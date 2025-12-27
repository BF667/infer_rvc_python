from infer_rvc_python.lib.log_config import logger
import os
import threading
from tqdm import tqdm
import numpy as np
import soundfile as sf

# Import the new inference module
from infer_rvc_python.modules.inference import run_inference_script

warnings.filterwarnings("ignore")


class Config:
    """
    Minimal Config class to map old parameters to the new system.
    """
    def __init__(self, only_cpu=False):
        self.device = "cuda:0" if not only_cpu else "cpu"
        
        # Determine is_half based on device/capabilities
        import torch
        self.is_half = True
        if only_cpu:
            self.is_half = False
        elif torch.cuda.is_available():
            # Simple heuristic: if not RTX 30/40 series, might prefer float32, 
            # but the new module handles this internally mostly. 
            # We keep is_half=True by default for CUDA as per the old behavior.
            pass
        else:
            self.is_half = False


class BaseLoader:
    def __init__(self, only_cpu=False, hubert_path=None, rmvpe_path=None):
        self.model_config = {}
        self.config = Config(only_cpu)
        self.only_cpu = only_cpu
        # The new module manages internal downloads, but we keep hubert_path if needed for specific overrides
        self.hubert_path = hubert_path 
        self.rmvpe_path = rmvpe_path
        self.output_list = []

    def apply_conf(
        self,
        tag="base_model",
        file_model="",
        pitch_algo="rmvpe",  # Default to rmvpe as it's generally preferred now
        pitch_lvl=0,
        file_index="",
        index_influence=0.66,
        respiration_median_filtering=3,
        envelope_ratio=0.25,
        consonant_breath_protection=0.33,
        resample_sr=0,
        file_pitch_algo="",
    ):

        if not file_model:
            raise ValueError("Model not found")

        if file_index is None:
            file_index = ""

        if file_pitch_algo is None:
            file_pitch_algo = ""

        self.model_config[tag] = {
            "file_model": file_model,
            "pitch_algo": pitch_algo,
            "pitch_lvl": pitch_lvl,
            "file_index": file_index,
            "index_influence": index_influence,
            "respiration_median_filtering": respiration_median_filtering,
            "envelope_ratio": envelope_ratio,
            "consonant_breath_protection": consonant_breath_protection,
            "resample_sr": resample_sr,
            "file_pitch_algo": file_pitch_algo,
        }
        return f"CONFIGURATION APPLIED FOR {tag}: {file_model}"

    def _execute_inference(
        self,
        tag,
        params,
        input_audio_path,
        output_audio_path
    ):
        """
        Internal method to call the new run_inference_script logic.
        Maps old parameter names to the new function's arguments.
        """
        
        # Mapping old config to new inference script arguments
        # Note: Some parameters like 'f0_autotune' were not in the old BaseLoader,
        # so we set them to defaults or False.
        run_inference_script(
            config=self.config,
            pitch=params["pitch_lvl"],
            filter_radius=params["respiration_median_filtering"],
            index_rate=params["index_influence"],
            volume_envelope=params["envelope_ratio"],
            protect=params["consonant_breath_protection"],
            hop_length=64, # Default in new script
            f0_method=params["pitch_algo"],
            input_path=input_audio_path,
            output_path=output_audio_path,
            pth_path=params["file_model"],
            index_path=params["file_index"],
            export_format=os.path.splitext(output_audio_path)[1][1:], # wav, mp3, etc.
            embedder_model="contentvec_base", # Default embedder
            resample_sr=params["resample_sr"],
            f0_autotune=False,
            split_audio=False, # Can be enabled if needed, but sticking to old behavior for now
            clean_audio=False,
            clean_strength=0.7,
            formant_shifting=False,
            formant_qfrency=0.8,
            formant_timbre=0.8,
            proposal_pitch=False
        )

    def run_threads(self, threads):
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def unload_models(self):
        # The new module manages model lifecycle via the VoiceConverter class 
        # instantiated inside run_inference_script. 
        # We can clear our output list or config here if necessary, 
        # but explicit GPU clearing is handled internally by the new script's exception handling.
        self.output_list = []
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __call__(
        self,
        audio_files=[],
        tag_list=[],
        overwrite=False,
        parallel_workers=1,
        type_output=None,
    ):
        logger.info(f"Parallel workers: {str(parallel_workers)}")
        
        self.output_list = []

        if not self.model_config:
            raise ValueError("No model has been configured for inference")

        if isinstance(audio_files, str):
            audio_files = [audio_files]
        if isinstance(tag_list, str):
            tag_list = [tag_list]

        if not audio_files:
            raise ValueError("No audio found to convert")
        if not tag_list:
            tag_list = [list(self.model_config.keys())[-1]] * len(audio_files)

        if len(audio_files) > len(tag_list):
            logger.info("Extend tag list to match audio files")
            extend_number = len(audio_files) - len(tag_list)
            tag_list.extend([tag_list[0]] * extend_number)

        if len(audio_files) < len(tag_list):
            logger.info("Cut list tags")
            tag_list = tag_list[:len(audio_files)]

        threads = []
        progress_bar = tqdm(total=len(tag_list), desc="Progress")
        
        # We no longer need to sort by tag for model loading because the new module
        # loads/unloads efficiently inside the call. However, we keep the pairing logic.
        
        tag_file_pairs = list(zip(tag_list, audio_files))

        for i, (id_tag, input_audio_path) in enumerate(tag_file_pairs):
            if id_tag not in self.model_config.keys():
                logger.info(f"No configured model for {id_tag} with {input_audio_path}")
                continue

            params = self.model_config[id_tag]
            
            # Determine output path
            if overwrite:
                output_audio_path = input_audio_path
                # Ensure extension matches type_output if specified
                if type_output:
                    output_audio_path = os.path.splitext(output_audio_path)[0] + f".{type_output}"
            else:
                basename = os.path.basename(input_audio_path)
                dirname = os.path.dirname(input_audio_path)
                name_, ext_ = os.path.splitext(basename)
                ext = type_output if type_output else ext_.lstrip(".")
                output_audio_path = os.path.join(dirname, f"{name_}_edited.{ext}")

            # Check if file exists (from previous run or external)
            if os.path.exists(output_audio_path):
                # Overwrite logic: remove existing if overwrite is True (though we constructed path as overwrite)
                # If overwrite=False, we appended _edited, so it shouldn't exist unless rerun.
                # We follow the old logic: run_threads implies waiting, but here we thread.
                # The new script handles "os.remove" inside it, but for thread safety, let's just pass the path.
                # If overwrite=True in old code, it set input_path = output_path. 
                # run_inference_script has internal check `if os.path.exists(output_path): os.remove(output_path)`
                pass

            # Define the target for the thread
            def inference_wrapper(tag, p, in_path, out_path):
                try:
                    self._execute_inference(tag, p, in_path, out_path)
                    
                    # Store result in the main thread's list safely
                    # The new module prints to stdout, but let's capture the path for the return value
                    self.output_list.append(out_path)
                    
                    if tag in self.model_config and "result" not in self.model_config[tag]:
                        self.model_config[tag]["result"] = []
                    
                    # Note: Threading list append is atomic in CPython, good enough here
                    if tag in self.model_config:
                        self.model_config[tag]["result"].append(out_path)
                        
                except Exception as e:
                    logger.error(f"Error converting {in_path}: {e}")

            thread = threading.Thread(
                target=inference_wrapper,
                args=(
                    id_tag,
                    params,
                    input_audio_path,
                    output_audio_path
                )
            )
            threads.append(thread)

            # Manage thread pool
            if len(threads) >= parallel_workers:
                self.run_threads(threads)
                progress_bar.update(len(threads))
                threads = []

        # Run remaining threads
        if threads:
            self.run_threads(threads)
            progress_bar.update(len(threads))
        
        progress_bar.close()
        
        # Final result aggregation
        final_result = []
        # Return output_list directly as it was populated in order of completion (roughly)
        # or reconstruct based on model_config structure if strictly required by old API.
        # The old code returned model_config results flattened.
        for tag in set(tag_list):
            if tag in self.model_config and "result" in self.model_config[tag]:
                final_result.extend(self.model_config[tag]["result"])
        
        return final_result

    def generate_from_cache(
        self,
        audio_data=None, 
        tag=None,
        reload=False,
    ):
        """
        This method was designed to keep models in memory to avoid reloading.
        The new run_inference_script instantiates VoiceConverter every time.
        To strictly follow the request to 'use infer_rvc_python/modules/inference.py',
        we must adapt 'generate_from_cache' to save temp files and call the script,
        as the new module primarily operates on file paths (input_path/output_path).
        
        Note: True caching (keeping the model in memory) is harder to guarantee 
        without modifying the imported module. We will assume the overhead of 
        re-initializing the VoiceConverter is acceptable or handled internally 
        by the script logic if we were to pass objects (which we don't, we pass paths).
        """
        
        if not self.model_config:
            raise ValueError("No model has been configured for inference")

        if not audio_data:
            raise ValueError(
                "An audio file or tuple with "
                "(<numpy data audio>,<sampling rate>) is needed"
            )

        if tag not in self.model_config.keys():
            raise ValueError(f"No configured model for {tag}")

        import tempfile
        params = self.model_config[tag]

        # Handle input path or numpy array
        if isinstance(audio_data, str):
            input_audio_path = audio_data
            cleanup_input = False
        elif isinstance(audio_data, tuple):
            # (numpy_array, sr)
            data, sr = audio_data
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(temp_input.name, data, sr)
            input_audio_path = temp_input.name
            cleanup_input = True
        else:
            raise ValueError("audio_data must be a file path or a tuple (numpy_array, sample_rate)")

        # Handle output (usually return array in this method)
        # We need a temp file to write to, then read back
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        output_audio_path = temp_output.name

        try:
            # Execute
            self._execute_inference(tag, params, input_audio_path, output_audio_path)
            
            # Read back result
            audio_out, sr_out = sf.read(output_audio_path)
            
            # Cleanup
            if cleanup_input and os.path.exists(input_audio_path):
                os.remove(input_audio_path)
            if os.path.exists(output_audio_path):
                os.remove(output_audio_path)
                
            return audio_out, sr_out

        except Exception as e:
            logger.error(f"Cache inference failed: {e}")
            if os.path.exists(input_audio_path):
                os.remove(input_audio_path)
            if os.path.exists(output_audio_path):
                os.remove(output_audio_path)
            raise e
