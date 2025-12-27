import os
import platform
import pkg_resources
from setuptools import find_packages, setup

setup(
    name="infer_rvc_python",
    version="1.2.0",
    description="Python wrapper for fast inference with rvc",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    readme="README.md",
    python_requires=">=3.10",
    author="R3gm",
    url="https://github.com/R3gm/infer_rvc_python",
    license="MIT",
    packages=find_packages(),
    package_data={'': ['*.txt', '*.rep', '*.pickle']},
    install_requires=[
        "pip>=23.3",
        "wheel",
        "omegaconf>=2.0.6",
        "torch>=2.3.1",
        "torchvision>=0.18.1",
        "torchaudio>=2.3.1",
        "faiss-cpu>=1.7.3",
        "einops>=0.8.0",
        "librosa>=0.10.2"
        "praat-parselmouth",
        "soundfile>=0.13.0",
        "numpy>=1.25.2,<2.0.0",
        "numba>=0.57.0",
        "scipy>=1.15.0",
        "requests>=2.32.3",
        "aiohttp",
        "ffmpy==0.3.1",
        "ffmpeg-python>=0.2.0",

    ],
    include_package_data=True,
    extras_require={"all": [
        "yt-dlp",
        "edge-tts"
        ]},

)
