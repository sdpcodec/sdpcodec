# devel image: nvcc is required when rebuilding CUDA extensions from source.
# Match the CUDA toolkit to the pinned PyTorch cu128 wheels to avoid ABI skew
# when building causal-conv1d / mamba-ssm from source.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ARG PYTORCH_CUDA=cu128
ARG TORCH_VERSION=2.8.0
ARG TORCHAUDIO_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCHCODEC_VERSION=0.7.0
ARG CAUSAL_CONV1D_VERSION=1.6.1
ARG MAMBA_SSM_VERSION=2.3.1

RUN apt update && apt install -y \
    wget \
    git \
    curl \
    bzip2 \
    build-essential \
    libtheora0 \
    sox && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/sdpcodec
ENV CUDA_HOME=/usr/local/cuda

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh
ENV CONDA_DIR=/opt/conda
ENV CONDA_ENV_NAME=sdpcodec
ENV PATH=/opt/conda/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

COPY environment.yml ./

# Official references used for this lock:
# - PyTorch wheels are available for torch 2.8.0 + cu128.
# - torchaudio compatibility matrix: torch 2.8 <-> torchaudio 2.8
# - torchcodec compatibility table: torchcodec 0.7 <-> torch 2.8
# - mamba-ssm v2.3.1 publishes release wheels for torch 2.6/2.7, so with torch 2.8
#   we rebuild mamba-ssm and causal-conv1d from source against the pinned torch ABI.
RUN --mount=type=cache,target=/root/.cache/pip \
    conda env create -f environment.yml && \
    conda run -n sdpcodec ffmpeg -version && \
    conda run -n sdpcodec python -m pip install -U pip setuptools wheel packaging ninja && \
    conda run -n sdpcodec python -m pip install --no-cache-dir git+https://github.com/sarulab-speech/UTMOSv2.git && \
    conda run -n sdpcodec python -m pip install --no-cache-dir --force-reinstall \
      --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}" \
      "torch==${TORCH_VERSION}" \
      "torchaudio==${TORCHAUDIO_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" && \
    conda run -n sdpcodec python -m pip install --no-cache-dir --force-reinstall \
      "fsspec==2025.3.0" && \
    conda run -n sdpcodec python -m pip install --no-cache-dir --force-reinstall --no-deps \
      --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}" \
      "torchcodec==${TORCHCODEC_VERSION}" && \
    conda run -n sdpcodec python -m pip uninstall -y mamba-ssm causal-conv1d || true && \
    conda run -n sdpcodec env CAUSAL_CONV1D_FORCE_BUILD=TRUE python -m pip install \
      --no-cache-dir \
      --no-build-isolation \
      --force-reinstall \
      --no-deps \
      "causal-conv1d==${CAUSAL_CONV1D_VERSION}" && \
    conda run -n sdpcodec env MAMBA_FORCE_BUILD=TRUE python -m pip install \
      --no-cache-dir \
      --no-build-isolation \
      --force-reinstall \
      --no-deps \
      "mamba-ssm==${MAMBA_SSM_VERSION}" && \
    conda run -n sdpcodec python -m pip check

# TorchAudio/TorchCodec need the conda env FFmpeg shared libraries at runtime.
ENV LD_LIBRARY_PATH=/opt/conda/envs/sdpcodec/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

SHELL ["conda", "run", "-n", "sdpcodec", "/bin/bash", "-c"]

RUN python - <<'PY'
import importlib.metadata as md

expected = {
    "torch": "2.8.0",
    "torchaudio": "2.8.0",
    "torchvision": "0.23.0",
    "torchcodec": "0.7.0",
    "causal-conv1d": "1.6.1",
    "mamba-ssm": "2.3.1",
}

for pkg, want in expected.items():
    got = md.version(pkg)
    if got != want:
        raise SystemExit(f"{pkg} version mismatch: expected {want}, got {got}")

import torch
import torchaudio
import torchvision
import torchcodec
import causal_conv1d
from mamba_ssm import Mamba2

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchaudio", torchaudio.__version__)
print("torchvision", torchvision.__version__)
print("torchcodec", md.version("torchcodec"))
print("causal-conv1d", md.version("causal-conv1d"))
print("mamba-ssm", md.version("mamba-ssm"))
print("Mamba2 import ok")
PY
