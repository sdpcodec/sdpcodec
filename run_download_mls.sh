#!/bin/bash

export WANDB_MODE=online
# MLS(Multilingual LibriSpeech) Hugging Face 다운로드 실행 스크립트
#
# conda 없이: pip install datasets 만 하면 됨.
#   python download_mls.py [옵션]
#
# sdpcodec conda 환경 쓰려면:
#   conda run --no-capture-output -n sdpcodec python download_mls.py [옵션]
#

set -e
cd "$(dirname "$0")"

# 캐시 경로 (run_mls_*.sh 와 맞추려면 아래처럼 설정)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

# conda 사용 여부: CONDA_ENV=sdpcodec 이면 conda run 사용
if [ -n "${CONDA_ENV}" ]; then
  exec conda run --no-capture-output -n "$CONDA_ENV" python -u download_mls.py "$@"
else
  # PEP 668(externally-managed-environment) 환경에서는 시스템 Python에 pip install이 막힘
  # → 로컬 venv를 만들어 그 안에 datasets를 설치해서 실행
  VENV_DIR="${VENV_DIR:-$(pwd)/.venv_mls}"

  if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "[run_download_mls] Creating venv at: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi

  "$VENV_DIR/bin/python" -m pip install -q -U pip
  "$VENV_DIR/bin/python" -m pip install -q datasets
  exec "$VENV_DIR/bin/python" -u download_mls.py "$@"
fi
