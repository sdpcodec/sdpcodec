#!/usr/bin/env python3
"""
MLS (Multilingual LibriSpeech)를 Hugging Face에서 미리 다운로드하는 스크립트.

data_module.py의 load_mls_splits()와 동일한 방식으로
facebook/multilingual_librispeech를 받습니다.

필요 패키지: pip install datasets
conda 없이 Python + pip 만 있으면 실행 가능합니다.
"""
from __future__ import annotations

import argparse
import os
import sys

# HF 타임아웃 (data_module.py와 동일)
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

try:
    from datasets import load_dataset, DownloadConfig
except ModuleNotFoundError:
    print(
        "Missing dependency: 'datasets'\n\n"
        "Install it in a venv (recommended):\n"
        "  python3 -m venv .venv && . .venv/bin/activate && pip install datasets\n\n"
        "Or run the helper script:\n"
        "  ./run_download_mls.sh\n",
        file=sys.stderr,
    )
    raise


def main():
    parser = argparse.ArgumentParser(
        description="Download MLS (Multilingual LibriSpeech) from Hugging Face"
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="parler-tts/mls_eng",
        help=(
            "HF dataset id. Examples: parler-tts/mls_eng (english-only), "
            "facebook/multilingual_librispeech (multilingual, requires config=language)"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory. Default: HF_DATASETS_CACHE or HF_HOME/datasets",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["english"],
        help=(
            "Language configs (only used for facebook/multilingual_librispeech). "
            "Available: dutch french german italian polish portuguese spanish. "
            "Ignored for parler-tts/mls_eng."
        ),
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "dev", "test"],
        help="Splits to download. Default: train dev test",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream dataset instead of downloading full cache (dry-run friendly).",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="If >0, iterate and print first N examples per split (use with --streaming).",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=16,
        help="Number of processes for downloading. Default: 4",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = os.environ.get(
            "HF_DATASETS_CACHE",
            os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "datasets"),
        )
    cache_dir = os.path.abspath(cache_dir)
    print(f"Cache directory: {cache_dir}")

    dc = DownloadConfig(
        max_retries=20,
        resume_download=True,
        use_etag=True,
        local_files_only=(os.environ.get("HF_HUB_OFFLINE", "0") == "1"
                          or os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"),
    )

    dataset_id = str(args.dataset_id)
    languages = [str(x).lower() for x in args.languages]
    splits = [str(x) for x in args.splits]

    for split in splits:
        if dataset_id == "parler-tts/mls_eng":
            print(f"\n--- Downloading MLS(eng) | dataset={dataset_id} | split={split} ---")
            try:
                ds = load_dataset(
                    dataset_id,
                    split=split,
                    num_proc=args.num_proc,
                    cache_dir=cache_dir,
                    download_config=dc,
                    streaming=args.streaming,
                )
                if args.streaming:
                    print("  Loaded streaming dataset.")
                else:
                    print(f"  Loaded {len(ds)} samples.")
                if args.preview and args.streaming:
                    for i, ex in enumerate(ds.take(int(args.preview))):
                        ex_id = ex.get("id") or ex.get("original_path") or ex.get("file")
                        audio = ex.get("audio", {})
                        sr = None
                        if isinstance(audio, dict):
                            sr = audio.get("sampling_rate")
                        print(f"  preview[{i}] id={ex_id} sr={sr} keys={list(ex.keys())}")
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            for lang in languages:
                print(f"\n--- Downloading MLS | dataset={dataset_id} | config={lang} | split={split} ---")
                try:
                    ds = load_dataset(
                        dataset_id,
                        lang,
                        split=split,
                        num_proc=args.num_proc,
                        cache_dir=cache_dir,
                        download_config=dc,
                        streaming=args.streaming,
                    )
                    if args.streaming:
                        print("  Loaded streaming dataset.")
                    else:
                        print(f"  Loaded {len(ds)} samples.")
                    if args.preview and args.streaming:
                        for i, ex in enumerate(ds.take(int(args.preview))):
                            ex_id = ex.get("id") or ex.get("original_path") or ex.get("file")
                            audio = ex.get("audio", {})
                            sr = None
                            if isinstance(audio, dict):
                                sr = audio.get("sampling_rate")
                            print(f"  preview[{i}] id={ex_id} sr={sr} keys={list(ex.keys())}")
                except Exception as e:
                    print(f"  Error: {e}", file=sys.stderr)
                    sys.exit(1)

    print("\nDone. MLS datasets are cached at:", cache_dir)


if __name__ == "__main__":
    main()
