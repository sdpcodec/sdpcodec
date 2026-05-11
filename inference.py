import os
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import librosa
import soundfile as sf
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from ptl.trixcodec.lightning_module import TriXCodecLightningModule


MODE_ALIASES = {
    "reconstruct": "same",
    "reconstruction": "same",
    "rec": "same",
    "same": "same",
    "vc": "vc",
    "voice_conversion": "vc",
    "vc_f0": "vc_f0",
}


def _resolve_optional_path(value: Optional[str]) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None
    return Path(to_absolute_path(str(value))).expanduser()


def _resolve_required_path(value: Optional[str], name: str) -> Path:
    path = _resolve_optional_path(value)
    if path is None:
        raise ValueError(f"`{name}` must be set.")
    return path


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def _load_wav(path: Path, sr: int) -> torch.Tensor:
    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    if wav.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    return torch.from_numpy(wav).float()


def _find_wavs(input_dir: Path) -> List[Path]:
    wavs = sorted(input_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No .wav files found under {input_dir}")
    return wavs


def _reference_for_source(
    source_path: Path,
    mode: str,
    ref_dir: Optional[Path],
    ref_wav: Optional[Path],
) -> Path:
    if mode == "same":
        return source_path
    if ref_wav is not None:
        return ref_wav
    if ref_dir is None:
        raise ValueError("Voice conversion mode requires `ref_wav=...` or `ref_dir=...`.")
    candidate = ref_dir / source_path.name
    if not candidate.is_file():
        raise FileNotFoundError(f"Reference wav not found for {source_path.name}: {candidate}")
    return candidate


def _build_batch(source_path: Path, reference_path: Path, target_sr: int, device: torch.device) -> Dict[str, torch.Tensor]:
    source_16k = _load_wav(source_path, 16000)
    source_target = source_16k if target_sr == 16000 else _load_wav(source_path, target_sr)
    reference_16k = source_16k if reference_path == source_path else _load_wav(reference_path, 16000)

    return {
        "wav": source_16k.unsqueeze(0).to(device),
        "wav_24k": source_target.unsqueeze(0).to(device),
        "ref_wav": reference_16k.unsqueeze(0).to(device),
    }


def _output_path(source_path: Path, reference_path: Path, output_dir: Path, mode: str) -> Path:
    if mode == "same":
        return output_dir / source_path.name
    stem = f"{source_path.stem}-{reference_path.stem}_{mode}"
    return output_dir / f"{stem}.wav"


def _load_model(cfg: DictConfig, ckpt_path: Path, device: torch.device) -> TriXCodecLightningModule:
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Download the checkpoint and place it at the path in `ckpt=...`."
        )

    cfg.ckpt = str(ckpt_path)
    model = TriXCodecLightningModule(cfg)
    try:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")

    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load] missing keys ignored: {len(missing)}")
    if unexpected:
        print(f"[load] unexpected keys ignored: {len(unexpected)}")

    model.eval().to(device)
    try:
        model._remove_all_weight_norms()
    except Exception as exc:
        print(f"[warn] failed to remove weight norm wrappers: {exc}")
    return model


@hydra.main(config_path="config", config_name="release/sdpcodec_24_s", version_base=None)
def main(cfg: DictConfig) -> None:
    mode = str(OmegaConf.select(cfg, "mode", default=OmegaConf.select(cfg, "voice_conversion", default="same")))
    mode = MODE_ALIASES.get(mode.lower(), mode.lower())
    if mode not in {"same", "vc", "vc_f0"}:
        raise ValueError(f"Unsupported mode: {mode}. Expected one of: same, vc, vc_f0.")

    input_dir = _resolve_required_path(OmegaConf.select(cfg, "input_dir"), "input_dir")
    output_dir = _resolve_required_path(OmegaConf.select(cfg, "output_dir"), "output_dir")
    ckpt_path = _resolve_required_path(OmegaConf.select(cfg, "ckpt"), "ckpt")
    ref_dir = _resolve_optional_path(OmegaConf.select(cfg, "ref_dir", default=None))
    ref_wav = _resolve_optional_path(OmegaConf.select(cfg, "ref_wav", default=None))
    device = _select_device(str(OmegaConf.select(cfg, "device", default="auto")))
    overwrite = bool(OmegaConf.select(cfg, "overwrite", default=False))
    target_sr = int(cfg.preprocess.audio.sr)

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_paths = _find_wavs(input_dir)
    model = _load_model(cfg, ckpt_path, device)
    model.vc_f0 = mode == "vc_f0"

    print(f"[inference] model_sr={target_sr} mode={mode} device={device} files={len(wav_paths)}")
    with torch.inference_mode():
        for source_path in tqdm(wav_paths):
            reference_path = _reference_for_source(source_path, mode, ref_dir, ref_wav)
            target_path = _output_path(source_path, reference_path, output_dir, mode)
            if target_path.exists() and not overwrite:
                continue

            batch = _build_batch(source_path, reference_path, target_sr, device)
            output = model(batch)
            wav = output["gen_wav"].squeeze().detach().cpu().float().clamp(-1.0, 1.0).numpy()
            sf.write(str(target_path), wav, target_sr)

    print(f"Done. Outputs written to: {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()
