#!/usr/bin/env python3
"""
TSM Bench: Run multiple time-stretch backends (WSOLA/phase-vocoder/neural hooks) from Python/PyTorch.

Backends implemented out-of-the-box:
  - torchaudio_phase_vocoder  (PyTorch/CUDA optional)
  - sox_tempo                 (SoX WSOLA via pysox)
  - audiotsm_wsola            (Pure-Python WSOLA, optional accel)
  - audiotsm_phasevocoder     (Phase vocoder in Python)
  - rubberband_tempo          (If pyrubberband + system rubberband installed)

Neural placeholders (provide your own packages/weights to enable):
  - tsmnet_pytorch
  - scalergan
  - controllable_lpcnet

Optional metrics if packages exist:
  - PESQ (pesq>=0.0.4), STOI (pystoi>=0.3.3)

Usage:
  python tsm_bench.py --input in.wav --rate 1.10 --methods all --outdir out
"""
import argparse, time, math, os, sys, importlib, shutil
from dataclasses import dataclass
from typing import Dict, Callable, Optional

import numpy as np

DEFAULT_PYTSMOD_DIR = os.environ.get("PYTSMOD_DIR")
DEFAULT_SCALERGAN_DIR = os.environ.get("SCALERGAN_DIR")

# --- Audio I/O ---
def _lazy_import_soundfile():
    try:
        import soundfile as sf
        return sf
    except Exception as e:
        print("[WARN] soundfile not available. Install with `pip install soundfile`.", file=sys.stderr)
        raise

def load_wav(path, target_sr: Optional[int]=None):
    sf = _lazy_import_soundfile()
    wav, sr = sf.read(path, dtype='float32', always_2d=True)
    wav = wav.mean(axis=1)  # mono
    if target_sr and target_sr != sr:
        # resample with torchaudio if available, else simple polyphase via librosa
        try:
            import torchaudio, torch
            wav_t = torch.from_numpy(wav).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            wav = resampler(wav_t).squeeze(0).numpy()
            sr = target_sr
        except Exception:
            try:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            except Exception:
                raise RuntimeError("Need torchaudio or librosa to resample.")
    return wav, sr

def save_wav(path, wav, sr):
    sf = _lazy_import_soundfile()
    sf.write(path, wav.astype(np.float32), sr)

# --- Helpers ---
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def rtf(process_sec, audio_dur_sec):
    return float(process_sec) / max(1e-6, float(audio_dur_sec))

# --- Backends ---
def backend_torchaudio_phase_vocoder(wav, sr, rate, n_fft=1024, hop=256, win=None, device=None):
    """
    STFT -> phase vocoder -> iSTFT. Keeps pitch roughly constant while changing time by `rate`.
    """
    import torch, torchaudio
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(wav).to(device)
    if win is None:
        win = torch.hann_window(n_fft, device=device)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, window=win, return_complex=True, center=True)
    # phase advance heuristic similar to torchaudio docs
    freq_bins = spec.size(-2)
    phase_advance = torch.linspace(0, math.pi * hop, freq_bins, device=device)[..., None]
    t0 = time.time()
    spec_st = torchaudio.functional.phase_vocoder(spec.unsqueeze(0), rate=rate, phase_advance=phase_advance).squeeze(0)
    y = torch.istft(spec_st, n_fft=n_fft, hop_length=hop, window=win, length=int(round(len(wav)/rate)) if rate>1 else None)
    proc = time.time() - t0
    return y.detach().cpu().numpy(), proc

def backend_sox_tempo(wav, sr, rate):
    """
    SoX WSOLA tempo via pysox. Writes temp files under /tmp.
    """
    import tempfile, uuid, subprocess
    import soundfile as sf
    tmp_in = os.path.join(tempfile.gettempdir(), f"tsm_in_{uuid.uuid4().hex}.wav")
    tmp_out = os.path.join(tempfile.gettempdir(), f"tsm_out_{uuid.uuid4().hex}.wav")
    sf.write(tmp_in, wav.astype(np.float32), sr)
    t0 = time.time()
    try:
        # Preferred: pysox if available
        import sox as _pysox
        tfm = _pysox.Transformer()
        tfm.tempo(factor=rate)
        tfm.build_file(tmp_in, tmp_out)
    except Exception:
        # Fallback: SoX CLI
        if shutil.which("sox") is None:
            try:
                os.remove(tmp_in)
            except Exception:
                pass
            raise RuntimeError("SoX not available: install pysox or system 'sox' binary.")
        cmd = ["sox", tmp_in, tmp_out, "tempo", str(rate)]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            try:
                os.remove(tmp_in)
            except Exception:
                pass
            raise RuntimeError(f"SoX CLI failed: {res.stderr.decode(errors='ignore')}")
    proc = time.time() - t0
    y, _ = load_wav(tmp_out, target_sr=sr)
    try:
        os.remove(tmp_in); os.remove(tmp_out)
    except Exception:
        pass
    return y, proc

def backend_audiotsm_wsola(wav, sr, rate):
    """
    Pure-Python WSOLA (audiotsm). Slower than SoX but dependency-light.
    """
    import io
    import audiotsm, audiotsm.io.array
    from audiotsm import wsola
    # ArrayReader expects data with shape (channels, n)
    data = wav[np.newaxis, :] if getattr(wav, 'ndim', 1) == 1 else wav
    in_stream = audiotsm.io.array.ArrayReader(data)
    out_stream = audiotsm.io.array.ArrayWriter(1)
    tsm = wsola(1, rate)
    t0 = time.time()
    tsm.run(in_stream, out_stream)
    proc = time.time() - t0
    y = out_stream.data.astype(np.float32).flatten()
    return y, proc

def backend_audiotsm_phasevocoder(wav, sr, rate):
    import audiotsm, audiotsm.io.array
    from audiotsm import phasevocoder
    # ArrayReader expects data with shape (channels, n)
    data = wav[np.newaxis, :] if getattr(wav, 'ndim', 1) == 1 else wav
    in_stream = audiotsm.io.array.ArrayReader(data)
    out_stream = audiotsm.io.array.ArrayWriter(1)
    tsm = phasevocoder(1, rate)
    t0 = time.time()
    tsm.run(in_stream, out_stream)
    proc = time.time() - t0
    y = out_stream.data.astype(np.float32).flatten()
    return y, proc

def backend_rubberband_tempo(wav, sr, rate):
    """
    pyrubberband requires the system library 'rubberband' installed.
    """
    import pyrubberband as prb
    t0 = time.time()
    y = prb.time_stretch(wav, sr, rate)
    proc = time.time() - t0
    return y.astype(np.float32), proc

# --- Neural placeholders ---
def backend_tsmnet_pytorch(wav, sr, rate, checkpoint=None, device=None):
    """
    Example hook: requires a package/model that exposes `infer(wav, sr, rate) -> wav_out`.
    Users can adapt this to their TSM-Net implementation.
    """
    mod = importlib.import_module("tsmnet_infer")  # user-provided module
    t0 = time.time()
    y = mod.infer(wav, sr, rate, checkpoint=checkpoint, device=device)
    proc = time.time() - t0
    return y.astype(np.float32), proc

_SCALERGAN_CACHE = {}

def backend_scalergan(wav, sr, rate, checkpoint=None, device=None, hifi_ckpt=None, hifi_conf=None, repo_dir=None):
    """
    ScalerGAN inference using the local ScalerGAN repo with in-memory mel + HiFi-GAN.
    - `rate` > 1.0 means faster → time_scale = 1/rate for the generator
    - Resamples internally to the ScalerGAN mel sampling rate (default 22050)
    """
    import json
    import torch
    import importlib
    import importlib.util as _ilspec
    import types as _types
    # Resolve paths
    repo_dir = repo_dir or DEFAULT_SCALERGAN_DIR
    if not repo_dir:
        raise RuntimeError("ScalerGAN backend requires `repo_dir` or SCALERGAN_DIR.")
    sys.path.insert(0, os.path.join(repo_dir, "hifi_gan"))
    # Dynamically import ScalerGAN modules from file paths to avoid package __init__ (which requires GitPython)
    def _load_module(name, file_path):
        spec = _ilspec.spec_from_file_location(name, file_path)
        mod = _ilspec.module_from_spec(spec)
        assert spec and spec.loader, f"Cannot load module from {file_path}"
        spec.loader.exec_module(mod)
        return mod
    def _register_pkg(mod_name, dir_path):
        if mod_name in sys.modules:
            return sys.modules[mod_name]
        pkg = _types.ModuleType(mod_name)
        pkg.__path__ = [dir_path]
        sys.modules[mod_name] = pkg
        return pkg
    sg_root = os.path.join(repo_dir, "scaler_gan")
    _register_pkg("scaler_gan", sg_root)
    _register_pkg("scaler_gan.scalergan_utils", os.path.join(sg_root, "scalergan_utils"))
    _sg_utils = _load_module("scaler_gan.scalergan_utils.scalergan_utils", os.path.join(sg_root, "scalergan_utils/scalergan_utils.py"))
    mel_spectrogram = _sg_utils.mel_spectrogram
    _register_pkg("scaler_gan.network_topology", os.path.join(sg_root, "network_topology"))
    _networks = _load_module("scaler_gan.network_topology.networks", os.path.join(sg_root, "network_topology/networks.py"))
    SGGenerator = _networks.Generator
    # Config paths
    mel_conf_path = os.path.join(repo_dir, "scaler_gan/configs/mel_config.json")
    hifi_conf_path = hifi_conf or os.path.join(repo_dir, "scaler_gan/configs/hifi_config.json")
    # Device and cache key
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g_ckpt = checkpoint or os.path.join(repo_dir, "pretrained_models/lj_speech_model.pth.tar")
    h_ckpt = hifi_ckpt or os.path.join(repo_dir, "pretrained_models/hifi_checkpoint_v1")
    cache_key = (repo_dir, g_ckpt, h_ckpt, mel_conf_path, hifi_conf_path, device)
    cached = _SCALERGAN_CACHE.get(cache_key)
    if cached is None:
        # Load mel + hifi configs
        with open(mel_conf_path, "r") as f:
            mel_cfg = json.load(f)
        from models import Generator as HifiGenerator
        from env import AttrDict as HifiAttrDict
        with open(hifi_conf_path, "r") as f:
            hifi_h = HifiAttrDict(json.load(f))
        # Load models once
        G = SGGenerator(64, 6, 3, True, True).to(device)
        state = torch.load(g_ckpt, map_location=device)
        if "G" in state:
            G.load_state_dict(state["G"])
        elif "generator" in state:
            G.load_state_dict(state["generator"])
        else:
            raise RuntimeError("ScalerGAN checkpoint missing generator weights")
        G.eval()
        hifi = HifiGenerator(hifi_h).to(device)
        hifi_state = torch.load(h_ckpt, map_location=device)
        if "generator" in hifi_state:
            hifi.load_state_dict(hifi_state["generator"])
        else:
            hifi.load_state_dict(hifi_state)
        hifi.eval()
        _SCALERGAN_CACHE[cache_key] = (G, hifi, mel_cfg, hifi_h)
        cached = _SCALERGAN_CACHE[cache_key]
    G, hifi, mel_cfg, hifi_h = cached
    # Unpack mel params
    model_sr = int(mel_cfg.get("sampling_rate", 22050))
    n_fft = int(mel_cfg.get("n_fft", 1024))
    num_mels = int(mel_cfg.get("num_mels", 80))
    hop_size = int(mel_cfg.get("hop_size", 256))
    win_size = int(mel_cfg.get("win_size", 1024))
    fmin = int(mel_cfg.get("fmin", 0))
    fmax = int(mel_cfg.get("fmax", 8000))
    must_divide = 8
    # Prepare audio tensor at model SR
    x = wav.astype(np.float32)
    if sr != model_sr:
        try:
            import torchaudio, torch
            x_t = torch.from_numpy(x).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, model_sr)
            x = resampler(x_t).squeeze(0).numpy().astype(np.float32)
        except Exception:
            try:
                import librosa
                x = librosa.resample(x, orig_sr=sr, target_sr=model_sr).astype(np.float32)
            except Exception:
                raise RuntimeError("Need torchaudio or librosa to resample for ScalerGAN.")
    x_t = torch.from_numpy(x).float().unsqueeze(0)  # [1, T]
    # Build mel [1, n_mels, frames]
    mel = mel_spectrogram(
        x_t,
        n_fft=n_fft,
        num_mels=num_mels,
        sampling_rate=model_sr,
        hop_size=hop_size,
        win_size=win_size,
        fmin=fmin,
        fmax=fmax,
        center=False,
    ).unsqueeze(0)
    # Time-scale
    time_scale = float(1.0 / max(rate, 1e-6))
    in_h, in_w = int(mel.shape[2]), int(mel.shape[3]) if mel.dim()==4 else (num_mels, mel.shape[-1])
    out_h = in_h  # freq scale fixed to 1
    out_w = int(np.floor(time_scale * in_w / must_divide) * must_divide)
    out_w = max(must_divide, out_w)
    # Run generator to scale mel
    input_mel = mel.to(device)
    t0 = time.time()
    with torch.no_grad():
        g_pred = G(input_mel, output_size=(out_h, out_w), random_affine=None)
    # HiFi-GAN synthesis (cached)
    with torch.no_grad():
        # g_pred: [1, 1, n_mels, T] or [1, n_mels, T]; ensure [n_mels, T]
        mel_for_vocoder = g_pred.squeeze(0)
        if mel_for_vocoder.dim() == 3:
            mel_for_vocoder = mel_for_vocoder.squeeze(0)
        y_hat = hifi(mel_for_vocoder)
        audio = y_hat.squeeze().detach().cpu().numpy().astype(np.float32)
    # Ensure output is at requested sampling rate
    try:
        out_sr = int(getattr(hifi_h, "sampling_rate", model_sr))
    except Exception:
        out_sr = model_sr
    if out_sr != sr:
        try:
            import torchaudio
            a_t = torch.from_numpy(audio).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(out_sr, sr)
            audio = resampler(a_t).squeeze(0).numpy().astype(np.float32)
        except Exception:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=out_sr, target_sr=sr).astype(np.float32)
            except Exception:
                # Fallback: return as-is; caller will misinterpret duration
                pass
    proc = time.time() - t0
    return audio, proc

def backend_controllable_lpcnet(wav, sr, rate, pitch_shift=0.0, **kw):
    """
    Example hook for controllable LPCNet bindings, expecting `clpcnet.infer(wav, sr, rate, pitch_shift)`.
    """
    mod = importlib.import_module("clpcnet_infer")
    t0 = time.time()
    y = mod.infer(wav, sr, rate, pitch_shift=pitch_shift)
    proc = time.time() - t0
    return y.astype(np.float32), proc

# --- Metrics (optional) ---
def optional_metrics(ref, est, sr):
    out = {}
    dur = len(ref)/sr
    # PESQ narrow-band 8k or wideband 16k
    from pesq import pesq
    if sr in (8000, 16000):
        out["PESQ"] = float(pesq(sr, ref, est, 'wb' if sr==16000 else 'nb'))
    else:
        out["PESQ"] = None
    out["PESQ"] = None
    from pystoi import stoi
    out["STOI"] = float(stoi(ref, est, sr, extended=False))
    # Simple SNR
    minlen = min(len(ref), len(est))
    ref_c = ref[:minlen]; est_c = est[:minlen]
    noise = ref_c - est_c
    num = np.sum(ref_c**2) + 1e-12
    den = np.sum(noise**2) + 1e-12
    out["SNR_dB"] = 10*np.log10(num/den)
    return out

# --- Registry ---
BACKENDS: Dict[str, Callable] = {
    "torchaudio_phase_vocoder": backend_torchaudio_phase_vocoder,
    "sox_tempo": backend_sox_tempo,
    "audiotsm_wsola": backend_audiotsm_wsola,
    "audiotsm_phasevocoder": backend_audiotsm_phasevocoder,
    "rubberband_tempo": backend_rubberband_tempo,
    # neural hooks
    "tsmnet_pytorch": backend_tsmnet_pytorch,
    "scalergan": backend_scalergan,
    "controllable_lpcnet": backend_controllable_lpcnet,
}

# --- PyTSMod (local repo) ---
def backend_pytsmod_wsola(wav, sr, rate, win_size=1024, syn_hop_size=512, tolerance=512):
    import_types_ok = True
    try:
        if DEFAULT_PYTSMOD_DIR:
            sys.path.insert(0, DEFAULT_PYTSMOD_DIR)
        from pytsmod import wsolatsm as _w
    except Exception:
        import_types_ok = False
    if not import_types_ok:
        raise RuntimeError("PyTSMod not available. Install pytsmod or set PYTSMOD_DIR.")
    s = float(1.0 / max(rate, 1e-6))
    t0 = time.time()
    y = _w.wsola(wav, s, win_type='hann', win_size=win_size, syn_hop_size=syn_hop_size, tolerance=tolerance)
    proc = time.time() - t0
    return y.astype(np.float32), proc

def backend_pytsmod_phasevocoder(wav, sr, rate, win_size=2048, syn_hop_size=512, zero_pad=0, restore_energy=False, fft_shift=False, phase_lock=False):
    import_types_ok = True
    try:
        if DEFAULT_PYTSMOD_DIR:
            sys.path.insert(0, DEFAULT_PYTSMOD_DIR)
        from pytsmod import pvtsm as _pv
    except Exception:
        import_types_ok = False
    if not import_types_ok:
        raise RuntimeError("PyTSMod not available. Install pytsmod or set PYTSMOD_DIR.")
    s = float(1.0 / max(rate, 1e-6))
    t0 = time.time()
    y = _pv.phase_vocoder(wav, s, win_type='sin', win_size=win_size, syn_hop_size=syn_hop_size, zero_pad=zero_pad, restore_energy=restore_energy, fft_shift=fft_shift, phase_lock=phase_lock)
    proc = time.time() - t0
    return y.astype(np.float32), proc

# Register PyTSMod backends
BACKENDS.update({
    "pytsmod_wsola": backend_pytsmod_wsola,
    "pytsmod_phasevocoder": backend_pytsmod_phasevocoder,
})

def detect_available_backends():
    """
    Return a list of backend names that appear usable on this system by
    probing for their required Python dependencies. This is a lightweight
    check meant for UI/notebook usage and does not guarantee runtime success.
    """
    import importlib.util as _ils
    dep_map = {
        "torchaudio_phase_vocoder": ["torch", "torchaudio"],
        # For SoX, allow either pysox module OR system sox CLI
        "sox_tempo": [],
        "audiotsm_wsola": ["audiotsm"],
        "audiotsm_phasevocoder": ["audiotsm"],
        "rubberband_tempo": ["pyrubberband"],
        # neural placeholders (user-provided)
        "tsmnet_pytorch": ["tsmnet_infer"],
        # local ScalerGAN repo check is handled below
        "scalergan": [],
        "controllable_lpcnet": ["clpcnet_infer"],
        # local PyTSMod backends
        "pytsmod_wsola": [],
        "pytsmod_phasevocoder": [],
    }
    avail = []
    for name, mods in dep_map.items():
        ok = True
        if mods:
            for m in mods:
                if _ils.find_spec(m) is None:
                    ok = False
                    break
        # Special-case local repos
        if name.startswith("pytsmod"):
            ok = ok and (
                _ils.find_spec("pytsmod") is not None
                or bool(DEFAULT_PYTSMOD_DIR and os.path.isdir(os.path.join(DEFAULT_PYTSMOD_DIR, "pytsmod")))
            )
        if name == "scalergan":
            ok = ok and bool(
                DEFAULT_SCALERGAN_DIR
                and os.path.isdir(os.path.join(DEFAULT_SCALERGAN_DIR, "scaler_gan"))
            )
        if not ok:
            continue
        # Extra runtime requirement checks / special cases
        if name == "sox_tempo":
            # consider available if either pysox module OR sox CLI exists
            ok_py = _ils.find_spec("sox") is not None
            ok_cli = shutil.which("sox") is not None
            ok = ok and (ok_py or ok_cli)
        if ok:
            avail.append(name)
    return avail

def run_tsm(method: str, wav: np.ndarray, sr: int, rate: float, device: Optional[str]=None, **kwargs):
    """
    Run a single TSM method by name on an in-memory waveform.
    Returns (y, info_dict) where info contains processing time and rtf.
    """
    if method not in BACKENDS:
        raise ValueError(f"Unknown method: {method}")
    fn = BACKENDS[method]
    if method == "torchaudio_phase_vocoder":
        y, tsec = fn(wav, sr, rate, device=device, **kwargs)
    elif method in ("sox_tempo", "audiotsm_wsola", "audiotsm_phasevocoder", "rubberband_tempo", "pytsmod_wsola", "pytsmod_phasevocoder"):
        y, tsec = fn(wav, sr, rate, **kwargs)
    elif method == "tsmnet_pytorch":
        y, tsec = fn(wav, sr, rate, device=device, **kwargs)
    elif method == "scalergan":
        y, tsec = fn(wav, sr, rate, device=device, **kwargs)
    elif method == "controllable_lpcnet":
        y, tsec = fn(wav, sr, rate, **kwargs)
    else:
        raise RuntimeError("Unhandled backend.")
    info = {
        "backend": method,
        "proc_sec": float(tsec),
        "rtf": rtf(tsec, len(wav)/float(sr) if sr else 0.0),
        "in_dur_sec": len(wav)/float(sr) if sr else None,
    }
    return y, info

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to input WAV")
    p.add_argument("--rate", type=float, required=True, help=">1.0 faster, <1.0 slower")
    p.add_argument("--methods", nargs="+", default=["all"], help="Choose from registry names or 'all'")
    p.add_argument("--outdir", default="tsm_out", help="Output directory")
    p.add_argument("--target_sr", type=int, default=16000, help="Resample to this SR for fair comparison")
    p.add_argument("--metrics", action="store_true", help="Compute PESQ/STOI if available + SNR")
    p.add_argument("--device", default=None, help="'cuda' or 'cpu' (for torch-based backends)")
    p.add_argument("--tsmnet_ckpt", default=None)
    p.add_argument("--scalergan_ckpt", default=None)
    p.add_argument("--scalergan_hifi_ckpt", default=None)
    p.add_argument("--scalergan_hifi_conf", default=None)
    p.add_argument("--scalergan_repo_dir", default=DEFAULT_SCALERGAN_DIR)
    p.add_argument("--lpcnet_pitch_shift", type=float, default=0.0)
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.outdir)
    wav, sr = load_wav(args.input, target_sr=args.target_sr)
    dur = len(wav)/args.target_sr

    selected = list(BACKENDS.keys()) if (len(args.methods)==1 and args.methods[0].lower()=="all") else args.methods
    results = []
    for name in selected:
        if name not in BACKENDS:
            print(f"[SKIP] Unknown method: {name}")
            continue
        print(f"[RUN] {name} (rate={args.rate}) ...", flush=True)
        fn = BACKENDS[name]
        if name == "torchaudio_phase_vocoder":
            y, tsec = fn(wav, args.target_sr, args.rate, device=args.device)
        elif name == "sox_tempo":
            y, tsec = fn(wav, args.target_sr, args.rate)
        elif name == "audiotsm_wsola":
            y, tsec = fn(wav, args.target_sr, args.rate)
        elif name == "audiotsm_phasevocoder":
            y, tsec = fn(wav, args.target_sr, args.rate)
        elif name == "rubberband_tempo":
            y, tsec = fn(wav, args.target_sr, args.rate)
        elif name == "pytsmod_wsola":
            y, tsec = fn(wav, args.target_sr, args.rate)
        elif name == "pytsmod_phasevocoder":
            y, tsec = fn(wav, args.target_sr, args.rate)
        elif name == "tsmnet_pytorch":
            y, tsec = fn(wav, args.target_sr, args.rate, checkpoint=args.tsmnet_ckpt, device=args.device)
        elif name == "scalergan":
            y, tsec = fn(
                wav,
                args.target_sr,
                args.rate,
                checkpoint=args.scalergan_ckpt,
                device=args.device,
                hifi_ckpt=args.scalergan_hifi_ckpt,
                hifi_conf=args.scalergan_hifi_conf,
                repo_dir=args.scalergan_repo_dir,
            )
        elif name == "controllable_lpcnet":
            y, tsec = fn(wav, args.target_sr, args.rate, pitch_shift=args.lpcnet_pitch_shift)
        else:
            raise RuntimeError("Unhandled backend.")
        outpath = os.path.join(args.outdir, f"{os.path.splitext(os.path.basename(args.input))[0]}_{name}_{args.rate:.2f}.wav")
        save_wav(outpath, y, args.target_sr)
        row = {"backend": name, "rtf": rtf(tsec, dur), "proc_sec": tsec, "in_dur_sec": dur, "out": outpath}
        if args.metrics:
            m = optional_metrics(wav, y, args.target_sr)
            row.update(m)
        results.append(row)
        print(f"[OK] {name}: RTF={row['rtf']:.3f}, wrote {outpath}")

    # Save JSON summary
    import json
    summary_path = os.path.join(args.outdir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
    # Pretty table
    from tabulate import tabulate
    headers = ["backend", "rtf", "proc_sec", "in_dur_sec", "PESQ", "STOI", "SNR_dB", "out"]
    table = []
    for r in results:
        table.append([r.get(h, None) for h in headers])
    print("\n" + tabulate(table, headers=headers))

if __name__ == "__main__":
    main()
