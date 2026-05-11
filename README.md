# SDP-Codec

Official implementation for **SDP-Codec: A Speaker-Decoupled Speech Codec with Pitch Injection for Low-Bitrate Coding and Zero-Shot Voice Conversion**.

SDP-Codec is a low-bitrate neural speech codec that separates local content/prosody tokens from global speaker information. It supports waveform reconstruction and zero-shot voice conversion with a single-codebook local stream and explicit pitch injection.

Paper and demo links will be added after release.

## Checkpoints

We will release the following checkpoints through Google Drive. Download links are placeholders for now and will be filled in later.

| Model | Setting | Bitrate | Training data | Config | Checkpoint path |
|---|---:|---:|---|---|---|
| SDP-Codec-24-S | 24 kHz small | 0.45 kbps | LibriTTS | `release/sdpcodec_24_s` | `checkpoints/SDP-Codec-24-S.ckpt` |
| SDP-Codec-16-S | 16 kHz small | 0.45 kbps | LibriSpeech | `release/sdpcodec_16_s` | `checkpoints/SDP-Codec-16-S.ckpt` |
| SDP-Codec-16-L | 16 kHz large | 0.52 kbps | MLS English + LibriSpeech | `release/sdpcodec_16_l` | `checkpoints/SDP-Codec-16-L.ckpt` |

The repository does not include checkpoint files. Place downloaded files under `checkpoints/` using the filenames above, or override `ckpt=/path/to/model.ckpt`.

Some release configs use frozen upstream models at runtime. If a run fails because an external model is missing, place the required file at the configured path, for example `pretrained_models/wavlm/WavLM-Large.pt` or `pretrained_models/pitch_estimator/fcpe/assets/fcpe_c_v001.pt`, or override the corresponding config path.

## Setup

```bash
conda create -n sdpcodec python=3.10 -y
conda activate sdpcodec
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA driver if the default `pip` build is not appropriate for your machine.

## Inference

Reconstruction:

```bash
python inference.py --config-name release/sdpcodec_24_s \
  input_dir=/path/to/input_wavs \
  output_dir=/path/to/output_wavs
```

Use another released model by changing `--config-name`:

```bash
python inference.py --config-name release/sdpcodec_16_s input_dir=/path/to/wavs output_dir=/path/to/out
python inference.py --config-name release/sdpcodec_16_l input_dir=/path/to/wavs output_dir=/path/to/out
```

Zero-shot voice conversion with a matching reference directory:

```bash
python inference.py --config-name release/sdpcodec_24_s \
  input_dir=/path/to/source_wavs \
  ref_dir=/path/to/reference_wavs \
  output_dir=/path/to/vc_outputs \
  mode=vc
```

For a single reference speaker WAV:

```bash
python inference.py --config-name release/sdpcodec_24_s \
  input_dir=/path/to/source_wavs \
  ref_wav=/path/to/reference.wav \
  output_dir=/path/to/vc_outputs \
  mode=vc
```

Input files should be mono WAV or readable by `librosa`; they are resampled internally as needed.

## Training

The research training scripts and Hydra configs are included for reproducibility. Prepare LibriSpeech-style filelists with:

```bash
python preprocess.py hydra.output_subdir=null hydra.job.chdir=False preprocess.datasets.LibriSpeech.root=/path/to/LibriSpeech
```

Then train with the desired config, for example:

```bash
python train.py --config-name release/sdpcodec_24_s train.trainer.devices=1 ckpt=null
```

Release configs are exact inference-compatible snapshots of the models above; for new training runs, review dataset paths, batch sizes, trainer settings, and external pretrained model paths before launching.

## Citation

Citation information will be added after the paper is public.

## Acknowledgements

This codebase is adapted from BigCodec and includes components derived from related open-source speech projects. Please keep the original license headers in source files when redistributing.

## License

MIT. See `LICENSE`.
