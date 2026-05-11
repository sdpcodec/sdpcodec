import os
import random
import hydra
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from vq import CodecEncoder, CodecDecoder
from vq.temporal_config import (
    CodecDecoderF0ConditionConfig,
    CodecDecoderSpeakerConditionConfig,
    CodecDecoderTemporalConfig,
    CodecDecoderVocosConfig,
    CodecTemporalConfig,
    F0CodecEncoderConfig,
    F0CodecDecoderConfig,
    F0CodecSpeakerConditionConfig,
)
from vq.codec_encoder import VQW2VEncoder
from vq.speaker.speaker_encoder import (
    SpeakerEncoder,
    build_speaker_quantizer_kwargs,
    resolve_speaker_encoder_token_dim,
)
from vq.f0.codec import F0Encoder, F0Decoder
from module import (
    APNet2MultiResolutionDiscriminator,
    BigVGANMultiResolutionDiscriminator,
    HiFiGANMultiPeriodDiscriminator,
    LegacySpecDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
    VocosMultiResolutionDiscriminator,
)
from criterions import APNet2AuxLoss, GANLoss, MultiResolutionMelSpectrogramLoss
from common.schedulers import WarmupLR
from metrics.metrics import STOI
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio, ScaleInvariantSignalDistortionRatio
from pytorch_lightning.utilities.model_summary import ModelSummary
from contextlib import nullcontext
import torchaudio
from termcolor import colored
from s3prl.nn import S3PRLUpstream
from torch.amp import autocast
import matplotlib.pyplot as plt
import matplotlib
import wandb
import jiwer
import time
from transformers import AutoProcessor, HubertForCTC
from requests.exceptions import ConnectionError as RequestsConnectionError
from urllib3.exceptions import ProtocolError as Urllib3ProtocolError
from torch.nn.parameter import UninitializedParameter

from vq.ssl_codec_wrappers import (
    W2V2HiddenStateEncoder,
    W2VBert2HiddenStateEncoder,
    VQW2VCodecEncoderWrapper,
    W2V2CodecEncoderWrapper,
    HubertCodecEncoderWrapper,
    W2VBert2CodecEncoderWrapper,
    S3TokenizerEncoder,
    S3TokenizerCodecEncoderWrapper,
)

def _debug_tensor_stats(x):
    if x is None:
        return None
    if not isinstance(x, torch.Tensor):
        return str(type(x))
    with torch.no_grad():
        x_detached = x.detach().float()
        finite_mask = torch.isfinite(x_detached)
        stats = {
            "shape": tuple(x_detached.shape),
            "non_finite": int((~finite_mask).sum().item()),
        }
        if finite_mask.any():
            finite_vals = x_detached[finite_mask]
            stats.update(
                {
                    "min": float(finite_vals.min().item()),
                    "max": float(finite_vals.max().item()),
                    "mean": float(finite_vals.mean().item()),
                    "absmax": float(finite_vals.abs().max().item()),
                }
            )
        return stats


def _set_s3prl_download_dir(cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["S3PRL_CACHE_ROOT"] = cache_dir
    try:
        from s3prl.util.download import set_dir as s3prl_set_dir

        s3prl_set_dir(os.path.join(cache_dir, "download"))
    except Exception as exc:
        print(colored(f"Warning: failed to set s3prl download dir: {exc}", "yellow"))


class TriXCodecLightningModule(pl.LightningModule):
    """
    TriXCodec: Neural Codec with Joint Content-F0 Quantization
    
    Architecture Overview:
    ----------------------
    Unlike TriCodec which quantizes content and F0 separately, TriXCodec performs
    joint quantization of content and F0 features through the following pipeline:
    
    1. Content Encoding:    wav → CodecEncoder → vq_emb (B, C, T)
    2. F0 Encoding:         wav → F0Extractor → F0Encoder → z_f0 (B, D_f0, T)
    3. Joint Projection:    [vq_emb; z_f0] → joint_mixer → mixed (B, vq_dim, T)
    4. Joint Quantization:  mixed → VQ → vq_post_emb (B, vq_dim, T)
    5. Disentanglement:     vq_post_emb → joint_to_audio_f0 → [content; f0]
    6. Separate Decoding:
       - content → CodecDecoder (+ speaker + F0 condition) → audio
       - f0 → F0Decoder → reconstructed F0
    
    Key Differences from TriCodec:
    ------------------------------
    - TriCodec:  Separate VQ for content and F0
    - TriXCodec: Joint VQ with projection layers (joint_mixer, joint_to_audio_f0)
    - Benefit:   Single codebook for both content and F0, potentially better
                 capturing of content-F0 correlations
    
    New Modules:
    -----------
    - joint_mixer:       Conv1d projecting concatenated features to VQ dimension
    - joint_to_audio_f0: Conv1d projecting quantized features back to disentangled dims
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_unnormf0_mse_loss = getattr(self.cfg.model.f0_codec, 'use_unnormf0_mse_loss', False)
        self.ocwd = hydra.utils.get_original_cwd()
        self.construct_model()
        self._train_weight_norm_strip_targets = tuple(getattr(self.cfg.train, 'remove_weight_norm_from_modules', []))
        if self._train_weight_norm_strip_targets:
            self._remove_weight_norm_from_submodules(self._train_weight_norm_strip_targets)
        self.construct_criteria()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.stoi = STOI(in_sr=cfg.preprocess.audio.sr, sr=cfg.preprocess.audio.sr)
        self.si_snr = ScaleInvariantSignalNoiseRatio()
        self.si_sdr = ScaleInvariantSignalDistortionRatio()
        self.register_buffer('total_step', torch.tensor(0, dtype=torch.long))
        self.vc_f0 = False
        self._timing_enabled = bool(getattr(self.cfg.train, 'profile_timing_enabled', False))
        self._timing_sync_cuda = bool(getattr(self.cfg.train, 'profile_timing_sync_cuda', True))
        self._timing_warmup_steps = int(getattr(self.cfg.train, 'profile_timing_warmup_steps', 5))
        self._timing_log_every = int(getattr(self.cfg.train, 'profile_timing_log_every', 20))
        self._timing_accumulator = {}
        self._timing_steps = 0
        self._f0_interp_warned = False
        self._parallel_branch_streams_enabled = os.environ.get("TRIXCODEC_PARALLEL_BRANCH_STREAMS", "0") == "1"
        self._parallel_branch_streams = {}
        self._parallel_branch_streams_announced = False
        self._last_f0_path_debug = {}
        # self.init_mel_transformer(self.cfg.model.mel_params)
        
        # UTMOS and WavLM for validation (optional)
        self.use_val_utmos = getattr(cfg.train, 'use_val_utmos', False)
        self.use_val_wavlm = getattr(cfg.train, 'use_val_wavlm', False)
        self.use_val_wer = getattr(cfg.train, 'use_val_wer', False)
        self.val_hubert_model_name = getattr(cfg.train, 'val_hubert_model', 'facebook/hubert-large-ls960-ft')
        self.utmos_predictor = None
        self.wavlm_model = None
        self.hubert_processor = None
        self.hubert_model = None
        self._wer_runtime_status = "disabled_by_config" if not self.use_val_wer else "pending_first_real_validation"
        self._wer_status_detail = "cfg.train.use_val_wer=False" if not self.use_val_wer else "sanity validation skips WER; first non-sanity validation will decide."
        self._wer_sanity_logged = False
        self._wer_transform = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.SubstituteRegexes({r"[_\u2010-\u2015\u2212-]+": " "}),
            jiwer.SubstituteRegexes({r"[^\w\s\uAC00-\uD7A3]": ""}),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ])
        
        if self.use_val_utmos or self.use_val_wavlm:
            print(colored("Validation will include UTMOS/WavLM metrics", "yellow"))

        # freeze → unfreeze 스케줄 설정 (없으면 비활성)
        self.unfreeze_encoder_step = getattr(cfg.train, 'unfreeze_encoder_step', None)
        self._encoder_frozen = None
        enccfg = self.cfg.model.codec_encoder
        self._freeze_schedule_enabled = self.unfreeze_encoder_step is not None
        use_vqw2v = bool(getattr(enccfg, 'use_vqw2v', False))
        freeze_vqw2v = bool(getattr(enccfg, 'freeze_vqw2v_encoder', True))
        use_w2v2 = bool(getattr(enccfg, 'use_w2v2', False))
        freeze_w2v2 = bool(getattr(enccfg, 'freeze_w2v2_encoder', True))
        use_hubert = bool(getattr(enccfg, 'use_hubert', False))
        freeze_hubert = bool(getattr(enccfg, 'freeze_hubert_encoder', True))
        use_w2vbert2 = bool(getattr(enccfg, 'use_w2vbert2', False))
        freeze_w2vbert2 = bool(getattr(enccfg, 'freeze_w2vbert2_encoder', True))
        use_s3tokenizer = bool(getattr(enccfg, 'use_s3tokenizer', False))
        freeze_s3tokenizer = bool(getattr(enccfg, 'freeze_s3tokenizer_encoder', True))

        # 스케줄 활성 여부: SSL encoder이고 freeze=False면 스케줄 무시
        if use_vqw2v and not freeze_vqw2v:
            self._freeze_schedule_enabled = False
            print(colored("freeze_vqw2v_encoder=False -> disabling unfreeze_encoder_step; vqw2v CodecEnc stays UNFROZEN.", "yellow"))
        elif use_w2v2 and not freeze_w2v2:
            self._freeze_schedule_enabled = False
            print(colored("freeze_w2v2_encoder=False -> disabling unfreeze_encoder_step; w2v2 CodecEnc stays UNFROZEN.", "yellow"))
        elif use_hubert and not freeze_hubert:
            self._freeze_schedule_enabled = False
            print(colored("freeze_hubert_encoder=False -> disabling unfreeze_encoder_step; hubert CodecEnc stays UNFROZEN.", "yellow"))
        elif use_w2vbert2 and not freeze_w2vbert2:
            self._freeze_schedule_enabled = False
            print(colored("freeze_w2vbert2_encoder=False -> disabling unfreeze_encoder_step; w2vbert2 CodecEnc stays UNFROZEN.", "yellow"))
        elif use_s3tokenizer and not freeze_s3tokenizer:
            self._freeze_schedule_enabled = False
            print(colored("freeze_s3tokenizer_encoder=False -> disabling unfreeze_encoder_step; s3tokenizer CodecEnc stays UNFROZEN.", "yellow"))
        else:
            print(colored(f"Encoder unfreeze step: {self.unfreeze_encoder_step}", "yellow"))
    
        # (선택) 동시 지정 안내
        if self.unfreeze_encoder_step is not None and freeze_vqw2v:
            print(colored("Both train.unfreeze_encoder_step and codec_encoder.freeze_vqw2v_encoder are set. Step-based schedule will override encoder freeze after unfreeze step.", "yellow"))
        if self.unfreeze_encoder_step is not None and freeze_w2v2:
            print(colored("Both train.unfreeze_encoder_step and codec_encoder.freeze_w2v2_encoder are set. Step-based schedule will override encoder freeze after unfreeze step.", "yellow"))
        if self.unfreeze_encoder_step is not None and freeze_hubert:
            print(colored("Both train.unfreeze_encoder_step and codec_encoder.freeze_hubert_encoder are set. Step-based schedule will override encoder freeze after unfreeze step.", "yellow"))
        if self.unfreeze_encoder_step is not None and freeze_w2vbert2:
            print(colored("Both train.unfreeze_encoder_step and codec_encoder.freeze_w2vbert2_encoder are set. Step-based schedule will override encoder freeze after unfreeze step.", "yellow"))
        if self.unfreeze_encoder_step is not None and freeze_s3tokenizer:
            print(colored("Both train.unfreeze_encoder_step and codec_encoder.freeze_s3tokenizer_encoder are set. Step-based schedule will override encoder freeze after unfreeze step.", "yellow"))

        # Validation sanity check can run before on_train_start in newer Lightning,
        # so visualization buffers/transforms must exist right after construction.
        self.val_step_plot_outputs = []

        sr, n_fft, hop_length, n_mels = 16000, 1024, 256, 80
        self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )

    def _timing_is_active(self) -> bool:
        if not self._timing_enabled or not self.training:
            return False
        if getattr(self, "global_rank", 0) != 0:
            return False
        return int(self.total_step.item()) >= self._timing_warmup_steps

    def _timing_now(self) -> float:
        if self._timing_sync_cuda and torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _timing_add(self, key: str, elapsed: float) -> None:
        if not self._timing_enabled:
            return
        self._timing_accumulator[key] = self._timing_accumulator.get(key, 0.0) + float(elapsed)

    def _parallel_branch_streams_active(self) -> bool:
        return (
            self._parallel_branch_streams_enabled
            and torch.cuda.is_available()
            and self.device.type == "cuda"
        )

    def _get_parallel_branch_stream(self, device: torch.device) -> torch.cuda.Stream:
        index = device.index if device.index is not None else torch.cuda.current_device()
        stream = self._parallel_branch_streams.get(index)
        if stream is None:
            stream = torch.cuda.Stream(device=index)
            self._parallel_branch_streams[index] = stream
        if not self._parallel_branch_streams_announced and getattr(self, "global_rank", 0) == 0:
            print(
                colored(
                    "TRIXCODEC_PARALLEL_BRANCH_STREAMS=1: overlap speaker and codec encoder branches with a dedicated CUDA stream",
                    "yellow",
                )
            )
            self._parallel_branch_streams_announced = True
        return stream

    def _timing_maybe_log(self) -> None:
        if not self._timing_is_active():
            return

        self._timing_steps += 1
        if self._timing_steps < self._timing_log_every:
            return

        avg = {
            key: value / max(self._timing_steps, 1)
            for key, value in sorted(self._timing_accumulator.items())
        }
        total = avg.get('step_total', 0.0)
        pct = {}
        if total > 0:
            pct = {
                key: (value / total) * 100.0
                for key, value in avg.items()
                if key != 'step_total'
            }
        metrics = " ".join(f"{k}={v:.4f}s" for k, v in avg.items())
        shares = " ".join(
            f"{k}={v:.1f}%"
            for k, v in sorted(pct.items(), key=lambda item: item[1], reverse=True)[:8]
        )
        print(colored(
            f"[timing][step={int(self.total_step.item())}] avg_over={self._timing_steps} {metrics}",
            "cyan",
        ))
        if shares:
            print(colored(f"[timing-share] {shares}", "cyan"))
        self._timing_accumulator = {}
        self._timing_steps = 0

    def _hubert_hub_cache_root(self) -> str:
        hf_hub_cache = os.environ.get("HF_HUB_CACHE")
        if hf_hub_cache:
            return str(hf_hub_cache)
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        return str(os.path.join(hf_home, "hub"))

    def _has_local_hubert_cache(self) -> bool:
        repo_id = str(self.val_hubert_model_name)
        if "/" not in repo_id:
            return False
        cache_dir = os.path.join(self._hubert_hub_cache_root(), f"models--{repo_id.replace('/', '--')}")
        return os.path.exists(cache_dir)

    def _print_wer_status(self, prefix: str) -> None:
        sanity_steps = int(getattr(self.trainer, "num_sanity_val_steps", 0)) if self.trainer is not None else 0
        offline_env = (
            os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        )
        cache_state = "present" if self._has_local_hubert_cache() else "missing"
        print(colored(
            f"[WER] {prefix}: enabled={self.use_val_wer}, status={self._wer_runtime_status}, cache={cache_state}, offline={offline_env}, sanity_steps={sanity_steps}, model={self.val_hubert_model_name}",
            "cyan",
        ))
        if self._wer_status_detail:
            print(colored(f"[WER] detail: {self._wer_status_detail}", "cyan"))

    def on_train_start(self):
        self._print_wer_status("train_start")

        self.val_step_plot_outputs.clear()


    def construct_model(self):
        from pretrained_models.pitch_estimator.fcpe.models import F0ExtractorWrapper
        f0_extractor = F0ExtractorWrapper(self.cfg, device='cpu')

        f0cfg = self.cfg.model.f0_codec
        use_unnormf0_mse_loss = f0cfg.get('use_unnormf0_mse_loss', False)
        if use_unnormf0_mse_loss:
            # assert f0cfg.use_normalized_f0, "use_unnormf0_mse_loss requires use_normalized_f0=True" -> use_normalized_f0=False 이면 f0_decoder_speaker_condition=False 로 두고 input output 전부 unnorm F0 로 학습됨.
            assert not f0cfg.get('use_fcpe_loss', False), "use_unnormf0_mse_loss is incompatible with use_fcpe_loss"
            print(colored("F0 codec: enabling unnormalized F0 MSE loss (targets use log1p raw F0)", "yellow"))
        if not f0cfg.zero_out_all_unvoiced:
            print(colored(f"F0 codec: using 1-dim input (raw f0 with -3 for unvoiced frames)", "yellow"))
        else: print(colored(f"F0 codec: using 2-dim input (raw f0 + vuv); zeroing out all unvoiced frames", "yellow"))
        if not f0cfg.use_normalized_f0:
            assert not f0cfg.zero_out_all_unvoiced, "zero_out_all_unvoiced must be False when not using normalized f0"
        
        # F0 extraction mode logging
        upsample_extracted_f0 = getattr(f0cfg, 'upsample_extracted_f0', True)
        if upsample_extracted_f0:
            print(colored(f"F0 extraction: Mode 1 - Interpolate to audio length during extraction (default)", "cyan"))
        else:
            print(colored(f"F0 extraction: Mode 2 - Extract at original frame rate (no interpolation)", "cyan", attrs=['bold']))
            print(colored(f"  → F0 output shape will be [B, T_frame] instead of [B, T_audio]", "yellow"))
            print(colored(f"  → Make sure encoder output length matches F0 frame length", "yellow"))
        
        spkcfg = self.cfg.model.speaker_encoder
        speaker_token_dim = resolve_speaker_encoder_token_dim(spkcfg)

        # Check if using CodecEncoder/Decoder structure for F0
        use_codec_structure = f0cfg.get('use_codec_structure', False)
        if use_unnormf0_mse_loss and not use_codec_structure:
            raise AssertionError("use_unnormf0_mse_loss requires f0_codec.use_codec_structure=True to enable speaker conditioning")
        
        # Initialize variables that will be used later for joint dimension calculation
        encoder_out_channels = None
        decoder_in_channels = None
        
        if use_codec_structure:
            print(colored(f"F0 codec: using CodecEncoder/CodecDecoder structure", "cyan", attrs=['bold']))
            from vq.f0.codec_structure import F0CodecEncoder, F0CodecDecoder, build_channel_schedule
            f0_encoder_cfg = F0CodecEncoderConfig.from_f0_cfg(f0cfg)
            f0_decoder_cfg = F0CodecDecoderConfig.from_f0_cfg(f0cfg)
            
            input_channels = 2 if f0cfg.zero_out_all_unvoiced else 1
            output_channels = 2 if f0cfg.zero_out_all_unvoiced else 1
            
            encoder_stage_channels = build_channel_schedule(
                f0_encoder_cfg.ngf,
                len(f0_encoder_cfg.up_ratios),
                max_channels=f0cfg.get('max_channels', None),
            )
            decoder_stage_channels = build_channel_schedule(
                f0_decoder_cfg.ngf,
                len(f0_decoder_cfg.up_ratios),
                max_channels=f0cfg.get('max_channels', None),
            )

            # Auto-calculate encoder_out_channels if not specified.
            # Default: hidden top channel after all EncoderBlocks with max_channels cap applied.
            encoder_out_channels = getattr(f0cfg, 'encoder_out_channels', None)
            if encoder_out_channels is None:
                encoder_out_channels = encoder_stage_channels[-1]
            
            # Auto-calculate decoder_in_channels if not specified
            decoder_in_channels = getattr(f0cfg, 'decoder_in_channels', None)
            if decoder_in_channels is None: decoder_in_channels = encoder_out_channels
            
            # decoder_upsample_initial_channel: must match the decoder's top hidden width.
            decoder_upsample_initial_channel = getattr(f0cfg, 'decoder_upsample_initial_channel', None)
            if decoder_upsample_initial_channel is None:
                decoder_upsample_initial_channel = decoder_stage_channels[-1]
            
            f0_encoder = F0CodecEncoder(
                input_channels=input_channels,
                ngf=f0_encoder_cfg.ngf,
                max_channels=f0cfg.get('max_channels', None),
                use_rnn=f0_encoder_cfg.use,
                rnn_bidirectional=f0_encoder_cfg.bidirectional,
                rnn_num_layers=f0_encoder_cfg.num_layers,
                rnn_type=f0_encoder_cfg.backbone,
                rnn_mamba=f0_encoder_cfg.mamba,
                up_ratios=f0_encoder_cfg.up_ratios,
                dilations=f0_encoder_cfg.dilations,
                out_channels=encoder_out_channels,
                activation_type=f0_encoder_cfg.activation_type,
                leaky_relu_params=f0_encoder_cfg.leaky_relu_params,
            )
            
            # F0 decoder speaker conditioning: enable when using normalized f0 + fcpe loss
            # This helps the model learn speaker-specific pitch ranges
            use_normalized_f0 = f0cfg.get('use_normalized_f0', False)
            use_fcpe_loss = f0cfg.get('use_fcpe_loss', False)
            legacy_f0_spk_cond = use_normalized_f0 and (use_fcpe_loss or use_unnormf0_mse_loss)
            f0_spk_cond_cfg = F0CodecSpeakerConditionConfig.from_f0_cfg(f0cfg)
            f0_decoder_use_concat = f0_spk_cond_cfg.resolve_concat(legacy_f0_spk_cond)
            f0_decoder_use_film = f0_spk_cond_cfg.resolve_film(legacy_f0_spk_cond)
            f0_decoder_speaker_condition = f0_decoder_use_concat or f0_decoder_use_film
            decoder_use_mhca = f0_spk_cond_cfg.resolve_mhca(legacy_f0_spk_cond)
            if (
                f0cfg.get('spk_cond', None) is None
                and f0cfg.get('decoder_use_mhca', f0cfg.get('use_mhca', False))
                and not decoder_use_mhca
            ):
                print(colored(f"Warning: decoder_use_mhca=True but speaker_condition=False. Disabling MHCA for F0 decoder.", "yellow"))
            configured_f0_mhca_key_dim = f0_spk_cond_cfg.key_dim
            f0_mhca_key_dim = speaker_token_dim
            f0_spk_cond_node = f0cfg.get('spk_cond', None)
            f0_mhca_use_sdpa = (
                f0_spk_cond_node.get('use_sdpa', f0cfg.get('decoder_mhca_use_sdpa', None))
                if f0_spk_cond_node is not None else f0cfg.get('decoder_mhca_use_sdpa', None)
            )
            if configured_f0_mhca_key_dim is not None and int(configured_f0_mhca_key_dim) != int(speaker_token_dim):
                print(colored(
                    f"Overriding f0 decoder MHCA key dim {configured_f0_mhca_key_dim} -> {speaker_token_dim} "
                    f"to match final speaker token dim",
                    "yellow",
                ))

            f0_decoder = F0CodecDecoder(
                in_channels=decoder_in_channels,
                upsample_initial_channel=decoder_upsample_initial_channel,
                output_channels=output_channels,
                ngf=f0_decoder_cfg.ngf,
                max_channels=f0cfg.get('max_channels', None),
                use_rnn=f0_decoder_cfg.use,
                rnn_bidirectional=f0_decoder_cfg.bidirectional,
                rnn_num_layers=f0_decoder_cfg.num_layers,
                rnn_type=f0_decoder_cfg.backbone,
                rnn_mamba=f0_decoder_cfg.mamba,
                up_ratios=f0_decoder_cfg.up_ratios,
                dilations=f0_decoder_cfg.dilations,
                activation_type=f0_decoder_cfg.activation_type,
                leaky_relu_params=f0_decoder_cfg.leaky_relu_params,
                use_fcpe_loss=use_fcpe_loss,
                fcpe_out_dims=f0cfg.get('fcpe_out_dims', 360),
                speaker_condition=f0_decoder_speaker_condition,
                condition_dim=spkcfg.out_dim if f0_decoder_speaker_condition else 0,
                use_spk_concat=f0_decoder_use_concat,
                use_spk_film=f0_decoder_use_film,
                use_mhca=decoder_use_mhca,
                mhca_num_heads=f0_spk_cond_cfg.num_heads,
                mhca_dropout=f0_spk_cond_cfg.dropout,
                mhca_key_dim=f0_mhca_key_dim,
                mhca_use_sdpa=f0_mhca_use_sdpa,
                use_split_condition_optimization=True,
            )
            self.f0_decoder_use_mhca = decoder_use_mhca
        else:
            print(colored(f"F0 codec: using original F0Encoder/F0Decoder structure", "yellow"))
            f0_encoder = F0Encoder(
                # input_emb_width=f0cfg.input_emb_width,
                input_emb_width= 2 if f0cfg.zero_out_all_unvoiced else 1,
                output_emb_width=f0cfg.output_emb_width,
                levels=f0cfg.levels,
                downs_t=f0cfg.downs_t,
                strides_t=f0cfg.enc_strides_t,
                width=f0cfg.width,
                depth=f0cfg.depth,
                m_conv=f0cfg.m_conv,
                channel_growth_rate=f0cfg.channel_growth_rate,
                dilation_growth_rate=f0cfg.dilation_growth_rate,
            )

            f0_decoder = F0Decoder(
                # input_emb_width=f0cfg.input_emb_width,
                input_emb_width= 2 if f0cfg.zero_out_all_unvoiced else 1,
                output_emb_width=f0cfg.output_emb_width,
                levels=f0cfg.levels,
                downs_t=f0cfg.downs_t,
                strides_t=f0cfg.dec_strides_t,
                width=f0cfg.width,
                depth=f0cfg.depth,
                m_conv=f0cfg.m_conv,
                channel_growth_rate=f0cfg.channel_growth_rate,
                dilation_growth_rate=f0cfg.dilation_growth_rate,
            )
            self.f0_decoder_use_mhca = False

        
        # Resolve WavLM checkpoint path (convert relative to absolute using original cwd)
        wavlm_ckpt = spkcfg.get('wavlm_checkpoint', None)
        if wavlm_ckpt is not None and not os.path.isabs(wavlm_ckpt):
            wavlm_ckpt = os.path.join(self.ocwd, wavlm_ckpt)
        
        speaker_encoder = SpeakerEncoder(
            mel_params=getattr(self.cfg.model, 'mel_params', None),
            speaker_encoder_type=spkcfg.speaker_encoder_type,
            use_perceiver_encoder=spkcfg.use_perceiver_encoder,
            use_memory_cattn=spkcfg.get('use_memory_cattn', False),
            input_dim=spkcfg.get('input_dim', 100),
            out_dim=spkcfg.out_dim,
            latent_dim=spkcfg.latent_dim,
            token_num=spkcfg.token_num,
            discretize_memory_attn=spkcfg.get('discretize_memory_attn', False),
            memory_attn_codebook_size=spkcfg.get('memory_attn_codebook_size', 128),
            memory_attn_share_across_heads=spkcfg.get('memory_attn_share_across_heads', True),
            norm_layer=spkcfg.get('norm_layer', 'bn'),
            use_normalized_f0=f0cfg.get('use_normalized_f0', False),
            use_quantizer=spkcfg.get('use_quantizer', True),
            stack=spkcfg.get('stack', None),
            wavlm_checkpoint=wavlm_ckpt,
            wavlm_output_layer=spkcfg.get('wavlm_output_layer', 6),
            freeze_wavlm=spkcfg.get('freeze_wavlm', True),
            perceiver_use_flash_attn=spkcfg.get('perceiver_use_flash_attn', False),
            frozen_wavlm_inference_mode=spkcfg.get('frozen_wavlm_inference_mode', False),
            frozen_wavlm_force_fp32=spkcfg.get('frozen_wavlm_force_fp32', False),
            **build_speaker_quantizer_kwargs(spkcfg),
        )
        enccfg = self.cfg.model.codec_encoder
        print(colored(f"codec_encoder.activation_type: {enccfg.get('activation_type', 'SnakeBeta')}", "cyan"))
        use_vqw2v = bool(getattr(enccfg, 'use_vqw2v', False))
        use_w2v2 = bool(getattr(enccfg, 'use_w2v2', False))
        use_hubert = bool(getattr(enccfg, 'use_hubert', False))
        use_w2vbert2 = bool(getattr(enccfg, 'use_w2vbert2', False))
        use_s3tokenizer = bool(getattr(enccfg, 'use_s3tokenizer', False))
        if (use_vqw2v + use_w2v2 + use_hubert + use_w2vbert2 + use_s3tokenizer) > 1:
            raise ValueError(
                "Only one of codec_encoder.use_vqw2v / use_w2v2 / use_hubert / use_w2vbert2 / use_s3tokenizer can be True."
            )

        if use_vqw2v:
            s3prl_cache_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "etc", "torch_hub_cache", "s3prl"
            )
            _set_s3prl_download_dir(s3prl_cache_dir)
            s3prl_upstream = S3PRLUpstream("vq_wav2vec_kmeans")
            model = s3prl_upstream.upstream.model
            use_vqw2v_continuous = bool(getattr(enccfg, 'use_vqw2v_continuous', True))
            encoder = VQW2VEncoder(
                feature_extractor=model.feature_extractor,
                vector_quantizer=model.vector_quantizer,
                use_continuous=use_vqw2v_continuous,
                frozen_feature_extractor_force_fp32=enccfg.get('frozen_upstream_force_fp32', False),
            )
            # 훅 제거: 수치 영향 없음, 메모리 누수 방지
            for m in [model.feature_extractor, model.vector_quantizer]:
                if hasattr(m, "_forward_hooks"): m._forward_hooks.clear()
                if hasattr(m, "_forward_pre_hooks"): m._forward_pre_hooks.clear()
                if hasattr(m, "_backward_hooks"): m._backward_hooks.clear()
            if use_vqw2v_continuous:
                print(colored("VQ-Wav2Vec: using CONTINUOUS features (pre-quantization)", "cyan"))
            else:
                print(colored("VQ-Wav2Vec: using DISCRETE features (quantized via pretrained codebook)", "cyan", attrs=["bold"]))

            if bool(getattr(enccfg, 'freeze_vqw2v_encoder', True)):
                print(colored("Freezing VQ-Wav2Vec encoder", "yellow"))
                for p in encoder.parameters():
                    p.requires_grad = False
                encoder.eval()
            else:
                for p in encoder.parameters():
                    p.requires_grad = True 

        elif use_w2v2:
            # HuggingFace Wav2Vec2 feature extractor / transformer hidden states
            from transformers import Wav2Vec2Model
            model_name = getattr(enccfg, 'w2v2_model_name', 'facebook/wav2vec2-large')
            use_transformer = bool(getattr(enccfg, 'w2v2_use_transformer', True))
            w2v2 = Wav2Vec2Model.from_pretrained(model_name)
            if use_transformer:
                layer_indices = getattr(enccfg, 'w2v2_hidden_layers', [11, 14, 16])
                mode = getattr(enccfg, 'w2v2_hidden_mode', 'avg')
                encoder = W2V2HiddenStateEncoder(w2v2, layer_indices=layer_indices, mode=mode)
            else:
                encoder = w2v2.feature_extractor

            if bool(getattr(enccfg, 'freeze_w2v2_encoder', True)):
                print(colored(f"Freezing Wav2Vec2 ({'transformer' if use_transformer else 'feature_extractor'}) ({model_name})", "yellow"))
                for p in w2v2.parameters():
                    p.requires_grad = False
                w2v2.eval()
            else:
                for p in w2v2.parameters():
                    p.requires_grad = True
        
        elif use_hubert:
            # HuggingFace HuBERT feature extractor / transformer hidden states
            from transformers import HubertModel
            model_name = getattr(enccfg, 'hubert_model_name', 'facebook/hubert-large-ls960-ft')
            use_transformer = bool(getattr(enccfg, 'hubert_use_transformer', True))
            hubert = HubertModel.from_pretrained(model_name)
            if use_transformer:
                layer_indices = getattr(enccfg, 'hubert_hidden_layers', [11, 14, 16])
                mode = getattr(enccfg, 'hubert_hidden_mode', 'avg')
                # Reuse the same hidden-state adapter (HuBERT output format matches W2V2)
                encoder = W2V2HiddenStateEncoder(hubert, layer_indices=layer_indices, mode=mode)
            else:
                encoder = hubert.feature_extractor

            if bool(getattr(enccfg, 'freeze_hubert_encoder', True)):
                print(colored(f"Freezing HuBERT ({'transformer' if use_transformer else 'feature_extractor'}) ({model_name})", "yellow"))
                for p in hubert.parameters():
                    p.requires_grad = False
                hubert.eval()
            else:
                for p in hubert.parameters():
                    p.requires_grad = True

        elif use_w2vbert2:
            # HuggingFace Wav2Vec2-BERT 2.0 (facebook/w2v-bert-2.0) - uses feature extractor for mel input
            # AutoProcessor loads tokenizer (ASR) which facebook/w2v-bert-2.0 lacks; use AutoFeatureExtractor
            from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
            model_name = getattr(enccfg, 'w2vbert2_model_name', 'facebook/w2v-bert-2.0')
            processor = AutoFeatureExtractor.from_pretrained(model_name)
            w2vbert2 = Wav2Vec2BertModel.from_pretrained(model_name)
            layer_indices = getattr(enccfg, 'w2vbert2_hidden_layers', (11, 18, 23))
            mode = getattr(enccfg, 'w2vbert2_hidden_mode', 'avg')
            sampling_rate = getattr(enccfg, 'w2vbert2_sampling_rate', 16000)
            normalize = getattr(enccfg, 'w2vbert2_hidden_normalize', True)
            encoder = W2VBert2HiddenStateEncoder(
                w2vbert2, processor,
                layer_indices=layer_indices,
                mode=mode,
                sampling_rate=sampling_rate,
                normalize=normalize,
            )
            if bool(getattr(enccfg, 'freeze_w2vbert2_encoder', True)):
                print(colored(f"Freezing Wav2Vec2-BERT 2.0 ({model_name})", "yellow"))
                for p in w2vbert2.parameters():
                    p.requires_grad = False
                w2vbert2.eval()
            else:
                for p in w2vbert2.parameters():
                    p.requires_grad = True

        elif use_s3tokenizer:
            # S3Tokenizer (CosyVoice) - mel-based encoder
            model_name = getattr(enccfg, 's3tokenizer_model_name', 'speech_tokenizer_v1')
            sampling_rate = getattr(enccfg, 's3tokenizer_sampling_rate', 16000)
            encoder = S3TokenizerEncoder(
                model_name=model_name,
                sampling_rate=sampling_rate,
                use_continuous=True,
            )
            if bool(getattr(enccfg, 'freeze_s3tokenizer_encoder', True)):
                print(colored(f"Freezing S3Tokenizer ({model_name})", "yellow"))
                for p in encoder.parameters():
                    p.requires_grad = False
                encoder.eval()
            else:
                for p in encoder.parameters():
                    p.requires_grad = True

        else:
            encoder = CodecEncoder(
                        ngf=enccfg.ngf,
                        temporal=CodecTemporalConfig.from_encoder_cfg(enccfg),
                        up_ratios=enccfg.up_ratios,
                        dilations=enccfg.dilations,
                        out_channels=enccfg.out_channels,

                        speaker_condition=enccfg.encoder_speaker_condition,
                        condition_dim=enccfg.condition_dim,
                        snake_logscale=enccfg.snake_logscale,
                        activation_type=enccfg.get('activation_type', 'SnakeBeta'),
                        leaky_relu_params=enccfg.get('leaky_relu_params', None),
                        snake_lite_taylor_degree=enccfg.get('snake_lite_taylor_degree', 8),
                    )
        if enccfg.get('up_ratios', None) is not None and use_vqw2v:
            print(colored(f"Using VQ-Wav2Vec Codec Encoder Wrapper", "red"))
            encoder = VQW2VCodecEncoderWrapper(
                temporal=CodecTemporalConfig.from_encoder_cfg(enccfg),
                encoder=encoder,
                up_ratios=enccfg.up_ratios,
                dilations=enccfg.dilations,
                ngf=512,
                out_channels=enccfg.out_channels,
                activation_type=enccfg.get('activation_type', 'SnakeBeta'),
                leaky_relu_params=enccfg.get('leaky_relu_params', None),
                snake_lite_taylor_degree=enccfg.get('snake_lite_taylor_degree', 8),)
        if use_w2v2:
            # Always use wrapper: projects w2v2 output to out_channels.
            # When up_ratios=None: no downsampling (keep w2v2 hop).
            print(colored(f"Using Wav2Vec2 Codec Encoder Wrapper (up_ratios={enccfg.get('up_ratios')})", "red"))
            w2v2_ngf = getattr(enccfg, 'w2v2_ngf', None)
            if w2v2_ngf is None:
                w2v2_ngf = 1024 if bool(getattr(enccfg, 'w2v2_use_transformer', True)) else 512
            encoder = W2V2CodecEncoderWrapper(
                temporal=CodecTemporalConfig.from_encoder_cfg(enccfg),
                encoder=encoder,
                up_ratios=enccfg.get('up_ratios'),
                dilations=enccfg.get('dilations', [1, 3, 9]),
                ngf=int(w2v2_ngf),
                out_channels=enccfg.out_channels,
                activation_type=enccfg.get('activation_type', 'SnakeBeta'),
                leaky_relu_params=enccfg.get('leaky_relu_params', None),
                encoder_force_fp32=enccfg.get('frozen_upstream_force_fp32', False),
                snake_lite_taylor_degree=enccfg.get('snake_lite_taylor_degree', 8),
            )
        if use_hubert:
            # Same wrapper behavior as W2V2; only the upstream encoder differs.
            print(colored(f"Using HuBERT Codec Encoder Wrapper (up_ratios={enccfg.get('up_ratios')})", "red"))
            hubert_ngf = getattr(enccfg, 'hubert_ngf', None)
            if hubert_ngf is None:
                hubert_ngf = 1024 if bool(getattr(enccfg, 'hubert_use_transformer', True)) else 512
            encoder = HubertCodecEncoderWrapper(
                temporal=CodecTemporalConfig.from_encoder_cfg(enccfg),
                encoder=encoder,
                up_ratios=enccfg.get('up_ratios'),
                dilations=enccfg.get('dilations', [1, 3, 9]),
                ngf=int(hubert_ngf),
                out_channels=enccfg.out_channels,
                activation_type=enccfg.get('activation_type', 'SnakeBeta'),
                leaky_relu_params=enccfg.get('leaky_relu_params', None),
                encoder_force_fp32=enccfg.get('frozen_upstream_force_fp32', False),
                snake_lite_taylor_degree=enccfg.get('snake_lite_taylor_degree', 8),
            )
        if use_w2vbert2:
            print(colored(f"Using Wav2Vec2-BERT 2.0 Codec Encoder Wrapper (up_ratios={enccfg.get('up_ratios')})", "red"))
            w2vbert2_ngf = getattr(enccfg, 'w2vbert2_ngf', None) or 1024
            encoder = W2VBert2CodecEncoderWrapper(
                temporal=CodecTemporalConfig.from_encoder_cfg(enccfg),
                encoder=encoder,
                up_ratios=enccfg.get('up_ratios'),
                dilations=enccfg.get('dilations', [1, 3, 9]),
                ngf=int(w2vbert2_ngf),
                out_channels=enccfg.out_channels,
                activation_type=enccfg.get('activation_type', 'SnakeBeta'),
                leaky_relu_params=enccfg.get('leaky_relu_params', None),
                encoder_force_fp32=enccfg.get('frozen_upstream_force_fp32', False),
                snake_lite_taylor_degree=enccfg.get('snake_lite_taylor_degree', 8),
            )
        if use_s3tokenizer:
            print(colored(f"Using S3Tokenizer Codec Encoder Wrapper (up_ratios={enccfg.get('up_ratios')})", "red"))
            s3_ngf = getattr(enccfg, 's3tokenizer_ngf', 1280)
            encoder = S3TokenizerCodecEncoderWrapper(
                temporal=CodecTemporalConfig.from_encoder_cfg(enccfg),
                encoder=encoder,
                up_ratios=enccfg.get('up_ratios'),
                dilations=enccfg.get('dilations', [1, 3, 9]),
                ngf=int(s3_ngf),
                out_channels=enccfg.out_channels,
                activation_type=enccfg.get('activation_type', 'SnakeBeta'),
                leaky_relu_params=enccfg.get('leaky_relu_params', None),
                encoder_force_fp32=enccfg.get('frozen_upstream_force_fp32', False),
                snake_lite_taylor_degree=enccfg.get('snake_lite_taylor_degree', 8),
            )

        deccfg = self.cfg.model.codec_decoder
        decoder_f0_cond_cfg = CodecDecoderF0ConditionConfig.from_decoder_cfg(deccfg)
        decoder_spk_cond_cfg = CodecDecoderSpeakerConditionConfig.from_decoder_cfg(deccfg)
        decoder_vocos_cfg = CodecDecoderVocosConfig.from_decoder_cfg(deccfg)
        decoder_spk_cond_node = deccfg.get('spk_cond', None)
        decoder_has_explicit_spk_cond = decoder_spk_cond_node is not None
        decoder_mhca_use_sdpa = (
            decoder_spk_cond_node.get('use_sdpa', deccfg.get('mhca_use_sdpa', None))
            if decoder_spk_cond_node is not None else deccfg.get('mhca_use_sdpa', None)
        )
        decoder_use_stage_speaker_film = (
            decoder_spk_cond_cfg.use_film
            if decoder_has_explicit_spk_cond
            else (
                decoder_spk_cond_cfg.use_film
                or deccfg.get('use_stage_speaker_film', False)
            )
        )
        if (
            decoder_has_explicit_spk_cond
            and deccfg.get('use_stage_speaker_film', False)
            and not decoder_spk_cond_cfg.use_film
        ):
            print(colored(
                "codec_decoder.use_stage_speaker_film=True is ignored because "
                "codec_decoder.spk_cond.type does not include 'film'.",
                "yellow",
            ))
        decoder_f0_stage_dims = None
        if deccfg.get('decoder_type', 'default').lower() in ['rndvoc', 'vocos', 'vocosformer', 'apnformer'] and decoder_f0_cond_cfg.use_concat:
            stage_count = int(
                deccfg.get(
                    'rndvoc_null_nstage',
                    decoder_vocos_cfg.num_layers,
                )
            ) if deccfg.get('decoder_type', 'default').lower() == 'rndvoc' else int(decoder_vocos_cfg.num_layers)
            decoder_f0_stage_dims = self._build_rndvoc_f0_stage_dims(
                f0_decoder=f0_decoder,
                f0cfg=f0cfg,
                null_nstage=stage_count,
            )
        # Get f0_width_list from f0_encoder (compatible with both F0Encoder and F0CodecEncoder)
        f0_width_list = f0_encoder.width_list
        decoder = CodecDecoder(
                    in_channels=deccfg.in_channels,
                    upsample_initial_channel=deccfg.get('upsample_initial_channel', 1536),

                    ngf=deccfg.get('ngf', 48),
                    temporal=CodecDecoderTemporalConfig.from_decoder_cfg(deccfg),
                    up_ratios=deccfg.get('up_ratios', (5, 5, 2, 2, 2)),
                    dilations=deccfg.get('dilations', (1, 3, 9)),

                    quantizer_type=deccfg.quantizer_type,
                    freeze_kvq_emb=deccfg.get('freeze_kvq_emb', True),

                    vq_num_quantizers=deccfg.get('vq_num_quantizers', 1),
                    vq_dim=deccfg.vq_dim,
                    vq_commit_weight=deccfg.vq_commit_weight,
                    vq_full_commit_loss=deccfg.get('vq_full_commit_loss', False),
                    quantizer_force_fp32=deccfg.get('quantizer_force_fp32', False),
                    codebook_size=deccfg.codebook_size,
                    codebook_dim=deccfg.get('codebook_dim', 8),
                    use_vqw2v_encoder=bool(use_vqw2v or use_w2v2 or use_hubert or use_w2vbert2 or use_s3tokenizer),
                    freeze_vqw2v_encoder=bool(
                        (use_vqw2v and bool(getattr(enccfg, 'freeze_vqw2v_encoder', True))) or
                        (use_w2v2 and bool(getattr(enccfg, 'freeze_w2v2_encoder', True))) or
                        (use_hubert and bool(getattr(enccfg, 'freeze_hubert_encoder', True))) or
                        (use_w2vbert2 and bool(getattr(enccfg, 'freeze_w2vbert2_encoder', True))) or
                        (use_s3tokenizer and bool(getattr(enccfg, 'freeze_s3tokenizer_encoder', True)))
                    ),
                    use_vqw2v_embed=deccfg.get('use_vqw2v_embed', True),

                    speaker_condition=deccfg.speaker_condition,
                    condition_dim=deccfg.condition_dim,
                    activation_type=deccfg.get('activation_type', 'SnakeBeta'),
                    leaky_relu_params=deccfg.get('leaky_relu_params', None),
                    snake_logscale=deccfg.get('snake_logscale', True),
                    snake_lite_taylor_degree=deccfg.get('snake_lite_taylor_degree', 8),
                    simvq_linear_layer_type=deccfg.get('simvq_linear_layer_type', 'linear'),
                    ema_decay=deccfg.get('ema_decay', 0.0),

                    f0_condition=decoder_f0_cond_cfg.use_concat,
                    f0_start_layer=decoder_f0_cond_cfg.start_layer,
                    f0_end_layer=decoder_f0_cond_cfg.end_layer,
                    f0_every=decoder_f0_cond_cfg.every,
                    f0_width_list=f0_width_list,
                    f0_speaker_condition=deccfg.get('f0_speaker_condition', False),
                    use_stage_speaker_film=decoder_use_stage_speaker_film,
                    
                    use_mhca=decoder_spk_cond_cfg.use_mhca,
                    spk_cond_use_concat=decoder_spk_cond_cfg.use_concat,
                    mhca_num_heads=decoder_spk_cond_cfg.num_heads,
                    mhca_dropout=decoder_spk_cond_cfg.dropout,
                    mhca_key_dim=speaker_token_dim,
                    mhca_use_sdpa=decoder_mhca_use_sdpa,
                    mhca_start_layer=decoder_spk_cond_cfg.start_layer,
                    mhca_end_layer=decoder_spk_cond_cfg.end_layer,
                    mhca_every=decoder_spk_cond_cfg.every,
                    decoder_type=deccfg.get('decoder_type', 'default'),
                    rndvoc_sampling_rate=self.cfg.preprocess.audio.sr,
                    rndvoc_n_fft=deccfg.get('rndvoc_n_fft', None),
                    rndvoc_hop_size=deccfg.get('rndvoc_hop_size', None),
                    rndvoc_win_size=deccfg.get('rndvoc_win_size', None),
                    rndvoc_input_channel=deccfg.get('rndvoc_input_channel', 256),
                    rndvoc_hidden_channel=deccfg.get('rndvoc_hidden_channel', 256),
                    rndvoc_squeeze_size=deccfg.get('rndvoc_squeeze_size', 64),
                    rndvoc_null_nstage=deccfg.get('rndvoc_null_nstage', 6),
                    rndvoc_nrep=deccfg.get('rndvoc_nrep', 2),
                    rndvoc_kernel_size=deccfg.get('rndvoc_kernel_size', 7),
                    rndvoc_causal=deccfg.get('rndvoc_causal', False),
                    rndvoc_use_rnd=deccfg.get('rndvoc_use_rnd', True),
                    rndvoc_time_type=deccfg.get('rndvoc_time_type', 'convnext_v2'),
                    rndvoc_freq_type=deccfg.get('rndvoc_freq_type', 'shuffler'),
                    rndvoc_f0_stage_dims=decoder_f0_stage_dims,
                    vocos_sampling_rate=self.cfg.preprocess.audio.sr,
                    vocos_n_fft=decoder_vocos_cfg.n_fft,
                    vocos_hop_size=decoder_vocos_cfg.hop_size,
                    vocos_win_size=decoder_vocos_cfg.win_size,
                    vocos_dim=decoder_vocos_cfg.dim,
                    vocos_intermediate_dim=decoder_vocos_cfg.intermediate_dim,
                    vocos_num_layers=decoder_vocos_cfg.num_layers,
                    vocos_kernel_size=decoder_vocos_cfg.kernel_size,
                    vocos_padding=decoder_vocos_cfg.padding,
                    vocos_f0_stage_dims=decoder_f0_stage_dims,
                    vocos_backbone=decoder_vocos_cfg.backbone,
                    vocos_num_heads=decoder_vocos_cfg.num_heads,
                    vocos_window_size=decoder_vocos_cfg.window_size,
                    vocos_attn_window_size=decoder_vocos_cfg.attn_window_size,
                    vocos_attention_impl=decoder_vocos_cfg.attention_impl,
                    vocos_use_rope=decoder_vocos_cfg.use_rope,
                    vocos_rope_base=decoder_vocos_cfg.rope_base,
                    vocos_max_position_embeddings=decoder_vocos_cfg.max_position_embeddings,
                    vocos_qk_norm=decoder_vocos_cfg.qk_norm,
                    vocos_norm_type=decoder_vocos_cfg.norm_type,
                    vocos_norm_eps=decoder_vocos_cfg.norm_eps,
                    vocos_dropout=decoder_vocos_cfg.dropout,
                    vocos_ffn_type=decoder_vocos_cfg.ffn_type,
                    vocos_ffn_mult=decoder_vocos_cfg.ffn_mult,
                    vocos_layerscale_gamma_init=decoder_vocos_cfg.layerscale_gamma_init,
                    vocos_apn_head_hidden_dim=decoder_vocos_cfg.apn_head_hidden_dim,
                    vocos_apn_phase_norm_eps=decoder_vocos_cfg.apn_phase_norm_eps,
                    use_gradient_checkpointing=deccfg.get('use_gradient_checkpointing', False),
                    use_split_condition_optimization=True,
                    # zero_out_all_unvoiced=f0cfg.zero_out_all_unvoiced,
                )


        mpdcfg = self.cfg.model.mpd
        use_bigvsan = self._effective_gan_mode() == 'bigvsan'
        mpd = HiFiGANMultiPeriodDiscriminator(
                    periods=mpdcfg.periods,
                    max_downsample_channels=mpdcfg.max_downsample_channels,
                    channels=mpdcfg.channels,
                    channel_increasing_factor=mpdcfg.channel_increasing_factor,
                    use_weight_norm=mpdcfg.get('use_weight_norm', True),
                    use_san=use_bigvsan,
                )
        spec_disc = None
        spec_disc_type = self._spec_discriminator_type()
        if use_bigvsan and spec_disc_type in {'bigvgan_mrd', 'vocos_mrd', 'apnet2_mrd'}:
            raise ValueError("BigVSAN loss is only supported with spec_discriminator.type=mstft or none.")
        if spec_disc_type == 'mstft':
            mstftcfg = self._resolved_mstft_cfg(spec_disc_type)
            force_legacy_from_preset = self._adversarial_preset() is not None
            if force_legacy_from_preset or mstftcfg.get('use', True):
                spec_disc = LegacySpecDiscriminator(
                            stft_params=mstftcfg.stft_params,
                            in_channels=mstftcfg.in_channels,
                            out_channels=mstftcfg.out_channels,
                            kernel_sizes=mstftcfg.kernel_sizes,
                            channels=mstftcfg.channels,
                            max_downsample_channels=mstftcfg.max_downsample_channels,
                            downsample_scales=mstftcfg.downsample_scales,
                            use_weight_norm=mstftcfg.use_weight_norm,
                        )
        elif spec_disc_type == 'bigvgan_mrd':
            spec_cfg = self.cfg.model.get('spec_discriminator', {})
            mrdcfg = spec_cfg.get('bigvgan_mrd', {})
            spec_disc = BigVGANMultiResolutionDiscriminator(
                resolutions=tuple(tuple(int(v) for v in resolution) for resolution in mrdcfg.get(
                    'resolutions',
                    ((1024, 120, 600), (2048, 240, 1200), (512, 50, 240)),
                )),
                channels=int(mrdcfg.get('channels', 32)),
                lrelu_slope=float(mrdcfg.get('lrelu_slope', 0.1)),
            )
        elif spec_disc_type == 'vocos_mrd':
            spec_cfg = self.cfg.model.get('spec_discriminator', {})
            mrdcfg = spec_cfg.get('vocos_mrd', {})
            spec_disc = VocosMultiResolutionDiscriminator(
                fft_sizes=tuple(int(v) for v in mrdcfg.get('fft_sizes', (2048, 1024, 512))),
                channels=int(mrdcfg.get('channels', 32)),
                hop_factor=float(mrdcfg.get('hop_factor', 0.25)),
                bands=tuple(tuple(float(v) for v in band) for band in mrdcfg.get(
                    'bands',
                    ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)),
                )),
                lrelu_slope=float(mrdcfg.get('lrelu_slope', 0.1)),
            )
        elif spec_disc_type == 'apnet2_mrd':
            spec_cfg = self.cfg.model.get('spec_discriminator', {})
            mrdcfg = spec_cfg.get('apnet2_mrd', {})
            spec_disc = APNet2MultiResolutionDiscriminator(
                resolutions=tuple(tuple(int(v) for v in resolution) for resolution in mrdcfg.get(
                    'resolutions',
                    ((1024, 256, 1024), (2048, 512, 2048), (512, 128, 512)),
                )),
                channels=int(mrdcfg.get('channels', 64)),
                lrelu_slope=float(mrdcfg.get('lrelu_slope', 0.1)),
            )
        cqt_disc = None
        cqtcfg = self.cfg.model.get('cqt_discriminator', None)
        if cqtcfg is not None and bool(self._resolved_adversarial_setup()['use_cqt']):
            cqt_disc = MultiScaleSubbandCQTDiscriminator(
                sample_rate=self.cfg.preprocess.audio.sr,
                cqt_params=cqtcfg.get('cqt_params', None),
                in_channels=cqtcfg.get('in_channels', 1),
                out_channels=cqtcfg.get('out_channels', 1),
                kernel_sizes=tuple(cqtcfg.get('kernel_sizes', (5, 3))),
                channels=cqtcfg.get('channels', 32),
                max_downsample_channels=cqtcfg.get('max_downsample_channels', 512),
                downsample_scales=tuple(cqtcfg.get('downsample_scales', (2, 2, 2))),
                use_weight_norm=cqtcfg.get('use_weight_norm', True),
                filters=cqtcfg.get('filters', None),
                max_filters=cqtcfg.get('max_filters', None),
                filters_scale=cqtcfg.get('filters_scale', 1),
                dilations=tuple(cqtcfg.get('dilations', (1, 2, 4))),
                stride=tuple(cqtcfg.get('stride', (1, 2))),
                norm=str(cqtcfg.get('norm', 'weight_norm' if cqtcfg.get('use_weight_norm', True) else 'none')),
                activation=str(cqtcfg.get('activation', 'LeakyReLU')),
                activation_params=cqtcfg.get('activation_params', {'negative_slope': 0.1}),
                use_san=use_bigvsan,
            )

        # TriXCodec: joint quantization of content and F0
        if use_codec_structure:
            # F0CodecEncoder: use encoder_out_channels (auto-calculated or specified)
            f0_dim = encoder_out_channels
            f0_dec_dim = decoder_in_channels
        else:
            # Original F0Encoder: use output_emb_width from config
            f0_dim = f0cfg.output_emb_width
            f0_dec_dim = f0cfg.output_emb_width
        
        joint_in_dim = enccfg.out_channels + f0_dim
        joint_disentangle_dim = deccfg.in_channels + f0_dec_dim
        
        # Project concatenated features before quantization
        joint_mixer = nn.Conv1d(joint_in_dim, deccfg.vq_dim, kernel_size=1)
        # Project quantized features back to disentangled dimensions
        joint_to_audio_f0 = nn.Conv1d(deccfg.vq_dim, joint_disentangle_dim, kernel_size=1)
        
        print(colored(f"TriXCodec: joint_in_dim={joint_in_dim}, vq_dim={deccfg.vq_dim}, joint_out_dim={joint_disentangle_dim}", "red", attrs=['bold']))
        print(colored(f"  - content dim: {enccfg.out_channels} → {deccfg.in_channels}", "cyan", attrs=['bold']))
        print(colored(f"  - f0 dim: {f0_dim} → {f0_dec_dim}", "blue", attrs=['bold']))
        print(colored(f"  - F0 codec structure: {'CodecEncoder/Decoder' if use_codec_structure else 'Original F0Encoder/Decoder'}", "yellow"))

        model = nn.ModuleDict({
                    'speaker_encoder': speaker_encoder,
                    'CodecEnc': encoder,
                    'generator': decoder,
                    'discriminator': mpd,
                    'f0_encoder': f0_encoder,
                    'f0_decoder': f0_decoder,
                    'f0_extractor': f0_extractor,
                    'joint_mixer': joint_mixer,
                    'joint_to_audio_f0': joint_to_audio_f0,
                })
        if spec_disc is not None:
            model['spec_discriminator'] = spec_disc
        if cqt_disc is not None:
            model['cqt_discriminator'] = cqt_disc

        for k, v in model.named_children():
            # print number of parameters
            num_params = self._count_initialized_params(v)
            print(f"{k}: {num_params}")
            # print number of trainable parameters
            num_trainable_params = self._count_initialized_params(v, trainable_only=True)
            print(f"{k} (trainable): {num_trainable_params}")
        self.model = model
        self._print_model_summary()

    @staticmethod
    def _build_rndvoc_f0_stage_dims(f0_decoder, f0cfg, null_nstage):
        if hasattr(f0_decoder, 'width_list'):
            stage_dims = list(reversed(f0_decoder.width_list[0]))
        else:
            stage_dims = [int(f0cfg.output_emb_width)]
            if hasattr(f0cfg, 'width_list'):
                stage_dims.extend(list(reversed(f0cfg.width_list[0])))

        if not getattr(f0_decoder, 'use_fcpe_loss', False):
            final_output_dim = int(getattr(f0_decoder, 'output_channels', 2 if bool(f0cfg.zero_out_all_unvoiced) else 1))
            stage_dims.append(final_output_dim)

        if len(stage_dims) < null_nstage:
            stage_dims.extend([stage_dims[-1]] * (null_nstage - len(stage_dims)))
        else:
            stage_dims = stage_dims[:null_nstage]
        return [int(dim) for dim in stage_dims]

    @staticmethod
    def _count_initialized_params(module, trainable_only=False):
        total = 0
        for param in module.parameters():
            if isinstance(param, UninitializedParameter):
                continue
            if trainable_only and not param.requires_grad:
                continue
            total += param.numel()
        return total

    def _print_model_summary(self):
        try:
            print(ModelSummary(self, max_depth=2))
        except ValueError as exc:
            print(colored(f"Skipping ModelSummary until lazy parameters are initialized: {exc}", "yellow"))

    def _iter_discriminator_modules(self):
        for name in ('discriminator', 'spec_discriminator', 'cqt_discriminator'):
            if name in self.model:
                yield name, self.model[name]

    @staticmethod
    def _disc_cache_key(name: str) -> str:
        return f'{name}_real_feats'

    @staticmethod
    def _disc_input_crop_key(name: str) -> str:
        return f'{name}_input_crop'

    def _adversarial_cfg(self):
        cfg = self.cfg.train.get('adversarial', None)
        return cfg if cfg is not None else {}

    def _adversarial_preset(self):
        preset = self._adversarial_cfg().get('preset', None)
        if preset is None:
            return None
        preset = str(preset).strip().lower().replace("-", "_")
        if preset in {"", "none", "null", "explicit", "manual"}:
            return None
        aliases = {
            "bigvgan2": "bigvgan_v2",
            "bigvganv2": "bigvgan_v2",
            "bigvganv1": "bigvgan",
        }
        preset = aliases.get(preset, preset)
        valid = {"legacy", "bigvgan", "bigvgan_v2", "vocos", "apnet2"}
        if preset not in valid:
            raise ValueError(f"Unsupported adversarial preset: {preset}. Expected one of {sorted(valid)}.")
        return preset

    def _adversarial_extra_discriminators(self) -> set[str]:
        extras = self._adversarial_cfg().get('extra_discriminators', [])
        if extras is None:
            return set()
        if isinstance(extras, str):
            extras = [extras]
        aliases = {
            "mstft": "mstft",
            "legacy": "mstft",
            "legacy_mstft": "mstft",
            "legacy_spec": "mstft",
            "bigvgan_mrd": "bigvgan_mrd",
            "vocos_mrd": "vocos_mrd",
            "apnet2_mrd": "apnet2_mrd",
            "cqt_discriminator": "cqt",
        }
        normalized = set()
        for name in extras:
            key = str(name).strip().lower().replace("-", "_")
            key = aliases.get(key, key)
            valid = {"mstft", "bigvgan_mrd", "vocos_mrd", "apnet2_mrd", "cqt"}
            if key not in valid:
                raise ValueError(
                    f"Unsupported extra discriminator: {name}. Expected one of {sorted(valid)}."
                )
            normalized.add(key)
        return normalized

    def _configured_gan_mode(self) -> str:
        gan_cfg = self.cfg.train.get('gan_loss', None)
        if gan_cfg is None:
            return 'lsgan'
        return 'bigvsan' if bool(gan_cfg.get('use_bigvsan', False)) else str(gan_cfg.get('mode', 'lsgan'))

    def _configured_spec_discriminator_type(self) -> str:
        spec_cfg = self.cfg.model.get('spec_discriminator', None)
        if spec_cfg is None:
            return 'mstft' if bool(self.cfg.model.get('mstft', {}).get('use', True)) else 'none'
        spec_type = str(spec_cfg.get('type', 'mstft')).lower()
        aliases = {
            'mstft': 'mstft',
            'legacy': 'mstft',
            'legacy_mstft': 'mstft',
            'legacy_spec': 'mstft',
        }
        spec_type = aliases.get(spec_type, spec_type)
        if spec_type not in {'mstft', 'bigvgan_mrd', 'vocos_mrd', 'apnet2_mrd', 'none'}:
            raise ValueError(f"Unsupported spec_discriminator.type: {spec_type}")
        return spec_type

    def _configured_cqt_enabled(self) -> bool:
        cqtcfg = self.cfg.model.get('cqt_discriminator', None)
        return bool(cqtcfg is not None and cqtcfg.get('use', False))

    def _configured_detach_f0_conds_for_audio(self):
        detach = self._adversarial_cfg().get('detach_f0_conds_for_audio', None)
        if detach is None:
            return None
        return bool(detach)

    def _configured_detach_f0_conds_for_audio_until_step(self):
        until_step = self._adversarial_cfg().get('detach_f0_conds_for_audio_until_step', None)
        if until_step is None:
            return None
        return int(until_step)

    def _configured_cqt_segment_samples(self):
        adv_cfg = self._adversarial_cfg()
        segment_samples = adv_cfg.get('cqt_segment_samples', None)
        if segment_samples is not None:
            segment_samples = int(segment_samples)
            return segment_samples if segment_samples > 0 else None
        segment_seconds = adv_cfg.get('cqt_segment_seconds', None)
        if segment_seconds is None:
            return None
        segment_samples = int(round(float(segment_seconds) * float(self.cfg.preprocess.audio.sr)))
        return segment_samples if segment_samples > 0 else None

    def _resolved_adversarial_setup(self) -> dict[str, object]:
        preset = self._adversarial_preset()
        if preset is None:
            gan_mode = self._configured_gan_mode()
            spec_type = self._configured_spec_discriminator_type()
            use_cqt = self._configured_cqt_enabled()
            detach_f0_conds_for_audio = bool(self._configured_detach_f0_conds_for_audio() or False)
        else:
            preset_map = {
                "legacy": {
                    "gan_mode": "lsgan",
                    "spec_type": "mstft",
                    "use_cqt": False,
                    "detach_f0_conds_for_audio": False,
                },
                "bigvgan": {
                    "gan_mode": "lsgan",
                    "spec_type": "bigvgan_mrd",
                    "use_cqt": False,
                    "detach_f0_conds_for_audio": False,
                },
                "bigvgan_v2": {
                    "gan_mode": "lsgan",
                    "spec_type": "none",
                    "use_cqt": True,
                    "detach_f0_conds_for_audio": False,
                },
                "vocos": {
                    "gan_mode": "apnet2",
                    "spec_type": "vocos_mrd",
                    "use_cqt": False,
                    "detach_f0_conds_for_audio": False,
                },
                "apnet2": {
                    "gan_mode": "apnet2",
                    "spec_type": "apnet2_mrd",
                    "use_cqt": False,
                    "detach_f0_conds_for_audio": False,
                },
            }
            resolved = preset_map[preset]
            gan_mode = resolved["gan_mode"]
            spec_type = resolved["spec_type"]
            use_cqt = resolved["use_cqt"]
            detach_f0_conds_for_audio = resolved["detach_f0_conds_for_audio"]

        configured_detach = self._configured_detach_f0_conds_for_audio()
        if configured_detach is not None:
            detach_f0_conds_for_audio = configured_detach

        extras = self._adversarial_extra_discriminators()
        extra_spec = {name for name in extras if name != "cqt"}
        if len(extra_spec) > 1:
            raise ValueError(
                "Only one extra spectral discriminator can be active at a time. "
                f"Received: {sorted(extra_spec)}"
            )
        if extra_spec:
            extra_name = next(iter(extra_spec))
            if spec_type != "none" and spec_type != extra_name:
                raise ValueError(
                    "This configuration already selects one spectral discriminator "
                    f"({spec_type}); adding {extra_name} would require multiple spectral slots."
                )
            spec_type = extra_name
        if "cqt" in extras:
            use_cqt = True

        return {
            "preset": preset,
            "gan_mode": str(gan_mode),
            "spec_type": str(spec_type),
            "use_cqt": bool(use_cqt),
            "detach_f0_conds_for_audio": bool(detach_f0_conds_for_audio),
        }

    def _effective_gan_mode(self) -> str:
        return str(self._resolved_adversarial_setup()['gan_mode'])

    def _spec_discriminator_type(self) -> str:
        return str(self._resolved_adversarial_setup()['spec_type'])

    def _resolved_mstft_cfg(self, spec_disc_type: str):
        spec_cfg = self.cfg.model.get('spec_discriminator', {})
        legacy_cfg = self.cfg.model.get('mstft', None)
        if spec_disc_type == 'mstft':
            return spec_cfg.get('mstft', legacy_cfg)
        return legacy_cfg

    def _detach_f0_conds_for_audio(self) -> bool:
        if bool(self._resolved_adversarial_setup()['detach_f0_conds_for_audio']):
            return True
        until_step = self._configured_detach_f0_conds_for_audio_until_step()
        if until_step is None:
            return False
        return int(self.total_step.item()) < until_step

    def _spec_discriminator_weight(self) -> float:
        spec_type = self._spec_discriminator_type()
        spec_cfg = self.cfg.model.get('spec_discriminator', {})
        if spec_type in {'bigvgan_mrd', 'vocos_mrd', 'apnet2_mrd'}:
            return float(spec_cfg.get('mrd_weight', 0.1))
        return float(spec_cfg.get('legacy_weight', 1.0))

    def _spec_discriminator_label(self) -> str:
        spec_type = self._spec_discriminator_type()
        if spec_type == 'bigvgan_mrd':
            return 'BigVGAN/UnivNet-faithful MRD'
        if spec_type == 'vocos_mrd':
            return 'Vocos-faithful MRD'
        if spec_type == 'apnet2_mrd':
            return 'APNet2-faithful MRD'
        if spec_type == 'mstft':
            return 'Legacy MSTFT'
        if spec_type == 'none':
            return 'No spectral discriminator'
        return 'Legacy spectral discriminator'

    def _discriminator_weight(self, name: str) -> float:
        if name == 'spec_discriminator':
            return self._spec_discriminator_weight()
        return 1.0

    def _detach_disc_feature_tree(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach()
        if isinstance(value, list):
            return [self._detach_disc_feature_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._detach_disc_feature_tree(item) for item in value)
        return value

    def _apn_aux_cfg(self):
        apn_cfg = self.cfg.train.get('apn_aux_loss', None)
        decoder_is_apn = str(self.cfg.model.codec_decoder.get('decoder_type', 'default')) == 'apnformer'
        if apn_cfg is None:
            return {} if decoder_is_apn else None
        use_flag = apn_cfg.get('use', None)
        if use_flag is None:
            return apn_cfg if decoder_is_apn else None
        return apn_cfg if bool(use_flag) else None

    def _use_apn_aux_loss(self) -> bool:
        return self._apn_aux_cfg() is not None

    def _apn_aux_scale(self) -> float:
        apn_cfg = self._apn_aux_cfg()
        if apn_cfg is None:
            return 1.0
        warmup_steps = apn_cfg.get('warmup_steps', None)
        max_scale = float(apn_cfg.get('max_scale', 1.0))
        if max_scale <= 0.0:
            return 0.0
        if warmup_steps is None:
            return max_scale
        warmup_steps = int(warmup_steps)
        if warmup_steps <= 0:
            return max_scale
        init_scale = float(apn_cfg.get('warmup_init_scale', 0.0))
        init_scale = max(0.0, min(max_scale, init_scale))
        step = int(self.total_step.item())
        if step >= warmup_steps:
            return max_scale
        alpha = float(step) / float(warmup_steps)
        scale = init_scale + (max_scale - init_scale) * alpha
        return max(0.0, min(max_scale, scale))

    def _effective_mel_loss_lambda(self) -> float:
        apn_cfg = self._apn_aux_cfg()
        if apn_cfg is not None:
            mel_lambda = float(apn_cfg.get('mel_weight', 45.0))
            if bool(apn_cfg.get('scale_mel_with_aux', True)):
                mel_lambda *= self._apn_aux_scale()
            return mel_lambda
        if self._effective_gan_mode() == 'bigvsan':
            return 45.0
        return float(self.cfg.train.lambdas.get('lambda_mel_loss', 15.0))

    @staticmethod
    def _disc_forward(disc: nn.Module, x: torch.Tensor, san_train: bool = False):
        if san_train and getattr(disc, 'supports_san_training', False):
            return disc(x, flg_train=True)
        return disc(x)

    @staticmethod
    def _match_audio_pair_lengths(real: torch.Tensor, fake: torch.Tensor):
        if real.size(-1) == fake.size(-1):
            return real, fake
        min_len = min(real.size(-1), fake.size(-1))
        return real[..., :min_len], fake[..., :min_len]

    def _prepare_discriminator_inputs(self, name: str, real: torch.Tensor, fake: torch.Tensor, crop=None):
        real, fake = self._match_audio_pair_lengths(real, fake)
        if name != 'cqt_discriminator':
            return real, fake, {'start': 0, 'length': int(real.size(-1))}

        segment_samples = self._configured_cqt_segment_samples()
        total_len = int(real.size(-1))
        if segment_samples is None or segment_samples >= total_len:
            return real, fake, {'start': 0, 'length': total_len}

        segment_samples = max(1, min(int(segment_samples), total_len))
        max_start = total_len - segment_samples
        if crop is None:
            start = int(torch.randint(0, max_start + 1, (1,), device=real.device).item()) if max_start > 0 else 0
        else:
            start = int(crop.get('start', 0))
            cached_length = int(crop.get('length', segment_samples))
            segment_samples = max(1, min(cached_length, total_len))
            start = max(0, min(start, total_len - segment_samples))
        end = start + segment_samples
        return real[..., start:end], fake[..., start:end], {'start': start, 'length': segment_samples}

    def _normalize_discriminator_post_weights(self) -> None:
        for _, disc in self._iter_discriminator_modules():
            if hasattr(disc, 'normalize_san_weights'):
                disc.normalize_san_weights()

    def configure_model(self):
        """Called after model is moved to device, before training starts.
        This is the right place to apply torch.compile."""
        if self.cfg.train.get('use_torch_compile', False):
            # Disable all disk caching to prevent I/O bloat during long training
            # Compilation will happen once per run (in memory), cache cleared on restart
            import torch._inductor.config as inductor_config
            inductor_config.fx_graph_cache = False  # Disable FX graph cache DB
            # Set cache dir to /tmp (auto-cleaned on reboot)
            os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torch_compile_cache_temp'
            # Disable Triton persistent cache
            os.environ['TRITON_CACHE_DIR'] = '/tmp/triton_cache_temp'
            print(colored("⚠ Inductor disk cache DISABLED to prevent I/O bloat", "yellow"))
            print(colored("  → Compilation happens once per run (stored in memory)", "cyan"))
            print(colored("  → Caches redirected to /tmp (auto-cleaned on reboot)", "cyan"))
            
            compile_mode = self.cfg.train.get('compile_mode', 'default')
            print(colored(f"Applying torch.compile (mode={compile_mode}) to model components...", "yellow"))
            
            # Only use dynamic=True for modules that actually need it
            # For most models, default (dynamic=False) gives better performance
            if hasattr(self, 'model'):
                # Speaker Encoder - fixed length after Perceiver resampling
                if 'speaker_encoder' in self.model:
                    self.model['speaker_encoder'] = torch.compile(
                        self.model['speaker_encoder'], mode=compile_mode
                    )
                    print(colored("  ✓ Speaker encoder compiled", "green"))
                
                # CodecEnc - fixed downsampling ratio
                if 'CodecEnc' in self.model:
                    self.model['CodecEnc'] = torch.compile(
                        self.model['CodecEnc'], mode=compile_mode
                    )
                    print(colored("  ✓ CodecEnc compiled", "green"))
                
                # Generator (decoder) - SKIP: SnakeBeta + dynamic upsampling causes symbolic shape issues
                # Error occurs in decoder's upsampling blocks with conditional SnakeBeta activation
                
                # Discriminators - fixed architecture
                disc_labels = {
                    'discriminator': 'MPD',
                    'spec_discriminator': self._spec_discriminator_label(),
                    'cqt_discriminator': 'CQT discriminator',
                }
                for name, module in list(self._iter_discriminator_modules()):
                    self.model[name] = torch.compile(module, mode=compile_mode)
                    print(colored(f"  ✓ {disc_labels.get(name, name)} compiled", "green"))
                
                # F0 codec components - fixed architecture
                if 'f0_encoder' in self.model:
                    self.model['f0_encoder'] = torch.compile(
                        self.model['f0_encoder'], mode=compile_mode
                    )
                    print(colored("  ✓ F0 encoder compiled", "green"))
                
                if 'f0_decoder' in self.model:
                    self.model['f0_decoder'] = torch.compile(
                        self.model['f0_decoder'], mode=compile_mode
                    )
                    print(colored("  ✓ F0 decoder compiled", "green"))
                
                # TriXCodec specific - simple Conv1d layers
                if 'joint_mixer' in self.model:
                    self.model['joint_mixer'] = torch.compile(
                        self.model['joint_mixer'], mode=compile_mode
                    )
                    print(colored("  ✓ Joint mixer compiled", "green"))
                
                if 'joint_to_audio_f0' in self.model:
                    self.model['joint_to_audio_f0'] = torch.compile(
                        self.model['joint_to_audio_f0'], mode=compile_mode
                    )
                    print(colored("  ✓ Joint disentangler compiled", "green"))
                
                print(colored("✓ torch.compile applied successfully", "green"))
                print(colored("⚠ Generator skipped (SnakeBeta + dynamic shapes not compatible)", "yellow"))
                print(colored("⚠ First epoch will be slower due to compilation (JIT tracing)", "yellow"))


    def load_state_dict(self, state_dict, strict=True):
        """
        Checkpoint -> Runtime state_dict reconciliation.
        - Drop keys for modules that are not part of the runtime model
        - Normalize torch.compile `_orig_mod` wrapper prefixes under `model.*`
          so that compiled / non-compiled checkpoints can be loaded either way.
        """
        filtered = {}
        skipped_keys = []

        def _has_submodule(name: str):
            if isinstance(self.model, nn.ModuleDict):
                return name in self.model
            return hasattr(self.model, name)

        def _get_submodule(name: str):
            if isinstance(self.model, nn.ModuleDict):
                return self.model[name] if name in self.model else None
            return getattr(self.model, name, None)

        # Detect old VQ-Wav2Vec checkpoint format: encoder.conv_layers instead of encoder.feature_extractor.conv_layers
        old_vqw2v_prefix = 'model.CodecEnc.encoder.conv_layers.'
        has_old_vqw2v_format = any(k.startswith(old_vqw2v_prefix) for k in state_dict)
        f0_decoder_module = _get_submodule('f0_decoder')
        runtime_fcpe_head = getattr(f0_decoder_module, 'fcpe_head', None) if f0_decoder_module is not None else None
        runtime_fcpe_head_is_sequential = isinstance(runtime_fcpe_head, nn.Sequential)

        for k, v in state_dict.items():
            # 1) Drop keys we never load in the runtime model
            skip_reason = None
            if k.startswith('mel_spectrogram_transform.'):
                skip_reason = 'runtime-only module'
            elif k.startswith('utmos_predictor.'):
                skip_reason = 'runtime-only module'
            elif k.startswith('wavlm_model.'):
                skip_reason = 'runtime-only module'
            elif k.startswith('hubert_model.'):
                skip_reason = 'runtime-only module'
            elif '.f0_mu_predictor.' in k:
                skip_reason = 'removed legacy f0_mu_predictor'
            if skip_reason is not None:
                skipped_keys.append((k, skip_reason))
                continue

            # 2) Remap old VQ-Wav2Vec encoder layout: encoder.conv_layers -> encoder.feature_extractor.conv_layers
            new_key = k
            if has_old_vqw2v_format and k.startswith('model.CodecEnc.encoder.conv_layers.'):
                new_key = k.replace(
                    'model.CodecEnc.encoder.conv_layers.',
                    'model.CodecEnc.encoder.feature_extractor.conv_layers.',
                    1,
                )
            if new_key.startswith('model.f0_decoder.fcpe_head.0.') and not runtime_fcpe_head_is_sequential:
                new_key = new_key.replace('model.f0_decoder.fcpe_head.0.', 'model.f0_decoder.fcpe_head.', 1)
            elif (
                new_key.startswith('model.f0_decoder.fcpe_head.')
                and not new_key.startswith('model.f0_decoder.fcpe_head.0.')
                and runtime_fcpe_head_is_sequential
            ):
                new_key = new_key.replace('model.f0_decoder.fcpe_head.', 'model.f0_decoder.fcpe_head.0.', 1)

            # 3) Normalize torch.compile wrapper prefixes for any submodule under model.*
            if new_key.startswith('model.'):
                parts = new_key.split('.')
                if len(parts) >= 3:
                    sub_name = parts[1]  # model.<sub>...
                    if _has_submodule(sub_name):
                        submodule = _get_submodule(sub_name)
                        compiled_now = hasattr(submodule, '_orig_mod')
                        plain_prefix = f'model.{sub_name}.'
                        compiled_prefix = f'model.{sub_name}._orig_mod.'
                        # ckpt compiled, runtime plain -> strip
                        if new_key.startswith(compiled_prefix) and not compiled_now:
                            new_key = plain_prefix + new_key[len(compiled_prefix):]
                        # ckpt plain, runtime compiled -> insert
                        elif new_key.startswith(plain_prefix) and compiled_now and not new_key.startswith(compiled_prefix):
                            new_key = compiled_prefix + new_key[len(plain_prefix):]

            filtered[new_key] = v

        # Always load with strict=False, then emulate strict behavior manually
        missing, unexpected = super().load_state_dict(filtered, strict=False)

        # Allow missing keys for non-trainable / runtime-only modules
        allowed_missing_prefixes = list((
            'mel_spectrogram_transform.',
            'utmos_predictor.',
            'wavlm_model.',
            'hubert_model.',
        ))
        if has_old_vqw2v_format:
            allowed_missing_prefixes.append('model.CodecEnc.encoder.vector_quantizer.')
        allowed_missing = [k for k in missing if any(k.startswith(p) for p in allowed_missing_prefixes)]
        real_missing = [k for k in missing if k not in allowed_missing]

        if skipped_keys:
            print("[load_state_dict] skipped checkpoint keys:")
            for key, reason in skipped_keys:
                print(f"  - {key} ({reason})")
        if allowed_missing:
            print("[load_state_dict] (ignored missing non-trainable):",
                  allowed_missing[:8], "..." if len(allowed_missing) > 8 else "")
        if unexpected:
            print("[load_state_dict] (unexpected):", unexpected[:8], "..." if len(unexpected) > 8 else "")

        # If strict=True, propagate real errors (excluding allowed missing keys)
        if strict and (real_missing or unexpected):
            error_msgs = []
            if real_missing:
                error_msgs.append(
                    "Missing key(s) in state_dict: " +
                    ", ".join(f'"{k}"' for k in real_missing) + "."
                )
            if unexpected:
                error_msgs.append(
                    "Unexpected key(s) in state_dict: " +
                    ", ".join(f'"{k}"' for k in unexpected) + "."
                )
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t" +
                "\n\t".join(error_msgs)
            )

        return real_missing, unexpected

    def construct_criteria(self):
        cfg = self.cfg.train
        criteria = nn.ModuleDict()
        if cfg.use_mel_loss:
            criteria['mel_loss'] = MultiResolutionMelSpectrogramLoss(sample_rate=self.cfg.preprocess.audio.sr)
        if cfg.use_feat_match_loss:
            criteria['fm_loss'] = nn.L1Loss()
        criteria['gan_loss'] = GANLoss(mode=self._effective_gan_mode())
        criteria['l1_loss'] = torch.nn.L1Loss()
        criteria['l2_loss'] = torch.nn.MSELoss()
        criteria['bcewlogits_loss'] = torch.nn.BCEWithLogitsLoss()
        criteria['bce_loss'] = torch.nn.BCELoss()
        if self._use_apn_aux_loss():
            vocos_cfg = CodecDecoderVocosConfig.from_decoder_cfg(self.cfg.model.codec_decoder)
            criteria['apn_aux_loss'] = APNet2AuxLoss(
                n_fft=vocos_cfg.n_fft,
                hop_length=vocos_cfg.hop_size,
                win_length=vocos_cfg.win_size,
            )
        self.criteria = criteria
        # print(criteria)

    def _no_sync_if_needed(self, enable: bool):
        # enable=True면 통신을 끔(no_sync 사용)
        if enable and hasattr(self.trainer, "strategy"):
            strat = self.trainer.strategy
            model = getattr(strat, "model", None)
            if model is not None and hasattr(model, "no_sync"):
                return model.no_sync()
        return nullcontext()

    def _sync_needed(self):
        try:
            return getattr(self.trainer.strategy, "world_size", 1) > 1
        except Exception:
            return False

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        def move(x):
            return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x
        if isinstance(batch, dict):
            return {k: move(v) for k, v in batch.items()}
        return batch

    def pad_for_wav2vec(self, source, J=160, RF=465):
        # source: (B, T)
        B, L = source.shape
        pad_left = RF - J  # 305
        # ceil(L/J) 프레임 확보용 오른쪽 패딩
        rem = L % J
        pad_right = (J - rem) % J
        return F.pad(source, (pad_left, pad_right)), pad_left, pad_right

    def pad_for_w2v2(self, source, J=320, RF=400):
        """
        Wav2Vec2 (HF) conv feature extractor padding.
        Default values match Wav2Vec2Model conv frontend:
          - effective stride J = 320 (5 * 2^6)
          - receptive field RF = 400
        """
        B, L = source.shape
        pad_left = RF - J  # 80
        rem = L % J
        pad_right = (J - rem) % J
        return F.pad(source, (pad_left, pad_right)), pad_left, pad_right

    def _w2v2_normalize(self, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """
        Torch implementation of Wav2Vec2FeatureExtractor zero-mean/unit-variance normalization.
        x: (B, T)
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + eps)


    def extract_f0(self, batch, ref=False):
        if not ref:
            # if self.cfg.preprocess.audio.sr == 16000:
            audio = batch['wav'].unsqueeze(-1)
            # else:
            #     print(colored(f"Using 24kHz audio for F0 extraction", "red"))
            #     audio = batch['wav_24k'].unsqueeze(-1)
        else:
            audio = batch['ref_wav'].unsqueeze(-1)
        
        # Two modes for F0 extraction:
        # Mode 1 (default): Interpolate to audio length (directly upsample during extraction)
        # Mode 2 (optional): Extract at original frame rate (no interpolation in extractor)
        if self.cfg.model.f0_codec.get('upsample_extracted_f0', True):
            # Mode 1: Interpolate to audio sample length
            output_interp_target_length = audio.shape[1] if self.cfg.preprocess.audio.sr == 16000 else int(audio.shape[1] * 1.5)
        else:
            # Mode 2: Return original frame length (e.g., T_audio / hop_length)
            target_length = int(audio.shape[1]/self.cfg.model.f0_codec.get('hop_length', 160))
            output_interp_target_length = target_length if self.cfg.preprocess.audio.sr == 16000 else int(target_length * 1.5)
        
        f0 = self.model['f0_extractor'](audio,
                                        sr=16000,
                                        output_interp_target_length=output_interp_target_length)  # (B, T)

        # print("f0>0 min max mean var :", (f0[f0>0].min().item(), f0[f0>0].max().item(), f0[f0>0].mean().item(), f0[f0>0].var().item()))

        # 1) NaN/Inf 제거
        f0 = torch.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)


        # 2) V/UV 마스크
        vuv = (f0 > 0).float()

        # 3) 전부 무성인 샘플 처리 (count=0 → 바로 zero feature 반환)
        count = vuv.sum(dim=1, keepdim=True)  # (B,1)
        all_unvoiced = (count == 0)
        # log-f0 (무성은 임시 0; 통계엔 vuv로 걸러짐)
        f0_log = torch.log(torch.clamp(f0, min=1e-5))

        # 4) voiced 값만으로 통계
        voiced_log = f0_log * vuv
        count_safe = count.clamp(min=1)
        mu = voiced_log.sum(dim=1, keepdim=True) / count_safe
        # CRITICAL FIX: unvoiced 프레임을 variance 계산에서 제외
        # 이전: diff = (voiced_log - mu)  → unvoiced가 -mu가 되어 variance 폭발!
        # 수정: diff = (f0_log - mu) * vuv → unvoiced는 0으로 유지
        diff = (f0_log - mu) * vuv  # voiced 프레임에서만 편차 계산
        var = (diff.pow(2).sum(dim=1, keepdim=True) / count_safe)
        sig = var.sqrt().clamp(min=1e-4)
        # print("f0 voiced log mu :", (mu.min().item(), mu.max().item(), mu.mean().item(), mu.var().item()))
        # print("f0 voiced log sig:", (sig.min().item(), sig.max().item(), sig.mean().item(), sig.var().item()))

        # 5) 정규화: unvoiced 프레임은 연산 전에 제거 → 큰 음수 생성 안 됨 (다시 넣자, 이거 std 도 speaker정보가 채워줘야하는 global info라 없애는 게 맞는 듯 251028)
        z = ((f0_log - mu) * vuv) / sig  # (B,T)
        # variance 정규화 제거 버전
        # z = ((f0_log - mu) * vuv) # (B,T)

        # unvoiced 부분을 0 이 아닌 -3.0 으로 채우기
        if not self.cfg.model.f0_codec.zero_out_all_unvoiced:
            uv_mask = vuv < 0.5       # 0/1 또는 확률인 경우
            # 채널 차원이 있으면 확장
            if z.dim() == 3:  # (B, C, T)
                uv_mask = uv_mask.unsqueeze(1)  # (B, 1, T)

            # -3으로 채우기 (벡터화, 병렬)
            z = z.masked_fill(uv_mask, -3.0)

        # print("f0 normalized min max mean var :", (z[vuv>0].min().item(), z[vuv>0].max().item(), z[vuv>0].mean().item(), z[vuv>0].var().item()))
        # 6) 클램프 (이제 극단값 거의 안 생김; 범위 더 타이트 가능)
        # z = torch.clamp(z, -8.0, 8.0)

        # 7) 전부 무성인 샘플 zero 처리 (혹시 위 계산 중 미세값 남았을 수 있음)
        if all_unvoiced.any():
            z[all_unvoiced.expand_as(z)] = 0.0 if self.cfg.model.f0_codec.zero_out_all_unvoiced else -3.0

        # 8) (선택) 디버그
        # print("f0_log range", f0_log.min().item(), f0_log.max().item(), "z range", z.min().item(), z.max().item())

        return (
            f0.unsqueeze(1),  # (B,1,T) (raw log-f0; 무성 프레임은 log(1e-5))
            z.unsqueeze(1),       # (B,1,T) normalized voiced-only
            vuv.unsqueeze(1),     # (B,1,T)
        )

    
    def extract_loudness(self, batch):
        audio = batch['wav'].unsqueeze(-1)
        return torch.log1p(torch.sqrt(torch.mean(audio**2, dim=-1)))

    
    def shift_f0(self, batch, f0, vuv):
        assert self.vc_f0 == True
        f0_ref, _, vuv_ref = self.extract_f0(batch, ref=True)  # (B, 1, T)
        
        # 유성음 구간만 추출
        voiced = f0 * vuv  # (B, 1, T)
        voiced_ref = f0_ref * vuv_ref  # (B, 1, T)
        
        # 각 배치의 유성음 프레임 수 계산
        count = vuv.sum(dim=2, keepdim=True)  # (B, 1, 1)
        count_ref = vuv_ref.sum(dim=2, keepdim=True)  # (B, 1, 1)
        
        # 0으로 나누기 방지
        count_safe = count.clamp(min=1)
        count_safe_ref = count_ref.clamp(min=1)
        
        # 평균 계산
        mu = voiced.sum(dim=2, keepdim=True) / count_safe  # (B, 1, 1)
        mu_ref = voiced_ref.sum(dim=2, keepdim=True) / count_safe_ref  # (B, 1, 1)
        
        print("Shift f0 mu:", mu.mean().item(), "->", mu_ref.mean().item())
        print("shapes f0, mu, mu_ref:", f0.shape, mu.shape, mu_ref.shape)
        
        # 전부 무성음인 경우 체크
        all_unvoiced = (count == 0)
        all_unvoiced_ref = (count_ref == 0)
        
        # F0 shift: log 공간에서의 덧셈은 원 공간에서의 곱셈
        # 무성음 구간은 그대로 유지
        if not (all_unvoiced.all() or all_unvoiced_ref.all()):
            # 유성음 구간만 shift 적용
            f0_shifted = f0.clone()
            B = f0.shape[0]
            for b in range(B):
                shift = mu_ref[b] - mu[b]
                voiced_mask = vuv[b] > 0.5
                # 유성음 구간에서 f0 + shift의 최소값 확인
                min_voiced = (f0[b][voiced_mask] + shift).min().item() if voiced_mask.any() else 0.0
                # 만약 음수가 있으면 전체적으로 올려줌
                if min_voiced < 1e-5:
                    shift = shift + (1e-5 - min_voiced)
                f0_shifted[b] = torch.where(
                    voiced_mask,
                    f0[b] + shift,
                    f0[b]
                )
            f0 = f0_shifted
        else:
            print("Warning: All frames are unvoiced in source or reference, skipping F0 shift")


        return f0

    # @autocast(device_type='cuda', enabled=False)
    def forward(self, batch):
        timing_active = self._timing_is_active()
        if timing_active:
            step_t0 = self._timing_now()
            seg_t0 = step_t0

        wav = batch['wav']
        wav_24k = batch['wav_24k']
        ref_wav = batch['ref_wav']
        use_mhca = CodecDecoderSpeakerConditionConfig.from_decoder_cfg(self.cfg.model.codec_decoder).use_mhca
        
        spkcfg = self.cfg.model.speaker_encoder
        spk_stack_cfg = spkcfg.get('stack', None)
        use_speaker_stack = bool(spk_stack_cfg is not None and spk_stack_cfg.get('use', False))
        use_quantizer = getattr(spkcfg, 'use_quantizer', True)

        # encoder가 freeze 상태면 no_grad로 forward
        enc_no_grad = self.training and self._is_encoder_frozen()

        parallel_branch_streams = (
            self._parallel_branch_streams_active()
            and wav.is_cuda
            and ref_wav.is_cuda
        )
        speaker_stream = None
        if parallel_branch_streams:
            speaker_stream = self._get_parallel_branch_stream(ref_wav.device)
            speaker_stream.wait_stream(torch.cuda.current_stream(ref_wav.device))
            with torch.cuda.stream(speaker_stream):
                x_vector, d_vector, x_quantized, _ = self.model['speaker_encoder'](ref_wav)
        else:
            x_vector, d_vector, x_quantized, _ = self.model['speaker_encoder'](ref_wav)
            if timing_active:
                now = self._timing_now()
                self._timing_add('forward_speaker_encoder', now - seg_t0)
                seg_t0 = now

        use_vqw2v = bool(getattr(self.cfg.model.codec_encoder, 'use_vqw2v', False))
        use_w2v2 = bool(getattr(self.cfg.model.codec_encoder, 'use_w2v2', False))
        use_hubert = bool(getattr(self.cfg.model.codec_encoder, 'use_hubert', False))
        use_w2vbert2 = bool(getattr(self.cfg.model.codec_encoder, 'use_w2vbert2', False))
        use_s3tokenizer = bool(getattr(self.cfg.model.codec_encoder, 'use_s3tokenizer', False))
        if use_vqw2v:
            wav_ = self.pad_for_wav2vec(wav)[0]
            with torch.set_grad_enabled(not enc_no_grad):
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
        elif use_s3tokenizer:
            wav_n = self._w2v2_normalize(wav)
            wav_ = self.pad_for_w2v2(wav_n)[0]
            with torch.set_grad_enabled(not enc_no_grad):
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
        elif use_w2v2:
            # Backward-compat: allow reproducing legacy behavior (incorrect vq-w2v padding on w2v2).
            enc_cfg = self.cfg.model.codec_encoder
            legacy = bool(getattr(enc_cfg, "w2v2_legacy_input", False))
            if legacy:
                wav_ = self.pad_for_wav2vec(wav)[0]
            else:
                wav_n = self._w2v2_normalize(wav)
                wav_ = self.pad_for_w2v2(wav_n)[0]
            with torch.set_grad_enabled(not enc_no_grad):
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
        elif use_hubert:
            # HuBERT: always normalize + pad_for_w2v2 (no legacy path).
            wav_n = self._w2v2_normalize(wav)
            wav_ = self.pad_for_w2v2(wav_n)[0]
            with torch.set_grad_enabled(not enc_no_grad):
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
        elif use_w2vbert2:
            # W2VBert2: mel frontend has different frame formula than W2V2 conv.
            # pad_for_w2v2 ensures output frame count matches pipeline (F0/decoder) expectations.
            wav_n = self._w2v2_normalize(wav)
            wav_ = self.pad_for_w2v2(wav_n)[0]
            with torch.set_grad_enabled(not enc_no_grad):
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
        else:
            raise NotImplementedError("Forward is only available when using an SSL encoder (vq-wav2vec, w2v2, hubert, w2vbert2, or s3tokenizer).")
        if parallel_branch_streams:
            current_stream = torch.cuda.current_stream(ref_wav.device)
            current_stream.wait_stream(speaker_stream)
            for tensor in (x_vector, d_vector, x_quantized):
                tensor.record_stream(current_stream)
            if timing_active:
                now = self._timing_now()
                self._timing_add('forward_parallel_branches', now - seg_t0)
                seg_t0 = now
        else:
            if timing_active:
                now = self._timing_now()
                self._timing_add('forward_codec_encoder', now - seg_t0)
                seg_t0 = now

        if spkcfg.use_perceiver_encoder or use_speaker_stack:
            if use_quantizer:
                if self.total_step < self.cfg.train.transit_step:
                    spk_cond_vec = x_vector
                else:
                    spk_cond_vec = d_vector
            else:
                spk_cond_vec = d_vector
        else:
            spk_cond_vec = x_vector

        # Prepare spk_cond: if use_mhca, pass as tuple (gq_vector, x_quantized)
        if use_mhca:
            spk_cond = (spk_cond_vec, x_quantized)
        else:
            spk_cond = spk_cond_vec

        # ============================================================
        # TriXCodec Pipeline:
        # 1. Extract and encode F0
        # 2. Concatenate content (vq_emb) and F0 embeddings
        # 3. Project to VQ dimension and jointly quantize
        # 4. Project back to disentangled dimensions
        # 5. Decode F0 and audio separately
        # ============================================================
        
        # Step 1: Extract and encode F0
        f0, f0_normalized, vuv = self.extract_f0(batch)  # (B, 1, T_wav)
        if self.vc_f0:
            f0 = self.shift_f0(batch, f0, vuv)
        if self.cfg.model.f0_codec.use_normalized_f0: input_f0 = f0_normalized
        else: input_f0 = torch.log1p(f0)
        # Always concatenate f0 and vuv to 2 channels
        # F0Encoder will select the appropriate channels based on input_channels
        f0_vuv = torch.cat([input_f0, vuv], dim=1)  # (B, 2, T)
        z_f0 = self.model['f0_encoder'](f0_vuv)  # (B, D, T)
        if z_f0.shape[-1] != vq_emb.shape[-1]:
            if not self._f0_interp_warned:
                print(colored(
                    f"Interpolating f0 embedding from {z_f0.shape} to match content embedding {vq_emb.shape}",
                    "red",
                ))
                self._f0_interp_warned = True
            z_f0_ = F.interpolate(z_f0, size=vq_emb.shape[-1], mode='nearest')
        else: z_f0_ = z_f0
        content_f0_emb = torch.cat([vq_emb, z_f0_], dim=1) # (B, C+D, T)
        content_f0_emb = self.model['joint_mixer'](content_f0_emb)
        if timing_active:
            now = self._timing_now()
            self._timing_add('forward_f0_path', now - seg_t0)
            seg_t0 = now
        
        vq_post_emb, vq_code, vq_loss, perplexity, active_num = self.model['generator'](content_f0_emb, total_step=self.total_step, vq=True)
        if timing_active:
            now = self._timing_now()
            self._timing_add('forward_joint_vq', now - seg_t0)
            seg_t0 = now

        content_f0_post_emb = self.model['joint_to_audio_f0'](vq_post_emb)
        content_f0_post_emb = torch.split(content_f0_post_emb, [vq_emb.shape[1], z_f0.shape[1]], dim=1)
        content_post_emb, f0_post_emb = content_f0_post_emb
        if getattr(self.cfg, "debug", False):
            self._last_f0_path_debug = {
                "vq_emb": _debug_tensor_stats(vq_emb),
                "z_f0": _debug_tensor_stats(z_f0),
                "z_f0_aligned": _debug_tensor_stats(z_f0_),
                "content_f0_emb": _debug_tensor_stats(content_f0_emb),
                "vq_post_emb": _debug_tensor_stats(vq_post_emb),
                "content_post_emb": _debug_tensor_stats(content_post_emb),
                "f0_post_emb": _debug_tensor_stats(f0_post_emb),
                "spk_cond_vec": _debug_tensor_stats(spk_cond_vec),
                "x_quantized": _debug_tensor_stats(x_quantized),
            }

        f0_decoder_module = self.model['f0_decoder']
        f0_spk_cond = None
        if getattr(f0_decoder_module, 'use_mhca', False):
            mhca_global = spk_cond_vec if getattr(f0_decoder_module, 'speaker_condition', False) else None
            f0_spk_cond = (mhca_global, x_quantized)
        elif getattr(f0_decoder_module, 'speaker_condition', False):
            f0_spk_cond = spk_cond_vec

        # F0 decoder output: (outs, fcpe_latent) if use_fcpe_loss else outs
        if f0_spk_cond is not None:
            f0_decoder_out = f0_decoder_module(f0_post_emb, spk_emb=f0_spk_cond)
        else:
            f0_decoder_out = f0_decoder_module(f0_post_emb)
        
        if self.cfg.model.f0_codec.get('use_fcpe_loss', False):
            f0_vuv_recons, f0_fcpe_latent = f0_decoder_out  # unpack tuple
            f0_fcpe_logits = getattr(f0_decoder_module, 'last_fcpe_logits', None)
            # FCPE mode: decode F0 from latent
            # latent (B, 360, T) → cents → raw Hz F0
            # Transpose to (B, T, 360) for FCPE decoder
            fcpe_latent_transposed = f0_fcpe_latent.transpose(1, 2)  # (B, T, 360)
            
            # Access FCPE model directly (F0ExtractorWrapper doesn't wrap latent2cents_local_decoder)
            f0_extractor = self.model['f0_extractor']
            if f0_extractor.is_infer:
                fcpe_model = f0_extractor.extractor.model
            else:
                fcpe_model = f0_extractor.model
            
            # Decode using local_argmax (same as FCPE inference)
            cents_pred = fcpe_model.latent2cents_local_decoder(
                fcpe_latent_transposed, threshold=0.006
            )  # (B, T, 1)
            f0_raw_pred = fcpe_model.cent_to_f0(cents_pred)  # (B, T, 1), raw Hz
            # Transpose back to (B, 1, T)
            f0_raw_pred = f0_raw_pred.transpose(1, 2)  # (B, 1, T)
            
            # Convert to same format as gt_f0_vuv (normalized or log1p)
            if self.cfg.model.f0_codec.use_normalized_f0:
                # raw Hz → log → normalize (simplified: use log1p for now)
                # TODO: proper normalization with speaker mu/sigma -> but not important because it's just for visualization
                f0_pred_formatted = torch.log1p(f0_raw_pred)
            else:
                f0_pred_formatted = torch.log1p(f0_raw_pred)
            
            # Create gen_f0_vuv: (B, 2, T) with [f0, vuv]
            # Ensure temporal dimensions align before concatenation
            if f0_pred_formatted.shape[-1] != vuv.shape[-1]:
                target_len = min(f0_pred_formatted.shape[-1], vuv.shape[-1])
                if f0_pred_formatted.shape[-1] != target_len:
                    f0_pred_formatted = f0_pred_formatted[..., :target_len]
                if vuv.shape[-1] != target_len:
                    vuv = vuv[..., :target_len]
            # Use predicted F0 and GT vuv (vuv prediction can be added later if needed)
            gen_f0_vuv_output = torch.cat([f0_pred_formatted, vuv], dim=1)  # (B, 2, T)
        else:
            f0_vuv_recons = f0_decoder_out
            f0_fcpe_latent = None
            f0_fcpe_logits = None
            gen_f0_vuv_output = f0_vuv_recons[-1]  # Use predicted F0
        if timing_active:
            now = self._timing_now()
            self._timing_add('forward_f0_decoder', now - seg_t0)
            seg_t0 = now
        
        content_post_emb_ = content_post_emb + spk_cond_vec.unsqueeze(-1)
        f0_audio_conds = (
            self._detach_disc_feature_tree(f0_vuv_recons)
            if self._detach_f0_conds_for_audio()
            else f0_vuv_recons
        )
        y_ = self.model['generator'](content_post_emb_, vq=False, spk_cond=spk_cond, f0_conds=f0_audio_conds, vuv=vuv) # [B, 1, T]
        decoder_aux = None
        if hasattr(self.model['generator'], 'get_latest_decoder_aux'):
            decoder_aux = self.model['generator'].get_latest_decoder_aux()
        if timing_active:
            now = self._timing_now()
            self._timing_add('forward_audio_decoder', now - seg_t0)
            self._timing_add('forward_total', now - step_t0)
        y = wav.unsqueeze(1) if self.cfg.preprocess.audio.sr == 16000 else wav_24k.unsqueeze(1)
        if y_.size(-1) != y.size(-1):
            min_len = min(y_.size(-1), y.size(-1))
            y_ = y_[..., :min_len]
            y = y[..., :min_len]
        gt_f0_log = torch.log1p(f0)
        output = {
            'gt_wav': y,
            'gen_wav': y_,
            'vq_loss': vq_loss,
            'vq_code': vq_code,
            'vq_emb': vq_emb,
            'spk_emb': spk_cond_vec,
            'x_vector': x_vector,
            'd_vector': d_vector,
            'perplexity': perplexity,
            'active_num': active_num,

            'gt_f0_vuv': f0_vuv,
            'gen_f0_vuv': gen_f0_vuv_output,  # GT in FCPE mode, predicted in normal mode
            'f0_vq_loss': None,  # TriXCodec: no separate F0 VQ loss (joint quantization)
            'f0_fcpe_latent': f0_fcpe_latent,  # FCPE-style latent (B, 360, T) or None
            'f0_fcpe_logits': f0_fcpe_logits,  # Pre-sigmoid FCPE logits for stable BCEWithLogits loss
            'gt_f0': f0,  # Raw F0 for FCPE loss computation
            'gt_f0_log': gt_f0_log,
            'decoder_aux': decoder_aux,
        }
        return output
    
    def tokenizer(self, batch):
        """
        TriXCodec tokenizer: produces joint tokens from content and F0.
        """
        spk_type = getattr(self.cfg.model.speaker_encoder, 'speaker_encoder_type', 'ecapa_tdnn')
        if spk_type == 'wavlm':
            global_tokens = self.model['speaker_encoder'].tokenize_wav(batch['ref_wav'])
        else:
            global_tokens = self.model['speaker_encoder'].tokenize(batch['mel'].transpose(1, 2))
        
        with torch.no_grad():
            use_vqw2v = bool(getattr(self.cfg.model.codec_encoder, 'use_vqw2v', False))
            use_w2v2 = bool(getattr(self.cfg.model.codec_encoder, 'use_w2v2', False))
            use_hubert = bool(getattr(self.cfg.model.codec_encoder, 'use_hubert', False))
            use_w2vbert2 = bool(getattr(self.cfg.model.codec_encoder, 'use_w2vbert2', False))
            use_s3tokenizer = bool(getattr(self.cfg.model.codec_encoder, 'use_s3tokenizer', False))
            if use_vqw2v:
                wav_ = self.pad_for_wav2vec(batch['wav'])[0]
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
            elif use_s3tokenizer:
                wav_n = self._w2v2_normalize(batch['wav'])
                wav_ = self.pad_for_w2v2(wav_n)[0]
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
            elif use_w2v2:
                enc_cfg = self.cfg.model.codec_encoder
                legacy = bool(getattr(enc_cfg, "w2v2_legacy_input", False))
                if legacy:
                    wav_ = self.pad_for_wav2vec(batch['wav'])[0]
                else:
                    wav_n = self._w2v2_normalize(batch['wav'])
                    wav_ = self.pad_for_w2v2(wav_n)[0]
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
            elif use_hubert:
                wav_n = self._w2v2_normalize(batch['wav'])
                wav_ = self.pad_for_w2v2(wav_n)[0]
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
            elif use_w2vbert2:
                wav_n = self._w2v2_normalize(batch['wav'])
                wav_ = self.pad_for_w2v2(wav_n)[0]
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
            else:
                raise NotImplementedError("Tokenizer is only available when using an SSL encoder (vq-wav2vec, w2v2, hubert, w2vbert2, or s3tokenizer).")
            
            # Extract and encode F0
            f0, f0_normalized, vuv = self.extract_f0(batch)
            if self.cfg.model.f0_codec.use_normalized_f0: 
                input_f0 = f0_normalized
            else: 
                input_f0 = torch.log1p(f0)
            # Always concatenate to 2 channels (F0Encoder will select appropriate channels)
            f0_vuv = torch.cat([input_f0, vuv], dim=1)  # (B, 2, T)
            z_f0 = self.model['f0_encoder'](f0_vuv)
            
            # Interpolate F0 embedding if length mismatch
            if z_f0.shape[-1] != vq_emb.shape[-1]:
                z_f0 = F.interpolate(z_f0, size=vq_emb.shape[-1], mode='nearest')
            
            # Joint encoding and quantization
            content_f0_emb = torch.cat([vq_emb, z_f0], dim=1)
            content_f0_mixed = self.model['joint_mixer'](content_f0_emb)
            _, vq_code, _, _, _ = self.model['generator'](content_f0_mixed, total_step=None, vq=True)

        return {
            'global_tokens': global_tokens,
            'semantic_tokens': vq_code  # TriXCodec: joint content+F0 tokens
        }

    def compute_disc_loss(self, batch, output):
        timing_active = self._timing_is_active()
        if timing_active:
            disc_t0 = self._timing_now()
        y, y_ = output['gt_wav'], output['gen_wav']
        y_ = y_.detach()
        use_bigvsan = self.criteria['gan_loss'].mode == 'bigvsan'
        real_loss_list, fake_loss_list = [], []
        disc_real_cache = {}
        disc_input_crops = {}
        for name, disc in self._iter_discriminator_modules():
            real_input, fake_input, crop = self._prepare_discriminator_inputs(name, y, y_)
            real_feats = self._disc_forward(disc, real_input, san_train=use_bigvsan)
            fake_feats = self._disc_forward(disc, fake_input, san_train=use_bigvsan)
            disc_real_cache[name] = self._detach_disc_feature_tree(real_feats)
            disc_input_crops[name] = crop
            disc_weight = self._discriminator_weight(name)
            for i in range(len(real_feats)):
                real_loss, fake_loss = self.criteria['gan_loss'].disc_loss(real_feats[i][-1], fake_feats[i][-1])
                real_loss_list.append(real_loss * disc_weight)
                fake_loss_list.append(fake_loss * disc_weight)
        
        real_loss = sum(real_loss_list)
        fake_loss = sum(fake_loss_list)

        disc_loss = real_loss + fake_loss
        disc_loss = self.cfg.train.lambdas.lambda_disc * disc_loss
        
        output = {
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'disc_loss': disc_loss,
            'disc_real_feats': disc_real_cache.get('discriminator'),
            'disc_real_feats_by_name': disc_real_cache,
            'disc_input_crops_by_name': disc_input_crops,
        }
        for name, feats in disc_real_cache.items():
            output[self._disc_cache_key(name)] = feats
        for name, crop in disc_input_crops.items():
            output[self._disc_input_crop_key(name)] = crop

        if timing_active:
            self._timing_add('disc_loss_total', self._timing_now() - disc_t0)

        return output
    
    def compute_gen_loss(self, batch, output, cached_real=None, cached_spec_real=None, cached_input_crops=None):
        timing_active = self._timing_is_active()
        if timing_active:
            gen_t0 = self._timing_now()
        y, y_ = output['gt_wav'], output['gen_wav']
        f0_vuv, f0_vuv_ = output['gt_f0_vuv'], output['gen_f0_vuv']
        vq_loss = output['vq_loss']
        f0_vq_loss =output['f0_vq_loss']
        x_vector, d_vector = output['x_vector'], output['d_vector']
        f0_fcpe_latent = output.get('f0_fcpe_latent', None)
        f0_fcpe_logits = output.get('f0_fcpe_logits', None)
        gt_f0 = output.get('gt_f0', None)
        gt_f0_log = output.get('gt_f0_log', None)
        gen_loss = torch.tensor(0.0, device=self.device)
        self.set_discriminator_gradients(False)
        loss_dict = {}
        cfg = self.cfg.train
        apn_cfg = self._apn_aux_cfg()
        
        if cfg.use_mel_loss:
            mel_t0 = self._timing_now() if timing_active else None
            # Safety: ensure lengths are equal before mel loss (BiCodec behavior)
            if y_.size(-1) != y.size(-1):
                min_len = min(y_.size(-1), y.size(-1))
                y_ = y_[..., :min_len]
                y = y[..., :min_len]
            mel_loss = self.criteria['mel_loss'](y_.squeeze(1), y.squeeze(1))
            gen_loss += mel_loss * self._effective_mel_loss_lambda()
            loss_dict['mel_loss'] = mel_loss
            if timing_active:
                self._timing_add('gen_loss_mel', self._timing_now() - mel_t0)
        
        # gan loss
        adv_t0 = self._timing_now() if timing_active else None
        adv_loss_list = []
        if isinstance(cached_real, dict):
            cached_real_map = cached_real
        else:
            cached_real_map = {}
            if cached_real is not None:
                cached_real_map['discriminator'] = cached_real
            if cached_spec_real is not None:
                cached_real_map['spec_discriminator'] = cached_spec_real
        cached_input_crop_map = cached_input_crops if isinstance(cached_input_crops, dict) else {}
        use_feat_match = bool(cfg.use_feat_match_loss)
        fm_loss = torch.tensor(0.0, device=self.device) if use_feat_match else None
        fm_loss_by_name = {}
        fm_multiplier = 2.0 if self.criteria['gan_loss'].mode == 'bigvsan' else 1.0
        for name, disc in self._iter_discriminator_modules():
            real_input, fake_input, _ = self._prepare_discriminator_inputs(
                name,
                y,
                y_,
                crop=cached_input_crop_map.get(name),
            )
            fake_feats = disc(fake_input)
            disc_weight = self._discriminator_weight(name)
            for i in range(len(fake_feats)):
                adv_loss_list.append(self.criteria['gan_loss'].gen_loss(fake_feats[i][-1]) * disc_weight)
            if use_feat_match:
                real_feats = cached_real_map.get(name)
                if real_feats is None:
                    with torch.no_grad():
                        real_feats = disc(real_input)
                real_feats = self._detach_disc_feature_tree(real_feats)
                disc_fm_loss = torch.tensor(0.0, device=self.device)
                for i in range(len(fake_feats)):
                    for j in range(len(fake_feats[i]) - 1):
                        disc_fm_loss += self.criteria['fm_loss'](fake_feats[i][j], real_feats[i][j])
                disc_fm_loss = disc_fm_loss * fm_multiplier * disc_weight
                fm_loss += disc_fm_loss
                fm_loss_by_name[name] = disc_fm_loss
                del real_feats
            del fake_feats
        adv_loss = sum(adv_loss_list)
        gen_loss += adv_loss * cfg.lambdas.lambda_adv
        loss_dict['adv_loss'] = adv_loss
        if timing_active:
            self._timing_add('gen_loss_adv', self._timing_now() - adv_t0)

        # fm loss
        if use_feat_match:
            fm_t0 = self._timing_now() if timing_active else None
            gen_loss += fm_loss * cfg.lambdas.lambda_feat_match_loss
            loss_dict['fm_loss'] = fm_loss
            if 'spec_discriminator' in fm_loss_by_name:
                loss_dict['spec_fm_loss'] = fm_loss_by_name['spec_discriminator']
            if 'cqt_discriminator' in fm_loss_by_name:
                loss_dict['cqt_fm_loss'] = fm_loss_by_name['cqt_discriminator']
            if timing_active:
                self._timing_add('gen_loss_feat_match', self._timing_now() - fm_t0)

        if apn_cfg is not None:
            if 'apn_aux_loss' not in self.criteria:
                raise RuntimeError("APNet2 auxiliary loss is enabled but not constructed.")
            pred_aux = output.get('decoder_aux')
            if pred_aux is None:
                raise RuntimeError("APNet2 auxiliary loss is enabled but decoder auxiliary outputs are missing.")
            apn_loss_t0 = self._timing_now() if timing_active else None
            apn_scale = self._apn_aux_scale()
            apn_losses = self.criteria['apn_aux_loss'](gt_wav=y, gen_wav=y_, pred_aux=pred_aux)
            spectral_recon_loss = (
                apn_losses['apn_consistency_loss']
                + float(apn_cfg.get('real_imag_weight', 2.25))
                * (apn_losses['apn_real_loss'] + apn_losses['apn_imag_loss'])
            )
            apn_total = (
                float(apn_cfg.get('amplitude_weight', 45.0)) * apn_losses['apn_amplitude_loss']
                + float(apn_cfg.get('phase_weight', 100.0)) * apn_losses['apn_phase_loss']
                + float(apn_cfg.get('consistency_weight', 20.0)) * spectral_recon_loss
            )
            apn_total = apn_total * apn_scale
            gen_loss += apn_total
            loss_dict.update(apn_losses)
            loss_dict['apn_spectral_recon_loss'] = spectral_recon_loss
            loss_dict['apn_aux_loss'] = apn_total
            loss_dict['apn_aux_scale'] = torch.tensor(apn_scale, device=self.device)
            loss_dict['mel_loss_lambda'] = torch.tensor(self._effective_mel_loss_lambda(), device=self.device)
            if timing_active:
                self._timing_add('gen_loss_apn_aux', self._timing_now() - apn_loss_t0)

        # vq
        if vq_loss is not None:
            if isinstance(vq_loss, list):
                vq_loss = sum(vq_loss)
            if isinstance(vq_loss, torch.Tensor) and vq_loss.dim() > 0:
                vq_loss = vq_loss.mean()
            loss_dict['vq_loss'] = vq_loss
            if self.cfg.model.codec_decoder.quantizer_type == "simvq":
                vq_loss = self.cfg.train.lambdas.lambda_vq_loss * vq_loss
            gen_loss = gen_loss + vq_loss

        spkcfg = self.cfg.model.speaker_encoder
        spk_stack_cfg = spkcfg.get('stack', None)
        use_pe = bool(spkcfg.use_perceiver_encoder or (spk_stack_cfg is not None and spk_stack_cfg.get('use', False)))
        use_quantizer = getattr(spkcfg, 'use_quantizer', True)
        transit_ok = self.total_step > self.cfg.train.transit_step
        fade_step = self.cfg.train.xd_loss_fade_step
        in_fade_window = (fade_step is None) or (self.total_step < fade_step)

        if use_pe and use_quantizer and transit_ok and in_fade_window:
            xd_loss = self.criteria['l2_loss'](x_vector.detach(), d_vector)
            gen_loss += xd_loss * cfg.lambdas.lambda_xd_loss
            loss_dict['xd_loss'] = xd_loss
        else:
            loss_dict['xd_loss'] = torch.tensor(0., device=self.device)
        
        # f0 VQ loss (TriXCodec: None, as F0 is jointly quantized with content)
        if f0_vq_loss is not None:
            if isinstance(f0_vq_loss, list):
                f0_vq_loss = sum(f0_vq_loss)
            if isinstance(f0_vq_loss, torch.Tensor) and f0_vq_loss.dim() > 0:
                f0_vq_loss = f0_vq_loss.mean()
            loss_dict['f0_vq_loss'] = f0_vq_loss
            f0_vq_loss = self.cfg.train.lambdas.lambda_f0_vq_loss * f0_vq_loss
            gen_loss = gen_loss + f0_vq_loss
        else:
            # TriXCodec: no separate F0 VQ loss
            loss_dict['f0_vq_loss'] = torch.tensor(0.0, device=self.device)

        # print('gen f0 max min', torch.max(f0_vuv_.select(1, 0)).item(), torch.min(f0_vuv_.select(1, 0)).item())
        # print('gen vuv max min', torch.max(f0_vuv_.select(1, 1)).item(), torch.min(f0_vuv_.select(1, 1)).item())
        # gt f0_vuv is always 2 channels: [0]=f0, [1]=vuv
        # FCPE mode: f0_vuv_ is GT (not predicted), skip F0 reconstruction loss
        use_fcpe = self.cfg.model.f0_codec.get('use_fcpe_loss', False)
        use_unnorm_mse = self.use_unnormf0_mse_loss
        if not use_fcpe:
            vuv = f0_vuv.select(1, 1)
            f0_pred = f0_vuv_.select(1, 0)
            if use_unnorm_mse:
                f0_target = gt_f0_log.squeeze(1)
            else:
                f0_target = f0_vuv.select(1, 0)

            if self.cfg.model.f0_codec.zero_out_all_unvoiced:
                vuv_pred = f0_vuv_.select(1, 1)
                vuv_bce_loss = self.criteria['bcewlogits_loss'](vuv_pred, vuv)
            else:
                vuv_bce_loss = torch.tensor(0.0, device=self.device)

            if f0_target.dim() == 3:
                f0_target = f0_target.squeeze(1)
            if f0_pred.dim() == 3:
                f0_pred = f0_pred.squeeze(1)
            vuv_mask = vuv
            if vuv_mask.dim() == 3:
                vuv_mask = vuv_mask.squeeze(1)
            f0 = f0_target
            f0_ = f0_pred
        else:
            # FCPE mode: no direct F0 reconstruction
            vuv_bce_loss = torch.tensor(0.0, device=self.device)
        
        # F0 reconstruction loss: skip if using FCPE loss
        # When use_fcpe_loss=True, we only train with FCPE latent BCE loss
        if not use_fcpe:
            f0_l2_loss = self.criteria['l2_loss'](f0_, f0)
        else:
            f0_l2_loss = torch.tensor(0.0, device=self.device)

        # FCPE-style loss (optional)
        if use_fcpe and f0_fcpe_latent is not None and gt_f0 is not None:
            # gt_f0: (B, 1, T), f0_fcpe_latent: (B, fcpe_out_dims, T)
            # Convert GT F0 to FCPE target latent (gaussian blurred cent)
            with torch.no_grad():
                # f0_to_cent and gaussian_blurred_cent2latent expect (B, T, 1) input
                # gt_f0 is (B, 1, T), so transpose it
                gt_f0_transposed = gt_f0.transpose(1, 2)  # (B, T, 1)
                
                # Convert to cent and then to gaussian blurred latent
                # Note: f0=0 (unvoiced) → cent=-inf, but gaussian_blurred_cent2latent's mask filters it out
                # The mask is (cents > 0.1), so -inf frames get masked to 0 in the final latent
                gt_cent_f0 = self.model['f0_extractor'].f0_to_cent(gt_f0_transposed)  # (B, T, 1), may contain -inf
                gt_fcpe_latent = self.model['f0_extractor'].gaussian_blurred_cent2latent(gt_cent_f0)  # (B, T, 360), -inf masked to 0
                
                # Transpose to (B, 360, T) to match f0_fcpe_latent shape
                gt_fcpe_latent = gt_fcpe_latent.transpose(1, 2)  # (B, 360, T)

            # Align temporal resolution if extractor/decoder lengths differ
            # (e.g., upsample_extracted_f0=False with decoder producing audio-length FCPE latent).
            if f0_fcpe_latent.shape[-1] != gt_fcpe_latent.shape[-1]:
                if f0_fcpe_latent.shape[-1] > gt_fcpe_latent.shape[-1]:
                    f0_fcpe_latent = F.interpolate(
                        f0_fcpe_latent,
                        size=gt_fcpe_latent.shape[-1],
                        mode="nearest",
                    )
                else:
                    gt_fcpe_latent = F.interpolate(
                        gt_fcpe_latent,
                        size=f0_fcpe_latent.shape[-1],
                        mode="nearest",
                    )
            if f0_fcpe_logits is None:
                raise RuntimeError("FCPE loss enabled but f0_decoder did not expose fcpe logits.")
            if f0_fcpe_logits.shape[-1] != gt_fcpe_latent.shape[-1]:
                f0_fcpe_logits = F.interpolate(
                    f0_fcpe_logits,
                    size=gt_fcpe_latent.shape[-1],
                    mode="nearest",
                )

            fcpe_logits_fp32 = f0_fcpe_logits.float()
            gt_fcpe_latent_fp32 = gt_fcpe_latent.float()
            if not torch.isfinite(fcpe_logits_fp32).all():
                finite_mask = torch.isfinite(fcpe_logits_fp32)
                finite_vals = fcpe_logits_fp32[finite_mask]
                stats = "all values are non-finite"
                if finite_vals.numel() > 0:
                    stats = (
                        f"finite_min={finite_vals.min().item():.4g}, "
                        f"finite_max={finite_vals.max().item():.4g}, "
                        f"finite_mean={finite_vals.mean().item():.4g}"
                    )
                decoder_debug = getattr(self.model['f0_decoder'], 'last_forward_debug', None)
                f0_path_debug = getattr(self, '_last_f0_path_debug', None)
                raise RuntimeError(
                    "Non-finite FCPE logits detected before BCEWithLogitsLoss "
                    f"(step={self.global_step}, rank={self.global_rank}, {stats}, "
                    f"non_finite={int((~finite_mask).sum().item())}, "
                    f"decoder_debug={decoder_debug}, "
                    f"f0_path_debug={f0_path_debug})."
                )
            if not torch.isfinite(gt_fcpe_latent_fp32).all():
                raise RuntimeError(
                    "Non-finite FCPE targets detected before BCEWithLogitsLoss "
                    f"(step={self.global_step}, rank={self.global_rank})."
                )

            # BCEWithLogits is more numerically stable than BCE on sigmoid outputs.
            with autocast(device_type=fcpe_logits_fp32.device.type, enabled=False):
                fcpe_loss = F.binary_cross_entropy_with_logits(
                    fcpe_logits_fp32,
                    gt_fcpe_latent_fp32,
                )
            loss_dict['fcpe_loss'] = fcpe_loss
            gen_loss = gen_loss + fcpe_loss * cfg.lambdas.get('lambda_fcpe_loss', 10.0)
        else:
            loss_dict['fcpe_loss'] = torch.tensor(0.0, device=self.device)

        loss_dict['f0_loss'], loss_dict['vuv_loss'] = f0_l2_loss, vuv_bce_loss
        gen_loss = gen_loss + vuv_bce_loss * cfg.lambdas.lambda_vuv_recon_loss
        gen_loss = gen_loss + f0_l2_loss * cfg.lambdas.lambda_f0_recon_loss

        self.set_discriminator_gradients(True)
        loss_dict['gen_loss'] = gen_loss
        if timing_active:
            self._timing_add('gen_loss_total', self._timing_now() - gen_t0)
        return loss_dict
    
    def training_step(self, batch, batch_idx):
        # 스텝별 encoder freeze/unfreeze 반영
        timing_active = self._timing_is_active()
        step_t0 = self._timing_now() if timing_active else None
        self._maybe_update_encoder_freeze()
        forward_t0 = self._timing_now() if timing_active else None
        output = self(batch)
        if timing_active:
            self._timing_add('step_forward_wrapper', self._timing_now() - forward_t0)

        opts = self.optimizers()
        gen_opt, disc_opt = opts
        gen_sche, disc_sche = self.lr_schedulers()
        
        accum = int(self.cfg.train.gradient_accumulation_steps)
        is_update_step = ((batch_idx + 1) % accum == 0)
        zero_grad_set_to_none = bool(getattr(self.cfg.train, "optimizer_zero_grad_set_to_none", False))

        # 1) Discriminator
        self.toggle_optimizer(disc_opt)
        disc_loss_t0 = self._timing_now() if timing_active else None
        disc_losses = self.compute_disc_loss(batch, output)
        disc_loss = disc_losses['disc_loss'] / accum
        if timing_active:
            self._timing_add('step_disc_loss_wrapper', self._timing_now() - disc_loss_t0)
        try:
            # DDP에서 그래드 적산 중에는 통신 생략
            disc_backward_t0 = self._timing_now() if timing_active else None
            with self._no_sync_if_needed(self._sync_needed() and not is_update_step):
                self.manual_backward(disc_loss)
            if timing_active:
                self._timing_add('step_disc_backward', self._timing_now() - disc_backward_t0)
            if is_update_step:
                disc_opt_t0 = self._timing_now() if timing_active else None
                self.clip_gradients(disc_opt, gradient_clip_val=self.cfg.train.disc_grad_clip, gradient_clip_algorithm='norm')
                disc_opt.step()
                if self.criteria['gan_loss'].mode == 'bigvsan':
                    self._normalize_discriminator_post_weights()
                disc_opt.zero_grad(set_to_none=zero_grad_set_to_none)
                if disc_sche is not None:
                    disc_sche.step()
                if timing_active:
                    self._timing_add('step_disc_optim', self._timing_now() - disc_opt_t0)
        finally:
            self.untoggle_optimizer(disc_opt)

        # 2) Generator
        self.toggle_optimizer(gen_opt)
        # toggle_optimizer가 requires_grad를 바꾸므로 즉시 재적용
        self._maybe_update_encoder_freeze()
        gen_loss_t0 = self._timing_now() if timing_active else None
        gen_losses = self.compute_gen_loss(
            batch, output,
            cached_real=disc_losses.get('disc_real_feats_by_name'),
            cached_input_crops=disc_losses.get('disc_input_crops_by_name'),
        )
        if timing_active:
            self._timing_add('step_gen_loss_wrapper', self._timing_now() - gen_loss_t0)

        gen_loss = gen_losses['gen_loss'] / accum
        try:
            # DDP에서 그래드 적산 중에는 통신 생략
            gen_backward_t0 = self._timing_now() if timing_active else None
            with self._no_sync_if_needed(self._sync_needed() and not is_update_step):
                self.manual_backward(gen_loss)
            if timing_active:
                self._timing_add('step_gen_backward', self._timing_now() - gen_backward_t0)
            if is_update_step:
                gen_opt_t0 = self._timing_now() if timing_active else None
                self.clip_gradients(gen_opt, gradient_clip_val=self.cfg.train.gen_grad_clip, gradient_clip_algorithm='norm')
                gen_opt.step()
                gen_opt.zero_grad(set_to_none=zero_grad_set_to_none)
                if gen_sche is not None:
                    gen_sche.step()
                if timing_active:
                    self._timing_add('step_gen_optim', self._timing_now() - gen_opt_t0)
        finally:
            self.untoggle_optimizer(gen_opt)

        if is_update_step:
            self.total_step += 1
            log_t0 = self._timing_now() if timing_active else None
            self._log_losses(
                stage='train',
                disc_losses=disc_losses, gen_losses=gen_losses, mi_losses=None,
                output=output, batch_size=self.cfg.dataset.train.batch_size,
                gen_opt=gen_opt, disc_opt=disc_opt,
                on_step=True, on_epoch=False
            )
            if timing_active:
                self._timing_add('step_logging', self._timing_now() - log_t0)
                self._timing_add('step_total', self._timing_now() - step_t0)
                self._timing_maybe_log()

    def _load_hubert_with_retry(self, max_retries=5, base_delay=2):
        """Load HuBERT model with retry logic for network errors.

        If the model is already cached locally, fall back to local-only loading
        when HF Hub rate limits or network errors occur.
        """
        print(colored("Loading HuBERT model for validation WER...", "yellow"))

        def _looks_like_rate_limit(err: Exception) -> bool:
            text = f"{type(err).__name__}: {err}"
            return "429" in text or "Too Many Requests" in text

        def _looks_like_network_error(err: Exception) -> bool:
            text = f"{type(err).__name__}: {err}"
            needles = (
                "ReadTimeout",
                "Read timed out",
                "ConnectionError",
                "ConnectTimeout",
                "Temporary failure in name resolution",
                "MaxRetryError",
                "HTTPSConnectionPool",
            )
            return any(needle in text for needle in needles)

        def _load_hubert(local_files_only: bool):
            self.hubert_processor = AutoProcessor.from_pretrained(
                self.val_hubert_model_name,
                local_files_only=local_files_only
            )
            self.hubert_model = HubertForCTC.from_pretrained(
                self.val_hubert_model_name,
                local_files_only=local_files_only
            ).to(self.device)
            self.hubert_model.eval()

        offline_env = (
            os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        )
        if offline_env and self._has_local_hubert_cache():
            _load_hubert(local_files_only=True)
            self._wer_runtime_status = "enabled_local_cache"
            self._wer_status_detail = "HuBERT loaded from local cache because offline mode is enabled."
            self._print_wer_status("validation_wer_ready")
            return

        for attempt in range(max_retries):
            try:
                _load_hubert(local_files_only=False)
                self._wer_runtime_status = "enabled_online_or_cache_validated"
                self._wer_status_detail = "HuBERT loaded with online hub access enabled."
                self._print_wer_status("validation_wer_ready")
                return
            except (RequestsConnectionError, ConnectionError, Urllib3ProtocolError, OSError, Exception) as e:
                if (_looks_like_rate_limit(e) or _looks_like_network_error(e)) and self._has_local_hubert_cache():
                    print(colored(
                        "HF Hub request failed or was rate limited; loading HuBERT from local cache.",
                        "yellow"
                    ))
                    _load_hubert(local_files_only=True)
                    self._wer_runtime_status = "enabled_local_cache_after_hub_failure"
                    self._wer_status_detail = f"HuBERT loaded from local cache after hub failure: {type(e).__name__}: {e}"
                    self._print_wer_status("validation_wer_ready")
                    return
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(colored(
                        f"Failed to load HuBERT model (attempt {attempt + 1}/{max_retries}): {str(e)}",
                        "yellow"
                    ))
                    print(colored(f"Retrying in {delay} seconds...", "yellow"))
                    time.sleep(delay)
                else:
                    print(colored(
                        f"Failed to load HuBERT model after {max_retries} attempts: {str(e)}",
                        "red"
                    ))
                    print(colored(
                        "Warning: Validation WER metrics will be skipped due to HuBERT loading failure.",
                        "yellow"
                    ))
                    # Disable WER validation to prevent future attempts
                    self.use_val_wer = False
                    self._wer_runtime_status = "disabled_after_hubert_load_failure"
                    self._wer_status_detail = f"WER disabled because HuBERT could not be loaded: {type(e).__name__}: {e}"
                    self._print_wer_status("validation_wer_disabled")
                    return

    def on_validation_epoch_start(self):
        self.stoi.reset()
        self.si_snr.reset()
        self.si_sdr.reset()
        
        # Skip heavy model loading during sanity check
        is_sanity_check = self.trainer.sanity_checking if hasattr(self.trainer, 'sanity_checking') else False
        
        # Initialize UTMOS/WavLM accumulators
        self.val_utmos_same_scores = []
        self.val_utmos_vc_scores = []
        self.val_wavlm_same_scores = []
        self.val_wavlm_vc_scores = []
        if self.use_val_wer:
            self.val_wer_same_scores = []
            self.val_wer_vc_scores = []
        
        # Skip loading heavy models during sanity check
        if is_sanity_check:
            if self.use_val_wer and not self._wer_sanity_logged:
                self._wer_sanity_logged = True
                self._wer_runtime_status = "pending_first_real_validation"
                self._wer_status_detail = "Sanity validation skips WER initialization. First non-sanity validation will decide."
                self._print_wer_status("sanity_validation_skip")
            return
        
        # Load UTMOS model if needed
        if self.use_val_utmos and self.utmos_predictor is None:
            print(colored("Loading UTMOS predictor for validation...", "yellow"))
            # 상대 경로를 절대 경로로 변환하여 캐시 재사용
            import os.path as osp
            torch_cache_dir = osp.join(osp.dirname(__file__), '..', '..', 'etc', 'torch_hub_cache')
            os.makedirs(torch_cache_dir, exist_ok=True)
            os.environ['TORCH_HOME'] = torch_cache_dir
            self.utmos_predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
            self.utmos_predictor = self.utmos_predictor.to(self.device)
            self.utmos_predictor.eval()
            print(colored("✓ UTMOS loaded", "green"))
        
        # Load WavLM model if needed
        if self.use_val_wavlm and self.wavlm_model is None:
            print(colored("Loading WavLM_large model for validation...", "yellow"))
            # s3prl 캐시 경로 설정
            import os.path as osp
            s3prl_cache_dir = osp.join(osp.dirname(__file__), '..', '..', 'etc', 'torch_hub_cache', 's3prl')
            _set_s3prl_download_dir(s3prl_cache_dir)
            
            import sys
            from pathlib import Path as _Path
            sv_dir = _Path(__file__).parent.parent.parent / "etc" / "UniSpeech" / "downstreams" / "speaker_verification"
            sys.path.append(str(sv_dir))
            from etc.unispeech_models.ecapa_tdnn import ECAPA_TDNN_SMALL
            self.wavlm_model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None).to(self.device)
            self.wavlm_model.eval()
            checkpoint_path = _Path(__file__).parent.parent.parent / "etc" / "unispeech_models" / "wavlm_large_finetune.pth"
            if checkpoint_path.exists():
                state_dict = torch.load(str(checkpoint_path), map_location=self.device)
                # Filter out keys that don't exist in the model (e.g., loss_calculator components)
                model_state_dict = self.wavlm_model.state_dict()
                filtered_state_dict = {k: v for k, v in state_dict['model'].items() if k in model_state_dict}
                missing_keys, unexpected_keys = self.wavlm_model.load_state_dict(filtered_state_dict, strict=False)
                if missing_keys:
                    print(colored(f"Warning: Missing keys when loading WavLM: {missing_keys}", "yellow"))
                if unexpected_keys:
                    print(colored(f"Warning: Unexpected keys in checkpoint (filtered out): {unexpected_keys}", "yellow"))
                print(colored("✓ WavLM loaded with checkpoint", "green"))
            else:
                raise FileNotFoundError(f"WavLM checkpoint not found at {checkpoint_path}")
        
        if self.use_val_wer and self.hubert_model is None:
            self._wer_runtime_status = "loading_hubert_for_validation"
            self._wer_status_detail = "Starting HuBERT load for the first non-sanity validation."
            self._print_wer_status("validation_wer_init")
            self._load_hubert_with_retry()

    def on_validation_epoch_end(self):
        assert self.si_snr.total == self.si_sdr.total == self.stoi.count, f"Metrics count mismatch: {self.si_snr.total}, {self.si_sdr.total}, {self.stoi.count}"
        self.log('val_stats/si_snr', self.si_snr.compute(), on_epoch=True, logger=True, sync_dist=True)
        self.log('val_stats/si_sdr', self.si_sdr.compute(), on_epoch=True, logger=True, sync_dist=True)
        self.log('val_stats/stoi', self.stoi.compute(), on_epoch=True, logger=True, sync_dist=True)
        
        # Log UTMOS scores if available
        if self.use_val_utmos and len(self.val_utmos_same_scores) > 0:
            mean_utmos_same = float(np.mean(self.val_utmos_same_scores))
            self.log('val_stats/utmos_same', mean_utmos_same, on_epoch=True, logger=True, sync_dist=True)
            print(colored(f"Mean UTMOS (same-speaker): {mean_utmos_same:.4f}", "cyan"))
            
        if self.use_val_utmos and len(self.val_utmos_vc_scores) > 0:
            mean_utmos_vc = float(np.mean(self.val_utmos_vc_scores))
            self.log('val_stats/utmos_vc', mean_utmos_vc, on_epoch=True, logger=True, sync_dist=True)
            print(colored(f"Mean UTMOS (VC): {mean_utmos_vc:.4f}", "cyan"))
        
        # Log WavLM speaker similarity scores if available
        if self.use_val_wavlm and len(self.val_wavlm_same_scores) > 0:
            mean_wavlm_same = float(np.mean(self.val_wavlm_same_scores))
            self.log('val_stats/wavlm_sim_same', mean_wavlm_same, on_epoch=True, logger=True, sync_dist=True)
            print(colored(f"Mean WavLM Similarity (same-speaker): {mean_wavlm_same:.4f}", "cyan"))
            
        if self.use_val_wavlm and len(self.val_wavlm_vc_scores) > 0:
            mean_wavlm_vc = float(np.mean(self.val_wavlm_vc_scores))
            self.log('val_stats/wavlm_sim_vc', mean_wavlm_vc, on_epoch=True, logger=True, sync_dist=True)
            print(colored(f"Mean WavLM Similarity (VC): {mean_wavlm_vc:.4f}", "cyan"))

        if self.use_val_wer and len(getattr(self, 'val_wer_same_scores', [])) > 0:
            mean_wer_same = float(np.mean(self.val_wer_same_scores))
            self.log('val_stats/wer_same', mean_wer_same, on_epoch=True, logger=True, sync_dist=True)
            print(colored(f"Mean WER (same-speaker, HuBERT): {mean_wer_same:.4f}", "cyan"))
        if self.use_val_wer and len(getattr(self, 'val_wer_vc_scores', [])) > 0:
            mean_wer_vc = float(np.mean(self.val_wer_vc_scores))
            self.log('val_stats/wer_vc', mean_wer_vc, on_epoch=True, logger=True, sync_dist=True)
            print(colored(f"Mean WER (VC, HuBERT): {mean_wer_vc:.4f}", "cyan"))

        # Clean up validation models to free memory
        if self.utmos_predictor is not None:
            del self.utmos_predictor
            self.utmos_predictor = None
            print(colored("✓ UTMOS predictor deleted", "cyan"))
        
        if self.wavlm_model is not None:
            del self.wavlm_model
            self.wavlm_model = None
            print(colored("✓ WavLM model deleted", "cyan"))
        
        if self.hubert_model is not None:
            del self.hubert_model
            self.hubert_model = None
            print(colored("✓ HuBERT model deleted", "cyan"))
        
        if self.hubert_processor is not None:
            del self.hubert_processor
            self.hubert_processor = None
            print(colored("✓ HuBERT processor deleted", "cyan"))
        
        torch.cuda.empty_cache()

        # 시각화 처리

        is_main = True
        if torch.distributed.is_initialized():
            is_main = (torch.distributed.get_rank() == 0)

        if self.val_step_plot_outputs and is_main:
            matplotlib.use('Agg')
            samples = self.val_step_plot_outputs
            sr, hop_length = 16000, 80
            
            # FCPE 모드: raw Hz 표시, Normal/Log1p 모드: normalized 값 표시
            use_fcpe_mode = (
                self.cfg.model.f0_codec.get('use_normalized_f0', False)
                and self.cfg.model.f0_codec.get('use_fcpe_loss', False)
            )
            use_log1p_mode = self.use_unnormf0_mse_loss
            
            # 3x3 grid layout (공통)
            rows, cols = 3, min(3, (len(samples) + 2) // 3)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
            
            for i, sample in enumerate(samples[:rows * cols]):
                r, c = i % rows, i // rows
                mel = sample['mel'].numpy()
                vuv_gt, vuv_rec = sample['vuv_gt'].numpy().squeeze(), sample['vuv_rec'].numpy().squeeze()
                
                # FCPE 모드: raw Hz로 변환
                if use_fcpe_mode:
                    f0_gt_raw = sample['f0_gt_raw']
                    f0_gt = f0_gt_raw.numpy().squeeze() if f0_gt_raw is not None else None
                    f0_rec = np.expm1(sample['f0_rec_z'].numpy().squeeze())  # log1p → Hz
                    ylabel, yscale, ylim = 'F0 (Hz)', 'log', (50, 500)
                elif use_log1p_mode:
                    # Unnormalized MSE 모드: log1p(raw F0)
                    f0_gt = sample['f0_gt_z'].numpy().squeeze()
                    f0_rec = sample['f0_rec_z'].numpy().squeeze()
                    ylabel, yscale, ylim = 'log1p F0', 'linear', None
                else:
                    # Normal 모드: normalized F0 사용
                    f0_gt = sample['f0_gt_z'].numpy().squeeze()
                    f0_rec = sample['f0_rec_z'].numpy().squeeze()
                    ylabel, yscale, ylim = 'Norm log-F0 (z)', 'linear', None
                
                times = librosa.times_like(f0_gt if f0_gt is not None else f0_rec, sr=sr, hop_length=hop_length)
                
                # Mel spectrogram
                librosa.display.specshow(mel, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='viridis', ax=axes[r][c])
                axes[r][c].set_title(sample['fid'])
                
                # F0 overlay
                ax2 = axes[r][c].twinx()
                if f0_gt is not None:
                    ax2.plot(times, f0_gt, 'cyan', lw=1.2, label='f0_gt', zorder=3)
                ax2.plot(times, f0_rec, 'r--', lw=1.0, label='f0_rec', zorder=2)
                
                # V/UV shading
                if vuv_gt.max() > 0 or vuv_rec.max() > 0:
                    if f0_gt is not None:
                        ymin, ymax = min(f0_gt.min(), f0_rec.min()), max(f0_gt.max(), f0_rec.max())
                    else:
                        ymin, ymax = f0_rec.min(), f0_rec.max()
                    if vuv_gt.max() > 0:
                        ax2.fill_between(times, ymin * 0.9, ymax * 1.1, where=vuv_gt > 0.5, 
                                       color='white', alpha=0.2, interpolate=True, label='voiced_gt', zorder=0)
                    if vuv_rec.max() > 0:
                        ax2.fill_between(times, ymin * 0.9, ymax * 1.1, where=vuv_rec > 0.5,
                                       color='magenta', alpha=0.15, interpolate=True, label='voiced_pred', zorder=1)
                    if not use_fcpe_mode:  # Normal & log1p 모드: 기존 margin 계산
                        margin = 0.1 * max(1e-6, abs(f0_gt).max() + abs(f0_rec).max()) if f0_gt is not None else 0.1
                        ax2.set_ylim(ymin - margin, ymax + margin)
                
                ax2.set_ylabel(ylabel)
                if yscale: ax2.set_yscale(yscale)
                if ylim: ax2.set_ylim(ylim)
                ax2.legend(loc='upper right', fontsize=7)
                ax2.grid(alpha=0.25, ls=':')

            plt.tight_layout()
            if self.logger and isinstance(self.logger, pl.loggers.wandb.WandbLogger):
                out_dir = "val_vis"
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, f"val_mel_f0{'_fcpe' if use_fcpe_mode else ''}_epoch_{self.current_epoch}.png")
                plt.savefig(path, dpi=150, bbox_inches='tight')
                self.logger.experiment.log({'val/mel_f0_grid': wandb.Image(path)}, step=int(self.global_step))
            plt.close()

        self.val_step_plot_outputs.clear()

    def _collect_val_vis_sample(self, batch, output, batch_idx):
        # 이미 충분히 모았으면 중단
        if len(self.val_step_plot_outputs) >= 9:
            return
        matplotlib.use('Agg')
        wav = batch['wav'][0].unsqueeze(0)  # [1, T_wav]
        self.mel_spectrogram_transform.to(wav.device)
        mel = self.mel_spectrogram_transform(wav)  # [1, n_mels, T_mel]
        amin = 1e-10
        ref_value = mel.max()
        mel_db = 10.0 * torch.log10(torch.clamp(mel, min=amin) / ref_value)

        target_len = output['gt_f0_vuv'].shape[-1]
        mel_db_interp = F.interpolate(mel_db, size=target_len, mode='linear', align_corners=False).squeeze(0)

        # gt_f0_vuv is always 2 channels: [0]=f0(encoder input), [1]=vuv
        if self.use_unnormf0_mse_loss:
            f0_gt_z = output['gt_f0_log'][0, 0].detach().cpu()
        else:
            f0_gt_z = output['gt_f0_vuv'][0, 0].detach().cpu()
        vuv_gt = output['gt_f0_vuv'][0, 1].detach().cpu()
        
        f0_rec_z = output['gen_f0_vuv'][0, 0].detach().cpu()
        vuv_rec = output['gen_f0_vuv'][0, 1].detach().cpu() if self.cfg.model.f0_codec.zero_out_all_unvoiced else vuv_gt

        # FCPE 모드: raw GT F0도 저장 (스케일 비교용)
        f0_gt_raw = output.get('gt_f0', None)  # (B, 1, T) raw Hz
        if f0_gt_raw is not None:
            f0_gt_raw = f0_gt_raw[0, 0].detach().cpu()  # (T,)
        
        sample = {
            'fid': batch.get('fid', ['unk'])[0],
            'mel': mel_db_interp.detach().cpu(),    # [n_mels, T]
            'f0_gt_z': f0_gt_z.unsqueeze(0),        # [1, T] normalized or log1p
            'f0_rec_z': f0_rec_z.unsqueeze(0),      # [1, T] normalized or log1p
            'f0_gt_raw': f0_gt_raw.unsqueeze(0) if f0_gt_raw is not None else None,  # [1, T] raw Hz
            'vuv_gt': vuv_gt.unsqueeze(0),          # [1, T]
            'vuv_rec': vuv_rec.unsqueeze(0),        # [1, T]
        }
        self.val_step_plot_outputs.append(sample)

    def _transcribe_hubert(self, audio_16k: torch.Tensor) -> str:
        if self.hubert_processor is None or self.hubert_model is None:
            raise RuntimeError("HuBERT processor/model not initialized. Enable cfg.train.use_val_wer.")
        if audio_16k.dim() > 1:
            audio_16k = audio_16k.squeeze(0)
        audio_np = audio_16k.detach().cpu().float().numpy()
        inputs = self.hubert_processor(audio_np, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            logits = self.hubert_model(input_values, attention_mask=attention_mask).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = self.hubert_processor.batch_decode(pred_ids)[0].strip()
        return text

    def _transcribe_hubert_batch(self, audio_16k: torch.Tensor) -> list[str]:
        if self.hubert_processor is None or self.hubert_model is None:
            raise RuntimeError("HuBERT processor/model not initialized. Enable cfg.train.use_val_wer.")
        if audio_16k.dim() == 1:
            audio_16k = audio_16k.unsqueeze(0)
        audio_list = [sample.detach().cpu().float().numpy() for sample in audio_16k]
        inputs = self.hubert_processor(
            audio_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            logits = self.hubert_model(input_values, attention_mask=attention_mask).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return [text.strip() for text in self.hubert_processor.batch_decode(pred_ids)]

    def _compute_wer_jiwer(self, ref_text: str, hyp_text: str) -> float:
        out = jiwer.process_words(
            reference=ref_text,
            hypothesis=hyp_text,
            reference_transform=self._wer_transform,
            hypothesis_transform=self._wer_transform
        )
        return float(out.wer)

    def _append_utmos_scores(self, score_store, audio_16k: torch.Tensor, tag: str) -> None:
        if self.utmos_predictor is None or audio_16k.numel() == 0:
            return
        try:
            scores = self.utmos_predictor(audio_16k, 16000)
            if isinstance(scores, torch.Tensor):
                score_values = scores.detach().cpu().reshape(-1).tolist()
            else:
                score_values = np.asarray(scores).reshape(-1).tolist()
            score_store.extend(float(score) for score in score_values)
            return
        except Exception as e:
            print(colored(f"UTMOS batch error ({tag}), falling back to per-sample: {e}", "red"))

        for i in range(audio_16k.size(0)):
            try:
                score = self.utmos_predictor(audio_16k[i:i+1], 16000)
                if isinstance(score, torch.Tensor):
                    score = score.item()
                score_store.append(float(score))
            except Exception as sub_e:
                print(colored(f"UTMOS error ({tag}) sample={i}: {sub_e}", "red"))

    def _append_wer_scores(
        self,
        score_store,
        ref_audio_16k: torch.Tensor,
        hyp_audio_16k: torch.Tensor,
        tag: str,
    ) -> None:
        if self.hubert_model is None or ref_audio_16k.numel() == 0 or hyp_audio_16k.numel() == 0:
            return
        try:
            ref_texts = self._transcribe_hubert_batch(ref_audio_16k)
            hyp_texts = self._transcribe_hubert_batch(hyp_audio_16k)
            for ref_text, hyp_text in zip(ref_texts, hyp_texts):
                score_store.append(self._compute_wer_jiwer(ref_text, hyp_text))
            return
        except Exception as e:
            print(colored(f"HuBERT WER batch error ({tag}), falling back to per-sample: {e}", "red"))

        for i in range(min(ref_audio_16k.size(0), hyp_audio_16k.size(0))):
            try:
                ref_text = self._transcribe_hubert(ref_audio_16k[i])
                hyp_text = self._transcribe_hubert(hyp_audio_16k[i])
                wer_value = self._compute_wer_jiwer(ref_text, hyp_text)
                score_store.append(wer_value)
            except Exception as sub_e:
                print(colored(f"HuBERT WER error ({tag}) sample={i}: {sub_e}", "red"))
                score_store.append(1.0)
    
    def validation_step(self, batch, batch_idx):
        # Same-speaker reconstruction
        output = self(batch)

        disc_losses = self.compute_disc_loss(batch, output)
        gen_losses = self.compute_gen_loss(
            batch,
            output,
            cached_real=disc_losses.get('disc_real_feats_by_name'),
            cached_input_crops=disc_losses.get('disc_input_crops_by_name'),
        )
        mi_losses = None

        self.si_snr.update(output['gt_wav'].squeeze(1), output['gen_wav'].squeeze(1))
        self.si_sdr.update(output['gt_wav'].squeeze(1), output['gen_wav'].squeeze(1))
        self.stoi.update(output['gt_wav'].squeeze(1), output['gen_wav'].squeeze(1))

        self._log_losses(
            stage='val',
            disc_losses=disc_losses, gen_losses=gen_losses, mi_losses=mi_losses, output=output,
            batch_size=self.cfg.dataset.val.batch_size,
            on_step=False, on_epoch=True)
        
        # Voice conversion (different speaker) - only for UTMOS/WavLM metrics, no loss logging
        output_vc = None
        if 'ref_wav_vc' in batch and (self.use_val_utmos or self.use_val_wavlm or self.use_val_wer):
            batch_vc = batch.copy()
            batch_vc['ref_wav'] = batch['ref_wav_vc']
            output_vc = self(batch_vc)
        
        # Compute UTMOS and WavLM metrics directly (without saving audio)
        if self.use_val_utmos or self.use_val_wavlm or self.use_val_wer:
            with torch.no_grad():
                # Resample to 16kHz if needed for metrics
                def to_16k(audio):
                    # Squeeze to (B, T) first
                    if audio.dim() == 3:
                        audio = audio.squeeze(1)  # (B, 1, T) -> (B, T)
                    
                    if self.cfg.preprocess.audio.sr != 16000:
                        audio = torchaudio.functional.resample(
                            audio, orig_freq=self.cfg.preprocess.audio.sr, new_freq=16000
                        )
                    return audio  # (B, T)
                
                # Prepare audio: squeeze and resample
                gen_same_16k = to_16k(output['gen_wav'])  # (B, T)
                gt_same_16k = to_16k(output['gt_wav'])    # (B, T)
                ref_same_16k = batch['ref_wav']  # Already 16kHz, (B, T)
                if ref_same_16k.dim() == 3:
                    ref_same_16k = ref_same_16k.squeeze(1)
                
                # UTMOS for same-speaker
                if self.use_val_utmos and self.utmos_predictor is not None:
                    self._append_utmos_scores(self.val_utmos_same_scores, gen_same_16k, tag="same")
                
                # WavLM for same-speaker
                if self.use_val_wavlm and self.wavlm_model is not None:
                    try:
                        # WavLM expects (B, T)
                        ref_emb = self.wavlm_model(ref_same_16k)  # (B, D)
                        gen_emb = self.wavlm_model(gen_same_16k)  # (B, D)
                        cosine_sim = torch.nn.functional.cosine_similarity(ref_emb, gen_emb, dim=-1)
                        self.val_wavlm_same_scores.extend(cosine_sim.cpu().tolist())
                    except Exception as e:
                        print(colored(f"WavLM error (same): {e}", "red"))

                # HuBERT-based WER for same-speaker reconstruction
                if self.use_val_wer and self.hubert_model is not None:
                    self._append_wer_scores(
                        self.val_wer_same_scores,
                        gt_same_16k,
                        gen_same_16k,
                        tag="same",
                    )
                
                # Voice conversion metrics
                if 'ref_wav_vc' in batch and output_vc is not None:
                    gen_vc_16k = to_16k(output_vc['gen_wav'])  # (B, T)
                    gt_vc_16k = to_16k(output_vc['gt_wav'])    # (B, T)
                    ref_vc_16k = batch['ref_wav_vc']  # Already 16kHz, (B, T)
                    if ref_vc_16k.dim() == 3:
                        ref_vc_16k = ref_vc_16k.squeeze(1)
                    
                    # UTMOS for VC
                    if self.use_val_utmos and self.utmos_predictor is not None:
                        self._append_utmos_scores(self.val_utmos_vc_scores, gen_vc_16k, tag="vc")
                    
                    # WavLM for VC
                    if self.use_val_wavlm and self.wavlm_model is not None:
                        try:
                            # WavLM expects (B, T)
                            ref_vc_emb = self.wavlm_model(ref_vc_16k)  # (B, D)
                            gen_vc_emb = self.wavlm_model(gen_vc_16k)  # (B, D)
                            cosine_sim_vc = torch.nn.functional.cosine_similarity(ref_vc_emb, gen_vc_emb, dim=-1)
                            self.val_wavlm_vc_scores.extend(cosine_sim_vc.cpu().tolist())
                        except Exception as e:
                            print(colored(f"WavLM error (VC): {e}", "red"))

                    if self.use_val_wer and self.hubert_model is not None:
                        self._append_wer_scores(
                            self.val_wer_vc_scores,
                            gt_vc_16k,
                            gen_vc_16k,
                            tag="vc",
                        )
                
                # Explicit memory cleanup after VC metrics
                if output_vc is not None:
                    del output_vc
                if 'batch_vc' in locals():
                    del batch_vc
                torch.cuda.empty_cache()

        is_main = True
        if torch.distributed.is_initialized():
            is_main = (torch.distributed.get_rank() == 0)

        is_sanity_check = self.trainer.sanity_checking if hasattr(self.trainer, 'sanity_checking') else False
        if is_main and (not is_sanity_check) and batch_idx < 9:
            self._collect_val_vis_sample(batch, output, batch_idx)

    def test_step(self, batch, batch_idx):
        # remove weight norm for modules in self.model
        batch_size = batch['wav'].size(0)
        target_type = self.cfg.voice_conversion
        self.vc_f0 = True if target_type == 'vc_f0' else False
        
        assert target_type in ['rec', 'vc', 'vc_f0', 'same'], f"Invalid target_type: {target_type}"
        
        # 먼저 이 배치에서 처리할 샘플이 있는지 확인
        indices_to_process = []
        for i in range(batch_size):
            source_filename = batch['fid'][i]
            if target_type in ['rec', 'vc', 'vc_f0']:
                target_filename = batch['target_id'][i]
            elif target_type == 'same':
                target_filename = source_filename
            
            # 생성할 파일 경로들
            gen_path = f"{self.cwd}/{source_filename}-{target_filename}_{target_type}.wav"
            gt_path = f"{self.cwd}/{source_filename}-{target_filename}_gt.wav"
            ref_path = f"{self.cwd}/{source_filename}-{target_filename}_ref.wav"
            
            # 모든 파일이 이미 존재하는지 확인
            if os.path.exists(gen_path) and os.path.exists(gt_path) and os.path.exists(ref_path):
                print(colored(f"[Skip] All files exist for {source_filename}-{target_filename}", "yellow"))
                continue
            
            indices_to_process.append(i)
        
        # 처리할 샘플이 없으면 forward pass 자체를 건너뜀
        if len(indices_to_process) == 0:
            print(colored(f"[Skip] Batch {batch_idx} - all files already exist", "green"))
            return
        
        # Forward pass (필요한 샘플이 있을 때만)
        output = self(batch)
        
        # 처리해야 할 샘플들만 저장
        for i in indices_to_process:
            source_filename = batch['fid'][i]
            if target_type in ['rec', 'vc', 'vc_f0']:
                target_filename = batch['target_id'][i]
            elif target_type == 'same':
                target_filename = source_filename
            
            # _ref.wav는 get_ref_clip 이후(ref_wav) 대신 raw ref_src를 저장해야 함 (test only)
            if isinstance(batch.get('ref_src', None), torch.Tensor):
                ref_i = batch['ref_src'][i]
            else:
                raise ValueError("ref_src is not found in batch")
            gt_i, gen_i = output['gt_wav'][i], output['gen_wav'][i]
            # gt_i와 gen_i는 이미 (1, T) shape이므로 squeeze하지 않음
            # ref_i만 확인
            if ref_i.dim() == 1: ref_i = ref_i.unsqueeze(0)
            
            gen_path = f"{self.cwd}/{source_filename}-{target_filename}_{target_type}.wav"
            gt_path = f"{self.cwd}/{source_filename}-{target_filename}_gt.wav"
            ref_path = f"{self.cwd}/{source_filename}-{target_filename}_ref.wav"
            
            # gt_wav와 gen_wav는 forward에서 이미 올바른 샘플링 레이트를 가짐
            torchaudio.save(gen_path, gen_i.float().detach().cpu(), self.cfg.preprocess.audio.sr)
            torchaudio.save(gt_path, gt_i.float().detach().cpu(), self.cfg.preprocess.audio.sr)
            # ref_wav는 항상 16kHz (데이터 로더에서 ref_wav_16k로 제공됨)
            torchaudio.save(ref_path, ref_i.float().detach().cpu(), 16000)
            # print(colored(f"[Saved] {source_filename}-{target_filename}_{target_type}.wav", "green"))

    def on_test_start(self):
        self._remove_all_weight_norms()
        self.cwd = os.path.join(os.getcwd(), os.path.splitext(self.cfg.ckpt)[0], str(self.cfg.dataset.min_ref_seconds))
        if str(self.cfg.dataset.get("name", "")).lower() == "vctk":
            self.cwd = os.path.join(self.cwd, "vctk")
        os.makedirs(self.cwd, exist_ok=True)
        print(colored(f'Test results will be saved in : {self.cwd}', 'yellow', attrs=['bold']))

    def _remove_all_weight_norms(self):
        if getattr(self, "_wn_removed", False):
            return
        removed = self._remove_weight_norm_from_module(self.model)
        self._wn_removed = True
        print(f"[test] removed weight_norm from {removed} submodules.")

    def _remove_weight_norm_from_module(self, module: nn.Module) -> int:
        removed = 0
        for submodule in module.modules():
            if hasattr(submodule, "weight_g") and hasattr(submodule, "weight_v"):
                try:
                    nn.utils.remove_weight_norm(submodule)
                    removed += 1
                except Exception:
                    pass
        return removed

    def _remove_weight_norm_from_submodules(self, module_names) -> None:
        if not isinstance(self.model, nn.ModuleDict):
            print(colored("Warning: selective weight_norm removal requires ModuleDict model container.", "yellow"))
            return

        removed_by_name = {}
        missing = []
        for module_name in module_names:
            if module_name not in self.model:
                missing.append(str(module_name))
                continue
            removed_by_name[str(module_name)] = self._remove_weight_norm_from_module(self.model[module_name])

        if removed_by_name:
            summary = ", ".join(f"{name}:{count}" for name, count in removed_by_name.items())
            print(colored(f"[train] removed weight_norm from modules -> {summary}", "cyan"))
        if missing:
            print(colored(f"[train] skipped unknown weight_norm targets: {', '.join(missing)}", "yellow"))

    # --- encoder freeze 스케줄 관련 유틸 ---
    def _is_encoder_frozen(self):
        if not self._freeze_schedule_enabled:
            return False
        # total_step은 optimizer step 기준으로 증가
        return int(self.total_step.item()) < int(self.unfreeze_encoder_step)

    def _maybe_update_encoder_freeze(self):
        if not self._freeze_schedule_enabled:
            return
        should_freeze = self._is_encoder_frozen()
        if self._encoder_frozen is None or self._encoder_frozen != should_freeze:
            self.set_encoder_gradients(not should_freeze)
            self._encoder_frozen = should_freeze
            status = "FROZEN" if should_freeze else "UNFROZEN"
            print(colored(f"[step {int(self.total_step.item())}] Encoder set to {status}", "cyan"))

    def set_encoder_gradients(self, flag=True):
        # CodecEnc은 vq_wav2vec이든 자체 Encoder든 공통 키 사용
        for p in self.model['CodecEnc'].parameters():
            p.requires_grad = flag
        # eval/train 전환으로 BN/Dropout 정지 및 활성화
        self.model['CodecEnc'].train(flag)

    # -------- Logging Helpers (NEW) --------
    def _zero(self):
        return torch.zeros((), device=self.device)

    def _gather_loss_dict(self, disc_losses, gen_losses, mi_losses=None):
        z = self._zero
        apn_cfg = self._apn_aux_cfg()
        losses = {
            'disc_loss': disc_losses['disc_loss'],
            'fake_loss': disc_losses['fake_loss'],
            'real_loss': disc_losses['real_loss'],
            'gen_loss': gen_losses['gen_loss'],
            'vq_loss': gen_losses.get('vq_loss', z()),
            'mel_loss': gen_losses.get('mel_loss', z()),
            'fm_loss': gen_losses.get('fm_loss', z()),
            'adv_loss': gen_losses.get('adv_loss', z()),
            'spec_fm_loss': gen_losses.get('spec_fm_loss', z()),
            'cqt_fm_loss': gen_losses.get('cqt_fm_loss', z()),
            'xd_loss': gen_losses.get('xd_loss', z()),
            'f0_loss': gen_losses.get('f0_loss', z()),
            'f0_vq_loss': gen_losses.get('f0_vq_loss', z()),
            'vuv_loss': gen_losses.get('vuv_loss', z()),
            'fcpe_loss': gen_losses.get('fcpe_loss', z()),
        }
        if apn_cfg is not None:
            losses.update({
                'apn_aux_loss': gen_losses.get('apn_aux_loss', z()),
                'apn_amplitude_loss': gen_losses.get('apn_amplitude_loss', z()),
                'apn_phase_loss': gen_losses.get('apn_phase_loss', z()),
                'apn_phase_ip_loss': gen_losses.get('apn_phase_ip_loss', z()),
                'apn_phase_gd_loss': gen_losses.get('apn_phase_gd_loss', z()),
                'apn_phase_ptd_loss': gen_losses.get('apn_phase_ptd_loss', z()),
                'apn_spectral_recon_loss': gen_losses.get('apn_spectral_recon_loss', z()),
                'apn_consistency_loss': gen_losses.get('apn_consistency_loss', z()),
                'apn_real_loss': gen_losses.get('apn_real_loss', z()),
                'apn_imag_loss': gen_losses.get('apn_imag_loss', z()),
            })

        # --- scaled (lambda 적용 후) ---
        lmb = self.cfg.train.lambdas
        def _lam(key, default=1.0):
            return lmb.get(key, default)

        def _scale(raw_key, lambda_key, already_scaled=False, default_lambda=1.0):
            if raw_key not in losses:
                return
            if already_scaled:
                # disc_loss 내부에서 이미 λ 곱했으면 별도 scaled 표시 생략 가능
                losses[f'{raw_key}_scaled'] = losses[raw_key]
            else:
                lam = _lam(lambda_key, default_lambda)
                losses[f'{raw_key}_scaled'] = losses[raw_key] * lam

        losses['mel_loss_scaled'] = losses['mel_loss'] * self._effective_mel_loss_lambda()
        _scale('adv_loss', 'lambda_adv')
        _scale('fm_loss', 'lambda_feat_match_loss')
        _scale('spec_fm_loss', 'lambda_feat_match_loss')
        _scale('cqt_fm_loss', 'lambda_feat_match_loss')
        _scale('vq_loss', 'lambda_vq_loss')
        _scale('xd_loss', 'lambda_xd_loss')
        # disc_loss: 이미 compute_disc_loss에서 lambda_disc 곱했으면 already_scaled=True
        already = True  # 현재 compute_disc_loss에서 lambda_disc 적용했다고 가정
        _scale('disc_loss', 'lambda_disc', already_scaled=already)
        _scale('f0_loss', 'lambda_f0_recon_loss')
        _scale('f0_vq_loss', 'lambda_f0_vq_loss')
        _scale('vuv_loss', 'lambda_vuv_recon_loss')
        _scale('fcpe_loss', 'lambda_fcpe_loss', default_lambda=10.0)
        if apn_cfg is not None:
            losses['apn_amplitude_loss_scaled'] = losses['apn_amplitude_loss'] * float(apn_cfg.get('amplitude_weight', 45.0))
            losses['apn_phase_loss_scaled'] = losses['apn_phase_loss'] * float(apn_cfg.get('phase_weight', 100.0))
            losses['apn_spectral_recon_loss_scaled'] = losses['apn_spectral_recon_loss'] * float(apn_cfg.get('consistency_weight', 20.0))
            losses['apn_aux_loss_scaled'] = losses['apn_aux_loss']

        return losses

    def _log_losses(self, stage, disc_losses, gen_losses, mi_losses, output,
                    batch_size, gen_opt=None, disc_opt=None,
                    on_step=False, on_epoch=False):
        losses = self._gather_loss_dict(disc_losses, gen_losses, mi_losses)
        for k, v in losses.items():
            self.log(f'{stage}_loss/{k}', v,
                     on_step=on_step, on_epoch=on_epoch,
                     logger=True, sync_dist=True, batch_size=batch_size)

        # stats
        if stage == 'train':
            self.log('train_stats/total_step', self.total_step.float(),
                     logger=True, sync_dist=False)
            if gen_opt is not None:
                self.log('train_stats/lr_g', gen_opt.param_groups[0]['lr'],
                         logger=True, sync_dist=True)
            if disc_opt is not None:
                self.log('train_stats/lr_d', disc_opt.param_groups[0]['lr'],
                         logger=True, sync_dist=True)
        else:
            self.log('val_stats/total_step', self.total_step.float(),
                     on_epoch=True, logger=True, sync_dist=False)

        self.log(f'{stage}_stats/perplexity', output['perplexity'],
                 on_step=on_step, on_epoch=on_epoch,
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log(f'{stage}_stats/cluster_size', output['active_num'],
                 on_step=on_step, on_epoch=on_epoch,
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)

        if (
            stage == 'train'
            and on_step
            and bool(getattr(self.cfg, 'debug', False))
            and int(getattr(self, 'global_rank', 0)) == 0
        ):
            every = 50
            trainer = getattr(self, 'trainer', None)
            if trainer is not None:
                every = max(1, int(getattr(trainer, 'log_every_n_steps', 50)))
            total_step = getattr(self, 'total_step', 0)
            if torch.is_tensor(total_step):
                step = int(total_step.item())
            else:
                step = int(total_step)
            if step == 0 or step % every == 0:
                def _scalar(name):
                    value = losses.get(name, None)
                    if value is None:
                        return None
                    if torch.is_tensor(value):
                        if value.numel() == 0:
                            return None
                        return float(value.detach().float().mean().cpu().item())
                    return float(value)

                payload = {
                    'step': step,
                    'gen': _scalar('gen_loss'),
                    'disc': _scalar('disc_loss'),
                    'mel': _scalar('mel_loss_scaled'),
                    'adv': _scalar('adv_loss'),
                    'fm': _scalar('fm_loss'),
                    'fcpe': _scalar('fcpe_loss_scaled'),
                    'apn': _scalar('apn_aux_loss_scaled'),
                    'apn_amp': _scalar('apn_amplitude_loss'),
                    'apn_phase': _scalar('apn_phase_loss'),
                    'apn_spec': _scalar('apn_spectral_recon_loss'),
                    'f0': _scalar('f0_loss_scaled'),
                    'vuv': _scalar('vuv_loss_scaled'),
                }
                if gen_opt is not None:
                    payload['lr_g'] = float(gen_opt.param_groups[0]['lr'])
                if disc_opt is not None:
                    payload['lr_d'] = float(disc_opt.param_groups[0]['lr'])

                pieces = [f"{k}={v:.4g}" for k, v in payload.items() if v is not None]
                print("[loss-debug] " + " ".join(pieces), flush=True)
    # -------- End Logging Helpers --------

    def configure_optimizers(self):
        from itertools import chain
        disc_param_groups = [module.parameters() for _, module in self._iter_discriminator_modules()]
        disc_params = chain(*disc_param_groups)
        
        # TriXCodec: include joint_mixer and joint_to_audio_f0 in generator params
        gen_params_list = [
            self.model['CodecEnc'].parameters(),
            self.model['generator'].parameters(),
            self.model['speaker_encoder'].parameters(),
            self.model['f0_encoder'].parameters(),
            self.model['f0_decoder'].parameters(),
            self.model['joint_mixer'].parameters(),  # TriXCodec: joint projection layers
            self.model['joint_to_audio_f0'].parameters(),
        ]

        if self.cfg.model.f0_codec.get('finetune_f0_extractor', False):
            print(colored("Defining f0_extractor parameters to optimizer", "yellow"))
            gen_params_list.append(self.model['f0_extractor'].parameters())
        
        gen_params = chain(*gen_params_list)


        gen_opt = optim.AdamW(gen_params, **self.cfg.train.gen_optim_params)
        disc_opt = optim.AdamW(disc_params, **self.cfg.train.disc_optim_params)
        opt_list = [gen_opt, disc_opt]

        gen_sche = WarmupLR(gen_opt, **self.cfg.train.gen_schedule_params)
        disc_sche = WarmupLR(disc_opt, **self.cfg.train.disc_schedule_params)
        # print(f'Generator optim: {gen_opt}')
        # print(f'Discriminator optim: {disc_opt}')
        return opt_list, [gen_sche, disc_sche]

    def set_discriminator_gradients(self, flag=True):
        for _, module in self._iter_discriminator_modules():
            for p in module.parameters():
                p.requires_grad = flag
