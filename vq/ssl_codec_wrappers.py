"""
Shared SSL (Self-Supervised Learning) codec encoder wrappers for BiCodec, TriCodec, TriXCodec.

Provides:
- W2V2HiddenStateEncoder, W2VBert2HiddenStateEncoder: HuggingFace SSL encoder adapters
- VQW2VCodecEncoderWrapper, W2V2CodecEncoderWrapper, HubertCodecEncoderWrapper, W2VBert2CodecEncoderWrapper
- S3TokenizerEncoder, S3TokenizerCodecEncoderWrapper: S3Tokenizer (CosyVoice) encoder adapter
"""

from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.amp import autocast
from typing import Optional

from .module import WNConv1d, EncoderBlock, build_codec_activation
from .temporal_config import CodecTemporalConfig


class VQW2VCodecEncoderWrapper(nn.Module):
    """Wrapper for VQ-Wav2Vec encoder output with RNN + downsampling blocks."""

    def __init__(
        self,
        encoder,
        activation_type="SnakeBeta",
        leaky_relu_params=None,
        temporal: Optional[CodecTemporalConfig] = None,
        up_ratios=(2, 2, 2, 5, 5),
        dilations=(1, 3, 9),
        ngf=32,
        out_channels=1024,
        encoder_force_fp32: bool = False,
        snake_lite_taylor_degree: int = 8,
    ):
        super().__init__()
        self.encoder = encoder
        self.temporal = temporal or CodecTemporalConfig()
        self.encoder_force_fp32 = bool(encoder_force_fp32)
        self.encoder_is_frozen = not any(
            param.requires_grad for param in self.encoder.parameters()
        )
        self.block = []
        if self.temporal.use:
            self.block += [self.temporal.build(ngf)]
        for i, stride in enumerate(up_ratios):
            ngf *= 2
            self.block += [
                EncoderBlock(
                    ngf,
                    stride=stride,
                    dilations=dilations,
                    activation_type=activation_type,
                    leaky_relu_params=leaky_relu_params,
                    snake_lite_taylor_degree=snake_lite_taylor_degree,
                )
            ]
        activation = build_codec_activation(
            dim=ngf,
            activation_type=activation_type,
            leaky_relu_params=leaky_relu_params,
            alpha_logscale=True,
            snake_lite_taylor_degree=snake_lite_taylor_degree,
            no_condition=True,
        )

        self.block += [
            activation,
            WNConv1d(ngf, out_channels, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*self.block)

    def forward(self, x, spk_cond=None):
        use_fp32_island = (
            x.is_cuda
            and self.encoder_is_frozen
            and self.encoder_force_fp32
        )
        autocast_ctx = (
            autocast(device_type=x.device.type, enabled=False)
            if use_fp32_island
            else nullcontext()
        )
        with autocast_ctx:
            x = self.encoder(x)
        x = self.block(x)
        return x


class W2V2CodecEncoderWrapper(nn.Module):
    """
    Wrapper for HuggingFace Wav2Vec2 feature extractor outputs.

    Expected usage:
      - `encoder` is a module that takes (B, T) float waveform and returns (B, C, T')
      - When up_ratios is not None: applies RNN + downsampling EncoderBlocks.
      - When up_ratios is None: no downsampling; optional RNN + projection to out_channels.
    """

    def __init__(
        self,
        encoder,
        activation_type="SnakeBeta",
        leaky_relu_params=None,
        temporal: Optional[CodecTemporalConfig] = None,
        up_ratios=(2, 2, 2, 5, 5),
        dilations=(1, 3, 9),
        ngf=512,
        out_channels=1024,
        encoder_force_fp32: bool = False,
        snake_lite_taylor_degree: int = 8,
    ):
        super().__init__()
        self.encoder = encoder
        self.up_ratios = up_ratios
        self.temporal = temporal or CodecTemporalConfig()
        self.encoder_force_fp32 = bool(encoder_force_fp32)
        self.encoder_is_frozen = not any(
            param.requires_grad for param in self.encoder.parameters()
        )

        def _make_activation(ch: int):
            return build_codec_activation(
                dim=ch,
                activation_type=activation_type,
                leaky_relu_params=leaky_relu_params,
                alpha_logscale=True,
                snake_lite_taylor_degree=snake_lite_taylor_degree,
                no_condition=True,
            )

        layers = []
        if self.temporal.use:
            layers += [self.temporal.build(ngf)]

        if up_ratios is not None and len(up_ratios) > 0:
            for stride in up_ratios:
                ngf *= 2
                print(f"EncoderBlock ngf size: {ngf}")
                layers += [
                    EncoderBlock(
                        ngf,
                        stride=stride,
                        dilations=dilations,
                        activation_type=activation_type,
                        leaky_relu_params=leaky_relu_params,
                        snake_lite_taylor_degree=snake_lite_taylor_degree,
                    )
                ]

        layers += [
            _make_activation(ngf),
            WNConv1d(ngf, out_channels, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x, spk_cond=None):
        use_fp32_island = (
            x.is_cuda
            and self.encoder_is_frozen
            and self.encoder_force_fp32
        )
        autocast_ctx = (
            autocast(device_type=x.device.type, enabled=False)
            if use_fp32_island
            else nullcontext()
        )
        with autocast_ctx:
            x = self.encoder(x)
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = self.block(x)
        return x


class HubertCodecEncoderWrapper(W2V2CodecEncoderWrapper):
    """
    HuBERT encoder wrapper. Same interface as W2V2CodecEncoderWrapper.
    HuBERT and Wav2Vec2 share the same conv+transformer output format.
    """
    pass


class W2VBert2CodecEncoderWrapper(W2V2CodecEncoderWrapper):
    """
    Wav2Vec2-BERT 2.0 (facebook/w2v-bert-2.0) encoder wrapper.
    Same interface as W2V2CodecEncoderWrapper; upstream handles mel preprocessing.
    """
    pass


def _load_s3tokenizer():
    """Lazy import of s3tokenizer to avoid hard dependency at import time."""
    try:
        import s3tokenizer
        return s3tokenizer
    except ImportError:
        raise ImportError(
            "s3tokenizer is required for S3TokenizerCodecEncoderWrapper. "
            "Install with: pip install s3tokenizer"
        )


class S3TokenizerEncoder(nn.Module):
    """
    S3Tokenizer (CosyVoice) encoder adapter that outputs continuous features.

    - Input:  (B, T) or (B, 1, T) waveform at 16kHz
    - Output: (B, n_audio_state, T') where n_audio_state=1280

    Uses S3Tokenizer's encoder (pre-quantization) for gradient flow during training.
    Reference: https://github.com/xingchensong/S3Tokenizer
    """

    def __init__(
        self,
        model_name: str = "speech_tokenizer_v1",
        sampling_rate: int = 16000,
        n_mels: int = 128,
        use_continuous: bool = True,
    ):
        super().__init__()
        s3 = _load_s3tokenizer()
        self.s3_model = s3.load_model(model_name)
        self.sampling_rate = int(sampling_rate)
        self.n_mels = int(n_mels)
        self.use_continuous = use_continuous
        # S3 encoder output dim
        self.out_dim = getattr(
            self.s3_model.config, "n_audio_state", 1280
        )

    def _compute_mel(self, wav: torch.Tensor, wav_lens: torch.Tensor):
        """Compute batched log-mel spectrogram matching S3Tokenizer format."""
        s3 = _load_s3tokenizer()
        B = wav.size(0)
        mels = []
        mel_lens = []
        for i in range(B):
            audio = wav[i]
            if audio.dim() > 1:
                audio = audio.squeeze(0)
            mel = s3.log_mel_spectrogram(audio, n_mels=self.n_mels)
            # mel: (n_mels, T)
            mels.append(mel)
            mel_lens.append(mel.size(1))
        mels_padded, mel_lens_t = s3.padding(mels)
        # mels_padded: (B, n_mels, T_max), mel_lens_t: (B,)
        return mels_padded, mel_lens_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)  # (B, 1, T) -> (B, T)
        B, T = x.shape
        wav_lens = torch.full((B,), T, device=x.device, dtype=torch.long)
        mel, mel_lens = self._compute_mel(x, wav_lens)
        # S3 encoder returns (hidden, x_len), hidden: (B, T', n_audio_state)
        hidden, _ = self.s3_model.encoder(mel, mel_lens)
        # (B, T', C) -> (B, C, T')
        out = hidden.transpose(1, 2).contiguous()
        return out


class S3TokenizerCodecEncoderWrapper(W2V2CodecEncoderWrapper):
    """
    S3Tokenizer (CosyVoice) encoder wrapper for codec training.

    Uses S3Tokenizer's encoder output (continuous, pre-quantization) and applies
    RNN + EncoderBlocks + projection, same interface as W2V2CodecEncoderWrapper.

    - Input:  (B, T) or (B, 1, T) waveform at 16kHz
    - Output: (B, out_channels, T')

    Config: codec_encoder.use_s3tokenizer=True, s3tokenizer_model_name, etc.
    """

    def __init__(
        self,
        encoder,
        activation_type="SnakeBeta",
        leaky_relu_params=None,
        temporal: Optional[CodecTemporalConfig] = None,
        up_ratios=(2, 2, 2, 5, 5),
        dilations=(1, 3, 9),
        ngf=1280,
        out_channels=1024,
        encoder_force_fp32: bool = False,
        snake_lite_taylor_degree: int = 8,
    ):
        super().__init__(
            encoder=encoder,
            activation_type=activation_type,
            leaky_relu_params=leaky_relu_params,
            temporal=temporal,
            up_ratios=up_ratios,
            dilations=dilations,
            ngf=ngf,
            out_channels=out_channels,
            encoder_force_fp32=encoder_force_fp32,
            snake_lite_taylor_degree=snake_lite_taylor_degree,
        )


class W2V2HiddenStateEncoder(nn.Module):
    """
    HuggingFace Wav2Vec2/HuBERT wrapper that returns a channels-first feature map.

    - Input:  (B, T) or (B, 1, T) waveform
    - Output: (B, H, F) where F is frame length after conv feature extractor

    Compatible with Wav2Vec2Model, HubertModel (both use raw waveform and output hidden_states).
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices=(11, 14, 16),
        mode: str = "avg",
    ):
        super().__init__()
        self.model = model
        self.layer_indices = tuple(int(i) for i in layer_indices)
        self.mode = str(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)  # (B, 1, T) -> (B, T)
        out = self.model(x, output_hidden_states=True)
        hs = out.hidden_states
        if hs is None:
            raise RuntimeError(
                "Model did not return hidden_states. Ensure output_hidden_states=True."
            )
        if self.mode == "avg":
            feat = sum(hs[i] for i in self.layer_indices) / float(
                len(self.layer_indices)
            )
        elif self.mode == "last":
            feat = out.last_hidden_state
        else:
            raise ValueError(f"Unsupported hidden-state mode: {self.mode}")
        # (B, F, H) -> (B, H, F)
        return feat.transpose(1, 2).contiguous()


class W2VBert2HiddenStateEncoder(nn.Module):
    """
    HuggingFace Wav2Vec2-BERT 2.0 (facebook/w2v-bert-2.0) wrapper.

    Wav2Vec2-BERT uses mel-spectrogram input via processor, not raw waveform.
    This wrapper handles processor + model and returns channels-first (B, H, F).

    - Input:  (B, T) or (B, 1, T) waveform at 16kHz
    - Output: (B, H, F) where H=hidden_size (1024), F=frame length

    Reference: https://huggingface.co/facebook/w2v-bert-2.0
    """

    def __init__(
        self,
        model: nn.Module,
        processor,
        layer_indices=(11, 18, 23),
        mode: str = "avg",
        sampling_rate: int = 16000,
        normalize: bool = True,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.layer_indices = tuple(int(i) for i in layer_indices)
        self.mode = str(mode)
        self.sampling_rate = int(sampling_rate)
        self.normalize = bool(normalize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)  # (B, 1, T) -> (B, T)
        # Wav2Vec2-BERT expects processor output (input_features or input_values)
        arrs = [x[i].cpu().float().numpy() for i in range(x.size(0))]
        inputs = self.processor(
            arrs,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(x.device) for k, v in inputs.items()}
        out = self.model(**inputs, output_hidden_states=True)

        hs = out.hidden_states
        if hs is None:
            raise RuntimeError(
                "Wav2Vec2-BERT did not return hidden_states. Ensure output_hidden_states=True."
            )
        if self.mode == "avg":
            feat = sum(hs[i] for i in self.layer_indices) / float(
                len(self.layer_indices)
            )
        elif self.mode == "last":
            feat = out.last_hidden_state
        else:
            raise ValueError(f"Unsupported hidden-state mode: {self.mode}")
        if self.normalize:
            mean = feat.mean(dim=1, keepdim=True)
            std = feat.std(dim=1, keepdim=True).clamp_min(1e-5)
            feat = (feat - mean) / std
        return feat.transpose(1, 2).contiguous()  # (B, F, H) -> (B, H, F)


__all__ = [
    "W2V2HiddenStateEncoder",
    "W2VBert2HiddenStateEncoder",
    "VQW2VCodecEncoderWrapper",
    "W2V2CodecEncoderWrapper",
    "HubertCodecEncoderWrapper",
    "W2VBert2CodecEncoderWrapper",
    "S3TokenizerEncoder",
    "S3TokenizerCodecEncoderWrapper",
]
