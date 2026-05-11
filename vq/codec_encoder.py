import os
from contextlib import nullcontext

import torch
from torch import nn
from torch.amp import autocast
import numpy as np
from typing import Optional
from .module import WNConv1d, EncoderBlock, build_codec_activation
from .temporal_config import CodecTemporalConfig
from .alias_free_torch import *

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

class CodecEncoder(nn.Module):
    def __init__(self,
                ngf=48,
                temporal: Optional[CodecTemporalConfig] = None,
                up_ratios=(2, 2, 2, 5, 5),
                dilations=(1, 3, 9),
                out_channels=1024,
                speaker_condition=False,
                condition_dim=1024,
                snake_logscale=False,
                activation_type='SnakeBeta',
                leaky_relu_params=None,
                snake_lite_taylor_degree: int = 8,
                ):
        super().__init__()
        self.temporal = temporal or CodecTemporalConfig()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios

        # Create first convolution
        d_model = ngf
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for i, stride in enumerate(up_ratios):
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride, dilations=dilations, speaker_condition=speaker_condition, condition_dim=condition_dim, activation_type=activation_type, leaky_relu_params=leaky_relu_params, snake_lite_taylor_degree=snake_lite_taylor_degree)]
        # Temporal block (ResLSTM or single MambaBlock wrapper)
        if self.temporal.use:
            self.block += [self.temporal.build(d_model)]
        # Create last convolution
        activation = build_codec_activation(
            dim=d_model,
            activation_type=activation_type,
            leaky_relu_params=leaky_relu_params,
            speaker_condition=speaker_condition,
            condition_dim=condition_dim,
            alpha_logscale=snake_logscale,
            snake_lite_taylor_degree=snake_lite_taylor_degree,
        )

        self.block += [
            # Activation1d(activation=activations.SnakeBeta(d_model, alpha_logscale=True)),
            activation,
            WNConv1d(d_model, out_channels, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model
        
        self.reset_parameters()

    def forward(self, x, spk_cond=None):
        for i, layer in enumerate(self.block):
            if isinstance(layer, Activation1dWithCondition) or isinstance(layer, EncoderBlock):
                x = layer(x, spk_cond)
            else:
                x = layer(x)

        return x

    def inference(self, x):
        return self.block(x)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)

class VQW2VEncoder(nn.Module):
    """
    VQ-Wav2Vec encoder with two modes:
    - use_continuous=True (default): Output pre-quantization features (feature_extractor only).
    - use_continuous=False: Quantize to discrete indices, lookup pretrained codebook, output discrete features.
    Both modes output (B, C, T) with same C for compatibility with VQW2VCodecEncoderWrapper.
    """
    def __init__(
        self,
        feature_extractor,
        vector_quantizer,
        use_continuous=True,
        frozen_feature_extractor_force_fp32: bool = False,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.vector_quantizer = vector_quantizer
        self.use_continuous = use_continuous
        self.feature_extractor_frozen = not any(
            param.requires_grad for param in self.feature_extractor.parameters()
        )
        self.frozen_feature_extractor_force_fp32 = bool(
            frozen_feature_extractor_force_fp32
        )
        self.use_inference_mode_for_frozen_feature_extractor = (
            self.feature_extractor_frozen
            and os.environ.get("CODEC_VQW2V_INFERENCE_MODE", "0") == "1"
        )
        if self.use_inference_mode_for_frozen_feature_extractor:
            print("CODEC_VQW2V_INFERENCE_MODE=1: frozen VQ-Wav2Vec feature extractor runs under torch.inference_mode()")
        if (
            self.feature_extractor_frozen
            and self.frozen_feature_extractor_force_fp32
        ):
            print("Frozen VQ-Wav2Vec feature extractor runs under fp32 autocast-disabled mode")

    def forward(self, x):
        # x: (B, T) waveform
        use_fp32_island = (
            x.is_cuda
            and self.feature_extractor_frozen
            and self.frozen_feature_extractor_force_fp32
        )
        autocast_ctx = (
            autocast(device_type=x.device.type, enabled=False)
            if use_fp32_island
            else nullcontext()
        )
        if self.use_inference_mode_for_frozen_feature_extractor:
            with autocast_ctx:
                with torch.inference_mode():
                    features = self.feature_extractor(x)  # (B, C, T)
            # Inference tensors cannot be saved for backward by downstream trainable layers.
            features = features.clone()
        else:
            with autocast_ctx:
                features = self.feature_extractor(x)  # (B, C, T)
        if self.use_continuous:
            return features
        # Discrete: quantize to idx, lookup pretrained codebook, return discrete features
        with torch.no_grad():
            zq, _ = self.vector_quantizer.forward_idx(features)
        return zq
