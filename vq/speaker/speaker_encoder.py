# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

from typing import Any, Dict, List, Tuple, Optional
from vq.vq.residual_fsq import ResidualFSQ
from vq.vq.simvq import SimVQ1D
from vq.speaker.ecapa_tdnn import ECAPA_TDNN_GLOB_c512, ECAPA_TDNN_SPEECHBRAIN
from vq.speaker.perceiver_encoder import (
    MemoryCrossAttentionEncoder,
    PerceiverResampler,
    SpeakerTokenMixer,
)

from termcolor import colored
import sys

from vq.speaker.wavlm.WavLM import WavLM, WavLMConfig


"""
x-vector + d-vector
"""


class ConvPromptPrenet(nn.Module):
    """
    Lightweight adaptation of vec2wav2.0 ConvPromptPrenet to compress WavLM prompts.

    Args:
        in_channels (int): input feature dimension.
        out_channels (int): output feature dimension.
        conv_layers (List[Tuple[int, int, int, int]]): sequence of (dim, kernel, stride, padding).
        dropout (float): dropout probability applied after each conv.
        skip_connections (bool): whether to use residual connections.
        residual_scale (float): scaling factor for residual paths.
        activation (Callable): activation factory (default nn.ReLU).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_layers: Optional[List[Tuple[int, int, int, int]]] = None,
        dropout: float = 0.1,
        skip_connections: bool = True,
        residual_scale: float = 0.25,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        if conv_layers is None or len(conv_layers) == 0:
            hidden_dim = max(out_channels, in_channels // 2)
            conv_layers = [
                (hidden_dim, 3, 1, 1),
                (out_channels, 3, 1, 1),
            ]

        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale) if skip_connections else 1.0
        act_factory = activation if activation is not None else nn.ReLU

        layers: List[nn.Module] = []
        residual_proj: List[Optional[nn.Module]] = []
        in_dim = int(in_channels)

        for dim, kernel, stride, padding in conv_layers:
            dim = int(dim)
            block = nn.Sequential(
                nn.Conv1d(in_dim, dim, kernel_size=kernel, stride=stride, padding=padding, bias=True),
                nn.Dropout(p=dropout),
                nn.GroupNorm(1, dim),
                act_factory(),
            )
            layers.append(block)

            if skip_connections and dim != in_dim:
                residual_proj.append(nn.Conv1d(in_dim, dim, kernel_size=1, bias=False))
            else:
                residual_proj.append(None)

            in_dim = dim

        if in_dim != out_channels:
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_dim, out_channels, kernel_size=1, bias=True),
                    nn.Dropout(p=dropout),
                    nn.GroupNorm(1, out_channels),
                    act_factory(),
                )
            )
            residual_proj.append(nn.Conv1d(in_dim, out_channels, kernel_size=1, bias=False) if skip_connections else None)

        self.conv_layers = nn.ModuleList(layers)
        self.residual_proj = nn.ModuleList(residual_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for proj, conv in zip(self.residual_proj, self.conv_layers):
            residual = x
            x = conv(x)
            if self.skip_connections:
                if proj is not None:
                    residual = proj(residual)
                x = (x + residual) * self.residual_scale
        return x


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_speaker_quantizer_kwargs(spkcfg: Any) -> Dict[str, Any]:
    quantizer_cfg = _cfg_get(spkcfg, 'quantizer')
    fsq_cfg = _cfg_get(quantizer_cfg, 'fsq')
    simvq_cfg = _cfg_get(quantizer_cfg, 'simvq')
    vq_type = _cfg_get(quantizer_cfg, 'type', _cfg_get(spkcfg, 'vq_type', 'fsq'))

    fsq_levels = _cfg_get(fsq_cfg, 'levels', _cfg_get(spkcfg, 'fsq_levels', [4, 4, 4, 4, 4, 4]))
    fsq_num_quantizers = _cfg_get(fsq_cfg, 'num_quantizers', _cfg_get(spkcfg, 'fsq_num_quantizers', 1))
    simvq_codebook_size = _cfg_get(simvq_cfg, 'codebook_size', _cfg_get(spkcfg, 'simvq_codebook_size'))
    if vq_type == 'simvq' and simvq_codebook_size is None:
        raise ValueError("speaker_encoder.quantizer.simvq.codebook_size must be set when quantizer.type=simvq")

    return {
        'vq_type': vq_type,
        'fsq_levels': fsq_levels,
        'fsq_num_quantizers': fsq_num_quantizers,
        'simvq_codebook_size': simvq_codebook_size,
        'simvq_commitment': _cfg_get(simvq_cfg, 'commitment', 0.25),
    }


def _stack_cfg_enabled(stack_cfg: Any) -> bool:
    return bool(stack_cfg is not None and _cfg_get(stack_cfg, 'use', False))


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {'', 'null', 'none'}:
            return None
        return int(stripped)
    return int(value)


def _normalize_stack_stage_name(raw_stage: Any) -> str:
    if raw_stage is None:
        raise ValueError("speaker_encoder.stack stage cannot be None")

    text = str(raw_stage).strip().lower().replace('-', '_')
    if any(char.isdigit() for char in text):
        raise ValueError(
            f"speaker_encoder.stack stage '{raw_stage}' should not include token counts. "
            "Use stack.layers=[mca,pe,tm,pe] and stack.token_nums=[64,64,null,16]."
        )

    raw_name = text.replace('_', '')
    stage_name = {
        'mca': 'mca',
        'memory': 'mca',
        'memorycattn': 'mca',
        'memorycrossattention': 'mca',
        'memorycrossattn': 'mca',
        'pe': 'pe',
        'perceiver': 'pe',
        'perceiverresampler': 'pe',
        'resampler': 'pe',
        'tm': 'tm',
        'mixer': 'tm',
        'tokenmixer': 'tm',
        'selfattn': 'tm',
        'selfattention': 'tm',
    }.get(raw_name)
    if stage_name is None:
        raise ValueError(
            f"Unsupported speaker_encoder.stack stage '{raw_stage}'. "
            "Expected one of mca / pe / tm."
        )

    return stage_name


def _parse_speaker_stack(stack_cfg: Any) -> List[Dict[str, Any]]:
    if not _stack_cfg_enabled(stack_cfg):
        return []

    raw_layers = (
        _cfg_get(stack_cfg, 'layers')
        or _cfg_get(stack_cfg, 'types')
        or _cfg_get(stack_cfg, 'stages')
        or _cfg_get(stack_cfg, 'modules')
        or _cfg_get(stack_cfg, 'specs')
    )
    if raw_layers is None or len(raw_layers) == 0:
        raise ValueError(
            "speaker_encoder.stack.use=True requires non-empty stack.layers "
            "(or stack.types / stack.stages / stack.modules / stack.specs)."
        )

    raw_token_nums = _cfg_get(stack_cfg, 'token_nums', _cfg_get(stack_cfg, 'tokens'))
    raw_depths = _cfg_get(stack_cfg, 'depths')
    raw_latent_dims = _cfg_get(stack_cfg, 'latent_dims', _cfg_get(stack_cfg, 'dims'))
    if raw_token_nums is None:
        raise ValueError(
            "speaker_encoder.stack.use=True requires stack.token_nums with the same length "
            "as stack.layers. Use null for tm stages."
        )
    if len(raw_token_nums) != len(raw_layers):
        raise ValueError(
            "speaker_encoder.stack.token_nums must have the same length as stack.layers"
        )
    if raw_depths is not None and len(raw_depths) != len(raw_layers):
        raise ValueError(
            "speaker_encoder.stack.depths must have the same length as stack.layers"
        )
    if raw_latent_dims is not None and len(raw_latent_dims) != len(raw_layers):
        raise ValueError(
            "speaker_encoder.stack.latent_dims must have the same length as stack.layers"
        )

    default_depth = int(_cfg_get(stack_cfg, 'depth', 2))
    num_heads = int(_cfg_get(stack_cfg, 'num_heads', 8))
    dim_head = int(_cfg_get(stack_cfg, 'dim_head', 64))
    ff_mult = int(_cfg_get(stack_cfg, 'ff_mult', 4))
    dropout = float(_cfg_get(stack_cfg, 'dropout', 0.0))
    use_flash_attn = bool(_cfg_get(stack_cfg, 'use_flash_attn', False))
    tm_type = str(_cfg_get(stack_cfg, 'tm_type', 'self_attn')).strip().lower()
    if tm_type not in {'self_attn', 'selfattention', 'attn'}:
        raise ValueError(
            f"Unsupported speaker_encoder.stack.tm_type: {tm_type}. "
            "Only 'self_attn' is supported for now."
        )

    stages: List[Dict[str, Any]] = []
    for idx, raw_stage in enumerate(raw_layers):
        stage_type = _normalize_stack_stage_name(raw_stage)
        token_num = _coerce_optional_int(raw_token_nums[idx])

        if stage_type in {'mca', 'pe'}:
            if token_num is None:
                raise ValueError(
                    f"speaker_encoder.stack stage '{raw_stage}' requires a numeric token_num."
                )
            token_num = int(token_num)
        elif token_num is not None:
            raise ValueError("speaker_encoder.stack tm stage must use null token_num")

        depth = default_depth if raw_depths is None else int(raw_depths[idx])
        if depth < 1:
            raise ValueError("speaker_encoder.stack stage depth must be >= 1")
        latent_dim = None if raw_latent_dims is None else _coerce_optional_int(raw_latent_dims[idx])
        if latent_dim is not None and latent_dim < 1:
            raise ValueError("speaker_encoder.stack stage latent_dim must be >= 1")

        stages.append(
            {
                'type': stage_type,
                'token_num': token_num,
                'depth': depth,
                'latent_dim': latent_dim,
                'num_heads': num_heads,
                'dim_head': dim_head,
                'ff_mult': ff_mult,
                'dropout': dropout,
                'use_flash_attn': use_flash_attn,
            }
        )

    return stages


def _infer_final_token_num_from_stack(stages: List[Dict[str, Any]]) -> Optional[int]:
    fixed_token_num: Optional[int] = None
    for stage in stages:
        if stage['type'] == 'pe':
            fixed_token_num = int(stage['token_num'])
    return fixed_token_num


def _infer_final_latent_dim_from_stack(stages: List[Dict[str, Any]]) -> Optional[int]:
    final_latent_dim: Optional[int] = None
    for stage in stages:
        stage_latent_dim = stage.get('latent_dim')
        if stage_latent_dim is not None:
            final_latent_dim = int(stage_latent_dim)
    return final_latent_dim


def resolve_speaker_encoder_token_dim(spkcfg: Any) -> int:
    stack_cfg = _cfg_get(spkcfg, 'stack')
    base_latent_dim = int(_cfg_get(spkcfg, 'latent_dim'))
    stages = _parse_speaker_stack(stack_cfg)
    if len(stages) == 0:
        return base_latent_dim

    final_latent_dim = _infer_final_latent_dim_from_stack(stages)
    if final_latent_dim is None:
        return base_latent_dim
    return int(final_latent_dim)


class SpeakerEncoder(nn.Module):
    """

    Args:
        input_dim (int): acoustic feature dimension
        out_dim (int): output dimension of x-vector and d-vector
        latent_dim (int): latent dimension before quantization
        token_num (int): sequence length of speaker tokens
        fsq_levels (List[int]): number of levels for each quantizer
        fsq_num_quantizers (int): number of quantizers

    Return:
        speaker_embs: (B, T2, out_dim)
    """

    def __init__(
        self,
        mel_params: Optional[Any] = None,
        speaker_encoder_type: str = 'ecapa_tdnn',
        use_perceiver_encoder: bool = True,
        use_memory_cattn: bool = False,
        input_dim: int = 100,
        out_dim: int = 512,
        latent_dim: int = 128,
        token_num: int = 32,
        discretize_memory_attn: bool = False,
        memory_attn_codebook_size: int = 128,
        memory_attn_share_across_heads: bool = True,
        fsq_levels: List[int] = [4, 4, 4, 4, 4, 4],
        fsq_num_quantizers: int = 1,
        norm_layer: str = 'bn',  # 'bn', 'in', 'ln'
        vq_type: str = 'fsq',
        simvq_codebook_size: Optional[int] = None,
        simvq_commitment: float = 0.25,
        use_normalized_f0: bool = False,
        use_quantizer: bool = True,
        stack: Optional[Any] = None,
        # WavLM specific (optional)
        wavlm_checkpoint: str = None,
        wavlm_output_layer: int = 6,
        freeze_wavlm: bool = True,
        perceiver_use_flash_attn: bool = False,
        frozen_wavlm_inference_mode: bool = False,
        frozen_wavlm_force_fp32: bool = False,
    ):
        super(SpeakerEncoder, self).__init__()
        self.vq_type = vq_type
        self.speaker_encoder_type = speaker_encoder_type
        self.token_stack = nn.ModuleList()
        self.stack_stages = _parse_speaker_stack(stack)
        self.use_stack = len(self.stack_stages) > 0
        self.use_perceiver_encoder = bool(use_perceiver_encoder or self.use_stack)
        self.use_memory_cattn = use_memory_cattn
        self.perceiver_use_flash_attn = bool(perceiver_use_flash_attn)
        self.use_normalized_f0 = use_normalized_f0
        self.use_quantizer = use_quantizer
        self.out_dim = out_dim
        self.base_latent_dim = int(latent_dim)
        self.final_latent_dim = int(latent_dim)

        print(colored(f'Importing {speaker_encoder_type} speaker encoder', 'red', attrs=['bold']))
        dim_context = None

        if speaker_encoder_type == 'ecapa_tdnn':
            self.speaker_encoder = ECAPA_TDNN_GLOB_c512(
                feat_dim=input_dim, embed_dim=out_dim, norm_layer=norm_layer,
            )
            if mel_params is None:
                raise ValueError("mel_params must be provided when speaker_encoder_type='ecapa_tdnn'.")
            self.init_mel_transformer(mel_params)
            dim_context = 512 * 3
        elif speaker_encoder_type == 'ecapa_tdnn_speechbrain':
            self.speaker_encoder = ECAPA_TDNN_SPEECHBRAIN(out_dim=out_dim)
            for param in self.speaker_encoder.parameters():
                param.requires_grad = True
            self.speaker_encoder.train()
            dim_context = 512 * 6
        elif speaker_encoder_type == 'wavlm':
            if WavLM is None or WavLMConfig is None:
                raise ImportError("Failed to import WavLM. Ensure vec2wav2.0 is on PYTHONPATH or adjust sys.path.")
            # Resolve checkpoint
            ckpt_path = wavlm_checkpoint or "pretrained/WavLM-Large.pt"
            print(colored(f"Loading WavLM from {ckpt_path}", "yellow"))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self.wavlm_cfg = WavLMConfig(ckpt['cfg'])
            self.wavlm_model = WavLM(self.wavlm_cfg)
            self.wavlm_model.load_state_dict(ckpt['model'])
            # freezing policy
            if freeze_wavlm:
                for p in self.wavlm_model.parameters():
                    p.requires_grad = False
                self.wavlm_model.eval()
            self.freeze_wavlm = bool(freeze_wavlm)
            self.frozen_wavlm_inference_mode = bool(frozen_wavlm_inference_mode)
            self.frozen_wavlm_force_fp32 = bool(frozen_wavlm_force_fp32)
            self.use_inference_mode_for_frozen_wavlm = (
                self.freeze_wavlm
                and (
                    self.frozen_wavlm_inference_mode
                    or os.environ.get("SPEAKER_WAVLM_INFERENCE_MODE", "0") == "1"
                )
            )
            self.wavlm_output_layer = int(wavlm_output_layer)
            # Use normalize flag from checkpoint config (not external param)
            self.wavlm_normalize = bool(self.wavlm_cfg.normalize)
            self.wavlm_feature_dim = int(self.wavlm_cfg.encoder_embed_dim)
            print(colored(f"WavLM normalize from checkpoint cfg: {self.wavlm_normalize}", "yellow"))
            if self.use_inference_mode_for_frozen_wavlm:
                print(colored("Frozen WavLM runs under torch.inference_mode()", "yellow"))
            if self.freeze_wavlm and self.frozen_wavlm_force_fp32:
                print(colored("Frozen WavLM runs under fp32 autocast-disabled mode", "yellow"))
            
            # Apply torch.compile for faster inference (PyTorch 2.0+)
            # self.wavlm_model = torch.compile(self.wavlm_model, mode='reduce-overhead')
            # print(colored("✓ torch.compile applied to WavLM model", "green"))
            
            # x-vector projection head (global pooled WavLM -> out_dim)
            self.x_project = nn.Linear(self.wavlm_feature_dim, out_dim)
            dim_context = self.wavlm_feature_dim
        else:
            raise ValueError(f"Unknown speaker_encoder_type: {speaker_encoder_type}")
        
        self.perceiver_sampler = None
        self.memory_encoder = None
        self.quantizer = None
        self.project = None
        self.prompt_prenet = None
        self.final_token_num = None

        if use_perceiver_encoder or self.use_stack:
            print(colored(f"Using Perceiver encoder in Speaker Encoder", "yellow", attrs=['bold']))
            if self.use_stack:
                stage_labels = []
                current_dim = dim_context
                for stage in self.stack_stages:
                    stage_type = stage['type']
                    stage_token_num = stage['token_num']
                    stage_latent_dim = int(stage['latent_dim'] if stage['latent_dim'] is not None else self.base_latent_dim)
                    common_kwargs = dict(
                        dim=stage_latent_dim,
                        depth=stage['depth'],
                        dim_head=stage['dim_head'],
                        heads=stage['num_heads'],
                        ff_mult=stage['ff_mult'],
                        use_flash_attn=stage['use_flash_attn'],
                    )

                    if stage_type == 'mca':
                        module = MemoryCrossAttentionEncoder(
                            dim_context=current_dim,
                            num_latents=stage_token_num,
                            discretize_attn=discretize_memory_attn,
                            attn_codebook_size=memory_attn_codebook_size,
                            attn_share_across_heads=memory_attn_share_across_heads,
                            **common_kwargs,
                        )
                        stage_labels.append(f"mca(kv={stage_token_num},d={stage_latent_dim})")
                        current_dim = stage_latent_dim
                    elif stage_type == 'pe':
                        module = PerceiverResampler(
                            dim_context=current_dim,
                            num_latents=stage_token_num,
                            **common_kwargs,
                        )
                        stage_labels.append(f"pe(n={stage_token_num},d={stage_latent_dim})")
                        current_dim = stage_latent_dim
                    elif stage_type == 'tm':
                        mixer = SpeakerTokenMixer(
                            dim=stage_latent_dim,
                            depth=stage['depth'],
                            dim_head=stage['dim_head'],
                            heads=stage['num_heads'],
                            ff_mult=stage['ff_mult'],
                            dropout=stage['dropout'],
                            use_flash_attn=stage['use_flash_attn'],
                        )
                        module = nn.Sequential(nn.Linear(current_dim, stage_latent_dim), mixer) if current_dim != stage_latent_dim else mixer
                        stage_labels.append(f"tm(d={stage_latent_dim})")
                        current_dim = stage_latent_dim
                    else:
                        raise ValueError(f"Unsupported speaker stack stage type: {stage_type}")

                    self.token_stack.append(module)

                self.final_token_num = _infer_final_token_num_from_stack(self.stack_stages)
                self.final_latent_dim = int(current_dim)
                if self.final_token_num is None:
                    raise ValueError(
                        "speaker_encoder.stack must contain at least one PE stage so the "
                        "final speaker token count is fixed for projection."
                    )
                print(
                    colored(
                        f"Speaker encoder stack enabled: {' -> '.join(stage_labels)} "
                        f"(final_token_num={self.final_token_num})",
                        "yellow",
                        attrs=['bold'],
                    )
                )
            else:
                if self.use_memory_cattn:
                    print(colored("Speaker encoder memory cross-attention is enabled", "yellow", attrs=['bold']))
                    self.memory_encoder = MemoryCrossAttentionEncoder(
                        dim=self.base_latent_dim,
                        dim_context=dim_context,
                        num_latents=token_num,
                        use_flash_attn=self.perceiver_use_flash_attn,
                        discretize_attn=discretize_memory_attn,
                        attn_codebook_size=memory_attn_codebook_size,
                        attn_share_across_heads=memory_attn_share_across_heads,
                    )
                self.perceiver_sampler = PerceiverResampler(
                    dim=self.base_latent_dim,
                    dim_context=self.base_latent_dim if self.use_memory_cattn else dim_context,
                    num_latents=token_num,
                    use_flash_attn=self.perceiver_use_flash_attn,
                )
                self.final_token_num = int(token_num)
                self.final_latent_dim = self.base_latent_dim

            if self.use_quantizer:
                print(colored(f'Importing {vq_type} quantizer', 'red', attrs=['bold']))
                print(f'Importing {vq_type} quantizer')
                if vq_type == 'fsq':
                    self.quantizer = ResidualFSQ(
                        levels=fsq_levels,
                        num_quantizers=fsq_num_quantizers,
                        dim=self.final_latent_dim,
                        is_channel_first=True,
                        quantize_dropout=False,
                    )
                elif vq_type == 'simvq':
                    self.quantizer = SimVQ1D(
                        codebook_size=simvq_codebook_size,
                        codebook_dim=self.final_latent_dim,
                        commitment=simvq_commitment,
                        sane_index_shape=True,
                    )
            
            self.project = nn.Linear(self.final_latent_dim * int(self.final_token_num), out_dim)
        else:
            if dim_context is None:
                raise ValueError("dim_context must be defined when use_perceiver_encoder=False")
            print(colored("Perceiver disabled: using Conv1d prompt prenet instead of quantization", "yellow", attrs=['bold']))
            mid_dim = max(self.base_latent_dim, dim_context // 2)
            conv_cfg = [
                (dim_context, 3, 1, 1),
                (mid_dim, 5, 1, 2),
                (self.base_latent_dim, 3, 1, 1),
            ]
            self.prompt_prenet = ConvPromptPrenet(
                in_channels=dim_context,
                out_channels=self.base_latent_dim,
                conv_layers=conv_cfg,
                dropout=0.1,
                skip_connections=True,
                residual_scale=0.25,
                activation=nn.ReLU,
            )
            self.final_latent_dim = self.base_latent_dim

    def init_mel_transformer(self, cfg):
        """
        Initializes the MelSpectrogram transformer based on the provided configuration.

        Args:
            config (dict): Configuration parameters for MelSpectrogram.
        """
        import torchaudio.transforms as TT
        self.mel_transformer = TT.MelSpectrogram(
            cfg.sample_rate,
            cfg.n_fft,
            cfg.win_length,
            cfg.hop_length,
            cfg.mel_fmin,
            cfg.mel_fmax,
            n_mels=cfg.num_mels,
            power=1,
            norm="slaney",
            mel_scale="slaney",
        )

    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if not self.use_perceiver_encoder or self.quantizer is None:
            raise RuntimeError("Quantization is disabled when use_perceiver_encoder=False.")
        if self.vq_type == 'simvq':
            return self.quantizer.get_output_from_indices(indices)
        zq = self.quantizer.get_codes_from_indices(indices.transpose(1, 2))
        return zq.transpose(1, 2)

    def get_indices(self, mels: torch.Tensor) -> torch.Tensor:
        if not self.use_perceiver_encoder or self.perceiver_sampler is None or self.quantizer is None:
            raise RuntimeError("Quantization is disabled when use_perceiver_encoder=False.")
        mels = mels.transpose(1, 2)
        x = self._encode_speaker_tokens(mels).transpose(1, 2)
        out = self._normalize_quantizer_output(self.quantizer(x), batch_size=x.shape[0], token_count=x.shape[-1])
        return out['indices']

    def _encode_speaker_tokens(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T_ctx, D_ctx) backbone speaker features

        Returns:
            Speaker tokens with fixed token length after optional memory cross-attn
            and Perceiver resampling: (B, token_num, latent_dim)
        """
        x = features
        if self.use_stack:
            if len(self.token_stack) == 0:
                raise RuntimeError("speaker_encoder.stack is enabled but no token stack modules were built.")
            for layer in self.token_stack:
                x = layer(x)
            return x
        if self.use_memory_cattn:
            assert self.memory_encoder is not None, "memory_encoder must be defined when use_memory_cattn=True."
            x = self.memory_encoder(x)  # (B, T_ctx, latent_dim)
        assert self.perceiver_sampler is not None, "perceiver_sampler must be defined when use_perceiver_encoder=True."
        return self.perceiver_sampler(x)  # (B, token_num, latent_dim)

    def forward(self, ref_wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            ref_wav: (B, T1)

        Return:
            x_vector: (B, out_dim) continuous speaker embedding from the backbone
            d_vector: (B, out_dim) projected embedding after Perceiver / memory cross-attn (+ optional VQ)
            x_quantized: (B, latent_dim, T_spk) speaker token sequence before global pooling
        """
        # mels = mels.transpose(1,2)

        if self.speaker_encoder_type == 'ecapa_tdnn':
            mel = self.mel_transformer(ref_wav).squeeze(1)  # [B, n_mel, T]
            x_vector, features = self.speaker_encoder(mel.transpose(1, 2), True)
            # x_vector: (B, out_dim), features: (B, C_ctx, T_ctx)
        elif self.speaker_encoder_type == 'ecapa_tdnn_speechbrain':
            x_vector, features = self.speaker_encoder(ref_wav)
            # x_vector: (B, out_dim), features: (B, C_ctx, T_ctx)
        elif self.speaker_encoder_type == 'wavlm':
            wav = ref_wav  # [B, T]
            if self.wavlm_normalize:
                wav = torch.nn.functional.layer_norm(wav, wav.shape)
            # features: [B, T, D]
            # Use no_grad for frozen WavLM to save memory and speed up
            use_fp32_island = (
                wav.is_cuda
                and self.freeze_wavlm
                and self.frozen_wavlm_force_fp32
            )
            autocast_ctx = (
                autocast(device_type=wav.device.type, enabled=False)
                if use_fp32_island
                else nullcontext()
            )
            if self.use_inference_mode_for_frozen_wavlm:
                with autocast_ctx:
                    with torch.inference_mode():
                        features = self.wavlm_model.extract_features(
                            wav,
                            output_layer=self.wavlm_output_layer,
                        )[0]
                # Inference tensors cannot be saved for backward by downstream trainable layers.
                features = features.clone()
            else:
                with autocast_ctx:
                    with torch.no_grad():
                        features = self.wavlm_model.extract_features(
                            wav,
                            output_layer=self.wavlm_output_layer,
                        )[0]
            # global pooled -> x_vector (B, out_dim)
            x_vector = self.x_project(features.mean(dim=1))
            # x_vector: (B, out_dim), features: (B, T_ctx, D_ctx)
        
        if self.use_perceiver_encoder:
            # Speaker token flow:
            # - default: backbone features -> PerceiverResampler -> fixed token_num tokens
            # - use_memory_cattn=True: backbone features -> MemoryCrossAttentionEncoder
            #   -> PerceiverResampler -> fixed token_num tokens
            if self.speaker_encoder_type in ['ecapa_tdnn', 'ecapa_tdnn_speechbrain']:
                x = self._encode_speaker_tokens(features.transpose(1, 2)).transpose(1, 2)  # (B, latent_dim, token_num)
            elif self.speaker_encoder_type == 'wavlm':
                x = self._encode_speaker_tokens(features).transpose(1, 2)  # (B, latent_dim, token_num)

            if self.use_quantizer and self.quantizer is not None:
                out = self._normalize_quantizer_output(
                    self.quantizer(x),
                    batch_size=x.shape[0],
                    token_count=x.shape[-1],
                )
                z_q = out['z_q']  # (B, latent_dim, token_num)
            else:
                z_q = x  # (B, latent_dim, token_num)
                out = None
            
            # x_vector and d_vector have the same shape (B, out_dim), but come from
            # different stages. x_vector is the backbone global embedding, while
            # d_vector is rebuilt from the fixed-length speaker token sequence.
            pooled = z_q.reshape(z_q.shape[0], -1)  # (B, latent_dim * token_num)
            gq_vector = self.project(pooled)  # (B, out_dim)
            x_quantized_projected = z_q  # (B, latent_dim, token_num)
        else:
            assert self.prompt_prenet is not None, "prompt_prenet must be defined when use_perceiver_encoder=False."
            if self.speaker_encoder_type in ['ecapa_tdnn', 'ecapa_tdnn_speechbrain']:
                feats = features  # (B, C, T)
            elif self.speaker_encoder_type == 'wavlm':
                feats = features.transpose(1, 2)  # (B, D, T)
            else:
                raise ValueError(f"Unsupported speaker_encoder_type for prompt prenet: {self.speaker_encoder_type}")
            x_quantized_projected = self.prompt_prenet(feats)  # (B, latent_dim, T_spk)
            return x_vector, None, x_quantized_projected, None

        if self.use_quantizer and self.quantizer is not None:
            if self.vq_type == 'fsq':
                return x_vector, gq_vector, x_quantized_projected, None
            elif self.vq_type == 'simvq':
                return x_vector, gq_vector, x_quantized_projected, out['vq_loss']
            else:
                raise ValueError(f"Unsupported vq_type: {self.vq_type}")
        else:
            return x_vector, gq_vector, x_quantized_projected, None
    
    def tokenize(self, mels: torch.Tensor) -> torch.Tensor:
        """tokenize the input mel spectrogram"""
        if not self.use_perceiver_encoder or self.perceiver_sampler is None or self.quantizer is None:
            raise RuntimeError("Quantization is disabled when use_perceiver_encoder=False.")
        if self.speaker_encoder_type in ['ecapa_tdnn', 'ecapa_tdnn_speechbrain']:
            _, features = self.speaker_encoder(mels, True)
            x = self._encode_speaker_tokens(features.transpose(1, 2)).transpose(1, 2)
            out = self._normalize_quantizer_output(
                self.quantizer(x),
                batch_size=x.shape[0],
                token_count=x.shape[-1],
            )
            return out['indices']
        else:
            raise NotImplementedError("tokenize(mels) is only supported for ECAPA-based encoders. Use tokenize_wav for WavLM.")

    def tokenize_wav(self, wav: torch.Tensor) -> torch.Tensor:
        """tokenize from waveform using WavLM (expects 16kHz wav)"""
        assert self.speaker_encoder_type == 'wavlm', "tokenize_wav is only available for WavLM encoder"
        if not self.use_perceiver_encoder or self.perceiver_sampler is None or self.quantizer is None:
            raise RuntimeError("Quantization is disabled when use_perceiver_encoder=False.")
        x = wav
        if self.wavlm_normalize:
            x = torch.nn.functional.layer_norm(x, x.shape)
        # Use no_grad for frozen WavLM
        with torch.no_grad():
            feats = self.wavlm_model.extract_features(x, output_layer=self.wavlm_output_layer)[0]  # (B, T, D)
        q_in = self._encode_speaker_tokens(feats).transpose(1, 2)  # (B, latent_dim, token_num)
        out = self._normalize_quantizer_output(
            self.quantizer(q_in),
            batch_size=q_in.shape[0],
            token_count=q_in.shape[-1],
        )
        return out['indices']
    
    def detokenize(self, indices: torch.Tensor) -> torch.Tensor:
        """detokenize the input indices to d-vector"""
        if not self.use_perceiver_encoder or self.quantizer is None:
            raise RuntimeError("Quantization is disabled when use_perceiver_encoder=False.")
        if self.vq_type == 'simvq':
            zq = self.quantizer.get_output_from_indices(indices)
        elif self.vq_type == 'fsq':
            zq = self.quantizer.get_output_from_indices(indices.transpose(1, 2)).transpose(1, 2)

        pooled = zq.reshape(zq.shape[0], -1)  # (B, latent_dim * token_num)
        gq_vector = self.project(pooled)  # (B, out_dim)
        return gq_vector

    def _normalize_quantizer_output(
        self,
        out: Any,
        batch_size: int,
        token_count: int,
    ) -> Dict[str, Any]:
        if isinstance(out, dict):
            return {
                'z_q': out['z_q'],
                'indices': out['indices'],
                'vq_loss': out.get('vq_loss'),
            }

        if isinstance(out, (tuple, list)) and len(out) >= 3:
            z_q, indices, vq_loss = out[:3]
            if indices.dim() == 1:
                indices = indices.view(batch_size, 1, token_count)
            elif indices.dim() == 2:
                if indices.shape == (batch_size, token_count):
                    indices = indices.unsqueeze(1)
                elif indices.shape == (batch_size * token_count, 1):
                    indices = indices.view(batch_size, 1, token_count)
                else:
                    raise ValueError(f"Unsupported SimVQ index shape: {tuple(indices.shape)}")
            return {
                'z_q': z_q,
                'indices': indices,
                'vq_loss': vq_loss,
            }

        raise TypeError(f"Unsupported quantizer output type: {type(out)}")

if __name__ == "__main__":
    model = SpeakerEncoder(
        input_dim=100,
        latent_dim=128,
        token_num=32,
        fsq_levels=[4, 4, 4, 4, 4, 4],
        fsq_num_quantizers=1,
    )
    mel = torch.randn(8, 200, 100)
    x_vector, d_vector = model(mel)
    print("x-vector shape", x_vector.shape)
    print("d-vector shape", d_vector.shape)

    indices = model.tokenize(mel)
    print("indices shape", indices.shape)
    d_vector_post = model.detokenize(indices)
    print("d-vector shape", d_vector_post.shape)
    if d_vector_post.all() == d_vector.all():
        print("d-vector post and d-vector are the same")
    else:
        print("d-vector post and d-vector are different")
    num_params = sum(param.numel() for param in model.parameters())
    print("{} M".format(num_params / 1e6))
