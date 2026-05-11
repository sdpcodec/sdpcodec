import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Sequence
from typing import Any, Optional, Union

from .module import CrossAttentionBlock, MambaBlock, _as_plain_dict
from torch.utils.checkpoint import checkpoint


class LayerNorm1d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.channels))
        self.bias = nn.Parameter(torch.zeros(self.channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.layer_norm(x, (self.channels,), self.weight, self.bias, self.eps)
        return x.transpose(1, 2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-2):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(dim)))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        return (self._norm(x_float) * self.weight).to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0, device=None):
        super().__init__()
        self.dim = int(dim)
        self.max_position_embeddings = int(max_position_embeddings)
        self.base = float(base)

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.max_seq_len_cached = self.max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = int(seq_len)
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _build_token_norm(dim: int, norm_type: str, norm_eps: float) -> nn.Module:
    resolved = str(norm_type).lower()
    if resolved == "rmsnorm":
        return RMSNorm(dim, eps=norm_eps)
    if resolved == "layernorm":
        return nn.LayerNorm(dim, eps=norm_eps)
    raise ValueError(f"Unsupported norm_type: {norm_type}. Supported: rmsnorm, layernorm.")


def _resolve_attn_window_size(
    window_size: Optional[Union[int, Sequence[int]]],
    attn_window_size: Optional[Sequence[int]],
) -> tuple[int, int]:
    if attn_window_size is not None:
        if len(attn_window_size) != 2:
            raise ValueError("attn_window_size must contain exactly two integers: [left, right].")
        left, right = (max(0, int(attn_window_size[0])), max(0, int(attn_window_size[1])))
        return left, right

    if window_size is None:
        return -1, -1

    if isinstance(window_size, Sequence) and not isinstance(window_size, (str, bytes)):
        if len(window_size) != 2:
            raise ValueError("window_size sequence must contain exactly two integers: [left, right].")
        left, right = (max(0, int(window_size[0])), max(0, int(window_size[1])))
        return left, right

    symmetric = max(0, int(window_size))
    return symmetric, symmetric


def _resolve_block_range(total_layers: int, start_layer: int, end_layer: Optional[int], every: int) -> tuple[int, int, int]:
    start = max(0, min(int(start_layer), total_layers))
    if end_layer is None:
        end = total_layers
    else:
        end = max(start, min(int(end_layer), total_layers))
    stride = max(1, int(every))
    return start, end, stride


class LayerScale(nn.Module):
    def __init__(self, dim: int, gamma_init: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(int(dim)) * float(gamma_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(2 * (int(dim) * float(mult)) / 3)
        multiple_of = 256
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class ConvGELUFFN(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, int(intermediate_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(intermediate_dim), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_ffn(
    dim: int,
    intermediate_dim: int,
    ffn_type: str,
    ffn_mult: float,
    dropout: float,
) -> nn.Module:
    resolved = str(ffn_type).lower()
    if resolved == "swiglu":
        return SwiGLUFFN(dim=dim, mult=ffn_mult, dropout=dropout)
    if resolved == "conv_gelu":
        return ConvGELUFFN(dim=dim, intermediate_dim=intermediate_dim, dropout=dropout)
    raise ValueError(f"Unsupported ffn_type: {ffn_type}. Supported: swiglu, conv_gelu.")


class ConvNeXtBlock1d(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel_size: int = 7,
        layer_scale_init_value: float = 1 / 8,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.norm = LayerNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, intermediate_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(intermediate_dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma.view(1, -1, 1)
        return x + residual


class MambaBlock1d(nn.Module):
    def __init__(self, dim: int, mamba_kwargs: Optional[dict] = None):
        super().__init__()
        self.dim = int(dim)
        self.block = MambaBlock(cin=self.dim, cout=self.dim, **_as_plain_dict(mamba_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.block(x)
        return x.transpose(1, 2)


class ConvNeXtMambaBlock1d(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel_size: int = 7,
        layer_scale_init_value: float = 1 / 8,
        mamba_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.convnext = ConvNeXtBlock1d(
            dim=dim,
            intermediate_dim=intermediate_dim,
            kernel_size=kernel_size,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.mamba = MambaBlock1d(dim=dim, mamba_kwargs=mamba_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convnext(x)
        return self.mamba(x)


class TimeTransformerBlock1d(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: Optional[int] = None,
        attn_window_size: Optional[Sequence[int]] = None,
        intermediate_dim: int = 1536,
        dropout: float = 0.0,
        layer_scale_init_value: float = 1.0,
        attention_impl: str = "flash_attn",
        use_rope: bool = True,
        rope_base: float = 10000.0,
        max_position_embeddings: int = 2048,
        qk_norm: bool = True,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-2,
        ffn_type: str = "swiglu",
        ffn_mult: float = 4.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.attn_norm_in = _build_token_norm(self.dim, norm_type=norm_type, norm_eps=norm_eps)
        self.ffn_norm_in = _build_token_norm(self.dim, norm_type=norm_type, norm_eps=norm_eps)
        self.attn = FlashAttentionSelfAttention(
            dim=self.dim,
            num_heads=int(num_heads),
            window_size=window_size,
            attn_window_size=attn_window_size,
            dropout=float(dropout),
            attention_impl=attention_impl,
            use_rope=use_rope,
            rope_base=rope_base,
            max_position_embeddings=max_position_embeddings,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
        )
        self.ffn = _build_ffn(
            dim=self.dim,
            intermediate_dim=int(intermediate_dim),
            ffn_type=ffn_type,
            ffn_mult=ffn_mult,
            dropout=float(dropout),
        )
        self.attn_scale = LayerScale(self.dim, gamma_init=layer_scale_init_value)
        self.ffn_scale = LayerScale(self.dim, gamma_init=layer_scale_init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.transpose(1, 2)
        tokens = tokens + self.attn_scale(self.attn(self.attn_norm_in(tokens)))
        tokens = tokens + self.ffn_scale(self.ffn(self.ffn_norm_in(tokens)))
        return tokens.transpose(1, 2)


class FlashAttentionSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: Optional[int] = None,
        attn_window_size: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
        attention_impl: str = "flash_attn",
        use_rope: bool = True,
        rope_base: float = 10000.0,
        max_position_embeddings: int = 2048,
        qk_norm: bool = True,
        norm_eps: float = 1e-2,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.window_size = _resolve_attn_window_size(window_size=window_size, attn_window_size=attn_window_size)
        self.dropout = float(dropout)
        self.attention_impl = str(attention_impl).lower()
        self.use_rope = bool(use_rope)
        self.qk_norm = bool(qk_norm)
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads}).")
        self.head_dim = self.dim // self.num_heads
        self.qkv_proj = nn.Linear(self.dim, 3 * self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps) if self.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps) if self.qk_norm else nn.Identity()
        self.rotary_emb = (
            RotaryEmbedding(
                dim=self.head_dim,
                max_position_embeddings=max_position_embeddings,
                base=rope_base,
            )
            if self.use_rope else None
        )

        if self.attention_impl != "flash_attn":
            raise ValueError(
                f"Unsupported attention_impl: {attention_impl}. "
                "This vocosformer path is hard-switched to flash_attn only."
            )
        try:
            from flash_attn import flash_attn_qkvpacked_func as _flash_attn_qkvpacked_func
        except Exception as exc:
            raise ImportError(
                "flash_attn is required for vocosformer attention. "
                "Install the project dependencies including flash-attn."
            ) from exc
        self.flash_attn_qkvpacked_func = _flash_attn_qkvpacked_func

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        qkv = self.qkv_proj(x).view(batch, tokens, 3, self.num_heads, self.head_dim)
        q = self.q_norm(qkv[:, :, 0])
        k = self.k_norm(qkv[:, :, 1])
        v = qkv[:, :, 2]
        if self.use_rope and self.rotary_emb is not None:
            cos, sin = self.rotary_emb(qkv, seq_len=tokens)
            position_ids = torch.arange(tokens, device=x.device, dtype=torch.long).unsqueeze(0).expand(batch, tokens)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=2)
        qkv = torch.stack((q, k, v), dim=2)
        dropout_p = self.dropout if self.training else 0.0
        y = self.flash_attn_qkvpacked_func(
            qkv,
            dropout_p=dropout_p,
            softmax_scale=self.head_dim ** -0.5,
            causal=False,
            window_size=self.window_size,
        )
        y = y.reshape(batch, tokens, self.dim)
        return self.out_proj(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype not in (torch.float16, torch.bfloat16):
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type='cuda', dtype=target_dtype):
                y = self._forward_impl(x)
            return y.to(dtype=x.dtype)
        return self._forward_impl(x)


class ISTFTHead(nn.Module):
    def __init__(
        self,
        dim: int,
        n_fft: int,
        hop_length: int,
        win_length: Optional[int] = None,
    ):
        super().__init__()
        self.dim = int(dim)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length) if win_length is not None else int(n_fft)
        self.out_channels = 2 * (self.n_fft // 2 + 1)
        self.proj = nn.Conv1d(dim, self.out_channels, kernel_size=1)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        self.last_outputs: Optional[dict[str, torch.Tensor]] = None

    def forward(self, x: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        x = self.proj(x)
        log_mag, phase = x.chunk(2, dim=1)
        log_mag = log_mag.float()
        phase = phase.float()
        # Keep spectral parameterization numerically stable under AMP.
        mag = torch.exp(torch.clamp(log_mag, min=-10.0, max=10.0))
        phase = torch.pi * torch.tanh(phase)
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        self.last_outputs = {
            "log_amplitude": log_mag,
            "phase": phase,
            "real": real,
            "imag": imag,
        }
        spec = torch.polar(mag, phase)
        wav = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(spec.device),
            length=length,
        )
        return wav.unsqueeze(1)


class APNISTFTHead(nn.Module):
    def __init__(
        self,
        dim: int,
        n_fft: int,
        hop_length: int,
        win_length: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        phase_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = int(dim)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length) if win_length is not None else int(n_fft)
        self.freq_bins = self.n_fft // 2 + 1
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else int(dim)
        self.phase_norm_eps = float(phase_norm_eps)

        self.amplitude_branch = nn.Sequential(
            nn.Conv1d(self.dim, self.hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(self.hidden_dim, self.freq_bins, kernel_size=1),
        )
        self.phase_branch = nn.Sequential(
            nn.Conv1d(self.dim, self.hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(self.hidden_dim, 2 * self.freq_bins, kernel_size=1),
        )
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        self.last_outputs: Optional[dict[str, torch.Tensor]] = None

    def forward(self, x: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        log_mag = self.amplitude_branch(x).float()
        phase_real, phase_imag = self.phase_branch(x).chunk(2, dim=1)
        phase_real = phase_real.float()
        phase_imag = phase_imag.float()

        phase_norm = torch.sqrt(phase_real.pow(2) + phase_imag.pow(2) + self.phase_norm_eps)
        phase_real = phase_real / phase_norm
        phase_imag = phase_imag / phase_norm
        mag = torch.exp(torch.clamp(log_mag, min=-10.0, max=10.0))
        phase = torch.atan2(phase_imag, phase_real)
        real = mag * phase_real
        imag = mag * phase_imag
        self.last_outputs = {
            "log_amplitude": log_mag,
            "phase": phase,
            "real": real,
            "imag": imag,
        }
        spec = torch.polar(mag, phase)
        wav = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(spec.device),
            length=length,
        )
        return wav.unsqueeze(1)


class VocosLatentDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        sampling_rate: int,
        hop_size: int,
        n_fft: Optional[int] = None,
        win_size: Optional[int] = None,
        dim: int = 512,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
        kernel_size: int = 7,
        padding: str = "same",
        speaker_condition: bool = False,
        condition_dim: int = 1024,
        f0_condition: bool = False,
        f0_start_layer: int = 0,
        f0_end_layer: Optional[int] = None,
        f0_every: int = 1,
        f0_speaker_condition: bool = False,
        use_stage_speaker_film: bool = False,
        f0_stage_dims: Optional[Sequence[int]] = None,
        leaky_relu_params: Optional[dict] = None,
        use_mhca: bool = False,
        spk_cond_use_concat: bool = False,
        mhca_num_heads: int = 8,
        mhca_dropout: float = 0.1,
        mhca_key_dim: int = 128,
        mhca_use_sdpa: Optional[bool] = None,
        num_heads: int = 8,
        window_size: Optional[int] = None,
        mhca_start_layer: int = 0,
        mhca_end_layer: Optional[int] = None,
        mhca_every: int = 1,
        backbone_type: str = "convnext",
        mamba_every: int = 2,
        mamba_start_layer: Optional[int] = None,
        mamba_end_layer: Optional[int] = None,
        mamba_kwargs: Optional[dict] = None,
        temporal: Optional[Any] = None,
    ):
        super().__init__()
        del num_heads, window_size
        self.latent_dim = int(latent_dim)
        self.sampling_rate = int(sampling_rate)
        self.hop_size = int(hop_size)
        self.n_fft = int(n_fft) if n_fft is not None else int(self.hop_size * 4)
        self.win_size = int(win_size) if win_size is not None else int(self.n_fft)
        self.dim = int(dim)
        self.intermediate_dim = int(intermediate_dim)
        self.num_layers = int(num_layers)
        self.speaker_condition = bool(speaker_condition)
        self.f0_condition = bool(f0_condition)
        self.f0_start_layer = int(f0_start_layer)
        self.f0_end_layer = f0_end_layer
        self.f0_every = max(1, int(f0_every))
        self.f0_speaker_condition = bool(f0_speaker_condition)
        self.use_stage_speaker_film = bool(use_stage_speaker_film)
        self.use_stage_speaker_concat = bool(spk_cond_use_concat)
        self.use_mhca = bool(use_mhca)
        self.mhca_start_layer = int(mhca_start_layer)
        self.mhca_end_layer = mhca_end_layer
        self.mhca_every = max(1, int(mhca_every))
        self.spk_cond_start_layer = int(mhca_start_layer)
        self.spk_cond_end_layer = mhca_end_layer
        self.spk_cond_every = max(1, int(mhca_every))
        self.padding = str(padding)
        self.f0_stage_dims = None if f0_stage_dims is None else [int(stage_dim) for stage_dim in f0_stage_dims]
        self.backbone_type = "convnext"
        self.temporal = temporal
        negative_slope = 0.1 if leaky_relu_params is None else leaky_relu_params.get("negative_slope", 0.1)

        if self.latent_dim != self.dim:
            raise ValueError(
                f"VocosLatentDecoder expects latent_dim == dim when input_proj is disabled, "
                f"but got latent_dim={self.latent_dim}, dim={self.dim}."
            )
        if str(backbone_type).lower() != "convnext":
            print(
                f"Warning: vocos_backbone={backbone_type} is ignored. "
                "Vocos decoder now uses fixed ConvNeXt blocks with temporal modules inserted before selected blocks."
            )
        if self.f0_condition:
            if self.f0_stage_dims is None or len(self.f0_stage_dims) == 0:
                raise ValueError("VocosLatentDecoder requires f0_stage_dims when f0_condition=True.")
            if len(self.f0_stage_dims) < self.num_layers:
                padded_dims = list(self.f0_stage_dims)
                padded_dims.extend([padded_dims[-1]] * (self.num_layers - len(padded_dims)))
                self.f0_stage_dims = padded_dims
            else:
                self.f0_stage_dims = list(self.f0_stage_dims[:self.num_layers])
        self.temporal_start_layer = 0
        self.temporal_end_layer = 0
        self.temporal_every = 1
        self.temporal_block_enabled = [False] * self.num_layers
        self.temporal_blocks = nn.ModuleList()
        self.input_temporal = nn.Identity()
        if self.temporal is not None and self.temporal.use:
            if self.temporal.backbone not in {"lstm", "mamba"}:
                raise ValueError(
                    f"Unsupported vocos temporal type: {self.temporal.backbone}. Supported: lstm, mamba."
                )
            self.input_temporal = self.temporal.build_res_path(self.dim)
            self.temporal_start_layer, self.temporal_end_layer, self.temporal_every = self.temporal.resolve_vocos_block_range(
                self.num_layers
            )

        for idx in range(self.num_layers):
            enabled = (
                self.temporal is not None
                and self.temporal.use
                and self.temporal_start_layer <= idx < self.temporal_end_layer
                and ((idx - self.temporal_start_layer) % self.temporal_every == 0)
            )
            self.temporal_block_enabled[idx] = enabled
            if enabled:
                self.temporal_blocks.append(self.temporal.build_res_path(self.dim))
            else:
                self.temporal_blocks.append(nn.Identity())

        self.condition_fuser = None
        if self.speaker_condition:
            self.condition_fuser = nn.Sequential(
                nn.Conv1d(self.dim + condition_dim, self.dim, kernel_size=1),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.Conv1d(self.dim, self.dim, kernel_size=1),
            )

        if self.f0_condition:
            self.f0_start_layer, self.f0_end_layer, self.f0_every = _resolve_block_range(
                self.num_layers,
                self.f0_start_layer,
                self.f0_end_layer,
                self.f0_every,
            )
        else:
            self.f0_start_layer = 0
            self.f0_end_layer = 0

        self.f0_concat_block_enabled = [False] * self.num_layers
        self.f0_mhca_block_enabled = [False] * self.num_layers
        self.spk_concat_block_enabled = [False] * self.num_layers
        self.spk_film_block_enabled = [False] * self.num_layers
        self.f0_stage_fusers = nn.ModuleList()
        self.f0_mhca_list = nn.ModuleList()
        self.speaker_stage_fusers = nn.ModuleList()
        self.speaker_stage_film = nn.ModuleList()
        for idx in range(self.num_layers):
            enable_f0_concat = (
                self.f0_condition
                and self.f0_start_layer <= idx < self.f0_end_layer
                and ((idx - self.f0_start_layer) % self.f0_every == 0)
            )
            enable_f0_mhca = False
            enable_spk_concat = (
                self.speaker_condition
                and self.use_stage_speaker_concat
                and self._is_spk_cond_block_enabled(idx)
            )
            enable_spk_film = (
                self.speaker_condition
                and self.use_stage_speaker_film
                and self._is_spk_cond_block_enabled(idx)
            )
            self.f0_concat_block_enabled[idx] = enable_f0_concat
            self.f0_mhca_block_enabled[idx] = enable_f0_mhca
            self.spk_concat_block_enabled[idx] = enable_spk_concat
            self.spk_film_block_enabled[idx] = enable_spk_film
            if enable_f0_concat:
                self.f0_stage_fusers.append(
                    nn.Sequential(
                        nn.Conv1d(self.dim + self.f0_stage_dims[idx], self.dim, kernel_size=1),
                        nn.LeakyReLU(negative_slope=negative_slope),
                        nn.Conv1d(self.dim, self.dim, kernel_size=1),
                    )
                )
            else:
                self.f0_stage_fusers.append(nn.Identity())
            self.f0_mhca_list.append(nn.Identity())
            if enable_spk_concat:
                self.speaker_stage_fusers.append(
                    nn.Sequential(
                        nn.Conv1d(self.dim + condition_dim, self.dim, kernel_size=1),
                        nn.LeakyReLU(negative_slope=negative_slope),
                        nn.Conv1d(self.dim, self.dim, kernel_size=1),
                    )
                )
            else:
                self.speaker_stage_fusers.append(nn.Identity())
            if enable_spk_film:
                self.speaker_stage_film.append(
                    nn.Sequential(
                        nn.Linear(condition_dim, self.dim * 2),
                        nn.LeakyReLU(negative_slope=negative_slope),
                        nn.Linear(self.dim * 2, self.dim * 2),
                    )
                )
            else:
                self.speaker_stage_film.append(nn.Identity())

        self.backbone = nn.ModuleList(
            [
                ConvNeXtBlock1d(
                    dim=self.dim,
                    intermediate_dim=self.intermediate_dim,
                    kernel_size=kernel_size,
                    layer_scale_init_value=1 / max(self.num_layers, 1),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = LayerNorm1d(self.dim)
        self.head = ISTFTHead(
            dim=self.dim,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
        )

        self.mhca_block_enabled = [False] * self.num_layers
        if self.use_mhca:
            mhca_start = max(0, min(self.mhca_start_layer, self.num_layers))
            if self.mhca_end_layer is None:
                mhca_end = self.num_layers
            else:
                mhca_end = max(mhca_start, min(int(self.mhca_end_layer), self.num_layers))
            self.mhca_start_layer = mhca_start
            self.mhca_end_layer = mhca_end
            self.mhca_list = nn.ModuleList(
                [
                    CrossAttentionBlock(
                        query_dim=self.dim,
                        key_dim=mhca_key_dim,
                        num_heads=mhca_num_heads,
                        dropout=mhca_dropout,
                        use_sdpa=mhca_use_sdpa,
                    )
                    if (
                        mhca_start <= idx < mhca_end
                        and ((idx - mhca_start) % self.mhca_every == 0)
                    ) else nn.Identity()
                    for idx in range(self.num_layers)
                ]
            )
            for idx in range(self.num_layers):
                self.mhca_block_enabled[idx] = (
                    mhca_start <= idx < mhca_end
                    and ((idx - mhca_start) % self.mhca_every == 0)
                )
        else:
            self.mhca_list = None
            self.mhca_start_layer = 0
            self.mhca_end_layer = 0

        print(
            "VocosLatentDecoder init: "
            f"latent_dim={self.latent_dim}, sr={self.sampling_rate}, hop_size={self.hop_size}, "
            f"n_fft={self.n_fft}, win_size={self.win_size}, dim={self.dim}, "
            f"intermediate_dim={self.intermediate_dim}, num_layers={self.num_layers}, "
            f"speaker_condition={self.speaker_condition}, f0_condition={self.f0_condition}, "
            f"f0_layer_range=({self.f0_start_layer}, {self.f0_end_layer}), "
            f"f0_every={self.f0_every}, "
            f"f0_speaker_condition={self.f0_speaker_condition}, "
            f"use_stage_speaker_concat={self.use_stage_speaker_concat}, "
            f"use_stage_speaker_film={self.use_stage_speaker_film}, "
            f"use_mhca={self.use_mhca}, global_condition_fuser={self.condition_fuser is not None}, "
            f"padding={self.padding}, input_proj=False, backbone_type={self.backbone_type}, "
            f"temporal_type={getattr(self.temporal, 'backbone', None)}, "
            f"temporal_layer_range=({self.temporal_start_layer}, {self.temporal_end_layer}), "
            f"temporal_every={self.temporal_every}, "
            f"spk_cond_layer_range=({self.spk_cond_start_layer}, {self.spk_cond_end_layer}), "
            f"spk_cond_every={self.spk_cond_every}, "
            f"mhca_layer_range=({self.mhca_start_layer}, {self.mhca_end_layer}), "
            f"mhca_every={self.mhca_every}"
        )
        if len(self.f0_stage_fusers) > 0:
            print(f"VocosLatentDecoder stage conditioning enabled for {len(self.f0_stage_fusers)} layers.")
            print(f"VocosLatentDecoder f0_stage_dims={self.f0_stage_dims}")
        print(f"VocosLatentDecoder input temporal enabled={not isinstance(self.input_temporal, nn.Identity)}")
        print(
            f"VocosLatentDecoder temporal blocks enabled on layers="
            f"{[idx for idx, enabled in enumerate(self.temporal_block_enabled) if enabled]}"
        )
        print(
            f"VocosLatentDecoder MHCA enabled on layers="
            f"{[idx for idx, enabled in enumerate(self.mhca_block_enabled) if enabled]}"
        )
        print(
            f"VocosLatentDecoder global condition fuser enabled="
            f"{self.condition_fuser is not None}"
        )
        print(
            f"VocosLatentDecoder f0 concat enabled on layers="
            f"{[idx for idx, enabled in enumerate(self.f0_concat_block_enabled) if enabled]}"
        )
        print(
            f"VocosLatentDecoder f0 MHCA enabled on layers="
            f"{[idx for idx, enabled in enumerate(self.f0_mhca_block_enabled) if enabled]}"
        )
        print(
            f"VocosLatentDecoder speaker concat enabled on layers="
            f"{[idx for idx, enabled in enumerate(self.spk_concat_block_enabled) if enabled]}"
        )
        print(
            f"VocosLatentDecoder speaker FiLM enabled on layers="
            f"{[idx for idx, enabled in enumerate(self.spk_film_block_enabled) if enabled]}"
        )

        self.reset_parameters()
        self.latest_aux: Optional[dict[str, torch.Tensor]] = None

    def _build_stage_f0_conditions(
        self,
        f0_conds: Optional[Sequence[Optional[torch.Tensor]]],
    ) -> list[Optional[torch.Tensor]]:
        if not self.f0_condition:
            return [None] * len(self.backbone)

        if f0_conds is None:
            return [None] * len(self.backbone)

        if isinstance(f0_conds, torch.Tensor):
            valid = [f0_conds]
        else:
            valid = [tensor for tensor in f0_conds if isinstance(tensor, torch.Tensor)]

        if not valid:
            return [None] * len(self.backbone)

        if len(valid) >= len(self.backbone):
            return valid[:len(self.backbone)]

        padded = list(valid)
        while len(padded) < len(self.backbone):
            padded.append(valid[-1])
        return padded

    def _apply_spk_mhca(self, x: torch.Tensor, x_quantized: Optional[torch.Tensor], block_idx: int) -> torch.Tensor:
        if not self.use_mhca or self.mhca_list is None or x_quantized is None:
            return x
        if not self.mhca_block_enabled[block_idx]:
            return x
        return self.mhca_list[block_idx](x, x_quantized)

    def _is_spk_cond_block_enabled(self, block_idx: int) -> bool:
        if not (self.use_mhca or self.use_stage_speaker_film or self.use_stage_speaker_concat):
            return False
        start = self.spk_cond_start_layer
        if self.spk_cond_end_layer is None:
            end = self.num_layers
        else:
            end = self.spk_cond_end_layer
        return start <= block_idx < end and ((block_idx - start) % self.spk_cond_every == 0)

    def _apply_f0_stage_conditions(
        self,
        x: torch.Tensor,
        f0_stage_cond: Optional[torch.Tensor],
        block_idx: int,
    ) -> torch.Tensor:
        if not self.f0_concat_block_enabled[block_idx] or f0_stage_cond is None:
            return x
        if f0_stage_cond.dim() == 2:
            f0_stage_cond = f0_stage_cond.unsqueeze(1)
        if f0_stage_cond.shape[-1] != x.shape[-1]:
            f0_stage_cond = F.interpolate(f0_stage_cond, size=x.shape[-1], mode="nearest")
        return self.f0_stage_fusers[block_idx](torch.cat([x, f0_stage_cond], dim=1))

    def _apply_f0_mhca(
        self,
        x: torch.Tensor,
        f0_stage_cond: Optional[torch.Tensor],
        block_idx: int,
    ) -> torch.Tensor:
        if not self.f0_mhca_block_enabled[block_idx] or f0_stage_cond is None:
            return x
        if f0_stage_cond.dim() == 2:
            f0_stage_cond = f0_stage_cond.unsqueeze(1)
        if f0_stage_cond.shape[-1] != x.shape[-1]:
            f0_stage_cond = F.interpolate(f0_stage_cond, size=x.shape[-1], mode="nearest")
        return self.f0_mhca_list[block_idx](x, f0_stage_cond)

    def _apply_conditions(self, x: torch.Tensor, spk_cond: Optional[torch.Tensor]) -> torch.Tensor:
        if self.condition_fuser is None or spk_cond is None:
            return x
        spk_cond_time = spk_cond.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return self.condition_fuser(torch.cat([x, spk_cond_time], dim=1))

    def _apply_stage_speaker_film(
        self,
        x: torch.Tensor,
        spk_cond: Optional[torch.Tensor],
        block_idx: int,
    ) -> torch.Tensor:
        if not self.spk_film_block_enabled[block_idx] or spk_cond is None:
            return x
        gamma_beta = self.speaker_stage_film[block_idx](spk_cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return x * (1.0 + gamma) + beta

    def _apply_stage_speaker_concat(
        self,
        x: torch.Tensor,
        spk_cond: Optional[torch.Tensor],
        block_idx: int,
    ) -> torch.Tensor:
        if not self.spk_concat_block_enabled[block_idx] or spk_cond is None:
            return x
        spk_cond_time = spk_cond.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return self.speaker_stage_fusers[block_idx](torch.cat([x, spk_cond_time], dim=1))

    def forward(
        self,
        x: torch.Tensor,
        spk_cond: Optional[torch.Tensor] = None,
        f0_conds: Optional[Sequence[Optional[torch.Tensor]]] = None,
        x_quantized: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_frames = int(x.shape[-1])
        self.latest_aux = None
        stage_f0_conds = self._build_stage_f0_conditions(f0_conds)
        x = self.input_temporal(x)
        x = self._apply_conditions(x, spk_cond=spk_cond)

        for idx, block in enumerate(self.backbone):
            x = self._apply_f0_stage_conditions(x, stage_f0_conds[idx], idx)
            x = self._apply_f0_mhca(x, stage_f0_conds[idx], idx)
            x = self.temporal_blocks[idx](x)
            x = self._apply_stage_speaker_concat(x, spk_cond, idx)
            x = self._apply_stage_speaker_film(x, spk_cond, idx)
            x = self._apply_spk_mhca(x, x_quantized=x_quantized, block_idx=idx)
            x = block(x)

        x = self.final_norm(x)
        wav = self.head(x, length=input_frames * self.hop_size)
        self.latest_aux = getattr(self.head, "last_outputs", None)
        return wav

    def reset_parameters(self):
        def _init(module: nn.Module):
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_init)


class VocosFormerDecoder(nn.Module):
    """
    Transformer-backed variant of :class:`VocosLatentDecoder`.

    The conditioning pipeline intentionally mirrors the Vocos decoder:
    - optional input temporal block before any conditioning
    - global speaker condition fuser before the stack
    - per-layer order: f0 conditioning -> temporal block -> speaker conditioning -> transformer block
    - output head: ISTFT
    """

    def __init__(
        self,
        latent_dim: int,
        sampling_rate: int,
        hop_size: int,
        n_fft: Optional[int] = None,
        win_size: Optional[int] = None,
        dim: int = 512,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
        speaker_condition: bool = False,
        condition_dim: int = 1024,
        f0_condition: bool = False,
        f0_start_layer: int = 0,
        f0_end_layer: Optional[int] = None,
        f0_every: int = 1,
        f0_speaker_condition: bool = False,
        use_stage_speaker_film: bool = False,
        f0_stage_dims: Optional[Sequence[int]] = None,
        leaky_relu_params: Optional[dict] = None,
        use_mhca: bool = False,
        spk_cond_use_concat: bool = False,
        mhca_num_heads: int = 8,
        mhca_dropout: float = 0.1,
        mhca_key_dim: int = 128,
        mhca_use_sdpa: Optional[bool] = None,
        num_heads: int = 8,
        window_size: Optional[int] = None,
        attn_window_size: Optional[Sequence[int]] = None,
        mhca_start_layer: int = 0,
        mhca_end_layer: Optional[int] = None,
        mhca_every: int = 1,
        attention_impl: str = "flash_attn",
        use_rope: bool = True,
        rope_base: float = 10000.0,
        max_position_embeddings: int = 2048,
        qk_norm: bool = True,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-2,
        dropout: float = 0.0,
        ffn_type: str = "swiglu",
        ffn_mult: float = 4.0,
        layerscale_gamma_init: float = 1.0,
        temporal: Optional[Any] = None,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.sampling_rate = int(sampling_rate)
        self.hop_size = int(hop_size)
        self.n_fft = int(n_fft) if n_fft is not None else int(self.hop_size * 4)
        self.win_size = int(win_size) if win_size is not None else int(self.n_fft)
        self.dim = int(dim)
        self.intermediate_dim = int(intermediate_dim)
        self.num_layers = int(num_layers)
        self.speaker_condition = bool(speaker_condition)
        self.f0_condition = bool(f0_condition)
        self.f0_start_layer = int(f0_start_layer)
        self.f0_end_layer = f0_end_layer
        self.f0_every = max(1, int(f0_every))
        self.f0_speaker_condition = bool(f0_speaker_condition)
        self.use_stage_speaker_film = bool(use_stage_speaker_film)
        self.use_stage_speaker_concat = bool(spk_cond_use_concat)
        self.use_mhca = bool(use_mhca)
        self.mhca_start_layer = int(mhca_start_layer)
        self.mhca_end_layer = mhca_end_layer
        self.mhca_every = max(1, int(mhca_every))
        self.spk_cond_start_layer = int(mhca_start_layer)
        self.spk_cond_end_layer = mhca_end_layer
        self.spk_cond_every = max(1, int(mhca_every))
        self.temporal = temporal
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.f0_stage_dims = None if f0_stage_dims is None else [int(stage_dim) for stage_dim in f0_stage_dims]
        self.num_heads = max(1, int(num_heads))
        self.window_size = window_size
        self.attn_window_size = _resolve_attn_window_size(window_size=window_size, attn_window_size=attn_window_size)
        self.attention_impl = str(attention_impl).lower()
        self.use_rope = bool(use_rope)
        self.rope_base = float(rope_base)
        self.max_position_embeddings = int(max_position_embeddings)
        self.qk_norm = bool(qk_norm)
        self.norm_type = str(norm_type).lower()
        self.norm_eps = float(norm_eps)
        self.dropout = float(dropout)
        self.ffn_type = str(ffn_type).lower()
        self.ffn_mult = float(ffn_mult)
        self.layerscale_gamma_init = float(layerscale_gamma_init)

        if self.latent_dim != self.dim:
            raise ValueError(
                f"VocosFormerDecoder expects latent_dim == dim when input projection is disabled, "
                f"but got latent_dim={self.latent_dim}, dim={self.dim}."
            )
        if self.f0_condition:
            if self.f0_stage_dims is None or len(self.f0_stage_dims) == 0:
                raise ValueError("VocosFormerDecoder requires f0_stage_dims when f0_condition=True.")
            if len(self.f0_stage_dims) < self.num_layers:
                padded_dims = list(self.f0_stage_dims)
                padded_dims.extend([padded_dims[-1]] * (self.num_layers - len(padded_dims)))
                self.f0_stage_dims = padded_dims
            else:
                self.f0_stage_dims = list(self.f0_stage_dims[:self.num_layers])

        if self.speaker_condition and not (self.use_stage_speaker_concat or self.use_stage_speaker_film or self.use_mhca):
            print("VocosFormerDecoder: speaker_condition=True but concat/FiLM/MHCA are all disabled on decoder blocks.")
        if self.speaker_condition and not self.use_stage_speaker_concat and not self.use_stage_speaker_film and self.use_mhca:
            print("VocosFormerDecoder: block-wise speaker conditioning uses MHCA only (concat/FiLM disabled).")

        negative_slope = 0.1 if leaky_relu_params is None else leaky_relu_params.get("negative_slope", 0.1)
        self.backbone = nn.ModuleList(
            [
                TimeTransformerBlock1d(
                    dim=self.dim,
                    num_heads=self.num_heads,
                    window_size=self.window_size,
                    attn_window_size=self.attn_window_size,
                    intermediate_dim=self.intermediate_dim,
                    dropout=self.dropout,
                    layer_scale_init_value=self.layerscale_gamma_init,
                    attention_impl=self.attention_impl,
                    use_rope=self.use_rope,
                    rope_base=self.rope_base,
                    max_position_embeddings=self.max_position_embeddings,
                    qk_norm=self.qk_norm,
                    norm_type=self.norm_type,
                    norm_eps=self.norm_eps,
                    ffn_type=self.ffn_type,
                    ffn_mult=self.ffn_mult,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.temporal_start_layer = 0
        self.temporal_end_layer = 0
        self.temporal_every = 1
        self.hybrid_temporal_enabled = [False] * self.num_layers
        self.temporal_blocks = nn.ModuleList()
        self.input_temporal = nn.Identity()
        if self.temporal is not None and self.temporal.use:
            if self.temporal.backbone not in {"lstm", "mamba"}:
                raise ValueError(
                    f"Unsupported hybrid temporal type: {self.temporal.backbone}. Supported: lstm, mamba."
                )
            self.input_temporal = self.temporal.build_res_path(self.dim)
            self.temporal_start_layer, self.temporal_end_layer, self.temporal_every = self.temporal.resolve_vocos_block_range(
                self.num_layers
            )
            for idx in range(self.num_layers):
                enabled = (
                    self.temporal_start_layer <= idx < self.temporal_end_layer
                    and ((idx - self.temporal_start_layer) % self.temporal_every == 0)
                )
                self.hybrid_temporal_enabled[idx] = enabled
                if enabled:
                    self.temporal_blocks.append(self.temporal.build_res_path(self.dim))
                else:
                    self.temporal_blocks.append(nn.Identity())
        else:
            self.temporal_blocks = nn.ModuleList([nn.Identity() for _ in range(self.num_layers)])

        self.condition_fuser = None
        if self.speaker_condition:
            self.condition_fuser = nn.Sequential(
                nn.Conv1d(self.dim + condition_dim, self.dim, kernel_size=1),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.Conv1d(self.dim, self.dim, kernel_size=1),
            )

        if self.f0_condition:
            self.f0_start_layer, self.f0_end_layer, self.f0_every = _resolve_block_range(
                self.num_layers,
                self.f0_start_layer,
                self.f0_end_layer,
                self.f0_every,
            )
        else:
            self.f0_start_layer = 0
            self.f0_end_layer = 0

        self.f0_stage_fusers = nn.ModuleList()
        self.f0_concat_block_enabled = [False] * self.num_layers
        self.f0_mhca_block_enabled = [False] * self.num_layers
        self.f0_mhca_list = nn.ModuleList()
        self.spk_concat_block_enabled = [False] * self.num_layers
        self.speaker_stage_fusers = nn.ModuleList()
        self.spk_film_block_enabled = [False] * self.num_layers
        self.speaker_stage_film = nn.ModuleList()
        for idx in range(self.num_layers):
            enable_f0_concat = (
                self.f0_condition
                and self.f0_start_layer <= idx < self.f0_end_layer
                and ((idx - self.f0_start_layer) % self.f0_every == 0)
            )
            enable_f0_mhca = False
            enable_spk_concat = (
                self.speaker_condition
                and self.use_stage_speaker_concat
                and self._is_spk_cond_block_enabled(idx)
            )
            enable_spk_film = (
                self.speaker_condition
                and self.use_stage_speaker_film
                and self._is_spk_cond_block_enabled(idx)
            )
            self.f0_concat_block_enabled[idx] = enable_f0_concat
            self.f0_mhca_block_enabled[idx] = enable_f0_mhca
            self.spk_concat_block_enabled[idx] = enable_spk_concat
            self.spk_film_block_enabled[idx] = enable_spk_film
            if enable_f0_concat:
                self.f0_stage_fusers.append(
                    nn.Sequential(
                        nn.Conv1d(self.dim + self.f0_stage_dims[idx], self.dim, kernel_size=1),
                        nn.LeakyReLU(negative_slope=negative_slope),
                        nn.Conv1d(self.dim, self.dim, kernel_size=1),
                    )
                )
            else:
                self.f0_stage_fusers.append(nn.Identity())
            self.f0_mhca_list.append(nn.Identity())
            if enable_spk_concat:
                self.speaker_stage_fusers.append(
                    nn.Sequential(
                        nn.Conv1d(self.dim + condition_dim, self.dim, kernel_size=1),
                        nn.LeakyReLU(negative_slope=negative_slope),
                        nn.Conv1d(self.dim, self.dim, kernel_size=1),
                    )
                )
            else:
                self.speaker_stage_fusers.append(nn.Identity())
            if enable_spk_film:
                self.speaker_stage_film.append(
                    nn.Sequential(
                        nn.Linear(condition_dim, self.dim * 2),
                        nn.LeakyReLU(negative_slope=negative_slope),
                        nn.Linear(self.dim * 2, self.dim * 2),
                    )
                )
            else:
                self.speaker_stage_film.append(nn.Identity())

        self.mhca_block_enabled = [False] * self.num_layers
        if self.use_mhca:
            mhca_start = max(0, min(self.mhca_start_layer, self.num_layers))
            if self.mhca_end_layer is None:
                mhca_end = self.num_layers
            else:
                mhca_end = max(mhca_start, min(int(self.mhca_end_layer), self.num_layers))
            self.mhca_start_layer = mhca_start
            self.mhca_end_layer = mhca_end
            self.mhca_list = nn.ModuleList(
                [
                    CrossAttentionBlock(
                        query_dim=self.dim,
                        key_dim=mhca_key_dim,
                        num_heads=mhca_num_heads,
                        dropout=mhca_dropout,
                        use_sdpa=mhca_use_sdpa,
                    )
                    if (
                        mhca_start <= idx < mhca_end
                        and ((idx - mhca_start) % self.mhca_every == 0)
                    ) else nn.Identity()
                    for idx in range(self.num_layers)
                ]
            )
            for idx in range(self.num_layers):
                self.mhca_block_enabled[idx] = (
                    mhca_start <= idx < mhca_end
                    and ((idx - mhca_start) % self.mhca_every == 0)
                )
        else:
            self.mhca_list = None
            self.mhca_start_layer = 0
            self.mhca_end_layer = 0

        self.final_norm = _build_token_norm(self.dim, norm_type=self.norm_type, norm_eps=self.norm_eps)
        self.head = ISTFTHead(
            dim=self.dim,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
        )

        print(
            "VocosFormerDecoder init: "
            f"latent_dim={self.latent_dim}, sr={self.sampling_rate}, hop_size={self.hop_size}, "
            f"n_fft={self.n_fft}, win_size={self.win_size}, dim={self.dim}, "
            f"intermediate_dim={self.intermediate_dim}, num_layers={self.num_layers}, "
            f"speaker_condition={self.speaker_condition}, f0_condition={self.f0_condition}, "
            f"f0_layer_range=({self.f0_start_layer}, {self.f0_end_layer}), "
            f"f0_every={self.f0_every}, "
            f"f0_speaker_condition={self.f0_speaker_condition}, "
            f"use_stage_speaker_concat={self.use_stage_speaker_concat}, "
            f"use_stage_speaker_film={self.use_stage_speaker_film}, "
            f"use_mhca={self.use_mhca}, global_condition_fuser={self.condition_fuser is not None}, "
            f"temporal_type={getattr(self.temporal, 'backbone', None)}, "
            f"temporal_layer_range=({self.temporal_start_layer}, {self.temporal_end_layer}), "
            f"temporal_every={self.temporal_every}, "
            f"spk_cond_layer_range=({self.spk_cond_start_layer}, {self.spk_cond_end_layer}), "
            f"spk_cond_every={self.spk_cond_every}, "
            f"mhca_layer_range=({self.mhca_start_layer}, {self.mhca_end_layer}), "
            f"mhca_every={self.mhca_every}, "
            f"num_heads={self.num_heads}, "
            f"window_size={self.attn_window_size}, "
            f"attention_impl={self.attention_impl}, "
            f"use_rope={self.use_rope}, rope_base={self.rope_base}, "
            f"max_position_embeddings={self.max_position_embeddings}, "
            f"qk_norm={self.qk_norm}, norm_type={self.norm_type}, norm_eps={self.norm_eps}, "
            f"dropout={self.dropout}, ffn_type={self.ffn_type}, ffn_mult={self.ffn_mult}, "
            f"layerscale_gamma_init={self.layerscale_gamma_init}, "
            f"use_gradient_checkpointing={self.use_gradient_checkpointing}"
        )
        print(
            f"VocosFormerDecoder hybrid temporal layers="
            f"{[idx for idx, enabled in enumerate(self.hybrid_temporal_enabled) if enabled]}"
        )
        print(f"VocosFormerDecoder input temporal enabled={not isinstance(self.input_temporal, nn.Identity)}")
        print(
            f"VocosFormerDecoder global condition fuser enabled="
            f"{self.condition_fuser is not None}"
        )
        print(
            f"VocosFormerDecoder f0 concat enabled on layers="
            f"{[idx for idx, enabled in enumerate(self.f0_concat_block_enabled) if enabled]}"
        )
        print(
            f"VocosFormerDecoder f0 MHCA enabled on layers="
            f"{[idx for idx, enabled in enumerate(self.f0_mhca_block_enabled) if enabled]}"
        )
        print(
            f"VocosFormerDecoder speaker concat enabled on layers="
            f"{[idx for idx, enabled in enumerate(self.spk_concat_block_enabled) if enabled]}"
        )
        print(
            f"VocosFormerDecoder speaker FiLM enabled on layers="
            f"{[idx for idx, enabled in enumerate(self.spk_film_block_enabled) if enabled]}"
        )
        print(
            f"VocosFormerDecoder MHCA enabled on layers="
            f"{[idx for idx, enabled in enumerate(self.mhca_block_enabled) if enabled]}"
        )

        self.reset_parameters()
        self.latest_aux: Optional[dict[str, torch.Tensor]] = None

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.use_gradient_checkpointing and x.requires_grad:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def _build_stage_f0_conditions(
        self,
        f0_conds: Optional[Sequence[Optional[torch.Tensor]]],
    ) -> list[Optional[torch.Tensor]]:
        if not self.f0_condition:
            return [None] * self.num_layers
        if f0_conds is None:
            return [None] * self.num_layers
        if isinstance(f0_conds, torch.Tensor):
            valid = [f0_conds]
        else:
            valid = [tensor for tensor in f0_conds if isinstance(tensor, torch.Tensor)]
        if not valid:
            return [None] * self.num_layers
        if len(valid) >= self.num_layers:
            return valid[:self.num_layers]
        padded = list(valid)
        while len(padded) < self.num_layers:
            padded.append(valid[-1])
        return padded

    def _apply_spk_mhca(self, x: torch.Tensor, x_quantized: Optional[torch.Tensor], block_idx: int) -> torch.Tensor:
        if not self.use_mhca or self.mhca_list is None or x_quantized is None:
            return x
        if not self.mhca_block_enabled[block_idx]:
            return x
        return self.mhca_list[block_idx](x, x_quantized)

    def _is_spk_cond_block_enabled(self, block_idx: int) -> bool:
        if not (self.use_mhca or self.use_stage_speaker_film or self.use_stage_speaker_concat):
            return False
        start = self.spk_cond_start_layer
        if self.spk_cond_end_layer is None:
            end = self.num_layers
        else:
            end = self.spk_cond_end_layer
        return start <= block_idx < end and ((block_idx - start) % self.spk_cond_every == 0)

    def _apply_f0_stage_conditions(
        self,
        x: torch.Tensor,
        f0_stage_cond: Optional[torch.Tensor],
        block_idx: int,
    ) -> torch.Tensor:
        if not self.f0_concat_block_enabled[block_idx] or f0_stage_cond is None:
            return x
        if f0_stage_cond.dim() == 2:
            f0_stage_cond = f0_stage_cond.unsqueeze(1)
        if f0_stage_cond.shape[-1] != x.shape[-1]:
            f0_stage_cond = F.interpolate(f0_stage_cond, size=x.shape[-1], mode="nearest")
        return self.f0_stage_fusers[block_idx](torch.cat([x, f0_stage_cond], dim=1))

    def _apply_f0_mhca(
        self,
        x: torch.Tensor,
        f0_stage_cond: Optional[torch.Tensor],
        block_idx: int,
    ) -> torch.Tensor:
        if not self.f0_mhca_block_enabled[block_idx] or f0_stage_cond is None:
            return x
        if f0_stage_cond.dim() == 2:
            f0_stage_cond = f0_stage_cond.unsqueeze(1)
        if f0_stage_cond.shape[-1] != x.shape[-1]:
            f0_stage_cond = F.interpolate(f0_stage_cond, size=x.shape[-1], mode="nearest")
        return self.f0_mhca_list[block_idx](x, f0_stage_cond)

    def _apply_conditions(self, x: torch.Tensor, spk_cond: Optional[torch.Tensor]) -> torch.Tensor:
        if self.condition_fuser is None or spk_cond is None:
            return x
        spk_cond_time = spk_cond.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return self.condition_fuser(torch.cat([x, spk_cond_time], dim=1))

    def _apply_stage_speaker_film(
        self,
        x: torch.Tensor,
        spk_cond: Optional[torch.Tensor],
        block_idx: int,
    ) -> torch.Tensor:
        if not self.spk_film_block_enabled[block_idx] or spk_cond is None:
            return x
        gamma_beta = self.speaker_stage_film[block_idx](spk_cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return x * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

    def _apply_stage_speaker_concat(
        self,
        x: torch.Tensor,
        spk_cond: Optional[torch.Tensor],
        block_idx: int,
    ) -> torch.Tensor:
        if not self.spk_concat_block_enabled[block_idx] or spk_cond is None:
            return x
        spk_cond_time = spk_cond.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return self.speaker_stage_fusers[block_idx](torch.cat([x, spk_cond_time], dim=1))

    def forward(
        self,
        x: torch.Tensor,
        spk_cond: Optional[torch.Tensor] = None,
        f0_conds: Optional[Sequence[Optional[torch.Tensor]]] = None,
        x_quantized: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_frames = int(x.shape[-1])
        self.latest_aux = None
        stage_f0_conds = self._build_stage_f0_conditions(f0_conds)
        x = self.input_temporal(x)
        x = self._apply_conditions(x, spk_cond=spk_cond)

        for idx, block in enumerate(self.backbone):
            x = self._apply_f0_stage_conditions(x, stage_f0_conds[idx], idx)
            x = self._apply_f0_mhca(x, stage_f0_conds[idx], idx)
            x = self._run_block(self.temporal_blocks[idx], x)
            x = self._apply_stage_speaker_concat(x, spk_cond, idx)
            x = self._apply_stage_speaker_film(x, spk_cond, idx)
            x = self._apply_spk_mhca(x, x_quantized=x_quantized, block_idx=idx)
            x = self._run_block(block, x)

        x = self.final_norm(x.transpose(1, 2)).transpose(1, 2)
        wav = self.head(x, length=input_frames * self.hop_size)
        self.latest_aux = getattr(self.head, "last_outputs", None)
        return wav

    def reset_parameters(self):
        def _init(module: nn.Module):
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_init)


class APNFormerDecoder(VocosFormerDecoder):
    def __init__(
        self,
        *args,
        apn_head_hidden_dim: Optional[int] = None,
        apn_phase_norm_eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.head = APNISTFTHead(
            dim=self.dim,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            hidden_dim=apn_head_hidden_dim,
            phase_norm_eps=apn_phase_norm_eps,
        )
        self.head.apply(self._init_head_parameters)
        print(
            "APNFormerDecoder head: "
            f"hidden_dim={self.head.hidden_dim}, "
            f"phase_norm_eps={apn_phase_norm_eps}"
        )

    @staticmethod
    def _init_head_parameters(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
