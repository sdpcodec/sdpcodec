import warnings
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from . import activations
from .alias_free_torch import *
from torch.nn.utils import weight_norm
from torch.nn.utils.weight_norm import _weight_norm
from termcolor import colored

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def _resolve_conv_weight_bias(conv: nn.Module, ref: torch.Tensor):
    if hasattr(conv, "weight_v") and hasattr(conv, "weight_g"):
        weight = _weight_norm(conv.weight_v, conv.weight_g, 0)
        bias = conv.bias
    else:
        weight = conv.weight
        bias = conv.bias

    weight = weight.to(device=ref.device, dtype=ref.dtype)
    if bias is not None:
        bias = bias.to(device=ref.device, dtype=ref.dtype)
    return weight, bias


def _apply_repeated_global_condition_conv1d(
    global_condition: torch.Tensor,
    global_weight: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """Exact conv contribution of a zero-padded repeated global condition tensor."""
    if global_weight.ndim != 3:
        raise ValueError(f"Expected [out_channels, cond_channels, kernel], got {tuple(global_weight.shape)}")

    kernel_size = int(global_weight.shape[-1])
    padding = kernel_size // 2
    if kernel_size % 2 == 0:
        raise ValueError("Split global-condition conv optimization requires an odd kernel size.")

    if target_length <= 2 * padding:
        outputs = []
        for position in range(target_length):
            k_start = max(0, padding - position)
            k_end = min(kernel_size, target_length + padding - position)
            position_weight = global_weight[:, :, k_start:k_end].sum(dim=-1)
            outputs.append(F.linear(global_condition, position_weight).unsqueeze(-1))
        return torch.cat(outputs, dim=-1)

    full_weight = global_weight.sum(dim=-1)
    out = F.linear(global_condition, full_weight).unsqueeze(-1).expand(-1, -1, target_length).clone()

    for position in range(padding):
        left_weight = global_weight[:, :, padding - position:].sum(dim=-1)
        out[:, :, position] = F.linear(global_condition, left_weight)

        right_position = target_length - 1 - position
        right_weight = global_weight[:, :, :padding + position + 1].sum(dim=-1)
        out[:, :, right_position] = F.linear(global_condition, right_weight)

    return out


def _apply_split_conditioned_conv1d(
    conv: nn.Module,
    timed_input: torch.Tensor,
    global_condition: torch.Tensor | None,
) -> torch.Tensor:
    """Apply a conv over [timed_input, repeated_global_condition] without materializing the repeated tensor."""
    ref = timed_input if timed_input is not None else global_condition
    if ref is None:
        raise ValueError("Expected at least one input tensor for conditioned conv.")

    if getattr(conv, "stride", (1,))[0] != 1:
        raise ValueError("Split conditioned conv only supports stride=1.")
    if getattr(conv, "dilation", (1,))[0] != 1:
        raise ValueError("Split conditioned conv only supports dilation=1.")
    if getattr(conv, "groups", 1) != 1:
        raise ValueError("Split conditioned conv only supports groups=1.")

    weight, bias = _resolve_conv_weight_bias(conv, ref)
    timed_channels = 0 if timed_input is None else int(timed_input.shape[1])
    global_channels = 0 if global_condition is None else int(global_condition.shape[1])
    expected_channels = int(conv.in_channels)
    if timed_channels + global_channels != expected_channels:
        raise ValueError(
            f"Conditioned conv channel mismatch: expected {expected_channels}, "
            f"got timed={timed_channels}, global={global_channels}."
        )

    out = None
    if timed_input is not None:
        padding = conv.padding if isinstance(conv.padding, tuple) else (conv.padding,)
        out = F.conv1d(
            timed_input,
            weight[:, :timed_channels, :],
            bias=bias,
            stride=1,
            padding=padding[0],
            dilation=1,
        )
    elif bias is not None:
        batch = int(global_condition.shape[0])
        target_length = 1
        out = bias.view(1, -1, 1).expand(batch, -1, target_length)

    if global_condition is not None:
        global_out = _apply_repeated_global_condition_conv1d(
            global_condition=global_condition,
            global_weight=weight[:, timed_channels:, :],
            target_length=int(timed_input.shape[-1]) if timed_input is not None else 1,
        )
        out = global_out if out is None else out + global_out

    if out is None:
        batch = 1 if global_condition is None else int(global_condition.shape[0])
        target_length = 1 if timed_input is None else int(timed_input.shape[-1])
        out = weight.new_zeros((batch, weight.shape[0], target_length))

    return out


def build_codec_activation(
    dim: int,
    activation_type: str = "SnakeBeta",
    leaky_relu_params: Optional[Dict[str, Any]] = None,
    speaker_condition: bool = False,
    condition_dim: int = 1024,
    f0_speaker_condition: bool = False,
    f0_condition_dim: int = 128,
    alpha_logscale: bool = True,
    snake_lite_taylor_degree: int = 8,
    no_condition: bool = False,
    log_context: Optional[str] = None,
    log_details: Optional[str] = None,
) -> nn.Module:
    if activation_type == "LeakyReLU":
        if not leaky_relu_params or "negative_slope" not in leaky_relu_params:
            raise ValueError("LeakyReLU activation requires leaky_relu_params['negative_slope'].")
        return nn.LeakyReLU(negative_slope=leaky_relu_params["negative_slope"])

    family = None
    if isinstance(activation_type, str):
        if activation_type.startswith("SnakeLiteTriton"):
            family = "SnakeLiteTriton"
        elif activation_type.startswith("SnakeBeta"):
            family = "SnakeBeta"
        elif activation_type.startswith("SnakeLite"):
            family = "SnakeLite"

    if family is None:
        raise ValueError(
            "Unsupported activation: "
            f"{activation_type}. Supported activations are 'SnakeBeta', 'SnakeLite', 'SnakeLiteTriton', and 'LeakyReLU'."
        )

    activation_name = family
    activation_kwargs: Dict[str, Any] = {
        "alpha_logscale": alpha_logscale,
    }
    if family.startswith("SnakeLite"):
        activation_kwargs["snake_lite_taylor_degree"] = snake_lite_taylor_degree
    if no_condition or not speaker_condition:
        activation_cls = getattr(activations, family)
        activation = Activation1d(activation=activation_cls(dim, **activation_kwargs))
    elif speaker_condition and not f0_speaker_condition:
        activation_name = f"{family}WithCondition"
        activation_cls = getattr(activations, activation_name)
        activation = Activation1dWithCondition(
            activation=activation_cls(dim, condition_dim, **activation_kwargs)
        )
    else:
        activation_name = f"{family}WithTimeVaryingCondition"
        activation_cls = getattr(activations, activation_name)
        activation = Activation1dWithCondition(
            activation=activation_cls(
                dim,
                condition_dim + f0_condition_dim,
                **activation_kwargs,
            )
        )

    if log_context is not None:
        details = f": {log_details}" if log_details else ""
        if family.startswith("SnakeLite"):
            details = f"{details}, degree={snake_lite_taylor_degree}" if details else f": degree={snake_lite_taylor_degree}"
        print(colored(f"{log_context} using {activation_name}{details}", "yellow"))

    return activation


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16,
                 dilation: int = 1,
                 speaker_condition=False,
                 f0_speaker_condition=False,
                 condition_dim=1024,
                 f0_condition_dim=128,
                 activation_type = 'SnakeBeta',
                 leaky_relu_params = None,
                 snake_lite_taylor_degree: int = 8,
                 no_condition=False):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        activation = build_codec_activation(
            dim=dim,
            activation_type=activation_type,
            leaky_relu_params=leaky_relu_params,
            speaker_condition=speaker_condition,
            condition_dim=condition_dim,
            f0_speaker_condition=f0_speaker_condition,
            f0_condition_dim=f0_condition_dim,
            alpha_logscale=True,
            snake_lite_taylor_degree=snake_lite_taylor_degree,
            no_condition=no_condition,
            log_context=None if no_condition or activation_type == "LeakyReLU" else "ResidualUnit",
            log_details=f"dilation={dilation}",
        )

        self.block = nn.Sequential(
            activation,
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            activation,
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x, condition=None):
        res = x
        for i, layer in enumerate(self.block):
            if isinstance(layer, Activation1dWithCondition):
                x = layer(x, condition)
            else:
                x = layer(x)
        return x + res

class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1, dilations = (1, 3, 9), speaker_condition = False, condition_dim = 1024, activation_type = 'SnakeBeta', leaky_relu_params = None, snake_lite_taylor_degree: int = 8, input_dim: int = None):
        super().__init__()
        input_dim = dim // 2 if input_dim is None else input_dim
        runits = [ResidualUnit(input_dim, dilation=d, speaker_condition=speaker_condition, condition_dim=condition_dim, activation_type=activation_type, leaky_relu_params=leaky_relu_params, snake_lite_taylor_degree=snake_lite_taylor_degree) for d in dilations]
        activation = build_codec_activation(
            dim=input_dim,
            activation_type=activation_type,
            leaky_relu_params=leaky_relu_params,
            speaker_condition=speaker_condition,
            condition_dim=condition_dim,
            alpha_logscale=True,
            snake_lite_taylor_degree=snake_lite_taylor_degree,
        )
        # stride=1: use kernel_size=3, padding=1 to maintain temporal length
        # stride>1: use kernel_size=2*stride for downsampling
        if stride == 1:
            conv = WNConv1d(input_dim, dim, kernel_size=3, stride=1, padding=1)
        else:
            conv = WNConv1d(
                input_dim,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2 + stride % 2,
            )
        
        self.block = nn.Sequential(
            *runits,
            activation,
            conv,
        )

    def forward(self, x, condition=None):
        for layer in self.block:
            if isinstance(layer, Activation1dWithCondition) or isinstance(layer, ResidualUnit):
                x = layer(x, condition)
            else:
                x = layer(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1, dilations = (1, 3, 9), speaker_condition = False, condition_dim = 1024, f0_condition = False, f0_condition_dim: int = 128, f0_speaker_condition = False, activation_type = 'SnakeBeta', leaky_relu_params = None, snake_lite_taylor_degree: int = 8, use_split_condition_optimization: bool = True):
        super().__init__()
        self.f0_condition = f0_condition
        self.f0_speaker_condition = f0_speaker_condition
        self.speaker_condition = speaker_condition
        self.use_split_condition_optimization = bool(use_split_condition_optimization)
        self.use_condition_only = speaker_condition and not (self.f0_condition or self.f0_speaker_condition)
        self.f0_condition_dim = f0_condition_dim if f0_condition_dim is not None else 0
        activation = build_codec_activation(
            dim=input_dim,
            activation_type=activation_type,
            leaky_relu_params=leaky_relu_params,
            speaker_condition=speaker_condition,
            condition_dim=condition_dim,
            f0_speaker_condition=f0_speaker_condition,
            f0_condition_dim=self.f0_condition_dim,
            alpha_logscale=True,
            snake_lite_taylor_degree=snake_lite_taylor_degree,
            log_context=None if activation_type == "LeakyReLU" else "DecoderBlock",
            log_details=f"f0_condition={f0_condition}, f0_speaker_condition={f0_speaker_condition}",
        )
        # stride=1: use Conv1d to maintain temporal length (no upsampling)
        # stride>1: use ConvTranspose1d for upsampling
        if stride == 1:
            # Use regular Conv1d with kernel_size=3, padding=1 to maintain length
            self.block = nn.Sequential(
                activation,
                WNConv1d(input_dim, output_dim, kernel_size=3, padding=1)
            )
        else:
            self.block = nn.Sequential(
                activation,
                WNConvTranspose1d(
                    input_dim,
                    output_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=stride // 2 + stride % 2,
                    output_padding=stride % 2,
                )
            )
        self.block.extend([
            ResidualUnit(
                output_dim,
                dilation=d,
                speaker_condition=speaker_condition,
                f0_speaker_condition=False, # ResidualUnit does not support f0_speaker_condition !! only speaker info
                f0_condition_dim=self.f0_condition_dim,
                condition_dim=condition_dim,
                activation_type=activation_type,
                leaky_relu_params=leaky_relu_params,
                snake_lite_taylor_degree=snake_lite_taylor_degree,
            )
            for d in dilations
        ])
        
        print(colored(f"DecoderBlock: f0_condition={self.f0_condition}, f0_speaker_condition={self.f0_speaker_condition}", "yellow"))
        if self.f0_speaker_condition:
            if self.f0_condition:
                concat_dim = self.f0_condition_dim + condition_dim + input_dim
            else: concat_dim = condition_dim + input_dim
        else:
            if self.f0_condition: concat_dim = self.f0_condition_dim + input_dim
        if self.f0_condition or self.f0_speaker_condition:
            self.f0_cond_conv = WNConv1d(concat_dim, input_dim, kernel_size=7, padding=3)
        elif self.use_condition_only:
            print(colored(f"DecoderBlock using WNConv1d for condition only: condition_dim={condition_dim}, input_dim={input_dim}", "yellow"))
            self.condition_conv = WNConv1d(condition_dim + input_dim, input_dim, kernel_size=7, padding=3)

    def forward(self, x, condition=None, f0_cond=None):
        if self.f0_condition or self.f0_speaker_condition:
            if self.f0_condition:
                # Handle 2D tensor (B, T) -> (B, 1, T) for interpolate
                if f0_cond is not None and f0_cond.dim() == 2:
                    f0_cond = f0_cond.unsqueeze(1)  # (B, T) -> (B, 1, T)
                if f0_cond is not None and x.shape[-1] != f0_cond.shape[-1]:
                    # print(colored(f"Interpolating f0 condition from {f0_cond.shape} to match x {x.shape}", "yellow"))
                    f0_cond = torch.nn.functional.interpolate(f0_cond, size=x.shape[-1], mode='nearest')
                if f0_cond is not None:
                    assert x.shape[-1] == f0_cond.shape[-1], f"f0 shape {f0_cond.shape} does not match x shape {x.shape}"
            if self.f0_speaker_condition:
                if self.f0_condition:
                    condition_ = condition.unsqueeze(-1).expand(-1, -1, x.shape[-1])
                    x = torch.cat([x, condition_, f0_cond], dim=1)
                else:
                    # print('Concatenating speaker condition without f0 condition.')
                    condition_ = condition.unsqueeze(-1).expand(-1, -1, x.shape[-1])
                    x = torch.cat([x, condition_], dim=1)
            else:
                if self.f0_condition:
                    x = torch.cat([x, f0_cond], dim=1)
            x = self.f0_cond_conv(x)
        elif self.use_condition_only and condition is not None:
            condition_ = condition.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x = torch.cat([x, condition_], dim=1)
            x = self.condition_conv(x)

        def build_time_condition(seq_len):
            if not self.f0_speaker_condition:
                return condition
            if self.use_split_condition_optimization and self.f0_condition:
                return (condition, f0_cond)
            condition_time = condition.unsqueeze(-1).expand(-1, -1, seq_len)
            if self.f0_condition:
                condition_time = torch.cat([condition_time, f0_cond], dim=1)
            return condition_time

        for i, layer in enumerate(self.block):
            if isinstance(layer, Activation1dWithCondition):
                cond_for_layer = build_time_condition(x.shape[-1]) if self.f0_speaker_condition else condition
                x = layer(x, cond_for_layer)
            elif isinstance(layer, ResidualUnit):
                x = layer(x, condition)
            else:
                x = layer(x)
        return x
    
class ResLSTM(nn.Module):
    def __init__(self, dimension: int,
                 num_layers: int = 2,
                 bidirectional: bool = False,
                 skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension if not bidirectional else dimension // 2,
                            num_layers, batch_first=True,
                            bidirectional=bidirectional)

    def forward(self, x):
        """
        Args:
            x: [B, F, T]

        Returns:
            y: [B, F, T]
        """
        x = rearrange(x, "b f t -> b t f")
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = rearrange(y, "b t f -> b f t")
        return y


# --- Mamba-2 temporal blocks (alternative to ResLSTM; requires pip install mamba-ssm) ---


def _mamba2_cls():
    try:
        from mamba_ssm import Mamba2
    except ImportError as e:
        raise ImportError(
            "encoder_rnn_type/decoder_rnn_type='mamba' requires mamba-ssm. "
            "Install with: pip install mamba-ssm"
        ) from e
    return Mamba2


class MambaBlock(nn.Module):
    """
    Neural audio codec latent block using causal / unidirectional Mamba-2.

    Input:
        x: (B, T, C_in)

    Output:
        y: (B, T, C_out)

    Modes:
        - use_conv = False:
            Linear -> RMSNorm -> Mamba2 -> Residual -> RMSNorm -> FFN -> Residual
        - use_conv = True:
            Linear -> RMSNorm -> DepthwiseConv1d -> PointwiseConv1d -> Act
                   -> Mamba2 -> Residual -> RMSNorm -> FFN -> Residual
    """

    def __init__(
        self,
        cin: int,
        cout: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.0,
        use_conv: bool = False,
        conv_kernel_size: int = 5,
        conv_bias: bool = True,
    ):
        super().__init__()

        assert cin > 0 and cout > 0
        assert conv_kernel_size >= 1 and conv_kernel_size % 2 == 1, (
            "conv_kernel_size should be odd for simple symmetric padding."
        )

        Mamba2 = _mamba2_cls()

        self.cin = cin
        self.cout = cout
        self.use_conv = use_conv
        self.conv_kernel_size = conv_kernel_size

        self.in_proj = nn.Linear(cin, cout) if cin != cout else nn.Identity()

        self.norm1 = nn.RMSNorm(cout)

        if use_conv:
            self.dwconv = nn.Conv1d(
                in_channels=cout,
                out_channels=cout,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=cout,
                bias=conv_bias,
            )
            self.pwconv = nn.Conv1d(
                in_channels=cout,
                out_channels=cout,
                kernel_size=1,
                bias=conv_bias,
            )
            self.conv_act = nn.SiLU()
        else:
            self.dwconv = None
            self.pwconv = None
            self.conv_act = None

        self.ssm = Mamba2(
            d_model=cout,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.RMSNorm(cout)
        self.ffn = nn.Sequential(
            nn.Linear(cout, ff_mult * cout),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * cout, cout),
        )
        self.drop2 = nn.Dropout(dropout)

    def extra_repr(self) -> str:
        return (
            f"cin={self.cin}, cout={self.cout}, "
            f"use_conv={self.use_conv}, conv_kernel_size={self.conv_kernel_size}"
        )

    def _conv_frontend(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.conv_act(x)
        x = x.transpose(1, 2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"Expected 3D input (B, T, C), got shape {tuple(x.shape)}"
        assert x.size(-1) == self.cin, f"Expected input channel {self.cin}, got {x.size(-1)}"

        x = self.in_proj(x)

        y = self.norm1(x)
        if self.use_conv:
            y = self._conv_frontend(y)
        y = self.ssm(y)
        x = x + self.drop1(y)

        y = self.ffn(self.norm2(x))
        x = x + self.drop2(y)

        return x


class TemporalMamba(nn.Module):
    """
    Single MambaBlock with the same I/O contract as ResLSTM: (B, F, T) -> (B, F, T).
    """

    def __init__(
        self,
        dimension: int,
        bidirectional: bool = False,
        **mamba_kwargs: Any,
    ):
        super().__init__()
        if bidirectional:
            warnings.warn(
                "TemporalMamba uses causal Mamba-2; bidirectional=True is ignored.",
                UserWarning,
                stacklevel=2,
            )
        self.block = MambaBlock(cin=dimension, cout=dimension, **mamba_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b f t -> b t f")
        x = self.block(x)
        x = rearrange(x, "b t f -> b f t")
        return x


def _as_plain_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    except Exception:
        pass
    return dict(cfg) if isinstance(cfg, dict) else {}


def build_res_temporal(
    dimension: int,
    rnn_type: str,
    num_layers: int,
    bidirectional: bool,
    skip: bool = True,
    mamba: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """
    Build ResLSTM or a single-block TemporalMamba from a unified ``rnn_type`` string.

    ``rnn_type``:
        - ``lstm``: :class:`ResLSTM`
        - ``mamba``: :class:`TemporalMamba` (extra kwargs via ``mamba``)
    """
    t = (rnn_type or "lstm").lower()
    if t == "lstm":
        return ResLSTM(
            dimension,
            num_layers=num_layers,
            bidirectional=bidirectional,
            skip=skip,
        )
    if t == "mamba":
        kw: Dict[str, Any] = _as_plain_dict(mamba)
        if int(num_layers) != 1:
            warnings.warn(
                f"Ignoring requested mamba stack depth ({num_layers}); using a single MambaBlock.",
                UserWarning,
                stacklevel=2,
            )
        return TemporalMamba(
            dimension,
            bidirectional=bidirectional,
            **kw,
        )
    raise ValueError(
        f"Unsupported temporal type: {rnn_type!r}. "
        f"Expected 'lstm' or 'mamba'."
    )


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross Attention module for conditioning decoder input with speaker embeddings.
    
    Args:
        query_dim: dimension of query (decoder hidden state)
        key_dim: dimension of key/value (speaker embedding)
        num_heads: number of attention heads
        dropout: dropout probability
    """
    def __init__(self, query_dim: int, key_dim: int, num_heads: int = 8, dropout: float = 0.1, use_sdpa: Optional[bool] = None):
        super().__init__()
        assert query_dim % num_heads == 0, f"query_dim {query_dim} must be divisible by num_heads {num_heads}"
        
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
        # Preserve the current default (SDPA) while allowing callers to opt back
        # into the legacy manual attention path for exact old-model alignment.
        self.use_sdpa = True if use_sdpa is None else bool(use_sdpa)
    
    def forward(self, query, key_value):
        """
        Args:
            query: [B, C, T] - decoder hidden state
            key_value: [B, C_kv, T_kv] - speaker embedding sequence
        
        Returns:
            output: [B, C, T] - attended output
        """
        B, C, T = query.shape
        _, C_kv, T_kv = key_value.shape
        
        # Reshape to [B, T, C] for attention
        q = query.transpose(1, 2)  # [B, T, C]
        kv = key_value.transpose(1, 2)  # [B, T_kv, C_kv]
        
        # Project and reshape for multi-head attention
        q = self.q_proj(q).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = self.k_proj(kv).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T_kv, D]
        v = self.v_proj(kv).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T_kv, D]
        
        if self.use_sdpa:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, T_kv]
            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        out = self.out_proj(out)
        
        # Reshape back to [B, C, T]
        out = out.transpose(1, 2)  # [B, C, T]
        
        return out


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    Args:
        dim: input/output dimension
        hidden_dim: hidden layer dimension (typically 4 * dim)
        dropout: dropout probability
    """
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, T]
        Returns:
            output: [B, C, T]
        """
        # Transpose to [B, T, C] for linear layers
        x = x.transpose(1, 2)
        x = self.net(x)
        # Transpose back to [B, C, T]
        return x.transpose(1, 2)


class CrossAttentionBlock(nn.Module):
    """
    Complete Cross-Attention block with Pre-Normalization:
    1. Layer Norm -> Multi-Head Cross Attention -> Residual
    2. Layer Norm -> Feed-Forward Network -> Residual
    
    Args:
        query_dim: dimension of query (decoder hidden state)
        key_dim: dimension of key/value (speaker embedding)
        num_heads: number of attention heads
        ffn_hidden_dim: FFN hidden dimension (default: 4 * query_dim)
        dropout: dropout probability
    """
    def __init__(self, query_dim: int, key_dim: int, num_heads: int = 8, 
                 ffn_hidden_dim: int = None, dropout: float = 0.1, use_sdpa: Optional[bool] = None):
        super().__init__()
        
        self.query_dim = query_dim
        
        # Multi-Head Cross Attention
        self.cross_attn = MultiHeadCrossAttention(
            query_dim=query_dim,
            key_dim=key_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_sdpa=use_sdpa,
        )
        
        # Layer Norm for attention (works on channel dim in [B, C, T] format)
        self.norm1 = nn.LayerNorm(query_dim)
        
        # Feed-Forward Network
        self.ffn = FeedForwardNetwork(
            dim=query_dim,
            hidden_dim=ffn_hidden_dim,
            dropout=dropout
        )
        
        # Layer Norm for FFN
        self.norm2 = nn.LayerNorm(query_dim)
        
        print(colored(f"CrossAttentionBlock: query_dim={query_dim}, key_dim={key_dim}, num_heads={num_heads}, ffn_hidden={ffn_hidden_dim or 4*query_dim}", "cyan", attrs=['bold']))
    
    def forward(self, query, key_value):
        """
        Args:
            query: [B, C, T] - decoder hidden state
            key_value: [B, C_kv, T_kv] - speaker embedding sequence
        
        Returns:
            output: [B, C, T] - processed output
        """
        # Pre-Norm: Normalize -> Cross-Attention -> Residual
        # Layer norm: transpose to [B, T, C], normalize, transpose back
        normed = self.norm1(query.transpose(1, 2)).transpose(1, 2)
        attn_out = self.cross_attn(normed, key_value)
        # Residual connection
        x = query + attn_out
        
        # Pre-Norm: Normalize -> FFN -> Residual
        normed = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        ffn_out = self.ffn(normed)
        # Residual connection
        x = x + ffn_out
        
        return x
