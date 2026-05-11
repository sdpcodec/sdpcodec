import math
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import CrossAttentionBlock, _as_plain_dict, build_res_temporal


class ChannelNormalization(nn.Module):
    def __init__(self, num_channels: int, affine: bool = True):
        super().__init__()
        self.eps = 1e-5
        if affine:
            self.gain = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter("gain", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps)
        x = (x - mean) / std
        if self.gain is not None and self.bias is not None:
            x = x * self.gain + self.bias
        return x


class TimeGlobalNormalization(nn.Module):
    def __init__(self, num_channels: int, affine: bool = True):
        super().__init__()
        self.eps = 1e-5
        if affine:
            self.gain = nn.Parameter(torch.ones(1, 1, 1, num_channels))
            self.bias = nn.Parameter(torch.zeros(1, 1, 1, num_channels))
        else:
            self.register_parameter("gain", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = torch.sqrt(torch.var(x, dim=(2, 3), keepdim=True, unbiased=False) + self.eps)
        x = (x - mean) / std
        if self.gain is not None and self.bias is not None:
            x = x * self.gain + self.bias
        return x


class BandwiseLayerNorm(nn.Module):
    def __init__(self, nband: int, feature_dim: int):
        super().__init__()
        self.nband = nband
        self.eps = 1e-5
        self.gain = nn.Parameter(torch.ones(1, nband, feature_dim, 1))
        self.bias = nn.Parameter(torch.zeros(1, nband, feature_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-2, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=-2, keepdim=True, unbiased=False) + self.eps)
        batch_nband, channels, seq_len = x.shape
        batch_size = batch_nband // self.nband
        x = x.view(batch_size, self.nband, channels, seq_len)
        mean = mean.view(batch_size, self.nband, 1, seq_len)
        std = std.view(batch_size, self.nband, 1, seq_len)
        x = self.gain * ((x - mean) / std) + self.bias
        return x.view(batch_nband, channels, seq_len)


class GRN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class LinearGroup(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_groups: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_groups, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[-1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.einsum("...gh,gkh->...gk", x, self.weight)
        if self.bias is not None:
            y = y + self.bias[None, ...]
        return y


def build_band_widths(sr: int, n_fft: int) -> list[int]:
    fft_reso = float(sr) / float(n_fft)
    bw_250 = max(1, int(math.floor(250.0 / fft_reso)))
    bw_500 = max(1, int(math.floor(500.0 / fft_reso)))
    bw_1k = max(1, int(math.floor(1000.0 / fft_reso)))
    widths = [bw_250] * 12 + [bw_500] * 8
    total_bins = n_fft // 2 + 1
    used_bins = sum(widths)
    while used_bins + bw_1k < total_bins:
        widths.append(bw_1k)
        used_bins += bw_1k
    if used_bins < total_bins:
        widths.append(total_bins - used_bins)
    return widths


class BandSplit(nn.Module):
    def __init__(self, sr: int, n_fft: int, feature_dim: int):
        super().__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.band_width = build_band_widths(sr, n_fft)
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    ChannelNormalization(width * 2 + 1),
                    nn.Conv1d(width * 2 + 1, feature_dim, kernel_size=1),
                )
                for width in self.band_width
            ]
        )

    def get_nband(self) -> int:
        return len(self.band_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        band_idx = 0
        for idx, width in enumerate(self.band_width):
            current = x[:, band_idx: band_idx + width].transpose(1, 2).contiguous()
            power = torch.sqrt(torch.norm(current, dim=-1, keepdim=True).pow(2).sum(dim=-2, keepdim=True) + self.eps)
            batch_size, seq_len, _, _ = current.shape
            current = (current / power).view(batch_size, seq_len, -1)
            current = torch.cat([current, torch.log(power.squeeze(-1))], dim=-1).transpose(-2, -1).contiguous()
            outputs.append(self.encoder[idx](current))
            band_idx += width
        return torch.stack(outputs, dim=1)


class BandMerge(nn.Module):
    def __init__(self, sr: int, n_fft: int, feature_dim: int):
        super().__init__()
        self.band_width = build_band_widths(sr, n_fft)
        self.mag_decoder = nn.ModuleList(
            [
                nn.Sequential(
                    ChannelNormalization(feature_dim),
                    nn.Conv1d(feature_dim, feature_dim * 2, kernel_size=1),
                    nn.GELU(),
                    nn.Conv1d(feature_dim * 2, width, kernel_size=1),
                )
                for width in self.band_width
            ]
        )
        self.phase_decoder = nn.ModuleList(
            [
                nn.Sequential(
                    ChannelNormalization(feature_dim),
                    nn.Conv1d(feature_dim, feature_dim * 2, kernel_size=1),
                    nn.GELU(),
                    nn.Conv1d(feature_dim * 2, width * 2, kernel_size=1),
                )
                for width in self.band_width
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mags = []
        phases = []
        for idx in range(len(self.band_width)):
            mag = torch.exp(self.mag_decoder[idx](x[:, idx].contiguous()))
            comp = self.phase_decoder[idx](x[:, idx].contiguous())
            real, imag = comp.chunk(2, dim=1)
            phase = torch.atan2(imag, real)
            mags.append(mag)
            phases.append(phase)
        return torch.cat(mags, dim=1), torch.cat(phases, dim=1)


class BandShuffler(nn.Module):
    def __init__(self, nband: int, input_size: int, squeeze_size: int = 64, f_kernel_size: int = 3, f_conv_groups: int = 8):
        super().__init__()
        self.fconv1 = nn.Sequential(
            ChannelNormalization(input_size),
            nn.Conv1d(input_size, input_size, kernel_size=f_kernel_size, groups=f_conv_groups, padding="same"),
            nn.PReLU(input_size),
        )
        self.fconv2 = nn.Sequential(
            ChannelNormalization(input_size),
            nn.Conv1d(input_size, input_size, kernel_size=f_kernel_size, groups=f_conv_groups, padding="same"),
            nn.PReLU(input_size),
        )
        self.squeeze = nn.Sequential(nn.Conv1d(input_size, squeeze_size, kernel_size=1), nn.SiLU())
        self.unsqueeze = nn.Sequential(nn.Conv1d(squeeze_size, input_size, kernel_size=1), nn.SiLU())
        self.full = LinearGroup(nband, nband, squeeze_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, nband = x.shape
        x = x.view(batch_size * seq_len, channels, nband)
        residual = x
        x = residual + self.fconv1(x)
        residual = x
        x = self.squeeze(x)
        x = self.full(x)
        x = self.unsqueeze(x)
        x = residual + x
        residual = x
        x = residual + self.fconv2(x)
        return x.view(batch_size, seq_len, channels, nband)


class TimeResRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(input_size) if causal else TimeGlobalNormalization(input_size)
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=not causal)
        self.proj = nn.Linear(hidden_size * (2 if not causal else 1), input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, nband, channels, seq_len = x.shape
        y = x.transpose(-2, -1).contiguous()
        y = self.norm(y)
        y = y.view(batch_size * nband, seq_len, channels)
        y, _ = self.rnn(self.dropout(y))
        y = self.proj(y).transpose(-2, -1).contiguous().view_as(x)
        return x + y


class FreqResRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=not causal)
        self.proj = nn.Linear(hidden_size * (2 if not causal else 1), input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, nband = x.shape
        y = x.transpose(-2, -1).contiguous()
        y = self.norm(y)
        y = y.view(batch_size * seq_len, nband, channels)
        y, _ = self.rnn(self.dropout(y))
        y = self.proj(y).transpose(-2, -1).contiguous().view_as(x)
        return x + y


class BandWiseTimeModule(nn.Module):
    def __init__(self, nband: int, nrep: int, input_channel: int, hidden_channel: int, kernel_size: int):
        super().__init__()
        self.nband = nband
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_channel, input_channel, kernel_size, padding="same", groups=input_channel),
                    BandwiseLayerNorm(nband, input_channel),
                    nn.Conv1d(input_channel, hidden_channel, kernel_size=1),
                    nn.GELU(),
                    GRN(hidden_channel),
                    nn.Conv1d(hidden_channel, input_channel, kernel_size=1),
                )
                for _ in range(nrep)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, nband, channels, seq_len = x.shape
        y = x.view(batch_size * nband, channels, seq_len)
        for block in self.blocks:
            y = y + block(y)
        return y.view(batch_size, nband, channels, seq_len)


class VocModule(nn.Module):
    def __init__(
        self,
        nrep: int,
        nband: int,
        input_channel: int,
        squeeze_size: int,
        hidden_channel: int,
        kernel_size: int,
        causal: bool = False,
        time_type: str = "convnext_v2",
        freq_type: str = "shuffler",
    ):
        super().__init__()
        freq_type = freq_type.lower()
        time_type = time_type.lower()
        if freq_type == "shuffler":
            self.bandnet = BandShuffler(nband, input_channel, squeeze_size)
        elif freq_type == "lstm":
            self.bandnet = FreqResRNN(input_channel, hidden_channel, causal=False)
        elif freq_type in {"none", "identity"}:
            self.bandnet = nn.Identity()
        else:
            raise ValueError(f"Unsupported RNDVoC freq_type: {freq_type}")

        if time_type == "convnext_v2":
            self.timenet = BandWiseTimeModule(nband, nrep, input_channel, hidden_channel, kernel_size)
        elif time_type == "lstm":
            self.timenet = TimeResRNN(input_channel, hidden_channel, causal=causal)
        elif time_type in {"none", "identity"}:
            self.timenet = nn.Identity()
        else:
            raise ValueError(f"Unsupported RNDVoC time_type: {time_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bandnet(x)
        x = x.transpose(1, 3).contiguous()
        x = self.timenet(x)
        return x.transpose(1, 3).contiguous()


class RNDLatentDecoder(nn.Module):
    """
    RNDVoC-inspired latent-to-waveform decoder.

    The original RNDVoC expects log-mel inputs. Here we keep the band-split /
    time-frequency refinement stack, but replace mel-specific initialization
    with a learned latent-to-magnitude projection so codec latents can be used
    directly.
    """

    def __init__(
        self,
        latent_dim: int,
        sampling_rate: int,
        hop_size: int,
        n_fft: Optional[int] = None,
        win_size: Optional[int] = None,
        input_channel: int = 256,
        hidden_channel: int = 256,
        squeeze_size: int = 64,
        null_nstage: int = 6,
        nrep: int = 2,
        kernel_size: int = 7,
        causal: bool = False,
        use_rnd: bool = True,
        time_type: str = "convnext_v2",
        freq_type: str = "shuffler",
        pre_temporal: Optional[Any] = None,
        speaker_condition: bool = False,
        condition_dim: int = 1024,
        f0_condition: bool = False,
        f0_start_layer: int = 0,
        f0_end_layer: Optional[int] = None,
        f0_every: int = 1,
        f0_speaker_condition: bool = False,
        use_stage_speaker_film: bool = True,
        f0_stage_dims: Optional[Sequence[int]] = None,
        leaky_relu_params: Optional[dict] = None,
        use_mhca: bool = False,
        mhca_num_heads: int = 8,
        mhca_dropout: float = 0.1,
        mhca_key_dim: int = 128,
        mhca_use_sdpa: Optional[bool] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.sampling_rate = int(sampling_rate)
        self.hop_size = int(hop_size)
        self.n_fft = int(n_fft) if n_fft is not None else int(self.hop_size * 4)
        self.win_size = int(win_size) if win_size is not None else int(self.n_fft)
        self.input_channel = int(input_channel)
        self.use_rnd = bool(use_rnd)
        self.speaker_condition = bool(speaker_condition)
        self.f0_condition = bool(f0_condition)
        self.f0_start_layer = int(f0_start_layer)
        self.f0_end_layer = f0_end_layer
        self.f0_every = max(1, int(f0_every))
        self.f0_speaker_condition = bool(f0_speaker_condition)
        self.use_stage_speaker_film = bool(use_stage_speaker_film)
        self.use_mhca = bool(use_mhca)
        pre_temporal_cfg = _as_plain_dict(pre_temporal)
        self.pre_temporal_use = bool(pre_temporal_cfg.get("use", False))
        self.pre_temporal_type = str(
            pre_temporal_cfg.get("type", pre_temporal_cfg.get("backbone", "lstm"))
        ).lower()
        self.pre_temporal_num_layers = int(pre_temporal_cfg.get("num_layers", 2))
        self.pre_temporal_bidirectional = bool(pre_temporal_cfg.get("bidirectional", False))
        self.pre_temporal_mamba = _as_plain_dict(pre_temporal_cfg.get("mamba", {}))
        self.eps = torch.finfo(torch.float32).eps
        self.f0_stage_dims = None if f0_stage_dims is None else [int(dim) for dim in f0_stage_dims]
        negative_slope = 0.1 if leaky_relu_params is None else leaky_relu_params.get("negative_slope", 0.1)

        self.pre_temporal = None
        if self.pre_temporal_use:
            if self.pre_temporal_type not in {"lstm", "mamba"}:
                raise ValueError(
                    f"Unsupported RNDVoC pre-temporal type: {self.pre_temporal_type!r}. "
                    "Expected 'lstm' or 'mamba'."
                )
            self.pre_temporal = build_res_temporal(
                latent_dim,
                self.pre_temporal_type,
                num_layers=self.pre_temporal_num_layers,
                bidirectional=self.pre_temporal_bidirectional,
                mamba=self.pre_temporal_mamba,
            )

        self.input_adapter = nn.Sequential(
            ChannelNormalization(latent_dim),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
        )
        self.condition_fuser = None
        if self.speaker_condition:
            self.condition_fuser = nn.Sequential(
                nn.Conv1d(latent_dim + condition_dim, latent_dim, kernel_size=1),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.Conv1d(latent_dim, latent_dim, kernel_size=1),
            )
        self.f0_block_enabled = [False] * null_nstage
        self.f0_stage_fusers = None
        if self.f0_condition:
            if self.f0_stage_dims is None or len(self.f0_stage_dims) == 0:
                raise ValueError("RNDLatentDecoder requires f0_stage_dims when f0_condition=True.")
            self.f0_start_layer = max(0, min(int(self.f0_start_layer), null_nstage))
            if self.f0_end_layer is None:
                self.f0_end_layer = null_nstage
            else:
                self.f0_end_layer = max(self.f0_start_layer, min(int(self.f0_end_layer), null_nstage))
            for idx in range(null_nstage):
                self.f0_block_enabled[idx] = (
                    self.f0_start_layer <= idx < self.f0_end_layer
                    and ((idx - self.f0_start_layer) % self.f0_every == 0)
                )
            if len(self.f0_stage_dims) < null_nstage:
                padded_dims = list(self.f0_stage_dims)
                padded_dims.extend([padded_dims[-1]] * (null_nstage - len(padded_dims)))
                self.f0_stage_dims = padded_dims
            else:
                self.f0_stage_dims = list(self.f0_stage_dims[:null_nstage])
            self.f0_stage_fusers = nn.ModuleList(
                [
                    (
                        nn.Sequential(
                            nn.Conv1d(self.input_channel + self.f0_stage_dims[idx], self.input_channel, kernel_size=1),
                            nn.LeakyReLU(negative_slope=negative_slope),
                            nn.Conv1d(self.input_channel, self.input_channel, kernel_size=1),
                        )
                        if self.f0_block_enabled[idx] else nn.Identity()
                    )
                    for idx in range(null_nstage)
                ]
            )
        else:
            self.f0_start_layer = 0
            self.f0_end_layer = 0
        self.f0_speaker_film = None
        if self.speaker_condition and self.use_stage_speaker_film:
            self.f0_speaker_film = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(condition_dim, self.input_channel * 2),
                        nn.LeakyReLU(negative_slope=negative_slope),
                        nn.Linear(self.input_channel * 2, self.input_channel * 2),
                    )
                    for _ in range(null_nstage)
                ]
            )

        self.latent_to_mag = nn.Sequential(
            ChannelNormalization(latent_dim),
            nn.Conv1d(latent_dim, latent_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(latent_dim * 2, self.n_fft // 2 + 1, kernel_size=1),
        )

        self.null_enc = BandSplit(sr=self.sampling_rate, n_fft=self.n_fft, feature_dim=self.input_channel)
        self.null_dec = BandMerge(sr=self.sampling_rate, n_fft=self.n_fft, feature_dim=self.input_channel)
        self.null_nband = self.null_enc.get_nband()
        self.null_module_list = nn.ModuleList(
            [
                VocModule(
                    nrep=nrep,
                    nband=self.null_nband,
                    input_channel=self.input_channel,
                    squeeze_size=squeeze_size,
                    hidden_channel=hidden_channel,
                    kernel_size=kernel_size,
                    causal=causal,
                    time_type=time_type,
                    freq_type=freq_type,
                )
                for _ in range(null_nstage)
            ]
        )
        if self.use_mhca:
            self.mhca_list = nn.ModuleList(
                [
                    CrossAttentionBlock(
                        query_dim=self.input_channel,
                        key_dim=mhca_key_dim,
                        num_heads=mhca_num_heads,
                        dropout=mhca_dropout,
                        use_sdpa=mhca_use_sdpa,
                    )
                    for _ in range(null_nstage)
                ]
            )
        else:
            self.mhca_list = None
        self.alpha = nn.Parameter(torch.ones(1, 1, self.input_channel, self.null_nband))
        self.register_buffer("window", torch.hann_window(self.win_size), persistent=False)
        print(
            "RNDLatentDecoder init: "
            f"latent_dim={self.latent_dim}, sr={self.sampling_rate}, hop_size={self.hop_size}, "
            f"n_fft={self.n_fft}, win_size={self.win_size}, input_channel={self.input_channel}, "
            f"hidden_channel={hidden_channel}, null_nstage={null_nstage}, nband={self.null_nband}, "
            f"speaker_condition={self.speaker_condition}, f0_condition={self.f0_condition}, "
            f"f0_layer_range=({self.f0_start_layer}, {self.f0_end_layer}), f0_every={self.f0_every}, "
            f"f0_speaker_condition={self.f0_speaker_condition}, "
            f"use_stage_speaker_film={self.use_stage_speaker_film}, "
            f"use_mhca={self.use_mhca}, use_rnd={self.use_rnd}, "
            f"pre_temporal_use={self.pre_temporal_use}, pre_temporal_type={self.pre_temporal_type}, "
            f"time_type={time_type}, freq_type={freq_type}"
        )
        if self.pre_temporal is not None:
            print(
                f"RNDLatentDecoder pre-temporal enabled: "
                f"type={self.pre_temporal_type}, layers={self.pre_temporal_num_layers}, "
                f"bidirectional={self.pre_temporal_bidirectional}"
            )
        if self.f0_stage_fusers is not None:
            print(f"RNDLatentDecoder f0 stage-wise fusion enabled for {len(self.f0_stage_fusers)} stages.")
            print(f"RNDLatentDecoder f0_stage_dims={self.f0_stage_dims}")
            print(f"RNDLatentDecoder f0 enabled on stages {[idx for idx, enabled in enumerate(self.f0_block_enabled) if enabled]}")
        if self.f0_speaker_film is not None:
            print(f"RNDLatentDecoder 2-stage conditioning enabled for {len(self.f0_speaker_film)} stages: f0 conv -> speaker FiLM.")
        self.reset_parameters()

    def _build_stage_f0_conditions(
        self,
        f0_conds: Optional[Sequence[Optional[torch.Tensor]]],
    ) -> list[Optional[torch.Tensor]]:
        if not self.f0_condition or self.f0_stage_fusers is None:
            return [None] * len(self.null_module_list)

        if f0_conds is None:
            return [None] * len(self.null_module_list)

        if isinstance(f0_conds, torch.Tensor):
            valid = [f0_conds]
        else:
            valid = [tensor for tensor in f0_conds if isinstance(tensor, torch.Tensor)]

        if not valid:
            return [None] * len(self.null_module_list)

        if len(valid) >= len(self.null_module_list):
            return valid[:len(self.null_module_list)]

        padded = list(valid)
        while len(padded) < len(self.null_module_list):
            padded.append(valid[-1])
        return padded

    def _apply_conditions(
        self,
        x: torch.Tensor,
        spk_cond: Optional[torch.Tensor] = None,
        f0_conds: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        if self.condition_fuser is None:
            return x

        cond_inputs = [x]
        if self.speaker_condition and spk_cond is not None:
            cond_inputs.append(spk_cond.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        if len(cond_inputs) == 1:
            return x
        return self.condition_fuser(torch.cat(cond_inputs, dim=1))

    def _apply_mhca(self, x: torch.Tensor, x_quantized: Optional[torch.Tensor], block_idx: int) -> torch.Tensor:
        if not self.use_mhca or self.mhca_list is None or x_quantized is None:
            return x

        batch_size, seq_len, channels, nband = x.shape
        query = x.permute(0, 3, 2, 1).contiguous().view(batch_size * nband, channels, seq_len)
        key_value = x_quantized.unsqueeze(1).expand(-1, nband, -1, -1).contiguous()
        key_value = key_value.view(batch_size * nband, x_quantized.shape[1], x_quantized.shape[2])
        query = self.mhca_list[block_idx](query, key_value)
        return query.view(batch_size, nband, channels, seq_len).permute(0, 3, 2, 1).contiguous()

    def _apply_stage_f0(
        self,
        x: torch.Tensor,
        f0_stage_cond: Optional[torch.Tensor],
        block_idx: int,
    ) -> torch.Tensor:
        if (
            not self.f0_condition
            or self.f0_stage_fusers is None
            or f0_stage_cond is None
            or not self.f0_block_enabled[block_idx]
        ):
            return x

        if f0_stage_cond.dim() == 2:
            f0_stage_cond = f0_stage_cond.unsqueeze(1)

        batch_size, seq_len, channels, nband = x.shape
        if f0_stage_cond.shape[-1] != seq_len:
            f0_stage_cond = F.interpolate(f0_stage_cond, size=seq_len, mode="nearest")

        x_band = x.permute(0, 3, 2, 1).contiguous().view(batch_size * nband, channels, seq_len)
        f0_band = f0_stage_cond.unsqueeze(1).expand(-1, nband, -1, -1).contiguous()
        f0_band = f0_band.view(batch_size * nband, f0_stage_cond.shape[1], seq_len)
        x_band = self.f0_stage_fusers[block_idx](torch.cat([x_band, f0_band], dim=1))
        return x_band.view(batch_size, nband, channels, seq_len).permute(0, 3, 2, 1).contiguous()

    def _apply_stage_speaker_film(
        self,
        x: torch.Tensor,
        spk_cond: Optional[torch.Tensor],
        block_idx: int,
    ) -> torch.Tensor:
        if (
            not self.speaker_condition
            or not self.use_stage_speaker_film
            or self.f0_speaker_film is None
            or spk_cond is None
        ):
            return x

        batch_size, seq_len, channels, nband = x.shape
        gamma_beta = self.f0_speaker_film[block_idx](spk_cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.view(batch_size, 1, channels, 1).expand(-1, nband, -1, seq_len)
        beta = beta.view(batch_size, 1, channels, 1).expand(-1, nband, -1, seq_len)

        x_band = x.permute(0, 3, 2, 1).contiguous()
        x_band = x_band * (1.0 + gamma) + beta
        return x_band.permute(0, 3, 2, 1).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        spk_cond: Optional[torch.Tensor] = None,
        f0_conds: Optional[Sequence[Optional[torch.Tensor]]] = None,
        x_quantized: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_frames = int(x.shape[-1])
        if self.pre_temporal is not None:
            x = self.pre_temporal(x)
        x = x + self.input_adapter(x)
        x = self._apply_conditions(x, spk_cond=spk_cond, f0_conds=f0_conds)
        stage_f0_conds = self._build_stage_f0_conditions(f0_conds)

        init_mag = F.softplus(self.latent_to_mag(x)) + self.eps
        init_spec = torch.stack([init_mag, torch.zeros_like(init_mag)], dim=-1)

        null_x = self.null_enc(init_spec).transpose(1, 3).contiguous()
        x = null_x
        for idx, module in enumerate(self.null_module_list):
            x = self._apply_mhca(x, x_quantized, idx)
            x = self._apply_stage_f0(x, stage_f0_conds[idx], idx)
            x = self._apply_stage_speaker_film(x, spk_cond, idx)
            x = module(x)

        if self.use_rnd:
            refined_input = (self.alpha * null_x + x).transpose(1, 3).contiguous()
            residual_mag, phase = self.null_dec(refined_input)
            out_mag = init_mag + residual_mag
        else:
            refined_input = x.transpose(1, 3).contiguous()
            out_mag, phase = self.null_dec(refined_input)

        real = out_mag * torch.cos(phase)
        imag = out_mag * torch.sin(phase)
        spec = torch.complex(real, imag)
        wav = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window.to(spec.device),
            length=input_frames * self.hop_size,
        )
        return wav.unsqueeze(1)

    def reset_parameters(self):
        def _init(module: nn.Module):
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                if isinstance(getattr(module, "weight", None), nn.parameter.UninitializedParameter):
                    return
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_init)
