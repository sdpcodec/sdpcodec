from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import warnings
import hashlib
import json
import os
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.nn.utils import spectral_norm, weight_norm

from module.mstft import NormConv2d, get_2d_padding
from module.san import SANConv2d


def _expand_per_scale(value: Any, num_scales: int, key: str) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == num_scales:
            return list(value)
        if len(value) == 1:
            return list(value) * num_scales
        raise ValueError(f"{key} must have length 1 or {num_scales}, but got {len(value)}.")
    return [value] * num_scales


def _default_filterbank_cache_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "etc" / "filterbank_cache"


def _resolve_filterbank_cache_dir() -> Path:
    raw = os.environ.get("BIGCODEC_FILTERBANK_CACHE_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return _default_filterbank_cache_dir()


def _cache_file(kind: str, params: dict[str, Any]) -> Path:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    cache_dir = _resolve_filterbank_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{kind}_{digest}.npz"


def _maybe_norm_2d(module: nn.Module, norm: str) -> nn.Module:
    norm = str(norm).lower()
    if norm == "weight_norm":
        return weight_norm(module)
    if norm == "spectral_norm":
        return spectral_norm(module)
    if norm in {"none", "identity", "false"}:
        return module
    raise ValueError(f"Unsupported norm type: {norm}")


class FixedComplexFilterbank(nn.Module):
    def __init__(
        self,
        basis: np.ndarray,
        hop_length: int,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        if basis.ndim != 2:
            raise ValueError(f"Expected 2D complex basis, but got shape={basis.shape}.")

        kernel = np.flip(np.conj(basis), axis=-1).copy()
        self.num_bins = int(kernel.shape[0])
        self.kernel_size = int(kernel.shape[1])
        self.hop_length = int(hop_length)
        self.pad_mode = str(pad_mode)
        self.register_buffer(
            "kernel_real",
            torch.from_numpy(np.real(kernel)).float().unsqueeze(1),
            persistent=False,
        )
        self.register_buffer(
            "kernel_imag",
            torch.from_numpy(np.imag(kernel)).float().unsqueeze(1),
            persistent=False,
        )

    @classmethod
    def from_cqt(
        cls,
        sample_rate: int,
        hop_length: int,
        fmin: float,
        n_bins: int,
        bins_per_octave: int = 12,
        filter_scale: float = 1.0,
        window: str = "hann",
        pad_fft: bool = True,
        pad_mode: str = "reflect",
    ) -> "FixedComplexFilterbank":
        cache_params = {
            "sample_rate": int(sample_rate),
            "hop_length": int(hop_length),
            "fmin": float(fmin),
            "n_bins": int(n_bins),
            "bins_per_octave": int(bins_per_octave),
            "filter_scale": float(filter_scale),
            "window": str(window),
            "pad_fft": bool(pad_fft),
            "pad_mode": str(pad_mode),
        }
        cache_path = _cache_file("cqt", cache_params)
        if cache_path.is_file():
            cached = np.load(cache_path, allow_pickle=False)
            basis = cached["basis"]
            print(
                f"[filterbank-cache] loaded CQT basis "
                f"sr={int(sample_rate)} hop={int(hop_length)} bins={int(n_bins)} "
                f"bpo={int(bins_per_octave)} from {cache_path}"
            )
        else:
            print(
                f"[filterbank-cache] building CQT basis "
                f"sr={int(sample_rate)} hop={int(hop_length)} bins={int(n_bins)} "
                f"bpo={int(bins_per_octave)}"
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                basis, _ = librosa.filters.constant_q(
                    sr=int(sample_rate),
                    fmin=float(fmin),
                    n_bins=int(n_bins),
                    bins_per_octave=int(bins_per_octave),
                    filter_scale=float(filter_scale),
                    window=window,
                    pad_fft=bool(pad_fft),
                )
            tmp_path = cache_path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp_path, basis=basis)
            os.replace(tmp_path, cache_path)
            print(f"[filterbank-cache] saved CQT basis to {cache_path}")
        return cls(basis=basis, hop_length=hop_length, pad_mode=pad_mode)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"Expected waveform tensor shaped [B, T], but got {tuple(x.shape)}.")

        x = x.float().unsqueeze(1)
        pad = self.kernel_size // 2
        pad_mode = self.pad_mode
        if pad_mode == "reflect" and x.shape[-1] <= pad:
            pad_mode = "constant"
        x = F.pad(x, (pad, pad), mode=pad_mode)
        real = F.conv1d(x, self.kernel_real, stride=self.hop_length)
        imag = F.conv1d(x, self.kernel_imag, stride=self.hop_length)
        return real, imag


class ComplexSubbandDiscriminator(nn.Module):
    def __init__(
        self,
        frontend: FixedComplexFilterbank,
        num_splits: int,
        split_bins: int,
        in_channels: int = 1,
        out_channels: int = 1,
        filters: int = 32,
        max_filters: int = 1024,
        filters_scale: int = 1,
        kernel_size: tuple[int, int] = (3, 9),
        dilations: Sequence[int] = (1, 2, 4),
        stride: tuple[int, int] = (1, 2),
        norm: str = "weight_norm",
        activation: str = "LeakyReLU",
        activation_params: dict | None = None,
        use_san: bool = False,
    ):
        super().__init__()
        activation_params = activation_params or {"negative_slope": 0.1}
        self.frontend = frontend
        self.num_splits = max(1, int(num_splits))
        self.split_bins = max(1, int(split_bins))
        self.use_san = bool(use_san)
        self.supports_san_training = self.use_san
        self.activation = getattr(torch.nn, activation)(**activation_params)

        in_spec_channels = int(in_channels) * 2
        filters = int(filters)
        max_filters = int(max_filters)
        filters_scale = int(filters_scale)
        kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        stride = (int(stride[0]), int(stride[1]))

        self.conv_pres = nn.ModuleList(
            [
                NormConv2d(
                    in_spec_channels,
                    in_spec_channels,
                    kernel_size=kernel_size,
                    padding=get_2d_padding(kernel_size),
                    norm=norm,
                )
                for _ in range(self.num_splits)
            ]
        )

        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(
                in_spec_channels,
                filters,
                kernel_size=kernel_size,
                padding=get_2d_padding(kernel_size),
                norm=norm,
            )
        )
        in_chs = min(filters_scale * filters, max_filters)
        for idx, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (idx + 1)) * filters, max_filters)
            self.convs.append(
                NormConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(int(dilation), 1),
                    padding=get_2d_padding(kernel_size, (int(dilation), 1)),
                    norm=norm,
                )
            )
            in_chs = out_chs
        out_chs = min((filters_scale ** (len(tuple(dilations)) + 1)) * filters, max_filters)
        self.convs.append(
            NormConv2d(
                in_chs,
                out_chs,
                kernel_size=(kernel_size[0], kernel_size[0]),
                padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                norm=norm,
            )
        )
        if self.use_san:
            self.conv_post = SANConv2d(
                out_chs,
                int(out_channels),
                kernel_size=(kernel_size[0], kernel_size[0]),
                padding=get_2d_padding((kernel_size[0], kernel_size[0])),
            )
        else:
            self.conv_post = NormConv2d(
                out_chs,
                int(out_channels),
                kernel_size=(kernel_size[0], kernel_size[0]),
                padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                norm=norm,
            )

    def _split_bands(self, z: torch.Tensor) -> list[torch.Tensor]:
        latent_z = []
        for idx in range(self.num_splits):
            start = idx * self.split_bins
            end = min((idx + 1) * self.split_bins, z.shape[-1])
            if end <= start:
                break
            latent_z.append(self.conv_pres[idx](z[:, :, :, start:end]))
        return latent_z

    def forward(self, x: torch.Tensor, flg_train: bool = False) -> list[torch.Tensor | list[torch.Tensor]]:
        real, imag = self.frontend(x)
        z = torch.cat([real.unsqueeze(1), imag.unsqueeze(1)], dim=1)
        z = torch.permute(z, (0, 1, 3, 2))

        split_latents = self._split_bands(z)
        if not split_latents:
            raise RuntimeError("No subband slices were produced for the configured discriminator.")
        latent_z = torch.cat(split_latents, dim=-1)

        fmap: list[torch.Tensor | list[torch.Tensor]] = []
        for layer in self.convs:
            latent_z = layer(latent_z)
            latent_z = self.activation(latent_z)
            fmap.append(latent_z)

        if self.use_san and flg_train:
            z_fun, z_dir = self.conv_post(latent_z, flg_train=True)
            logits = [torch.flatten(z_fun, 1, -1), torch.flatten(z_dir, 1, -1)]
        else:
            latent_z = self.conv_post(latent_z)
            logits = torch.flatten(latent_z, 1, -1)
        fmap.append(logits)
        return fmap

    def normalize_san_weights(self) -> None:
        if self.use_san and hasattr(self.conv_post, "normalize_weight"):
            self.conv_post.normalize_weight()


class NNAudioCQTDiscriminator(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        n_octaves: int,
        bins_per_octave: int,
        fmin: float = 32.7,
        filter_scale: float = 1.0,
        in_channels: int = 1,
        out_channels: int = 1,
        filters: int = 128,
        max_filters: int = 1024,
        filters_scale: int = 1,
        kernel_size: tuple[int, int] = (3, 9),
        dilations: Sequence[int] = (1, 2, 4),
        stride: tuple[int, int] = (1, 2),
        norm: str = "weight_norm",
        activation: str = "LeakyReLU",
        activation_params: dict | None = None,
        use_san: bool = False,
        resample_factor: int = 2,
        normalize_volume: bool = False,
        cqt_pad_mode: str = "constant",
        cqt_window: str = "hann",
        cqt_earlydownsample: bool = True,
        cqt_verbose: bool = True,
    ):
        super().__init__()
        activation_params = activation_params or {"negative_slope": 0.1}
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.n_octaves = int(n_octaves)
        self.bins_per_octave = int(bins_per_octave)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.filters = int(filters)
        self.max_filters = int(max_filters)
        self.filters_scale = int(filters_scale)
        self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        self.dilations = tuple(int(v) for v in dilations)
        self.stride = (int(stride[0]), int(stride[1]))
        self.use_san = bool(use_san)
        self.supports_san_training = self.use_san
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.resample_factor = int(resample_factor)
        self.normalize_volume = bool(normalize_volume)

        try:
            from nnAudio import features
        except ImportError as exc:
            raise ImportError(
                "nnAudio is required for the CQT discriminator. "
                "Install it with `pip install nnAudio==0.3.4` in the sdpcodec environment."
            ) from exc

        self.cqt_transform = features.cqt.CQT2010v2(
            sr=self.sample_rate * self.resample_factor,
            hop_length=self.hop_length,
            fmin=float(fmin),
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            filter_scale=float(filter_scale),
            window=str(cqt_window),
            pad_mode=str(cqt_pad_mode),
            earlydownsample=bool(cqt_earlydownsample),
            output_format="Complex",
            verbose=bool(cqt_verbose),
        )

        self.conv_pres = nn.ModuleList(
            [
                nn.Conv2d(
                    self.in_channels * 2,
                    self.in_channels * 2,
                    kernel_size=self.kernel_size,
                    padding=get_2d_padding(self.kernel_size),
                )
                for _ in range(self.n_octaves)
            ]
        )

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    self.in_channels * 2,
                    self.filters,
                    kernel_size=self.kernel_size,
                    padding=get_2d_padding(self.kernel_size),
                )
            ]
        )

        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for idx, dilation in enumerate(self.dilations):
            out_chs = min((self.filters_scale ** (idx + 1)) * self.filters, self.max_filters)
            self.convs.append(
                _maybe_norm_2d(
                    nn.Conv2d(
                        in_chs,
                        out_chs,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=(int(dilation), 1),
                        padding=get_2d_padding(self.kernel_size, (int(dilation), 1)),
                    ),
                    norm=norm,
                )
            )
            in_chs = out_chs

        out_chs = min((self.filters_scale ** (len(self.dilations) + 1)) * self.filters, self.max_filters)
        self.convs.append(
            _maybe_norm_2d(
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                    padding=get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
                ),
                norm=norm,
            )
        )

        if self.use_san:
            self.conv_post = SANConv2d(
                out_chs,
                self.out_channels,
                kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                padding=get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
            )
        else:
            self.conv_post = _maybe_norm_2d(
                nn.Conv2d(
                    out_chs,
                    self.out_channels,
                    kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                    padding=get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
                ),
                norm=norm,
            )

        self.resample = T.Resample(
            orig_freq=self.sample_rate,
            new_freq=self.sample_rate * self.resample_factor,
        )

        if self.normalize_volume:
            print(
                "[INFO] cqt_discriminator.normalize_volume=True; "
                "applying DC-offset removal and peak normalization before CQTD."
            )

    def forward(self, x: torch.Tensor, flg_train: bool = False) -> list[torch.Tensor | list[torch.Tensor]]:
        if x.dim() == 3:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"Expected waveform tensor shaped [B, T], but got {tuple(x.shape)}.")

        if self.normalize_volume:
            x = x - x.mean(dim=-1, keepdim=True)
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

        x = self.resample(x)
        z = self.cqt_transform(x)
        if z.dim() != 4 or z.shape[-1] != 2:
            raise RuntimeError(
                "Expected nnAudio CQT output shaped [B, bins, frames, 2], "
                f"but got {tuple(z.shape)}."
            )

        z_real = z[:, :, :, 0].unsqueeze(1)
        z_imag = z[:, :, :, 1].unsqueeze(1)
        z = torch.cat([z_real, z_imag], dim=1)
        z = torch.permute(z, (0, 1, 3, 2))

        latent_z = []
        for idx in range(self.n_octaves):
            start = idx * self.bins_per_octave
            end = (idx + 1) * self.bins_per_octave
            latent_z.append(self.conv_pres[idx](z[:, :, :, start:end]))
        latent_z = torch.cat(latent_z, dim=-1)

        fmap: list[torch.Tensor | list[torch.Tensor]] = []
        for layer in self.convs:
            latent_z = layer(latent_z)
            latent_z = self.activation(latent_z)
            fmap.append(latent_z)

        if self.use_san and flg_train:
            z_fun, z_dir = self.conv_post(latent_z, flg_train=True)
            logits = [torch.flatten(z_fun, 1, -1), torch.flatten(z_dir, 1, -1)]
        else:
            latent_z = self.conv_post(latent_z)
            logits = torch.flatten(latent_z, 1, -1)

        fmap.append(logits)
        return fmap

    def normalize_san_weights(self) -> None:
        if self.use_san and hasattr(self.conv_post, "normalize_weight"):
            self.conv_post.normalize_weight()


class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        cqt_params: dict[str, Any] | None = None,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: tuple[int, int] | Sequence[int] = (3, 9),
        channels: int | None = None,
        max_downsample_channels: int | None = None,
        downsample_scales: tuple[int, ...] | Sequence[int] | None = None,
        use_weight_norm: bool = True,
        filters: int | None = None,
        max_filters: int | None = None,
        filters_scale: int = 1,
        dilations: Sequence[int] = (1, 2, 4),
        stride: Sequence[int] = (1, 2),
        norm: str | None = None,
        activation: str = "LeakyReLU",
        activation_params: dict | None = None,
        use_san: bool = False,
    ):
        super().__init__()
        cqt_params = cqt_params or {}
        if norm is None:
            norm = "weight_norm" if use_weight_norm else "none"
        filters = int(filters if filters is not None else (channels if channels is not None else 128))
        max_filters = int(
            max_filters if max_filters is not None else (
                max_downsample_channels if max_downsample_channels is not None else 1024
            )
        )
        if downsample_scales is not None and tuple(dilations) == (1, 2, 4):
            dilations = tuple(int(v) for v in downsample_scales)

        hop_lengths = list(cqt_params.get("hop_lengths", [512, 256, 256]))
        num_scales = len(hop_lengths)
        n_octaves = _expand_per_scale(cqt_params.get("n_octaves", None), num_scales, "cqt_params.n_octaves")
        bins_per_octaves = _expand_per_scale(
            cqt_params.get("bins_per_octaves", cqt_params.get("bins_per_octave", None)),
            num_scales,
            "cqt_params.bins_per_octaves",
        )
        fmins = _expand_per_scale(cqt_params.get("fmins", 32.7), num_scales, "cqt_params.fmins")
        filter_scales = _expand_per_scale(cqt_params.get("filter_scales", 1.0), num_scales, "cqt_params.filter_scales")
        resample_factor = int(cqt_params.get("resample_factor", 2))
        cqt_pad_mode = str(cqt_params.get("pad_mode", "constant"))
        cqt_window = str(cqt_params.get("window", "hann"))
        cqt_earlydownsample = bool(cqt_params.get("earlydownsample", True))
        cqt_verbose = bool(cqt_params.get("verbose", True))
        normalize_volume = bool(cqt_params.get("normalize_volume", False))

        if n_octaves[0] is None:
            n_bins = _expand_per_scale(cqt_params.get("n_bins", 216), num_scales, "cqt_params.n_bins")
            n_octaves = [int(nb) // int(bpo) for nb, bpo in zip(n_bins, bins_per_octaves)]

        self.discriminators = nn.ModuleList()
        for idx, (hop_length, octave_count, bins_per_octave, fmin, filter_scale) in enumerate(zip(
            hop_lengths,
            n_octaves,
            bins_per_octaves,
            fmins,
            filter_scales,
        ), start=1):
            print(
                f"[cqt-disc] preparing scale {idx}/{num_scales}: "
                f"hop={int(hop_length)} octaves={int(octave_count)} "
                f"bpo={int(bins_per_octave)} bins={int(octave_count) * int(bins_per_octave)} "
                f"(frontend=nnAudio.CQT2010v2)"
            )
            scale_disc = NNAudioCQTDiscriminator(
                sample_rate=int(sample_rate),
                hop_length=int(hop_length),
                n_octaves=int(octave_count),
                bins_per_octave=int(bins_per_octave),
                fmin=float(fmin),
                filter_scale=float(filter_scale),
                in_channels=int(in_channels),
                out_channels=int(out_channels),
                filters=filters,
                max_filters=max_filters,
                filters_scale=filters_scale,
                kernel_size=tuple(int(v) for v in kernel_sizes),
                dilations=tuple(int(v) for v in dilations),
                stride=tuple(int(v) for v in stride),
                norm=norm,
                activation=activation,
                activation_params=activation_params,
                use_san=use_san,
                resample_factor=resample_factor,
                normalize_volume=normalize_volume,
                cqt_pad_mode=cqt_pad_mode,
                cqt_window=cqt_window,
                cqt_earlydownsample=cqt_earlydownsample,
                cqt_verbose=cqt_verbose,
            )
            self.discriminators.append(scale_disc)
            print(
                f"[cqt-disc] scale {idx}/{num_scales} ready: "
                f"resample_factor={resample_factor}"
            )

        self.supports_san_training = bool(use_san)

    def forward(self, x: torch.Tensor, flg_train: bool = False) -> list[list[torch.Tensor | list[torch.Tensor]]]:
        results = []
        for disc in self.discriminators:
            results.append(disc(x, flg_train=flg_train))
        return results

    def normalize_san_weights(self) -> None:
        for disc in self.discriminators:
            disc.normalize_san_weights()
