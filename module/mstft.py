from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torchaudio
from einops import rearrange
from torch.nn.utils import spectral_norm, weight_norm

from common.audio import stft as legacy_stft
from module.san import SANConv2d


def get_2d_padding(
    kernel_size: tuple[int, int],
    dilation: tuple[int, int] = (1, 1),
) -> tuple[int, int]:
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


def _maybe_norm(module: nn.Module, norm: str) -> nn.Module:
    norm = str(norm).lower()
    if norm == "weight_norm":
        return weight_norm(module)
    if norm == "spectral_norm":
        return spectral_norm(module)
    if norm in {"none", "identity", "false"}:
        return module
    raise ValueError(f"Unsupported norm type: {norm}")


class NormConv2d(nn.Module):
    def __init__(self, *args, norm: str = "weight_norm", **kwargs):
        super().__init__()
        self.conv = _maybe_norm(nn.Conv2d(*args, **kwargs), norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DiscriminatorSTFT(nn.Module):
    def __init__(
        self,
        filters: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        max_filters: int = 1024,
        filters_scale: int = 1,
        kernel_size: tuple[int, int] = (3, 9),
        dilations: Sequence[int] = (1, 2, 4),
        stride: tuple[int, int] = (1, 2),
        normalized: bool = True,
        norm: str = "weight_norm",
        activation: str = "LeakyReLU",
        activation_params: dict | None = None,
        use_san: bool = False,
    ):
        super().__init__()
        activation_params = activation_params or {"negative_slope": 0.2}
        self.use_san = bool(use_san)
        self.supports_san_training = self.use_san
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            win_length=int(win_length),
            window_fn=torch.hann_window,
            normalized=bool(normalized),
            center=False,
            pad_mode=None,
            power=None,
        )

        spec_channels = 2 * int(in_channels)
        filters = int(filters)
        max_filters = int(max_filters)
        filters_scale = int(filters_scale)
        kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        stride = (int(stride[0]), int(stride[1]))

        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(
                spec_channels,
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

    def forward(self, x: torch.Tensor, flg_train: bool = False) -> list[torch.Tensor | list[torch.Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        fmap: list[torch.Tensor | list[torch.Tensor]] = []
        z = self.spec_transform(x)
        z = torch.cat([z.real, z.imag], dim=1)
        z = rearrange(z, "b c f t -> b c t f")

        for layer in self.convs:
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)

        if self.use_san and flg_train:
            z_fun, z_dir = self.conv_post(z, flg_train=True)
            logits = [torch.flatten(z_fun, 1, -1), torch.flatten(z_dir, 1, -1)]
        else:
            z = self.conv_post(z)
            logits = torch.flatten(z, 1, -1)

        fmap.append(logits)
        return fmap

    def normalize_san_weights(self) -> None:
        if self.use_san and hasattr(self.conv_post, "normalize_weight"):
            self.conv_post.normalize_weight()


class LegacyNLayerSpecDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: Sequence[int] = (5, 3),
        channels: int = 32,
        max_downsample_channels: int = 512,
        downsample_scales: Sequence[int] = (2, 2, 2),
    ):
        super().__init__()
        kernel_sizes = tuple(int(v) for v in kernel_sizes)
        if kernel_sizes[0] % 2 != 1 or kernel_sizes[1] % 2 != 1:
            raise ValueError(f"LegacyNLayerSpecDiscriminator expects odd kernel sizes, got {kernel_sizes}.")

        model = nn.ModuleDict()
        model["layer_0"] = nn.Sequential(
            nn.Conv2d(
                int(in_channels),
                int(channels),
                kernel_size=kernel_sizes[0],
                stride=2,
                padding=kernel_sizes[0] // 2,
            ),
            nn.LeakyReLU(0.2, True),
        )

        in_chs = int(channels)
        max_downsample_channels = int(max_downsample_channels)
        for idx, downsample_scale in enumerate(tuple(int(v) for v in downsample_scales)):
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)
            model[f"layer_{idx + 1}"] = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=downsample_scale * 2 + 1,
                    stride=downsample_scale,
                    padding=downsample_scale,
                ),
                nn.LeakyReLU(0.2, True),
            )
            in_chs = out_chs

        out_chs = min(in_chs * 2, max_downsample_channels)
        model[f"layer_{len(tuple(downsample_scales)) + 1}"] = nn.Sequential(
            nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size=kernel_sizes[1],
                padding=kernel_sizes[1] // 2,
            ),
            nn.LeakyReLU(0.2, True),
        )
        model[f"layer_{len(tuple(downsample_scales)) + 2}"] = nn.Conv2d(
            out_chs,
            int(out_channels),
            kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1] // 2,
        )
        self.model = model

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        results = []
        for _, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class LegacySpecDiscriminator(nn.Module):
    def __init__(
        self,
        stft_params: dict | None = None,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: Sequence[int] = (7, 3),
        channels: int = 32,
        max_downsample_channels: int = 512,
        downsample_scales: Sequence[int] = (2, 2, 2),
        use_weight_norm: bool = True,
    ):
        super().__init__()
        if stft_params is None:
            stft_params = {
                "fft_sizes": [1024, 2048, 512],
                "hop_sizes": [120, 240, 50],
                "win_lengths": [600, 1200, 240],
                "window": "hann_window",
            }
        self.stft_params = stft_params
        self.model = nn.ModuleDict()
        for idx in range(len(stft_params["fft_sizes"])):
            self.model[f"disc_{idx}"] = LegacyNLayerSpecDiscriminator(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                channels=channels,
                max_downsample_channels=max_downsample_channels,
                downsample_scales=downsample_scales,
            )
        if use_weight_norm:
            self.apply_weight_norm()
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> list[list[torch.Tensor]]:
        results = []
        x = x.squeeze(1)
        for idx, disc in enumerate(self.model.values()):
            win_length = int(self.stft_params["win_lengths"][idx])
            spec = legacy_stft(
                x,
                int(self.stft_params["fft_sizes"][idx]),
                int(self.stft_params["hop_sizes"][idx]),
                win_length,
                window=getattr(torch, self.stft_params["window"])(win_length),
            )
            spec = spec.transpose(1, 2).unsqueeze(1)
            results.append(disc(spec))
        return results

    def remove_weight_norm(self) -> None:
        def _remove_weight_norm(module: nn.Module) -> None:
            try:
                torch.nn.utils.remove_weight_norm(module)
            except ValueError:
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self) -> None:
        def _apply_weight_norm(module: nn.Module) -> None:
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.utils.weight_norm(module)

        self.apply(_apply_weight_norm)

    def reset_parameters(self) -> None:
        def _reset_parameters(module: nn.Module) -> None:
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
                module.weight.data.normal_(0.0, 0.02)

        self.apply(_reset_parameters)


class SpecDiscriminator(nn.Module):
    def __init__(
        self,
        stft_params: dict | None = None,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: Sequence[int] | None = None,
        channels: int | None = None,
        max_downsample_channels: int | None = None,
        downsample_scales: Sequence[int] | None = None,
        use_weight_norm: bool = True,
        filters: int | None = None,
        max_filters: int | None = None,
        filters_scale: int = 1,
        kernel_size: Sequence[int] | None = None,
        dilations: Sequence[int] = (1, 2, 4),
        stride: Sequence[int] = (1, 2),
        normalized: bool = True,
        norm: str | None = None,
        activation: str = "LeakyReLU",
        activation_params: dict | None = None,
        use_san: bool = False,
    ):
        super().__init__()
        if stft_params is None:
            stft_params = {
                "fft_sizes": [1024, 2048, 512],
                "hop_sizes": [256, 512, 256],
                "win_lengths": [1024, 2048, 512],
            }
        if norm is None:
            norm = "weight_norm" if use_weight_norm else "none"

        filters = int(filters if filters is not None else (channels if channels is not None else 32))
        max_filters = int(
            max_filters if max_filters is not None else (
                max_downsample_channels if max_downsample_channels is not None else 1024
            )
        )
        if kernel_size is None:
            if kernel_sizes is not None and len(tuple(kernel_sizes)) == 2:
                kernel_size = tuple(int(v) for v in kernel_sizes)
            else:
                kernel_size = (3, 9)
        if downsample_scales is not None and dilations == (1, 2, 4):
            dilations = tuple(int(v) for v in downsample_scales)

        fft_sizes = list(stft_params["fft_sizes"])
        hop_sizes = list(stft_params["hop_sizes"])
        win_lengths = list(stft_params["win_lengths"])
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorSTFT(
                    filters=filters,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_fft=int(fft_sizes[idx]),
                    hop_length=int(hop_sizes[idx]),
                    win_length=int(win_lengths[idx]),
                    max_filters=max_filters,
                    filters_scale=filters_scale,
                    kernel_size=tuple(int(v) for v in kernel_size),
                    dilations=tuple(int(v) for v in dilations),
                    stride=tuple(int(v) for v in stride),
                    normalized=normalized,
                    norm=norm,
                    activation=activation,
                    activation_params=activation_params,
                    use_san=use_san,
                )
                for idx in range(len(fft_sizes))
            ]
        )
        self.supports_san_training = bool(use_san)

    def forward(self, x: torch.Tensor, flg_train: bool = False) -> list[list[torch.Tensor | list[torch.Tensor]]]:
        return [disc(x, flg_train=flg_train) for disc in self.discriminators]

    def normalize_san_weights(self) -> None:
        for disc in self.discriminators:
            disc.normalize_san_weights()
