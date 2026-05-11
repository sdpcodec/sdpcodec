from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm
from torchaudio.transforms import Spectrogram


class VocosDiscriminatorR(nn.Module):
    def __init__(
        self,
        window_length: int,
        channels: int = 32,
        hop_factor: float = 0.25,
        bands: Sequence[Sequence[float]] = (
            (0.0, 0.1),
            (0.1, 0.25),
            (0.25, 0.5),
            (0.5, 0.75),
            (0.75, 1.0),
        ),
        lrelu_slope: float = 0.1,
    ):
        super().__init__()
        self.window_length = int(window_length)
        self.hop_factor = float(hop_factor)
        self.lrelu_slope = float(lrelu_slope)
        self.spec_fn = Spectrogram(
            n_fft=self.window_length,
            hop_length=int(self.window_length * self.hop_factor),
            win_length=self.window_length,
            power=None,
        )
        n_fft_bins = self.window_length // 2 + 1
        self.bands = [(int(band[0] * n_fft_bins), int(band[1] * n_fft_bins)) for band in bands]

        def _band_stack() -> nn.ModuleList:
            return nn.ModuleList(
                [
                    weight_norm(nn.Conv2d(2, channels, (3, 9), (1, 1), padding=(1, 4))),
                    weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                    weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                    weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                    weight_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1))),
                ]
            )

        self.band_convs = nn.ModuleList([_band_stack() for _ in self.bands])
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1)))
        self.supports_san_training = False

    def spectrogram(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = x - x.mean(dim=-1, keepdim=True)
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        x = self.spec_fn(x)
        x = torch.view_as_real(x)
        x = rearrange(x, "b f t c -> b c t f")
        return [x[..., start:end] for start, end in self.bands]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.dim() == 3:
            x = x.squeeze(1)

        x_bands = self.spectrogram(x)
        fmap: list[torch.Tensor] = []
        band_outputs = []
        for band, stack in zip(x_bands, self.band_convs):
            for idx, layer in enumerate(stack):
                band = layer(band)
                band = F.leaky_relu(band, self.lrelu_slope)
                if idx > 0:
                    fmap.append(band)
            band_outputs.append(band)

        logits = self.conv_post(torch.cat(band_outputs, dim=-1))
        fmap.append(logits)
        return fmap + [torch.flatten(logits, 1, -1)]


class BigVGANDiscriminatorR(nn.Module):
    def __init__(
        self,
        resolution: Sequence[int],
        channels: int = 32,
        lrelu_slope: float = 0.1,
    ):
        super().__init__()
        self.resolution = tuple(int(v) for v in resolution)
        self.lrelu_slope = float(lrelu_slope)
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, channels, (3, 9), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))
        self.supports_san_training = False

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        pad = int((n_fft - hop_length) / 2)
        x = F.pad(x.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=False,
            return_complex=True,
        )
        return torch.abs(x)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.dim() == 3:
            x = x.squeeze(1)

        x = self.spectrogram(x).unsqueeze(1)
        fmap: list[torch.Tensor] = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        logits = self.conv_post(x)
        fmap.append(logits)
        return fmap + [torch.flatten(logits, 1, -1)]


class BigVGANMultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        resolutions: Sequence[Sequence[int]] = (
            (1024, 120, 600),
            (2048, 240, 1200),
            (512, 50, 240),
        ),
        channels: int = 32,
        lrelu_slope: float = 0.1,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                BigVGANDiscriminatorR(
                    resolution=resolution,
                    channels=channels,
                    lrelu_slope=lrelu_slope,
                )
                for resolution in resolutions
            ]
        )
        self.supports_san_training = False

    def forward(self, x: torch.Tensor) -> list[list[torch.Tensor]]:
        return [disc(x) for disc in self.discriminators]


class VocosMultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        fft_sizes: Sequence[int] = (2048, 1024, 512),
        channels: int = 32,
        hop_factor: float = 0.25,
        bands: Sequence[Sequence[float]] = (
            (0.0, 0.1),
            (0.1, 0.25),
            (0.25, 0.5),
            (0.5, 0.75),
            (0.75, 1.0),
        ),
        lrelu_slope: float = 0.1,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                VocosDiscriminatorR(
                    window_length=int(fft_size),
                    channels=channels,
                    hop_factor=hop_factor,
                    bands=bands,
                    lrelu_slope=lrelu_slope,
                )
                for fft_size in fft_sizes
            ]
        )
        self.supports_san_training = False

    def forward(self, x: torch.Tensor) -> list[list[torch.Tensor]]:
        return [disc(x) for disc in self.discriminators]


class APNet2DiscriminatorR(nn.Module):
    def __init__(
        self,
        resolution: Sequence[int],
        channels: int = 64,
        lrelu_slope: float = 0.1,
    ):
        super().__init__()
        self.resolution = tuple(int(v) for v in resolution)
        self.lrelu_slope = float(lrelu_slope)
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, channels, kernel_size=(7, 5), stride=(2, 2), padding=(3, 2))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 1), padding=1)),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 2), padding=1)),
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))
        self.supports_san_training = False

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        return torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=None,
            center=True,
            return_complex=True,
        ).abs()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.dim() == 3:
            x = x.squeeze(1)

        x = self.spectrogram(x).unsqueeze(1)
        fmap: list[torch.Tensor] = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        logits = self.conv_post(x)
        fmap.append(logits)
        return fmap + [torch.flatten(logits, 1, -1)]


class APNet2MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        resolutions: Sequence[Sequence[int]] = (
            (1024, 256, 1024),
            (2048, 512, 2048),
            (512, 128, 512),
        ),
        channels: int = 64,
        lrelu_slope: float = 0.1,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                APNet2DiscriminatorR(
                    resolution=resolution,
                    channels=channels,
                    lrelu_slope=lrelu_slope,
                )
                for resolution in resolutions
            ]
        )
        self.supports_san_training = False

    def forward(self, x: torch.Tensor) -> list[list[torch.Tensor]]:
        return [disc(x) for disc in self.discriminators]
