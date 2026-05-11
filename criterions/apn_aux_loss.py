from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn


class APNet2AuxLoss(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        gd_matrix = (
            torch.triu(torch.ones(self.n_fft // 2 + 1, self.n_fft // 2 + 1), diagonal=1)
            - torch.triu(torch.ones(self.n_fft // 2 + 1, self.n_fft // 2 + 1), diagonal=2)
            - torch.eye(self.n_fft // 2 + 1)
        )
        self.register_buffer("gd_matrix", gd_matrix, persistent=False)
        self.mse = nn.MSELoss()
        self._ptd_cache: Dict[int, torch.Tensor] = {}

    @staticmethod
    def anti_wrapping_function(x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

    def _ptd_matrix(self, frames: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cached = self._ptd_cache.get(frames)
        if cached is None or cached.device != device or cached.dtype != dtype:
            cached = (
                torch.triu(torch.ones(frames, frames, device=device, dtype=dtype), diagonal=1)
                - torch.triu(torch.ones(frames, frames, device=device, dtype=dtype), diagonal=2)
                - torch.eye(frames, device=device, dtype=dtype)
            )
            self._ptd_cache[frames] = cached
        return cached

    def stft_components(self, wav: torch.Tensor) -> Dict[str, torch.Tensor]:
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        spec = torch.stft(
            wav.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=wav.device),
            center=True,
            return_complex=True,
        )
        real = spec.real
        imag = spec.imag
        log_amplitude = torch.log(torch.abs(spec) + 1e-5)
        phase = torch.atan2(imag, real)
        return {
            "log_amplitude": log_amplitude,
            "phase": phase,
            "real": real,
            "imag": imag,
        }

    @staticmethod
    def _align_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freq = min(a.shape[-2], b.shape[-2])
        frames = min(a.shape[-1], b.shape[-1])
        return a[..., :freq, :frames], b[..., :freq, :frames]

    def phase_loss(
        self,
        phase_r: torch.Tensor,
        phase_g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase_r, phase_g = self._align_pair(phase_r, phase_g)
        gd_matrix = self.gd_matrix.to(device=phase_g.device, dtype=phase_g.dtype)
        gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
        gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

        frames = phase_g.shape[-1]
        ptd_matrix = self._ptd_matrix(frames, device=phase_g.device, dtype=phase_g.dtype)
        ptd_r = torch.matmul(phase_r, ptd_matrix)
        ptd_g = torch.matmul(phase_g, ptd_matrix)

        ip_loss = torch.mean(self.anti_wrapping_function(phase_r - phase_g))
        gd_loss = torch.mean(self.anti_wrapping_function(gd_r - gd_g))
        ptd_loss = torch.mean(self.anti_wrapping_function(ptd_r - ptd_g))
        return ip_loss, gd_loss, ptd_loss

    def forward(
        self,
        gt_wav: torch.Tensor,
        gen_wav: torch.Tensor,
        pred_aux: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        gt = self.stft_components(gt_wav)
        gen_from_wav = self.stft_components(gen_wav)

        pred_log_amplitude, gt_log_amplitude = self._align_pair(pred_aux["log_amplitude"], gt["log_amplitude"])
        pred_phase, gt_phase = self._align_pair(pred_aux["phase"], gt["phase"])
        pred_real, gt_real = self._align_pair(pred_aux["real"], gt["real"])
        pred_imag, gt_imag = self._align_pair(pred_aux["imag"], gt["imag"])
        pred_real_cons, gen_real = self._align_pair(pred_aux["real"], gen_from_wav["real"])
        pred_imag_cons, gen_imag = self._align_pair(pred_aux["imag"], gen_from_wav["imag"])

        amplitude_loss = self.mse(gt_log_amplitude, pred_log_amplitude)
        ip_loss, gd_loss, ptd_loss = self.phase_loss(gt_phase, pred_phase)
        consistency_loss = torch.mean(torch.mean((pred_real_cons - gen_real) ** 2 + (pred_imag_cons - gen_imag) ** 2, dim=(1, 2)))
        real_loss = torch.mean(torch.abs(gt_real - pred_real))
        imag_loss = torch.mean(torch.abs(gt_imag - pred_imag))

        return {
            "apn_amplitude_loss": amplitude_loss,
            "apn_phase_ip_loss": ip_loss,
            "apn_phase_gd_loss": gd_loss,
            "apn_phase_ptd_loss": ptd_loss,
            "apn_phase_loss": ip_loss + gd_loss + ptd_loss,
            "apn_consistency_loss": consistency_loss,
            "apn_real_loss": real_loss,
            "apn_imag_loss": imag_loss,
        }
