import torch
from torch import nn
import numpy as np
from .module import WNConv1d, EncoderBlock, ResLSTM
from .alias_free_torch import *
from . import activations
import math

"""MI module
    Modified from: https://github.com/Linear95/CLUB
    Modified from: https://github.com/PecholaL/MAIN-VC/tree/main
"""

class CLUBSample_group(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size, per_dim=True):
        super().__init__()
        h = hidden_size // 2
        self.per_dim = per_dim
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, y_dim),
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, y_dim),
        )
        self.register_buffer('LOG2E', torch.tensor(1.4426950408889634))

    def get_mu_logvar(self, x):
        mu = self.p_mu(x)
        logvar = self.p_logvar(x).clamp_(-6, 6)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples, to_bits=True):
        mu, logvar = self.get_mu_logvar(x_samples)
        B, T, Dy = y_samples.shape
        mu = mu.unsqueeze(1).expand(-1, T, -1).reshape(-1, Dy)
        logvar = logvar.unsqueeze(1).expand(-1, T, -1).reshape(-1, Dy)
        y_flat = y_samples.reshape(-1, Dy)
        var = logvar.exp()
        ll = -0.5 * (((y_flat - mu) ** 2) / var + logvar)  # (N,Dy)
        ll = ll.sum(dim=-1).mean()  # total log-lik (상수항 제외)
        if to_bits:
            ll = ll * self.LOG2E
        if self.per_dim:
            ll = ll / Dy
        return ll

    def mi_est(self, x_samples, y_samples, to_bits=True):
        # E[ log p(y|x) - log p(y|x') ]  (Gaussian 상수항 상쇄)
        B, T, Dy = y_samples.shape
        N = B * T
        x_rep = x_samples.unsqueeze(1).expand(-1, T, -1).reshape(N, -1)
        y_flat = y_samples.reshape(N, Dy)

        mu, logvar = self.get_mu_logvar(x_rep)
        var = logvar.exp()

        log_p_pos = -0.5 * (((y_flat - mu) ** 2) / var + logvar).sum(dim=-1)  # (N,)
        perm = torch.randperm(N, device=y_flat.device)
        y_neg = y_flat[perm]
        log_p_neg = -0.5 * (((y_neg - mu) ** 2) / var + logvar).sum(dim=-1)

        diff = (log_p_pos - log_p_neg).mean()  # nats
        if to_bits:
            diff = diff * self.LOG2E
        if self.per_dim:
            diff = diff / Dy
        return diff


class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size, per_dim=True):
        super().__init__()
        self.per_dim = per_dim
        self.T_func = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.register_buffer('LOG2E', torch.tensor(1.4426950408889634))

    def forward(self, x_samples, y_samples, to_bits=True):
        B, T, Dy = y_samples.shape
        Dx = x_samples.shape[-1]
        N = B * T

        x_flat = x_samples.unsqueeze(1).expand(-1, T, -1).reshape(N, Dx)
        y_flat = y_samples.reshape(N, Dy)

        joint_xy = torch.cat([x_flat, y_flat], dim=1)
        perm = torch.randperm(N, device=y_flat.device)
        y_shuffle = y_flat[perm]
        marg_xy = torch.cat([x_flat, y_shuffle], dim=1)

        all_inputs = torch.cat([joint_xy, marg_xy], dim=0)
        logits = self.T_func(all_inputs).squeeze(-1)
        T_joint, T_marg = logits[:N], logits[N:]

        log_mean_exp = torch.logsumexp(T_marg, dim=0) - math.log(N)
        lower_nats = T_joint.mean() - log_mean_exp
        if to_bits:
            lower = lower_nats * self.LOG2E
        else:
            lower = lower_nats
        if self.per_dim:
            lower = lower / Dy
        return lower

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)