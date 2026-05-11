import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    def __init__(self, mode: str = "lsgan"):
        super().__init__()
        self.mode = str(mode).lower()
        valid_modes = {"lsgan", "hinge", "apnet2", "bigvsan"}
        if self.mode not in valid_modes:
            raise ValueError(f"Unsupported GAN loss mode: {mode}. Expected one of {sorted(valid_modes)}.")

    def disc_loss(self, real: torch.Tensor, fake: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "lsgan":
            real_loss = F.mse_loss(real, torch.ones_like(real))
            fake_loss = F.mse_loss(fake, torch.zeros_like(fake))
            return real_loss, fake_loss

        if self.mode == "hinge":
            real_loss = torch.mean(F.relu(1.0 - real))
            fake_loss = torch.mean(F.relu(1.0 + fake))
            return real_loss, fake_loss

        if self.mode == "apnet2":
            real_loss = torch.mean(torch.clamp(1.0 - real, min=0.0))
            fake_loss = torch.mean(torch.clamp(1.0 + fake, min=0.0))
            return real_loss, fake_loss

        if not isinstance(real, (list, tuple)) or not isinstance(fake, (list, tuple)) or len(real) != 2 or len(fake) != 2:
            raise ValueError("BigVSAN discriminator loss expects SAN outputs shaped as [fun, dir].")
        real_fun, real_dir = real
        fake_fun, fake_dir = fake
        real_loss_fun = torch.mean(F.softplus(1.0 - real_fun).pow(2))
        fake_loss_fun = torch.mean(F.softplus(fake_fun).pow(2))
        real_loss_dir = torch.mean(F.softplus(1.0 - real_dir).pow(2))
        fake_loss_dir = torch.mean(-F.softplus(1.0 - fake_dir).pow(2))
        real_loss = real_loss_fun + real_loss_dir
        fake_loss = fake_loss_fun + fake_loss_dir
        return real_loss, fake_loss

    def gen_loss(self, fake: torch.Tensor) -> torch.Tensor:
        if self.mode == "lsgan":
            return F.mse_loss(fake, torch.ones_like(fake))

        if self.mode == "hinge":
            return -torch.mean(fake)

        if self.mode == "apnet2":
            return torch.mean(torch.clamp(1.0 - fake, min=0.0))

        return torch.mean(F.softplus(1.0 - fake).pow(2))
