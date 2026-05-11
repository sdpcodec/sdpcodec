import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize(tensor: torch.Tensor, dim) -> torch.Tensor:
    denom = tensor.norm(p=2.0, dim=dim, keepdim=True).clamp_min(1e-12)
    return tensor / denom


class SANConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        scale = self.weight.norm(p=2.0, dim=[1, 2, 3], keepdim=True).clamp_min(1e-12)
        self.weight = nn.Parameter(self.weight / scale.expand_as(self.weight))
        self.scale = nn.Parameter(scale.view(out_channels))
        if bias:
            self.bias = nn.Parameter(torch.zeros(in_channels, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor, flg_train: bool = False):
        if self.bias is not None:
            input = input + self.bias.view(self.in_channels, 1, 1)
        normalized_weight = self._get_normalized_weight()
        scale = self.scale.view(self.out_channels, 1, 1)
        if flg_train:
            out_fun = F.conv2d(
                input,
                normalized_weight.detach(),
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            out_dir = F.conv2d(
                input.detach(),
                normalized_weight,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            return [out_fun * scale, out_dir * scale.detach()]
        out = F.conv2d(
            input,
            normalized_weight,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return out * scale

    @torch.no_grad()
    def normalize_weight(self) -> None:
        self.weight.data = self._get_normalized_weight()

    def _get_normalized_weight(self) -> torch.Tensor:
        return _normalize(self.weight, dim=[1, 2, 3])
