# Copyright (c) 2021 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2022 Hongji Wang (jijijiang77@gmail.com)
#               2023 Bing Han (hanbing97@sjtu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" This implementation is adapted from github repo:
    https://github.com/lawlict/ECAPA-TDNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import vq.speaker.pooling_layers as pooling_layers

from speechbrain.inference.speaker import EncoderClassifier


class Res2Conv1dReluBn(nn.Module):
    """
    in_channels == out_channels == channels
    """

    def __init__(
        self,
        channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        scale=4,
    ):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    bias=bias,
                )
            )
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            # Order: conv -> relu -> bn
            if i >= 1:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)

        return out


""" Conv1d + BatchNorm1d + ReLU
"""


class Conv1dReluBn(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


""" The SE connection of 1D case.
"""


class SE_Connect(nn.Module):

    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)

        return out


""" SE-Res2Block of the ECAPA-TDNN architecture.
"""


class SE_Res2Block(nn.Module):

    def __init__(self, channels, kernel_size, stride, padding, dilation, scale):
        super().__init__()
        self.se_res2block = nn.Sequential(
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            Res2Conv1dReluBn(
                channels, kernel_size, stride, padding, dilation, scale=scale
            ),
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            SE_Connect(channels),
        )

    def forward(self, x):
        return x + self.se_res2block(x)


class ECAPA_TDNN(nn.Module):

    def __init__(
        self,
        channels=512,
        feat_dim=80,
        embed_dim=192,
        pooling_func="ASTP",
        global_context_att=False,
        emb_bn=False,
        norm_layer='bn',
    ):
        super().__init__()

        self.layer1 = Conv1dReluBn(feat_dim, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(
            channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8
        )
        self.layer3 = SE_Res2Block(
            channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8
        )
        self.layer4 = SE_Res2Block(
            channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8
        )

        cat_channels = channels * 3
        out_channels = 512 * 3
        self.conv = nn.Conv1d(cat_channels, out_channels, kernel_size=1)
        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=out_channels, global_context_att=global_context_att
        )
        self.pool_out_dim = self.pool.get_out_dim()
        if norm_layer == 'bn':
            self.bn = nn.BatchNorm1d(self.pool_out_dim)
        elif norm_layer == 'in':
            self.bn = nn.InstanceNorm1d(self.pool_out_dim, affine=False)
        elif norm_layer == 'ln':
            self.bn = nn.LayerNorm(self.pool_out_dim, elementwise_affine=False)
        self.linear = nn.Linear(self.pool_out_dim, embed_dim)
        self.emb_bn = emb_bn
        if emb_bn:  # better in SSL for SV
            self.bn2 = nn.BatchNorm1d(embed_dim)
        else:
            self.bn2 = nn.Identity()

    def forward(self, x, return_latent=False):
        x = x.permute(0, 2, 1)  # (B,T,F) -> (B,F,T)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = torch.cat([out2, out3, out4], dim=1)
        latent = F.relu(self.conv(out))
        out = self.bn(self.pool(latent))
        out = self.linear(out)
        if self.emb_bn:
            out = self.bn2(out)

        if return_latent:
            return out, latent
        return out



def ECAPA_TDNN_c1024(feat_dim, embed_dim, pooling_func="ASTP", emb_bn=False):
    return ECAPA_TDNN(
        channels=1024,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        emb_bn=emb_bn,
    )


def ECAPA_TDNN_GLOB_c1024(feat_dim, embed_dim, pooling_func="ASTP", emb_bn=False):
    return ECAPA_TDNN(
        channels=1024,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        global_context_att=True,
        emb_bn=emb_bn,
    )


def ECAPA_TDNN_c512(feat_dim, embed_dim, pooling_func="ASTP", emb_bn=False):
    return ECAPA_TDNN(
        channels=512,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        emb_bn=emb_bn,
    )


def ECAPA_TDNN_GLOB_c512(feat_dim, embed_dim, pooling_func="ASTP", emb_bn=False, norm_layer=False):
    return ECAPA_TDNN(
        channels=512,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        global_context_att=True,
        emb_bn=emb_bn,
        norm_layer=norm_layer
    )


class EmbWrapper(nn.Module):
    def __init__(self, base: nn.Module, out_dim: int):
        super().__init__()
        self.base = base
        self.base.fc = nn.Conv1d(self.base.fc.conv.in_channels, out_dim,
                                 kernel_size=self.base.fc.conv.kernel_size,
                                 stride=self.base.fc.conv.stride,)
        
    def forward(self, x, lengths=None):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        lengths : torch.Tensor
            Corresponding relative lengths of inputs.

        Returns
        -------
        x : torch.Tensor ()
            Embedding vector.
        """
        # Minimize transpose for efficiency
        x = x.transpose(1, 2)

        xl = []
        for layer in self.base.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[1:], dim=1)
        latent = self.base.mfa(x)

        # Attentive Statistical Pooling
        x = self.base.asp(latent, lengths=lengths)
        x = self.base.asp_bn(x)

        # Final linear transformation
        x = self.base.fc(x)

        x = x.transpose(1, 2)
        return x.squeeze(), latent # (B, T), (B, D, frames)

class EncoderClassifierWrapper(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, wavs, return_latent=True):
        """Encodes the input audio into a single vector embedding.

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = <this>.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.
        normalize : bool
            If True, it normalizes the embeddings with the statistics
            contained in mean_var_norm_emb.

        Returns
        -------
        torch.Tensor
            The encoded batch
        """
        wav_lens=None
        normalize=False
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=wavs.device)

        # Storing waveform in the specified device
        # wavs, wav_lens = wavs.to(wavs.device), wav_lens.to(self.device)
        wavs = wavs.float()

        # Computing features and embeddings
        feats = self.base.mods.compute_features(wavs)
        feats = self.base.mods.mean_var_norm(feats, wav_lens)
        embeddings, latent = self.base.mods.embedding_model(feats, wav_lens)
        if normalize:
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings, torch.ones(embeddings.shape[0], device=wavs.device)
            )
        return embeddings, latent

def ECAPA_TDNN_SPEECHBRAIN(out_dim, source="speechbrain/spkrec-ecapa-voxceleb"):
    classifier = EncoderClassifier.from_hparams(source=source, savedir="./.cache")
    wrapper = EmbWrapper(classifier.mods.embedding_model, out_dim=out_dim)
    classifier.mods.embedding_model = wrapper
    classifier = EncoderClassifierWrapper(classifier)
    return classifier


if __name__ == "__main__":
    x = torch.zeros(1, 200, 100)
    model = ECAPA_TDNN_GLOB_c512(feat_dim=100, embed_dim=256, pooling_func="ASTP")
    model.eval()
    out, latent = model(x, True)
    print(out.shape)
    print(latent.shape)

    num_params = sum(param.numel() for param in model.parameters())
    print("{} M".format(num_params / 1e6))

    # from thop import profile
    # x_np = torch.randn(1, 200, 80)
    # flops, params = profile(model, inputs=(x_np, ))
    # print("FLOPs: {} G, Params: {} M".format(flops / 1e9, params / 1e6))
