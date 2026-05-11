# Adapted from https://github.com/openai/jukebox
# Adapted from https://github.com/facebookresearch/speech-resynthesis

import numpy as np
import torch.nn as nn
from vq.f0.resnet import Resnet1D

from typing import List
from omegaconf import ListConfig

def assert_shape(x, exp_shape):
    assert x.shape == exp_shape, f"Expected {exp_shape} got {x.shape}"

def build_width_list(width, levels, downs_t, channel_growth_rate, output_emb_width):
    width_list = []
    cur_width = width
    for k in range(levels):
        level_list = []
        for i in range(len(downs_t[k])):
            level_list.append(cur_width)
            cur_width = int(min(cur_width * (channel_growth_rate), output_emb_width))
        width_list.append(level_list)
    return width_list


class EncoderConvBlock(nn.Module):
    def __init__(self, input_emb_width, widths: List, output_emb_width, down_t, stride_t, depth, m_conv,
                 dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False):
        super().__init__()
        blocks = []
        if isinstance(stride_t, (tuple, list, ListConfig)):
            # start = True
            for i, (s_t, d_t) in enumerate(zip(stride_t, down_t)):
                if s_t % 2 == 0: filter_t, pad_t = s_t * 2, s_t // 2
                else: filter_t, pad_t = s_t * 2 + 1, s_t // 2 + 1
                assert d_t == 1, "Only d_t == 1 is supported for now"
                block = nn.Sequential(
                    nn.Conv1d(input_emb_width if i==0 else widths[i-1], widths[i], filter_t, s_t, pad_t),
                    Resnet1D(widths[i], depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out, res_scale), )
                blocks.append(block)
                # width = min(width * (m_conv ** depth), output_emb_width)
                # start = False
            block = nn.Conv1d(widths[i], output_emb_width, 3, 1, 1)
            blocks.append(block)
        else: # Not implemented yet
            pass
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class DecoderConvBlock(nn.Module):
    def __init__(self, input_emb_width, widths: List, output_emb_width, down_t, stride_t, depth, m_conv,
                 dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False,
                 reverse_decoder_dilation=False, checkpoint_res=False):
        super().__init__()
        blocks = []
        widths = list(reversed(widths))

        if isinstance(stride_t, (tuple, list, ListConfig)):
            block = nn.Conv1d(output_emb_width, widths[0], 3, 1, 1)
            blocks.append(block)
            for i, (s_t, d_t) in enumerate(zip(stride_t, down_t)):
                if s_t % 2 == 0: filter_t, pad_t = s_t * 2, s_t // 2
                else: filter_t, pad_t = s_t * 2 + 1, s_t // 2 + 1
                assert d_t == 1, "Only d_t == 1 is supported for now"
                block = nn.Sequential(
                    Resnet1D(widths[i], depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out=zero_out,
                                res_scale=res_scale, reverse_dilation=reverse_decoder_dilation,
                                checkpoint_res=checkpoint_res),
                    nn.ConvTranspose1d(widths[i], input_emb_width if i == len(down_t)-1 else widths[i+1], filter_t, s_t, pad_t))

                blocks.append(block)
        else: # Not implemented yet
            pass
  
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        outs = [] # we need to extract the intermediate features for conditioning the codec decoder
        current_feature = x

        for i, block in enumerate(self.model):
            if isinstance(block, nn.Sequential):
                for j, sub_block in enumerate(block):
                    if isinstance(sub_block, nn.ConvTranspose1d):
                        current_feature = sub_block(current_feature)
                        outs.append(current_feature)
                    else:
                        current_feature = sub_block(current_feature)
            else:
                current_feature = block(current_feature)
        
        return outs
        # return self.model(x)


class F0Encoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t, strides_t, width, depth, channel_growth_rate, m_conv, dilation_growth_rate=3,):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t
        self.width_list = build_width_list(width, levels, downs_t, channel_growth_rate, output_emb_width)

        block_kwargs_copy = dict(depth=depth, m_conv=m_conv,
                                 dilation_growth_rate=dilation_growth_rate)
        if 'reverse_decoder_dilation' in block_kwargs_copy:
            del block_kwargs_copy['reverse_decoder_dilation']
        level_block = lambda level, down_t, stride_t: EncoderConvBlock(input_emb_width,
            self.width_list[level], output_emb_width, down_t, stride_t,
            **block_kwargs_copy)
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))
        
        # print("Encoder level blocks: ", self.level_blocks)
        print("Number of parameters in f0encoder: ", sum(p.numel() for p in self.parameters()) / 1e6)
        print("Number of parameters in f0encoder level blocks: ", sum(p.numel() for p in self.level_blocks.parameters()) / 1e6)

    def forward(self, x):
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        if self.input_emb_width == 1: x = x[:, :1, :]  # use only f0 for encoding
        assert_shape(x, (N, emb, T)) # [B, emb(=1), T(=200)]
        xs = []

        # 64, 32, ...
        iterator = zip(list(range(self.levels)), self.downs_t, self.strides_t)
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            if isinstance(stride_t, (tuple, list, ListConfig)):
                # emb, T = self.output_emb_width, T // np.prod([s ** d for s, d in zip(stride_t, down_t)])
                emb, T = self.output_emb_width, T // np.prod([s for s in zip(stride_t)])
            else:
                pass
            assert_shape(x, (N, emb, T))
            xs.append(x)
        
        assert self.levels == 1, "Only one level is supported for now"

        return xs[0] # [B, emb(=128), T(=12)]


class F0Decoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t, strides_t, width, depth, channel_growth_rate, m_conv,
                 dilation_growth_rate=3,):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t
        self.width_list = build_width_list(width, levels, downs_t, channel_growth_rate, output_emb_width)

        level_block = lambda level, down_t, stride_t: DecoderConvBlock(
            input_emb_width, self.width_list[-level-1], output_emb_width, down_t, stride_t,
            **dict(depth=depth, m_conv=m_conv,
                   dilation_growth_rate=dilation_growth_rate))

        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

        print("Number of parameters in f0decoder: ", sum(p.numel() for p in self.parameters()) / 1e6)
        print("Number of parameters in f0decoder level blocks: ", sum(p.numel() for p in self.level_blocks.parameters()) / 1e6)

    def forward(self, xs, all_levels=True):
        # if all_levels:
        #     assert len(xs) == self.levels
        # else:
        #     assert len(xs) == 1
        # x = xs[-1]
        x = xs
        N, T = x.shape[0], x.shape[-1]
        emb = self.output_emb_width
        assert_shape(x, (N, emb, T))

        outs = []
        # 32, 64 ...
        outs.append(x)
        iterator = reversed(list(zip(list(range(self.levels)), self.downs_t, self.strides_t)))
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            # if isinstance(stride_t, (tuple, list, ListConfig)):
                # emb, T = 2, T * np.prod([s ** d for s, d in zip(stride_t, down_t)])
            # else:
                # pass
            # assert_shape(x[-1], (N, emb, T))
            outs.extend(x)

        # x = self.out(x[-1])
        # outs.append(x)
        return outs