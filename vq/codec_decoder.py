import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.nn.utils.weight_norm import _weight_norm
from .vq.residual_vq import ResidualVQ
from .vq.simvq import SimVQ1D, KmeansVectorQuantizer_
from .module import WNConv1d, DecoderBlock, CrossAttentionBlock, build_codec_activation
from .temporal_config import CodecDecoderSpeakerConditionConfig, CodecDecoderTemporalConfig
from .alias_free_torch import *
from .rndvoc_decoder import RNDLatentDecoder
from .vocos_decoder import APNFormerDecoder, VocosFormerDecoder, VocosLatentDecoder
from termcolor import colored
from .hyper_lstm import HyperLSTM

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        if isinstance(getattr(m, "weight", None), nn.parameter.UninitializedParameter):
            return
        if m.bias is not None:
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)


class BroadcastConditionedPointwiseConv1d(nn.Module):
    """Exact 1x1 conv over [x, repeated_condition] without materializing the repeated tensor."""

    def __init__(self, input_dim: int, condition_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.condition_dim = int(condition_dim)
        self.conv = WNConv1d(self.input_dim + self.condition_dim, output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        weight = _weight_norm(self.conv.weight_v, self.conv.weight_g, 0)
        bias = self.conv.bias
        x_out = F.conv1d(x, weight[:, :self.input_dim, :], bias=bias)
        cond_out = F.linear(condition, weight[:, self.input_dim:, 0]).unsqueeze(-1)
        return x_out + cond_out


class SpeakerStageFuser(nn.Module):
    def __init__(self, input_dim: int, condition_dim: int):
        super().__init__()
        self.cond_conv = BroadcastConditionedPointwiseConv1d(
            input_dim=input_dim,
            condition_dim=condition_dim,
            output_dim=input_dim,
        )
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.out_conv = WNConv1d(input_dim, input_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = self.cond_conv(x, condition)
        x = self.activation(x)
        x = self.out_conv(x)
        return x

class CodecDecoder(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 upsample_initial_channel=1536,
                 ngf=48,
                 temporal: Optional[CodecDecoderTemporalConfig] = None,
                 up_ratios=(5, 5, 2, 2, 2),
                 dilations=(1, 3, 9),
                 quantizer_type="rvq",
                 vq_num_quantizers=1,
                 vq_dim=1024,
                 vq_commit_weight=0.25,
                 vq_weight_init=False,
                 vq_full_commit_loss=False,
                 quantizer_force_fp32: bool = False,
                 codebook_size=8192,
                 codebook_dim=8,
                 speaker_condition=False,
                 condition_dim=1024,
                 activation_type='SnakeBeta',
                 leaky_relu_params=None,
                 snake_logscale=True,
                 snake_lite_taylor_degree: int = 8,
                 simvq_linear_layer_type='linear',
                 ema_decay=0.0,
                 freeze_vqw2v_encoder=False,
                 use_vqw2v_encoder=False,
                 freeze_kvq_emb=True,
                 use_vqw2v_embed=True,
                 f0_condition=False,
                 f0_start_layer=0,
                 f0_end_layer=None,
                 f0_every=1,
                 f0_width_list=None,
                 f0_speaker_condition=False,
                 use_stage_speaker_film=True,
                 use_mhca=False,
                 spk_cond_use_concat=False,
                 mhca_num_heads=8,
                 mhca_dropout=0.1,
                 mhca_key_dim=128,
                 mhca_use_sdpa: Optional[bool] = None,
                 mhca_start_layer=0,
                 mhca_end_layer=None,
                 mhca_every=1,
                 decoder_type='default',
                 rndvoc_sampling_rate=16000,
                 rndvoc_n_fft=None,
                 rndvoc_hop_size=None,
                 rndvoc_win_size=None,
                 rndvoc_input_channel=256,
                 rndvoc_hidden_channel=256,
                 rndvoc_squeeze_size=64,
                 rndvoc_null_nstage=6,
                 rndvoc_nrep=2,
                 rndvoc_kernel_size=7,
                 rndvoc_causal=False,
                 rndvoc_use_rnd=True,
                 rndvoc_time_type='convnext_v2',
                 rndvoc_freq_type='shuffler',
                 rndvoc_f0_stage_dims=None,
                 vocos_sampling_rate=16000,
                 vocos_n_fft=None,
                 vocos_hop_size=None,
                 vocos_win_size=None,
                 vocos_dim=512,
                 vocos_intermediate_dim=1536,
                 vocos_num_layers=8,
                 vocos_kernel_size=7,
                 vocos_padding='same',
                 vocos_f0_stage_dims=None,
                 vocos_backbone='convnext',
                 vocos_num_heads=8,
                 vocos_window_size=None,
                 vocos_attn_window_size=None,
                 vocos_attention_impl='flash_attn',
                 vocos_use_rope=True,
                 vocos_rope_base=10000.0,
                 vocos_max_position_embeddings=2048,
                 vocos_qk_norm=True,
                 vocos_norm_type='rmsnorm',
                 vocos_norm_eps=1e-2,
                 vocos_dropout=0.0,
                 vocos_ffn_type='swiglu',
                 vocos_ffn_mult=4.0,
                 vocos_layerscale_gamma_init=1.0,
                 vocos_apn_head_hidden_dim=None,
                 vocos_apn_phase_norm_eps=1e-6,
                 vocos_mhca_every=None,
                 vocos_mhca_start_layer=None,
                 vocos_mhca_end_layer=None,
                 vocos_mhca_align_to_mamba=True,
                 vocos_mhca_position='after',
                 vocos_mamba_every=2,
                 vocos_mamba_start_layer=None,
                 vocos_mamba_end_layer=None,
                 vocos_mamba=None,
                 use_gradient_checkpointing=False,
                 use_split_condition_optimization: bool = True,
                #  zero_out_all_unvoiced=True,
                ):
        super().__init__()
        self.temporal = temporal or CodecDecoderTemporalConfig()
        self._decoder_temporal_type = self.temporal.backbone

        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios
        self.f0_condition = bool(f0_condition)
        self.f0_start_layer = int(f0_start_layer)
        self.f0_end_layer = f0_end_layer
        self.f0_every = max(1, int(f0_every))
        self.f0_width_list = []
        self.f0_block_enabled = [False] * len(up_ratios)
        self.spk_start_layer = int(mhca_start_layer)
        self.spk_end_layer = mhca_end_layer
        self.spk_every = max(1, int(mhca_every))
        self.spk_block_enabled = [False] * len(up_ratios)
        self.spk_concat_block_enabled = [False] * len(up_ratios)
        self.spk_film_block_enabled = [False] * len(up_ratios)
        self.use_mhca = use_mhca
        self.spk_cond_use_concat = bool(spk_cond_use_concat)
        self.use_stage_speaker_film = bool(use_stage_speaker_film)
        self.mhca_key_dim = mhca_key_dim
        self.mhca_start_layer = mhca_start_layer
        self.mhca_end_layer = mhca_end_layer
        self.mhca_every = max(1, int(mhca_every))
        self.decoder_type = decoder_type.lower()
        self.vocos_pre_proj = None
        self.use_split_condition_optimization = bool(use_split_condition_optimization)

        if not use_vqw2v_encoder: # using any encoder
            if quantizer_type == "rvq":
                if quantizer_force_fp32:
                    print(colored("CodecDecoder RVQ quantizer runs under fp32 autocast-disabled mode", "yellow", attrs=['bold']))
                self.quantizer = ResidualVQ(
                    num_quantizers=vq_num_quantizers,
                    dim=vq_dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    commitment=vq_commit_weight,
                    force_quantization_f32=quantizer_force_fp32,
                    weight_init=vq_weight_init,
                    full_commit_loss=vq_full_commit_loss,)
            
            elif quantizer_type == "simvq": 
                self.quantizer = SimVQ1D(
                    codebook_size=codebook_size,
                    codebook_dim=vq_dim,
                    commitment=vq_commit_weight,
                    )
            elif quantizer_type == "kvq":
                print(colored(f"Using KmeansVectorQuantizer_ with trainable codec encoder", "yellow", attrs=['bold']))
                print((colored(f"freeze_kvq_emb: {freeze_kvq_emb}", "yellow", attrs=['bold'])))
                print((colored(f"use_vqw2v_embed: {use_vqw2v_embed}", "yellow", attrs=['bold'])))
                self.quantizer = KmeansVectorQuantizer_(groups=2,
                                                combine_groups=True,
                                                dim=512,
                                                num_vars=320,
                                                vq_dim=512,
                                                time_first=False,
                                                gamma=0.25,
                                                freeze_embed=freeze_kvq_emb,
                                                use_vqw2v_embed=use_vqw2v_embed,
                                                ) 
            
        if use_vqw2v_encoder:
            if quantizer_type == "kvq":
                if freeze_vqw2v_encoder: print(colored(f"Freezing VQ-Wav2Vec encoder and using KmeansVectorQuantizer_", "yellow", attrs=['bold']))
                else: print(colored(f"Using KmeansVectorQuantizer_ with trainable VQ-Wav2Vec encoder", "yellow", attrs=['bold']))
                print((colored(f"freeze_kvq_emb: {freeze_kvq_emb}", "yellow", attrs=['bold'])))
                print((colored(f"use_vqw2v_embed: {use_vqw2v_embed}", "yellow", attrs=['bold'])))
                self.quantizer = KmeansVectorQuantizer_(groups=2,
                                                combine_groups=True,
                                                dim=512,
                                                num_vars=320,
                                                vq_dim=512,
                                                time_first=False,
                                                gamma=0.25,
                                                freeze_embed=freeze_kvq_emb,
                                                use_vqw2v_embed=use_vqw2v_embed,
                                                )
                # for p in self.quantizer.parameters():
                #     p.requires_grad = False
            elif quantizer_type == "simvq":
                if freeze_vqw2v_encoder: print(colored(f"Freezing VQ-Wav2Vec encoder and using SimVQ", "yellow", attrs=['bold']))
                else: print(colored(f"Using SimVQ with trainable VQ-Wav2Vec encoder", "yellow", attrs=['bold']))
                print(colored(f"WARNING: quantizer_type==simvq option changed KmeansVectorQuantizer_SimVQ to simple SimVQ1D from 251021"))
                # self.quantizer = KmeansVectorQuantizer_SimVQ(groups=2,
                #                                 combine_groups=True,
                #                                 dim=512,
                #                                 num_vars=320,
                #                                 vq_dim=512,
                #                                 time_first=False,
                #                                 gamma=0.25,
                #                                 simvq_linear_layer_type=simvq_linear_layer_type,
                #                                 ema_decay=ema_decay
                #                                 )
                self.quantizer = SimVQ1D(
                    codebook_size=codebook_size,
                    codebook_dim=vq_dim,
                    commitment=vq_commit_weight,
                    )

            elif quantizer_type == "rvq":
                print(colored(f"Using ResidualVQ for VQ-Wav2Vec encoder (possibly for vqw2v_enc_24kHz_480hs_tribase)", "yellow", attrs=['bold']))
                if quantizer_force_fp32:
                    print(colored("CodecDecoder RVQ quantizer runs under fp32 autocast-disabled mode", "yellow", attrs=['bold']))
                self.quantizer = ResidualVQ(
                    num_quantizers=vq_num_quantizers,
                    dim=vq_dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    commitment=vq_commit_weight,
                    force_quantization_f32=quantizer_force_fp32,
                    weight_init=vq_weight_init,
                    full_commit_loss=vq_full_commit_loss,)
            
        total_decoder_blocks = len(self.up_ratios)
        if speaker_condition:
            self.spk_start_layer = max(0, min(int(self.spk_start_layer), total_decoder_blocks))
            if self.spk_end_layer is None:
                self.spk_end_layer = total_decoder_blocks
            else:
                self.spk_end_layer = max(self.spk_start_layer, min(int(self.spk_end_layer), total_decoder_blocks))
            for idx in range(total_decoder_blocks):
                self.spk_block_enabled[idx] = (
                    self.spk_start_layer <= idx < self.spk_end_layer
                    and ((idx - self.spk_start_layer) % self.spk_every == 0)
                )
                self.spk_concat_block_enabled[idx] = bool(self.spk_cond_use_concat and self.spk_block_enabled[idx])
                self.spk_film_block_enabled[idx] = bool(self.use_stage_speaker_film and self.spk_block_enabled[idx])
        else:
            self.spk_start_layer = 0
            self.spk_end_layer = 0

        if self.f0_condition:
            self.f0_width_list = list(reversed(f0_width_list[0]))
            # if not self.zero_out_all_unvoiced:
            #     print(colored(f"Introducing null f0 embeddings for unvoiced frames", "yellow", attrs=['bold']))
            #     self.f0_null_embeddings = nn.ParameterList()
            #     for w in self.f0_width_list:
            #         self.f0_null_embeddings.append(nn.Parameter(torch.randn(1, w, 1)))
            self.f0_start_layer = max(0, min(int(self.f0_start_layer), total_decoder_blocks))
            if self.f0_end_layer is None:
                self.f0_end_layer = total_decoder_blocks
            else:
                self.f0_end_layer = max(self.f0_start_layer, min(int(self.f0_end_layer), total_decoder_blocks))
            for idx in range(total_decoder_blocks):
                self.f0_block_enabled[idx] = (
                    self.f0_start_layer <= idx < self.f0_end_layer
                    and ((idx - self.f0_start_layer) % self.f0_every == 0)
                )
        else:
            self.f0_start_layer = 0
            self.f0_end_layer = 0

        print(colored(
            "CodecDecoder init: "
            f"decoder_type={self.decoder_type}, in_channels={in_channels}, "
            f"upsample_initial_channel={upsample_initial_channel}, up_ratios={list(up_ratios)}, "
            f"hop_length={self.hop_length}, quantizer_type={quantizer_type}, "
            f"speaker_condition={speaker_condition}, f0_condition={self.f0_condition}, "
            f"spk_layer_range=({self.spk_start_layer}, {self.spk_end_layer}), spk_every={self.spk_every}, "
            f"f0_layer_range=({self.f0_start_layer}, {self.f0_end_layer}), f0_every={self.f0_every}, "
            f"f0_speaker_condition={f0_speaker_condition}, "
            f"use_stage_speaker_film={use_stage_speaker_film}, use_mhca={use_mhca}, "
            "global_condition_fuser=True(vocos when speaker_condition=True), "
            f"activation_type={activation_type}",
            "cyan",
            attrs=["bold"],
        ))
        if self.f0_width_list:
            print(colored(f"CodecDecoder f0_width_list(reversed)={self.f0_width_list}", "cyan"))
            print(colored(f"CodecDecoder f0 enabled on decoder blocks {[idx for idx, enabled in enumerate(self.f0_block_enabled) if enabled]}", "cyan"))
        if speaker_condition:
            print(colored(f"CodecDecoder speaker enabled on decoder blocks {[idx for idx, enabled in enumerate(self.spk_block_enabled) if enabled]}", "cyan"))
        if any(self.spk_concat_block_enabled):
            print(colored(f"CodecDecoder speaker concat enabled on decoder blocks {[idx for idx, enabled in enumerate(self.spk_concat_block_enabled) if enabled]}", "cyan"))
        if any(self.spk_film_block_enabled):
            print(colored(f"CodecDecoder speaker FiLM enabled on decoder blocks {[idx for idx, enabled in enumerate(self.spk_film_block_enabled) if enabled]}", "cyan"))

        if self.decoder_type == 'rndvoc':
            if activation_type != 'LeakyReLU':
                print(colored("RNDVoC decoder path ignores Snake/SnakeLite activations; using latent-aware RNDVoC blocks instead.", "yellow"))
            rndvoc_pre_temporal = None
            if self.temporal.use:
                tt = self.temporal.backbone
                if tt not in ('lstm', 'mamba'):
                    raise ValueError(
                        f"Unsupported temporal type for rndvoc pre-RNN: {tt}. "
                        f"Only 'lstm' and 'mamba' are supported."
                    )
                _label = "ResLSTM" if tt == "lstm" else "MambaBlock"
                print(
                    colored(
                        f"Using rndvoc pre-RNN ({_label}) with {self.temporal.effective_num_layers()} layers and "
                        f"bidirectional={self.temporal.bidirectional} at dim={in_channels}",
                        "blue",
                        attrs=["bold"],
                    )
                )
                rndvoc_pre_temporal = {
                    "use": True,
                    "type": tt,
                    "num_layers": self.temporal.effective_num_layers(),
                    "bidirectional": self.temporal.bidirectional,
                    "mamba": self.temporal.mamba,
                }
            hop_size = rndvoc_hop_size if rndvoc_hop_size is not None else self.hop_length
            self.rnd_decoder = RNDLatentDecoder(
                latent_dim=in_channels,
                sampling_rate=rndvoc_sampling_rate,
                hop_size=hop_size,
                n_fft=rndvoc_n_fft,
                win_size=rndvoc_win_size,
                input_channel=rndvoc_input_channel,
                hidden_channel=rndvoc_hidden_channel,
                squeeze_size=rndvoc_squeeze_size,
                null_nstage=rndvoc_null_nstage,
                nrep=rndvoc_nrep,
                kernel_size=rndvoc_kernel_size,
                causal=rndvoc_causal,
                use_rnd=rndvoc_use_rnd,
                time_type=rndvoc_time_type,
                freq_type=rndvoc_freq_type,
                pre_temporal=rndvoc_pre_temporal,
                speaker_condition=speaker_condition,
                condition_dim=condition_dim,
                f0_condition=f0_condition,
                f0_start_layer=f0_start_layer,
                f0_end_layer=f0_end_layer,
                f0_every=f0_every,
                f0_speaker_condition=f0_speaker_condition,
                use_stage_speaker_film=use_stage_speaker_film,
                f0_stage_dims=rndvoc_f0_stage_dims,
                leaky_relu_params=leaky_relu_params,
                use_mhca=use_mhca,
                mhca_num_heads=mhca_num_heads,
                mhca_dropout=mhca_dropout,
                mhca_key_dim=mhca_key_dim,
                mhca_use_sdpa=mhca_use_sdpa,
            )
            self.model = None
            self.reset_parameters()
            return
        if self.decoder_type == 'vocos':
            vocos_input_dim = in_channels
            if in_channels != vocos_dim:
                print(
                    colored(
                        f"Using vocos input projection: {in_channels} -> {vocos_dim}",
                        "blue",
                        attrs=["bold"],
                    )
                )
                self.vocos_pre_proj = WNConv1d(in_channels, vocos_dim, kernel_size=1)
                vocos_input_dim = vocos_dim

            hop_size = vocos_hop_size if vocos_hop_size is not None else self.hop_length
            self.vocos_decoder = VocosLatentDecoder(
                latent_dim=vocos_input_dim,
                sampling_rate=vocos_sampling_rate,
                hop_size=hop_size,
                n_fft=vocos_n_fft,
                win_size=vocos_win_size,
                dim=vocos_dim,
                intermediate_dim=vocos_intermediate_dim,
                num_layers=vocos_num_layers,
                kernel_size=vocos_kernel_size,
                padding=vocos_padding,
                speaker_condition=speaker_condition,
                condition_dim=condition_dim,
                f0_condition=f0_condition,
                f0_start_layer=f0_start_layer,
                f0_end_layer=f0_end_layer,
                f0_every=f0_every,
                f0_speaker_condition=f0_speaker_condition,
                use_stage_speaker_film=use_stage_speaker_film,
                f0_stage_dims=vocos_f0_stage_dims,
                leaky_relu_params=leaky_relu_params,
                use_mhca=use_mhca,
                spk_cond_use_concat=spk_cond_use_concat,
                mhca_num_heads=mhca_num_heads,
                mhca_dropout=mhca_dropout,
                mhca_key_dim=mhca_key_dim,
                mhca_use_sdpa=mhca_use_sdpa,
                num_heads=vocos_num_heads,
                window_size=vocos_window_size,
                backbone_type=vocos_backbone,
                temporal=self.temporal,
                mhca_start_layer=mhca_start_layer,
                mhca_end_layer=mhca_end_layer,
                mhca_every=mhca_every,
            )
            self.model = None
            self.reset_parameters()
            return
        if self.decoder_type in {'vocosformer', 'apnformer'}:
            hybrid_input_dim = in_channels
            if in_channels != vocos_dim:
                print(
                    colored(
                        f"Using {self.decoder_type} input projection: {in_channels} -> {vocos_dim}",
                        "blue",
                        attrs=["bold"],
                    )
                )
                self.vocos_pre_proj = WNConv1d(in_channels, vocos_dim, kernel_size=1)
                hybrid_input_dim = vocos_dim

            hop_size = vocos_hop_size if vocos_hop_size is not None else self.hop_length
            vocos_decoder_cls = APNFormerDecoder if self.decoder_type == 'apnformer' else VocosFormerDecoder
            vocos_decoder_kwargs = dict(
                latent_dim=hybrid_input_dim,
                sampling_rate=vocos_sampling_rate,
                hop_size=hop_size,
                n_fft=vocos_n_fft,
                win_size=vocos_win_size,
                dim=vocos_dim,
                intermediate_dim=vocos_intermediate_dim,
                num_layers=vocos_num_layers,
                speaker_condition=speaker_condition,
                condition_dim=condition_dim,
                f0_condition=f0_condition,
                f0_start_layer=f0_start_layer,
                f0_end_layer=f0_end_layer,
                f0_every=f0_every,
                f0_speaker_condition=f0_speaker_condition,
                use_stage_speaker_film=use_stage_speaker_film,
                f0_stage_dims=vocos_f0_stage_dims,
                leaky_relu_params=leaky_relu_params,
                use_mhca=use_mhca,
                spk_cond_use_concat=spk_cond_use_concat,
                mhca_num_heads=mhca_num_heads,
                mhca_dropout=mhca_dropout,
                mhca_key_dim=mhca_key_dim,
                mhca_use_sdpa=mhca_use_sdpa,
                num_heads=vocos_num_heads,
                window_size=vocos_window_size,
                attn_window_size=vocos_attn_window_size,
                attention_impl=vocos_attention_impl,
                use_rope=vocos_use_rope,
                rope_base=vocos_rope_base,
                max_position_embeddings=vocos_max_position_embeddings,
                qk_norm=vocos_qk_norm,
                norm_type=vocos_norm_type,
                norm_eps=vocos_norm_eps,
                dropout=vocos_dropout,
                ffn_type=vocos_ffn_type,
                ffn_mult=vocos_ffn_mult,
                layerscale_gamma_init=vocos_layerscale_gamma_init,
                temporal=self.temporal,
                mhca_start_layer=mhca_start_layer,
                mhca_end_layer=mhca_end_layer,
                mhca_every=mhca_every,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
            if self.decoder_type == 'apnformer':
                vocos_decoder_kwargs.update(
                    apn_head_hidden_dim=vocos_apn_head_hidden_dim,
                    apn_phase_norm_eps=vocos_apn_phase_norm_eps,
                )
            self.vocos_decoder = vocos_decoder_cls(**vocos_decoder_kwargs)
            self.model = None
            self.reset_parameters()
            return
        if self.decoder_type != 'default':
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")

        channels = upsample_initial_channel
        layers = [WNConv1d(in_channels, channels, kernel_size=7, padding=3)]
        
        if self.temporal.use:
            tt = self.temporal.backbone
            if tt in ('lstm', 'mamba'):
                _label = "LSTM" if tt == "lstm" else "Mamba"
                print(colored(
                    f"Using {_label} with {self.temporal.effective_num_layers()} layers and bidirectional={self.temporal.bidirectional}",
                    "blue",
                    attrs=['bold'],
                ))
                layers += [self.temporal.build_res_path(channels)]
            elif tt in ('statichyperlstm', 'dynamichyperlstm'):
                print(colored(f"Using {tt} with {self.temporal.effective_num_layers()} layers and bidirectional={self.temporal.bidirectional}", "blue", attrs=['bold']))
                layers += [
                    HyperLSTM(input_size=channels, hidden_size=channels, cond_size=(condition_dim+self.f0_width_list[0] if self.f0_condition else condition_dim) if speaker_condition else 0,
                            residual=True,
                            num_layers=self.temporal.effective_num_layers(),
                            bidirectional=self.temporal.bidirectional,
                            hyper='static' if tt=='statichyperlstm' else 'dynamic',
                            )
                ]
            else:
                raise ValueError(
                    f"Unsupported decoder temporal type: {tt}. "
                    f"Supported: 'lstm', 'mamba', 'statichyperlstm', 'dynamichyperlstm'."
                )
        
        for i, stride in enumerate(up_ratios):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            # In the default decoder path, the legacy in-block speaker-conditioned
            # path now maps to explicit concat-style speaker conditioning only.
            block_speaker_condition = bool(self.spk_concat_block_enabled[i])
            block_f0_condition = bool(self.f0_condition and self.f0_block_enabled[i])
            # Legacy f0_speaker_condition is no longer used as a separate gate.
            # When speaker conditioning and f0 conditioning overlap on a block,
            # use the joint speaker+f0 path automatically.
            block_f0_speaker_condition = bool(block_speaker_condition and block_f0_condition)
            layers += [DecoderBlock(input_dim, output_dim, stride, dilations, block_speaker_condition, condition_dim=condition_dim, f0_condition=block_f0_condition,
                                    f0_condition_dim=self.f0_width_list[i + (len(self.f0_width_list) - len(self.up_ratios))] if block_f0_condition else None,
                                    f0_speaker_condition=block_f0_speaker_condition,
                                    activation_type=activation_type,
                                    leaky_relu_params=leaky_relu_params,
                                    snake_lite_taylor_degree=snake_lite_taylor_degree,
                                    use_split_condition_optimization=self.use_split_condition_optimization,
                                    )]

        final_speaker_condition = bool(self.spk_concat_block_enabled and self.spk_concat_block_enabled[-1])
        activation = build_codec_activation(
            dim=output_dim,
            activation_type=activation_type,
            leaky_relu_params=leaky_relu_params,
            speaker_condition=final_speaker_condition,
            condition_dim=condition_dim,
            alpha_logscale=snake_logscale,
            snake_lite_taylor_degree=snake_lite_taylor_degree,
        )
        # elif speaker_condition and f0_speaker_condition:
        #     activation = Activation1dWithCondition(activation=activations.SnakeBetaWithTimeVaryingCondition(output_dim, condition_dim + (self.f0_width_list[-1] if f0_condition else 0), alpha_logscale=True))
        layers += [
            activation,
            WNConv1d(output_dim, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

        self.speaker_stage_fusers = nn.ModuleList()
        self.speaker_stage_film = nn.ModuleList()
        for i, _stride in enumerate(up_ratios):
            input_dim = channels // 2**i
            # Keep legacy default-decoder behavior: speaker concat happens only
            # inside DecoderBlock, not through an additional pre-block fuser.
            self.speaker_stage_fusers.append(nn.Identity())

            if self.spk_film_block_enabled[i]:
                self.speaker_stage_film.append(
                    nn.Linear(condition_dim, input_dim * 2)
                )
            else:
                self.speaker_stage_film.append(nn.Identity())
        
        # Cross-Attention Block for each DecoderBlock
        self.mhca_block_enabled = [False] * len(up_ratios)
        if use_mhca:
            print(colored(f"Using Cross-Attention Blocks for speaker conditioning at each DecoderBlock", "cyan", attrs=['bold']))
            self.mhca_list = nn.ModuleList()
            mhca_cfg = CodecDecoderSpeakerConditionConfig(
                use=use_mhca,
                type="mhca",
                num_heads=mhca_num_heads,
                dropout=mhca_dropout,
                start_layer=mhca_start_layer,
                end_layer=mhca_end_layer,
                every=mhca_every,
            )
            start, end, every = mhca_cfg.resolve_block_range(len(up_ratios))
            for i, stride in enumerate(up_ratios):
                input_dim = channels // 2**i
                enabled = start <= i < end and ((i - start) % every == 0)
                self.mhca_block_enabled[i] = enabled
                self.mhca_list.append(
                    CrossAttentionBlock(
                        query_dim=input_dim,
                        key_dim=mhca_key_dim,  # x_quantized_projected dim (out_dim or latent_dim)
                        num_heads=mhca_num_heads,
                        ffn_hidden_dim=None,  # defaults to 4 * query_dim
                        dropout=mhca_dropout,
                        use_sdpa=mhca_use_sdpa,
                    )
                    if enabled else nn.Identity()
                )
            print(colored(f"Created {sum(self.mhca_block_enabled)} Cross-Attention Blocks for DecoderBlocks (key_dim={mhca_key_dim})", "cyan"))
            print(colored(f"MHCA enabled on decoder blocks {[idx for idx, enabled in enumerate(self.mhca_block_enabled) if enabled]}", "cyan"))
        else:
            self.mhca_list = None
        
        self.reset_parameters()
        self.latest_decoder_aux = None

    def forward(self, x, total_step=None, vq=True, spk_cond=None, f0_conds=None, vuv=None):
        if vq is True:
            x, q, commit_loss, perplexity, active_num = self.quantizer(x, total_step=total_step, produce_targets=True)
            return x, q, commit_loss, perplexity, active_num

        # Parse speaker conditioning: when available, tuple is (global_spk, speaker_tokens)
        if isinstance(spk_cond, tuple):
            gq_vector, x_quantized = spk_cond
            spk_cond_global = gq_vector
        else:
            spk_cond_global = spk_cond
            x_quantized = None

        if self.decoder_type == 'rndvoc':
            self.latest_decoder_aux = None
            return self.rnd_decoder(x, spk_cond=spk_cond_global, f0_conds=f0_conds, x_quantized=x_quantized)
        if self.decoder_type in {'vocos', 'vocosformer', 'apnformer'}:
            if self.vocos_pre_proj is not None:
                x = self.vocos_pre_proj(x)
            wav = self.vocos_decoder(x, spk_cond=spk_cond_global, f0_conds=f0_conds, x_quantized=x_quantized)
            self.latest_decoder_aux = getattr(self.vocos_decoder, 'latest_aux', None)
            return wav
        
        # x = self.model(x, condition=condition)
        # decoder_block_num = 0
        decoder_block_num = len(self.f0_width_list) - len(self.up_ratios) if self.f0_condition else 0
        decoder_block_idx = 0
        for i, layer in enumerate(self.model):
            if isinstance(layer, Activation1dWithCondition):
                x = layer(x, spk_cond_global)
            elif isinstance(layer, DecoderBlock):
                if (
                    decoder_block_idx < len(self.spk_film_block_enabled)
                    and self.spk_film_block_enabled[decoder_block_idx]
                    and spk_cond_global is not None
                ):
                    gamma_beta = self.speaker_stage_film[decoder_block_idx](spk_cond_global)
                    gamma, beta = gamma_beta.chunk(2, dim=1)
                    x = x * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

                # Apply Cross-Attention Block before each DecoderBlock
                if (
                    self.use_mhca
                    and x_quantized is not None
                    and self.mhca_list is not None
                    and decoder_block_idx < len(self.mhca_block_enabled)
                    and self.mhca_block_enabled[decoder_block_idx]
                ):
                    x = self.mhca_list[decoder_block_idx](x, x_quantized)
                
                # f0_cond = f0_conds[decoder_block_num].detach() if self.f0_condition else None
                f0_cond = (
                    f0_conds[decoder_block_num]
                    if self.f0_condition and self.f0_block_enabled[decoder_block_idx]
                    else None
                )
                x = layer(x, spk_cond_global, f0_cond=f0_cond)
                decoder_block_num += 1
                decoder_block_idx += 1
            elif isinstance(layer, HyperLSTM):
                # condition speaker and f0, with speaker broadcasted to the time dimension
                assert spk_cond_global is not None, "spk_cond is required for HyperLSTM"
                if self.f0_condition:
                    assert f0_conds is not None, "f0_conds is required for HyperLSTM with f0_condition"
                    cond = torch.cat([spk_cond_global.unsqueeze(1).expand(-1, x.size(2), -1), f0_conds[decoder_block_num].transpose(1,2)], dim=-1)
                else:
                    cond = spk_cond_global.unsqueeze(1).expand(-1, x.size(2), -1)
                x, _ = layer(x.transpose(1,2), cond) # (B, C, T) -> (B, T, C) -> (B, C, T)
                x = x.transpose(1,2)
            else:
                x = layer(x)
        
        self.latest_decoder_aux = None
        return x

    def vq2emb(self, vq):
        self.quantizer = self.quantizer.eval()
        x = self.quantizer.vq2emb(vq)
        return x

    def get_emb(self):
        self.quantizer = self.quantizer.eval()
        embs = self.quantizer.get_emb()
        return embs

    def inference_vq(self, vq):
        x = vq[None,:,:]
        if self.decoder_type == 'rndvoc':
            x = self.rnd_decoder(x)
        elif self.decoder_type in {'vocos', 'vocosformer', 'apnformer'}:
            x = self.vocos_decoder(x)
        else:
            x = self.model(x)
        return x

    def inference_0(self, x):
        x, q, loss, perp = self.quantizer(x)
        if self.decoder_type == 'rndvoc':
            x = self.rnd_decoder(x)
        else:
            x = self.vocos_decoder(x) if self.decoder_type in {'vocos', 'vocosformer', 'apnformer'} else self.model(x)
        return x, None
    
    def inference(self, x):
        if self.decoder_type == 'rndvoc':
            x = self.rnd_decoder(x)
        else:
            x = self.vocos_decoder(x) if self.decoder_type in {'vocos', 'vocosformer', 'apnformer'} else self.model(x)
        return x, None


    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)

    def get_latest_decoder_aux(self):
        return self.latest_decoder_aux
