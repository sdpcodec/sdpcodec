import os
import random
import hydra
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from vq import CodecEncoder, CodecDecoder
from vq.temporal_config import (
    CodecDecoderF0ConditionConfig,
    CodecDecoderSpeakerConditionConfig,
    CodecDecoderTemporalConfig,
    CodecDecoderVocosConfig,
    CodecTemporalConfig,
    F0CodecEncoderConfig,
    F0CodecDecoderConfig,
    F0CodecSpeakerConditionConfig,
)
from vq.ssl_codec_wrappers import VQW2VCodecEncoderWrapper
from vq.speaker.speaker_encoder import (
    SpeakerEncoder,
    build_speaker_quantizer_kwargs,
    resolve_speaker_encoder_token_dim,
)
from vq.f0.codec import F0Encoder, F0Decoder
from vq.vq.factorized_vector_quantize import FactorizedVectorQuantize
from module import HiFiGANMultiPeriodDiscriminator, SpecDiscriminator
from criterions import GANLoss, MultiResolutionMelSpectrogramLoss
from common.schedulers import WarmupLR
from metrics.metrics import STOI, PESQ
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio, ScaleInvariantSignalDistortionRatio
from pytorch_lightning.utilities.model_summary import ModelSummary
from contextlib import nullcontext
import torchaudio
from termcolor import colored
from s3prl.nn import S3PRLUpstream
from torch.amp import autocast
import matplotlib.pyplot as plt
import matplotlib
import wandb
import jiwer
from transformers import AutoProcessor, HubertForCTC
from torch.nn.parameter import UninitializedParameter

class TriCodecLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_unnormf0_mse_loss = getattr(self.cfg.model.f0_codec, 'use_unnormf0_mse_loss', False)
        self.ocwd = hydra.utils.get_original_cwd()
        self.construct_model()
        self.construct_criteria()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.stoi = STOI(in_sr=cfg.preprocess.audio.sr, sr=cfg.preprocess.audio.sr)
        self.pesq_wb = PESQ(in_sr=cfg.preprocess.audio.sr, sr=16000, mode='wb', n_processor=cfg.train.num_pesq_processor)
        # self.pesq_nb = PESQ(in_sr=cfg.preprocess.audio.sr, sr=8000, mode='nb')
        self.si_snr = ScaleInvariantSignalNoiseRatio()
        self.si_sdr = ScaleInvariantSignalDistortionRatio()
        self.register_buffer('total_step', torch.tensor(0, dtype=torch.long))
        self.vc_f0 = False
        # self.init_mel_transformer(self.cfg.model.mel_params)
        
        # UTMOS and WavLM for validation (optional)
        self.use_val_utmos = getattr(cfg.train, 'use_val_utmos', False)
        self.use_val_wavlm = getattr(cfg.train, 'use_val_wavlm', False)
        self.use_val_wer = getattr(cfg.train, 'use_val_wer', False)
        self.val_hubert_model_name = getattr(cfg.train, 'val_hubert_model', 'facebook/hubert-large-ls960-ft')
        self.utmos_predictor = None
        self.wavlm_model = None
        self.hubert_processor = None
        self.hubert_model = None
        self._wer_transform = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.SubstituteRegexes({r"[_\u2010-\u2015\u2212-]+": " "}),
            jiwer.SubstituteRegexes({r"[^\w\s\uAC00-\uD7A3]": ""}),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ])
        
        if self.use_val_utmos or self.use_val_wavlm or self.use_val_wer:
            print(colored("Validation will include UTMOS/WavLM/WER metrics", "yellow"))

        # freeze → unfreeze 스케줄 설정 (없으면 비활성)
        self.unfreeze_encoder_step = getattr(cfg.train, 'unfreeze_encoder_step', None)
        self._encoder_frozen = None
        # 스케줄 활성 여부: vq_wav2vec이고 freeze_vqw2v_encoder=False면 스케줄 무시
        enccfg = self.cfg.model.codec_encoder
        self._freeze_schedule_enabled = self.unfreeze_encoder_step is not None
        if enccfg.use_vqw2v and not enccfg.freeze_vqw2v_encoder:
            self._freeze_schedule_enabled = False
            print(colored("freeze_vqw2v_encoder=False -> disabling unfreeze_encoder_step; vqw2v CodecEnc stays UNFROZEN.", "yellow"))
        else:
            print(colored(f"Encoder unfreeze step: {self.unfreeze_encoder_step}", "yellow"))
    
        # (선택) 동시 지정 안내
        if self.unfreeze_encoder_step is not None and getattr(self.cfg.model.codec_encoder, 'freeze_vqw2v_encoder', False):
            print(colored("Both train.unfreeze_encoder_step and codec_encoder.freeze_vqw2v_encoder are set. Step-based schedule will override encoder freeze after unfreeze step.", "yellow"))
        
        # 시각화용 버퍼
        self.val_step_plot_outputs = []

        # Mel 변환기 (필요시 파라미터 조정)
        sr, n_fft, hop_length, n_mels = 16000, 1024, 256, 80
        self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length, 
            n_mels=n_mels,
            power=2.0
        )


    def construct_model(self):
        from pretrained_models.pitch_estimator.fcpe.models import F0ExtractorWrapper
        f0_extractor = F0ExtractorWrapper(self.cfg, device='cpu')

        f0cfg = self.cfg.model.f0_codec
        use_unnormf0_mse_loss = f0cfg.get('use_unnormf0_mse_loss', False)
        if use_unnormf0_mse_loss:
            assert f0cfg.use_normalized_f0, "use_unnormf0_mse_loss requires use_normalized_f0=True"
            assert not f0cfg.get('use_fcpe_loss', False), "use_unnormf0_mse_loss is incompatible with use_fcpe_loss"
            print(colored("F0 codec: enabling unnormalized F0 MSE loss (targets use log1p raw F0)", "yellow"))
        if not f0cfg.zero_out_all_unvoiced:
            print(colored(f"F0 codec: using 1-dim input (raw f0 with -3 for unvoiced frames)", "yellow"))
        else: print(colored(f"F0 codec: using 2-dim input (raw f0 + vuv); zeroing out all unvoiced frames", "yellow"))
        if not f0cfg.use_normalized_f0:
            assert not f0cfg.zero_out_all_unvoiced, "zero_out_all_unvoiced must be False when not using normalized f0"
        
        # F0 extraction mode logging
        upsample_extracted_f0 = getattr(f0cfg, 'upsample_extracted_f0', True)
        if upsample_extracted_f0:
            print(colored(f"F0 extraction: Mode 1 - Interpolate to audio length during extraction (default)", "cyan"))
        else:
            print(colored(f"F0 extraction: Mode 2 - Extract at original frame rate (no interpolation)", "cyan", attrs=['bold']))
            print(colored(f"  → F0 output shape will be [B, T_frame] instead of [B, T_audio]", "yellow"))
            print(colored(f"  → Make sure encoder output length matches F0 frame length", "yellow"))
        
        spkcfg = self.cfg.model.speaker_encoder
        speaker_token_dim = resolve_speaker_encoder_token_dim(spkcfg)

        # Check if using CodecEncoder/Decoder structure for F0
        use_codec_structure = f0cfg.get('use_codec_structure', False)
        if use_unnormf0_mse_loss and not use_codec_structure:
            raise AssertionError("use_unnormf0_mse_loss requires f0_codec.use_codec_structure=True to enable speaker conditioning")
        
        if use_codec_structure:
            print(colored(f"F0 codec: using CodecEncoder/CodecDecoder structure", "cyan", attrs=['bold']))
            from vq.f0.codec_structure import F0CodecEncoder, F0CodecDecoder, build_channel_schedule
            f0_encoder_cfg = F0CodecEncoderConfig.from_f0_cfg(f0cfg)
            f0_decoder_cfg = F0CodecDecoderConfig.from_f0_cfg(f0cfg)
            
            input_channels = 2 if f0cfg.zero_out_all_unvoiced else 1
            output_channels = 2 if f0cfg.zero_out_all_unvoiced else 1
            
            encoder_stage_channels = build_channel_schedule(
                f0_encoder_cfg.ngf,
                len(f0_encoder_cfg.up_ratios),
                max_channels=f0cfg.get('max_channels', None),
            )
            decoder_stage_channels = build_channel_schedule(
                f0_decoder_cfg.ngf,
                len(f0_decoder_cfg.up_ratios),
                max_channels=f0cfg.get('max_channels', None),
            )

            # Auto-calculate encoder_out_channels if not specified.
            # Default: hidden top channel after all EncoderBlocks with max_channels cap applied.
            encoder_out_channels = getattr(f0cfg, 'encoder_out_channels', None)
            if encoder_out_channels is None:
                encoder_out_channels = encoder_stage_channels[-1]
            
            # Auto-calculate decoder_in_channels if not specified
            decoder_in_channels = getattr(f0cfg, 'decoder_in_channels', None)
            if decoder_in_channels is None: decoder_in_channels = encoder_out_channels
            
            # decoder_upsample_initial_channel: must match the decoder's top hidden width.
            decoder_upsample_initial_channel = getattr(f0cfg, 'decoder_upsample_initial_channel', None)
            if decoder_upsample_initial_channel is None:
                decoder_upsample_initial_channel = decoder_stage_channels[-1]
            
            f0_encoder = F0CodecEncoder(
                input_channels=input_channels,
                ngf=f0_encoder_cfg.ngf,
                max_channels=f0cfg.get('max_channels', None),
                use_rnn=f0_encoder_cfg.use,
                rnn_bidirectional=f0_encoder_cfg.bidirectional,
                rnn_num_layers=f0_encoder_cfg.num_layers,
                rnn_type=f0_encoder_cfg.backbone,
                rnn_mamba=f0_encoder_cfg.mamba,
                up_ratios=f0_encoder_cfg.up_ratios,
                dilations=f0_encoder_cfg.dilations,
                out_channels=encoder_out_channels,
                activation_type=f0_encoder_cfg.activation_type,
                leaky_relu_params=f0_encoder_cfg.leaky_relu_params,
            )
            
            # F0 decoder speaker conditioning: enable when using normalized f0 + fcpe loss
            # This helps the model learn speaker-specific pitch ranges
            use_normalized_f0 = f0cfg.get('use_normalized_f0', False)
            use_fcpe_loss = f0cfg.get('use_fcpe_loss', False)
            legacy_f0_spk_cond = use_normalized_f0 and (use_fcpe_loss or use_unnormf0_mse_loss)
            f0_spk_cond_cfg = F0CodecSpeakerConditionConfig.from_f0_cfg(f0cfg)
            f0_decoder_use_concat = f0_spk_cond_cfg.resolve_concat(legacy_f0_spk_cond)
            f0_decoder_use_film = f0_spk_cond_cfg.resolve_film(legacy_f0_spk_cond)
            f0_decoder_speaker_condition = f0_decoder_use_concat or f0_decoder_use_film
            decoder_use_mhca = f0_spk_cond_cfg.resolve_mhca(legacy_f0_spk_cond)
            if (
                f0cfg.get('spk_cond', None) is None
                and f0cfg.get('decoder_use_mhca', f0cfg.get('use_mhca', False))
                and not decoder_use_mhca
            ):
                print(colored(f"Warning: decoder_use_mhca=True but speaker_condition=False. Disabling MHCA for F0 decoder.", "yellow"))
            configured_f0_mhca_key_dim = f0_spk_cond_cfg.key_dim
            f0_mhca_key_dim = speaker_token_dim
            f0_spk_cond_node = f0cfg.get('spk_cond', None)
            f0_mhca_use_sdpa = (
                f0_spk_cond_node.get('use_sdpa', f0cfg.get('decoder_mhca_use_sdpa', None))
                if f0_spk_cond_node is not None else f0cfg.get('decoder_mhca_use_sdpa', None)
            )
            if configured_f0_mhca_key_dim is not None and int(configured_f0_mhca_key_dim) != int(speaker_token_dim):
                print(colored(
                    f"Overriding f0 decoder MHCA key dim {configured_f0_mhca_key_dim} -> {speaker_token_dim} "
                    f"to match final speaker token dim",
                    "yellow",
                ))

            f0_decoder = F0CodecDecoder(
                in_channels=decoder_in_channels,
                upsample_initial_channel=decoder_upsample_initial_channel,
                output_channels=output_channels,
                ngf=f0_decoder_cfg.ngf,
                max_channels=f0cfg.get('max_channels', None),
                use_rnn=f0_decoder_cfg.use,
                rnn_bidirectional=f0_decoder_cfg.bidirectional,
                rnn_num_layers=f0_decoder_cfg.num_layers,
                rnn_type=f0_decoder_cfg.backbone,
                rnn_mamba=f0_decoder_cfg.mamba,
                up_ratios=f0_decoder_cfg.up_ratios,
                dilations=f0_decoder_cfg.dilations,
                activation_type=f0_decoder_cfg.activation_type,
                leaky_relu_params=f0_decoder_cfg.leaky_relu_params,
                use_fcpe_loss=use_fcpe_loss,
                fcpe_out_dims=f0cfg.get('fcpe_out_dims', 360),
                speaker_condition=f0_decoder_speaker_condition,
                condition_dim=spkcfg.out_dim if f0_decoder_speaker_condition else 0,
                use_spk_concat=f0_decoder_use_concat,
                use_spk_film=f0_decoder_use_film,
                use_mhca=decoder_use_mhca,
                mhca_num_heads=f0_spk_cond_cfg.num_heads,
                mhca_dropout=f0_spk_cond_cfg.dropout,
                mhca_key_dim=f0_mhca_key_dim,
                mhca_use_sdpa=f0_mhca_use_sdpa,
                use_split_condition_optimization=True,
            )
            self.f0_decoder_use_mhca = decoder_use_mhca
        else:
            print(colored(f"F0 codec: using original F0Encoder/F0Decoder structure", "yellow"))
            f0_encoder = F0Encoder(
                # input_emb_width=f0cfg.input_emb_width,
                input_emb_width= 2 if f0cfg.zero_out_all_unvoiced else 1,
                output_emb_width=f0cfg.output_emb_width,
                levels=f0cfg.levels,
                downs_t=f0cfg.downs_t,
                strides_t=f0cfg.enc_strides_t,
                width=f0cfg.width,
                depth=f0cfg.depth,
                m_conv=f0cfg.m_conv,
                channel_growth_rate=f0cfg.channel_growth_rate,
                dilation_growth_rate=f0cfg.dilation_growth_rate,
            )

            f0_decoder = F0Decoder(
                # input_emb_width=f0cfg.input_emb_width,
                input_emb_width= 2 if f0cfg.zero_out_all_unvoiced else 1,
                output_emb_width=f0cfg.output_emb_width,
                levels=f0cfg.levels,
                downs_t=f0cfg.downs_t,
                strides_t=f0cfg.dec_strides_t,
                width=f0cfg.width,
                depth=f0cfg.depth,
                m_conv=f0cfg.m_conv,
                channel_growth_rate=f0cfg.channel_growth_rate,
                dilation_growth_rate=f0cfg.dilation_growth_rate,
            )
            self.f0_decoder_use_mhca = False

        f0_quantizer = FactorizedVectorQuantize(
            dim=f0cfg.output_emb_width if not use_codec_structure else encoder_out_channels,
            codebook_size=f0cfg.codebook_size,
            codebook_dim=f0cfg.codebook_dim,
            commitment=f0cfg.commitment,
            codebook_loss_weight=f0cfg.codebook_loss_weight,
            threshold_ema_dead_code=f0cfg.threshold_ema_dead_code,
        )

        # Resolve WavLM checkpoint path (convert relative to absolute using original cwd)
        wavlm_ckpt = spkcfg.get('wavlm_checkpoint', None)
        if wavlm_ckpt is not None and not os.path.isabs(wavlm_ckpt):
            wavlm_ckpt = os.path.join(self.ocwd, wavlm_ckpt)
        
        speaker_encoder = SpeakerEncoder(
            mel_params=getattr(self.cfg.model, 'mel_params', None),
            speaker_encoder_type=spkcfg.speaker_encoder_type,
            use_perceiver_encoder=spkcfg.use_perceiver_encoder,
            use_memory_cattn=spkcfg.get('use_memory_cattn', False),
            input_dim=spkcfg.get('input_dim', 100),
            out_dim=spkcfg.out_dim,
            latent_dim=spkcfg.latent_dim,
            token_num=spkcfg.token_num,
            discretize_memory_attn=spkcfg.get('discretize_memory_attn', False),
            memory_attn_codebook_size=spkcfg.get('memory_attn_codebook_size', 128),
            memory_attn_share_across_heads=spkcfg.get('memory_attn_share_across_heads', True),
            norm_layer=spkcfg.get('norm_layer', 'bn'),
            use_normalized_f0=f0cfg.get('use_normalized_f0', False),
            use_quantizer=spkcfg.get('use_quantizer', True),
            stack=spkcfg.get('stack', None),
            perceiver_use_flash_attn=spkcfg.get('perceiver_use_flash_attn', False),
            wavlm_checkpoint=wavlm_ckpt,
            wavlm_output_layer=spkcfg.get('wavlm_output_layer', 6),
            freeze_wavlm=spkcfg.get('freeze_wavlm', True),
            frozen_wavlm_inference_mode=spkcfg.get('frozen_wavlm_inference_mode', False),
            frozen_wavlm_force_fp32=spkcfg.get('frozen_wavlm_force_fp32', False),
            **build_speaker_quantizer_kwargs(spkcfg),
        )
        enccfg = self.cfg.model.codec_encoder
        print(colored(f"codec_encoder.activation_type: {enccfg.get('activation_type', 'SnakeBeta')}", "cyan"))
        if enccfg.use_vqw2v:
            encoder = S3PRLUpstream("vq_wav2vec_kmeans")
            encoder = encoder.upstream.model.feature_extractor
            # 훅 제거: 수치 영향 없음, 메모리 누수 방지
            for m in encoder.modules():
                if hasattr(m, "_forward_hooks"): m._forward_hooks.clear()
                if hasattr(m, "_forward_pre_hooks"): m._forward_pre_hooks.clear()
                if hasattr(m, "_backward_hooks"): m._backward_hooks.clear()

            if enccfg.freeze_vqw2v_encoder:
                print(colored("Freezing VQ-Wav2Vec encoder", "yellow"))
                for p in encoder.parameters():
                    p.requires_grad = False
                encoder.eval()
            else:
                for p in encoder.parameters():
                    p.requires_grad = True 

        elif not enccfg.use_vqw2v:
            encoder = CodecEncoder(
                        ngf=enccfg.ngf,
                        temporal=CodecTemporalConfig.from_encoder_cfg(enccfg),
                        up_ratios=enccfg.up_ratios,
                        dilations=enccfg.dilations,
                        out_channels=enccfg.out_channels,

                        speaker_condition=enccfg.encoder_speaker_condition,
                        condition_dim=enccfg.condition_dim,
                        snake_logscale=enccfg.snake_logscale,
                        activation_type=enccfg.get('activation_type', 'SnakeBeta'),
                        leaky_relu_params=enccfg.get('leaky_relu_params', None),
                        snake_lite_taylor_degree=enccfg.get('snake_lite_taylor_degree', 8),
                    )
        if enccfg.get('up_ratios', None) is not None:
            print(colored(f"Using VQ-Wav2Vec Codec Encoder Wrapper", "red"))
            encoder = VQW2VCodecEncoderWrapper(
                temporal=CodecTemporalConfig.from_encoder_cfg(enccfg),
                encoder=encoder,
                up_ratios=enccfg.up_ratios,
                dilations=enccfg.dilations,
                ngf=512,
                out_channels=enccfg.out_channels,
                activation_type=enccfg.get('activation_type', 'SnakeBeta'),
                leaky_relu_params=enccfg.get('leaky_relu_params', None),
                snake_lite_taylor_degree=enccfg.get('snake_lite_taylor_degree', 8),
                encoder_force_fp32=enccfg.get('frozen_upstream_force_fp32', False),)

        
        deccfg = self.cfg.model.codec_decoder
        decoder_f0_cond_cfg = CodecDecoderF0ConditionConfig.from_decoder_cfg(deccfg)
        decoder_spk_cond_cfg = CodecDecoderSpeakerConditionConfig.from_decoder_cfg(deccfg)
        decoder_vocos_cfg = CodecDecoderVocosConfig.from_decoder_cfg(deccfg)
        decoder_spk_cond_node = deccfg.get('spk_cond', None)
        decoder_mhca_use_sdpa = (
            decoder_spk_cond_node.get('use_sdpa', deccfg.get('mhca_use_sdpa', None))
            if decoder_spk_cond_node is not None else deccfg.get('mhca_use_sdpa', None)
        )
        decoder_f0_stage_dims = None
        if deccfg.get('decoder_type', 'default').lower() in ['rndvoc', 'vocos', 'vocosformer'] and decoder_f0_cond_cfg.use_concat:
            stage_count = int(
                deccfg.get(
                    'rndvoc_null_nstage',
                    decoder_vocos_cfg.num_layers,
                )
            ) if deccfg.get('decoder_type', 'default').lower() == 'rndvoc' else int(decoder_vocos_cfg.num_layers)
            decoder_f0_stage_dims = self._build_rndvoc_f0_stage_dims(
                f0_decoder=f0_decoder,
                f0cfg=f0cfg,
                null_nstage=stage_count,
            )
        decoder = CodecDecoder(
                    in_channels=deccfg.in_channels,
                    upsample_initial_channel=deccfg.get('upsample_initial_channel', 1536),

                    ngf=deccfg.get('ngf', 48),
                    temporal=CodecDecoderTemporalConfig.from_decoder_cfg(deccfg),
                    up_ratios=deccfg.get('up_ratios', (5, 5, 2, 2, 2)),
                    dilations=deccfg.get('dilations', (1, 3, 9)),

                    quantizer_type=deccfg.quantizer_type,
                    freeze_kvq_emb=deccfg.get('freeze_kvq_emb', True),

                    vq_num_quantizers=deccfg.get('vq_num_quantizers', 1),
                    vq_dim=deccfg.vq_dim,
                    vq_commit_weight=deccfg.vq_commit_weight,
                    vq_full_commit_loss=deccfg.get('vq_full_commit_loss', False),
                    quantizer_force_fp32=deccfg.get('quantizer_force_fp32', False),
                    codebook_size=deccfg.codebook_size,
                    codebook_dim=deccfg.get('codebook_dim', 8),
                    use_vqw2v_encoder=enccfg.use_vqw2v,
                    freeze_vqw2v_encoder=enccfg.freeze_vqw2v_encoder,
                    use_vqw2v_embed=deccfg.get('use_vqw2v_embed', True),

                    speaker_condition=deccfg.speaker_condition,
                    condition_dim=deccfg.condition_dim,
                    activation_type=deccfg.get('activation_type', 'SnakeBeta'),
                    leaky_relu_params=deccfg.get('leaky_relu_params', None),
                    snake_logscale=deccfg.get('snake_logscale', True),
                    snake_lite_taylor_degree=deccfg.get('snake_lite_taylor_degree', 8),
                    simvq_linear_layer_type=deccfg.get('simvq_linear_layer_type', 'linear'),
                    ema_decay=deccfg.get('ema_decay', 0.0),

                    f0_condition=decoder_f0_cond_cfg.use_concat,
                    f0_start_layer=decoder_f0_cond_cfg.start_layer,
                    f0_end_layer=decoder_f0_cond_cfg.end_layer,
                    f0_every=decoder_f0_cond_cfg.every,
                    f0_width_list=f0_encoder.width_list,
                    f0_speaker_condition=deccfg.get('f0_speaker_condition', False),
                    use_stage_speaker_film=(decoder_spk_cond_cfg.use_film or deccfg.get('use_stage_speaker_film', False)),
                    
                    use_mhca=decoder_spk_cond_cfg.use_mhca,
                    spk_cond_use_concat=decoder_spk_cond_cfg.use_concat,
                    mhca_num_heads=decoder_spk_cond_cfg.num_heads,
                    mhca_dropout=decoder_spk_cond_cfg.dropout,
                    mhca_key_dim=speaker_token_dim,
                    mhca_use_sdpa=decoder_mhca_use_sdpa,
                    mhca_start_layer=decoder_spk_cond_cfg.start_layer,
                    mhca_end_layer=decoder_spk_cond_cfg.end_layer,
                    mhca_every=decoder_spk_cond_cfg.every,
                    decoder_type=deccfg.get('decoder_type', 'default'),
                    rndvoc_sampling_rate=self.cfg.preprocess.audio.sr,
                    rndvoc_n_fft=deccfg.get('rndvoc_n_fft', None),
                    rndvoc_hop_size=deccfg.get('rndvoc_hop_size', None),
                    rndvoc_win_size=deccfg.get('rndvoc_win_size', None),
                    rndvoc_input_channel=deccfg.get('rndvoc_input_channel', 256),
                    rndvoc_hidden_channel=deccfg.get('rndvoc_hidden_channel', 256),
                    rndvoc_squeeze_size=deccfg.get('rndvoc_squeeze_size', 64),
                    rndvoc_null_nstage=deccfg.get('rndvoc_null_nstage', 6),
                    rndvoc_nrep=deccfg.get('rndvoc_nrep', 2),
                    rndvoc_kernel_size=deccfg.get('rndvoc_kernel_size', 7),
                    rndvoc_causal=deccfg.get('rndvoc_causal', False),
                    rndvoc_use_rnd=deccfg.get('rndvoc_use_rnd', True),
                    rndvoc_time_type=deccfg.get('rndvoc_time_type', 'convnext_v2'),
                    rndvoc_freq_type=deccfg.get('rndvoc_freq_type', 'shuffler'),
                    rndvoc_f0_stage_dims=decoder_f0_stage_dims,
                    vocos_sampling_rate=self.cfg.preprocess.audio.sr,
                    vocos_n_fft=decoder_vocos_cfg.n_fft,
                    vocos_hop_size=decoder_vocos_cfg.hop_size,
                    vocos_win_size=decoder_vocos_cfg.win_size,
                    vocos_dim=decoder_vocos_cfg.dim,
                    vocos_intermediate_dim=decoder_vocos_cfg.intermediate_dim,
                    vocos_num_layers=decoder_vocos_cfg.num_layers,
                    vocos_kernel_size=decoder_vocos_cfg.kernel_size,
                    vocos_padding=decoder_vocos_cfg.padding,
                    vocos_f0_stage_dims=decoder_f0_stage_dims,
                    vocos_backbone=decoder_vocos_cfg.backbone,
                    vocos_num_heads=decoder_vocos_cfg.num_heads,
                    vocos_window_size=decoder_vocos_cfg.window_size,
                    vocos_attn_window_size=decoder_vocos_cfg.attn_window_size,
                    vocos_attention_impl=decoder_vocos_cfg.attention_impl,
                    vocos_use_rope=decoder_vocos_cfg.use_rope,
                    vocos_rope_base=decoder_vocos_cfg.rope_base,
                    vocos_max_position_embeddings=decoder_vocos_cfg.max_position_embeddings,
                    vocos_qk_norm=decoder_vocos_cfg.qk_norm,
                    vocos_norm_type=decoder_vocos_cfg.norm_type,
                    vocos_norm_eps=decoder_vocos_cfg.norm_eps,
                    vocos_dropout=decoder_vocos_cfg.dropout,
                    vocos_ffn_type=decoder_vocos_cfg.ffn_type,
                    vocos_ffn_mult=decoder_vocos_cfg.ffn_mult,
                    vocos_layerscale_gamma_init=decoder_vocos_cfg.layerscale_gamma_init,
                    # zero_out_all_unvoiced=f0cfg.zero_out_all_unvoiced,
                )


        mpdcfg = self.cfg.model.mpd
        mpd = HiFiGANMultiPeriodDiscriminator(
                    periods=mpdcfg.periods,
                    max_downsample_channels=mpdcfg.max_downsample_channels,
                    channels=mpdcfg.channels,
                    channel_increasing_factor=mpdcfg.channel_increasing_factor,
                    use_weight_norm=mpdcfg.get('use_weight_norm', True),
                )
        mstftcfg = self.cfg.model.mstft
        mstft = SpecDiscriminator(
                    stft_params=mstftcfg.stft_params,
                    in_channels=mstftcfg.in_channels,
                    out_channels=mstftcfg.out_channels,
                    kernel_sizes=mstftcfg.kernel_sizes,
                    channels=mstftcfg.channels,
                    max_downsample_channels=mstftcfg.max_downsample_channels,
                    downsample_scales=mstftcfg.downsample_scales,
                    use_weight_norm=mstftcfg.use_weight_norm,
                )
        model = nn.ModuleDict({
                    'speaker_encoder': speaker_encoder,
                    'CodecEnc': encoder,
                    'generator': decoder,
                    'discriminator': mpd,
                    'spec_discriminator': mstft,
                    'f0_encoder': f0_encoder,
                    'f0_decoder': f0_decoder,
                    'f0_quantizer': f0_quantizer,
                    'f0_extractor': f0_extractor,
                })
        
        # add mi if use_mi
        if self.cfg.model.get('use_mi', False):
            print(colored(f"Using MI network: CLUBSample_group and MINE", "yellow"))
            print(colored(f"Lambda_mi_loss: {self.cfg.train.lambdas.get('lambda_mi_loss', 0.0)}", "yellow"))
            print(colored(f"CMI steps: {self.cfg.train.cmi_steps}", "yellow"))
            from vq.mi import CLUBSample_group, MINE
            club_dim, mine_dim = self.cfg.model.mi.club_dim, self.cfg.model.mi.mine_dim
            x_dim, y_dim = self.cfg.model.speaker_encoder.out_dim, self.cfg.model.codec_decoder.vq_dim
            mi_club = CLUBSample_group(
                x_dim=x_dim,
                y_dim=y_dim,
                hidden_size=club_dim)
            mi_mine = MINE(
                x_dim=x_dim,
                y_dim=y_dim,
                hidden_size=mine_dim)

            model['mi_club'] = mi_club
            model['mi_mine'] = mi_mine

        for k, v in model.named_children():
            # print number of parameters
            num_params = self._count_initialized_params(v)
            print(f"{k}: {num_params}")
            # print number of trainable parameters
            num_trainable_params = self._count_initialized_params(v, trainable_only=True)
            print(f"{k} (trainable): {num_trainable_params}")
        self.model = model
        self._print_model_summary()

    @staticmethod
    def _build_rndvoc_f0_stage_dims(f0_decoder, f0cfg, null_nstage):
        if hasattr(f0_decoder, 'width_list'):
            stage_dims = list(reversed(f0_decoder.width_list[0]))
        else:
            stage_dims = [int(f0cfg.output_emb_width)]

        if not getattr(f0_decoder, 'use_fcpe_loss', False):
            final_output_dim = int(getattr(f0_decoder, 'output_channels', 2 if bool(f0cfg.zero_out_all_unvoiced) else 1))
            stage_dims.append(final_output_dim)

        if len(stage_dims) < null_nstage:
            stage_dims.extend([stage_dims[-1]] * (null_nstage - len(stage_dims)))
        else:
            stage_dims = stage_dims[:null_nstage]
        return [int(dim) for dim in stage_dims]

    @staticmethod
    def _count_initialized_params(module, trainable_only=False):
        total = 0
        for param in module.parameters():
            if isinstance(param, UninitializedParameter):
                continue
            if trainable_only and not param.requires_grad:
                continue
            total += param.numel()
        return total

    def _print_model_summary(self):
        try:
            print(ModelSummary(self, max_depth=2))
        except ValueError as exc:
            print(colored(f"Skipping ModelSummary until lazy parameters are initialized: {exc}", "yellow"))

    def configure_model(self):
        """Called after model is moved to device, before training starts.
        This is the right place to apply torch.compile."""
        if self.cfg.train.get('use_torch_compile', False):
            # Disable all disk caching to prevent I/O bloat during long training
            # Compilation will happen once per run (in memory), cache cleared on restart
            import torch._inductor.config as inductor_config
            inductor_config.fx_graph_cache = False  # Disable FX graph cache DB
            # Set cache dir to /tmp (auto-cleaned on reboot)
            os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torch_compile_cache_temp'
            # Disable Triton persistent cache
            os.environ['TRITON_CACHE_DIR'] = '/tmp/triton_cache_temp'
            print(colored("⚠ Inductor disk cache DISABLED to prevent I/O bloat", "yellow"))
            print(colored("  → Compilation happens once per run (stored in memory)", "cyan"))
            print(colored("  → Caches redirected to /tmp (auto-cleaned on reboot)", "cyan"))
            
            compile_mode = self.cfg.train.get('compile_mode', 'default')
            print(colored(f"Applying torch.compile (mode={compile_mode}) to model components...", "yellow"))
            
            # Compile each model component separately for better control
            # Note: WavLM inside speaker_encoder is already compiled separately in __init__
            # Only use dynamic=True for modules that actually need it
            try:
                if hasattr(self, 'model'):
                    # Speaker Encoder - fixed length after Perceiver resampling
                    if 'speaker_encoder' in self.model:
                        self.model['speaker_encoder'] = torch.compile(
                            self.model['speaker_encoder'], mode=compile_mode
                        )
                        print(colored("  ✓ Speaker encoder compiled", "green"))
                    
                    # CodecEnc - fixed downsampling ratio
                    # if 'CodecEnc' in self.model:
                    #     self.model['CodecEnc'] = torch.compile(
                    #         self.model['CodecEnc'], mode=compile_mode
                    #     )
                    #     print(colored("  ✓ CodecEnc compiled", "green"))
                    
                    # Generator (decoder) - SKIP: SnakeBeta + dynamic upsampling causes symbolic shape issues
                    # The error is in the decoder's upsampling blocks with conditional SnakeBeta
                    
                    # Discriminators - fixed architecture
                    if 'discriminator' in self.model:
                        self.model['discriminator'] = torch.compile(
                            self.model['discriminator'], mode=compile_mode
                        )
                        print(colored("  ✓ MPD compiled", "green"))
                    
                    if 'spec_discriminator' in self.model:
                        self.model['spec_discriminator'] = torch.compile(
                            self.model['spec_discriminator'], mode=compile_mode
                        )
                        print(colored("  ✓ MSTFT discriminator compiled", "green"))
                    
                    # F0 codec components - fixed architecture
                    if 'f0_encoder' in self.model:
                        self.model['f0_encoder'] = torch.compile(
                            self.model['f0_encoder'], mode=compile_mode
                        )
                        print(colored("  ✓ F0 encoder compiled", "green"))
                    
                    if 'f0_decoder' in self.model:
                        self.model['f0_decoder'] = torch.compile(
                            self.model['f0_decoder'], mode=compile_mode
                        )
                        print(colored("  ✓ F0 decoder compiled", "green"))
                    
                    if 'f0_quantizer' in self.model:
                        self.model['f0_quantizer'] = torch.compile(
                            self.model['f0_quantizer'], mode=compile_mode
                        )
                        print(colored("  ✓ F0 quantizer compiled", "green"))
                    
                    print(colored("✓ torch.compile applied successfully", "green"))
                    print(colored("⚠ Generator skipped (SnakeBeta + dynamic shapes not compatible)", "yellow"))
                    print(colored("⚠ First epoch will be slower due to compilation (JIT tracing)", "yellow"))
            except Exception as e:
                print(colored(f"⚠ torch.compile failed: {e}", "red"))
                print(colored("  Continuing without compilation...", "yellow"))

    def load_state_dict(self, state_dict, strict=True):
        """
        Checkpoint -> Runtime state_dict reconciliation.
        - Drop keys for modules that are not part of the runtime model
        - Normalize torch.compile `_orig_mod` wrapper prefixes under `model.*`
          so that compiled / non-compiled checkpoints can be loaded either way.
        """
        filtered = {}
        skipped_keys = []

        def _has_submodule(name: str):
            if isinstance(self.model, nn.ModuleDict):
                return name in self.model
            return hasattr(self.model, name)

        def _get_submodule(name: str):
            if isinstance(self.model, nn.ModuleDict):
                return self.model[name] if name in self.model else None
            return getattr(self.model, name, None)

        for k, v in state_dict.items():
            # 1) Drop keys we never load in the runtime model
            skip_reason = None
            if k.startswith('mel_spectrogram_transform.'):
                skip_reason = 'runtime-only module'
            elif k.startswith('utmos_predictor.'):
                skip_reason = 'runtime-only module'
            elif k.startswith('wavlm_model.'):
                skip_reason = 'runtime-only module'
            elif k.startswith('hubert_model.'):
                skip_reason = 'runtime-only module'
            elif '.f0_mu_predictor.' in k:
                skip_reason = 'removed legacy f0_mu_predictor'
            if skip_reason is not None:
                skipped_keys.append((k, skip_reason))
                continue

            # 2) Normalize torch.compile wrapper prefixes for any submodule under model.*
            new_key = k
            if k.startswith('model.'):
                parts = k.split('.')
                if len(parts) >= 3:
                    sub_name = parts[1]  # model.<sub>...
                    if _has_submodule(sub_name):
                        submodule = _get_submodule(sub_name)
                        compiled_now = hasattr(submodule, '_orig_mod')
                        plain_prefix = f'model.{sub_name}.'
                        compiled_prefix = f'model.{sub_name}._orig_mod.'
                        # ckpt compiled, runtime plain -> strip
                        if k.startswith(compiled_prefix) and not compiled_now:
                            new_key = plain_prefix + k[len(compiled_prefix):]
                        # ckpt plain, runtime compiled -> insert
                        elif k.startswith(plain_prefix) and compiled_now and not k.startswith(compiled_prefix):
                            new_key = compiled_prefix + k[len(plain_prefix):]

            filtered[new_key] = v

        # Always load with strict=False, then emulate strict behavior manually
        missing, unexpected = super().load_state_dict(filtered, strict=False)

        # Allow missing keys for non-trainable / runtime-only modules
        allowed_missing_prefixes = (
            'mel_spectrogram_transform.',
            'utmos_predictor.',
            'wavlm_model.',
            'hubert_model.',
        )
        allowed_missing = [k for k in missing if k.startswith(allowed_missing_prefixes)]
        real_missing = [k for k in missing if k not in allowed_missing]

        if skipped_keys:
            print("[load_state_dict] skipped checkpoint keys:")
            for key, reason in skipped_keys:
                print(f"  - {key} ({reason})")
        if allowed_missing:
            print("[load_state_dict] (ignored missing non-trainable):",
                  allowed_missing[:8], "..." if len(allowed_missing) > 8 else "")
        if unexpected:
            print("[load_state_dict] (unexpected):", unexpected[:8], "..." if len(unexpected) > 8 else "")

        # If strict=True, propagate real errors (excluding allowed missing keys)
        if strict and (real_missing or unexpected):
            error_msgs = []
            if real_missing:
                error_msgs.append(
                    "Missing key(s) in state_dict: " +
                    ", ".join(f'"{k}"' for k in real_missing) + "."
                )
            if unexpected:
                error_msgs.append(
                    "Unexpected key(s) in state_dict: " +
                    ", ".join(f'"{k}"' for k in unexpected) + "."
                )
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t" +
                "\n\t".join(error_msgs)
            )

        return real_missing, unexpected

    def construct_criteria(self):
        cfg = self.cfg.train
        criteria = nn.ModuleDict()
        if cfg.use_mel_loss:
            criteria['mel_loss'] = MultiResolutionMelSpectrogramLoss(sample_rate=self.cfg.preprocess.audio.sr)
        if cfg.use_feat_match_loss:
            criteria['fm_loss'] = nn.L1Loss()
        criteria['gan_loss'] = GANLoss()
        criteria['l1_loss'] = torch.nn.L1Loss()
        criteria['l2_loss'] = torch.nn.MSELoss()
        criteria['bcewlogits_loss'] = torch.nn.BCEWithLogitsLoss()
        criteria['bce_loss'] = torch.nn.BCELoss()
        self.criteria = criteria
        # print(criteria)

    def _no_sync_if_needed(self, enable: bool):
        # enable=True면 통신을 끔(no_sync 사용)
        if enable and hasattr(self.trainer, "strategy"):
            strat = self.trainer.strategy
            model = getattr(strat, "model", None)
            if model is not None and hasattr(model, "no_sync"):
                return model.no_sync()
        return nullcontext()

    def _sync_needed(self):
        try:
            return getattr(self.trainer.strategy, "world_size", 1) > 1
        except Exception:
            return False

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        def move(x):
            return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x
        if isinstance(batch, dict):
            return {k: move(v) for k, v in batch.items()}
        return batch

    def pad_for_wav2vec(self, source, J=160, RF=465):
        # source: (B, T)
        B, L = source.shape
        pad_left = RF - J  # 305
        # ceil(L/J) 프레임 확보용 오른쪽 패딩
        rem = L % J
        pad_right = (J - rem) % J
        return F.pad(source, (pad_left, pad_right)), pad_left, pad_right


    def extract_f0(self, batch, ref=False):
        if not ref:
            # if self.cfg.preprocess.audio.sr == 16000:
            audio = batch['wav'].unsqueeze(-1)
            # else:
            #     print(colored(f"Using 24kHz audio for F0 extraction", "red"))
            #     audio = batch['wav_24k'].unsqueeze(-1)
        else:
            audio = batch['ref_wav'].unsqueeze(-1)
        
        # Two modes for F0 extraction:
        # Mode 1 (default): Interpolate to audio length (directly upsample during extraction)
        # Mode 2 (optional): Extract at original frame rate (no interpolation in extractor)
        if self.cfg.model.f0_codec.get('upsample_extracted_f0', True):
            # Mode 1: Interpolate to audio sample length
            output_interp_target_length = audio.shape[1] if self.cfg.preprocess.audio.sr == 16000 else int(audio.shape[1] * 1.5)
        else:
            # Mode 2: Return original frame length (e.g., T_audio / hop_length)
            target_length = int(audio.shape[1]/self.cfg.model.f0_codec.get('hop_length', 160))
            output_interp_target_length = target_length if self.cfg.preprocess.audio.sr == 16000 else int(target_length * 1.5)
        
        f0 = self.model['f0_extractor'](audio,
                                        sr=16000,
                                        output_interp_target_length=output_interp_target_length)  # (B, T)

        # print("f0>0 min max mean var :", (f0[f0>0].min().item(), f0[f0>0].max().item(), f0[f0>0].mean().item(), f0[f0>0].var().item()))

        # 1) NaN/Inf 제거
        f0 = torch.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)


        # 2) V/UV 마스크
        vuv = (f0 > 0).float()

        # 3) 전부 무성인 샘플 처리 (count=0 → 바로 zero feature 반환)
        count = vuv.sum(dim=1, keepdim=True)  # (B,1)
        all_unvoiced = (count == 0)
        # log-f0 (무성은 임시 0; 통계엔 vuv로 걸러짐)
        f0_log = torch.log(torch.clamp(f0, min=1e-5))

        # 4) voiced 값만으로 통계
        voiced_log = f0_log * vuv
        count_safe = count.clamp(min=1)
        mu = voiced_log.sum(dim=1, keepdim=True) / count_safe
        # CRITICAL FIX: unvoiced 프레임을 variance 계산에서 제외
        # 이전: diff = (voiced_log - mu)  → unvoiced가 -mu가 되어 variance 폭발!
        # 수정: diff = (f0_log - mu) * vuv → unvoiced는 0으로 유지
        diff = (f0_log - mu) * vuv  # voiced 프레임에서만 편차 계산
        var = (diff.pow(2).sum(dim=1, keepdim=True) / count_safe)
        sig = var.sqrt().clamp(min=1e-4)
        # print("f0 voiced log mu :", (mu.min().item(), mu.max().item(), mu.mean().item(), mu.var().item()))
        # print("f0 voiced log sig:", (sig.min().item(), sig.max().item(), sig.mean().item(), sig.var().item()))

        # 5) 정규화: unvoiced 프레임은 연산 전에 제거 → 큰 음수 생성 안 됨 (다시 넣자, 이거 std 도 speaker정보가 채워줘야하는 global info라 없애는 게 맞는 듯 251028)
        z = ((f0_log - mu) * vuv) / sig  # (B,T)
        # variance 정규화 제거 버전
        # z = ((f0_log - mu) * vuv) # (B,T)

        # unvoiced 부분을 0 이 아닌 -3.0 으로 채우기
        if not self.cfg.model.f0_codec.zero_out_all_unvoiced:
            uv_mask = vuv < 0.5       # 0/1 또는 확률인 경우
            # 채널 차원이 있으면 확장
            if z.dim() == 3:  # (B, C, T)
                uv_mask = uv_mask.unsqueeze(1)  # (B, 1, T)

            # -3으로 채우기 (벡터화, 병렬)
            z = z.masked_fill(uv_mask, -3.0)

        # print("f0 normalized min max mean var :", (z[vuv>0].min().item(), z[vuv>0].max().item(), z[vuv>0].mean().item(), z[vuv>0].var().item()))
        # 6) 클램프 (이제 극단값 거의 안 생김; 범위 더 타이트 가능)
        # z = torch.clamp(z, -8.0, 8.0)

        # 7) 전부 무성인 샘플 zero 처리 (혹시 위 계산 중 미세값 남았을 수 있음)
        if all_unvoiced.any():
            z[all_unvoiced.expand_as(z)] = 0.0 if self.cfg.model.f0_codec.zero_out_all_unvoiced else -3.0

        # 8) (선택) 디버그
        # print("f0_log range", f0_log.min().item(), f0_log.max().item(), "z range", z.min().item(), z.max().item())

        return (
            f0.unsqueeze(1),  # (B,1,T) (raw log-f0; 무성 프레임은 log(1e-5))
            z.unsqueeze(1),       # (B,1,T) normalized voiced-only
            vuv.unsqueeze(1),     # (B,1,T)
        )

    
    def extract_loudness(self, batch):
        audio = batch['wav'].unsqueeze(-1)
        return torch.log1p(torch.sqrt(torch.mean(audio**2, dim=-1)))

    
    def forward_f0_codec(self, batch, spk_cond_vec=None, x_quantized=None):
        f0, f0_normalized, vuv = self.extract_f0(batch)  # (B, 1, T)
        if self.vc_f0:
            f0_ref, _, vuv_ref = self.extract_f0(batch, ref=True)  # (B, 1, T)
            
            # 유성음 구간만 추출
            voiced = f0 * vuv  # (B, 1, T)
            voiced_ref = f0_ref * vuv_ref  # (B, 1, T)
            
            # 각 배치의 유성음 프레임 수 계산
            count = vuv.sum(dim=2, keepdim=True)  # (B, 1, 1)
            count_ref = vuv_ref.sum(dim=2, keepdim=True)  # (B, 1, 1)
            
            # 0으로 나누기 방지
            count_safe = count.clamp(min=1)
            count_safe_ref = count_ref.clamp(min=1)
            
            # 평균 계산
            mu = voiced.sum(dim=2, keepdim=True) / count_safe  # (B, 1, 1)
            mu_ref = voiced_ref.sum(dim=2, keepdim=True) / count_safe_ref  # (B, 1, 1)
            
            print("Shift f0 mu:", mu.mean().item(), "->", mu_ref.mean().item())
            print("shapes f0, mu, mu_ref:", f0.shape, mu.shape, mu_ref.shape)
            
            # 전부 무성음인 경우 체크
            all_unvoiced = (count == 0)
            all_unvoiced_ref = (count_ref == 0)
            
            # F0 shift: log 공간에서의 덧셈은 원 공간에서의 곱셈
            # 무성음 구간은 그대로 유지
            if not (all_unvoiced.all() or all_unvoiced_ref.all()):
                # 유성음 구간만 shift 적용
                f0_shifted = f0.clone()
                B = f0.shape[0]
                for b in range(B):
                    shift = mu_ref[b] - mu[b]
                    voiced_mask = vuv[b] > 0.5
                    # 유성음 구간에서 f0 + shift의 최소값 확인
                    min_voiced = (f0[b][voiced_mask] + shift).min().item() if voiced_mask.any() else 0.0
                    # 만약 음수가 있으면 전체적으로 올려줌
                    if min_voiced < 1e-5:
                        shift = shift + (1e-5 - min_voiced)
                    f0_shifted[b] = torch.where(
                        voiced_mask,
                        f0[b] + shift,
                        f0[b]
                    )
                f0 = f0_shifted
            else:
                print("Warning: All frames are unvoiced in source or reference, skipping F0 shift")

        if self.cfg.model.f0_codec.use_normalized_f0: input_f0 = f0_normalized
        else: input_f0 = torch.log1p(f0)
        f0_vuv = torch.cat([input_f0 , vuv], dim=1)  # (B, 2, T)
        z_f0 = self.model['f0_encoder'](f0_vuv)  # (B, C, T)
        f0_output_dict = self.model['f0_quantizer'](z_f0)
        
        # F0 decoder with speaker conditioning and MHCA support
        f0_decoder_module = self.model['f0_decoder']
        f0_spk_cond = None
        if getattr(f0_decoder_module, 'use_mhca', False):
            mhca_global = spk_cond_vec if getattr(f0_decoder_module, 'speaker_condition', False) else None
            f0_spk_cond = (mhca_global, x_quantized) if x_quantized is not None else (mhca_global,)
        elif getattr(f0_decoder_module, 'speaker_condition', False):
            f0_spk_cond = spk_cond_vec
        
        # F0 decoder output: (outs, fcpe_latent) if use_fcpe_loss else outs
        if f0_spk_cond is not None:
            f0_decoder_out = f0_decoder_module(f0_output_dict['z_q'], spk_emb=f0_spk_cond)
        else:
            f0_decoder_out = f0_decoder_module(f0_output_dict['z_q'])
        
        if self.cfg.model.f0_codec.get('use_fcpe_loss', False):
            f0_vuv_recons, f0_fcpe_latent = f0_decoder_out  # unpack tuple
            f0_fcpe_logits = getattr(f0_decoder_module, 'last_fcpe_logits', None)
            # FCPE mode: decode F0 from latent
            # latent (B, 360, T) → cents → raw Hz F0
            # Transpose to (B, T, 360) for FCPE decoder
            fcpe_latent_transposed = f0_fcpe_latent.transpose(1, 2)  # (B, T, 360)
            
            # Access FCPE model directly (F0ExtractorWrapper doesn't wrap latent2cents_local_decoder)
            f0_extractor = self.model['f0_extractor']
            if f0_extractor.is_infer:
                fcpe_model = f0_extractor.extractor.model
            else:
                fcpe_model = f0_extractor.model
            
            # Decode using local_argmax (same as FCPE inference)
            cents_pred = fcpe_model.latent2cents_local_decoder(
                fcpe_latent_transposed, threshold=0.006
            )  # (B, T, 1)
            f0_raw_pred = fcpe_model.cent_to_f0(cents_pred)  # (B, T, 1), raw Hz
            # Transpose back to (B, 1, T)
            f0_raw_pred = f0_raw_pred.transpose(1, 2)  # (B, 1, T)
            
            # Convert to same format as gt_f0_vuv (normalized or log1p)
            if self.cfg.model.f0_codec.use_normalized_f0:
                # raw Hz → log → normalize (simplified: use log1p for now)
                # TODO: proper normalization with speaker mu/sigma
                f0_pred_formatted = torch.log1p(f0_raw_pred)
            else:
                f0_pred_formatted = torch.log1p(f0_raw_pred)
            
            # Create gen_f0_vuv: (B, 2, T) with [f0, vuv]
            # Ensure temporal dimensions align before concatenation
            if f0_pred_formatted.shape[-1] != vuv.shape[-1]:
                target_len = min(f0_pred_formatted.shape[-1], vuv.shape[-1])
                if f0_pred_formatted.shape[-1] != target_len:
                    f0_pred_formatted = f0_pred_formatted[..., :target_len]
                if vuv.shape[-1] != target_len:
                    vuv = vuv[..., :target_len]
            # Use predicted F0 and GT vuv (vuv prediction can be added later if needed)
            gen_f0_vuv_output = torch.cat([f0_pred_formatted, vuv], dim=1)  # (B, 2, T)
        else:
            f0_vuv_recons = f0_decoder_out
            f0_fcpe_latent = None
            f0_fcpe_logits = None
            gen_f0_vuv_output = f0_vuv_recons[-1]  # Use predicted F0
        
        return {
            'f0_vuv': f0_vuv,
            'z_f0': z_f0,
            'f0_vuv_recons': f0_vuv_recons,  # Pass full list for CodecDecoder f0_conds
            'f0_vq_code': f0_output_dict['indices'],
            'f0_vq_loss': f0_output_dict['vq_loss'],
            'f0_perplexity': f0_output_dict['perplexity'],
            'f0_active_num': f0_output_dict['active_num'],
            'f0_fcpe_latent': f0_fcpe_latent if self.cfg.model.f0_codec.get('use_fcpe_loss', False) else None,
            'f0_fcpe_logits': f0_fcpe_logits if self.cfg.model.f0_codec.get('use_fcpe_loss', False) else None,
            'gt_f0': f0,  # Raw F0 for FCPE loss computation
            'gt_f0_log': torch.log1p(f0),
            'gen_f0_vuv': gen_f0_vuv_output,  # For visualization and loss computation
        }

    # @autocast(device_type='cuda', enabled=False)
    def forward(self, batch):
        wav = batch['wav']
        wav_24k = batch['wav_24k']
        ref_wav = batch['ref_wav']
        x_vector, d_vector, x_quantized, _ = self.model['speaker_encoder'](ref_wav)
        
        use_mhca = CodecDecoderSpeakerConditionConfig.from_decoder_cfg(self.cfg.model.codec_decoder).use_mhca
        
        spkcfg = self.cfg.model.speaker_encoder
        spk_stack_cfg = spkcfg.get('stack', None)
        use_speaker_stack = bool(spk_stack_cfg is not None and spk_stack_cfg.get('use', False))
        use_quantizer = getattr(spkcfg, 'use_quantizer', True)
        if spkcfg.use_perceiver_encoder or use_speaker_stack:
            if use_quantizer:
                if self.total_step < self.cfg.train.transit_step: 
                    spk_cond_vec = x_vector
                else: 
                    spk_cond_vec = d_vector
            else:
                spk_cond_vec = d_vector
        else:
            spk_cond_vec = x_vector
        
        # Prepare spk_cond: if use_mhca, pass as tuple (gq_vector, x_quantized)
        if use_mhca:
            spk_cond = (spk_cond_vec, x_quantized)
        else:
            spk_cond = spk_cond_vec

        # encoder가 freeze 상태면 no_grad로 forward
        enc_no_grad = self.training and self._is_encoder_frozen()

        if self.cfg.model.codec_encoder.use_vqw2v:
            wav_ = self.pad_for_wav2vec(wav)[0]
            with torch.set_grad_enabled(not enc_no_grad):
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
        else:
            raise NotImplementedError("Forward is only available when using VQ-Wav2Vec encoder, for tricodec")

        f0_outputs = self.forward_f0_codec(batch, spk_cond_vec=spk_cond_vec, x_quantized=x_quantized)
        vq_post_emb, vq_code, vq_loss, perplexity, active_num = self.model['generator'](vq_emb, total_step=self.total_step, vq=True)
        vq_post_emb_ = vq_post_emb + spk_cond_vec.unsqueeze(-1)
        y_ = self.model['generator'](vq_post_emb_, vq=False, spk_cond=spk_cond, f0_conds=f0_outputs['f0_vuv_recons'], vuv=f0_outputs['f0_vuv'][:, 1:, :]) # [B, 1, T]
        y = wav.unsqueeze(1) if self.cfg.preprocess.audio.sr == 16000 else wav_24k.unsqueeze(1)
        output = {
            'gt_wav': y,
            'gen_wav': y_,
            'vq_loss': vq_loss,
            'vq_code': vq_code,
            'vq_emb': vq_emb,
            'spk_emb': spk_cond_vec,
            'x_vector': x_vector,
            'd_vector': d_vector,
            'perplexity': perplexity,
            'active_num': active_num,

            'f0_vq_loss': f0_outputs['f0_vq_loss'],
            'gt_f0_vuv': f0_outputs['f0_vuv'],
            'gen_f0_vuv': f0_outputs['gen_f0_vuv'],
            'f0_perplexity': f0_outputs['f0_perplexity'],
            'f0_active_num': f0_outputs['f0_active_num'],
            'f0_fcpe_latent': f0_outputs.get('f0_fcpe_latent', None),
            'gt_f0': f0_outputs.get('gt_f0', None),
            'gt_f0_log': f0_outputs.get('gt_f0_log', None),
        }
        return output
    
    def tokenizer(self, batch):
        spk_type = getattr(self.cfg.model.speaker_encoder, 'speaker_encoder_type', 'ecapa_tdnn')
        if spk_type == 'wavlm':
            global_tokens = self.model['speaker_encoder'].tokenize_wav(batch['ref_wav'])
        else:
            global_tokens = self.model['speaker_encoder'].tokenize(batch['mel'].transpose(1, 2))
        if self.cfg.model.codec_encoder.use_vqw2v:
            wav_ = self.pad_for_wav2vec(batch['wav'])[0]
            with torch.no_grad():
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
        else:
            raise NotImplementedError("Tokenizer is only available when using VQ-Wav2Vec encoder.")
        _, vq_code, _, _, _= self.model['generator'](vq_emb, total_step=None, vq=True)

        return {
            'global_tokens': global_tokens,
            'semantic_tokens': vq_code
        }

    def mi_training_step(self, mi_opt, spk_emb, vq_emb):
        cfgtrain = self.cfg.train
        cmi_steps = cfgtrain.cmi_steps
        # detach 원본
        emb_tmp = spk_emb.detach()
        mu_tmp_full = vq_emb.transpose(1, 2).detach()  # (B, T, D_vq)

        last = {}
        for _ in range(cmi_steps):
            mi_opt.zero_grad()

            # forward (마지막 step 것만 기록됨)
            club_ll = self.model['mi_club'].loglikeli(emb_tmp, mu_tmp_full)
            club_loss = -club_ll

            mi_lower = self.model['mi_mine'](emb_tmp, mu_tmp_full)   # lower bound (forward)
            mine_loss = -mi_lower

            mi_upper = self.model['mi_club'].mi_est(emb_tmp, mu_tmp_full)  # upper surrogate
            delta = mi_upper - mi_lower
            gap_loss = torch.relu(delta)

            total = club_loss + mine_loss + gap_loss
            self.manual_backward(total)
            self.clip_gradients(mi_opt,
                                gradient_clip_val=self.cfg.train.mi_grad_clip,
                                gradient_clip_algorithm='norm')
            mi_opt.step()

            last = {
                'club_loss': club_loss.detach(),
                'mine_loss': mine_loss.detach(),
                'gap_loss': gap_loss.detach(),
                'mi_upper': mi_upper.detach(),
                'mi_lower': mi_lower.detach(),
                'delta': delta.detach()
            }

        return last

    def _compute_mi_metrics(self, spk_emb, vq_emb, use_penalty_relu=True):
        vq_seq = vq_emb.transpose(1, 2)
        # penalty 경로(upper)만 grad 유지
        mi_upper = self.model['mi_club'].mi_est(spk_emb, vq_seq)
        if use_penalty_relu:
            mi_penalty = torch.relu(mi_upper)
        else:
            mi_penalty = mi_upper
        # 나머지 로깅 값은 그래프 불필요
        with torch.no_grad():
            mi_lower = self.model['mi_mine'](spk_emb, vq_seq)
            club_ll = self.model['mi_club'].loglikeli(spk_emb, vq_seq)
            club_loss = -club_ll
            mine_loss = -mi_lower
            delta = mi_upper - mi_lower
            gap_loss = torch.relu(delta)
        return {
            'mi_loss': mi_penalty.detach(),
            'club_loss': club_loss,
            'mine_loss': mine_loss,
            'gap_loss': gap_loss,
            'mi_upper': mi_upper.detach(),
            'mi_lower': mi_lower.detach(),
            'delta': delta.detach(),
        }, mi_penalty

    def compute_disc_loss(self, batch, output):
        y, y_ = output['gt_wav'], output['gen_wav']
        y_ = y_.detach()
        p = self.model['discriminator'](y)
        p_ = self.model['discriminator'](y_)
        
        real_loss_list, fake_loss_list = [], []
        for i in range(len(p)):
            real_loss, fake_loss = self.criteria['gan_loss'].disc_loss(p[i][-1], p_[i][-1])
            real_loss_list.append(real_loss)
            fake_loss_list.append(fake_loss)

        if 'spec_discriminator' in self.model:
            sd_p = self.model['spec_discriminator'](y)
            sd_p_ = self.model['spec_discriminator'](y_)

            for i in range(len(sd_p)):
                real_loss, fake_loss = self.criteria['gan_loss'].disc_loss(sd_p[i][-1], sd_p_[i][-1])
                real_loss_list.append(real_loss)
                fake_loss_list.append(fake_loss)
        
        real_loss = sum(real_loss_list)
        fake_loss = sum(fake_loss_list)

        disc_loss = real_loss + fake_loss
        disc_loss = self.cfg.train.lambdas.lambda_disc * disc_loss
        
        output = {
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'disc_loss': disc_loss,
            'disc_real_feats': p,  # D(real) 결과 캐시
        }
        if 'spec_discriminator' in self.model:
            output['spec_disc_real_feats'] = sd_p

        return output
    
    def compute_gen_loss(self, batch, output, cached_real=None, cached_spec_real=None):
        y, y_ = output['gt_wav'], output['gen_wav']
        f0_vuv, f0_vuv_ = output['gt_f0_vuv'], output['gen_f0_vuv']
        vq_loss = output['vq_loss']
        f0_vq_loss =output['f0_vq_loss']
        x_vector, d_vector = output['x_vector'], output['d_vector']
        f0_fcpe_latent = output.get('f0_fcpe_latent', None)
        gt_f0 = output.get('gt_f0', None)
        gt_f0_log = output.get('gt_f0_log', None)
        gen_loss = torch.tensor(0.0, device=self.device)
        self.set_discriminator_gradients(False)
        loss_dict = {}
        cfg = self.cfg.train
        
        if cfg.use_mel_loss:
            mel_loss = self.criteria['mel_loss'](y_.squeeze(1), y.squeeze(1))
            gen_loss += mel_loss * cfg.lambdas.lambda_mel_loss
            loss_dict['mel_loss'] = mel_loss
        
        # gan loss
        p_ = self.model['discriminator'](y_)
        adv_loss_list = []
        for i in range(len(p_)):
            adv_loss_list.append(self.criteria['gan_loss'].gen_loss(p_[i][-1]))
        if 'spec_discriminator' in self.model:
            sd_p_ = self.model['spec_discriminator'](y_)
            for i in range(len(sd_p_)):
                adv_loss_list.append(self.criteria['gan_loss'].gen_loss(sd_p_[i][-1]))
        adv_loss = sum(adv_loss_list)
        gen_loss += adv_loss * cfg.lambdas.lambda_adv
        loss_dict['adv_loss'] = adv_loss

        # fm loss
        if cfg.use_feat_match_loss:
            fm_loss = torch.tensor(0.0, device=self.device)
            with torch.no_grad():
                p = cached_real if cached_real is not None else self.model['discriminator'](y)
            for i in range(len(p_)):
                for j in range(len(p_[i]) - 1):
                    fm_loss += self.criteria['fm_loss'](p_[i][j], p[i][j].detach())
            gen_loss += fm_loss * cfg.lambdas.lambda_feat_match_loss
            loss_dict['fm_loss'] = fm_loss
            if 'spec_discriminator' in self.model:
                spec_fm_loss = torch.tensor(0.0, device=self.device)
                with torch.no_grad():
                    sd_p = cached_spec_real if cached_spec_real is not None else self.model['spec_discriminator'](y)
                for i in range(len(sd_p_)):
                    for j in range(len(sd_p_[i]) - 1):
                        spec_fm_loss += self.criteria['fm_loss'](sd_p_[i][j], sd_p[i][j].detach())
                gen_loss += spec_fm_loss * cfg.lambdas.lambda_feat_match_loss
                loss_dict['spec_fm_loss'] = spec_fm_loss

        # vq
        if vq_loss is not None:
            if isinstance(vq_loss, list):
                vq_loss = sum(vq_loss)
            if isinstance(vq_loss, torch.Tensor) and vq_loss.dim() > 0:
                vq_loss = vq_loss.mean()
            loss_dict['vq_loss'] = vq_loss
            if self.cfg.model.codec_decoder.quantizer_type == "simvq":
                vq_loss = self.cfg.train.lambdas.lambda_vq_loss * vq_loss
            gen_loss = gen_loss + vq_loss

        spkcfg = self.cfg.model.speaker_encoder
        spk_stack_cfg = spkcfg.get('stack', None)
        use_pe = bool(spkcfg.use_perceiver_encoder or (spk_stack_cfg is not None and spk_stack_cfg.get('use', False)))
        use_quantizer = getattr(spkcfg, 'use_quantizer', True)
        transit_ok = self.total_step > self.cfg.train.transit_step
        fade_step = self.cfg.train.xd_loss_fade_step
        in_fade_window = (fade_step is None) or (self.total_step < fade_step)

        if use_pe and use_quantizer and transit_ok and in_fade_window:
            xd_loss = self.criteria['l2_loss'](x_vector.detach(), d_vector)
            gen_loss += xd_loss * cfg.lambdas.lambda_xd_loss
            loss_dict['xd_loss'] = xd_loss
        else:
            loss_dict['xd_loss'] = torch.tensor(0., device=self.device)
        
        # f0 loss
        if f0_vq_loss is not None:
            if isinstance(f0_vq_loss, list):
                f0_vq_loss = sum(f0_vq_loss)
            if isinstance(f0_vq_loss, torch.Tensor) and f0_vq_loss.dim() > 0:
                f0_vq_loss = f0_vq_loss.mean()
            loss_dict['f0_vq_loss'] = f0_vq_loss
            f0_vq_loss = self.cfg.train.lambdas.lambda_f0_vq_loss * f0_vq_loss
            gen_loss = gen_loss + f0_vq_loss
        else:
            # TriCodec: no separate F0 VQ loss (when using codec structure without separate quantizer)
            loss_dict['f0_vq_loss'] = torch.tensor(0.0, device=self.device)

        # print('gen f0 max min', torch.max(f0_vuv_.select(1, 0)).item(), torch.min(f0_vuv_.select(1, 0)).item())
        # print('gen vuv max min', torch.max(f0_vuv_.select(1, 1)).item(), torch.min(f0_vuv_.select(1, 1)).item())
        # gt f0_vuv is always 2 channels: [0]=f0, [1]=vuv
        # FCPE mode: f0_vuv_ is GT (not predicted), skip F0 reconstruction loss
        use_fcpe = self.cfg.model.f0_codec.get('use_fcpe_loss', False)
        use_unnorm_mse = self.use_unnormf0_mse_loss
        if not use_fcpe:
            vuv = f0_vuv.select(1, 1)
            f0_pred = f0_vuv_.select(1, 0)
            if use_unnorm_mse:
                f0_target = gt_f0_log.squeeze(1) if gt_f0_log is not None else f0_vuv.select(1, 0)
            else:
                f0_target = f0_vuv.select(1, 0)

            if self.cfg.model.f0_codec.zero_out_all_unvoiced:
                vuv_pred = f0_vuv_.select(1, 1)
                vuv_bce_loss = self.criteria['bcewlogits_loss'](vuv_pred, vuv)
            else:
                vuv_bce_loss = torch.tensor(0.0, device=self.device)

            if f0_target.dim() == 3:
                f0_target = f0_target.squeeze(1)
            if f0_pred.dim() == 3:
                f0_pred = f0_pred.squeeze(1)
            vuv_mask = vuv
            if vuv_mask.dim() == 3:
                vuv_mask = vuv_mask.squeeze(1)
            f0 = f0_target
            f0_ = f0_pred
        else:
            # FCPE mode: no direct F0 reconstruction
            vuv_bce_loss = torch.tensor(0.0, device=self.device)
        
        # F0 reconstruction loss: skip if using FCPE loss
        # When use_fcpe_loss=True, we only train with FCPE latent BCE loss
        if not use_fcpe:
            f0_l2_loss = self.criteria['l2_loss'](f0_, f0)
        else:
            f0_l2_loss = torch.tensor(0.0, device=self.device)

        # FCPE-style loss (optional)
        f0_fcpe_logits = output.get('f0_fcpe_logits', None)
        if use_fcpe and f0_fcpe_latent is not None and gt_f0 is not None:
            # gt_f0: (B, 1, T), f0_fcpe_latent: (B, fcpe_out_dims, T)
            # Convert GT F0 to FCPE target latent (gaussian blurred cent)
            with torch.no_grad():
                # f0_to_cent and gaussian_blurred_cent2latent expect (B, T, 1) input
                # gt_f0 is (B, 1, T), so transpose it
                gt_f0_transposed = gt_f0.transpose(1, 2)  # (B, T, 1)
                
                # Convert to cent and then to gaussian blurred latent
                # Note: f0=0 (unvoiced) → cent=-inf, but gaussian_blurred_cent2latent's mask filters it out
                # The mask is (cents > 0.1), so -inf frames get masked to 0 in the final latent
                gt_cent_f0 = self.model['f0_extractor'].f0_to_cent(gt_f0_transposed)  # (B, T, 1), may contain -inf
                gt_fcpe_latent = self.model['f0_extractor'].gaussian_blurred_cent2latent(gt_cent_f0)  # (B, T, 360), -inf masked to 0
                
                # Transpose to (B, 360, T) to match f0_fcpe_latent shape
                gt_fcpe_latent = gt_fcpe_latent.transpose(1, 2)  # (B, 360, T)
            
            # Align temporal dimensions if needed.
            # FCPE latent loss requires exact shape match; different hop sizes / downsample ratios
            # can cause a systematic mismatch (e.g. 24k@480 vs 16k@320 → ×1.5).
            if f0_fcpe_latent.shape[-1] != gt_fcpe_latent.shape[-1]:
                gt_fcpe_latent = F.interpolate(
                    gt_fcpe_latent,
                    size=int(f0_fcpe_latent.shape[-1]),
                    mode='linear',
                    align_corners=False,
                )

            if f0_fcpe_logits is None:
                raise RuntimeError("FCPE loss enabled but f0_decoder did not expose fcpe logits.")
            if f0_fcpe_logits.shape[-1] != gt_fcpe_latent.shape[-1]:
                f0_fcpe_logits = F.interpolate(
                    f0_fcpe_logits,
                    size=int(gt_fcpe_latent.shape[-1]),
                    mode='linear',
                    align_corners=False,
                )

            fcpe_logits_fp32 = f0_fcpe_logits.float()
            gt_fcpe_latent_fp32 = gt_fcpe_latent.float()
            if not torch.isfinite(fcpe_logits_fp32).all():
                finite_mask = torch.isfinite(fcpe_logits_fp32)
                finite_vals = fcpe_logits_fp32[finite_mask]
                stats = "all values are non-finite"
                if finite_vals.numel() > 0:
                    stats = (
                        f"finite_min={finite_vals.min().item():.4g}, "
                        f"finite_max={finite_vals.max().item():.4g}, "
                        f"finite_mean={finite_vals.mean().item():.4g}"
                    )
                raise RuntimeError(
                    "Non-finite FCPE logits detected before BCEWithLogitsLoss "
                    f"(step={self.global_step}, rank={self.global_rank}, {stats}, "
                    f"non_finite={int((~finite_mask).sum().item())})."
                )
            if not torch.isfinite(gt_fcpe_latent_fp32).all():
                raise RuntimeError(
                    "Non-finite FCPE targets detected before BCEWithLogitsLoss "
                    f"(step={self.global_step}, rank={self.global_rank})."
                )

            with autocast(device_type=fcpe_logits_fp32.device.type, enabled=False):
                fcpe_loss = F.binary_cross_entropy_with_logits(
                    fcpe_logits_fp32,
                    gt_fcpe_latent_fp32,
                )
            loss_dict['fcpe_loss'] = fcpe_loss
            gen_loss = gen_loss + fcpe_loss * cfg.lambdas.get('lambda_fcpe_loss', 10.0)
        else:
            loss_dict['fcpe_loss'] = torch.tensor(0.0, device=self.device)

        loss_dict['f0_loss'], loss_dict['vuv_loss'] = f0_l2_loss, vuv_bce_loss
        gen_loss = gen_loss + vuv_bce_loss * cfg.lambdas.lambda_vuv_recon_loss
        gen_loss = gen_loss + f0_l2_loss * cfg.lambdas.lambda_f0_recon_loss

        self.set_discriminator_gradients(True)
        loss_dict['gen_loss'] = gen_loss
        return loss_dict
    
    def training_step(self, batch, batch_idx):
        # 스텝별 encoder freeze/unfreeze 반영
        self._maybe_update_encoder_freeze()
        output = self(batch)

        opts = self.optimizers()
        if self.cfg.model.get('use_mi', False):
            gen_opt, disc_opt, mi_opt = opts
        else:
            gen_opt, disc_opt = opts
            mi_opt = None
        gen_sche, disc_sche = self.lr_schedulers()
        
        accum = int(self.cfg.train.gradient_accumulation_steps)
        is_update_step = ((batch_idx + 1) % accum == 0)
    
        if mi_opt:
            self.toggle_optimizer(mi_opt)
            mi_losses = self.mi_training_step(mi_opt, output['spk_emb'], output['vq_emb'])
            self.untoggle_optimizer(mi_opt)
        else:
            mi_losses = None

        # 1) Discriminator
        self.toggle_optimizer(disc_opt)
        disc_losses = self.compute_disc_loss(batch, output)
        disc_loss = disc_losses['disc_loss'] / accum
        try:
            # DDP에서 그래드 적산 중에는 통신 생략
            with self._no_sync_if_needed(self._sync_needed() and not is_update_step):
                self.manual_backward(disc_loss)
            if is_update_step:
                self.clip_gradients(disc_opt, gradient_clip_val=self.cfg.train.disc_grad_clip, gradient_clip_algorithm='norm')
                disc_opt.step()
                disc_opt.zero_grad()
                if disc_sche is not None:
                    disc_sche.step()
        finally:
            self.untoggle_optimizer(disc_opt)

        # 2) Generator
        self.toggle_optimizer(gen_opt)
        # toggle_optimizer가 requires_grad를 바꾸므로 즉시 재적용
        self._maybe_update_encoder_freeze()
        gen_losses = self.compute_gen_loss(
            batch, output,
            cached_real=disc_losses.get('disc_real_feats'),
            cached_spec_real=disc_losses.get('spec_disc_real_feats')
        )
        lambda_mi = self.cfg.train.lambdas.get('lambda_mi_loss', 0.0)
        if lambda_mi > 0 and self.cfg.model.get('use_mi', False):
            # Lightning toggle 으로 mi_club / mi_mine 파라미터는 requires_grad=False 상태
            # 재계산(메모리 절약): speaker / vq 경로만 gradient
            if mi_losses is None:
                mi_losses = {}
            metrics, mi_penalty = self._compute_mi_metrics(output['spk_emb'], output['vq_emb'])
            gen_losses['gen_loss'] = gen_losses['gen_loss'] + lambda_mi * mi_penalty
            # --- add (ensure mi_loss logged) ---
            mi_losses.update(metrics)

        gen_loss = gen_losses['gen_loss'] / accum
        try:
            # DDP에서 그래드 적산 중에는 통신 생략
            with self._no_sync_if_needed(self._sync_needed() and not is_update_step):
                self.manual_backward(gen_loss)
            if is_update_step:
                self.clip_gradients(gen_opt, gradient_clip_val=self.cfg.train.gen_grad_clip, gradient_clip_algorithm='norm')
                gen_opt.step()
                gen_opt.zero_grad()
                if gen_sche is not None:
                    gen_sche.step()
        finally:
            self.untoggle_optimizer(gen_opt)

        if is_update_step:
            self.total_step += 1
            self._log_losses(
                stage='train',
                disc_losses=disc_losses, gen_losses=gen_losses, mi_losses=mi_losses,
                output=output, batch_size=self.cfg.dataset.train.batch_size,
                gen_opt=gen_opt, disc_opt=disc_opt,
                on_step=True, on_epoch=False
            )

    def on_validation_epoch_start(self):
        self.stoi.reset()
        self.pesq_wb.reset()
        # self.pesq_nb.reset()
        self.si_snr.reset()
        self.si_sdr.reset()
        
        # Skip heavy model loading during sanity check
        is_sanity_check = self.trainer.sanity_checking if hasattr(self.trainer, 'sanity_checking') else False
        
        # Initialize UTMOS/WavLM accumulators
        self.val_utmos_same_scores = []
        self.val_utmos_vc_scores = []
        self.val_wavlm_same_scores = []
        self.val_wavlm_vc_scores = []
        if self.use_val_wer:
            self.val_wer_same_scores = []
            self.val_wer_vc_scores = []
        
        # Skip loading heavy models during sanity check
        if is_sanity_check:
            return
        
        # Load UTMOS model if needed
        if self.use_val_utmos and self.utmos_predictor is None:
            print(colored("Loading UTMOS predictor for validation...", "yellow"))
            # 상대 경로를 절대 경로로 변환하여 캐시 재사용
            import os.path as osp
            torch_cache_dir = osp.join(osp.dirname(__file__), '..', '..', 'etc', 'torch_hub_cache')
            os.makedirs(torch_cache_dir, exist_ok=True)
            os.environ['TORCH_HOME'] = torch_cache_dir
            self.utmos_predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
            self.utmos_predictor = self.utmos_predictor.to(self.device)
            self.utmos_predictor.eval()
            print(colored("✓ UTMOS loaded", "green"))
        
        # Load WavLM model if needed
        if self.use_val_wavlm and self.wavlm_model is None:
            print(colored("Loading WavLM_large model for validation...", "yellow"))
            # s3prl 캐시 경로 설정
            import os.path as osp
            s3prl_cache_dir = osp.join(osp.dirname(__file__), '..', '..', 'etc', 'torch_hub_cache', 's3prl')
            os.makedirs(s3prl_cache_dir, exist_ok=True)
            os.environ['S3PRL_CACHE_ROOT'] = s3prl_cache_dir
            
            import sys
            from pathlib import Path as _Path
            sv_dir = _Path(__file__).parent.parent.parent / "etc" / "UniSpeech" / "downstreams" / "speaker_verification"
            sys.path.append(str(sv_dir))
            from etc.unispeech_models.ecapa_tdnn import ECAPA_TDNN_SMALL
            self.wavlm_model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None).to(self.device)
            self.wavlm_model.eval()
            checkpoint_path = _Path(__file__).parent.parent.parent / "etc" / "unispeech_models" / "wavlm_large_finetune.pth"
            if checkpoint_path.exists():
                state_dict = torch.load(str(checkpoint_path), map_location=self.device)
                # Filter out keys that don't exist in the model (e.g., loss_calculator components)
                model_state_dict = self.wavlm_model.state_dict()
                filtered_state_dict = {k: v for k, v in state_dict['model'].items() if k in model_state_dict}
                missing_keys, unexpected_keys = self.wavlm_model.load_state_dict(filtered_state_dict, strict=False)
                if missing_keys:
                    print(colored(f"Warning: Missing keys when loading WavLM: {missing_keys}", "yellow"))
                if unexpected_keys:
                    print(colored(f"Warning: Unexpected keys in checkpoint (filtered out): {unexpected_keys}", "yellow"))
                print(colored("✓ WavLM loaded with checkpoint", "green"))
            else:
                raise FileNotFoundError(f"WavLM checkpoint not found at {checkpoint_path}")
        
        if self.use_val_wer and self.hubert_model is None:
            print(colored("Loading HuBERT model for validation WER...", "yellow"))
            self.hubert_processor = AutoProcessor.from_pretrained(self.val_hubert_model_name)
            self.hubert_model = HubertForCTC.from_pretrained(self.val_hubert_model_name).to(self.device)
            self.hubert_model.eval()
            print(colored("✓ HuBERT loaded", "green"))

    def on_validation_epoch_end(self):
        assert self.si_snr.total == self.si_sdr.total == self.stoi.count == self.pesq_wb.count, f"Metrics count mismatch: {self.si_snr.total}, {self.si_sdr.total}, {self.stoi.count}, {self.pesq_wb.count}"
        self.log('val_stats/si_snr', self.si_snr.compute(), on_epoch=True, logger=True, sync_dist=True)
        self.log('val_stats/si_sdr', self.si_sdr.compute(), on_epoch=True, logger=True, sync_dist=True)
        self.log('val_stats/stoi', self.stoi.compute(), on_epoch=True, logger=True, sync_dist=True)
        self.log('val_stats/pesq_wb', self.pesq_wb.compute(), on_epoch=True, logger=True, sync_dist=True)
        
        # Log UTMOS scores if available
        if self.use_val_utmos and len(self.val_utmos_same_scores) > 0:
            mean_utmos_same = float(np.mean(self.val_utmos_same_scores))
            self.log('val_stats/utmos_same', mean_utmos_same, on_epoch=True, logger=True, sync_dist=True)
            print(colored(f"Mean UTMOS (same-speaker): {mean_utmos_same:.4f}", "cyan"))
            
        if self.use_val_utmos and len(self.val_utmos_vc_scores) > 0:
            mean_utmos_vc = float(np.mean(self.val_utmos_vc_scores))
            self.log('val_stats/utmos_vc', mean_utmos_vc, on_epoch=True, logger=True, sync_dist=True)
            print(colored(f"Mean UTMOS (VC): {mean_utmos_vc:.4f}", "cyan"))
        
        # Log WavLM speaker similarity scores if available
        if self.use_val_wavlm and len(self.val_wavlm_same_scores) > 0:
            mean_wavlm_same = float(np.mean(self.val_wavlm_same_scores))
            self.log('val_stats/wavlm_sim_same', mean_wavlm_same, on_epoch=True, logger=True, sync_dist=True)
            print(colored(f"Mean WavLM Similarity (same-speaker): {mean_wavlm_same:.4f}", "cyan"))
            
        if self.use_val_wavlm and len(self.val_wavlm_vc_scores) > 0:
            mean_wavlm_vc = float(np.mean(self.val_wavlm_vc_scores))
            self.log('val_stats/wavlm_sim_vc', mean_wavlm_vc, on_epoch=True, logger=True, sync_dist=True)
            print(colored(f"Mean WavLM Similarity (VC): {mean_wavlm_vc:.4f}", "cyan"))

        if self.use_val_wer and len(getattr(self, 'val_wer_same_scores', [])) > 0:
            mean_wer_same = float(np.mean(self.val_wer_same_scores))
            self.log('val_stats/wer_same', mean_wer_same, on_epoch=True, logger=True, sync_dist=True)
            print(colored(f"Mean WER (same-speaker, HuBERT): {mean_wer_same:.4f}", "cyan"))
        if self.use_val_wer and len(getattr(self, 'val_wer_vc_scores', [])) > 0:
            mean_wer_vc = float(np.mean(self.val_wer_vc_scores))
            self.log('val_stats/wer_vc', mean_wer_vc, on_epoch=True, logger=True, sync_dist=True)
            print(colored(f"Mean WER (VC, HuBERT): {mean_wer_vc:.4f}", "cyan"))

        # Clean up validation models to free memory
        if self.utmos_predictor is not None:
            del self.utmos_predictor
            self.utmos_predictor = None
            print(colored("✓ UTMOS predictor deleted", "cyan"))
        
        if self.wavlm_model is not None:
            del self.wavlm_model
            self.wavlm_model = None
            print(colored("✓ WavLM model deleted", "cyan"))
        
        if self.hubert_model is not None:
            del self.hubert_model
            self.hubert_model = None
            print(colored("✓ HuBERT model deleted", "cyan"))
        
        if self.hubert_processor is not None:
            del self.hubert_processor
            self.hubert_processor = None
            print(colored("✓ HuBERT processor deleted", "cyan"))
        
        torch.cuda.empty_cache()

        # 시각화 처리

        is_main = True
        if torch.distributed.is_initialized():
            is_main = (torch.distributed.get_rank() == 0)

        if self.val_step_plot_outputs and is_main:
            matplotlib.use('Agg')
            samples = self.val_step_plot_outputs
            sr, hop_length = 16000, 80
            
            # FCPE 모드: raw Hz 표시, Normal/Log1p 모드: normalized 값 표시
            use_fcpe_mode = (
                self.cfg.model.f0_codec.get('use_normalized_f0', False)
                and self.cfg.model.f0_codec.get('use_fcpe_loss', False)
            )
            use_log1p_mode = self.use_unnormf0_mse_loss
            
            # 3x3 grid layout (공통)
            rows, cols = 3, min(3, (len(samples) + 2) // 3)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)

            for i, sample in enumerate(samples[:rows * cols]):
                r, c = i % rows, i // rows
                mel = sample['mel'].numpy()
                vuv_gt, vuv_rec = sample['vuv_gt'].numpy().squeeze(), sample['vuv_rec'].numpy().squeeze()
                
                # FCPE 모드: raw Hz로 변환
                if use_fcpe_mode:
                    f0_gt_raw = sample['f0_gt_raw']
                    f0_gt = f0_gt_raw.numpy().squeeze() if f0_gt_raw is not None else None
                    f0_rec = np.expm1(sample['f0_rec_z'].numpy().squeeze())  # log1p → Hz
                    ylabel, yscale, ylim = 'F0 (Hz)', 'log', (50, 500)
                elif use_log1p_mode:
                    # Unnormalized MSE 모드: log1p(raw F0)
                    f0_gt = sample['f0_gt_z'].numpy().squeeze()
                    f0_rec = sample['f0_rec_z'].numpy().squeeze()
                    ylabel, yscale, ylim = 'log1p F0', 'linear', None
                else:
                    # Normal 모드: normalized F0 사용
                    f0_gt = sample['f0_gt_z'].numpy().squeeze()
                    f0_rec = sample['f0_rec_z'].numpy().squeeze()
                    ylabel, yscale, ylim = 'Norm log-F0 (z)', 'linear', None
                
                times = librosa.times_like(f0_gt if f0_gt is not None else f0_rec, sr=sr, hop_length=hop_length)
                
                # Mel spectrogram
                librosa.display.specshow(mel, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='viridis', ax=axes[r][c])
                axes[r][c].set_title(sample['fid'])
                
                # F0 overlay
                ax2 = axes[r][c].twinx()
                if f0_gt is not None:
                    ax2.plot(times, f0_gt, 'cyan', lw=1.2, label='f0_gt', zorder=3)
                ax2.plot(times, f0_rec, 'r--', lw=1.0, label='f0_rec', zorder=2)

                # V/UV shading
                if vuv_gt.max() > 0 or vuv_rec.max() > 0:
                    if f0_gt is not None:
                        ymin, ymax = min(f0_gt.min(), f0_rec.min()), max(f0_gt.max(), f0_rec.max())
                    else:
                        ymin, ymax = f0_rec.min(), f0_rec.max()
                    if vuv_gt.max() > 0:
                        ax2.fill_between(times, ymin * 0.9, ymax * 1.1, where=vuv_gt > 0.5, 
                                       color='white', alpha=0.2, interpolate=True, label='voiced_gt', zorder=0)
                    if vuv_rec.max() > 0:
                        ax2.fill_between(times, ymin * 0.9, ymax * 1.1, where=vuv_rec > 0.5,
                                       color='magenta', alpha=0.15, interpolate=True, label='voiced_pred', zorder=1)
                    if not use_fcpe_mode:  # Normal & log1p 모드: 기존 margin 계산
                        margin = 0.1 * max(1e-6, abs(f0_gt).max() + abs(f0_rec).max()) if f0_gt is not None else 0.1
                        ax2.set_ylim(ymin - margin, ymax + margin)
                
                ax2.set_ylabel(ylabel)
                if yscale: ax2.set_yscale(yscale)
                if ylim: ax2.set_ylim(ylim)
                ax2.legend(loc='upper right', fontsize=7)
                ax2.grid(alpha=0.25, ls=':')

            plt.tight_layout()
            if self.logger and isinstance(self.logger, pl.loggers.wandb.WandbLogger):
                out_dir = "val_vis"
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, f"val_mel_f0{'_fcpe' if use_fcpe_mode else ''}_epoch_{self.current_epoch}.png")
                plt.savefig(path, dpi=150, bbox_inches='tight')
                self.logger.experiment.log({'val/mel_f0_grid': wandb.Image(path)}, step=int(self.global_step))
            plt.close()

        self.val_step_plot_outputs.clear()

    def _transcribe_hubert(self, audio_16k: torch.Tensor) -> str:
        if self.hubert_processor is None or self.hubert_model is None:
            raise RuntimeError("HuBERT processor/model not initialized. Enable cfg.train.use_val_wer.")
        if audio_16k.dim() > 1:
            audio_16k = audio_16k.squeeze(0)
        audio_np = audio_16k.detach().cpu().float().numpy()
        inputs = self.hubert_processor(audio_np, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            logits = self.hubert_model(input_values, attention_mask=attention_mask).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = self.hubert_processor.batch_decode(pred_ids)[0].strip()
        return text

    def _compute_wer_jiwer(self, ref_text: str, hyp_text: str) -> float:
        out = jiwer.process_words(
            reference=ref_text,
            hypothesis=hyp_text,
            reference_transform=self._wer_transform,
            hypothesis_transform=self._wer_transform
        )
        return float(out.wer)

    def _collect_val_vis_sample(self, batch, output, batch_idx):
        # 이미 충분히 모았으면 중단
        if len(self.val_step_plot_outputs) >= 9:
            return
        matplotlib.use('Agg')
        wav = batch['wav'][0].unsqueeze(0)  # [1, T_wav]
        self.mel_spectrogram_transform.to(wav.device)
        mel = self.mel_spectrogram_transform(wav)  # [1, n_mels, T_mel]
        amin = 1e-10
        ref_value = mel.max()
        mel_db = 10.0 * torch.log10(torch.clamp(mel, min=amin) / ref_value)

        target_len = output['gt_f0_vuv'].shape[-1]
        mel_db_interp = F.interpolate(mel_db, size=target_len, mode='linear', align_corners=False).squeeze(0)

        # gt_f0_vuv is always 2 channels: [0]=f0(encoder input), [1]=vuv
        if self.use_unnormf0_mse_loss:
            f0_gt_z = output['gt_f0_log'][0, 0].detach().cpu() if output.get('gt_f0_log') is not None else output['gt_f0_vuv'][0, 0].detach().cpu()
        else:
            f0_gt_z = output['gt_f0_vuv'][0, 0].detach().cpu()
        vuv_gt = output['gt_f0_vuv'][0, 1].detach().cpu()
        
        f0_rec_z = output['gen_f0_vuv'][0, 0].detach().cpu()
        vuv_rec = output['gen_f0_vuv'][0, 1].detach().cpu() if self.cfg.model.f0_codec.zero_out_all_unvoiced else vuv_gt

        # FCPE 모드: raw GT F0도 저장 (스케일 비교용)
        f0_gt_raw = output.get('gt_f0', None)  # (B, 1, T) raw Hz
        if f0_gt_raw is not None:
            f0_gt_raw = f0_gt_raw[0, 0].detach().cpu()  # (T,)

        sample = {
            'fid': batch.get('fid', ['unk'])[0],
            'mel': mel_db_interp.detach().cpu(),    # [n_mels, T]
            'f0_gt_z': f0_gt_z.unsqueeze(0),        # [1, T] normalized or log1p
            'f0_rec_z': f0_rec_z.unsqueeze(0),      # [1, T] normalized or log1p
            'f0_gt_raw': f0_gt_raw.unsqueeze(0) if f0_gt_raw is not None else None,  # [1, T] raw Hz
            'vuv_gt': vuv_gt.unsqueeze(0),          # [1, T]
            'vuv_rec': vuv_rec.unsqueeze(0),        # [1, T]
        }
        self.val_step_plot_outputs.append(sample)
    
    def validation_step(self, batch, batch_idx):
        # Same-speaker reconstruction
        output = self(batch)

        disc_losses = self.compute_disc_loss(batch, output)
        gen_losses = self.compute_gen_loss(batch, output)
        mi_losses = {}
        if self.cfg.model.get('use_mi', False):
            with torch.no_grad():
                metrics, _ = self._compute_mi_metrics(output['spk_emb'], output['vq_emb'], use_penalty_relu=True)
                mi_losses.update(metrics)

        self.si_snr.update(output['gt_wav'].squeeze(1), output['gen_wav'].squeeze(1))
        self.si_sdr.update(output['gt_wav'].squeeze(1), output['gen_wav'].squeeze(1))
        self.stoi.update(output['gt_wav'].squeeze(1), output['gen_wav'].squeeze(1))
        self.pesq_wb.update(output['gt_wav'].squeeze(1), output['gen_wav'].squeeze(1))

        self._log_losses(
            stage='val',
            disc_losses=disc_losses, gen_losses=gen_losses, mi_losses=mi_losses, output=output,
            batch_size=self.cfg.dataset.val.batch_size,
            on_step=False, on_epoch=True)
        
        # Voice conversion (different speaker) - only for UTMOS/WavLM/WER metrics, no loss logging
        output_vc = None
        if 'ref_wav_vc' in batch and (self.use_val_utmos or self.use_val_wavlm or self.use_val_wer):
            batch_vc = batch.copy()
            batch_vc['ref_wav'] = batch['ref_wav_vc']
            output_vc = self(batch_vc)
        
        # Compute UTMOS and WavLM metrics directly (without saving audio)
        if self.use_val_utmos or self.use_val_wavlm or self.use_val_wer:
            with torch.no_grad():
                # Resample to 16kHz if needed for metrics
                def to_16k(audio):
                    # Squeeze to (B, T) first
                    if audio.dim() == 3:
                        audio = audio.squeeze(1)  # (B, 1, T) -> (B, T)
                    
                    if self.cfg.preprocess.audio.sr != 16000:
                        audio = torchaudio.functional.resample(
                            audio, orig_freq=self.cfg.preprocess.audio.sr, new_freq=16000
                        )
                    return audio  # (B, T)
                
                # Prepare audio: squeeze and resample
                gen_same_16k = to_16k(output['gen_wav'])  # (B, T)
                gt_same_16k = to_16k(output['gt_wav'])    # (B, T)
                ref_same_16k = batch['ref_wav']  # Already 16kHz, (B, T)
                if ref_same_16k.dim() == 3:
                    ref_same_16k = ref_same_16k.squeeze(1)
                
                # UTMOS for same-speaker
                if self.use_val_utmos and self.utmos_predictor is not None:
                    for i in range(gen_same_16k.size(0)):
                        try:
                            # UTMOS expects (1, T) or (T,)
                            audio_sample = gen_same_16k[i].unsqueeze(0)  # (1, T)
                            utmos_score = self.utmos_predictor(audio_sample, 16000)
                            if isinstance(utmos_score, torch.Tensor):
                                utmos_score = utmos_score.item()
                            self.val_utmos_same_scores.append(float(utmos_score))
                        except Exception as e:
                            print(colored(f"UTMOS error (same): {e}", "red"))
                
                # WavLM for same-speaker
                if self.use_val_wavlm and self.wavlm_model is not None:
                    try:
                        # WavLM expects (B, T)
                        ref_emb = self.wavlm_model(ref_same_16k)  # (B, D)
                        gen_emb = self.wavlm_model(gen_same_16k)  # (B, D)
                        cosine_sim = torch.nn.functional.cosine_similarity(ref_emb, gen_emb, dim=-1)
                        self.val_wavlm_same_scores.extend(cosine_sim.cpu().tolist())
                    except Exception as e:
                        print(colored(f"WavLM error (same): {e}", "red"))

                # HuBERT-based WER for same-speaker reconstruction
                if self.use_val_wer and self.hubert_model is not None:
                    for i in range(gen_same_16k.size(0)):
                        try:
                            ref_text = self._transcribe_hubert(gt_same_16k[i])
                            hyp_text = self._transcribe_hubert(gen_same_16k[i])
                            wer_value = self._compute_wer_jiwer(ref_text, hyp_text)
                            self.val_wer_same_scores.append(wer_value)
                        except Exception as e:
                            print(colored(f"HuBERT WER error (same): {e}", "red"))
                            self.val_wer_same_scores.append(1.0)
                
                # Voice conversion metrics
                if 'ref_wav_vc' in batch and output_vc is not None:
                    gen_vc_16k = to_16k(output_vc['gen_wav'])  # (B, T)
                    gt_vc_16k = to_16k(output_vc['gt_wav'])    # (B, T)
                    ref_vc_16k = batch['ref_wav_vc']  # Already 16kHz, (B, T)
                    if ref_vc_16k.dim() == 3:
                        ref_vc_16k = ref_vc_16k.squeeze(1)
                    
                    # UTMOS for VC
                    if self.use_val_utmos and self.utmos_predictor is not None:
                        for i in range(gen_vc_16k.size(0)):
                            try:
                                audio_sample = gen_vc_16k[i].unsqueeze(0)  # (1, T)
                                utmos_score = self.utmos_predictor(audio_sample, 16000)
                                if isinstance(utmos_score, torch.Tensor):
                                    utmos_score = utmos_score.item()
                                self.val_utmos_vc_scores.append(float(utmos_score))
                            except Exception as e:
                                print(colored(f"UTMOS error (VC): {e}", "red"))
                    
                    # WavLM for VC
                    if self.use_val_wavlm and self.wavlm_model is not None:
                        try:
                            # WavLM expects (B, T)
                            ref_vc_emb = self.wavlm_model(ref_vc_16k)  # (B, D)
                            gen_vc_emb = self.wavlm_model(gen_vc_16k)  # (B, D)
                            cosine_sim_vc = torch.nn.functional.cosine_similarity(ref_vc_emb, gen_vc_emb, dim=-1)
                            self.val_wavlm_vc_scores.extend(cosine_sim_vc.cpu().tolist())
                        except Exception as e:
                            print(colored(f"WavLM error (VC): {e}", "red"))

                    if self.use_val_wer and self.hubert_model is not None:
                        for i in range(gen_vc_16k.size(0)):
                            try:
                                ref_text = self._transcribe_hubert(gt_vc_16k[i])
                                hyp_text = self._transcribe_hubert(gen_vc_16k[i])
                                wer_value = self._compute_wer_jiwer(ref_text, hyp_text)
                                self.val_wer_vc_scores.append(wer_value)
                            except Exception as e:
                                print(colored(f"HuBERT WER error (VC): {e}", "red"))
                                self.val_wer_vc_scores.append(1.0)
                
                # Explicit memory cleanup after VC metrics
                if output_vc is not None:
                    del output_vc
                if 'batch_vc' in locals():
                    del batch_vc
                torch.cuda.empty_cache()

        is_main = True
        if torch.distributed.is_initialized():
            is_main = (torch.distributed.get_rank() == 0)

        if is_main and batch_idx < 9:
            self._collect_val_vis_sample(batch, output, batch_idx)

    def test_step(self, batch, batch_idx):
        # remove weight norm for modules in self.model
        batch_size = batch['wav'].size(0)
        target_type = self.cfg.voice_conversion
        self.vc_f0 = True if target_type == 'vc_f0' else False
        
        assert target_type in ['rec', 'vc', 'vc_f0', 'same'], f"Invalid target_type: {target_type}"
        
        # 먼저 이 배치에서 처리할 샘플이 있는지 확인
        indices_to_process = []
        for i in range(batch_size):
            source_filename = batch['fid'][i]
            if target_type in ['rec', 'vc', 'vc_f0']:
                target_filename = batch['target_id'][i]
            elif target_type == 'same':
                target_filename = source_filename
            
            # 생성할 파일 경로들
            gen_path = f"{self.cwd}/{source_filename}-{target_filename}_{target_type}.wav"
            gt_path = f"{self.cwd}/{source_filename}-{target_filename}_gt.wav"
            ref_path = f"{self.cwd}/{source_filename}-{target_filename}_ref.wav"
            
            # 모든 파일이 이미 존재하는지 확인
            if os.path.exists(gen_path) and os.path.exists(gt_path) and os.path.exists(ref_path):
                print(colored(f"[Skip] All files exist for {source_filename}-{target_filename}", "yellow"))
                continue
            
            indices_to_process.append(i)
        
        # 처리할 샘플이 없으면 forward pass 자체를 건너뜀
        if len(indices_to_process) == 0:
            print(colored(f"[Skip] Batch {batch_idx} - all files already exist", "green"))
            return
        
        # Forward pass (필요한 샘플이 있을 때만)
        output = self(batch)
        
        # 처리해야 할 샘플들만 저장
        for i in indices_to_process:
            source_filename = batch['fid'][i]
            if target_type in ['rec', 'vc', 'vc_f0']:
                target_filename = batch['target_id'][i]
            elif target_type == 'same':
                target_filename = source_filename
            
            # _ref.wav는 get_ref_clip 이후(ref_wav) 대신 raw ref_src를 저장해야 함 (test only)
            if isinstance(batch.get('ref_src', None), torch.Tensor):
                ref_i = batch['ref_src'][i]
            else:
                raise ValueError("ref_src is not found in batch")
            gt_i, gen_i = output['gt_wav'][i], output['gen_wav'][i]
            # gt_i와 gen_i는 이미 (1, T) shape이므로 squeeze하지 않음
            # ref_i만 확인
            if ref_i.dim() == 1: ref_i = ref_i.unsqueeze(0)
            
            gen_path = f"{self.cwd}/{source_filename}-{target_filename}_{target_type}.wav"
            gt_path = f"{self.cwd}/{source_filename}-{target_filename}_gt.wav"
            ref_path = f"{self.cwd}/{source_filename}-{target_filename}_ref.wav"
            
            # gt_wav와 gen_wav는 forward에서 이미 올바른 샘플링 레이트를 가짐
            torchaudio.save(gen_path, gen_i.float().detach().cpu(), self.cfg.preprocess.audio.sr)
            torchaudio.save(gt_path, gt_i.float().detach().cpu(), self.cfg.preprocess.audio.sr)
            # ref_wav는 항상 16kHz (데이터 로더에서 ref_wav_16k로 제공됨)
            torchaudio.save(ref_path, ref_i.float().detach().cpu(), 16000)
            # print(colored(f"[Saved] {source_filename}-{target_filename}_{target_type}.wav", "green"))

    def on_test_start(self):
        self._remove_all_weight_norms()
        self.cwd = os.path.join(os.getcwd(), os.path.splitext(self.cfg.ckpt)[0])
        os.makedirs(self.cwd, exist_ok=True)
        print(colored(f'Test results will be saved in : {self.cwd}', 'yellow', attrs=['bold']))

    def _remove_all_weight_norms(self):
        if getattr(self, "_wn_removed", False):
            return
        removed = 0
        for m in self.model.modules():
            # weight_norm 적용된 모듈은 weight_g/weight_v를 가짐
            if hasattr(m, "weight_g") and hasattr(m, "weight_v"):
                try:
                    nn.utils.remove_weight_norm(m)
                    removed += 1
                except Exception:
                    pass
        self._wn_removed = True
        print(f"[test] removed weight_norm from {removed} submodules.")

    # --- encoder freeze 스케줄 관련 유틸 ---
    def _is_encoder_frozen(self):
        if not self._freeze_schedule_enabled:
            return False
        # total_step은 optimizer step 기준으로 증가
        return int(self.total_step.item()) < int(self.unfreeze_encoder_step)

    def _maybe_update_encoder_freeze(self):
        if not self._freeze_schedule_enabled:
            return
        should_freeze = self._is_encoder_frozen()
        if self._encoder_frozen is None or self._encoder_frozen != should_freeze:
            self.set_encoder_gradients(not should_freeze)
            self._encoder_frozen = should_freeze
            status = "FROZEN" if should_freeze else "UNFROZEN"
            print(colored(f"[step {int(self.total_step.item())}] Encoder set to {status}", "cyan"))

    def set_encoder_gradients(self, flag=True):
        # CodecEnc은 vq_wav2vec이든 자체 Encoder든 공통 키 사용
        for p in self.model['CodecEnc'].parameters():
            p.requires_grad = flag
        # eval/train 전환으로 BN/Dropout 정지 및 활성화
        self.model['CodecEnc'].train(flag)

    # -------- Logging Helpers (NEW) --------
    def _zero(self):
        return torch.zeros((), device=self.device)

    def _gather_loss_dict(self, disc_losses, gen_losses, mi_losses=None):
        z = self._zero
        losses = {
            'disc_loss': disc_losses['disc_loss'],
            'fake_loss': disc_losses['fake_loss'],
            'real_loss': disc_losses['real_loss'],
            'gen_loss': gen_losses['gen_loss'],
            'vq_loss': gen_losses.get('vq_loss', z()),
            'mel_loss': gen_losses.get('mel_loss', z()),
            'fm_loss': gen_losses.get('fm_loss', z()),
            'adv_loss': gen_losses.get('adv_loss', z()),
            'spec_fm_loss': gen_losses.get('spec_fm_loss', z()),
            'xd_loss': gen_losses.get('xd_loss', z()),
            'f0_loss': gen_losses.get('f0_loss', z()),
            'f0_vq_loss': gen_losses.get('f0_vq_loss', z()),
            'vuv_loss': gen_losses.get('vuv_loss', z()),
            'fcpe_loss': gen_losses.get('fcpe_loss', z()),
        }
        if mi_losses is not None:
            losses.update({
                # 'mi_loss': mi_losses.get('mi_loss', z()),
                'club_loss': mi_losses.get('club_loss', z()),
                # 'mine_loss': mi_losses.get('mine_loss', z()),
                'gap_loss': mi_losses.get('gap_loss', z()),
                'mi_upper': mi_losses.get('mi_upper', z()),
                'mi_lower': mi_losses.get('mi_lower', z()),
                # 'delta': mi_losses.get('delta', z()),
            })

        # --- scaled (lambda 적용 후) ---
        lmb = self.cfg.train.lambdas
        def _lam(key, default=1.0):
            return lmb.get(key, default)

        def _scale(raw_key, lambda_key, already_scaled=False, default_lambda=1.0):
            if raw_key not in losses:
                return
            if already_scaled:
                # disc_loss 내부에서 이미 λ 곱했으면 별도 scaled 표시 생략 가능
                losses[f'{raw_key}_scaled'] = losses[raw_key]
            else:
                lam = _lam(lambda_key, default_lambda)
                losses[f'{raw_key}_scaled'] = losses[raw_key] * lam

        _scale('mel_loss', 'lambda_mel_loss')
        _scale('adv_loss', 'lambda_adv')
        _scale('fm_loss', 'lambda_feat_match_loss')
        _scale('spec_fm_loss', 'lambda_feat_match_loss')
        _scale('vq_loss', 'lambda_vq_loss')
        _scale('xd_loss', 'lambda_xd_loss')
        _scale('mi_upper', 'lambda_mi_loss')
        # disc_loss: 이미 compute_disc_loss에서 lambda_disc 곱했으면 already_scaled=True
        already = True  # 현재 compute_disc_loss에서 lambda_disc 적용했다고 가정
        _scale('disc_loss', 'lambda_disc', already_scaled=already)
        _scale('f0_loss', 'lambda_f0_recon_loss')
        _scale('f0_vq_loss', 'lambda_f0_vq_loss')
        _scale('vuv_loss', 'lambda_vuv_recon_loss')
        _scale('fcpe_loss', 'lambda_fcpe_loss', default_lambda=10.0)

        return losses

    def _log_losses(self, stage, disc_losses, gen_losses, mi_losses, output,
                    batch_size, gen_opt=None, disc_opt=None,
                    on_step=False, on_epoch=False):
        losses = self._gather_loss_dict(disc_losses, gen_losses, mi_losses)
        for k, v in losses.items():
            self.log(f'{stage}_loss/{k}', v,
                     on_step=on_step, on_epoch=on_epoch,
                     logger=True, sync_dist=True, batch_size=batch_size)

        # stats
        if stage == 'train':
            self.log('train_stats/total_step', self.total_step.float(),
                     logger=True, sync_dist=False)
            if gen_opt is not None:
                self.log('train_stats/lr_g', gen_opt.param_groups[0]['lr'],
                         logger=True, sync_dist=True)
            if disc_opt is not None:
                self.log('train_stats/lr_d', disc_opt.param_groups[0]['lr'],
                         logger=True, sync_dist=True)
        else:
            self.log('val_stats/total_step', self.total_step.float(),
                     on_epoch=True, logger=True, sync_dist=False)

        self.log(f'{stage}_stats/perplexity', output['perplexity'],
                 on_step=on_step, on_epoch=on_epoch,
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log(f'{stage}_stats/cluster_size', output['active_num'],
                 on_step=on_step, on_epoch=on_epoch,
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
    # -------- End Logging Helpers --------

    def configure_optimizers(self):
        from itertools import chain
        disc_params = self.model['discriminator'].parameters()
        if 'spec_discriminator' in self.model:
            disc_params = chain(disc_params, self.model['spec_discriminator'].parameters())
        # xd_loss를 줄이려면 speaker_encoder(특히 perceiver + quantizer + project)도 학습 대상이어야 함
        gen_params_list = [
            self.model['CodecEnc'].parameters(),
            self.model['generator'].parameters(),
            self.model['speaker_encoder'].parameters(),
            self.model['f0_encoder'].parameters(),
            self.model['f0_decoder'].parameters(),
            self.model['f0_quantizer'].parameters(),
        ]

        if self.cfg.model.f0_codec.finetune_f0_extractor:
            print(colored("Defining f0_extractor parameters to optimizer", "yellow"))
            gen_params_list.append(self.model['f0_extractor'].parameters())
        
        gen_params = chain(*gen_params_list)


        gen_opt = optim.AdamW(gen_params, **self.cfg.train.gen_optim_params)
        disc_opt = optim.AdamW(disc_params, **self.cfg.train.disc_optim_params)
        opt_list = [gen_opt, disc_opt]

        if self.cfg.model.get('use_mi', False):
            mi_opt = optim.AdamW(
                chain(self.model['mi_club'].parameters(), self.model['mi_mine'].parameters()),
                **self.cfg.train.mi_optim_params)
            opt_list.append(mi_opt)

        gen_sche = WarmupLR(gen_opt, **self.cfg.train.gen_schedule_params)
        disc_sche = WarmupLR(disc_opt, **self.cfg.train.disc_schedule_params)
        # print(f'Generator optim: {gen_opt}')
        # print(f'Discriminator optim: {disc_opt}')
        # print(f'MI optim: {mi_opt}')
        return opt_list, [gen_sche, disc_sche]

    def set_discriminator_gradients(self, flag=True):
        for p in self.model['discriminator'].parameters():
            p.requires_grad = flag
        
        if 'spec_discriminator' in self.model:
            for p in self.model['spec_discriminator'].parameters():
                p.requires_grad = flag
