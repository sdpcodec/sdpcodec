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
from vq.temporal_config import CodecDecoderSpeakerConditionConfig, CodecDecoderVocosConfig, CodecTemporalConfig, CodecDecoderTemporalConfig
from vq.ssl_codec_wrappers import (
    W2V2HiddenStateEncoder,
    W2VBert2HiddenStateEncoder,
    VQW2VCodecEncoderWrapper,
    W2V2CodecEncoderWrapper,
    W2VBert2CodecEncoderWrapper,
    S3TokenizerEncoder,
    S3TokenizerCodecEncoderWrapper,
)
from vq.speaker.speaker_encoder import (
    SpeakerEncoder,
    build_speaker_quantizer_kwargs,
    resolve_speaker_encoder_token_dim,
)
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
import jiwer
import time
from transformers import AutoProcessor, HubertForCTC
from requests.exceptions import ConnectionError as RequestsConnectionError
from urllib3.exceptions import ProtocolError as Urllib3ProtocolError
from torch.nn.parameter import UninitializedParameter


class BiCodecLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
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
        self._wer_runtime_status = "disabled_by_config" if not self.use_val_wer else "pending_first_real_validation"
        self._wer_status_detail = "cfg.train.use_val_wer=False" if not self.use_val_wer else "sanity validation skips WER; first non-sanity validation will decide."
        self._wer_sanity_logged = False
        self._wer_transform = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.SubstituteRegexes({r"[_\u2010-\u2015\u2212-]+": " "}),
            jiwer.SubstituteRegexes({r"[^\w\s\uAC00-\uD7A3]": ""}),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ])
        
        if self.use_val_utmos or self.use_val_wavlm:
            print(colored("Validation will include UTMOS/WavLM metrics", "yellow"))

        # freeze → unfreeze 스케줄 설정 (없으면 비활성)
        self.unfreeze_encoder_step = getattr(cfg.train, 'unfreeze_encoder_step', None)
        self._encoder_frozen = None
        # 스케줄 활성 여부: vq_wav2vec이고 freeze_vqw2v_encoder=False면 스케줄 무시
        enccfg = self.cfg.model.codec_encoder
        self._freeze_schedule_enabled = self.unfreeze_encoder_step is not None
        use_vqw2v = bool(getattr(enccfg, 'use_vqw2v', False))
        freeze_vqw2v = bool(getattr(enccfg, 'freeze_vqw2v_encoder', True))
        use_w2v2 = bool(getattr(enccfg, 'use_w2v2', False))
        freeze_w2v2 = bool(getattr(enccfg, 'freeze_w2v2_encoder', True))
        use_w2vbert2 = bool(getattr(enccfg, 'use_w2vbert2', False))
        freeze_w2vbert2 = bool(getattr(enccfg, 'freeze_w2vbert2_encoder', True))
        use_s3tokenizer = bool(getattr(enccfg, 'use_s3tokenizer', False))
        freeze_s3tokenizer = bool(getattr(enccfg, 'freeze_s3tokenizer_encoder', True))

        if use_vqw2v and not freeze_vqw2v:
            self._freeze_schedule_enabled = False
            print(colored("freeze_vqw2v_encoder=False -> disabling unfreeze_encoder_step; vqw2v CodecEnc stays UNFROZEN.", "yellow"))
        elif use_w2v2 and not freeze_w2v2:
            self._freeze_schedule_enabled = False
            print(colored("freeze_w2v2_encoder=False -> disabling unfreeze_encoder_step; w2v2 CodecEnc stays UNFROZEN.", "yellow"))
        elif use_w2vbert2 and not freeze_w2vbert2:
            self._freeze_schedule_enabled = False
            print(colored("freeze_w2vbert2_encoder=False -> disabling unfreeze_encoder_step; w2vbert2 CodecEnc stays UNFROZEN.", "yellow"))
        elif use_s3tokenizer and not freeze_s3tokenizer:
            self._freeze_schedule_enabled = False
            print(colored("freeze_s3tokenizer_encoder=False -> disabling unfreeze_encoder_step; s3tokenizer CodecEnc stays UNFROZEN.", "yellow"))
        else:
            print(colored(f"Encoder unfreeze step: {self.unfreeze_encoder_step}", "yellow"))
    
        # (선택) 동시 지정 안내
        if self.unfreeze_encoder_step is not None and freeze_vqw2v:
            print(colored("Both train.unfreeze_encoder_step and codec_encoder.freeze_vqw2v_encoder are set. Step-based schedule will override encoder freeze after unfreeze step.", "yellow"))
        if self.unfreeze_encoder_step is not None and freeze_w2v2:
            print(colored("Both train.unfreeze_encoder_step and codec_encoder.freeze_w2v2_encoder are set. Step-based schedule will override encoder freeze after unfreeze step.", "yellow"))
        if self.unfreeze_encoder_step is not None and freeze_w2vbert2:
            print(colored("Both train.unfreeze_encoder_step and codec_encoder.freeze_w2vbert2_encoder are set. Step-based schedule will override encoder freeze after unfreeze step.", "yellow"))
        if self.unfreeze_encoder_step is not None and freeze_s3tokenizer:
            print(colored("Both train.unfreeze_encoder_step and codec_encoder.freeze_s3tokenizer_encoder are set. Step-based schedule will override encoder freeze after unfreeze step.", "yellow"))

    def _hubert_hub_cache_root(self) -> str:
        hf_hub_cache = os.environ.get("HF_HUB_CACHE")
        if hf_hub_cache:
            return str(hf_hub_cache)
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        return str(os.path.join(hf_home, "hub"))

    def _has_local_hubert_cache(self) -> bool:
        repo_id = str(self.val_hubert_model_name)
        if "/" not in repo_id:
            return False
        cache_dir = os.path.join(self._hubert_hub_cache_root(), f"models--{repo_id.replace('/', '--')}")
        return os.path.exists(cache_dir)

    def _print_wer_status(self, prefix: str) -> None:
        sanity_steps = int(getattr(self.trainer, "num_sanity_val_steps", 0)) if self.trainer is not None else 0
        offline_env = (
            os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        )
        cache_state = "present" if self._has_local_hubert_cache() else "missing"
        print(colored(
            f"[WER] {prefix}: enabled={self.use_val_wer}, status={self._wer_runtime_status}, cache={cache_state}, offline={offline_env}, sanity_steps={sanity_steps}, model={self.val_hubert_model_name}",
            "cyan",
        ))
        if self._wer_status_detail:
            print(colored(f"[WER] detail: {self._wer_status_detail}", "cyan"))

    def on_train_start(self):
        self._print_wer_status("train_start")

    def construct_model(self):
        spkcfg = self.cfg.model.speaker_encoder
        speaker_token_dim = resolve_speaker_encoder_token_dim(spkcfg)
        
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
        use_vqw2v = bool(getattr(enccfg, 'use_vqw2v', False))
        use_w2v2 = bool(getattr(enccfg, 'use_w2v2', False))
        use_w2vbert2 = bool(getattr(enccfg, 'use_w2vbert2', False))
        use_s3tokenizer = bool(getattr(enccfg, 'use_s3tokenizer', False))
        if (use_vqw2v + use_w2v2 + use_w2vbert2 + use_s3tokenizer) > 1:
            raise ValueError(
                "Only one of codec_encoder.use_vqw2v / use_w2v2 / use_w2vbert2 / use_s3tokenizer can be True."
            )

        if use_vqw2v:
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
        elif use_w2v2:
            # HuggingFace Wav2Vec2 feature extractor (conv frontend)
            from transformers import Wav2Vec2Model
            model_name = getattr(enccfg, 'w2v2_model_name', 'facebook/wav2vec2-large')
            use_transformer = bool(getattr(enccfg, 'w2v2_use_transformer', True))
            w2v2 = Wav2Vec2Model.from_pretrained(model_name)
            if use_transformer:
                layer_indices = getattr(enccfg, 'w2v2_hidden_layers', [11, 14, 16])
                mode = getattr(enccfg, 'w2v2_hidden_mode', 'avg')
                encoder = W2V2HiddenStateEncoder(w2v2, layer_indices=layer_indices, mode=mode)
            else:
                encoder = w2v2.feature_extractor

            if bool(getattr(enccfg, 'freeze_w2v2_encoder', True)):
                print(colored(f"Freezing Wav2Vec2 ({'transformer' if use_transformer else 'feature_extractor'}) ({model_name})", "yellow"))
                for p in w2v2.parameters():
                    p.requires_grad = False
                w2v2.eval()
            else:
                for p in w2v2.parameters():
                    p.requires_grad = True

        elif use_w2vbert2:
            # AutoProcessor loads tokenizer which facebook/w2v-bert-2.0 lacks; use AutoFeatureExtractor
            from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
            model_name = getattr(enccfg, 'w2vbert2_model_name', 'facebook/w2v-bert-2.0')
            processor = AutoFeatureExtractor.from_pretrained(model_name)
            w2vbert2 = Wav2Vec2BertModel.from_pretrained(model_name)
            layer_indices = getattr(enccfg, 'w2vbert2_hidden_layers', (11, 18, 23))
            mode = getattr(enccfg, 'w2vbert2_hidden_mode', 'avg')
            sampling_rate = getattr(enccfg, 'w2vbert2_sampling_rate', 16000)
            normalize = getattr(enccfg, 'w2vbert2_hidden_normalize', True)
            encoder = W2VBert2HiddenStateEncoder(
                w2vbert2, processor,
                layer_indices=layer_indices,
                mode=mode,
                sampling_rate=sampling_rate,
                normalize=normalize,
            )
            if bool(getattr(enccfg, 'freeze_w2vbert2_encoder', True)):
                print(colored(f"Freezing Wav2Vec2-BERT 2.0 ({model_name})", "yellow"))
                for p in w2vbert2.parameters():
                    p.requires_grad = False
                w2vbert2.eval()
            else:
                for p in w2vbert2.parameters():
                    p.requires_grad = True

        elif use_s3tokenizer:
            model_name = getattr(enccfg, 's3tokenizer_model_name', 'speech_tokenizer_v1')
            sampling_rate = getattr(enccfg, 's3tokenizer_sampling_rate', 16000)
            encoder = S3TokenizerEncoder(
                model_name=model_name,
                sampling_rate=sampling_rate,
                use_continuous=True,
            )
            if bool(getattr(enccfg, 'freeze_s3tokenizer_encoder', True)):
                print(colored(f"Freezing S3Tokenizer ({model_name})", "yellow"))
                for p in encoder.parameters():
                    p.requires_grad = False
                encoder.eval()
            else:
                for p in encoder.parameters():
                    p.requires_grad = True

        else:
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

        if enccfg.get('up_ratios', None) is not None and use_vqw2v:
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
                encoder_force_fp32=enccfg.get('frozen_upstream_force_fp32', False),
                snake_lite_taylor_degree=enccfg.get('snake_lite_taylor_degree', 8),
            )
        if use_w2v2:
            # Always use wrapper: projects w2v2 output to out_channels.
            # When up_ratios=None: no downsampling (w2v2 hop 320 -> decoder up [8,5,4,3] -> 24kHz).
            # When up_ratios is set: adds RNN + EncoderBlocks for additional downsampling.
            print(colored(f"Using Wav2Vec2 Codec Encoder Wrapper (up_ratios={enccfg.get('up_ratios')})", "red"))
            w2v2_ngf = getattr(enccfg, 'w2v2_ngf', None)
            if w2v2_ngf is None:
                # auto infer: transformer hidden size vs conv channels
                w2v2_ngf = 1024 if bool(getattr(enccfg, 'w2v2_use_transformer', True)) else 512
            encoder = W2V2CodecEncoderWrapper(
                temporal=CodecTemporalConfig.from_encoder_cfg(enccfg),
                encoder=encoder,
                up_ratios=enccfg.get('up_ratios'),
                dilations=enccfg.get('dilations', [1, 3, 9]),
                ngf=int(w2v2_ngf),
                out_channels=enccfg.out_channels,
                activation_type=enccfg.get('activation_type', 'SnakeBeta'),
                leaky_relu_params=enccfg.get('leaky_relu_params', None),
                encoder_force_fp32=enccfg.get('frozen_upstream_force_fp32', False),
                snake_lite_taylor_degree=enccfg.get('snake_lite_taylor_degree', 8),
            )
        if use_w2vbert2:
            print(colored(f"Using Wav2Vec2-BERT 2.0 Codec Encoder Wrapper (up_ratios={enccfg.get('up_ratios')})", "red"))
            w2vbert2_ngf = getattr(enccfg, 'w2vbert2_ngf', None) or 1024
            encoder = W2VBert2CodecEncoderWrapper(
                temporal=CodecTemporalConfig.from_encoder_cfg(enccfg),
                encoder=encoder,
                up_ratios=enccfg.get('up_ratios'),
                dilations=enccfg.get('dilations', [1, 3, 9]),
                ngf=int(w2vbert2_ngf),
                out_channels=enccfg.out_channels,
                activation_type=enccfg.get('activation_type', 'SnakeBeta'),
                leaky_relu_params=enccfg.get('leaky_relu_params', None),
                encoder_force_fp32=enccfg.get('frozen_upstream_force_fp32', False),
                snake_lite_taylor_degree=enccfg.get('snake_lite_taylor_degree', 8),
            )
        if use_s3tokenizer:
            print(colored(f"Using S3Tokenizer Codec Encoder Wrapper (up_ratios={enccfg.get('up_ratios')})", "red"))
            s3_ngf = getattr(enccfg, 's3tokenizer_ngf', 1280)
            encoder = S3TokenizerCodecEncoderWrapper(
                temporal=CodecTemporalConfig.from_encoder_cfg(enccfg),
                encoder=encoder,
                up_ratios=enccfg.get('up_ratios'),
                dilations=enccfg.get('dilations', [1, 3, 9]),
                ngf=int(s3_ngf),
                out_channels=enccfg.out_channels,
                activation_type=enccfg.get('activation_type', 'SnakeBeta'),
                leaky_relu_params=enccfg.get('leaky_relu_params', None),
                encoder_force_fp32=enccfg.get('frozen_upstream_force_fp32', False),
                snake_lite_taylor_degree=enccfg.get('snake_lite_taylor_degree', 8),
            )

        deccfg = self.cfg.model.codec_decoder
        decoder_spk_cond_cfg = CodecDecoderSpeakerConditionConfig.from_decoder_cfg(deccfg)
        decoder_vocos_cfg = CodecDecoderVocosConfig.from_decoder_cfg(deccfg)
        decoder_spk_cond_node = deccfg.get('spk_cond', None)
        decoder_mhca_use_sdpa = (
            decoder_spk_cond_node.get('use_sdpa', deccfg.get('mhca_use_sdpa', None))
            if decoder_spk_cond_node is not None else deccfg.get('mhca_use_sdpa', None)
        )
        use_ssl_encoder = bool(use_vqw2v or use_w2v2 or use_w2vbert2 or use_s3tokenizer)
        freeze_ssl_encoder = bool(
            (use_vqw2v and bool(getattr(enccfg, 'freeze_vqw2v_encoder', False))) or
            (use_w2v2 and bool(getattr(enccfg, 'freeze_w2v2_encoder', True))) or
            (use_w2vbert2 and bool(getattr(enccfg, 'freeze_w2vbert2_encoder', True))) or
            (use_s3tokenizer and bool(getattr(enccfg, 'freeze_s3tokenizer_encoder', True)))
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
                    use_vqw2v_encoder=use_ssl_encoder,
                    freeze_vqw2v_encoder=freeze_ssl_encoder,
                    use_vqw2v_embed=deccfg.get('use_vqw2v_embed', True),

                    speaker_condition=deccfg.speaker_condition,
                    condition_dim=deccfg.condition_dim,
                    activation_type=deccfg.get('activation_type', 'SnakeBeta'),
                    leaky_relu_params=deccfg.get('leaky_relu_params', None),
                    snake_logscale=deccfg.get('snake_logscale', True),
                    snake_lite_taylor_degree=deccfg.get('snake_lite_taylor_degree', 8),
                    simvq_linear_layer_type=deccfg.get('simvq_linear_layer_type', 'linear'),
                    ema_decay=deccfg.get('ema_decay', 0.0),

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
                    vocos_sampling_rate=self.cfg.preprocess.audio.sr,
                    vocos_n_fft=decoder_vocos_cfg.n_fft,
                    vocos_hop_size=decoder_vocos_cfg.hop_size,
                    vocos_win_size=decoder_vocos_cfg.win_size,
                    vocos_dim=decoder_vocos_cfg.dim,
                    vocos_intermediate_dim=decoder_vocos_cfg.intermediate_dim,
                    vocos_num_layers=decoder_vocos_cfg.num_layers,
                    vocos_kernel_size=decoder_vocos_cfg.kernel_size,
                    vocos_padding=decoder_vocos_cfg.padding,
                    vocos_f0_stage_dims=None,
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
                })
        
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
            
            # Use default settings (dynamic=False) for better performance
            # Only Generator is skipped due to SnakeBeta incompatibility
            try:
                if hasattr(self, 'model'):
                    # Speaker Encoder - fixed length after Perceiver resampling
                    if 'speaker_encoder' in self.model:
                        self.model['speaker_encoder'] = torch.compile(
                            self.model['speaker_encoder'], mode=compile_mode
                        )
                        print(colored("  ✓ Speaker encoder compiled", "green"))
                    
                    # CodecEnc - fixed downsampling ratio
                    enc_cfg = self.cfg.model.codec_encoder
                    if 'CodecEnc' in self.model and (getattr(enc_cfg, 'use_vqw2v', False) or getattr(enc_cfg, 'use_w2v2', False) or getattr(enc_cfg, 'use_w2vbert2', False) or getattr(enc_cfg, 'use_s3tokenizer', False)):
                        self.model['CodecEnc'] = torch.compile(
                            self.model['CodecEnc'], mode=compile_mode
                        )
                        print(colored("  ✓ CodecEnc compiled", "green"))
                    
                    # Generator (decoder) - SKIP: SnakeBeta activation causes compile errors
                    
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
                    
                    print(colored("✓ torch.compile applied successfully", "green"))
                    print(colored("⚠ Generator skipped (SnakeBeta activation incompatible)", "yellow"))
                    print(colored("⚠ First epoch will be slower due to compilation (JIT tracing)", "yellow"))
            except Exception as e:
                print(colored(f"⚠ torch.compile failed: {e}", "red"))
                print(colored("  Continuing without compilation...", "yellow"))

    def load_state_dict(self, state_dict, strict=True):
        # Reconcile checkpoint keys with current module structure
        # - Drop keys for validation-only modules
        # - Fix _orig_mod prefix mismatches introduced by torch.compile
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

    def pad_for_w2v2(self, source, J=320, RF=400):
        """
        Wav2Vec2 (HF) conv feature extractor padding.
        Default values match Wav2Vec2Model conv frontend:
          - effective stride J = 320 (5 * 2^6)
          - receptive field RF = 400
        """
        B, L = source.shape
        pad_left = RF - J  # 80
        rem = L % J
        pad_right = (J - rem) % J
        return F.pad(source, (pad_left, pad_right)), pad_left, pad_right

    def _w2v2_normalize(self, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """
        Torch implementation of Wav2Vec2FeatureExtractor zero-mean/unit-variance normalization.
        x: (B, T)
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + eps)


    # @autocast(device_type='cuda', enabled=False)
    def forward(self, batch):
        wav = batch['wav']  # 16 kHz branch (required by VQ-Wav2Vec)
        wav_24k = batch.get('wav_24k')
        sr = self.cfg.preprocess.audio.sr
        if sr == 24000:
            if not isinstance(wav_24k, torch.Tensor):
                raise RuntimeError("Expected 24 kHz waveform `wav_24k` in batch when preprocess.audio.sr=24000")
            target_wav = wav_24k
        else:
            target_wav = wav
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

        enc_cfg = self.cfg.model.codec_encoder
        use_vqw2v = bool(getattr(enc_cfg, 'use_vqw2v', False))
        use_w2v2 = bool(getattr(enc_cfg, 'use_w2v2', False))
        use_w2vbert2 = bool(getattr(enc_cfg, 'use_w2vbert2', False))
        use_s3tokenizer = bool(getattr(enc_cfg, 'use_s3tokenizer', False))
        if use_vqw2v:
            wav_ = self.pad_for_wav2vec(wav)[0]
            with torch.set_grad_enabled(not enc_no_grad):
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
        elif use_s3tokenizer:
            wav_n = self._w2v2_normalize(wav)
            wav_ = self.pad_for_w2v2(wav_n)[0]
            with torch.set_grad_enabled(not enc_no_grad):
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
        elif use_w2v2:
            wav_n = self._w2v2_normalize(wav)
            wav_ = self.pad_for_w2v2(wav_n)[0]
            with torch.set_grad_enabled(not enc_no_grad):
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
        elif use_w2vbert2:
            wav_n = self._w2v2_normalize(wav)
            wav_ = self.pad_for_w2v2(wav_n)[0]
            with torch.set_grad_enabled(not enc_no_grad):
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
        else:
            with torch.set_grad_enabled(not enc_no_grad):
                # Non-SSL path should follow training sample rate (e.g., 24 kHz)
                enc_input = wav_24k if int(self.cfg.preprocess.audio.sr) == 24000 else wav
                vq_emb = self.model['CodecEnc'](enc_input.unsqueeze(1), spk_cond=spk_cond_vec)

        vq_post_emb, vq_code, vq_loss, perplexity, active_num = self.model['generator'](vq_emb, total_step=self.total_step, vq=True)
        vq_post_emb_ = vq_post_emb + spk_cond_vec.unsqueeze(-1)
        y_ = self.model['generator'](vq_post_emb_, vq=False, spk_cond=spk_cond) # [B, 1, T]
        y = target_wav.unsqueeze(1)
        # Ensure waveform lengths are aligned (match TriX behavior of aligned comparison)
        if y_.size(-1) != y.size(-1):
            min_len = min(y_.size(-1), y.size(-1))
            y_ = y_[..., :min_len]
            y = y[..., :min_len]
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
            'active_num': active_num
        }
        return output
    
    def tokenizer(self, batch):
        spk_type = getattr(self.cfg.model.speaker_encoder, 'speaker_encoder_type', 'ecapa_tdnn')
        if spk_type == 'wavlm':
            global_tokens = self.model['speaker_encoder'].tokenize_wav(batch['ref_wav'])
        else:
            global_tokens = self.model['speaker_encoder'].tokenize(batch['mel'].transpose(1, 2))
        
        enc_cfg = self.cfg.model.codec_encoder
        use_vqw2v = bool(getattr(enc_cfg, 'use_vqw2v', False))
        use_w2v2 = bool(getattr(enc_cfg, 'use_w2v2', False))
        use_w2vbert2 = bool(getattr(enc_cfg, 'use_w2vbert2', False))
        use_s3tokenizer = bool(getattr(enc_cfg, 'use_s3tokenizer', False))
        if use_vqw2v:
            wav_ = self.pad_for_wav2vec(batch['wav'])[0]
            with torch.no_grad():
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
        elif use_s3tokenizer:
            wav_n = self._w2v2_normalize(batch['wav'])
            wav_ = self.pad_for_w2v2(wav_n)[0]
            with torch.no_grad():
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
        elif use_w2v2:
            wav_n = self._w2v2_normalize(batch['wav'])
            wav_ = self.pad_for_w2v2(wav_n)[0]
            with torch.no_grad():
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
        elif use_w2vbert2:
            wav_n = self._w2v2_normalize(batch['wav'])
            wav_ = self.pad_for_w2v2(wav_n)[0]
            with torch.no_grad():
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
        else:
            raise NotImplementedError("Tokenizer is only available when using VQ-Wav2Vec / Wav2Vec2 / W2VBert2 / S3Tokenizer encoder.")
        _, vq_code, _, _, _= self.model['generator'](vq_emb, total_step=None, vq=True)

        return {
            'global_tokens': global_tokens,
            'semantic_tokens': vq_code
        }

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
        vq_loss, vq_code = output['vq_loss'], output['vq_code']
        x_vector, d_vector = output['x_vector'], output['d_vector']
        gen_loss = torch.tensor(0.0, device=self.device)
        self.set_discriminator_gradients(False)
        loss_dict = {}
        cfg = self.cfg.train
        
        if cfg.use_mel_loss:
            # Safety: ensure lengths are equal before mel loss (redundant with forward trim)
            if y_.size(-1) != y.size(-1):
                min_len = min(y_.size(-1), y.size(-1))
                y_ = y_[..., :min_len]
                y = y[..., :min_len]
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

        self.set_discriminator_gradients(True)
        loss_dict['gen_loss'] = gen_loss
        return loss_dict
    
    def training_step(self, batch, batch_idx):
        # 스텝별 encoder freeze/unfreeze 반영
        self._maybe_update_encoder_freeze()
        output = self(batch)

        opts = self.optimizers()
        gen_opt, disc_opt = opts
        gen_sche, disc_sche = self.lr_schedulers()
        
        
        accum = int(self.cfg.train.gradient_accumulation_steps)
        is_update_step = ((batch_idx + 1) % accum == 0)

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
                disc_losses=disc_losses, gen_losses=gen_losses, mi_losses=None,
                output=output, batch_size=self.cfg.dataset.train.batch_size,
                gen_opt=gen_opt, disc_opt=disc_opt,
                on_step=True, on_epoch=False
            )

    def _load_hubert_with_retry(self, max_retries=5, base_delay=2):
        """Load HuBERT model with retry logic for network errors.

        If the model is already cached locally, fall back to local-only loading
        when HF Hub rate limits or network errors occur.
        """
        print(colored("Loading HuBERT model for validation WER...", "yellow"))

        def _looks_like_rate_limit(err: Exception) -> bool:
            text = f"{type(err).__name__}: {err}"
            return "429" in text or "Too Many Requests" in text

        def _looks_like_network_error(err: Exception) -> bool:
            text = f"{type(err).__name__}: {err}"
            needles = (
                "ReadTimeout",
                "Read timed out",
                "ConnectionError",
                "ConnectTimeout",
                "Temporary failure in name resolution",
                "MaxRetryError",
                "HTTPSConnectionPool",
            )
            return any(needle in text for needle in needles)

        def _load_hubert(local_files_only: bool):
            self.hubert_processor = AutoProcessor.from_pretrained(
                self.val_hubert_model_name,
                local_files_only=local_files_only,
            )
            self.hubert_model = HubertForCTC.from_pretrained(
                self.val_hubert_model_name,
                local_files_only=local_files_only,
            ).to(self.device)
            self.hubert_model.eval()

        offline_env = (
            os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        )
        if offline_env and self._has_local_hubert_cache():
            _load_hubert(local_files_only=True)
            self._wer_runtime_status = "enabled_local_cache"
            self._wer_status_detail = "HuBERT loaded from local cache because offline mode is enabled."
            self._print_wer_status("validation_wer_ready")
            return

        for attempt in range(max_retries):
            try:
                _load_hubert(local_files_only=False)
                self._wer_runtime_status = "enabled_online_or_cache_validated"
                self._wer_status_detail = "HuBERT loaded with online hub access enabled."
                self._print_wer_status("validation_wer_ready")
                return
            except (RequestsConnectionError, ConnectionError, Urllib3ProtocolError, OSError, Exception) as e:
                if (_looks_like_rate_limit(e) or _looks_like_network_error(e)) and self._has_local_hubert_cache():
                    print(colored(
                        "HF Hub request failed or was rate limited; loading HuBERT from local cache.",
                        "yellow"
                    ))
                    _load_hubert(local_files_only=True)
                    self._wer_runtime_status = "enabled_local_cache_after_hub_failure"
                    self._wer_status_detail = f"HuBERT loaded from local cache after hub failure: {type(e).__name__}: {e}"
                    self._print_wer_status("validation_wer_ready")
                    return
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(colored(
                        f"Failed to load HuBERT model (attempt {attempt + 1}/{max_retries}): {str(e)}",
                        "yellow"
                    ))
                    print(colored(f"Retrying in {delay} seconds...", "yellow"))
                    time.sleep(delay)
                else:
                    print(colored(
                        f"Failed to load HuBERT model after {max_retries} attempts: {str(e)}",
                        "red"
                    ))
                    print(colored(
                        "Warning: Validation WER metrics will be skipped due to HuBERT loading failure.",
                        "yellow"
                    ))
                    # Disable WER validation to prevent future attempts
                    self.use_val_wer = False
                    self._wer_runtime_status = "disabled_after_hubert_load_failure"
                    self._wer_status_detail = f"WER disabled because HuBERT could not be loaded: {type(e).__name__}: {e}"
                    self._print_wer_status("validation_wer_disabled")
                    return

    def on_validation_epoch_start(self):
        self.stoi.reset()
        self.pesq_wb.reset()
        # self.pesq_nb.reset()
        self.si_snr.reset()
        self.si_sdr.reset()
        
        is_sanity_check = getattr(self.trainer, 'sanity_checking', False) if hasattr(self.trainer, 'sanity_checking') else False

        # Initialize UTMOS/WavLM accumulators
        self.val_utmos_same_scores = []
        self.val_utmos_vc_scores = []
        self.val_wavlm_same_scores = []
        self.val_wavlm_vc_scores = []
        if self.use_val_wer:
            self.val_wer_same_scores = []
            self.val_wer_vc_scores = []
        
        if is_sanity_check:
            if self.use_val_wer and not self._wer_sanity_logged:
                self._wer_sanity_logged = True
                self._wer_runtime_status = "pending_first_real_validation"
                self._wer_status_detail = "Sanity validation skips WER initialization. First non-sanity validation will decide."
                self._print_wer_status("sanity_validation_skip")
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
                print(colored("⚠ WavLM checkpoint not found, using random weights", "yellow"))

        if self.use_val_wer and self.hubert_model is None:
            self._wer_runtime_status = "loading_hubert_for_validation"
            self._wer_status_detail = "Starting HuBERT load for the first non-sanity validation."
            self._print_wer_status("validation_wer_init")
            self._load_hubert_with_retry()

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


    def validation_step(self, batch, batch_idx):
        # Same-speaker reconstruction
        output = self(batch)

        disc_losses = self.compute_disc_loss(batch, output)
        gen_losses = self.compute_gen_loss(batch, output)
        mi_losses = None

        self.si_snr.update(output['gt_wav'].squeeze(1), output['gen_wav'].squeeze(1))
        self.si_sdr.update(output['gt_wav'].squeeze(1), output['gen_wav'].squeeze(1))
        self.stoi.update(output['gt_wav'].squeeze(1), output['gen_wav'].squeeze(1))
        self.pesq_wb.update(output['gt_wav'].squeeze(1), output['gen_wav'].squeeze(1))

        self._log_losses(
            stage='val',
            disc_losses=disc_losses, gen_losses=gen_losses, mi_losses=mi_losses, output=output,
            batch_size=self.cfg.dataset.val.batch_size,
            on_step=False, on_epoch=True)
        
        # Voice conversion (different speaker) - only for UTMOS/WavLM metrics, no loss logging
        output_vc = None
        if 'ref_wav_vc' in batch and (self.use_val_utmos or self.use_val_wavlm):
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

    def test_step(self, batch, batch_idx):
        # remove weight norm for modules in self.model
        sr = self.cfg.preprocess.audio.sr
        wav_24k = batch.get('wav_24k')
        if sr == 24000:
            if not isinstance(wav_24k, torch.Tensor):
                raise RuntimeError("Expected 24 kHz waveform `wav_24k` in batch when preprocess.audio.sr=24000")
            batch_audio = wav_24k
        else:
            batch_audio = batch['wav']
        batch_size = batch_audio.size(0)
        target_type = self.cfg.voice_conversion
        assert target_type in ['rec', 'vc', 'same'], f"Invalid target_type: {target_type}"

        indices_to_process = []
        for i in range(batch_size):
            source_filename = batch['fid'][i]
            if target_type in ['rec', 'vc']:
                target_filename = batch['target_id'][i]
            elif target_type == 'same':
                target_filename = source_filename

            gen_path = f"{self.cwd}/{source_filename}-{target_filename}_{target_type}.wav"
            gt_path = f"{self.cwd}/{source_filename}-{target_filename}_gt.wav"
            ref_path = f"{self.cwd}/{source_filename}-{target_filename}_ref.wav"

            if os.path.exists(gen_path) and os.path.exists(gt_path) and os.path.exists(ref_path):
                print(colored(f"[Skip] All files exist for {source_filename}-{target_filename}", "yellow"))
                continue

            indices_to_process.append(i)

        if len(indices_to_process) == 0:
            print(colored(f"[Skip] Batch {batch_idx} - all files already exist", "green"))
            return

        output = self(batch)

        for i in indices_to_process:
            source_filename = batch['fid'][i]
            if target_type in ['rec', 'vc']:
                target_filename = batch['target_id'][i]
            elif target_type == 'same':
                target_filename = source_filename

            # _ref.wav는 get_ref_clip 이후(ref_wav) 대신 raw ref_src를 저장해야 함 (test only)
            if isinstance(batch.get('ref_src', None), torch.Tensor):
                ref_i = batch['ref_src'][i]
            else:
                raise ValueError("ref_src is not found in batch")
            gt_i, gen_i = batch_audio[i], output['gen_wav'][i]
            if gt_i.dim() == 1:
                gt_i = gt_i.unsqueeze(0)
            if ref_i.dim() == 1:
                ref_i = ref_i.unsqueeze(0)
            if gen_i.dim() == 1:
                gen_i = gen_i.unsqueeze(0)

            gen_path = f"{self.cwd}/{source_filename}-{target_filename}_{target_type}.wav"
            gt_path = f"{self.cwd}/{source_filename}-{target_filename}_gt.wav"
            ref_path = f"{self.cwd}/{source_filename}-{target_filename}_ref.wav"

            torchaudio.save(gen_path, gen_i.float().detach().cpu(), self.cfg.preprocess.audio.sr)
            torchaudio.save(gt_path, gt_i.float().detach().cpu(), self.cfg.preprocess.audio.sr)
            torchaudio.save(ref_path, ref_i.float().detach().cpu(), 16000)
            # print(colored(f"[Saved] {source_filename}-{target_filename}_{target_type}.wav", "green"))

    def on_test_start(self):
        self._remove_all_weight_norms()
        self.cwd = os.path.join(os.getcwd(), os.path.splitext(self.cfg.ckpt)[0], str(self.cfg.dataset.min_ref_seconds))
        if str(self.cfg.dataset.get("name", "")).lower() == "vctk":
            self.cwd = os.path.join(self.cwd, "vctk")
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
        }

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
        # disc_loss: 이미 compute_disc_loss에서 lambda_disc 곱했으면 already_scaled=True
        already = True  # 현재 compute_disc_loss에서 lambda_disc 적용했다고 가정
        _scale('disc_loss', 'lambda_disc', already_scaled=already)

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
        gen_params = chain(
            self.model['CodecEnc'].parameters(),
            self.model['generator'].parameters(),
            self.model['speaker_encoder'].parameters())

        gen_opt = optim.AdamW(gen_params, **self.cfg.train.gen_optim_params)
        disc_opt = optim.AdamW(disc_params, **self.cfg.train.disc_optim_params)
        opt_list = [gen_opt, disc_opt]

        gen_sche = WarmupLR(gen_opt, **self.cfg.train.gen_schedule_params)
        disc_sche = WarmupLR(disc_opt, **self.cfg.train.disc_schedule_params)
        sche_list = [gen_sche, disc_sche]

        # print(f'Generator optim: {gen_opt}')
        # print(f'Discriminator optim: {disc_opt}')
        return opt_list, sche_list

    def set_discriminator_gradients(self, flag=True):
        for p in self.model['discriminator'].parameters():
            p.requires_grad = flag
        
        if 'spec_discriminator' in self.model:
            for p in self.model['spec_discriminator'].parameters():
                p.requires_grad = flag
