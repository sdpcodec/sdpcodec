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
from torch.nn.parameter import UninitializedParameter

class BigCodecLightningModule(pl.LightningModule):
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
        # encoder freeze schedule (BiCodec logic port, speaker/MI excluded)
        self.unfreeze_encoder_step = getattr(cfg.train, 'unfreeze_encoder_step', None)
        self._encoder_frozen = None
        enccfg = self.cfg.model.codec_encoder
        self._freeze_schedule_enabled = self.unfreeze_encoder_step is not None
        if enccfg.use_vqw2v and not enccfg.freeze_vqw2v_encoder:
            # if not frozen initially, schedule meaningless
            self._freeze_schedule_enabled = False
            print(colored("freeze_vqw2v_encoder=False -> disabling unfreeze_encoder_step; vqw2v CodecEnc stays UNFROZEN.", "yellow"))
        else:
            if self.unfreeze_encoder_step is not None:
                print(colored(f"Encoder unfreeze step: {self.unfreeze_encoder_step}", "yellow"))
        if self.unfreeze_encoder_step is not None and getattr(self.cfg.model.codec_encoder, 'freeze_vqw2v_encoder', False):
            print(colored("Both train.unfreeze_encoder_step and codec_encoder.freeze_vqw2v_encoder are set. Step-based schedule will override encoder freeze after unfreeze step.", "yellow"))

    def construct_model(self):
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
                        activation_type=enccfg.get('activation_type', 'SnakeBeta'),
                        leaky_relu_params=enccfg.get('leaky_relu_params', None),
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

                    speaker_condition=deccfg.speaker_condition,
                    condition_dim=deccfg.condition_dim,
                    activation_type=deccfg.get('activation_type', 'SnakeBeta'),
                    leaky_relu_params=deccfg.get('leaky_relu_params', None),
                    snake_logscale=deccfg.get('snake_logscale', True),
                    snake_lite_taylor_degree=deccfg.get('snake_lite_taylor_degree', 8),
                    simvq_linear_layer_type=deccfg.get('simvq_linear_layer_type', 'linear'),
                    ema_decay=deccfg.get('ema_decay', 0.0),
                    use_mhca=decoder_spk_cond_cfg.use_mhca,
                    spk_cond_use_concat=decoder_spk_cond_cfg.use_concat,
                    mhca_num_heads=decoder_spk_cond_cfg.num_heads,
                    mhca_dropout=decoder_spk_cond_cfg.dropout,
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
            try:
                if hasattr(self, 'model'):
                    # CodecEnc - fixed downsampling ratio
                    if 'CodecEnc' in self.model:
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


    def forward(self, batch):
        wav = batch['wav']

        # encoder가 freeze 상태면 no_grad로 forward
        enc_no_grad = self.training and self._is_encoder_frozen()

        if self.cfg.model.codec_encoder.use_vqw2v:
            wav_ = self.pad_for_wav2vec(wav)[0]
            with torch.set_grad_enabled(not enc_no_grad):
                vq_emb = self.model['CodecEnc'](wav_)  # [B, C, T]
        else:
            with torch.set_grad_enabled(not enc_no_grad):
                vq_emb = self.model['CodecEnc'](wav.unsqueeze(1))
                
        vq_post_emb, vq_code, vq_loss, perplexity, active_num = self.model['generator'](vq_emb, total_step=self.total_step, vq=True)
        y_ = self.model['generator'](vq_post_emb, vq=False) # [B, 1, T]
        y = wav.unsqueeze(1)
        output = {
            'gt_wav': y,
            'gen_wav': y_,
            'vq_loss': vq_loss,
            'vq_code': vq_code,
            'perplexity': perplexity,
            'active_num': active_num
        }
        return output
    
    # @torch.inference_mode()
    # def inference(self, wav):
    #     vq_emb = self.model['CodecEnc'](wav.unsqueeze(1))
    #     vq_post_emb, vq_code, vq_loss, perplexity, active_num = self.model['generator'](vq_emb, vq=True)
    #     y_ = self.model['generator'](vq_post_emb, vq=False).squeeze(1) # [B, T]
    #     return y_

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
        gen_loss = torch.tensor(0.0, device=self.device)
        self.set_discriminator_gradients(False)
        output = {}
        cfg = self.cfg.train
        
        if cfg.use_mel_loss:
            mel_loss = self.criteria['mel_loss'](y_.squeeze(1), y.squeeze(1))
            gen_loss += mel_loss * cfg.lambdas.lambda_mel_loss
            output['mel_loss'] = mel_loss
        
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
        output['adv_loss'] = adv_loss

        # fm loss
        if cfg.use_feat_match_loss:
            fm_loss = torch.tensor(0.0, device=self.device)
            with torch.no_grad():
                p = cached_real if cached_real is not None else self.model['discriminator'](y)
            for i in range(len(p_)):
                for j in range(len(p_[i]) - 1):
                    fm_loss += self.criteria['fm_loss'](p_[i][j], p[i][j].detach())
            gen_loss += fm_loss * cfg.lambdas.lambda_feat_match_loss
            output['fm_loss'] = fm_loss
            if 'spec_discriminator' in self.model:
                spec_fm_loss = torch.tensor(0.0, device=self.device)
                with torch.no_grad():
                    sd_p = cached_spec_real if cached_spec_real is not None else self.model['spec_discriminator'](y)
                for i in range(len(sd_p_)):
                    for j in range(len(sd_p_[i]) - 1):
                        spec_fm_loss += self.criteria['fm_loss'](sd_p_[i][j], sd_p[i][j].detach())
                gen_loss += spec_fm_loss * cfg.lambdas.lambda_feat_match_loss
                output['spec_fm_loss'] = spec_fm_loss

        # vq
        if vq_loss is not None:
            if isinstance(vq_loss, list):
                vq_loss = sum(vq_loss)
            if isinstance(vq_loss, torch.Tensor) and vq_loss.dim() > 0:
                vq_loss = vq_loss.mean()
            output['vq_loss'] = vq_loss
            if self.cfg.model.codec_decoder.quantizer_type == "simvq":
                vq_loss = self.cfg.train.lambdas.lambda_vq_loss * vq_loss
            gen_loss = gen_loss + vq_loss

        output['xd_loss'] = torch.tensor(0., device=self.device)

        self.set_discriminator_gradients(True)
        output['gen_loss'] = gen_loss
        return output
    

    def training_step(self, batch, batch_idx):
        # apply encoder freeze/unfreeze schedule
        self._maybe_update_encoder_freeze()
        output = self(batch)

        gen_opt, disc_opt = self.optimizers()
        gen_sche, disc_sche = self.lr_schedulers()

        accum = int(self.cfg.train.gradient_accumulation_steps)
        is_update_step = ((batch_idx + 1) % accum == 0)

        # 1) Discriminator
        self.toggle_optimizer(disc_opt)
        disc_losses = self.compute_disc_loss(batch, output)
        disc_loss = disc_losses['disc_loss'] / accum
        try:
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
                disc_losses=disc_losses, gen_losses=gen_losses,
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

    def on_validation_epoch_end(self):
        assert self.si_snr.total == self.si_sdr.total == self.stoi.count == self.pesq_wb.count, f"Metrics count mismatch: {self.si_snr.total}, {self.si_sdr.total}, {self.stoi.count}, {self.pesq_wb.count}"
        self.log('val_stats/si_snr', self.si_snr.compute(), on_epoch=True, logger=True, sync_dist=True)
        self.log('val_stats/si_sdr', self.si_sdr.compute(), on_epoch=True, logger=True, sync_dist=True)
        self.log('val_stats/stoi', self.stoi.compute(), on_epoch=True, logger=True, sync_dist=True)
        self.log('val_stats/pesq_wb', self.pesq_wb.compute(), on_epoch=True, logger=True, sync_dist=True)
    
    def validation_step(self, batch, batch_idx):
        output = self(batch)

        disc_losses = self.compute_disc_loss(batch, output)
        gen_losses = self.compute_gen_loss(batch, output)

        self.si_snr.update(output['gt_wav'].squeeze(1), output['gen_wav'].squeeze(1))
        self.si_sdr.update(output['gt_wav'].squeeze(1), output['gen_wav'].squeeze(1))
        self.stoi.update(output['gt_wav'].squeeze(1), output['gen_wav'].squeeze(1))
        self.pesq_wb.update(output['gt_wav'].squeeze(1), output['gen_wav'].squeeze(1))


        self._log_losses(
            stage='val',
            disc_losses=disc_losses, gen_losses=gen_losses, output=output,
            batch_size=self.cfg.dataset.val.batch_size,
            on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # remove weight norm for modules in self.model
        batch_size = batch['wav'].size(0)
        output = self(batch)
        target_type = self.cfg.voice_conversion  # 'rec' or 'same'
        assert target_type in ['rec', 'same'], f"Invalid target_type: {target_type}"
        for i in range(batch_size):
            source_filename = batch['fid'][i]
            if target_type == 'rec':
                target_filename = batch['target_id'][i]
            elif target_type == 'same':
                target_filename = source_filename
            gt_i, ref_i, gen_i = batch['wav'][i], batch['ref_wav'][i], output['gen_wav'][i]
            if gt_i.dim() == 1: gt_i = gt_i.unsqueeze(0)
            if ref_i.dim() == 1: ref_i = ref_i.unsqueeze(0)
            if gen_i.dim() == 1: gen_i = gen_i.unsqueeze(0)
            torchaudio.save(f"{self.cwd}/{source_filename}-{target_filename}_{target_type}.wav", gen_i.float().detach().cpu(), self.cfg.preprocess.audio.sr)
            torchaudio.save(f"{self.cwd}/{source_filename}-{target_filename}_gt.wav", gt_i.float().detach().cpu(), self.cfg.preprocess.audio.sr)
            torchaudio.save(f"{self.cwd}/{source_filename}-{target_filename}_ref.wav", ref_i.float().detach().cpu(), self.cfg.preprocess.audio.sr)

    def on_test_start(self):
        self._remove_all_weight_norms()
        self.cwd = os.path.join(os.getcwd(), os.path.splitext(self.cfg.ckpt)[0])
        os.makedirs(self.cwd, exist_ok=True)
        print(colored(f'Test results will be saved in : {self.cwd}', 'yellow', attrs=['bold']))

    # ---- weight norm removal for test (ported from bi module) ----
    def _remove_all_weight_norms(self):
        if getattr(self, "_wn_removed", False):
            return
        removed = 0
        for m in self.model.modules():
            if hasattr(m, 'weight_g') and hasattr(m, 'weight_v'):
                try:
                    torch.nn.utils.remove_weight_norm(m)
                    removed += 1
                except Exception:
                    pass
        self._wn_removed = True
        print(f"[test] removed weight_norm from {removed} submodules.")

    # --- encoder freeze schedule utils ---
    def _is_encoder_frozen(self):
        if not self._freeze_schedule_enabled:
            return False
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

    def _gather_loss_dict(self, disc_losses, gen_losses):
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

    def _log_losses(self, stage, disc_losses, gen_losses, output,
                    batch_size, gen_opt=None, disc_opt=None,
                    on_step=False, on_epoch=False):
        losses = self._gather_loss_dict(disc_losses, gen_losses)
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
        gen_params = chain(self.model['CodecEnc'].parameters(), self.model['generator'].parameters())
        
        gen_opt = optim.AdamW(gen_params, **self.cfg.train.gen_optim_params)
        disc_opt = optim.AdamW(disc_params, **self.cfg.train.disc_optim_params)

        gen_sche = WarmupLR(gen_opt, **self.cfg.train.gen_schedule_params)
        disc_sche = WarmupLR(disc_opt, **self.cfg.train.disc_schedule_params)
        # print(f'Generator optim: {gen_opt}')
        # print(f'Discriminator optim: {disc_opt}')
        return [gen_opt, disc_opt], [gen_sche, disc_sche]

    def set_discriminator_gradients(self, flag=True):
        for p in self.model['discriminator'].parameters():
            p.requires_grad = flag
        
        if 'spec_discriminator' in self.model:
            for p in self.model['spec_discriminator'].parameters():
                p.requires_grad = flag
