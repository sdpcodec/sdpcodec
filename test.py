import os
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from ptl.bigcodec.data_module import DataModule as BigCodecDataModule
from ptl.bigcodec.lightning_module import BigCodecLightningModule
from ptl.bicodec.data_module import DataModule as BiCodecDataModule
from ptl.bicodec.lightning_module import BiCodecLightningModule
from ptl.tricodec.lightning_module import TriCodecLightningModule
from ptl.trixcodec.lightning_module import TriXCodecLightningModule


def _infer_modules_from_hydra_config_yaml(cfg, ckpt_path: str):
    """
    구버전 run: hydra/config.yaml 만 있고 job.config_name 이 'config' 인 경우,
    체크포인트 state_dict 로 Lightning 모듈 종류를 추론한다.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    keys = "\n".join(ckpt.get("state_dict", {}).keys())
    if "joint_mixer" in keys:
        return BiCodecDataModule(cfg), TriXCodecLightningModule(cfg)
    if "f0_quantizer" in keys:
        return BiCodecDataModule(cfg), TriCodecLightningModule(cfg)
    if "speaker_encoder" in keys:
        return BiCodecDataModule(cfg), BiCodecLightningModule(cfg)
    return BigCodecDataModule(cfg), BigCodecLightningModule(cfg)


def _configure_reproducibility(cfg) -> int:
    seed = int(getattr(cfg, "seed", 1024))
    deterministic = bool(getattr(cfg, "deterministic", False))
    seed_everything(seed, workers=True)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    return seed

@hydra.main(config_path='config', config_name='default', version_base=None)
def train(cfg):
    # 1) 과거 실행 디렉터리(run_dir) 구성 병합
    run_dir = getattr(cfg, "run_dir", None) or os.environ.get("RUN_DIR", None)
    if run_dir:
        saved_cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
        if not os.path.isfile(saved_cfg_path):
            raise FileNotFoundError(f"Saved config not found: {saved_cfg_path}")
        saved_cfg = OmegaConf.load(saved_cfg_path)
        # saved_cfg 우선, 현재 CLI 오버라이드로 다시 덮어씀
        cfg = OmegaConf.merge(saved_cfg, cfg)
        # ckpt 상대경로면 run_dir 기준으로 절대경로화
        if "ckpt" in cfg and cfg.ckpt and not os.path.isabs(str(cfg.ckpt)):
            cfg.ckpt = os.path.join(run_dir, str(cfg.ckpt))

    seed = _configure_reproducibility(cfg)
    print(f"[repro] seed={seed}, deterministic={bool(getattr(cfg, 'deterministic', False))}")

    # 2) 현재 실행의 Hydra run dir
    try:
        hydra_run_dir = HydraConfig.get().runtime.output_dir  # ex: .../outputs/2025-08-26/22-16-23
    except Exception:
        hydra_run_dir = os.getcwd()

    ts_date = os.path.basename(os.path.dirname(hydra_run_dir))
    ts_time = os.path.basename(hydra_run_dir)
    ts_slug = f"{ts_date}-{ts_time}"
    print(f"ts_slug: {ts_slug}")
    print(f"logdir: {getattr(cfg, 'log_dir', None)}")

    # 3) 모듈 구성
    config_name = HydraConfig.get().job.config_name
    if config_name in ['default', 'base', 'vqw2v_enc_base', 'bigbase']:
        datamodule = BigCodecDataModule(cfg)
        lightning_module = BigCodecLightningModule(cfg)
    elif config_name in ['bicodec', 'midbicodec', 'bibase', 'vqw2v_enc_bibase', 'vqw2v_enc_bibase_mi', 'vqw2v_enc_24kHz_bibase', 'w2v2_enc_24kHz_bibase', 'w2vbert2_enc_24kHz_bibase']:
        datamodule = BiCodecDataModule(cfg)
        lightning_module = BiCodecLightningModule(cfg)
    elif config_name in ['vqw2v_enc_tribase', 'vqw2v_enc_tricodec', 'vqw2v_enc_24kHz_480hs_tribase']:
        # 기본값: bicodec 계열
        datamodule = BiCodecDataModule(cfg)
        lightning_module = TriCodecLightningModule(cfg)
    elif config_name in [
        'vqw2v_enc_24kHz_trixbase',
        'vqw2v_enc_rndvoc_24kHz_trixbase',
        'vqw2v_enc_vocos_24kHz_trixbase',
        'vqw2v_enc_vocos_16kHz_trixbase',
        'vqw2v_enc_16kHz_trixbase',
        'w2v2_enc_24kHz_trixbase',
        'mls_vqw2v_enc_16kHz_trixbase',
        'hubert_enc_24kHz_trixbase',
        'w2vbert2_enc_24kHz_trixbase',
        'hubert_enc_vocos_24kHz_trixbase',
        'hubert_enc_rndvoc_24kHz_trixbase',
    ]:
        datamodule = BiCodecDataModule(cfg)
        lightning_module = TriXCodecLightningModule(cfg)
    elif config_name == 'config':
        ckpt_for_infer = getattr(cfg, "ckpt", None)
        if not ckpt_for_infer:
            raise ValueError(
                "Hydra primary config is config.yaml; set ckpt=... so the architecture can be inferred."
            )
        datamodule, lightning_module = _infer_modules_from_hydra_config_yaml(cfg, str(ckpt_for_infer))
    else:
        raise ValueError(f"Unknown Hydra config_name for test.py: {config_name!r}")

    ckpt = getattr(cfg, "ckpt", None)
    if ckpt:
        print(f"Test from checkpoint: {ckpt}")


    trainer_kwargs = dict(**cfg.train.trainer)
    trainer_kwargs.setdefault("deterministic", bool(getattr(cfg, "deterministic", False)))
    # Only use DDP when devices > 1
    try:
        devices = int(trainer_kwargs.get("devices", 1))
    except Exception:
        devices = 1
    if devices and devices > 1:
        strategy = DDPStrategy(process_group_backend='nccl', find_unused_parameters=True)
        trainer_kwargs["strategy"] = strategy

    trainer = pl.Trainer(**trainer_kwargs)

    # 5) 테스트 실행
    trainer.test(lightning_module, datamodule=datamodule, ckpt_path=ckpt)
    print("Test end")

if __name__ == '__main__':
    train()
