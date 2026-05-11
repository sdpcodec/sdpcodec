import os
import sys
import warnings
import pytorch_lightning as pl
import hydra
import torch
import random
import time
from os.path import join, basename, exists
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from ptl.bigcodec.data_module import DataModule as BigCodecDataModule
from ptl.bigcodec.lightning_module import BigCodecLightningModule
from ptl.bicodec.data_module import DataModule as BiCodecDataModule
from ptl.bicodec.lightning_module import BiCodecLightningModule
from ptl.tricodec.lightning_module import TriCodecLightningModule
from ptl.trixcodec.lightning_module import TriXCodecLightningModule
from pytorch_lightning.loggers import WandbLogger
import datetime
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from omegaconf import OmegaConf
from termcolor import colored
import re


def _first_existing_file(*paths):
    for path in paths:
        if path and os.path.isfile(path):
            return path
    return None


def _resolve_saved_config_path(run_dir: str) -> str:
    run_dir = os.path.abspath(os.path.expanduser(str(run_dir)))
    saved_cfg_path = _first_existing_file(
        os.path.join(run_dir, "hydra", "config.yaml"),
        os.path.join(run_dir, ".hydra", "config.yaml"),
    )
    if saved_cfg_path is None:
        raise FileNotFoundError(
            f"Saved config not found under '{run_dir}' "
            f"(looked for hydra/config.yaml and .hydra/config.yaml)"
        )
    return saved_cfg_path


def _resolve_named_config_path(run_dir: str, config_name: str | None) -> str | None:
    run_dir = os.path.abspath(os.path.expanduser(str(run_dir)))
    if not config_name:
        return None
    return _first_existing_file(
        os.path.join(run_dir, f"{config_name}.yaml"),
        os.path.join(run_dir, "hydra", f"{config_name}.yaml"),
        os.path.join(run_dir, ".hydra", f"{config_name}.yaml"),
    )


def _build_resume_override_cfg() -> object:
    try:
        task_overrides = list(HydraConfig.get().overrides.task)
    except Exception:
        task_overrides = []

    filtered_overrides = [override for override in task_overrides if not override.startswith("run_dir=")]
    if not filtered_overrides:
        return OmegaConf.create()
    return OmegaConf.from_dotlist(filtered_overrides)


def _resolve_saved_config_name(run_dir: str, fallback: str | None = None) -> str:
    run_dir = os.path.abspath(os.path.expanduser(str(run_dir)))
    ignored_names = {"config.yaml", "hydra.yaml", "overrides.yaml"}
    candidate_names: list[str] = []
    for candidate_dir in (Path(run_dir) / "hydra", Path(run_dir) / ".hydra", Path(run_dir)):
        if not candidate_dir.is_dir():
            continue
        candidate_names.extend(
            path.stem for path in candidate_dir.glob("*.yaml") if path.name not in ignored_names
        )

    unique_candidate_names = sorted(set(candidate_names))
    specific_candidate_names = [name for name in unique_candidate_names if name != "default"]
    unique_candidate_name = unique_candidate_names[0] if len(unique_candidate_names) == 1 else None
    unique_specific_candidate_name = (
        specific_candidate_names[0] if len(specific_candidate_names) == 1 else None
    )

    saved_hydra_path = _first_existing_file(
        os.path.join(run_dir, "hydra", "hydra.yaml"),
        os.path.join(run_dir, ".hydra", "hydra.yaml"),
    )
    if saved_hydra_path is not None:
        try:
            saved_hydra_cfg = OmegaConf.load(saved_hydra_path)
            saved_name = OmegaConf.select(saved_hydra_cfg, "hydra.job.config_name")
            if saved_name and saved_name != "default":
                return str(saved_name)
        except Exception:
            pass

    if unique_specific_candidate_name is not None:
        return unique_specific_candidate_name

    if unique_candidate_name is not None:
        return unique_candidate_name

    return str(fallback or "default")


def _extract_dotlist_value(args: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def _prepare_resume_argv(argv: list[str]) -> list[str] | None:
    run_dir_value = _extract_dotlist_value(argv[1:], "run_dir")
    if not run_dir_value:
        return None

    run_dir = os.path.abspath(os.path.expanduser(run_dir_value))
    saved_config_name = _resolve_saved_config_name(run_dir, fallback="default")

    explicit_config_name = None
    normalized_args = [argv[0]]
    changed = False
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--config-name":
            if i + 1 < len(argv):
                explicit_config_name = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--config-name="):
            explicit_config_name = arg.split("=", 1)[1]
            i += 1
            continue
        if arg.startswith("hydra.run.dir="):
            changed = True
            i += 1
            continue
        if arg.startswith("run_dir="):
            normalized_args.append("+" + arg)
            changed = True
            i += 1
            continue
        normalized_args.append(arg)
        i += 1

    effective_config_name = explicit_config_name
    if effective_config_name in (None, "", "default"):
        effective_config_name = saved_config_name
    if effective_config_name != explicit_config_name:
        changed = True

    normalized_args = [normalized_args[0], "--config-name", effective_config_name] + normalized_args[1:]
    return normalized_args if changed else None


def _resolve_resume_ckpt(run_dir: str, ckpt_value) -> str:
    run_dir = os.path.abspath(os.path.expanduser(str(run_dir)))
    if ckpt_value:
        ckpt_path = os.path.expanduser(str(ckpt_value))
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(run_dir, ckpt_path)
        ckpt_path = os.path.abspath(ckpt_path)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    pl_log_dir = Path(run_dir) / "pl_log"
    preferred_last = pl_log_dir / "last.ckpt"
    if preferred_last.is_file():
        return str(preferred_last)

    ckpt_candidates = sorted(
        pl_log_dir.glob("*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if ckpt_candidates:
        return str(ckpt_candidates[0])

    raise FileNotFoundError(
        f"No checkpoint found under '{pl_log_dir}'. "
        f"Expected last.ckpt or at least one '*.ckpt' file."
    )

class SamplerStateCheckpointCallback(pl.Callback):
    """Saves/loads DataModule ResumableSampler state for seamless mid-epoch resume."""

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        dm = getattr(trainer, "datamodule", None)
        if dm is not None and getattr(dm, "_train_sampler", None) is not None:
            try:
                checkpoint["datamodule_sampler_state"] = dm._train_sampler.state_dict()
            except Exception:
                pass


class RankZeroProgressBar(TQDMProgressBar):
    """Progress bar that only updates on rank 0 to avoid duplicate lines in DDP."""
    @property
    def _is_enabled(self) -> bool:
        return self.trainer.global_rank == 0


def _configure_reproducibility(cfg) -> int:
    """
    Make training as reproducible as possible.

    Notes:
    - `seed_everything(..., workers=True)` is important when Dataset uses Python/random
      inside DataLoader workers (e.g., random crop / augmentation).
    - Full determinism on GPU can require disabling certain fast kernels and may impact speed.
    """
    seed = int(getattr(cfg, "seed", 1024))
    deterministic = bool(getattr(cfg, "deterministic", False))

    # Seed python / numpy / torch, and also DataLoader workers
    seed_everything(seed, workers=True)

    if deterministic:
        # Needed for deterministic cuBLAS GEMMs on some CUDA setups (must be set before use)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # Avoid non-deterministic algorithm selection
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Avoid TF32 numeric drift
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        # Enforce deterministic algorithms where possible
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

    return seed


def _configure_runtime_performance(cfg) -> None:
    deterministic = bool(getattr(cfg, "deterministic", False))
    train_cfg = getattr(cfg, "train", None)
    if train_cfg is None:
        return

    matmul_precision = getattr(train_cfg, "float32_matmul_precision", None)
    if torch.cuda.is_available() and matmul_precision:
        torch.set_float32_matmul_precision(str(matmul_precision))

    allow_tf32 = getattr(train_cfg, "allow_tf32", None)
    if allow_tf32 is None or deterministic:
        return

    allow_tf32 = bool(allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    except Exception:
        pass

# if torch.cuda.is_available():
#     torch.set_float32_matmul_precision("high")  # or "medium"
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

@hydra.main(config_path='config', config_name='default', version_base=None)
def train(cfg):
    # Silence PyTorch DDP NCCL barrier/init_process_group warning.
    # This warning is benign in our use (Lightning/torch choose the current device).
    warnings.filterwarnings(
        "ignore",
        message=r"No device id is provided via",
        category=UserWarning,
        module=r"torch\.distributed\.distributed_c10d",
    )

    # 1) 과거 실행 디렉터리(run_dir) 구성 병합
    run_dir = getattr(cfg, "run_dir", None) or os.environ.get("RUN_DIR", None)
    if run_dir:
        run_dir = os.path.abspath(os.path.expanduser(str(run_dir)))
        saved_config_name = _resolve_saved_config_name(run_dir, fallback=getattr(cfg, "config_name", None))
        saved_cfg_path = _resolve_named_config_path(run_dir, saved_config_name)
        if saved_cfg_path is None:
            saved_cfg_path = _resolve_saved_config_path(run_dir)
        saved_cfg = OmegaConf.load(saved_cfg_path)
        resume_override_cfg = _build_resume_override_cfg()
        # Resume uses the saved run config as the source of truth; only explicit CLI overrides are re-applied.
        cfg = OmegaConf.merge(saved_cfg, resume_override_cfg)
        cfg.run_dir = run_dir
        cfg.ckpt = _resolve_resume_ckpt(run_dir, getattr(cfg, "ckpt", None))

    # Reproducibility (after merge so resumed runs use the saved seed)
    seed = _configure_reproducibility(cfg)
    _configure_runtime_performance(cfg)
    print(colored(f"[repro] seed={seed}, deterministic={bool(getattr(cfg, 'deterministic', False))}", "cyan"))

    # 2) 현재 실행의 Hydra run dir
    try:
        hydra_run_dir = HydraConfig.get().runtime.output_dir  # ex) .../outputs/2025-08-26/22-16-23
        config_name = HydraConfig.get().job.config_name
    except Exception:
        hydra_run_dir = os.getcwd()
        config_name = "default"

    if run_dir:
        hydra_run_dir = run_dir
        config_name = _resolve_saved_config_name(run_dir, fallback=config_name)

    # hydra_run_dir을 절대 경로로 변환 (chdir로 인한 상대 경로 문제 방지)
    if not os.path.isabs(hydra_run_dir):
        hydra_run_dir = os.path.abspath(hydra_run_dir)

    # Hydra는 hydra/config.yaml 만 저장함. test.py에서 --config-name으로 쓰려면 실험명 yaml이 필요.
    _rank = 0
    try:
        import torch.distributed as _dist
        if _dist.is_initialized():
            _rank = _dist.get_rank()
    except Exception:
        pass
    if _rank == 0 and config_name and not run_dir:
        _named_cfg = Path(hydra_run_dir) / f"{config_name}.yaml"
        OmegaConf.save(cfg, _named_cfg)
        print(colored(f"[config] saved {_named_cfg}", "cyan"))
        try:
            hydra_output_subdir = HydraConfig.get().output_subdir
        except Exception:
            hydra_output_subdir = "hydra"
        if hydra_output_subdir:
            _named_hydra_cfg = Path(hydra_run_dir) / str(hydra_output_subdir) / f"{config_name}.yaml"
            _named_hydra_cfg.parent.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(cfg, _named_hydra_cfg)
            print(colored(f"[config] saved {_named_hydra_cfg}", "cyan"))
    elif _rank == 0 and config_name and run_dir:
        print(colored(f"[config] skip saving named config files while resuming from {run_dir}", "cyan"))

    # Hydra 런 디렉터리에서 날짜/시간 추출
    ts_date = os.path.basename(os.path.dirname(hydra_run_dir))  # 2025-08-26
    ts_time = os.path.basename(hydra_run_dir)                   # 22-16-23
    ts_slug = f"{ts_date}-{ts_time}"                            # 2025-08-26-22-16-23

    # log_dir을 절대 경로로 변환 (resume 시 같은 디렉토리에 저장하기 위해)
    if not os.path.isabs(str(cfg.log_dir)):
        log_dir_abs = os.path.join(hydra_run_dir, str(cfg.log_dir))
    else:
        log_dir_abs = str(cfg.log_dir)
    cfg.log_dir = log_dir_abs

    print(f"ts_slug : {ts_slug}")
    print(f"logdir : {cfg.log_dir}")

    trainer_kwargs = dict(**cfg.train.trainer)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    enable_checkpointing = bool(trainer_kwargs.get("enable_checkpointing", True))
    checkpoint_callback = None
    callbacks = [lr_monitor]
    if enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(dirpath=cfg.log_dir,
                                save_top_k=3, save_last=True,
                                monitor='val_stats/stoi', mode='max',
                                filename='step={val_stats/total_step:07}-stoi={val_stats/stoi:.4f}',
                                auto_insert_metric_name=False,
                                )
        sampler_state_cb = SamplerStateCheckpointCallback()
        callbacks = [checkpoint_callback, lr_monitor, sampler_state_cb]
    else:
        print("Checkpoint saving disabled for this run.")
    if config_name in ['default', 'base', 'vqw2v_enc_base']:
        datamodule = BigCodecDataModule(cfg)
        lightning_module = BigCodecLightningModule(cfg)
    elif config_name in ['bicodec', 'midbicodec', 'bibase', 'bibase_24kHz', 'vqw2v_enc_bibase', 'vqw2v_enc_bibase_mi', 'vqw2v_enc_24kHz_bibase', 'w2v2_enc_24kHz_bibase', 'w2vbert2_enc_24kHz_bibase']:
        datamodule = BiCodecDataModule(cfg)
        lightning_module = BiCodecLightningModule(cfg)
    elif config_name in ['vqw2v_enc_tribase', 'vqw2v_enc_tricodec', 'vqw2v_enc_24kHz_480hs_tribase']:
        datamodule = BiCodecDataModule(cfg)
        lightning_module = TriCodecLightningModule(cfg)
    elif config_name in ['vqw2v_enc_24kHz_trixbase', 'vqw2v_enc_rndvoc_24kHz_trixbase', 'vqw2v_enc_vocos_24kHz_trixbase', 'vqw2v_enc_vocos_16kHz_trixbase', 'vqw2v_enc_16kHz_trixbase', 'w2v2_enc_24kHz_trixbase', 'hubert_enc_24kHz_trixbase', 'mls_vqw2v_enc_16kHz_trixbase', 'w2vbert2_enc_24kHz_trixbase', 'hubert_enc_vocos_24kHz_trixbase', 'hubert_enc_vocosformer_24kHz_trixbase', 'hubert_enc_apnformer_24kHz_trixbase', 'hubert_enc_rndvoc_24kHz_trixbase']:
        datamodule = BiCodecDataModule(cfg)
        lightning_module = TriXCodecLightningModule(cfg)

    ckpt = cfg.ckpt

    # 출력 디렉터리 이름으로 id 지정 (datetime 대신)

    if cfg.id is not None: id = cfg.id
    else: id = ts_slug

    if ckpt is not None:
        print(colored("Continue training from checkpoint:", "red", attrs=['bold']), ckpt)

    # Weights & Biases logger (원래 사용하던 기본 설정으로 복원)
    wandb_logger_kwargs = dict(
        save_dir=os.path.join(hydra_run_dir, "logs"),
        name=ts_slug,
        project="SDP-Codec",
        offline=bool(
            getattr(cfg.train, "wandb_offline", False)
            or os.environ.get("WANDB_MODE", "").lower() == "offline"
        ),
        id=id,
    )
    if ckpt is not None:
        wandb_logger_kwargs["resume"] = "allow"
    wandb_logger = WandbLogger(**wandb_logger_kwargs)
    wandb_logger.log_hyperparams(cfg)

    # If the config doesn't explicitly set Trainer(deterministic=...), mirror the top-level flag.
    trainer_kwargs.setdefault("deterministic", bool(getattr(cfg, "deterministic", False)))
    # Only use DDP when devices > 1
    try:
        devices = int(trainer_kwargs.get("devices", 1))
    except Exception:
        devices = 1
    if devices and devices > 1:
        strategy = DDPStrategy(
            process_group_backend='nccl',
            find_unused_parameters=bool(getattr(cfg.train, "ddp_find_unused_parameters", True)),
            static_graph=bool(getattr(cfg.train, "ddp_static_graph", False)),
            gradient_as_bucket_view=bool(getattr(cfg.train, "ddp_gradient_as_bucket_view", False)),
        )
        trainer_kwargs["strategy"] = strategy
        # Progress bar only on rank 0 to avoid duplicate lines (each DDP process was printing)
        if bool(trainer_kwargs.get("enable_progress_bar", True)):
            callbacks = [RankZeroProgressBar()] + callbacks

    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
        # limit_train_batches=1.0 if not cfg.debug else 0.001,
        # detect_anomaly=True,
        logger=wandb_logger
    )
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=ckpt)
    if checkpoint_callback is not None:
        print(f'Training ends, best score: {checkpoint_callback.best_model_score}, ckpt path: {checkpoint_callback.best_model_path}')
    else:
        print('Training ends, checkpointing was disabled.')

if __name__ == '__main__':
    resume_argv = _prepare_resume_argv(sys.argv)
    if resume_argv is not None:
        os.execv(sys.executable, [sys.executable] + resume_argv)
    train()
