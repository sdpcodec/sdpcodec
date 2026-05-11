import re
import os
from copy import deepcopy

# Configure HF timeouts before any hub/datasets import so the libraries pick them up.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(120))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(120))

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import pytorch_lightning as pl
import random
import librosa
from os.path import basename, exists, join
from torch.utils.data import Dataset, DataLoader, Sampler
import hydra
import utils
from datasets import load_dataset, concatenate_datasets, DownloadConfig
from ptl.sampler import ResumableRandomSampler, ResumableDistributedSampler
import torch.distributed as dist
from typing import List, Optional
from termcolor import colored


def _dataset_cache_root(cache_dir: Optional[str]) -> str:
    if cache_dir:
        return str(cache_dir)
    hf_datasets_cache = os.environ.get("HF_DATASETS_CACHE")
    if hf_datasets_cache:
        return str(hf_datasets_cache)
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return str(os.path.join(hf_home, "datasets"))


def _has_local_dataset_cache(repo_id: str, cache_dir: Optional[str]) -> bool:
    if "/" not in repo_id:
        return False
    dataset_dir = os.path.join(_dataset_cache_root(cache_dir), repo_id.replace("/", "___"))
    return os.path.exists(dataset_dir)


def _looks_like_hf_network_timeout(err: Exception) -> bool:
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


def _make_offline_download_config(dc: Optional[DownloadConfig]) -> DownloadConfig:
    if dc is not None:
        try:
            dc = deepcopy(dc)
        except Exception:
            dc = None
    if dc is None:
        dc = DownloadConfig()
    dc.local_files_only = True
    dc.max_retries = 1
    return dc


def _load_dataset_cached(*args, **kwargs):
    try:
        return load_dataset(*args, **kwargs)
    except Exception as err:
        repo_id = str(args[0]) if args else ""
        cache_dir = kwargs.get("cache_dir")
        if not (_looks_like_hf_network_timeout(err) and _has_local_dataset_cache(repo_id, cache_dir)):
            raise
        print(colored(
            f"HF dataset metadata request timed out for {repo_id}. Falling back to local cached files only.",
            "yellow",
            attrs=["bold"],
        ))
        offline_kwargs = dict(kwargs)
        offline_kwargs["download_config"] = _make_offline_download_config(kwargs.get("download_config"))
        return load_dataset(*args, **offline_kwargs)


def _extract_speaker_ids(ds) -> List[str]:
    cols = getattr(ds, "column_names", [])
    if 'speaker_id' in cols:
        return [str(s) for s in ds['speaker_id']]
    ids = ds['id']
    spk_ids: List[str] = []
    for i in ids:
        s = str(i)
        m = re.match(r'^(\d+)[-_]', s)
        if m:
            spk_ids.append(m.group(1))
            continue
        spk_ids.append(re.split(r'[-_]', s)[0])
    return spk_ids


def add_same_spk_pair_in_test_dataset(cfg, test_dataset):
    """Add rec_pair_idx/rec_pair_id for same-speaker reconstruction (another utterance from same speaker)."""
    id_list: List[str] = test_dataset['id']
    spk_ids: List[str] = _extract_speaker_ids(test_dataset)
    spk2idxs = {}
    for idx, spk in enumerate(spk_ids):
        spk2idxs.setdefault(spk, []).append(idx)
    all_spks = list(spk2idxs.keys())
    seed = getattr(cfg.dataset, "pair_seed", 42)
    rng = random.Random(seed)
    pair_idx_list: List[int] = []
    for i, spk in enumerate(spk_ids):
        candidates = [j for j in spk2idxs[spk] if j != i]
        if len(candidates) == 0:
            other_spks = [s for s in all_spks if s != spk and len(spk2idxs[s]) > 0]
            if len(other_spks) == 0:
                j = i
            else:
                other_spk = rng.choice(other_spks)
                j = rng.choice(spk2idxs[other_spk])
        else:
            j = rng.choice(candidates)
        pair_idx_list.append(j)
    pair_id_list: List[str] = [id_list[j] for j in pair_idx_list]
    test_dataset = test_dataset.add_column('rec_pair_idx', pair_idx_list)
    test_dataset = test_dataset.add_column('rec_pair_id', pair_id_list)
    return test_dataset


def load_librispeech_dataset(cfg, dc):
    cache_dir = cfg.cache_dir
    print(f"Loading dataset in {cache_dir}...")
    return load_librispeech_splits(cfg, dc, load_train=True, load_val=True, load_test=True)


def load_librispeech_splits(cfg, dc, *, load_train: bool, load_val: bool, load_test: bool):
    cache_dir = cfg.cache_dir
    print(f"Loading dataset in {cache_dir}...")

    train_dataset = None
    val_dataset = None
    test_dataset = None

    if load_train:
        train_clean_100 = _load_dataset_cached('openslr/librispeech_asr',
                                       split="train.clean.100",
                                       num_proc=cfg.dataset.train.num_workers,
                                       cache_dir=cache_dir,
                                       download_config=dc)
        train_clean_360 = _load_dataset_cached('openslr/librispeech_asr',
                                       split="train.clean.360",
                                       num_proc=cfg.dataset.train.num_workers,
                                       cache_dir=cache_dir,
                                       download_config=dc)
        train_other_500 = _load_dataset_cached('openslr/librispeech_asr',
                                       split="train.other.500",
                                       num_proc=cfg.dataset.train.num_workers,
                                       cache_dir=cache_dir,
                                       download_config=dc)
        train_dataset = concatenate_datasets(
            [train_clean_100, train_clean_360, train_other_500])

    if load_val:
        val_dataset = _load_dataset_cached('openslr/librispeech_asr',
                                   split="validation.clean",
                                   num_proc=cfg.dataset.val.num_workers,
                                   cache_dir=cache_dir,
                                   download_config=dc)

    if load_test:
        test_dataset = _load_dataset_cached('openslr/librispeech_asr',
                                    split="test.clean",
                                    num_proc=cfg.dataset.test.num_workers,
                                    cache_dir=cache_dir,
                                    download_config=dc)

    return train_dataset, val_dataset, test_dataset


def load_libritts_dataset(cfg, dc):
    cache_dir = cfg.cache_dir
    print(f"Loading dataset in {cache_dir}...")
    return load_libritts_splits(cfg, dc, load_train=True, load_val=True, load_test=True)


def load_libritts_splits(cfg, dc, *, load_train: bool, load_val: bool, load_test: bool):
    cache_dir = cfg.cache_dir
    print(f"Loading dataset in {cache_dir}...")

    train_dataset = None
    val_dataset = None
    test_dataset = None

    if load_train:
        train_clean_100 = _load_dataset_cached('mythicinfinity/libritts', 'clean', split="train.clean.100",
                                       num_proc=cfg.dataset.train.num_workers, cache_dir=cache_dir, download_config=dc)
        train_clean_360 = _load_dataset_cached('mythicinfinity/libritts', 'clean', split="train.clean.360",
                                       num_proc=cfg.dataset.train.num_workers, cache_dir=cache_dir, download_config=dc)
        train_other_500 = _load_dataset_cached('mythicinfinity/libritts', 'other', split="train.other.500",
                                       num_proc=cfg.dataset.train.num_workers, cache_dir=cache_dir, download_config=dc)
        train_dataset = concatenate_datasets(
            [train_clean_100, train_clean_360, train_other_500])

    if load_val:
        val_dataset = _load_dataset_cached('mythicinfinity/libritts', 'dev', split="dev.clean",
                                   num_proc=cfg.dataset.val.num_workers, cache_dir=cache_dir, download_config=dc)

    if load_test:
        test_dataset = _load_dataset_cached('mythicinfinity/libritts', 'clean', split="test.clean",
                                    num_proc=cfg.dataset.test.num_workers, cache_dir=cache_dir, download_config=dc)

    return train_dataset, val_dataset, test_dataset


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        ocwd = hydra.utils.get_original_cwd()
        self.ocwd = ocwd
        offline_env = (os.environ.get("HF_HUB_OFFLINE", "0") == "1" or
                       os.environ.get("HF_DATASETS_OFFLINE", "0") == "1")
        self.dc = DownloadConfig(
            max_retries=20,
            resume_download=True,
            use_etag=True,
            local_files_only=offline_env,
        )
        self.dataset_name = self.cfg.dataset.get('name', 'librispeech').lower()
        if self.dataset_name == 'librispeech':
            print(colored("Using librispeech dataset.", "green", attrs=['bold']))
        elif self.dataset_name == 'libritts':
            print(colored("Using libritts dataset.", "green", attrs=['bold']))
        else:
            raise ValueError(
                f"Unsupported dataset: {self.dataset_name}. Supported datasets are librispeech and libritts.")

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Lazy-load splits to avoid loading train set during test."""
        if stage is not None:
            stage = str(stage).lower()

        def _load_splits(*, load_train: bool, load_val: bool, load_test: bool):
            if self.dataset_name == 'librispeech':
                return load_librispeech_splits(self.cfg, self.dc,
                                               load_train=load_train, load_val=load_val, load_test=load_test)
            return load_libritts_splits(self.cfg, self.dc,
                                        load_train=load_train, load_val=load_val, load_test=load_test)

        if stage in (None, 'fit'):
            if self.train_dataset is None or self.val_dataset is None:
                train_ds, val_ds, _ = _load_splits(load_train=True, load_val=True, load_test=False)
                self.train_dataset = train_ds
                self.val_dataset = val_ds
                print(f"Train dataset size: {len(self.train_dataset)}")
                print(f"Validation dataset size: {len(self.val_dataset)}")

        if stage in (None, 'validate'):
            if self.val_dataset is None:
                _, val_ds, _ = _load_splits(load_train=False, load_val=True, load_test=False)
                self.val_dataset = val_ds
                print(f"Validation dataset size: {len(self.val_dataset)}")

        if stage in (None, 'test'):
            if self.test_dataset is None:
                _, _, test_ds = _load_splits(load_train=False, load_val=False, load_test=True)
                self.test_dataset = add_same_spk_pair_in_test_dataset(self.cfg, test_ds)
                print(f"Test dataset size: {len(self.test_dataset)}")

    def get_loader(self, phase, dataset):
        phase_cfg = self.cfg.dataset.get(phase)
        batch_size = phase_cfg.batch_size
        ds = FSDataset(phase, self.cfg, dataset)
        dl = DataLoader(ds, batch_size=batch_size,
                        shuffle=phase_cfg.shuffle,
                        num_workers=phase_cfg.num_workers,
                        prefetch_factor=(phase_cfg.prefetch_factor if phase_cfg.num_workers > 0 else None),
                        collate_fn=ds.collate_fn,
                        persistent_workers=(phase_cfg.num_workers > 0),
                        drop_last=phase_cfg.shuffle,
                        pin_memory=torch.cuda.is_available(),
                        )
        return dl

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup('fit')
        return self.get_loader('train', self.train_dataset)

    def val_dataloader(self):
        if self.val_dataset is None:
            self.setup('validate')
        return self.get_loader('val', self.val_dataset)

    def test_dataloader(self):
        if self.test_dataset is None:
            self.setup('test')
        return self.get_loader('test', self.test_dataset)


class FSDataset(Dataset):
    """Dataset for BigCodec (same-speaker reconstruction)."""

    def __init__(self, phase, cfg, dataset):
        self.phase = phase
        self.cfg = cfg
        self.phase_cfg = cfg.dataset.get(phase)
        self.dataset = dataset
        self.dataset_name = cfg.dataset.get('name', 'librispeech').lower()
        self.ocwd = hydra.utils.get_original_cwd()

        self.sr = cfg.preprocess.audio.sr
        self.min_audio_length = cfg.dataset.min_audio_length
        self.latent_hop_length = cfg.dataset.latent_hop_length

    def __len__(self):
        return len(self.dataset)

    def load_wav(self, path):
        wav, sr = librosa.load(path, sr=self.sr)
        return wav

    def __getitem__(self, idx):
        item = self.dataset[idx]
        wav = torch.from_numpy(item['audio']['array'])
        if wav.dtype != torch.float32:
            wav = wav.float()

        # LibriTTS: 24k -> 16k
        if self.dataset_name == 'libritts':
            orig_sr, target_sr = 24000, 16000
            if orig_sr != target_sr:
                wav = torchaudio.functional.resample(wav, orig_sr, target_sr)
                wav = wav.clamp(-1.0, 1.0)

        length = wav.shape[0]
        if length < self.min_audio_length:
            wav = F.pad(wav, (0, self.min_audio_length - length))
            length = wav.shape[0]

        use_full = (self.phase in ['test', 'predict'])
        if use_full:
            if self.latent_hop_length:
                length_aligned = (length // self.latent_hop_length) * self.latent_hop_length
                wav = wav[:length_aligned]
                length = wav.shape[0]
            target_wav = wav
        else:
            i = random.randint(0, length - self.min_audio_length)
            target_wav = wav[i:i + self.min_audio_length]

        # Reference: same-speaker rec uses rec_pair_idx, else same utterance
        if self.cfg.voice_conversion == 'rec' and self.phase == 'test':
            pair_idx = int(item['rec_pair_idx'])
            paired = self.dataset[pair_idx]
            ref_src = torch.from_numpy(paired['audio']['array']).to(wav.dtype)
            if self.dataset_name == 'libritts':
                ref_src = torchaudio.functional.resample(ref_src, 24000, 16000).clamp(-1.0, 1.0)
            ref_len = target_wav.shape[0]
            if ref_src.shape[0] >= ref_len:
                start = 0 if ref_src.shape[0] == ref_len else random.randint(0, ref_src.shape[0] - ref_len)
                ref_wav = ref_src[start:start + ref_len]
            else:
                ref_wav = F.pad(ref_src, (0, ref_len - ref_src.shape[0]))
            target_id = paired['id']
        else:
            ref_wav = target_wav
            target_id = None

        out = {
            'fid': item['id'],
            'wav': target_wav,
            'ref_wav': ref_wav,
            'target_id': target_id,
        }
        return out

    def collate_fn(self, bs):
        fids = [b['fid'] for b in bs]
        wavs = [b['wav'] for b in bs]
        ref_wavs = [b['ref_wav'] for b in bs]
        target_ids = [b['target_id'] for b in bs] if self.phase == 'test' else None
        if self.latent_hop_length and self.phase == 'test':
            max_len = max([w.shape[0] for w in wavs])
            max_len = (max_len // self.latent_hop_length) * self.latent_hop_length
            wavs = [F.pad(w, (0, max_len - w.shape[0])) for w in wavs]
            ref_wavs = [F.pad(r, (0, max_len - r.shape[0])) for r in ref_wavs]

        wavs = torch.stack(wavs)
        ref_wavs = torch.stack(ref_wavs)

        out = {
            'fid': fids,
            'wav': wavs,
            'ref_wav': ref_wavs,
            'target_id': target_ids,
        }
        return out
