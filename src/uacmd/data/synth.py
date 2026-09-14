from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
import numpy as np
import torch
MODALITIES = ('T', 'A', 'V')
_PRIORS_PATH = os.path.join(os.path.dirname(__file__), 'meld_priors.json')

@dataclass
class SynthConfig:
    num_classes: int = 7
    dims: dict = field(default_factory=lambda: {'T': 128, 'A': 64, 'V': 32})
    latent_dim: int = 24
    private_dim: int = 20
    snr: dict = field(default_factory=lambda: {'T': 0.6, 'A': 0.55, 'V': 0.48})
    shared_weight: dict = field(default_factory=lambda: {'T': 0.5, 'A': 0.38, 'V': 0.32})
    corrupt_rate: dict = field(default_factory=lambda: {'T': 0.05, 'A': 0.2, 'V': 0.25})
    corrupt_factor: float = 6.0
    class_sep: float = 0.6
    n_train: int = 9989
    n_val: int = 1109
    n_test: int = 2610
    seed: int = 0

    def to_dict(self):
        return asdict(self)

def load_meld_priors() -> dict:
    with open(_PRIORS_PATH) as fh:
        return json.load(fh)

class SynthMM:
    EMOTIONS = ('neutral', 'joy', 'surprise', 'anger', 'sadness', 'disgust', 'fear')

    def __init__(self, cfg: SynthConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.rng = rng
        d_lat, d_priv, K = (cfg.latent_dim, cfg.private_dim, cfg.num_classes)
        self.centroids = rng.normal(0, cfg.class_sep, size=(K, d_lat))
        self.private_centroids = {m: rng.normal(0, cfg.class_sep, size=(K, d_priv)) for m in MODALITIES}
        self.mixing = {m: rng.normal(0, 1.0 / np.sqrt(d_lat + d_priv), size=(d_lat + d_priv, cfg.dims[m])) for m in MODALITIES}
        self.priors = load_meld_priors()

    def _labels(self, n: int, split: str, rng) -> np.ndarray:
        p = np.array([self.priors[split][e] for e in self.EMOTIONS], dtype=np.float64)
        p = p / p.sum()
        return rng.choice(self.cfg.num_classes, size=n, p=p)

    def generate(self, split: str, n: int, seed: int) -> dict:
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        y = self._labels(n, split, rng)
        shared = self.centroids[y] + rng.normal(0, 1.0, size=(n, cfg.latent_dim))
        features, corrupt = ({}, {})
        for m in MODALITIES:
            private = self.private_centroids[m][y] + rng.normal(0, 1.0, size=(n, cfg.private_dim))
            latent = np.concatenate([cfg.shared_weight[m] * shared, (1.0 - cfg.shared_weight[m]) * private], axis=1)
            signal = cfg.snr[m] * (latent @ self.mixing[m])
            is_corrupt = rng.random(n) < cfg.corrupt_rate[m]
            scale = np.where(is_corrupt, cfg.corrupt_factor, 1.0)[:, None]
            noise = rng.normal(0, 1.0, size=(n, cfg.dims[m])) * scale
            features[m] = (signal + noise).astype(np.float32)
            corrupt[m] = is_corrupt.astype(np.float32)
        return {'x': {m: torch.from_numpy(features[m]) for m in MODALITIES}, 'y': torch.from_numpy(y.astype(np.int64)), 'corrupt': {m: torch.from_numpy(corrupt[m]) for m in MODALITIES}, 'split': split}

    def build(self, data_seed: int=0) -> dict:
        cfg = self.cfg
        return {'train': self.generate('train', cfg.n_train, 10000 + data_seed), 'val': self.generate('dev', cfg.n_val, 20000 + data_seed), 'test': self.generate('test', cfg.n_test, 30000 + data_seed)}

def standardize_(data: dict) -> dict:
    stats = {}
    for m in MODALITIES:
        xt = data['train']['x'][m]
        mu, sd = (xt.mean(0, keepdim=True), xt.std(0, keepdim=True).clamp(min=1e-06))
        stats[m] = (mu, sd)
    for split in ('train', 'val', 'test'):
        for m in MODALITIES:
            mu, sd = stats[m]
            data[split]['x'][m] = (data[split]['x'][m] - mu) / sd
    return data
