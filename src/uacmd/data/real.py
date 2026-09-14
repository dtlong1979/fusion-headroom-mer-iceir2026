from __future__ import annotations
import os
import pickle
from typing import Sequence
import numpy as np
import torch
MODALITIES = ('T', 'A', 'V')
_MMSA_KEYS = {'T': 'text', 'A': 'audio', 'V': 'vision'}
MELD_EMOTIONS = ('neutral', 'joy', 'surprise', 'anger', 'sadness', 'disgust', 'fear')
MOSEI_BINARY = ('negative', 'non-negative')

def _pool(x: np.ndarray, mode: str='mean') -> np.ndarray:
    if x.ndim == 2:
        return x.astype(np.float32)
    if mode == 'mean':
        mask = (np.abs(x).sum(axis=-1, keepdims=True) > 0).astype(np.float32)
        denom = np.clip(mask.sum(axis=1), 1.0, None)
        return ((x * mask).sum(axis=1) / denom).astype(np.float32)
    if mode == 'last':
        return x[:, -1, :].astype(np.float32)
    raise ValueError(f"unknown pooling '{mode}'")

def _sanitize(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def load_mmsa_pickle(path: str, label_key: str='labels', pooling: str='mean', binarize: bool=False) -> dict:
    with open(path, 'rb') as fh:
        raw = pickle.load(fh)
    split_alias = {'train': 'train', 'val': 'valid', 'test': 'test'}
    data = {}
    for ours, theirs in split_alias.items():
        if theirs not in raw:
            raise KeyError(f"split '{theirs}' missing from {path}; found {list(raw)}")
        part = raw[theirs]
        x = {}
        for m in MODALITIES:
            arr = _sanitize(np.asarray(part[_MMSA_KEYS[m]], dtype=np.float32))
            x[m] = torch.from_numpy(_pool(arr, pooling))
        y = np.asarray(part.get(label_key, part.get('regression_labels'))).squeeze()
        if binarize:
            y = (y >= 0).astype(np.int64)
        data[ours] = {'x': x, 'y': torch.from_numpy(np.asarray(y, dtype=np.int64)), 'split': ours}
    return data

def load_meld(annotations_dir: str, features_root: str, modality_dirs: dict | None=None, pooling: str='mean') -> dict:
    import pandas as pd
    modality_dirs = modality_dirs or {'T': 'text', 'A': 'audio', 'V': 'visual'}
    files = {'train': 'train_sent_emo.csv', 'val': 'dev_sent_emo.csv', 'test': 'test_sent_emo.csv'}
    label_index = {e: i for i, e in enumerate(MELD_EMOTIONS)}
    data = {}
    for split, csv_name in files.items():
        df = pd.read_csv(os.path.join(annotations_dir, csv_name))
        feats = {m: [] for m in MODALITIES}
        labels, kept = ([], 0)
        for _, row in df.iterrows():
            stem = f"dia{int(row['Dialogue_ID'])}_utt{int(row['Utterance_ID'])}.npy"
            paths = {m: os.path.join(features_root, split, modality_dirs[m], stem) for m in MODALITIES}
            if not all((os.path.exists(p) for p in paths.values())):
                continue
            for m in MODALITIES:
                feats[m].append(_pool(_sanitize(np.load(paths[m]))[None], pooling)[0])
            labels.append(label_index[str(row['Emotion']).strip().lower()])
            kept += 1
        if kept == 0:
            raise FileNotFoundError(f"no MELD features found for split '{split}' under {features_root}")
        data[split] = {'x': {m: torch.from_numpy(np.stack(feats[m])) for m in MODALITIES}, 'y': torch.from_numpy(np.asarray(labels, dtype=np.int64)), 'split': split, 'n_dropped': int(len(df) - kept)}
    return data

def infer_dims(data: dict) -> dict:
    return {m: int(data['train']['x'][m].shape[-1]) for m in MODALITIES}

def standardize_(data: dict) -> dict:
    stats = {}
    for m in MODALITIES:
        xt = data['train']['x'][m]
        stats[m] = (xt.mean(0, keepdim=True), xt.std(0, keepdim=True).clamp(min=1e-06))
    for split in data:
        for m in MODALITIES:
            mu, sd = stats[m]
            data[split]['x'][m] = (data[split]['x'][m] - mu) / sd
    return data

def class_distribution(data: dict, num_classes: int) -> dict:
    return {split: torch.bincount(data[split]['y'], minlength=num_classes).tolist() for split in data}
