from __future__ import annotations
import argparse
import importlib.util
import json
import pickle
from pathlib import Path
import numpy as np
import torch
_spec = importlib.util.spec_from_file_location('iemocap_context', Path(__file__).with_name('iemocap_context.py'))
IC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(IC)
BLOCKS = ('text', 'audio', 'video')

def load_meld_frozen(path: Path) -> dict:
    with open(path, 'rb') as fh:
        raw = pickle.load(fh)
    tr = [raw[s][d] for s in ('train', 'dev') for d in sorted(raw[s])]
    te = [raw['test'][d] for d in sorted(raw['test'])]
    blocks = [b for b in BLOCKS if b in tr[0]]
    maxlen = max((len(d['labels']) for d in tr + te))

    def pack(dias):
        n = len(dias)
        y = np.zeros((n, maxlen), dtype=np.int64)
        m = np.zeros((n, maxlen), dtype=np.float32)
        s = np.zeros((n, maxlen), dtype=np.int64)
        feats = {b: np.zeros((n, maxlen, dias[0][b].shape[1]), dtype=np.float32) for b in blocks}
        for i, d in enumerate(dias):
            L = len(d['labels'])
            y[i, :L] = d['labels']
            m[i, :L] = 1.0
            names = sorted(set(d['speakers']))
            s[i, :L] = [names.index(p) for p in d['speakers']]
            for b in blocks:
                feats[b][i, :L] = d[b]
        return (feats, y, m, s)
    f_tr, y_tr, m_tr, s_tr = pack(tr)
    f_te, y_te, m_te, s_te = pack(te)
    return {'name': 'MELD', 'feats': {b: (f_tr[b], f_te[b]) for b in blocks}, 'y': (y_tr, y_te), 'mask': (m_tr, m_te), 'speaker': (s_tr, s_te), 'num_classes': len(raw['meta']['emotions']), 'n_dialogues': (len(tr), len(te)), 'meta': raw['meta']}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', type=Path, required=True)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--out', type=Path, default=Path('results/meld_context.json'))
    args = ap.parse_args()
    torch.set_num_threads(4)
    seeds = list(range(args.seeds))
    data = load_meld_frozen(args.features)
    n_tr = int(data['mask'][0].sum())
    n_te = int(data['mask'][1].sum())
    print(f"MELD: {data['n_dialogues'][0]}/{data['n_dialogues'][1]} dialogues, {n_tr}/{n_te} utterances, {data['num_classes']} classes")
    print(f"blocks: { {k: v[0].shape[-1] for k, v in data['feats'].items()}}  ({data['meta']})\n")
    print('Label-only reference predictors (no words, no audio, no video):')
    refs = IC.label_only_references(data)
    print(f"{'':>40} {'acc':>7} {'wF1':>7} {'macroF1':>8} {'kappa':>7}")
    for name, m in refs.items():
        print(f"{name:>40} {m['acc'] * 100:7.2f} {m['w_f1'] * 100:7.2f} {m['macro_f1'] * 100:8.2f} {m['kappa']:7.4f}")
    print()
    cells = IC.run_grid(data, seeds)
    contr = IC.contrasts(cells)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'references': refs, 'cells': cells, 'contrasts': contr, 'meta': data['meta'], 'n_dialogues': data['n_dialogues'], 'n_utterances': [n_tr, n_te]}, indent=2))
    print(f'\nwritten to {args.out}')
if __name__ == '__main__':
    main()
