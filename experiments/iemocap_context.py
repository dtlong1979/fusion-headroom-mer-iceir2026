from __future__ import annotations
import argparse
import importlib.util
import json
import pickle
from collections import Counter
from pathlib import Path
import numpy as np
import torch
_spec = importlib.util.spec_from_file_location('headroom_main', Path(__file__).with_name('mosi_mosei_headroom.py'))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
EMOTIONS = ('hap', 'sad', 'neu', 'ang', 'exc', 'fru')
BLOCKS = {'text': 'text', 'audio': 'audio', 'video': 'video'}

def load_iemocap(path: Path, text_features: Path | None=None) -> dict:
    with open(path, 'rb') as fh:
        ids, spk, lab, txt, aud, vis, _sent, tr, te = pickle.load(fh, encoding='latin1')
    if text_features is not None:
        with open(text_features, 'rb') as fh:
            txt = pickle.load(fh)
    tr, te = (sorted(tr), sorted(te))
    maxlen = max((len(lab[v]) for v in tr + te))

    def pack(vids):
        n = len(vids)
        y = np.zeros((n, maxlen), dtype=np.int64)
        m = np.zeros((n, maxlen), dtype=np.float32)
        s = np.zeros((n, maxlen), dtype=np.int64)
        feats = {k: np.zeros((n, maxlen, np.array(src[vids[0]]).shape[1]), dtype=np.float32) for k, src in (('text', txt), ('audio', aud), ('video', vis))}
        for i, v in enumerate(vids):
            L = len(lab[v])
            y[i, :L] = lab[v]
            m[i, :L] = 1.0
            spk_ids = sorted(set(spk[v]))
            s[i, :L] = [spk_ids.index(p) for p in spk[v]]
            for k, src in (('text', txt), ('audio', aud), ('video', vis)):
                feats[k][i, :L] = np.asarray(src[v], dtype=np.float32)
        return (feats, y, m, s)
    f_tr, y_tr, m_tr, s_tr = pack(tr)
    f_te, y_te, m_te, s_te = pack(te)
    return {'feats': {k: (f_tr[k], f_te[k]) for k in f_tr}, 'y': (y_tr, y_te), 'mask': (m_tr, m_te), 'speaker': (s_tr, s_te), 'num_classes': 6, 'n_dialogues': (len(tr), len(te))}

def label_only_references(data: dict) -> dict:
    from uacmd.metrics import classification_metrics
    y_tr, y_te = data['y']
    m_tr, m_te = data['mask']
    s_te = data['speaker'][1]
    maj = Counter(y_tr[m_tr > 0].tolist()).most_common(1)[0][0]
    ys, p_maj, p_prev, p_own = ([], [], [], [])
    for i in range(len(y_te)):
        last = {}
        for t in range(int(m_te[i].sum())):
            ys.append(y_te[i, t])
            p_maj.append(maj)
            p_prev.append(y_te[i, t - 1] if t > 0 else maj)
            p_own.append(y_te[i, last[s_te[i, t]]] if s_te[i, t] in last else maj)
            last[s_te[i, t]] = t
    yt = torch.tensor(ys)
    out = {}
    for name, p in (('constant (training majority)', p_maj), ('copy the previous turn', p_prev), ("copy this speaker's own previous turn", p_own)):
        out[name] = classification_metrics(yt, torch.tensor(p), num_classes=data['num_classes'])
    return out

def run_grid(data: dict, seeds: list[int]) -> dict:
    y_tr, y_te = (torch.from_numpy(a) for a in data['y'])
    m_tr, m_te = (torch.from_numpy(a) for a in data['mask'])
    cells = {}
    blocks = tuple(data['feats'])
    grid = [('text', ('text',))] + ([('fused', blocks)] if len(blocks) > 1 else [])
    for cell, use in grid:
        for ctx in (False, True):
            tr = np.concatenate([data['feats'][k][0] for k in use], axis=-1)
            te = np.concatenate([data['feats'][k][1] for k in use], axis=-1)
            tr, te = H.standardise(tr, te, data['mask'][0])
            x_tr, x_te = (torch.from_numpy(tr), torch.from_numpy(te))
            a_tr, b_tr, c_tr = (x_tr, y_tr, m_tr)
            a_te, b_te, c_te = (x_te, y_te, m_te)
            if not ctx:
                sel = m_tr > 0
                a_tr, b_tr = (x_tr[sel].unsqueeze(1), y_tr[sel].unsqueeze(1))
                c_tr = torch.ones_like(b_tr, dtype=m_tr.dtype)
                sel = m_te > 0
                a_te, b_te = (x_te[sel].unsqueeze(1), y_te[sel].unsqueeze(1))
                c_te = torch.ones_like(b_te, dtype=m_te.dtype)
            runs = [H.run_one(a_tr, b_tr, c_tr, a_te, b_te, c_te, data['num_classes'], s)[0] for s in seeds]
            agg = {k: {'mean': float(np.mean([r[k] for r in runs])), 'std': float(np.std([r[k] for r in runs], ddof=1)) if len(runs) > 1 else 0.0, 'runs': [float(r[k]) for r in runs]} for k in runs[0]}
            key = f"{cell}/{('context' if ctx else 'no-context')}"
            cells[key] = agg
            print(f"  {data.get('name', 'IEMOCAP')} {key:22s} acc={agg['acc']['mean'] * 100:6.2f} wF1={agg['w_f1']['mean'] * 100:6.2f} macroF1={agg['macro_f1']['mean'] * 100:6.2f} kappa={agg['kappa']['mean']:.3f}", flush=True)
    return cells

def contrasts(cells: dict) -> dict:
    from scipy import stats
    out = {}
    print('\nContrasts:')
    for metric in ('acc', 'w_f1', 'macro_f1', 'kappa'):
        sc = 100 if metric != 'kappa' else 1
        print(f'  -- {metric} --')
        for label, a_key, b_key in (('H_modality@no-context', 'fused/no-context', 'text/no-context'), ('H_modality@context', 'fused/context', 'text/context'), ('H_context@text', 'text/context', 'text/no-context'), ('H_context@fused', 'fused/context', 'fused/no-context')):
            if a_key not in cells or b_key not in cells:
                continue
            a = np.array(cells[a_key][metric]['runs'])
            b = np.array(cells[b_key][metric]['runs'])
            d = a - b
            t, p = stats.ttest_rel(a, b) if len(d) > 1 else (0, float('nan'))
            print(f'    {label:24s} {d.mean() * sc:+7.2f}   p={p:.4f}')
            out.setdefault(metric, {})[label] = {'diff': float(d.mean()), 'p': float(p)}
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pickle', type=Path, required=True)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--out', type=Path, default=Path('results/iemocap_context.json'))
    ap.add_argument('--text-features', type=Path, default=None)
    ap.add_argument('--blocks', default='text,audio,video')
    args = ap.parse_args()
    torch.set_num_threads(4)
    seeds = list(range(args.seeds))
    blocks = [b.strip() for b in args.blocks.split(',') if b.strip()]
    if 'text' not in blocks:
        ap.error('--blocks must include text')
    data = load_iemocap(args.pickle, args.text_features)
    data['feats'] = {k: data['feats'][k] for k in ('text', 'audio', 'video') if k in blocks}
    print(f"text features: {args.text_features or 'released CNN (label-fit)'}; fused blocks: {list(data['feats'])}")
    n_tr = int(data['mask'][0].sum())
    n_te = int(data['mask'][1].sum())
    print(f"IEMOCAP: {data['n_dialogues'][0]}/{data['n_dialogues'][1]} dialogues, {n_tr}/{n_te} utterances, 6 classes")
    dims = {k: v[0].shape[-1] for k, v in data['feats'].items()}
    print(f'feature dims: {dims}\n')
    print('Label-only reference predictors (no words, no audio, no video):')
    refs = label_only_references(data)
    print(f"{'':>40} {'acc':>7} {'wF1':>7} {'macroF1':>8} {'kappa':>7}")
    for name, m in refs.items():
        print(f"{name:>40} {m['acc'] * 100:7.2f} {m['w_f1'] * 100:7.2f} {m['macro_f1'] * 100:8.2f} {m['kappa']:7.4f}")
    print()
    cells = run_grid(data, seeds)
    contr = contrasts(cells)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'references': refs, 'cells': cells, 'contrasts': contr, 'text_features': str(args.text_features or 'released'), 'blocks': list(data['feats']), 'n_dialogues': data['n_dialogues'], 'n_utterances': [n_tr, n_te]}, indent=2))
    print(f'\nwritten to {args.out}')
if __name__ == '__main__':
    main()
