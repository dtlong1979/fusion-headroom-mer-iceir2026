from __future__ import annotations
import argparse
import json
import pickle
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from uacmd.metrics import classification_metrics, majority_class_floor
MODALITIES = ('text', 'audio', 'video')

def load_mosi(root: Path) -> dict:
    feats, labels, masks = ({}, {}, {})
    for m in MODALITIES:
        with open(root / f'{m}_2way.pickle', 'rb') as fh:
            tr_x, tr_y, te_x, te_y, _maxlen, tr_len, te_len = pickle.load(fh, encoding='latin1')
        feats[m] = (tr_x.astype(np.float32), te_x.astype(np.float32))
        y = (tr_y.astype(np.int64), te_y.astype(np.int64))
        mk = tuple((np.array([[1.0 if j < L else 0.0 for j in range(x.shape[1])] for L in lens], dtype=np.float32) for x, lens in ((tr_x, tr_len), (te_x, te_len))))
        if labels:
            assert all((np.array_equal(a, b) for a, b in zip(labels['y'], y)))
            assert all((np.array_equal(a, b) for a, b in zip(masks['m'], mk)))
        labels['y'], masks['m'] = (y, mk)
    return {'feats': feats, 'y': labels['y'], 'mask': masks['m'], 'num_classes': 2}

def load_mosei(pickle_or_zip: Path) -> dict:
    if pickle_or_zip.suffix == '.zip':
        with zipfile.ZipFile(pickle_or_zip) as zf:
            name = [n for n in zf.namelist() if n.endswith('.pickle')][0]
            with zf.open(name) as fh:
                d = pickle.load(fh, encoding='latin1')
    else:
        with open(pickle_or_zip, 'rb') as fh:
            d = pickle.load(fh, encoding='latin1')
    feats = {m: (d[f'{m}_train'].astype(np.float32), d[f'{m}_test'].astype(np.float32)) for m in MODALITIES}
    y = tuple((d[f'{s}_label'].argmax(-1).astype(np.int64) for s in ('train', 'test')))
    mask = tuple((d[f'{s}_mask'].astype(np.float32) for s in ('train', 'test')))
    return {'feats': feats, 'y': y, 'mask': mask, 'num_classes': 3}

def assemble(data: dict, use: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    tr = np.concatenate([data['feats'][m][0] for m in use], axis=-1)
    te = np.concatenate([data['feats'][m][1] for m in use], axis=-1)
    return (tr, te)

def standardise(tr: np.ndarray, te: np.ndarray, mask_tr: np.ndarray):
    valid = tr[mask_tr > 0]
    mu, sd = (valid.mean(0), valid.std(0))
    sd[sd < 1e-06] = 1.0
    return ((tr - mu) / sd, (te - mu) / sd)

class ContextLSTM(nn.Module):

    def __init__(self, in_dim: int, num_classes: int, hidden: int=100, dropout: float=0.5):
        super().__init__()
        self.drop_in = nn.Dropout(dropout)
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
        self.drop_out = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, num_classes)

    def forward(self, x, lengths):
        z = self.drop_in(x)
        packed = nn.utils.rnn.pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        return self.head(self.drop_out(h))

def masked_ce(logits, y, mask):
    ll = torch.log_softmax(logits, dim=-1)
    picked = ll.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return -(picked * mask).sum() / mask.sum()

@torch.no_grad()
def predict(model, x, mask):
    model.eval()
    probs = []
    lengths = mask.sum(1).long().clamp(min=1)
    for i in range(0, len(x), 64):
        logits = model(x[i:i + 64], lengths[i:i + 64])
        probs.append(torch.softmax(logits, dim=-1))
    return torch.cat(probs)[mask > 0]

def fit(x_tr, y_tr, m_tr, num_classes, seed, epochs=50, patience=8, lr=0.001, hidden=100, dropout=0.5, val_frac=0.2):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(x_tr)
    order = np.random.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    val_idx, fit_idx = (order[:n_val], order[n_val:])
    model = ContextLSTM(x_tr.shape[-1], num_classes, hidden=hidden, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lens_tr = m_tr.sum(1).long().clamp(min=1)
    xf, yf, mf = (x_tr[fit_idx], y_tr[fit_idx], m_tr[fit_idx])
    lf = lens_tr[fit_idx]
    xv, yv, mv = (x_tr[val_idx], y_tr[val_idx], m_tr[val_idx])
    best, best_state, bad = (-np.inf, None, 0)
    for _ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xf))
        for i in range(0, len(xf), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            loss = masked_ce(model(xf[b], lf[b]), yf[b], mf[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        pv = predict(model, xv, mv).argmax(-1)
        acc = float((pv == yv[mv > 0]).float().mean())
        if acc > best:
            best, bad = (acc, 0)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return (model, best, val_idx)

def run_one(x_tr, y_tr, m_tr, x_te, y_te, m_te, num_classes, seed, **train_kw):
    model, best, _val_idx = fit(x_tr, y_tr, m_tr, num_classes, seed, **train_kw)
    if x_te is None:
        return (None, None, best)
    probs = predict(model, x_te, m_te)
    scores = classification_metrics(y_te[m_te > 0], probs.argmax(-1), num_classes=num_classes)
    return (scores, probs, best)
GRID = [(h, d) for h in (50, 100, 200) for d in (0.3, 0.5)]

def select_hparams(x_tr, y_tr, m_tr, num_classes, seeds, verbose_tag=''):
    best, best_hp = (-np.inf, GRID[0])
    for hidden, dropout in GRID:
        vals = []
        for s in seeds:
            _sc, _pr, val = run_one(x_tr, y_tr, m_tr, None, None, None, num_classes, s, hidden=hidden, dropout=dropout)
            vals.append(val)
        mean_val = float(np.mean(vals))
        if verbose_tag:
            print(f'    [{verbose_tag}] hidden={hidden} dropout={dropout} val={mean_val * 100:.2f}', flush=True)
        if mean_val > best:
            best, best_hp = (mean_val, (hidden, dropout))
    return (best_hp, best)

def headroom(runs: dict, fusion_labels, metric: str, modalities=MODALITIES) -> dict:
    val = lambda k: runs[k]['validation'][metric]['mean']
    tst = lambda k: runs[k][metric]['mean']
    best_uni = max(modalities, key=val)
    best_fus = max(fusion_labels, key=val)
    return {'selected_on': 'validation', 'best_unimodal': best_uni, 'best_unimodal_value': tst(best_uni), 'best_fusion': best_fus, 'fused_value': tst(best_fus), 'H': tst(best_fus) - tst(best_uni), 'H_if_selected_on_test': max(map(tst, fusion_labels)) - max(map(tst, modalities))}

def aggregate(scores: list[dict]) -> dict:
    return {k: {'mean': float(np.mean([s[k] for s in scores])), 'std': float(np.std([s[k] for s in scores], ddof=1)) if len(scores) > 1 else 0.0, 'runs': [float(s[k]) for s in scores]} for k in scores[0]}

def evaluate_corpus(name: str, data: dict, seeds: list[int], sel_seeds: list[int], grid: bool) -> dict:
    y_tr = torch.from_numpy(data['y'][0])
    y_te = torch.from_numpy(data['y'][1])
    m_tr = torch.from_numpy(data['mask'][0])
    m_te = torch.from_numpy(data['mask'][1])
    K = data['num_classes']
    y_flat = y_te[m_te > 0]
    floor = majority_class_floor(y_tr[m_tr > 0], y_flat, num_classes=K)
    out = {'corpus': name, 'num_classes': K, 'n_train_utterances': int((m_tr > 0).sum()), 'n_test_utterances': int((m_te > 0).sum()), 'majority_class_floor': floor, 'seeds': seeds, 'selection_seeds': sel_seeds if grid else [], 'grid': GRID if grid else [], 'runs': {}}
    uni_probs: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    val_idx_of: dict[int, np.ndarray] = {}
    modalities = tuple((m for m in MODALITIES if m in data['feats']))
    tag = '+'.join((m[0].upper() for m in modalities))
    configs = {m: (m,) for m in modalities}
    configs[f'fused-early ({tag})'] = modalities
    for label, use in configs.items():
        tr, te = assemble(data, use)
        tr, te = standardise(tr, te, data['mask'][0])
        x_tr, x_te = (torch.from_numpy(tr), torch.from_numpy(te))
        if grid:
            (hidden, dropout), val = select_hparams(x_tr, y_tr, m_tr, K, sel_seeds, verbose_tag=f'{name}/{label}')
        else:
            hidden, dropout, val = (100, 0.5, float('nan'))
        t0 = time.time()
        scores, val_scores, probs = ([], [], {})
        for s in seeds:
            model, _best, val_idx = fit(x_tr, y_tr, m_tr, K, s, hidden=hidden, dropout=dropout)
            assert np.array_equal(val_idx_of.setdefault(s, val_idx), val_idx)
            pr = predict(model, x_te, m_te)
            vpr = predict(model, x_tr[val_idx], m_tr[val_idx])
            scores.append(classification_metrics(y_flat, pr.argmax(-1), num_classes=K))
            val_scores.append(classification_metrics(y_tr[val_idx][m_tr[val_idx] > 0], vpr.argmax(-1), num_classes=K))
            probs[s] = (pr, vpr)
        if len(use) == 1:
            uni_probs[use[0]] = probs
        agg = aggregate(scores)
        agg['validation'] = aggregate(val_scores)
        agg.update({'input_dim': int(tr.shape[-1]), 'hidden': hidden, 'dropout': dropout, 'selection_val_acc': val, 'seconds': round(time.time() - t0, 1)})
        out['runs'][label] = agg
        print(f"  {name:6s} {label:20s} dim={agg['input_dim']:4d} h={hidden} p={dropout} acc={agg['acc']['mean'] * 100:.2f}+-{agg['acc']['std'] * 100:.2f} wF1={agg['w_f1']['mean'] * 100:.2f} kappa={agg['kappa']['mean']:.3f} ({agg['seconds']}s)", flush=True)
    late, late_val = ([], [])
    for s in seeds:
        vi = val_idx_of[s]
        avg = sum((uni_probs[m][s][0] for m in modalities)) / len(modalities)
        vavg = sum((uni_probs[m][s][1] for m in modalities)) / len(modalities)
        late.append(classification_metrics(y_flat, avg.argmax(-1), num_classes=K))
        late_val.append(classification_metrics(y_tr[vi][m_tr[vi] > 0], vavg.argmax(-1), num_classes=K))
    late_label = f'fused-late ({tag})'
    out['runs'][late_label] = aggregate(late)
    out['runs'][late_label]['validation'] = aggregate(late_val)
    print(f"  {name:6s} {late_label:20s} acc={out['runs'][late_label]['acc']['mean'] * 100:.2f} wF1={out['runs'][late_label]['w_f1']['mean'] * 100:.2f}", flush=True)
    fusion_labels = [f'fused-early ({tag})', late_label]
    for metric in ('acc', 'w_f1', 'macro_f1', 'kappa'):
        out.setdefault('headroom', {})[metric] = headroom(out['runs'], fusion_labels, metric, modalities)
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mosi-root', type=Path, required=False)
    ap.add_argument('--mosei-file', type=Path, required=False)
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--grid', action='store_true')
    ap.add_argument('--selection-seeds', type=int, default=2)
    ap.add_argument('--out', type=Path, default=Path('results/mosi_mosei_headroom.json'))
    args = ap.parse_args()
    torch.set_num_threads(4)
    seeds = list(range(args.seeds))
    results = []
    sel_seeds = list(range(100, 100 + args.selection_seeds))
    if args.mosi_root:
        results.append(evaluate_corpus('MOSI', load_mosi(args.mosi_root), seeds, sel_seeds, args.grid))
    if args.mosei_file:
        results.append(evaluate_corpus('MOSEI', load_mosei(args.mosei_file), seeds, sel_seeds, args.grid))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    for r in results:
        print(f"\n{r['corpus']}: floor acc={r['majority_class_floor']['acc'] * 100:.2f} wF1={r['majority_class_floor']['w_f1'] * 100:.2f}")
        for metric, h in r['headroom'].items():
            scale = 100 if metric != 'kappa' else 1
            print(f"  H[{metric}] = {h['H'] * scale:+.2f}  ({h['best_fusion']} {h['fused_value'] * scale:.2f} - {h['best_unimodal']} {h['best_unimodal_value'] * scale:.2f})")
    print(f'\nwritten to {args.out}')
if __name__ == '__main__':
    main()
