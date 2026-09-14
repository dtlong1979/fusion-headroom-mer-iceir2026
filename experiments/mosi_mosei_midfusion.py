from __future__ import annotations
import argparse
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
_spec = importlib.util.spec_from_file_location('headroom_main', Path(__file__).with_name('mosi_mosei_headroom.py'))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

class MidFusionLSTM(nn.Module):

    def __init__(self, dims: list[int], num_classes: int, hidden: int=100, dropout: float=0.5):
        super().__init__()
        self.dims = dims
        self.drop_in = nn.Dropout(dropout)
        self.encoders = nn.ModuleList([nn.LSTM(d, hidden, batch_first=True, bidirectional=True) for d in dims])
        self.drop_out = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden * len(dims), num_classes)

    def forward(self, x, lengths):
        z = self.drop_in(x)
        outs, start = ([], 0)
        for d, enc in zip(self.dims, self.encoders):
            block = z[..., start:start + d]
            start += d
            packed = nn.utils.rnn.pack_padded_sequence(block, lengths.cpu(), batch_first=True, enforce_sorted=False)
            h, _ = enc(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
            outs.append(h)
        return self.head(self.drop_out(torch.cat(outs, dim=-1)))

def train_mid(x_tr, y_tr, m_tr, x_te, y_te, m_te, dims, num_classes, seed, epochs=50, patience=8, lr=0.001, hidden=100, dropout=0.5, val_frac=0.2):
    torch.manual_seed(seed)
    np.random.seed(seed)
    order = np.random.permutation(len(x_tr))
    n_val = max(1, int(round(val_frac * len(x_tr))))
    val_idx, fit_idx = (order[:n_val], order[n_val:])
    model = MidFusionLSTM(dims, num_classes, hidden=hidden, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lens = m_tr.sum(1).long().clamp(min=1)
    xf, yf, mf, lf = (x_tr[fit_idx], y_tr[fit_idx], m_tr[fit_idx], lens[fit_idx])
    xv, yv, mv = (x_tr[val_idx], y_tr[val_idx], m_tr[val_idx])
    best, best_state, bad = (-np.inf, None, 0)
    for _ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xf))
        for i in range(0, len(xf), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            H.masked_ce(model(xf[b], lf[b]), yf[b], mf[b]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        acc = float((H.predict(model, xv, mv).argmax(-1) == yv[mv > 0]).float().mean())
        if acc > best:
            best, bad, best_state = (acc, 0, {k: v.detach().clone() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    probs = H.predict(model, x_te, m_te)
    val_probs = H.predict(model, xv, mv)
    return (H.classification_metrics(y_te[m_te > 0], probs.argmax(-1), num_classes=num_classes), H.classification_metrics(yv[mv > 0], val_probs.argmax(-1), num_classes=num_classes))

def evaluate(name: str, data: dict, seeds: list[int], hidden: int, dropout: float):
    y_tr, y_te = (torch.from_numpy(a) for a in data['y'])
    m_tr, m_te = (torch.from_numpy(a) for a in data['mask'])
    modalities = tuple((m for m in H.MODALITIES if m in data['feats']))
    dims = [data['feats'][m][0].shape[-1] for m in modalities]
    tr, te = H.assemble(data, modalities)
    tr, te = H.standardise(tr, te, data['mask'][0])
    x_tr, x_te = (torch.from_numpy(tr), torch.from_numpy(te))
    scores, val_scores, t0 = ([], [], time.time())
    for s in seeds:
        sc, vsc = train_mid(x_tr, y_tr, m_tr, x_te, y_te, m_te, dims, data['num_classes'], s, hidden=hidden, dropout=dropout)
        scores.append(sc)
        val_scores.append(vsc)
    agg = H.aggregate(scores)
    agg['validation'] = H.aggregate(val_scores)
    agg.update({'input_dim': int(tr.shape[-1]), 'hidden': hidden, 'dropout': dropout, 'per_modality_dims': dims, 'seconds': round(time.time() - t0, 1)})
    print(f"  {name:6s} {'fused-mid (T+A+V)':20s} dims={dims} h={hidden} p={dropout} acc={agg['acc']['mean'] * 100:.2f}+-{agg['acc']['std'] * 100:.2f} wF1={agg['w_f1']['mean'] * 100:.2f} kappa={agg['kappa']['mean']:.3f} ({agg['seconds']}s)", flush=True)
    return agg

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mosi-root', type=Path)
    ap.add_argument('--mosei-file', type=Path)
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--hidden', type=int, default=100)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--out', type=Path, default=Path('results/headroom_mosi_mosei_mid.json'))
    args = ap.parse_args()
    torch.set_num_threads(4)
    seeds = list(range(args.seeds))
    out = {}
    if args.mosi_root:
        out['MOSI'] = evaluate('MOSI', H.load_mosi(args.mosi_root), seeds, args.hidden, args.dropout)
    if args.mosei_file:
        out['MOSEI'] = evaluate('MOSEI', H.load_mosei(args.mosei_file), seeds, args.hidden, args.dropout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f'written to {args.out}')
if __name__ == '__main__':
    main()
