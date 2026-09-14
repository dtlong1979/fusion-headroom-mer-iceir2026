from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from scipy import stats

def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
H = _load('headroom_main', 'mosi_mosei_headroom.py')
MF = _load('midfusion', 'mosi_mosei_midfusion.py')
MC = _load('meld_context', 'meld_context.py')
P = _load('paired_stats', 'paired_stats.py')
METRICS = ('acc', 'w_f1', 'macro_f1', 'kappa')

def paired(runs: dict, a: str, b: str, metric: str) -> dict:
    x, y = (np.array(runs[k][metric]['runs']) for k in (a, b))
    st = P.paired(x, y)
    return {'H': st['mean'], 'ci95': st['ci95_t'], 'ci95_bootstrap': st['ci95_bootstrap'], 'p': st['p_t'], 'p_permutation': st['p_permutation'], 'dz': st['dz'], 'n_seeds': st['n']}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', type=Path, required=True)
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--out', type=Path, default=Path('results/meld_headroom.json'))
    args = ap.parse_args()
    torch.set_num_threads(4)
    seeds = list(range(args.seeds))
    data = MC.load_meld_frozen(args.features)
    modalities = tuple((m for m in H.MODALITIES if m in data['feats']))
    tag = '+'.join((m[0].upper() for m in modalities))
    print(f"MELD: blocks {modalities}, {data['meta']}\n")
    res = H.evaluate_corpus('MELD', data, seeds, [], grid=False)
    runs = dict(res['runs'])
    runs[f'fused-mid ({tag})'] = MF.evaluate('MELD', data, seeds, 100, 0.5)
    fusion = [k for k in runs if k.startswith('fused')]
    print(f"\n{'configuration':22s} {'held-out acc':>12s} {'test acc':>9s} {'held-out wF1':>12s} {'test wF1':>9s}")
    for k, v in runs.items():
        print(f"{k:22s} {v['validation']['acc']['mean'] * 100:12.2f} {v['acc']['mean'] * 100:9.2f} {v['validation']['w_f1']['mean'] * 100:12.2f} {v['w_f1']['mean'] * 100:9.2f}")
    out = {'corpus': 'MELD', 'features': data['meta'], 'seeds': seeds, 'majority_class_floor': res['majority_class_floor'], 'runs': runs, 'headroom': {}, 'per_style': {}}
    for metric in METRICS:
        sc = 1 if metric == 'kappa' else 100
        rec = H.headroom(runs, fusion, metric, modalities)
        rec.update(paired(runs, rec['best_fusion'], rec['best_unimodal'], metric))
        out['headroom'][metric] = rec
        out['per_style'][metric] = {f: paired(runs, f, rec['best_unimodal'], metric) for f in fusion}
        print(f"\nH[{metric}] = {rec['H'] * sc:+.2f}  ({rec['best_fusion']} - {rec['best_unimodal']}, chosen on held-out)  95% CI [{rec['ci95'][0] * sc:+.2f}, {rec['ci95'][1] * sc:+.2f}], p={rec['p']:.3f}  [if selected on test: {rec['H_if_selected_on_test'] * sc:+.2f}]")
        for f, s in out['per_style'][metric].items():
            print(f"    {f:22s} {s['H'] * sc:+.2f}  p={s['p']:.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f'\nwritten to {args.out}')
if __name__ == '__main__':
    main()
