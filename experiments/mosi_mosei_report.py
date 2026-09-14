from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import stats
import importlib.util
_spec = importlib.util.spec_from_file_location('headroom_main', Path(__file__).with_name('mosi_mosei_headroom.py'))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
_pspec = importlib.util.spec_from_file_location('paired_stats', Path(__file__).with_name('paired_stats.py'))
P = importlib.util.module_from_spec(_pspec)
_pspec.loader.exec_module(P)
METRICS = ('acc', 'w_f1', 'macro_f1', 'kappa')

def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None

def report(corpus: dict, mid: dict | None, floors: dict) -> dict:
    name = corpus['corpus']
    runs = dict(corpus['runs'])
    if mid:
        runs['fused-mid (T+A+V)'] = mid
    fusion = [k for k in runs if k.startswith('fused')]
    print(f"\n### {name}  ({corpus['n_train_utterances']} train / {corpus['n_test_utterances']} test utterances, {corpus['num_classes']} classes)")
    fl = corpus['majority_class_floor']
    print(f"  floor (train-majority class {fl['majority_class']}): acc={fl['acc'] * 100:.2f} wF1={fl['w_f1'] * 100:.2f} macroF1={fl['macro_f1'] * 100:.2f} kappa={fl['kappa']:.3f}")
    if floors:
        best_c = max(floors['constant_predictors'], key=lambda c: floors['constant_predictors'][c]['acc'])
        bc = floors['constant_predictors'][best_c]
        print(f"  best constant predictor (class {best_c}): acc={bc['acc'] * 100:.2f} wF1={bc['w_f1'] * 100:.2f} kappa={bc['kappa']:.3f}")
    print(f"  {'configuration':22s} {'acc':>14s} {'wF1':>14s} {'macroF1':>9s} {'kappa':>7s}")
    for k, v in runs.items():
        print(f"  {k:22s} {v['acc']['mean'] * 100:8.2f}+-{v['acc']['std'] * 100:4.2f} {v['w_f1']['mean'] * 100:8.2f}+-{v['w_f1']['std'] * 100:4.2f} {v['macro_f1']['mean'] * 100:9.2f} {v['kappa']['mean']:7.3f}")
    out = {'corpus': name, 'headroom': {}}
    for metric in METRICS:
        rec = H.headroom(runs, fusion, metric)
        best_uni, best_fus = (rec['best_unimodal'], rec['best_fusion'])
        a = np.array(runs[best_fus][metric]['runs'])
        b = np.array(runs[best_uni][metric]['runs'])
        if len(a) == len(b) and len(a) > 1:
            st = P.paired(a, b)
            rec.update({'p': st['p_t'], 'ci95': st['ci95_t'], 'ci95_bootstrap': st['ci95_bootstrap'], 'p_permutation': st['p_permutation'], 'dz': st['dz'], 'n_seeds': st['n']})
        out['headroom'][metric] = rec
        scale = 100 if metric != 'kappa' else 1
        ci = rec.get('ci95')
        ci_s = f"  95% CI [{ci[0] * scale:+.2f}, {ci[1] * scale:+.2f}], p={rec['p']:.3f}" if ci else ''
        print(f"  H[{metric}] = {rec['H'] * scale:+.2f}  ({best_fus} {rec['fused_value'] * scale:.2f} - {best_uni} {rec['best_unimodal_value'] * scale:.2f}){ci_s}  [if selected on test: {rec['H_if_selected_on_test'] * scale:+.2f}]")
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, default=Path('results'))
    ap.add_argument('--fixed', type=Path, default=None)
    ap.add_argument('--mid', type=Path, nargs='*', default=None)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    floors = load(args.results / 'mosi_mosei_floors.json') or {}
    if args.mid is not None:
        mid = {}
        for f in args.mid:
            mid.update(load(f) or {})
    else:
        mid = load(args.results / 'headroom_mosi_mosei_mid.json') or {}
    protocols = {'fixed': args.fixed} if args.fixed else {p: args.results / f'headroom_mosi_mosei_{p}.json' for p in ('fixed', 'grid')}
    summary = {}
    for protocol, path in protocols.items():
        data = load(path)
        if not data:
            continue
        print(f'\n## Protocol: {protocol}')
        summary[protocol] = [report(c, mid.get(c['corpus']) if protocol == 'fixed' else None, floors.get(c['corpus'], {})) for c in data]
    out = args.out or args.results / 'mosi_mosei_summary.json'
    out.write_text(json.dumps(summary, indent=2))
    print(f'\nwritten to {out}')
if __name__ == '__main__':
    main()
