import json
import numpy as np
from scipy import stats
RESULTS = 'experiments/results'

def per_seed(runs, method, metric, scen):
    return np.array([np.mean([r['methods'][method]['test'][s][metric] for s in scen]) for r in runs])

def main() -> None:
    ten = [r for f in ('synth_main.json', 'synth_main2.json') for r in json.load(open(f'{RESULTS}/{f}'))['runs']]
    print(f'ten-seed visual-only ({len(ten)} seeds): weighted F1 / macro F1')
    for m in ('uacmd', 'early_fusion', 'kd_jsd', 'attention_gate_md', 'mean_gate_md', 'early_fusion_md'):
        w = np.mean([r['methods'][m]['test']['visual_only']['w_f1'] for r in ten]) * 100
        mf = np.mean([r['methods'][m]['test']['visual_only']['macro_f1'] for r in ten]) * 100
        print(f'  {m:18s} {w:6.2f} {mf:6.2f}')
    print()
    d = json.load(open(f'{RESULTS}/synth_kappa.json'))
    runs, scenarios = (d['runs'], d['scenarios'])
    for base in ('attention_gate_md', 'kd_jsd', 'early_fusion_md', 'abl_no_align'):
        for label, scen in (('all7', scenarios), ('visual_only', ['visual_only'])):
            out = []
            for k in ('w_f1', 'macro_f1', 'kappa', 'mcc'):
                a, b = (per_seed(runs, 'uacmd', k, scen), per_seed(runs, base, k, scen))
                out.append(f'{k}={np.mean(a - b) * 100:+.2f}(p={stats.ttest_rel(a, b).pvalue:.4f})')
            print(f'{base:18s} {label:12s} ' + ' '.join(out))
if __name__ == '__main__':
    main()
