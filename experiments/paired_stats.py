from __future__ import annotations
import itertools
import numpy as np
from scipy import stats

def paired(x, y, n_boot: int=10000, seed: int=0) -> dict:
    x, y = (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    d = x - y
    n = len(d)
    out = {'mean': float(d.mean()), 'n': int(n)}
    if n < 2:
        return out
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    crit = stats.t.ppf(0.975, n - 1)
    out['ci95_t'] = [float(d.mean() - crit * se), float(d.mean() + crit * se)]
    out['p_t'] = float(stats.ttest_rel(x, y).pvalue)
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    out['ci95_bootstrap'] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    if n <= 16:
        signs = np.array(list(itertools.product([1, -1], repeat=n)), dtype=float)
    else:
        signs = rng.choice([1.0, -1.0], size=(n_boot, n))
    null = (signs * d).mean(axis=1)
    out['p_permutation'] = float((np.abs(null) >= abs(d.mean()) - 1e-12).mean())
    out['dz'] = float(d.mean() / sd) if sd > 0 else float('nan')
    return out

def fmt(rec: dict, scale: float=100.0) -> str:
    if 'ci95_bootstrap' not in rec:
        return f"{rec['mean'] * scale:+.2f}"
    lo, hi = rec['ci95_bootstrap']
    return f"{rec['mean'] * scale:+.2f}  bootstrap 95% [{lo * scale:+.2f}, {hi * scale:+.2f}]  p_perm={rec['p_permutation']:.3f}  dz={rec['dz']:+.2f}"
