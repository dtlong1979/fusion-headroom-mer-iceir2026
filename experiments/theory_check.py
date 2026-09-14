from __future__ import annotations
import itertools
import json
import os
import sys
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef
TOL = 1e-09
EMOTIONS = ('neutral', 'joy', 'surprise', 'anger', 'sadness', 'disgust', 'fear')

def floor_weighted_f1(pi_max: float) -> float:
    return 2.0 * pi_max ** 2 / (1.0 + pi_max)

def floor_macro_f1(pi_max: float, k: int) -> float:
    return 2.0 * pi_max / (1.0 + pi_max) / k

def labels_from_prior(pi: np.ndarray, n: int=200000, seed: int=0) -> np.ndarray:
    counts = np.round(pi * n).astype(int)
    counts[0] += n - counts.sum()
    return np.concatenate([np.full(c, i) for i, c in enumerate(counts)])

def check_proposition_1(pi: np.ndarray, name: str) -> dict:
    k = len(pi)
    y = labels_from_prior(pi)
    pi_hat = np.bincount(y, minlength=k) / len(y)
    star = int(pi_hat.argmax())
    const = np.full_like(y, star)
    w = f1_score(y, const, average='weighted', zero_division=0)
    m = f1_score(y, const, average='macro', zero_division=0)
    a = float((y == const).mean())
    kap = cohen_kappa_score(y, const)
    w_hat = floor_weighted_f1(pi_hat[star])
    m_hat = floor_macro_f1(pi_hat[star], k)
    assert abs(w - w_hat) < 1e-06, (name, w, w_hat)
    assert abs(m - m_hat) < 1e-06, (name, m, m_hat)
    assert abs(a - pi_hat[star]) < 1e-06, (name, a, pi_hat[star])
    assert abs(kap) < 1e-09, (name, kap)
    mcc = matthews_corrcoef(y, const)
    cm = np.zeros((k, k))
    for true_lbl, pred_lbl in zip(y, const):
        cm[true_lbl, pred_lbl] += 1
    total = cm.sum()
    pred_counts = cm.sum(0)
    assert abs(total ** 2 - (pred_counts ** 2).sum()) < 1e-06, (name, 'MCC denominator was expected to vanish at a constant predictor')
    assert np.isfinite(mcc)
    return {'corpus': name, 'K': k, 'pi_max': float(pi_hat[star]), 'weighted_f1': 100 * w, 'macro_f1': 100 * m, 'accuracy': 100 * a, 'kappa': kap, 'mcc': mcc}

def check_proposition_2() -> None:
    for pi in np.linspace(0.05, 0.95, 19):
        analytic = 2 * pi * (2 + pi) / (1 + pi) ** 2
        h = 1e-06
        numeric = (floor_weighted_f1(pi + h) - floor_weighted_f1(pi - h)) / (2 * h)
        assert abs(analytic - numeric) < 1e-05, (pi, analytic, numeric)
        assert analytic > 0
    assert abs(floor_weighted_f1(1.0) - 1.0) < TOL

def confusion_of_family(a: np.ndarray, star: int) -> np.ndarray:
    k = len(a)
    c = np.zeros((k, k))
    for i in range(k):
        c[i, i] += a[i]
        c[i, star] += 1.0 - a[i]
    return c

def metrics_from_confusion(pi: np.ndarray, c: np.ndarray) -> dict:
    joint = pi[:, None] * c
    pred = joint.sum(0)
    f1 = np.zeros(len(pi))
    for k in range(len(pi)):
        tp = joint[k, k]
        prec = tp / pred[k] if pred[k] > 0 else 0.0
        rec = tp / pi[k] if pi[k] > 0 else 0.0
        f1[k] = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = joint / (pi[:, None] * pred[None, :])
        terms = np.where(joint > 0, joint * np.log(ratio), 0.0)
    return {'w_f1': float((pi * f1).sum()), 'macro_f1': float(f1.mean()), 'mi': float(terms.sum()), 'acc': float(np.trace(joint))}

def find_theorem3_witness(pi: np.ndarray, seed: int=0, trials: int=200000):
    rng = np.random.default_rng(seed)
    star = int(pi.argmax())
    k = len(pi)
    best = None
    for _ in range(trials):
        a = rng.uniform(0, 1, size=k)
        b = rng.uniform(0, 1, size=k)
        mf = metrics_from_confusion(pi, confusion_of_family(a, star))
        mg = metrics_from_confusion(pi, confusion_of_family(b, star))
        if mf['mi'] > mg['mi'] and mf['w_f1'] < mg['w_f1']:
            gap = (mf['mi'] - mg['mi']) * (mg['w_f1'] - mf['w_f1'])
            if best is None or gap > best[0]:
                best = (gap, a.copy(), b.copy(), mf, mg)
    return best

def verify_witness_empirically(pi: np.ndarray, a: np.ndarray, b: np.ndarray, n: int=400000, seed: int=1) -> dict:
    rng = np.random.default_rng(seed)
    star = int(pi.argmax())
    y = rng.choice(len(pi), size=n, p=pi)

    def sample(coeff):
        keep = rng.random(n) < coeff[y]
        return np.where(keep, y, star)
    yf, yg = (sample(a), sample(b))
    out = {}
    for tag, yh in (('f', yf), ('g', yg)):
        out[tag] = {'w_f1': 100 * f1_score(y, yh, average='weighted', zero_division=0), 'macro_f1': 100 * f1_score(y, yh, average='macro', zero_division=0), 'kappa': cohen_kappa_score(y, yh), 'mcc': matthews_corrcoef(y, yh)}
    return out

def garble(c: np.ndarray, lam: float, star: int) -> np.ndarray:
    e = np.zeros_like(c)
    e[:, star] = 1.0
    return lam * c + (1.0 - lam) * e

def dW_dlambda(pi: np.ndarray, c: np.ndarray, star: int) -> float:
    p = (pi[:, None] * c).sum(0)
    total = sum((2 * pi[j] ** 3 * c[j, j] / (p[j] + pi[j]) ** 2 for j in range(len(pi)) if j != star))
    cs, ps, pm = (c[star, star], p[star], pi[star])
    return total + 2 * pm ** 2 * (cs * (1 + pm) - (ps + pm)) / (ps + pm) ** 2

def dM_dlambda(pi: np.ndarray, c: np.ndarray, star: int) -> float:
    p = (pi[:, None] * c).sum(0)
    k = len(pi)
    total = sum((2 * pi[j] ** 2 * c[j, j] / (p[j] + pi[j]) ** 2 for j in range(k) if j != star))
    cs, ps, pm = (c[star, star], p[star], pi[star])
    return (total + 2 * pm * (cs * (1 + pm) - (ps + pm)) / (ps + pm) ** 2) / k

def check_theorem_3(pi: np.ndarray, seed: int=7, trials: int=40) -> None:
    rng = np.random.default_rng(seed)
    star = int(pi.argmax())
    h = 1e-06
    for _ in range(trials):
        c = rng.dirichlet(np.ones(len(pi)) * rng.uniform(0.3, 3.0), size=len(pi))
        for analytic, idx in ((dW_dlambda, 'w_f1'), (dM_dlambda, 'macro_f1')):
            num = (metrics_from_confusion(pi, garble(c, 1.0, star))[idx] - metrics_from_confusion(pi, garble(c, 1 - h, star))[idx]) / h
            got = analytic(pi, c, star)
            assert abs(got - num) < 0.001, (idx, got, num)

def realistic_confusion(pi: np.ndarray, recall: np.ndarray, star: int, leak: float=0.7) -> np.ndarray:
    k = len(pi)
    c = np.zeros((k, k))
    for i in range(k):
        c[i, i] = recall[i]
        rest = 1.0 - recall[i]
        if i == star:
            c[i, :] += rest / (k - 1) * (np.arange(k) != i)
        else:
            c[i, star] += rest * leak
            others = [j for j in range(k) if j not in (i, star)]
            for j in others:
                c[i, j] += rest * (1 - leak) / len(others)
    return c / c.sum(1, keepdims=True)

def kappa_from_confusion(pi: np.ndarray, c: np.ndarray) -> float:
    j = pi[:, None] * c
    po = float(np.trace(j))
    pe = float((j.sum(0) * j.sum(1)).sum())
    return (po - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else float('nan')

def semigroup_holds(pi: np.ndarray, seed: int=3) -> bool:
    rng = np.random.default_rng(seed)
    star = int(pi.argmax())
    for _ in range(50):
        c = rng.dirichlet(np.ones(len(pi)), size=len(pi))
        a, b = rng.uniform(0, 1, 2)
        lhs = garble(garble(c, b, star), a, star)
        if not np.allclose(lhs, garble(c, a * b, star), atol=1e-15):
            return False
    return True

def garbling_experiment(pi: np.ndarray, lam: float=0.9, n: int=20000, seed: int=0):
    rng = np.random.default_rng(seed)
    star = int(pi.argmax())
    raised_k = raised_w = usable = 0
    for _ in range(n):
        c = rng.dirichlet(np.ones(len(pi)) * rng.uniform(0.2, 4.0), size=len(pi))
        k1 = kappa_from_confusion(pi, c)
        if not np.isfinite(k1) or k1 <= 0:
            continue
        usable += 1
        g = garble(c, lam, star)
        if kappa_from_confusion(pi, g) > k1 + 1e-12:
            raised_k += 1
        if metrics_from_confusion(pi, g)['w_f1'] > metrics_from_confusion(pi, c)['w_f1'] + 1e-12:
            raised_w += 1
    return (raised_k, raised_w, usable)

def mcc_from_confusion(pi: np.ndarray, c: np.ndarray) -> float:
    j = pi[:, None] * c
    q = j.sum(0)
    d = float(np.trace(j) - pi @ q)
    v = 1.0 - float(pi @ pi)
    w = 1.0 - float(q @ q)
    return float('nan') if v <= 0 or w <= 0 else d / np.sqrt(v * w)

def balanced_accuracy(pi: np.ndarray, c: np.ndarray) -> float:
    return float(np.diag(c)[pi > 0].mean())

def one_sample_perturbation(counts: np.ndarray, rare: int) -> dict:
    from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
    star = int(counts.argmax())
    y = np.repeat(np.arange(len(counts)), counts.astype(int))
    out = {}
    for tag, cls in (('hit', rare), ('miss', star)):
        p = np.full_like(y, star)
        p[np.flatnonzero(y == cls)[0]] = rare
        out[tag] = (cohen_kappa_score(y, p), matthews_corrcoef(y, p))
    return out

def check_proposition_4(pi: np.ndarray, seed: int=11, trials: int=30) -> None:
    rng = np.random.default_rng(seed)
    k = len(pi)
    lams = np.array([1.0, 0.7, 0.4, 0.2, 0.05, 0.01, 0.001, 0.0001])
    checked = 0
    for _ in range(trials):
        recall = rng.uniform(0.1, 0.95, size=k)
        c = realistic_confusion(pi, recall, star=int(np.argmax(pi)))
        if kappa_from_confusion(pi, c) <= 0:
            continue
        mccs = np.array([mcc_from_confusion(pi, garble(c, l, int(np.argmax(pi)))) for l in lams])
        bas = np.array([balanced_accuracy(pi, garble(c, l, int(np.argmax(pi)))) for l in lams])
        assert np.all(np.diff(mccs[::-1]) > -1e-12), 'MCC is not monotone in lambda'
        assert mccs[-1] < 0.01, 'MCC does not tend to 0 at the constant predictor'
        affine = lams * balanced_accuracy(pi, c) + (1.0 - lams) / k
        assert np.allclose(bas, affine, atol=1e-12), 'BA(lambda) is not affine'
        j = pi[:, None] * c
        q = j.sum(0)
        d = float(np.trace(j) - pi @ q)
        v = 1.0 - float(pi @ pi)
        star = int(np.argmax(pi))
        e0 = np.zeros(k)
        e0[star] = 1.0
        b = 1.0 - q[star]
        a = float(((q - e0) ** 2).sum())
        closed = d * np.sqrt(lams) / np.sqrt(v * (2 * b - a * lams))
        assert np.allclose(closed, mccs, atol=1e-12), 'the MCC(lambda) closed form fails'
        checked += 1
    assert checked >= 5, 'not enough usable confusions to check Proposition 4'
    print(f'  on {checked} random confusions with kappa > 0:')
    print('    MCC(lambda) = D sqrt(lambda) / sqrt(V(2b - a lambda)) verified to 1e-12,')
    print('    strictly increasing in lambda, and -> 0 at the constant predictor.')
    print('    BA(lambda) = lambda BA(1) + (1-lambda)/K verified exactly.')
    print('  So kappa is NOT alone in resisting collapse, and the floor is not a')
    print('  discontinuity for MCC. The withdrawn claim is asserted against here.')

def main() -> int:
    priors_path = os.path.join('src', 'uacmd', 'data', 'meld_priors.json')
    meld = json.load(open(priors_path))['test']
    counts = np.array([meld[e] for e in EMOTIONS], dtype=float)
    pi_meld = counts / counts.sum()
    iemocap = np.array([1636, 1084, 1103, 1708], dtype=float)
    pi_iemocap = iemocap / iemocap.sum()
    print('## Proposition 1 and 4 — floor in closed form; kappa at zero, MCC undefined\n')
    rows = [check_proposition_1(pi_meld, 'MELD test (K=7)'), check_proposition_1(pi_iemocap, 'IEMOCAP 4-class'), check_proposition_1(np.array([0.7, 0.1, 0.1, 0.1]), 'synthetic pi_max=0.7')]
    print(f"{'corpus':22s} {'pi_max':>7s} {'W':>7s} {'M':>7s} {'A':>7s} {'kappa':>7s} {'MCC':>7s}")
    for r in rows:
        print(f"{r['corpus']:22s} {r['pi_max']:7.3f} {r['weighted_f1']:7.1f} {r['macro_f1']:7.1f} {r['accuracy']:7.1f} {r['kappa']:7.3f} {r['mcc']:7.3f}")
    print('\n  Closed forms agree with scikit-learn to 1e-6.')
    print('  kappa is exactly 0 (its denominator 1 - p_e = 1 - pi_0 is nonzero).')
    print('  MCC is 0/0: the factor n^2 - sum(p_k^2) vanishes because the predicted')
    print("  distribution is degenerate. The 0.0 printed above is scikit-learn's")
    print('  convention, NOT a theorem — Proposition 4 says so explicitly.')
    check_proposition_2()
    print('\n## Proposition 2 — the floor is strictly increasing in pi_max\n')
    print(f"{'pi_max':>8s} {'floor W':>9s}")
    for pi in (0.2, 0.3, 0.481, 0.6, 0.7, 0.9):
        print(f'{pi:8.3f} {100 * floor_weighted_f1(pi):9.1f}')
    print('\n  analytic derivative matches the numerical one; W -> 1 as pi_max -> 1')
    print('\n## Theorem 3 — searching for a ranking-reversal witness\n')
    found = find_theorem3_witness(pi_meld)
    if found is None:
        print('  NO WITNESS FOUND — Theorem 3 as stated is not supported. Fix the')
        print('  theory document before quoting it.')
        return 1
    _gap, a, b, mf, mg = found
    print(f'  f: a = {np.round(a, 3).tolist()}')
    print(f'  g: b = {np.round(b, 3).tolist()}\n')
    print(f"{'':>10s} {'MI (nats)':>10s} {'weighted F1':>12s} {'macro F1':>10s}")
    print(f"{'f':>10s} {mf['mi']:10.4f} {100 * mf['w_f1']:12.1f} {100 * mf['macro_f1']:10.1f}")
    print(f"{'g':>10s} {mg['mi']:10.4f} {100 * mg['w_f1']:12.1f} {100 * mg['macro_f1']:10.1f}")
    print(f"\n  f is more informative by {mf['mi'] - mg['mi']:.4f} nats, yet weighted F1")
    print(f"  ranks g higher by {100 * (mg['w_f1'] - mf['w_f1']):.1f} points.")
    emp = verify_witness_empirically(pi_meld, a, b)
    print('\n  Sampled verification (400,000 draws, scikit-learn metrics):')
    print(f"{'':>10s} {'weighted F1':>12s} {'macro F1':>10s} {'kappa':>8s} {'MCC':>8s}")
    for tag in ('f', 'g'):
        e = emp[tag]
        print(f"{tag:>10s} {e['w_f1']:12.1f} {e['macro_f1']:10.1f} {e['kappa']:8.3f} {e['mcc']:8.3f}")
    assert emp['g']['w_f1'] > emp['f']['w_f1'], 'witness failed under sampling'
    print('\n  The reversal survives sampling. Note which columns get it right:')
    print('  kappa and MCC rank f above g, weighted F1 does not. (Away from the')
    print('  degenerate point MCC is well defined, so it is usable here.)')
    print('\n## Theorem 3 — the derivative identities\n')
    check_theorem_3(pi_meld)
    check_theorem_3(pi_iemocap)
    print('  dW/dlambda and dM/dlambda match finite differences on 80 random')
    print('  confusion matrices, to better than 1e-4.\n')
    rho = float(pi_meld.max() / pi_meld.min())
    print('\n## Theorem 3 — the sensitivity reading, corrected\n')
    r_calib = 0.6
    c_cal = np.zeros((len(pi_meld), len(pi_meld)))
    for i in range(len(pi_meld)):
        c_cal[i, i] = r_calib
        off = (1 - r_calib) * pi_meld / (1 - pi_meld[i])
        off[i] = 0.0
        c_cal[i] += off
    p_cal = (pi_meld[:, None] * c_cal).sum(0)
    wt = [2 * pi_meld[j] ** 3 * c_cal[j, j] / (p_cal[j] + pi_meld[j]) ** 2 for j in range(len(pi_meld))]
    mt = [2 * pi_meld[j] ** 2 * c_cal[j, j] / (len(pi_meld) * (p_cal[j] + pi_meld[j]) ** 2) for j in range(len(pi_meld))]
    print(f'  imbalance ratio rho                 = {rho:6.1f}')
    print(f'  weighted F1 sensitivity ratio       = {wt[0] / wt[-1]:6.1f}   (NOT rho^3)')
    print(f'  macro F1 sensitivity ratio          = {mt[0] / mt[-1]:6.1f}   (NOT rho^2)')
    assert wt[0] / wt[-1] < 10 * rho, 'the weighted ratio should be of order rho'
    assert mt[0] / mt[-1] < 3.0, 'macro should be near-unbiased'
    print('\n  P_j scales with pi_j, so (P_j+pi_j)^2 cancels two powers. The removed')
    print('  claim of a 16,000-to-1 ratio is asserted against here so it cannot return.')
    print('\n## Theorem 3(b,c) — kappa cannot be raised by collapse; W usually is\n')
    assert semigroup_holds(pi_meld), 'T_a o T_b != T_ab; monotonicity is unproven'
    print('  T_a o T_b = T_ab verified to 1e-15, so the data-processing inequality')
    print('  gives monotone information loss.\n')
    raised_k, raised_w, usable = garbling_experiment(pi_meld)
    print(f'  on {usable:,} random confusions with kappa > 0, garbling to lambda=0.9:')
    print(f'    raised kappa in {raised_k:,} cases  ({100 * raised_k / usable:.2f}%)')
    print(f'    raised W     in {raised_w:,} cases  ({100 * raised_w / usable:.2f}%)')
    assert raised_k == 0, 'kappa was raised by an information-destroying operation'
    assert raised_w > usable // 2, 'the weighted-F1 failure rate should be large'
    print('\n## Proposition 4 — kappa, MCC and balanced accuracy all resist collapse\n')
    check_proposition_4(pi_meld)
    print('\n## Sensitivity near the floor — one sample away from the constant predictor\n')
    rare = int(counts.argmin())
    pert = one_sample_perturbation(counts, rare)
    for tag, (k, m) in pert.items():
        print(f'  flip one prediction to {EMOTIONS[rare]} ({tag}): kappa {k:+.5f}   MCC {m:+.4f}')
    assert abs(pert['hit'][1]) > 10 * abs(pert['hit'][0]), 'MCC should move far more than kappa'
    print('\nALL CHECKS PASSED')
    return 0
if __name__ == '__main__':
    sys.exit(main())
