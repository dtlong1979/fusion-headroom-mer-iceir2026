from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef, roc_auc_score
__all__ = ['classification_metrics', 'expected_calibration_error', 'uncertainty_auroc', 'majority_class_floor']

def classification_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int | None=None) -> dict:
    yt = y_true.cpu().numpy()
    yp = y_pred.cpu().numpy()
    labels = list(range(num_classes)) if num_classes is not None else None
    return {'acc': float((yt == yp).mean()), 'w_f1': float(f1_score(yt, yp, average='weighted', labels=labels, zero_division=0)), 'macro_f1': float(f1_score(yt, yp, average='macro', labels=labels, zero_division=0)), 'kappa': float(cohen_kappa_score(yt, yp)), 'mcc': float(matthews_corrcoef(yt, yp))}

def expected_calibration_error(probs: torch.Tensor, y_true: torch.Tensor, n_bins: int=15) -> float:
    conf, pred = probs.max(dim=-1)
    correct = (pred == y_true).float()
    conf = conf.cpu().numpy()
    correct = correct.cpu().numpy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = (0.0, len(conf))
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.sum() == 0:
            continue
        ece += sel.sum() / n * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)

def uncertainty_auroc(uncertainty: torch.Tensor, is_corrupt: torch.Tensor) -> float:
    u = uncertainty.detach().cpu().numpy().ravel()
    c = is_corrupt.detach().cpu().numpy().ravel()
    if c.min() == c.max():
        return float('nan')
    return float(roc_auc_score(c, u))

def majority_class_floor(y_train: torch.Tensor, y_eval: torch.Tensor, num_classes: int | None=None) -> dict:
    majority = int(torch.bincount(y_train).argmax())
    const = torch.full_like(y_eval, majority)
    out = classification_metrics(y_eval, const, num_classes=num_classes)
    out['majority_class'] = majority
    out['eval_share_of_predicted_class'] = float((y_eval == majority).float().mean())
    out['eval_majority_class'] = int(torch.bincount(y_eval).argmax())
    return out
