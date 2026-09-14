from __future__ import annotations
import torch
import torch.nn.functional as F
from .evidential import expected_probability
from .gate import MODALITIES
__all__ = ['jsd_distillation', 'jsd_probs', 'uncertainty_weighted_alignment']
_EPS = 1e-08

def jsd_distillation(alpha_student: torch.Tensor, alpha_teacher: torch.Tensor, temperature: float=1.0) -> torch.Tensor:
    return jsd_probs(expected_probability(alpha_student), expected_probability(alpha_teacher), temperature)

def jsd_probs(p_student: torch.Tensor, p_teacher: torch.Tensor, temperature: float=1.0) -> torch.Tensor:
    if temperature != 1.0:
        p_student = p_student.pow(1.0 / temperature)
        p_student = p_student / p_student.sum(-1, keepdim=True)
        p_teacher = p_teacher.pow(1.0 / temperature)
        p_teacher = p_teacher / p_teacher.sum(-1, keepdim=True)
    m = 0.5 * (p_student + p_teacher)
    log_m = torch.log(m + _EPS)
    kl_s = (p_student * (torch.log(p_student + _EPS) - log_m)).sum(-1)
    kl_t = (p_teacher * (torch.log(p_teacher + _EPS) - log_m)).sum(-1)
    return (0.5 * (kl_s + kl_t)).mean()

def uncertainty_weighted_alignment(feats_student: dict, feats_teacher: dict, uncertainties: dict, masks: dict, weighting: str='uncertainty', normalize: bool=True, metric: str='l2') -> torch.Tensor:
    total, denom = (0.0, 0.0)
    for m in MODALITIES:
        mask = masks[m]
        if weighting == 'uncertainty':
            w = (1.0 - uncertainties[m].detach()) * mask
        elif weighting == 'uniform':
            w = mask
        else:
            raise ValueError(f"unknown weighting '{weighting}'")
        zs, zt = (feats_student[m], feats_teacher[m].detach())
        if metric == 'l2':
            diff = (zs - zt).pow(2).sum(-1, keepdim=True)
        elif metric == 'cosine':
            diff = 1.0 - F.cosine_similarity(zs, zt, dim=-1, eps=1e-08).unsqueeze(-1)
        else:
            raise ValueError(f"unknown alignment metric '{metric}'")
        total = total + (w * diff).sum()
        denom = denom + w.sum()
    if normalize:
        return total / torch.clamp(denom, min=1.0)
    return total / feats_student[MODALITIES[0]].shape[0]
