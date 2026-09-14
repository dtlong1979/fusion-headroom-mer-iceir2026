from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .evidential import dirichlet_mutual_information
MODALITIES = ('T', 'A', 'V')
GAMMA_FLOOR = 0.001
__all__ = ['DynamicUncertaintyGate', 'AttentionGate', 'MeanGate', 'MODALITIES']

class DynamicUncertaintyGate(nn.Module):

    def __init__(self, hidden_dim: int, detach_uncertainty: bool=True, eps: float=1e-06, normalize_u: bool=False, sharpness: bool=False, ema_momentum: float=0.05, weight_rule: str='confidence'):
        super().__init__()
        self.proj = nn.ModuleDict({m: nn.Linear(hidden_dim, hidden_dim) for m in MODALITIES})
        self.detach_uncertainty = detach_uncertainty
        self.eps = eps
        self.normalize_u = normalize_u
        self.ema_momentum = ema_momentum
        if weight_rule not in ('confidence', 'precision', 'mi'):
            raise ValueError(f"unknown weight_rule '{weight_rule}'")
        self.weight_rule = weight_rule
        if normalize_u:
            self.register_buffer('u_mean', torch.full((len(MODALITIES),), 0.5))
            self.register_buffer('u_var', torch.ones(len(MODALITIES)))
            self.kappa = nn.Parameter(torch.zeros(1))
        if sharpness:
            raw0 = torch.log(torch.expm1(torch.tensor(1.0 - GAMMA_FLOOR)))
            self.gamma_raw = nn.Parameter(raw0.reshape(1).clone())
        self.sharpness = sharpness

    def _update_stats(self, uncertainties: dict, masks: dict) -> None:
        with torch.no_grad():
            for i, m in enumerate(MODALITIES):
                sel = masks[m].squeeze(-1) > 0.5
                if sel.sum() < 2:
                    continue
                vals = uncertainties[m].squeeze(-1)[sel]
                mom = self.ema_momentum
                self.u_mean[i] = (1 - mom) * self.u_mean[i] + mom * vals.mean()
                self.u_var[i] = (1 - mom) * self.u_var[i] + mom * vals.var()

    def _base_confidence(self, m_index: int, u: torch.Tensor, alpha: torch.Tensor | None) -> torch.Tensor:
        if self.weight_rule == 'confidence':
            return torch.clamp(1.0 - u, min=self.eps)
        if self.weight_rule == 'precision':
            return torch.clamp(1.0 / torch.clamp(u, min=self.eps), min=self.eps)
        mi = dirichlet_mutual_information(alpha)
        if self.detach_uncertainty:
            mi = mi.detach()
        return 1.0 / (mi + 0.001)

    def forward(self, feats: dict, uncertainties: dict, masks: dict, alphas: dict | None=None):
        if self.normalize_u and self.training:
            self._update_stats(uncertainties, masks)
        gamma = None
        if self.sharpness:
            gamma = F.softplus(self.gamma_raw) + GAMMA_FLOOR
        confidences = {}
        for i, m in enumerate(MODALITIES):
            u = uncertainties[m]
            if self.detach_uncertainty:
                u = u.detach()
            alpha = alphas[m] if alphas is not None else None
            if self.weight_rule == 'mi' and alpha is None:
                raise ValueError("weight_rule='mi' needs the per-channel alphas")
            c = self._base_confidence(i, u, alpha)
            if gamma is not None:
                c = c.pow(gamma)
            if self.normalize_u:
                z = (self.u_mean[i] - u) / torch.sqrt(self.u_var[i] + 1e-06)
                c = c * torch.exp(torch.clamp(self.kappa * z, -6.0, 6.0))
            confidences[m] = c * masks[m]
        total = torch.clamp(sum((confidences[m] for m in MODALITIES)), min=self.eps)
        weights, fused = ({}, 0.0)
        for m in MODALITIES:
            w = confidences[m] / total
            weights[m] = w
            fused = fused + w * self.proj[m](feats[m])
        return (fused, weights)

class AttentionGate(nn.Module):

    def __init__(self, hidden_dim: int, **_):
        super().__init__()
        self.proj = nn.ModuleDict({m: nn.Linear(hidden_dim, hidden_dim) for m in MODALITIES})
        self.score = nn.ModuleDict({m: nn.Linear(hidden_dim, 1) for m in MODALITIES})

    def forward(self, feats: dict, uncertainties: dict, masks: dict, alphas: dict | None=None):
        scores = torch.cat([self.score[m](feats[m]) for m in MODALITIES], dim=-1)
        mask = torch.cat([masks[m] for m in MODALITIES], dim=-1)
        scores = scores.masked_fill(mask < 0.5, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        weights, fused = ({}, 0.0)
        for i, m in enumerate(MODALITIES):
            w = attn[:, i:i + 1]
            weights[m] = w
            fused = fused + w * self.proj[m](feats[m])
        return (fused, weights)

class MeanGate(nn.Module):

    def __init__(self, hidden_dim: int, **_):
        super().__init__()
        self.proj = nn.ModuleDict({m: nn.Linear(hidden_dim, hidden_dim) for m in MODALITIES})

    def forward(self, feats: dict, uncertainties: dict, masks: dict, alphas: dict | None=None):
        total = torch.clamp(sum((masks[m] for m in MODALITIES)), min=1.0)
        weights, fused = ({}, 0.0)
        for m in MODALITIES:
            w = masks[m] / total
            weights[m] = w
            fused = fused + w * self.proj[m](feats[m])
        return (fused, weights)
