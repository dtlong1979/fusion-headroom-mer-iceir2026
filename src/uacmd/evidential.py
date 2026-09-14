from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['dirichlet_mutual_information', 'conjunctive_combination', 'evidence_to_alpha', 'dirichlet_uncertainty', 'expected_probability', 'EvidentialHead', 'EvidentialLoss', 'kl_dirichlet_to_uniform']

def evidence_to_alpha(evidence: torch.Tensor) -> torch.Tensor:
    return evidence + 1.0

def dirichlet_uncertainty(alpha: torch.Tensor) -> torch.Tensor:
    num_classes = alpha.shape[-1]
    strength = alpha.sum(dim=-1, keepdim=True)
    return num_classes / strength

def expected_probability(alpha: torch.Tensor) -> torch.Tensor:
    return alpha / alpha.sum(dim=-1, keepdim=True)

class EvidentialHead(nn.Module):

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None=None):
        evidence = F.softplus(self.fc(z))
        if mask is not None:
            evidence = evidence * mask
        alpha = evidence_to_alpha(evidence)
        return (alpha, dirichlet_uncertainty(alpha))

def kl_dirichlet_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    num_classes = alpha.shape[-1]
    strength = alpha.sum(dim=-1, keepdim=True)
    log_gamma_k = torch.lgamma(torch.tensor(float(num_classes), device=alpha.device, dtype=alpha.dtype))
    term = torch.lgamma(strength) - log_gamma_k - torch.lgamma(alpha).sum(dim=-1, keepdim=True) + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(strength))).sum(dim=-1, keepdim=True)
    return term

class EvidentialLoss(nn.Module):

    def __init__(self, num_classes: int, annealing_epochs: int=10, kl_weight: float=1.0, class_weight: torch.Tensor | None=None):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_epochs = max(1, annealing_epochs)
        self.kl_weight = kl_weight
        if class_weight is not None:
            self.register_buffer('class_weight', class_weight)
        else:
            self.class_weight = None

    def forward(self, alpha: torch.Tensor, target: torch.Tensor, epoch: int=1, sample_weight: torch.Tensor | None=None) -> torch.Tensor:
        y = F.one_hot(target, self.num_classes).to(alpha.dtype)
        strength = alpha.sum(dim=-1, keepdim=True)
        risk = (y * (torch.digamma(strength) - torch.digamma(alpha))).sum(dim=-1, keepdim=True)
        alpha_tilde = y + (1.0 - y) * alpha
        kl = kl_dirichlet_to_uniform(alpha_tilde)
        coef = min(1.0, float(epoch) / self.annealing_epochs) * self.kl_weight
        per_sample = risk + coef * kl
        if self.class_weight is not None:
            per_sample = per_sample * self.class_weight[target].unsqueeze(-1)
        if sample_weight is not None:
            per_sample = per_sample * sample_weight.view(-1, 1)
        return per_sample.mean()

def dirichlet_mutual_information(alpha: torch.Tensor) -> torch.Tensor:
    strength = alpha.sum(dim=-1, keepdim=True)
    p = alpha / strength
    term = torch.log(p + 1e-12) - torch.digamma(alpha + 1.0) + torch.digamma(strength + 1.0)
    return torch.clamp(-(p * term).sum(dim=-1, keepdim=True), min=0.0)

def conjunctive_combination(alphas: dict, masks: dict) -> torch.Tensor:
    keys = list(alphas)
    total = None
    for m in keys:
        e = (alphas[m] - 1.0) * masks[m]
        total = e if total is None else total + e
    return total + 1.0
