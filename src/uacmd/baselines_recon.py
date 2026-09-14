from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .evidential import dirichlet_uncertainty, evidence_to_alpha
from .gate import MODALITIES
from .models import ModalityEncoder

class ImaginationBaseline(nn.Module):

    def __init__(self, dims, hidden_dim=128, num_classes=7, dropout=0.1, **_):
        super().__init__()
        self.encoders = nn.ModuleDict({m: ModalityEncoder(dims[m], hidden_dim, dropout) for m in MODALITIES})
        self.imagine = nn.ModuleDict({m: nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)) for m in MODALITIES})
        self.classifier = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        self.num_classes = num_classes

    def forward(self, x: dict, masks: dict):
        obs = {m: self.encoders[m](x[m], masks[m]) for m in MODALITIES}
        completed, recon = ({}, {})
        for m in MODALITIES:
            others = [o for o in MODALITIES if o != m]
            guess = self.imagine[m](torch.cat([obs[o] for o in others], dim=-1))
            recon[m] = guess
            completed[m] = masks[m] * obs[m] + (1.0 - masks[m]) * guess
        logits = self.classifier(torch.cat([completed[m] for m in MODALITIES], -1))
        alpha = evidence_to_alpha(F.softplus(logits))
        return {'logits': logits, 'alpha_fused': alpha, 'u_fused': dirichlet_uncertainty(alpha), 'alphas': {}, 'uncertainties': {}, 'feats': completed, 'recon': recon, 'observed': obs, 'fused_feat': torch.cat([completed[m] for m in MODALITIES], -1), 'gate_weights': {}}

def reconstruction_loss(out: dict, masks: dict) -> torch.Tensor:
    total, denom = (0.0, 0.0)
    for m in MODALITIES:
        w = masks[m]
        diff = (out['recon'][m] - out['observed'][m].detach()).pow(2).sum(-1, keepdim=True)
        total = total + (w * diff).sum()
        denom = denom + w.sum()
    return total / torch.clamp(denom, min=1.0)
