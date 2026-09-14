from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .evidential import EvidentialHead, conjunctive_combination, dirichlet_uncertainty, evidence_to_alpha
from .gate import AttentionGate, DynamicUncertaintyGate, MeanGate, MODALITIES
__all__ = ['UACMDStudent', 'CrossModalTeacher', 'ConcatBaseline', 'build_gate']
_GATES = {'uncertainty': DynamicUncertaintyGate, 'attention': AttentionGate, 'mean': MeanGate}

def build_gate(name: str, hidden_dim: int, **kwargs) -> nn.Module:
    if name not in _GATES:
        raise ValueError(f"unknown gate '{name}', expected one of {sorted(_GATES)}")
    return _GATES[name](hidden_dim, **kwargs)

class ModalityEncoder(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.net(x) * mask

class UACMDStudent(nn.Module):

    def __init__(self, dims: dict, hidden_dim: int=128, num_classes: int=7, gate: str='uncertainty', dropout: float=0.1, detach_uncertainty: bool=True, align_projector: bool=False, gate_normalize_u: bool=False, gate_sharpness: bool=False, gate_weight_rule: str='confidence', fusion_mode: str='gated'):
        super().__init__()
        self.encoders = nn.ModuleDict({m: ModalityEncoder(dims[m], hidden_dim, dropout) for m in MODALITIES})
        self.heads = nn.ModuleDict({m: EvidentialHead(hidden_dim, num_classes) for m in MODALITIES})
        self.align_proj = nn.ModuleDict({m: nn.Linear(hidden_dim, hidden_dim) for m in MODALITIES}) if align_projector else None
        gate_kwargs = {'detach_uncertainty': detach_uncertainty, 'normalize_u': gate_normalize_u, 'sharpness': gate_sharpness, 'weight_rule': gate_weight_rule} if gate == 'uncertainty' else {}
        self.gate = build_gate(gate, hidden_dim, **gate_kwargs)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        self.num_classes = num_classes
        if fusion_mode not in ('gated', 'conjunctive', 'hybrid'):
            raise ValueError(f"unknown fusion_mode '{fusion_mode}'")
        self.fusion_mode = fusion_mode

    def forward(self, x: dict, masks: dict):
        feats, alphas, uncs = ({}, {}, {})
        for m in MODALITIES:
            z = self.encoders[m](x[m], masks[m])
            a, u = self.heads[m](z, masks[m])
            feats[m], alphas[m], uncs[m] = (z, a, u)
        fused, weights = self.gate(feats, uncs, masks, alphas)
        logits = self.classifier(fused)
        gated_evidence = F.softplus(logits)
        if self.fusion_mode == 'gated':
            alpha_fused = evidence_to_alpha(gated_evidence)
        elif self.fusion_mode == 'conjunctive':
            alpha_fused = conjunctive_combination(alphas, masks)
        else:
            alpha_fused = conjunctive_combination(alphas, masks) + gated_evidence
        return {'logits': logits, 'alpha_fused': alpha_fused, 'u_fused': dirichlet_uncertainty(alpha_fused), 'alphas': alphas, 'uncertainties': uncs, 'feats': feats, 'align_feats': {m: self.align_proj[m](feats[m]) for m in MODALITIES} if self.align_proj is not None else feats, 'fused_feat': fused, 'gate_weights': weights}

class CrossModalBlock(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int=4, dropout: float=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim))

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        q = self.norm1(target).unsqueeze(1)
        kv = self.norm1(source).unsqueeze(1)
        attended, _ = self.attn(q, kv, kv, need_weights=False)
        h = target + attended.squeeze(1)
        return h + self.ffn(self.norm2(h))

class CrossModalTeacher(nn.Module):

    def __init__(self, dims: dict, hidden_dim: int=128, num_classes: int=7, dropout: float=0.1, num_heads: int=4):
        super().__init__()
        self.encoders = nn.ModuleDict({m: ModalityEncoder(dims[m], hidden_dim, dropout) for m in MODALITIES})
        self.cross = nn.ModuleDict({f'{tgt}<-{src}': CrossModalBlock(hidden_dim, num_heads, dropout) for tgt in MODALITIES for src in MODALITIES if tgt != src})
        self.merge = nn.ModuleDict({m: nn.Linear(2 * hidden_dim, hidden_dim) for m in MODALITIES})
        self.heads = nn.ModuleDict({m: EvidentialHead(hidden_dim, num_classes) for m in MODALITIES})
        self.fuse = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_classes))
        self.num_classes = num_classes

    def forward(self, x: dict, masks: dict | None=None):
        ones = {m: torch.ones_like(next(iter(x.values()))[:, :1]) for m in MODALITIES}
        masks = masks or ones
        base = {m: self.encoders[m](x[m], masks[m]) for m in MODALITIES}
        feats = {}
        for tgt in MODALITIES:
            others = [s for s in MODALITIES if s != tgt]
            enriched = [self.cross[f'{tgt}<-{s}'](base[tgt], base[s]) for s in others]
            feats[tgt] = self.merge[tgt](torch.cat(enriched, dim=-1))
        alphas, uncs = ({}, {})
        for m in MODALITIES:
            a, u = self.heads[m](feats[m], masks[m])
            alphas[m], uncs[m] = (a, u)
        fused = self.fuse(torch.cat([feats[m] for m in MODALITIES], dim=-1))
        alpha_fused = evidence_to_alpha(F.softplus(self.classifier(fused)))
        return {'alpha_fused': alpha_fused, 'u_fused': dirichlet_uncertainty(alpha_fused), 'alphas': alphas, 'uncertainties': uncs, 'feats': feats, 'fused_feat': fused}

class ConcatBaseline(nn.Module):

    def __init__(self, dims, hidden_dim=128, num_classes=7, dropout=0.1, **_):
        super().__init__()
        self.encoders = nn.ModuleDict({m: ModalityEncoder(dims[m], hidden_dim, dropout) for m in MODALITIES})
        self.classifier = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        self.num_classes = num_classes

    def forward(self, x: dict, masks: dict):
        feats = {m: self.encoders[m](x[m], masks[m]) for m in MODALITIES}
        logits = self.classifier(torch.cat([feats[m] for m in MODALITIES], dim=-1))
        alpha = evidence_to_alpha(F.softplus(logits))
        return {'logits': logits, 'alpha_fused': alpha, 'u_fused': dirichlet_uncertainty(alpha), 'alphas': {}, 'uncertainties': {}, 'feats': feats, 'fused_feat': torch.cat([feats[m] for m in MODALITIES], dim=-1), 'gate_weights': {}}
