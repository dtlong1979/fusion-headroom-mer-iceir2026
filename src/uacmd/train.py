from __future__ import annotations
import copy
import itertools
from dataclasses import dataclass, field, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .evidential import EvidentialLoss, expected_probability
from .losses import jsd_probs, uncertainty_weighted_alignment
from .masking import MODALITIES, TEST_SCENARIOS, fixed_masks, sample_random_masks
from .metrics import classification_metrics, expected_calibration_error, majority_class_floor, uncertainty_auroc
from .baselines_recon import ImaginationBaseline, reconstruction_loss
from .models import ConcatBaseline, CrossModalTeacher, UACMDStudent

@dataclass
class TrainConfig:
    hidden_dim: int = 128
    num_classes: int = 7
    dropout: float = 0.1
    lr: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 128
    teacher_epochs: int = 25
    student_epochs: int = 40
    drop_rate: float = 0.5
    annealing_epochs: int = 10
    kl_weight: float = 0.5
    lambda_unimodal: float = 0.5
    lambda_distill: float = 1.0
    lambda_align: float = 0.5
    lambda_recon: float = 1.0
    temperature: float = 2.0
    gate: str = 'uncertainty'
    gate_normalize_u: bool = False
    gate_sharpness: bool = False
    gate_weight_rule: str = 'confidence'
    fusion_mode: str = 'gated'
    teacher_masked: bool = False
    align_weighting: str = 'uncertainty'
    align_metric: str = 'l2'
    align_projector: bool = False
    use_distillation: bool = True
    use_evidential: bool = True
    detach_uncertainty: bool = True
    class_weighted_loss: bool = True
    early_stopping_patience: int = 8
    seed: int = 0

    def to_dict(self):
        return asdict(self)

def _class_weights(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(y, minlength=num_classes).float().clamp(min=1.0)
    w = counts.sum() / (num_classes * counts)
    return w / w.mean()

def _batches(n, batch_size, generator, shuffle=True):
    idx = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]

def _softmax_ce(logits, target, class_weight=None):
    return F.cross_entropy(logits, target, weight=class_weight)

def train_teacher(data, cfg: TrainConfig, dims, verbose=False):
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = CrossModalTeacher(dims, cfg.hidden_dim, cfg.num_classes, cfg.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    y_tr = data['train']['y']
    cw = _class_weights(y_tr, cfg.num_classes) if cfg.class_weighted_loss else None
    ev_loss = EvidentialLoss(cfg.num_classes, cfg.annealing_epochs, cfg.kl_weight, cw)
    best_state, best_score, patience = (None, -np.inf, 0)
    n = len(y_tr)
    for epoch in range(1, cfg.teacher_epochs + 1):
        model.train()
        for batch in _batches(n, cfg.batch_size, gen):
            x = {m: data['train']['x'][m][batch] for m in MODALITIES}
            y = y_tr[batch]
            masks = sample_random_masks(len(batch), cfg.drop_rate, gen) if cfg.teacher_masked else fixed_masks('full', len(batch))
            out = model(x, masks)
            loss = ev_loss(out['alpha_fused'], y, epoch)
            loss = loss + cfg.lambda_unimodal * sum((ev_loss(out['alphas'][m], y, epoch) for m in MODALITIES)) / 3.0
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        score = evaluate(model, data['val'], cfg, scenarios=['full'])['full']['w_f1']
        if score > best_score:
            best_score, best_state, patience = (score, copy.deepcopy(model.state_dict()), 0)
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                break
        if verbose:
            print(f'  teacher epoch {epoch:3d}  val w-F1 {score:.4f}')
    model.load_state_dict(best_state)
    model.eval()
    return (model, best_score)

def train_student(data, cfg: TrainConfig, dims, teacher=None, model=None, verbose=False):
    torch.manual_seed(cfg.seed + 1)
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    if model is None:
        model = UACMDStudent(dims, cfg.hidden_dim, cfg.num_classes, cfg.gate, cfg.dropout, cfg.detach_uncertainty, cfg.align_projector, cfg.gate_normalize_u, cfg.gate_sharpness, cfg.gate_weight_rule, cfg.fusion_mode)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    y_tr = data['train']['y']
    cw = _class_weights(y_tr, cfg.num_classes) if cfg.class_weighted_loss else None
    ev_loss = EvidentialLoss(cfg.num_classes, cfg.annealing_epochs, cfg.kl_weight, cw)
    is_concat = isinstance(model, ConcatBaseline)
    is_recon = isinstance(model, ImaginationBaseline)
    best_state, best_score, patience = (None, -np.inf, 0)
    n = len(y_tr)
    for epoch in range(1, cfg.student_epochs + 1):
        model.train()
        for batch in _batches(n, cfg.batch_size, gen):
            x = {m: data['train']['x'][m][batch] for m in MODALITIES}
            y = y_tr[batch]
            masks = sample_random_masks(len(batch), cfg.drop_rate, gen)
            out = model(x, masks)
            if cfg.use_evidential:
                loss = ev_loss(out['alpha_fused'], y, epoch)
                if out['alphas']:
                    per_mod, denom = (0.0, 0.0)
                    for m in MODALITIES:
                        w = masks[m].squeeze(-1)
                        if w.sum() == 0:
                            continue
                        per_mod = per_mod + ev_loss(out['alphas'][m], y, epoch, w) * w.sum()
                        denom = denom + w.sum()
                    if denom > 0:
                        loss = loss + cfg.lambda_unimodal * per_mod / denom
            else:
                loss = _softmax_ce(out['logits'], y, cw)
            if is_recon:
                loss = loss + cfg.lambda_recon * reconstruction_loss(out, masks)
            if cfg.use_distillation and teacher is not None:
                with torch.no_grad():
                    t_masks = masks if cfg.teacher_masked else fixed_masks('full', len(batch))
                    t_out = teacher(x, t_masks)
                p_student = expected_probability(out['alpha_fused']) if cfg.use_evidential else torch.softmax(out['logits'], dim=-1)
                loss = loss + cfg.lambda_distill * jsd_probs(p_student, expected_probability(t_out['alpha_fused']), cfg.temperature)
                if not is_concat and (not is_recon):
                    loss = loss + cfg.lambda_align * uncertainty_weighted_alignment(out.get('align_feats', out['feats']), t_out['feats'], out['uncertainties'], masks, weighting=cfg.align_weighting, metric=cfg.align_metric)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val = evaluate(model, data['val'], cfg)
        score = float(np.mean([val[s]['w_f1'] for s in TEST_SCENARIOS]))
        if score > best_score:
            best_score, best_state, patience = (score, copy.deepcopy(model.state_dict()), 0)
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                break
        if verbose:
            print(f'  student epoch {epoch:3d}  val mean w-F1 {score:.4f}')
    model.load_state_dict(best_state)
    model.eval()
    return (model, best_score)

@torch.no_grad()
def marginalize_over_submasks(model, x: dict, mask_spec: dict, use_evidential: bool):
    present = [m for m in MODALITIES if mask_spec[m] > 0.5]
    if not present:
        raise ValueError('at least one modality must be available')
    b = next(iter(x.values())).shape[0]
    total, n_sub = (0.0, 0)
    for r in range(1, len(present) + 1):
        for combo in itertools.combinations(present, r):
            masks = {m: torch.full((b, 1), 1.0 if m in combo else 0.0) for m in MODALITIES}
            out = model(x, masks)
            p = expected_probability(out['alpha_fused']) if use_evidential else torch.softmax(out['logits'], dim=-1)
            total = total + p
            n_sub += 1
    return total / n_sub

@torch.no_grad()
def evaluate(model, split_data, cfg: TrainConfig, scenarios=None, batch_size=512, log_prior: torch.Tensor | None=None) -> dict:
    model.eval()
    scenarios = scenarios or list(TEST_SCENARIOS)
    y = split_data['y']
    n = len(y)
    results = {}
    for scen in scenarios:
        preds, probs, add_preds, tta_preds = ([], [], [], [])
        unc = {m: [] for m in MODALITIES}
        for i in range(0, n, batch_size):
            sl = slice(i, min(i + batch_size, n))
            x = {m: split_data['x'][m][sl] for m in MODALITIES}
            masks = fixed_masks(scen, x['T'].shape[0])
            out = model(x, masks)
            if cfg.use_evidential:
                p = expected_probability(out['alpha_fused'])
            else:
                p = torch.softmax(out['logits'], dim=-1)
            probs.append(p)
            preds.append(p.argmax(-1))
            tta_preds.append(marginalize_over_submasks(model, x, TEST_SCENARIOS[scen], cfg.use_evidential).argmax(-1))
            if log_prior is not None and out.get('alphas'):
                lp = log_prior.to(p.device).view(1, -1)
                score = lp.expand(x['T'].shape[0], -1).clone()
                for m in MODALITIES:
                    ch = torch.log(expected_probability(out['alphas'][m]) + 1e-12)
                    score = score + masks[m] * (ch - lp)
                add_preds.append(score.argmax(-1))
            for m in MODALITIES:
                if out['uncertainties']:
                    unc[m].append(out['uncertainties'][m])
        preds = torch.cat(preds)
        probs = torch.cat(probs)
        res = classification_metrics(y, preds, num_classes=cfg.num_classes)
        if tta_preds:
            tta = classification_metrics(y, torch.cat(tta_preds), num_classes=cfg.num_classes)
            res['tta_acc'], res['tta_w_f1'] = (tta['acc'], tta['w_f1'])
            res['tta_macro_f1'] = tta['macro_f1']
        if add_preds:
            add = classification_metrics(y, torch.cat(add_preds), num_classes=cfg.num_classes)
            res['add_acc'], res['add_w_f1'] = (add['acc'], add['w_f1'])
            res['add_macro_f1'] = add['macro_f1']
        res['ece'] = expected_calibration_error(probs, y)
        res['mean_conf'] = float(probs.max(-1).values.mean())
        if unc['T']:
            for m in MODALITIES:
                res[f'u_{m}'] = float(torch.cat(unc[m]).mean())
            if scen == 'full' and 'corrupt' in split_data:
                for m in MODALITIES:
                    res[f'auroc_u_{m}'] = uncertainty_auroc(torch.cat(unc[m]), split_data['corrupt'][m])
        results[scen] = res
    return results

@torch.no_grad()
def gate_weight_report(model, split_data, batch_size=512) -> dict:
    model.eval()
    n = len(split_data['y'])
    acc = {m: {'clean': [], 'corrupt': []} for m in MODALITIES}
    for i in range(0, n, batch_size):
        sl = slice(i, min(i + batch_size, n))
        x = {m: split_data['x'][m][sl] for m in MODALITIES}
        out = model(x, fixed_masks('full', x['T'].shape[0]))
        if not out.get('gate_weights'):
            return {}
        for m in MODALITIES:
            w = out['gate_weights'][m].squeeze(-1)
            c = split_data['corrupt'][m][sl].bool()
            acc[m]['clean'].append(w[~c])
            acc[m]['corrupt'].append(w[c])
    report = {}
    for m in MODALITIES:
        clean = torch.cat(acc[m]['clean'])
        corrupt = torch.cat(acc[m]['corrupt'])
        report[m] = {'w_clean': float(clean.mean()) if len(clean) else float('nan'), 'w_corrupt': float(corrupt.mean()) if len(corrupt) else float('nan')}
    return report
