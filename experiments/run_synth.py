from __future__ import annotations
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from uacmd.data.synth import SynthConfig, SynthMM, standardize_
from uacmd.masking import TEST_SCENARIOS
from uacmd.baselines_recon import ImaginationBaseline
from uacmd.models import ConcatBaseline, UACMDStudent
from uacmd.metrics import majority_class_floor
from uacmd.train import TrainConfig, evaluate, gate_weight_report, train_student, train_teacher
METHODS = {'early_fusion': ('concat', dict(drop_rate=0.0, use_evidential=False, use_distillation=False), False), 'early_fusion_md': ('concat', dict(use_evidential=False, use_distillation=False), False), 'mean_gate_md': ('student', dict(gate='mean', use_evidential=False, use_distillation=False), False), 'attention_gate_md': ('student', dict(gate='attention', use_evidential=False, use_distillation=False), False), 'kd_jsd': ('concat', dict(use_evidential=False, use_distillation=True), True), 'recon_imagine': ('recon', dict(use_evidential=False, use_distillation=False), False), 'uacmd': ('student', dict(gate='uncertainty', use_evidential=True, use_distillation=True, align_weighting='uncertainty'), True), 'abl_no_distill': ('student', dict(gate='uncertainty', use_evidential=True, use_distillation=False), False), 'abl_no_align': ('student', dict(gate='uncertainty', use_evidential=True, use_distillation=True, lambda_align=0.0), True), 'abl_uniform_align': ('student', dict(gate='uncertainty', use_evidential=True, use_distillation=True, align_weighting='uniform'), True), 'abl_attention_gate': ('student', dict(gate='attention', use_evidential=True, use_distillation=True), True), 'abl_no_evidential': ('student', dict(gate='attention', use_evidential=False, use_distillation=True), True), 'abl_grad_through_u': ('student', dict(gate='uncertainty', use_evidential=True, use_distillation=True, detach_uncertainty=False), True), 'uacmd_cosine': ('student', dict(gate='uncertainty', use_evidential=True, use_distillation=True, align_metric='cosine'), True), 'uacmd_projector': ('student', dict(gate='uncertainty', use_evidential=True, use_distillation=True, align_projector=True), True), 'uacmd_cosine_projector': ('student', dict(gate='uncertainty', use_evidential=True, use_distillation=True, align_metric='cosine', align_projector=True), True), 'fix_normu': ('student', dict(gate='uncertainty', use_evidential=True, use_distillation=True, gate_normalize_u=True), True), 'fix_sharp': ('student', dict(gate='uncertainty', use_evidential=True, use_distillation=True, gate_sharpness=True), True), 'fix_lowkl': ('student', dict(gate='uncertainty', use_evidential=True, use_distillation=True, kl_weight=0.1), True), 'th_precision': ('student', dict(gate='uncertainty', use_evidential=True, use_distillation=True, gate_weight_rule='precision', kl_weight=0.1), True), 'th_mi': ('student', dict(gate='uncertainty', use_evidential=True, use_distillation=True, gate_weight_rule='mi', kl_weight=0.1), True), 'th_conjunctive': ('student', dict(gate='uncertainty', use_evidential=True, use_distillation=True, fusion_mode='conjunctive', kl_weight=0.1), True), 'th_hybrid': ('student', dict(gate='uncertainty', use_evidential=True, use_distillation=True, fusion_mode='hybrid', kl_weight=0.1), True), 'th_maskteacher': ('student', dict(gate='uncertainty', use_evidential=True, use_distillation=True, teacher_masked=True), True), 'th_all': ('student', dict(gate='uncertainty', use_evidential=True, use_distillation=True, kl_weight=0.1, gate_weight_rule='precision', fusion_mode='hybrid', teacher_masked=True, align_metric='cosine', align_projector=True), True)}

def build_model(kind: str, dims, cfg: TrainConfig):
    if kind == 'concat':
        return ConcatBaseline(dims, cfg.hidden_dim, cfg.num_classes, cfg.dropout)
    if kind == 'recon':
        return ImaginationBaseline(dims, cfg.hidden_dim, cfg.num_classes, cfg.dropout)
    return UACMDStudent(dims, cfg.hidden_dim, cfg.num_classes, cfg.gate, cfg.dropout, cfg.detach_uncertainty, cfg.align_projector, cfg.gate_normalize_u, cfg.gate_sharpness, cfg.gate_weight_rule, cfg.fusion_mode)

def run_seed(seed: int, args, data_cache: dict) -> dict:
    synth_cfg = SynthConfig(seed=0)
    dims = synth_cfg.dims
    if seed not in data_cache:
        gen = SynthMM(synth_cfg)
        data_cache[seed] = standardize_(gen.build(data_seed=seed))
    data = data_cache[seed]
    base = TrainConfig(seed=seed, teacher_epochs=args.teacher_epochs, student_epochs=args.student_epochs, drop_rate=args.drop_rate)
    teacher, teacher_val = train_teacher(data, base, dims, verbose=args.verbose)
    teacher_test = evaluate(teacher, data['test'], base, scenarios=['full'])
    teacher_masked = None
    wants_masked = any((ov.get('teacher_masked') for name, (_k, ov, _n) in METHODS.items() if not args.only or name in args.only))
    if wants_masked:
        masked_cfg = TrainConfig(seed=seed, teacher_epochs=args.teacher_epochs, student_epochs=args.student_epochs, drop_rate=args.drop_rate, teacher_masked=True)
        teacher_masked, teacher_masked_val = train_teacher(data, masked_cfg, dims, verbose=args.verbose)
    counts = torch.bincount(data['train']['y'], minlength=base.num_classes).float()
    log_prior = torch.log(counts / counts.sum())
    out = {'seed': seed, 'teacher': {'val_w_f1': teacher_val, 'test_full': teacher_test['full']}, 'majority_floor': majority_class_floor(data['train']['y'], data['test']['y'], num_classes=base.num_classes), 'methods': {}}
    if teacher_masked is not None:
        out['teacher_masked'] = {'val_w_f1': teacher_masked_val, 'test_full': evaluate(teacher_masked, data['test'], masked_cfg, scenarios=['full'])['full']}
    for name, (kind, overrides, needs_teacher) in METHODS.items():
        if args.only and name not in args.only:
            continue
        kwargs = dict(seed=seed, teacher_epochs=args.teacher_epochs, student_epochs=args.student_epochs, drop_rate=args.drop_rate)
        kwargs.update(overrides)
        cfg = TrainConfig(**kwargs)
        t0 = time.time()
        torch.manual_seed(cfg.seed + 1)
        model = build_model(kind, dims, cfg)
        chosen_teacher = None
        if needs_teacher:
            chosen_teacher = teacher_masked if cfg.teacher_masked else teacher
            if chosen_teacher is None:
                raise RuntimeError(f"method '{name}' asks for a mask-conditioned teacher but none was trained; this is the dead-flag failure mode")
        model, val_score = train_student(data, cfg, dims, teacher=chosen_teacher, model=model, verbose=args.verbose)
        res = evaluate(model, data['test'], cfg, log_prior=log_prior)
        entry = {'val_mean_w_f1': val_score, 'test': res, 'mean_w_f1': float(np.mean([res[s]['w_f1'] for s in TEST_SCENARIOS])), 'mean_acc': float(np.mean([res[s]['acc'] for s in TEST_SCENARIOS])), 'gate': gate_weight_report(model, data['test']), 'seconds': round(time.time() - t0, 1), 'n_params': int(sum((p.numel() for p in model.parameters())))}
        out['methods'][name] = entry
        print(f"  [seed {seed}] {name:22s} mean w-F1 {entry['mean_w_f1']:.4f} full {res['full']['w_f1']:.4f} text-only {res['text_only']['w_f1']:.4f} ({entry['seconds']}s)", flush=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed-start', type=int, default=0)
    ap.add_argument('--teacher-epochs', type=int, default=30)
    ap.add_argument('--student-epochs', type=int, default=50)
    ap.add_argument('--drop-rate', type=float, default=0.5)
    ap.add_argument('--out', default='experiments/results')
    ap.add_argument('--tag', default='main')
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(args.out, exist_ok=True)
    cache, runs = ({}, [])
    t0 = time.time()
    for s in range(args.seed_start, args.seed_start + args.seeds):
        print(f'=== seed {s} ===', flush=True)
        runs.append(run_seed(s, args, cache))
        cache.pop(s, None)
    payload = {'config': vars(args), 'synth_config': SynthConfig().to_dict(), 'train_defaults': TrainConfig().to_dict(), 'scenarios': list(TEST_SCENARIOS), 'runs': runs, 'total_seconds': round(time.time() - t0, 1), 'torch_version': torch.__version__}
    path = os.path.join(args.out, f'synth_{args.tag}.json')
    with open(path, 'w') as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {path}  ({payload['total_seconds']}s total)")
if __name__ == '__main__':
    main()
