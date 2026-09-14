from __future__ import annotations
import torch
MODALITIES = ('T', 'A', 'V')
TEST_SCENARIOS = {'full': {'T': 1, 'A': 1, 'V': 1}, 'no_visual': {'T': 1, 'A': 1, 'V': 0}, 'no_audio': {'T': 1, 'A': 0, 'V': 1}, 'no_text': {'T': 0, 'A': 1, 'V': 1}, 'text_only': {'T': 1, 'A': 0, 'V': 0}, 'audio_only': {'T': 0, 'A': 1, 'V': 0}, 'visual_only': {'T': 0, 'A': 0, 'V': 1}}
__all__ = ['TEST_SCENARIOS', 'fixed_masks', 'sample_random_masks', 'MODALITIES']

def fixed_masks(scenario: str, batch_size: int, device=None) -> dict:
    if scenario not in TEST_SCENARIOS:
        raise ValueError(f"unknown scenario '{scenario}'")
    spec = TEST_SCENARIOS[scenario]
    return {m: torch.full((batch_size, 1), float(spec[m]), device=device) for m in MODALITIES}

def sample_random_masks(batch_size: int, drop_rate: float, generator: torch.Generator | None=None, device=None) -> dict:
    keep = (torch.rand(batch_size, 3, generator=generator, device=device) >= drop_rate).float()
    empty = keep.sum(dim=1) == 0
    if empty.any():
        idx = torch.randint(0, 3, (int(empty.sum()),), generator=generator, device=device)
        rows = torch.nonzero(empty, as_tuple=False).squeeze(-1)
        keep[rows, idx] = 1.0
    return {m: keep[:, i:i + 1] for i, m in enumerate(MODALITIES)}
