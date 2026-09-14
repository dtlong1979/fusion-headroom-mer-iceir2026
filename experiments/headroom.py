from __future__ import annotations
import argparse
import json
import os
import sys
REQUIRED = ('corpus', 'modalities', 'fused', 'metric', 'source', 'protocol')

def validate(entry: dict) -> list[str]:
    problems = []
    for field in REQUIRED:
        if not entry.get(field):
            problems.append(f"missing '{field}'")
    if entry.get('modalities') and (not isinstance(entry['modalities'], dict)):
        problems.append("'modalities' must map a channel name to a score")
    if entry.get('same_architecture') is False:
        problems.append('unimodal and fused numbers come from different architectures: H is inflated and is reported as unusable')
    if entry.get('second_hand'):
        problems.append('number taken from a search summary, not the source')
    return problems

def headroom(entry: dict):
    best_m, best_v = max(entry['modalities'].items(), key=lambda kv: kv[1])
    return (entry['fused'] - best_v, best_m, best_v)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='docs/headroom-data.json')
    args = ap.parse_args()
    if not os.path.exists(args.data):
        print(f'no data file at {args.data}. Nothing has been collected yet.')
        return 1
    entries = json.load(open(args.data))['entries']
    rows, unusable = ([], [])
    for e in entries:
        problems = validate(e)
        if any((p.startswith(('missing', "'modalities'")) for p in problems)):
            unusable.append((e.get('corpus', '?'), problems))
            continue
        h, best_m, best_v = headroom(e)
        rows.append((e, h, best_m, best_v, problems))
    rows.sort(key=lambda r: -r[1])
    print('# Fusion headroom of candidate corpora\n')
    print('H = fused - best single modality, both from the same paper, table and')
    print('protocol. Larger H means the channels are complementary and a fusion or')
    print('missing-modality method has room to show a real effect.\n')
    print(f"{'corpus':22s} {'metric':>10s} {'best single':>22s} {'fused':>7s} {'H':>7s}  caveats")
    print('-' * 100)
    for e, h, best_m, best_v, problems in rows:
        caveat = '; '.join(problems) if problems else ''
        print(f"{e['corpus']:22s} {e['metric']:>10s} {best_m + ' ' + format(best_v, '.1f'):>22s} {e['fused']:7.1f} {h:7.1f}  {caveat}")
    print('\n## Sources\n')
    for e, h, *_ in rows:
        print(f"- **{e['corpus']}** (H = {h:+.1f}, {e['metric']}): {e['source']}")
        print(f"  protocol: {e['protocol']}")
        if e.get('note'):
            print(f"  note: {e['note']}")
    if unusable:
        print('\n## Not usable\n')
        for corpus, problems in unusable:
            print(f"- {corpus}: {'; '.join(problems)}")
    if rows:
        print('\n## Verdict\n')
        best = rows[0]
        ref = [r for r in rows if 'SynthMM' in r[0]['corpus']]
        if ref:
            gap = best[1] - ref[0][1]
            print(f"  Largest H: {best[0]['corpus']} at {best[1]:.1f} points.")
            print(f"  Reference (this study's benchmark): {ref[0][1]:.1f} points.")
            print(f'  Difference: {gap:+.1f} points of headroom.\n')
            if gap < 5:
                print('  A difference under about 5 points does not justify moving')
                print('  corpora: the new one is as redundant as the old, and a')
                print('  missing-modality method will have the same non-result there.')
            else:
                print('  A difference this large is worth acting on: the same method')
                print('  has materially more room on the new corpus.')
        else:
            print('  No reference row to compare against; add one before concluding.')
    return 0
if __name__ == '__main__':
    sys.exit(main())
