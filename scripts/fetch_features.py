from __future__ import annotations
import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
REPO = 'https://github.com/declare-lab/multimodal-deep-learning'
COMMIT = '111b401d867c14a1dcd930fd2b2ab58a5eca3f70'
SUB = 'contextual-attention-based-LSTM'
COPY = [f'{SUB}/dataset/mosi/raw/{m}_2way.pickle' for m in ('text', 'audio', 'video')]
UNZIP = [f'{SUB}/unimodal_mosei_3way.pickle.zip', f'{SUB}/dataset/iemocap/raw/IEMOCAP_features_raw.pkl.zip']
DIGESTS = {'text_2way.pickle': '52e94bcf12189a606765ead9f00fdca3', 'audio_2way.pickle': 'b503c104186243675659c4b3788fd70f', 'video_2way.pickle': '99444cd0d1c06751ed68a843a4d5bc2e', 'IEMOCAP_features_raw.pkl': '811b524cc12e5ff99434a19b9ff48704'}

def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('dest', nargs='?', default='data-features')
    ap.add_argument('--keep-clone', action='store_true')
    args = ap.parse_args()
    dest = Path(args.dest).expanduser().resolve()
    src = dest / '_src'
    dest.mkdir(parents=True, exist_ok=True)
    if not (src / '.git').exists():
        print(f'cloning {REPO}\n  into {src}')
        env_note = "GIT_LFS_SKIP_SMUDGE=1 is set: this repo's LFS objects are not needed."
        print(f'  ({env_note})')
        import os
        env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
        subprocess.run(['git', 'clone', '--depth', '1', REPO, str(src)], check=True, env=env)
    head = subprocess.run(['git', '-C', str(src), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    if head != COMMIT:
        print(f'\nnote: clone is at {head[:12]}, not the recorded {COMMIT[:12]}.')
        print('      the digest check below is what actually decides whether the')
        print('      files match what the experiments were run on.')
    for rel in COPY:
        shutil.copy2(src / rel, dest / Path(rel).name)
        print(f'copied  {Path(rel).name}')
    for rel in UNZIP:
        with zipfile.ZipFile(src / rel) as zf:
            for name in zf.namelist():
                if name.startswith('__MACOSX') or name.endswith('/'):
                    continue
                target = dest / Path(name).name
                with zf.open(name) as fin, open(target, 'wb') as fout:
                    shutil.copyfileobj(fin, fout)
                print(f'unzipped {target.name}')
    print('\nchecking the digests of the files the recorded results were run on')
    bad = 0
    for name, want in DIGESTS.items():
        got = md5(dest / name)
        ok = got == want
        bad += not ok
        print(f"  {('ok      ' if ok else 'MISMATCH')} {name}")
        if not ok:
            print(f'    expected {want}\n    got      {got}')
    if not args.keep_clone:
        shutil.rmtree(src, ignore_errors=True)
        print(f'\nremoved the working clone ({src.name}); pass --keep-clone to keep it')
    print(f'\nfeatures are in {dest}\n\nRun, for example:\n')
    print(f"  python experiments/iemocap_context.py --pickle {dest / 'IEMOCAP_features_raw.pkl'} --seeds 1 --out {dest / 'trial_iemocap_context.json'}")
    print(f"  python experiments/mosi_mosei_headroom.py --mosi-root {dest} --mosei-file {dest / 'unimodal_mosei_3way.pickle'} --out {dest / 'trial_mosi_mosei.json'}")
    print("\n(set PYTHONPATH to the repo's src/ directory, or run from the repo root")
    print(" with PYTHONPATH=src on Unix / $env:PYTHONPATH='src' in PowerShell)")
    return 1 if bad else 0
if __name__ == '__main__':
    sys.exit(main())
