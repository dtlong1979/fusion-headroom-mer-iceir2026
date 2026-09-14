from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from uacmd.data.real import MELD_EMOTIONS
SPLITS = {'train': 'train_sent_emo.csv', 'dev': 'dev_sent_emo.csv', 'test': 'test_sent_emo.csv'}
MEDIA_DIRS = {'train': 'train_splits', 'dev': 'dev_splits_complete', 'test': 'output_repeated_splits_test'}
RATE = 16000

def _finite_or_die(vecs: np.ndarray, model_name: str) -> np.ndarray:
    if not np.isfinite(vecs).all():
        raise SystemExit(f'{model_name}: non-finite features; rerun without fp16')
    return vecs

@torch.no_grad()
def encode_texts(texts: list[str], model_name: str, device: str, batch: int=64, max_length: int=128) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    if device == 'cuda':
        model = model.half()
    rows = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
        out = model(**enc).last_hidden_state
        mask = enc['attention_mask'].unsqueeze(-1).to(out.dtype)
        rows.append(((out * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu())
    return _finite_or_die(torch.cat(rows).numpy(), model_name)

def decode_audio(path: Path, rate: int=RATE) -> np.ndarray:
    import av
    with av.open(str(path)) as c:
        res = av.AudioResampler(format='flt', layout='mono', rate=rate)
        chunks = []
        for frame in c.decode(c.streams.audio[0]):
            chunks += [f.to_ndarray().reshape(-1) for f in res.resample(frame)]
        chunks += [f.to_ndarray().reshape(-1) for f in res.resample(None)]
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

def sample_frames(path: Path, n: int) -> list:
    import av
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.codec_context.thread_count = 2
        total = stream.frames
        if total > 0:
            want = set(np.linspace(0, total - 1, n).round().astype(int).tolist())
            frames = [f.to_image() for i, f in enumerate(c.decode(stream)) if i in want]
        else:
            frames = [f.reformat(width=256, height=144).to_image() for f in c.decode(stream)]
    if not frames:
        return []
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]

class _Clips(torch.utils.data.Dataset):

    def __init__(self, paths: list[Path], n: int, processor):
        self.paths, self.n, self.proc = (paths, n, processor)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            frames = sample_frames(self.paths[i], self.n) if self.paths[i].exists() else []
        except Exception:
            frames = []
        if not frames:
            return (torch.zeros(self.n, 3, 224, 224), False)
        return (self.proc(images=frames, return_tensors='pt')['pixel_values'], True)

@torch.no_grad()
def encode_video(paths: list[Path], model_name: str, device: str, frames: int, workers: int) -> tuple[np.ndarray, np.ndarray]:
    from transformers import AutoImageProcessor, CLIPModel
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    if device == 'cuda':
        model = model.half()
    loader = torch.utils.data.DataLoader(_Clips(paths, frames, proc), batch_size=16, num_workers=workers)
    vecs, missing = ([], [])
    for b, (px, ok) in enumerate(loader):
        x = px.flatten(0, 1).to(device)
        emb = model.get_image_features(pixel_values=x.half() if device == 'cuda' else x)
        if not torch.is_tensor(emb):
            emb = emb.pooler_output
        vecs.append(emb.float().view(len(px), frames, -1).mean(1).cpu())
        missing.append(~ok)
        if b % 50 == 0:
            print(f'    {b * 16:6d}/{len(paths)}', flush=True)
    vecs = torch.cat(vecs).numpy()
    missing = torch.cat(missing).numpy()
    vecs[missing] = 0.0
    return (_finite_or_die(vecs, model_name), missing)

@torch.no_grad()
def encode_audio(paths: list[Path], model_name: str, device: str, max_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    if device == 'cuda':
        model = model.half()
    vecs = np.zeros((len(paths), model.config.hidden_size), dtype=np.float32)
    missing = np.zeros(len(paths), dtype=bool)
    for i, p in enumerate(paths):
        try:
            wav = decode_audio(p)[:int(max_seconds * RATE)] if p.exists() else None
        except Exception as exc:
            print(f'    unreadable {p.name}: {type(exc).__name__}', flush=True)
            wav = None
        if wav is None or wav.size < 400:
            missing[i] = True
            continue
        x = fe(wav, sampling_rate=RATE, return_tensors='pt').input_values.to(device)
        h = model(x.half() if device == 'cuda' else x).last_hidden_state[0]
        vecs[i] = h.float().mean(0).cpu().numpy()
        if i % 1000 == 0:
            print(f'    {i:6d}/{len(paths)}', flush=True)
    return (_finite_or_die(vecs, model_name), missing)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--annotations', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--blocks', default='text')
    ap.add_argument('--text-model', default='bert-base-uncased')
    ap.add_argument('--audio-model', default='microsoft/wavlm-base-plus')
    ap.add_argument('--media', type=Path)
    ap.add_argument('--max-audio-seconds', type=float, default=30.0)
    ap.add_argument('--video-model', default='openai/clip-vit-base-patch16')
    ap.add_argument('--frames', type=int, default=8)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    blocks = [b.strip() for b in args.blocks.split(',') if b.strip()]
    if {'audio', 'video'} & set(blocks) and args.media is None:
        ap.error('--blocks audio/video needs --media')
    import pandas as pd
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    label_index = {e: i for i, e in enumerate(MELD_EMOTIONS)}
    frames = {s: pd.read_csv(args.annotations / f).sort_values(['Dialogue_ID', 'Utterance_ID']) for s, f in SPLITS.items()}
    if args.out.exists():
        with open(args.out, 'rb') as fh:
            out = pickle.load(fh)
        print(f'adding {blocks} to {args.out}')
    else:
        out = {'meta': {'pooling': 'mean, last layer', 'emotions': list(MELD_EMOTIONS)}}
        for s in SPLITS:
            out[s] = {int(dia): {'labels': np.array([label_index[e.strip().lower()] for e in g['Emotion']], dtype=np.int64), 'speakers': [str(p) for p in g['Speaker']]} for dia, g in frames[s].groupby('Dialogue_ID', sort=True)}
    order = [(s, int(r.Dialogue_ID), int(r.Utterance_ID), str(r.Utterance)) for s in SPLITS for r in frames[s].itertuples()]

    def scatter(name: str, rows: np.ndarray) -> None:
        i = 0
        for s in SPLITS:
            for dia in sorted(out[s]):
                n = len(out[s][dia]['labels'])
                out[s][dia][name] = rows[i:i + n]
                i += n
        assert i == len(rows)
    if 'text' in blocks:
        print(f'encoding {len(order)} utterances with a frozen {args.text_model} on {device}')
        scatter('text', encode_texts([o[3] for o in order], args.text_model, device))
        out['meta']['text_model'] = args.text_model
    if 'audio' in blocks:
        paths = [args.media / MEDIA_DIRS[s] / f'dia{d}_utt{u}.mp4' for s, d, u, _ in order]
        for s in SPLITS:
            if not (args.media / MEDIA_DIRS[s]).is_dir():
                print(f'  warning: {args.media / MEDIA_DIRS[s]} is missing; every {s} clip will be marked missing', flush=True)
        print(f'encoding {len(paths)} clips with a frozen {args.audio_model} on {device}')
        vecs, missing = encode_audio(paths, args.audio_model, device, args.max_audio_seconds)
        scatter('audio', vecs)
        scatter('audio_missing', missing)
        out['meta'].update(audio_model=args.audio_model, max_audio_seconds=args.max_audio_seconds, audio_missing=int(missing.sum()))
        print(f'  {int(missing.sum())} of {len(paths)} clips missing or unreadable')
    if 'video' in blocks:
        paths = [args.media / MEDIA_DIRS[s] / f'dia{d}_utt{u}.mp4' for s, d, u, _ in order]
        print(f'encoding {len(paths)} clips, {args.frames} frames each, with a frozen {args.video_model} on {device}')
        vecs, missing = encode_video(paths, args.video_model, device, args.frames, args.workers)
        scatter('video', vecs)
        scatter('video_missing', missing)
        out['meta'].update(video_model=args.video_model, video_frames=args.frames, video_pooling='mean of whole-frame image embeddings', video_missing=int(missing.sum()))
        print(f'  {int(missing.sum())} of {len(paths)} clips missing or unreadable')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'wb') as fh:
        pickle.dump(out, fh)
    print(f"{', '.join((f'{s}: {len(out[s])} dialogues' for s in SPLITS))}; written to {args.out}  ({out['meta']})")
if __name__ == '__main__':
    main()
