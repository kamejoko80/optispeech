#!/usr/bin/env python3
from pathlib import Path

SRC_WAVS = Path("datasets/LJSpeech-1.1/wavs").resolve()
OUT = Path("datasets/LJSpeech-1.1_optispeech").resolve()

def link_split(split: str):
    meta = (OUT / split / "metadata.csv").read_text(encoding="utf-8").splitlines()
    wav_dir = OUT / split / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for ln in meta:
        if not ln.strip():
            continue
        utt_id = ln.split("|", 1)[0]
        src = SRC_WAVS / f"{utt_id}.wav"
        dst = wav_dir / f"{utt_id}.wav"
        if not dst.exists():
            dst.symlink_to(src)
        n += 1
    print(f"{split}: linked {n} wavs into {wav_dir}")

link_split("train")
link_split("val")
