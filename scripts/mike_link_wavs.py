#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
from pathlib import Path


def iter_ids_from_metadata(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="|")
        for row in r:
            if not row:
                continue
            uid = row[0].strip()
            if uid:
                yield uid


def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def link_one(src: Path, dst: Path, mode: str):
    ensure_parent(dst)
    if dst.exists():
        return "exists"

    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return "linked"
    if mode == "hardlink":
        os.link(src, dst)
        return "linked"
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copied"

    raise ValueError("mode must be symlink|hardlink|copy")


def process_split(split_name: str, out_root: Path, wav_src_dir: Path, mode: str, ext: str):
    meta = out_root / split_name / "metadata.csv"
    out_wav_dir = out_root / split_name / "wav"
    out_wav_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    missing = 0
    exists = 0
    copied = 0
    linked = 0

    for uid in iter_ids_from_metadata(meta):
        src = wav_src_dir / f"{uid}{ext}"
        dst = out_wav_dir / f"{uid}{ext}"
        if not src.exists():
            missing += 1
            continue
        res = link_one(src, dst, mode)
        kept += 1
        if res == "exists":
            exists += 1
        elif res == "copied":
            copied += 1
        else:
            linked += 1

    print(f"[{split_name}] metadata: {meta}")
    print(f"[{split_name}] wav_src:  {wav_src_dir}")
    print(f"[{split_name}] wav_out:  {out_wav_dir}")
    print(f"[{split_name}] kept: {kept}  missing_wav: {missing}  already_exists: {exists}  linked: {linked}  copied: {copied}")
    return kept, missing


def main():
    ap = argparse.ArgumentParser(description="Link/copy wavs into OptiSpeech train/val wav folders based on metadata.csv IDs.")
    ap.add_argument("--out_root", required=True, help="datasets/hi-fi-captain_optispeech")
    ap.add_argument("--wav_src", required=True, help='datasets/hi-fi-captain/en-US/male/wav/train_parallel')
    ap.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    ap.add_argument("--ext", default=".wav")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    wav_src = Path(args.wav_src)

    process_split("train", out_root, wav_src, args.mode, args.ext)
    process_split("val", out_root, wav_src, args.mode, args.ext)


if __name__ == "__main__":
    main()
