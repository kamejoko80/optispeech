#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser(
        description="Filter OptiSpeech-prepared filelists by audio duration using .npz wav length."
    )
    ap.add_argument("--root", required=True,
                    help="Dataset root, e.g. data/LJSpeech-1.1")
    ap.add_argument("--max_s", type=float, required=True,
                    help="Max allowed audio duration (seconds).")
    ap.add_argument("--sr", type=int, default=24000,
                    help="Sample rate used when preparing dataset (default: 24000).")

    ap.add_argument("--train_in", default="train.txt", help="Train filelist name inside root.")
    ap.add_argument("--val_in", default="val.txt", help="Val filelist name inside root.")
    ap.add_argument("--train_out", default="train.filter.txt", help="Output train filelist name inside root.")
    ap.add_argument("--val_out", default="val.filter.txt", help="Output val filelist name inside root.")

    return ap.parse_args()

def read_lines(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

def resolve_base(root: Path, item: str):
    # item can be:
    # - "data/LJ001-0001" (relative to root)
    # - "LJ001-0001" (we'll try root/data/)
    # - absolute path to base
    ip = Path(item)
    if ip.is_absolute():
        return ip

    # If it's already like "data/xxx"
    cand1 = root / ip
    if (cand1.with_suffix(".npz")).exists() or (cand1.with_suffix(".json")).exists():
        return cand1

    # Otherwise assume it's under root/data/
    cand2 = root / "data" / ip
    return cand2

def wav_seconds_from_npz(npz_path: Path, sr: int):
    with np.load(npz_path, allow_pickle=False) as arr:
        if "wav" not in arr:
            raise KeyError(f"Missing 'wav' key in {npz_path}")
        wav_len = int(arr["wav"].shape[-1])
    return wav_len / float(sr)

def filter_filelist(root: Path, in_name: str, out_name: str, max_s: float, sr: int):
    in_path = root / in_name
    out_path = root / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = read_lines(in_path)

    kept = 0
    dropped = 0
    missing_npz = 0
    missing_json = 0
    bad = 0

    with out_path.open("w", encoding="utf-8") as g:
        for item in lines:
            base = resolve_base(root, item)
            npz_path = base.with_suffix(".npz")
            json_path = base.with_suffix(".json")

            if not npz_path.exists():
                missing_npz += 1
                dropped += 1
                continue
            if not json_path.exists():
                missing_json += 1
                dropped += 1
                continue

            try:
                dur = wav_seconds_from_npz(npz_path, sr)
            except Exception:
                bad += 1
                dropped += 1
                continue

            if dur <= max_s:
                g.write(item + "\n")
                kept += 1
            else:
                dropped += 1

    print(f"[{in_name} -> {out_name}]")
    print(f"  total:        {len(lines)}")
    print(f"  kept:         {kept}")
    print(f"  dropped:      {dropped}")
    print(f"  missing_npz:  {missing_npz}")
    print(f"  missing_json: {missing_json}")
    print(f"  bad_npz:      {bad}")
    print(f"  max_s:        {max_s}")
    print(f"  sr:           {sr}")
    print()

def main():
    args = parse_args()
    root = Path(args.root)

    filter_filelist(root, args.train_in, args.train_out, args.max_s, args.sr)
    filter_filelist(root, args.val_in, args.val_out, args.max_s, args.sr)

if __name__ == "__main__":
    main()
