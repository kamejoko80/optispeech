#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path


def read_id_text(txt_path: Path):
    items = []
    bad = 0
    for ln in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        parts = s.split(None, 1)
        if len(parts) < 2:
            bad += 1
            continue
        uid, text = parts[0].strip(), parts[1].strip()
        if not uid or not text:
            bad += 1
            continue
        items.append((uid, text))
    return items, bad


def write_metadata_csv(out_csv: Path, rows):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|", quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        for uid, text in rows:
            w.writerow([uid, text])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True, help="Source wav directory (contains *.wav)")
    ap.add_argument("--text_file", required=True, help="Source train_parallel.txt (ID <space> text...)")
    ap.add_argument("--out_dir", required=True, help="Output root, e.g. datasets/hi-fi-captain_optispeech")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--val_count", type=int, default=500)
    g.add_argument("--val_ratio", type=float, default=None)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--strict", action="store_true", help="Fail if any wav is missing")
    args = ap.parse_args()

    wav_dir = Path(args.wav_dir)
    txt_path = Path(args.text_file)
    out_dir = Path(args.out_dir)

    items, bad_lines = read_id_text(txt_path)
    seen = set()
    dedup = []
    dup = 0
    for uid, text in items:
        if uid in seen:
            dup += 1
            continue
        seen.add(uid)
        dedup.append((uid, text))
    items = dedup

    exists = []
    missing = 0
    for uid, text in items:
        if (wav_dir / f"{uid}.wav").is_file():
            exists.append((uid, text))
        else:
            missing += 1

    if args.strict and missing:
        raise SystemExit(f"Missing wav files: {missing}")

    rnd = random.Random(args.seed)
    rnd.shuffle(exists)

    if args.val_ratio is not None:
        if not (0.0 < args.val_ratio < 1.0):
            raise SystemExit("--val_ratio must be between 0 and 1")
        val_n = max(1, int(round(len(exists) * args.val_ratio)))
    else:
        val_n = int(args.val_count)
        if val_n < 1:
            val_n = 1
        if val_n > len(exists):
            val_n = max(1, int(round(len(exists) * 0.04)))

    val_rows = exists[:val_n]
    train_rows = exists[val_n:]

    train_root = out_dir / "train"
    val_root = out_dir / "val"
    (train_root / "wav").mkdir(parents=True, exist_ok=True)
    (val_root / "wav").mkdir(parents=True, exist_ok=True)

    write_metadata_csv(train_root / "metadata.csv", train_rows)
    write_metadata_csv(val_root / "metadata.csv", val_rows)

    print(f"text_file: {txt_path}")
    print(f"wav_dir:    {wav_dir}")
    print(f"out_dir:    {out_dir}")
    print(f"parsed: {len(items)}  bad_lines: {bad_lines}  duplicates_dropped: {dup}")
    print(f"wav_missing: {missing}  wav_found: {len(exists)}")
    print(f"train: {len(train_rows)}  val: {len(val_rows)}")
    print("Done (metadata only). Next run mike_link_wavs.py to populate wav/.")


if __name__ == "__main__":
    main()
