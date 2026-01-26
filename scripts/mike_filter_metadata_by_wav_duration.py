#!/usr/bin/env python3
import argparse
import csv
import shutil
from pathlib import Path

def wav_duration_seconds(wav_path: Path) -> float | None:
    # Prefer soundfile (supports more WAV encodings); fall back to wave (PCM).
    try:
        import soundfile as sf
        info = sf.info(str(wav_path))
        if info.samplerate <= 0:
            return None
        return float(info.frames) / float(info.samplerate)
    except Exception:
        pass

    try:
        import wave
        with wave.open(str(wav_path), "rb") as w:
            sr = w.getframerate()
            n = w.getnframes()
            if sr <= 0:
                return None
            return float(n) / float(sr)
    except Exception:
        return None

def filter_one_split(split_dir: Path, max_s: float, ext: str, make_backup: bool, dry_run: bool):
    meta_path = split_dir / "metadata.csv"
    wav_dir = split_dir / "wav"

    if not meta_path.exists():
        raise SystemExit(f"metadata.csv not found: {meta_path}")
    if not wav_dir.exists():
        raise SystemExit(f"wav dir not found: {wav_dir}")

    rows_in = []
    with meta_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="|")
        for row in r:
            rows_in.append(row)

    kept_rows = []
    kept = 0
    dropped = 0
    missing = 0
    bad = 0
    unreadable = 0

    for row in rows_in:
        if not row or len(row) < 2:
            bad += 1
            continue

        uid = (row[0] or "").strip()
        text = (row[1] or "").strip()
        if not uid or not text:
            bad += 1
            continue

        wav_path = wav_dir / f"{uid}{ext}"
        if not wav_path.exists():
            missing += 1
            continue

        dur = wav_duration_seconds(wav_path)
        if dur is None:
            unreadable += 1
            continue

        if dur > max_s:
            dropped += 1
            continue

        kept_rows.append([uid, text])  # keep EXACTLY 2 columns
        kept += 1

    print(f"[{split_dir.name}] metadata: {meta_path}")
    print(f"[{split_dir.name}] wav_dir:   {wav_dir}")
    print(f"[{split_dir.name}] max_s:     {max_s}")
    print(f"[{split_dir.name}] kept: {kept}")
    print(f"[{split_dir.name}] dropped_too_long: {dropped}")
    print(f"[{split_dir.name}] missing_wav: {missing}")
    print(f"[{split_dir.name}] unreadable_wav: {unreadable}")
    print(f"[{split_dir.name}] bad_rows: {bad}")
    print(f"[{split_dir.name}] dry_run: {dry_run}")
    print("")

    if dry_run:
        return

    if make_backup:
        bak = meta_path.with_suffix(".csv.bak")
        shutil.copy2(meta_path, bak)

    tmp_path = meta_path.with_suffix(".csv.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(
            f,
            delimiter="|",
            lineterminator="\n",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
        )
        for uid, text in kept_rows:
            w.writerow([uid, text])

    tmp_path.replace(meta_path)

def main():
    ap = argparse.ArgumentParser(
        description="Filter OptiSpeech train/val metadata.csv by wav duration > max_s (updates metadata.csv in place)."
    )
    ap.add_argument("--root", required=True, help="Dataset root (contains train/ and val/)")
    ap.add_argument("--max_s", type=float, required=True, help="Drop utterances with duration > max_s seconds (e.g. 5.0)")
    ap.add_argument("--ext", default=".wav", help="Audio extension (default .wav)")
    ap.add_argument("--backup", action="store_true", help="Create metadata.csv.bak before overwriting")
    ap.add_argument("--dry-run", action="store_true", help="Only print stats; do not overwrite or backup files")
    args = ap.parse_args()

    root = Path(args.root)
    train_dir = root / "train"
    val_dir = root / "val"

    filter_one_split(train_dir, args.max_s, args.ext, args.backup, args.dry_run)
    filter_one_split(val_dir, args.max_s, args.ext, args.backup, args.dry_run)

if __name__ == "__main__":
    main()
