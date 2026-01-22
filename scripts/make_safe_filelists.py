#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


DELIMS = ["|", "\t", ",", ";"]


@dataclass
class DetectResult:
    delim: Optional[str]
    path_col: Optional[int]
    mode: str  # "base" (expects .npz/.json) or "audio" (not used here)
    bases: List[Path]


def build_bases(in_file: Path) -> List[Path]:
    cwd = Path.cwd().resolve()
    bases = []

    def add(p: Path):
        p = p.resolve()
        if p not in bases:
            bases.append(p)

    add(cwd)
    add(cwd / "data")
    add(cwd / "optispeech")
    add(cwd / "optispeech" / "data")
    add(in_file.resolve().parent)
    add(in_file.resolve().parent.parent)
    add(in_file.resolve().parent.parent.parent)

    # If you're inside optispeech/ already (common), also add repo root guesses
    add(cwd.parent)
    add(cwd.parent / "data")
    return bases


def read_nonempty_lines(p: Path) -> List[str]:
    return [ln.strip("\n") for ln in p.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]


def split_line(line: str, delim: Optional[str]) -> List[str]:
    if delim is None:
        return [line]
    return line.split(delim)


def guess_delim(lines: List[str]) -> Optional[str]:
    best = None
    best_hits = -1
    for d in DELIMS:
        hits = 0
        for ln in lines[:400]:
            if len(ln.split(d)) >= 2:
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best = d
    if best_hits <= 0:
        return None
    return best


def resolve_base_path(raw: str, bases: List[Path]) -> Optional[Path]:
    s = raw.strip().strip('"').strip("'")
    if not s:
        return None

    p = Path(s)
    # If someone accidentally provided .npz/.json path, normalize to base
    if p.suffix.lower() in [".npz", ".json"]:
        p = p.with_suffix("")

    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        for b in bases:
            candidates.append((b / p).resolve())

    for c in candidates:
        npz = c.with_suffix(".npz")
        jsn = c.with_suffix(".json")
        if npz.exists() and jsn.exists():
            return c
    return None


def score_path_col(lines: List[str], delim: Optional[str], col: int, bases: List[Path]) -> int:
    score = 0
    for ln in lines[:600]:
        cols = split_line(ln, delim)
        if col >= len(cols):
            continue
        if resolve_base_path(cols[col], bases) is not None:
            score += 1
    return score


def detect_format(in_file: Path) -> DetectResult:
    lines = read_nonempty_lines(in_file)
    bases = build_bases(in_file)

    delim = guess_delim(lines)
    cols0 = split_line(lines[0], delim)
    max_cols = max(len(split_line(ln, delim)) for ln in lines[:400])

    # Find which column looks like the "base path" that has .npz+.json
    best_col = None
    best_score = -1
    for c in range(max_cols):
        sc = score_path_col(lines, delim, c, bases)
        if sc > best_score:
            best_score = sc
            best_col = c

    # If even best score is zero, we might be in "single column base paths" case (delim None)
    if best_score <= 0:
        # Try single-column mode
        if delim is not None:
            # maybe delimiter guessed wrong; fallback to no delimiter
            delim = None
        best_col = 0
        best_score = score_path_col(lines, delim, 0, bases)

    mode = "base"  # This script is designed for preprocessed outputs (.npz/.json)

    return DetectResult(delim=delim, path_col=best_col if best_score > 0 else None, mode=mode, bases=bases)


def load_meta_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def load_npz(p: Path) -> Optional[dict]:
    try:
        arr = np.load(str(p), allow_pickle=False)
        return {k: arr[k] for k in arr.files}
    except Exception:
        return None


@dataclass
class Report:
    in_file: str
    out_file: str
    kept: int
    dropped: int
    missing_npz_json: int
    bad_lines: int
    too_short: int
    too_long: int
    min_duration_s: float
    max_duration_s: float
    bases: List[str]
    detected_delim: str
    detected_path_col: int


def make_safe_filelist(
    in_file: Path,
    out_file: Path,
    min_duration_s: float,
    max_duration_s: float,
    max_mel_frames: int,
    max_phoneme: int,
    sample_rate: int,
    rewrite_paths: bool,
) -> Report:
    det = detect_format(in_file)
    lines = read_nonempty_lines(in_file)

    out_file.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    missing_npz_json = 0
    bad_lines = 0
    too_short = 0
    too_long = 0

    if det.path_col is None:
        # Nothing can be resolved -> write empty and report
        out_file.write_text("", encoding="utf-8")
        return Report(
            in_file=str(in_file),
            out_file=str(out_file),
            kept=0,
            dropped=len(lines),
            missing_npz_json=len(lines),
            bad_lines=0,
            too_short=0,
            too_long=0,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            bases=[str(b) for b in det.bases],
            detected_delim=repr(det.delim),
            detected_path_col=-1,
        )

    with in_file.open("r", encoding="utf-8", errors="replace") as f, out_file.open("w", encoding="utf-8") as g:
        for raw_ln in f:
            ln = raw_ln.rstrip("\n")
            if not ln.strip():
                continue

            cols = split_line(ln, det.delim)
            if det.path_col >= len(cols):
                bad_lines += 1
                dropped += 1
                continue

            base = resolve_base_path(cols[det.path_col], det.bases)
            if base is None:
                missing_npz_json += 1
                dropped += 1
                continue

            npz_path = base.with_suffix(".npz")
            json_path = base.with_suffix(".json")

            meta = load_meta_json(json_path)
            arr = load_npz(npz_path)
            if meta is None or arr is None:
                bad_lines += 1
                dropped += 1
                continue

            wav = arr.get("wav", None)
            mel = arr.get("mel", None)
            if wav is None or mel is None:
                bad_lines += 1
                dropped += 1
                continue

            try:
                wav_len = int(wav.shape[-1])
                mel_len = int(mel.shape[-1])
            except Exception:
                bad_lines += 1
                dropped += 1
                continue

            dur_s = wav_len / float(sample_rate)
            if dur_s < min_duration_s:
                too_short += 1
                dropped += 1
                continue
            if dur_s > max_duration_s:
                too_long += 1
                dropped += 1
                continue

            phoneme_ids = meta.get("phoneme_ids", [])
            xlen = len(phoneme_ids) if isinstance(phoneme_ids, list) else 0

            if mel_len > max_mel_frames or xlen > max_phoneme:
                dropped += 1
                continue

            if rewrite_paths:
                # rewrite only the detected base path column, preserve everything else & delimiter
                cols[det.path_col] = str(base)
                out_ln = cols[0] if det.delim is None else det.delim.join(cols)
            else:
                out_ln = ln

            g.write(out_ln + "\n")
            kept += 1

    return Report(
        in_file=str(in_file),
        out_file=str(out_file),
        kept=kept,
        dropped=dropped,
        missing_npz_json=missing_npz_json,
        bad_lines=bad_lines,
        too_short=too_short,
        too_long=too_long,
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
        bases=[str(b) for b in det.bases],
        detected_delim=repr(det.delim),
        detected_path_col=int(det.path_col),
    )


def print_report(r: Report):
    print(f"in_file: {r.in_file}")
    print(f"out_file: {r.out_file}")
    print(f"kept: {r.kept}")
    print(f"dropped: {r.dropped}")
    print(f"missing_npz_json: {r.missing_npz_json}")
    print(f"bad_lines: {r.bad_lines}")
    print(f"too_short: {r.too_short}")
    print(f"too_long: {r.too_long}")
    print(f"min_duration_s: {r.min_duration_s}")
    print(f"max_duration_s: {r.max_duration_s}")
    print(f"detected_delim: {r.detected_delim}")
    print(f"detected_path_col: {r.detected_path_col}")
    print("bases:")
    for b in r.bases:
        print(f"  - {b}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_in", required=True)
    ap.add_argument("--val_in", required=True)
    ap.add_argument("--train_out", required=True)
    ap.add_argument("--val_out", required=True)

    ap.add_argument("--min_s", type=float, default=0.2)
    ap.add_argument("--max_s", type=float, default=8.0)

    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--max_mel_frames", type=int, default=450)
    ap.add_argument("--max_phoneme", type=int, default=180)

    ap.add_argument("--no_rewrite_paths", action="store_true")
    args = ap.parse_args()

    rewrite_paths = not args.no_rewrite_paths

    r1 = make_safe_filelist(
        Path(args.train_in),
        Path(args.train_out),
        min_duration_s=args.min_s,
        max_duration_s=args.max_s,
        max_mel_frames=args.max_mel_frames,
        max_phoneme=args.max_phoneme,
        sample_rate=args.sr,
        rewrite_paths=rewrite_paths,
    )
    print_report(r1)
    print("")

    r2 = make_safe_filelist(
        Path(args.val_in),
        Path(args.val_out),
        min_duration_s=args.min_s,
        max_duration_s=args.max_s,
        max_mel_frames=args.max_mel_frames,
        max_phoneme=args.max_phoneme,
        sample_rate=args.sr,
        rewrite_paths=rewrite_paths,
    )
    print_report(r2)


if __name__ == "__main__":
    main()
