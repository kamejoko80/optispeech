import argparse
from pathlib import Path

def read_ljspeech_metadata(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            fid = parts[0].strip()
            text = (parts[2] if len(parts) >= 3 else parts[1]).strip()
            rows.append((fid, text))
    return rows

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        dst.symlink_to(src)
    except Exception:
        import shutil
        shutil.copy2(src, dst)

def write_split(out_dir: Path, rows, wavs_dir: Path):
    ensure_dir(out_dir / "wav")
    meta_path = out_dir / "metadata.csv"
    with meta_path.open("w", encoding="utf-8") as f:
        for fid, text in rows:
            wav_src = wavs_dir / f"{fid}.wav"
            wav_dst = out_dir / "wav" / f"{fid}.wav"
            if not wav_src.exists():
                raise FileNotFoundError(str(wav_src))
            link_or_copy(wav_src, wav_dst)
            f.write(f"{fid}|{text}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ljspeech_dir", required=True)
    ap.add_argument("--out_input_dir", required=True)
    ap.add_argument("--val_size", type=int, default=500)
    args = ap.parse_args()

    ljspeech_dir = Path(args.ljspeech_dir).resolve()
    out_input_dir = Path(args.out_input_dir).resolve()

    meta = ljspeech_dir / "metadata.csv"
    wavs = ljspeech_dir / "wavs_24k"
    if not meta.exists():
        raise FileNotFoundError(str(meta))
    if not wavs.exists():
        raise FileNotFoundError(str(wavs))

    rows = read_ljspeech_metadata(meta)
    if len(rows) <= args.val_size:
        raise ValueError(f"Not enough rows ({len(rows)}) for val_size={args.val_size}")

    train_rows = rows[:-args.val_size]
    val_rows = rows[-args.val_size:]

    write_split(out_input_dir / "train", train_rows, wavs)
    write_split(out_input_dir / "val", val_rows, wavs)

    print("Wrote:")
    print(" ", out_input_dir / "train" / "metadata.csv")
    print(" ", out_input_dir / "val" / "metadata.csv")
    print("Symlink/copy wavs into:")
    print(" ", out_input_dir / "train" / "wav")
    print(" ", out_input_dir / "val" / "wav")
    print("Train rows:", len(train_rows), "Val rows:", len(val_rows))

if __name__ == "__main__":
    main()
