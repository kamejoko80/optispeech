from pathlib import Path
import csv

root = Path("datasets/LJSpeech-1.1_optispeech")

def rewrite_2col(meta: Path, use_col: int):
    # use_col = 1 -> keep original transcript (2nd column)
    # use_col = 2 -> keep normalized transcript (3rd column)
    tmp = meta.with_suffix(".csv.tmp")
    kept = 0

    with meta.open("r", encoding="utf-8", newline="") as f, tmp.open("w", encoding="utf-8", newline="") as g:
        r = csv.reader(f, delimiter="|")
        w = csv.writer(g, delimiter="|", lineterminator="\n")
        for row in r:
            if not row or len(row) <= use_col:
                continue
            utt_id = row[0].strip()
            text = row[use_col].strip()
            if utt_id and text:
                w.writerow([utt_id, text])
                kept += 1

    tmp.replace(meta)
    return kept

for split in ("train", "val"):
    meta = root / split / "metadata.csv"
    if not meta.exists():
        raise SystemExit(f"Missing: {meta}")
    n = rewrite_2col(meta, use_col=1)  # keep column-2 text; change to 2 if you prefer normalized text
    print(f"{split}: rewrote {n} lines -> {meta}")
