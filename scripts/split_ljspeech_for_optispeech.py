#!/usr/bin/env python3
from pathlib import Path
import random

SRC = Path("datasets/LJSpeech-1.1/metadata.csv")
OUT = Path("datasets/LJSpeech-1.1_optispeech")
SEED = 1234
VAL_COUNT = 500  # adjust (e.g., 300/500). LJSpeech has ~13100 lines.

lines = SRC.read_text(encoding="utf-8").splitlines()
lines = [ln for ln in lines if ln.strip()]

random.Random(SEED).shuffle(lines)

val = lines[:VAL_COUNT]
train = lines[VAL_COUNT:]

(OUT / "train").mkdir(parents=True, exist_ok=True)
(OUT / "val").mkdir(parents=True, exist_ok=True)

(OUT / "train" / "metadata.csv").write_text("\n".join(train) + "\n", encoding="utf-8")
(OUT / "val" / "metadata.csv").write_text("\n".join(val) + "\n", encoding="utf-8")

print(f"train: {len(train)} lines -> {OUT/'train'/'metadata.csv'}")
print(f"val:   {len(val)} lines -> {OUT/'val'/'metadata.csv'}")
