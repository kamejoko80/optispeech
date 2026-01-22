from pathlib import Path
import soundfile as sf
import librosa

src = Path("datasets/LJSpeech-1.1/wavs")
dst = Path("datasets/LJSpeech-1.1/wavs_24k")
dst.mkdir(parents=True, exist_ok=True)

target_sr = 24000

wavs = sorted(src.glob("*.wav"))
print("found", len(wavs), "wavs")
for i, wp in enumerate(wavs, 1):
    y, sr = sf.read(wp, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y.astype("float32"), orig_sr=sr, target_sr=target_sr)
    outp = dst / wp.name
    sf.write(outp, y, target_sr)
    if i % 500 == 0:
        print("resampled", i)
print("done ->", dst)
