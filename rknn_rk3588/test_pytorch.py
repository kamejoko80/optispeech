import time
import torch
import soundfile as sf
from optispeech.model import OptiSpeech

def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x

def pick_attr(obj, names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Using device:", device)

ckpt_path = "./models/checkpoints/lightspeech/en-us/mike-checkpoint_epoch-729_step-305000.ckpt"
model = OptiSpeech.load_from_checkpoint(ckpt_path, map_location=device)
model = model.to(device).eval()

sentence = (
    "A rainbow is a meteorological phenomenon that is caused by reflection, "
    "refraction and dispersion of light in water droplets resulting in a spectrum "
    "of light appearing in the sky."
)

if device.type == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()

with torch.no_grad():
    inference_inputs = model.prepare_input(sentence)
    inference_inputs = to_device(inference_inputs, device)

    outs = model.synthesise(inference_inputs)
    wav_t = outs.wav

if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()

wav = wav_t.detach().float().cpu().numpy().squeeze()

sf.write("output.wav", wav, model.sample_rate)

audio_sec = float(len(wav)) / float(model.sample_rate)
wall_sec = t1 - t0
rtf = wall_sec / audio_sec if audio_sec > 0 else float("inf")

print(f"Wrote output.wav @ {model.sample_rate} Hz")
print(f"Audio length: {audio_sec:.3f} s")
print(f"Wall time:    {wall_sec:.3f} s")
print(f"RTF:          {rtf:.3f}")
