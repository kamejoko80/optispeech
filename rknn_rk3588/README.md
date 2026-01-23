### Install OptiSpeech on Linux x86

```bash
mkdir OptiSpeech
cd OptiSpeech
sudo apt install python3.11 python3.11-venv python3.11-dev
python3.11 -m venv venv
source venv/bin/activate
pip install -U pip
```

Install OptiSpeech from github repo:

```bash
https://github.com/kamejoko80/optispeech.git
cd optispeech
git checkout henry_rk3588
pip install -e .
```

Install some extra dependencies:

```bash
pip install transformers -U
pip install onnxruntime soundfile numpy
```

Test pytorch voice inference with the lightspeech model:


```bash
cd rknn_rk3588
mkdir -p models out

python3 -c '
from huggingface_hub import hf_hub_download
repo_id = "henrydang80/optispeech"
filename = "checkpoints/lightspeech/en-us/mike-checkpoint_epoch-729_step-305000.ckpt"
path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="models", local_dir_use_symlinks=False)
print("Saved to:", path)
'

python3 -c '
from huggingface_hub import hf_hub_download
repo_id = "henrydang80/optispeech"
filename = "checkpoints/lightspeech/en-us/hfc_female-en_us-checkpoint_epoch-2174_step-451407.ckpt"
path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="models", local_dir_use_symlinks=False)
print("Saved to:", path)
'

python3 python3 test_pytorch.py
```

Test onnx voice inference with the lightspeech model:


```bash

python3 -c '
from huggingface_hub import hf_hub_download
repo_id = "henrydang80/optispeech"
filename = "onnx/lightspeech/en-us/mike-step_305k.onnx"
path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="models", local_dir_use_symlinks=False)
print("Saved to:", path)
'

python3 -m optispeech.onnx.infer \
  models/onnx/lightspeech/en-us/mike-step_305k.onnx \
  "Hello! This is OptiSpeech running on x86 Linux." \
  out
```

### Convert ONNX to RKNN (on x86 linux pc)

Open a new terminal, follow the below steps to install RKNN-Toolkit2

```bash
mkdir RKNN-Toolkit2
cd RKNN-Toolkit2
```

Run bash Miniforge3-Linux-x86_64.sh and install in path = $PWD/env

Every time we open a new console we must activate the env:

```bash
source env/bin/activate
```

Create a Conda environment named "RKNN-Toolkit2" with Python 3.8 version:

```bash
conda create -n RKNN-Toolkit2 python=3.8
```

Activate RKNN-Toolkit2:

```bash
> conda activate RKNN-Toolkit2
```

To deactivate:

```bash
> conda deactivate
```

```bash
git clone https://github.com/airockchip/rknn-toolkit2.git
cd rknn-toolkit2
pip install -r rknn-toolkit2/packages/x86_64/requirements_cp310-2.3.2.txt
pip install rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

Covert ONNX to RKNN:

--------------------- T = 64 -----------------------------

```bash
python3 patch_all_ranges.py \
  --in models/onnx/lightspeech/en-us/mike-step_305k.onnx \
  --out mike_T64_patched.onnx \
  --T 64 --frames_per_token 4 --hop 25

python3 bench_optispeech_onnx_cli.py \
  --onnx mike_T64_patched.onnx \
  --T 64 \
  --text "Hello! This is a verification run for mike patched ONNX." \
  --outdir out_onnx \
  --warmup 1 --runs 1

python3 onnx_to_rknn_mike.py --onnx mike_T64_patched.onnx --T 64
```

--------------------- T = 128 -----------------------------

```bash
python3 patch_all_ranges.py \
  --in models/onnx/lightspeech/en-us/mike-step_305k.onnx \
  --out mike_T128_patched.onnx \
  --T 128 --frames_per_token 4 --hop 25

python3 bench_optispeech_onnx_cli.py \
  --onnx mike_T128_patched.onnx \
  --T 128 \
  --text "Hello! This is a verification run for mike patched ONNX." \
  --outdir out_onnx \
  --warmup 1 --runs 1

python3 onnx_to_rknn_mike.py --onnx mike_T128_patched.onnx --T 128
```

--------------------- T = 256 -----------------------------

```bash
python3 patch_all_ranges.py \
  --in models/onnx/lightspeech/en-us/mike-step_305k.onnx \
  --out mike_T256_patched.onnx \
  --T 256 --frames_per_token 4 --hop 25

python3 bench_optispeech_onnx_cli.py \
  --onnx mike_T256_patched.onnx \
  --T 256 \
  --text "Hello! This is a verification run for mike patched ONNX." \
  --outdir out_onnx \
  --warmup 1 --runs 1

python3 onnx_to_rknn_mike.py --onnx mike_T256_patched.onnx --T 256
```

### Install OptiSpeech on RK3588


```bash
mkdir OptiSpeech
cd OptiSpeech
```

Install a local python evnrironment:


```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
```

Run bash Miniforge3-Linux-aarch64.sh and install in path = $PWD/env

Every time we open a new console we must activate the env:

```bash
source env/bin/activate
```

Create a Conda environment named "RKNN-Toolkit2" with Python 3.11 version:

```bash
conda create -n RKNN-Toolkit2 python=3.11
```

Activate RKNN-Toolkit2:

```bash
> conda activate RKNN-Toolkit2
```

To deactivate:

```bash
> conda deactivate
```

Install RKNN-Toolkit2 & OptiSpeech

```bash
pip install -U pip
pip install rknn-toolkit-lite2
```

```bash
git clone https://github.com/kamejoko80/optispeech.git
cd optispeech
git checkout henry_rk3588
pip install -e .
```

Install some extra dependencies:

```bash
pip install -U huggingface_hub
pip install transformers -U
pip install onnxruntime soundfile numpy
```

Test pytorch voice inference with the lightspeech model:

```bash
cd optispeech/rknn_rk3588
mkdir -p models out

python3 -c '
from huggingface_hub import hf_hub_download
repo_id = "henrydang80/optispeech"
filename = "checkpoints/lightspeech/en-us/mike-checkpoint_epoch-729_step-305000.ckpt"
path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="models", local_dir_use_symlinks=False)
print("Saved to:", path)
'

python3 -c '
from huggingface_hub import hf_hub_download
repo_id = "henrydang80/optispeech"
filename = "checkpoints/lightspeech/en-us/hfc_female-en_us-checkpoint_epoch-2174_step-451407.ckpt"
path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="models", local_dir_use_symlinks=False)
print("Saved to:", path)
'

python3 test_pytorch.py
```

Run RKNN with the coverted models:


--------------------- T = 64 -----------------------------

```bash
python3 bench_optispeech_rknn_cli.py \
  --rknn mike_T64.rknn \
  --T 64 \
  --text "Are you aware that dozens of American citizens have been harassed, beaten, and locked up?" \
  --outdir out_rknn \
  --warmup 1 --runs 1
```

--------------------- T = 128 -----------------------------

```bash
python3 bench_optispeech_rknn_cli.py \
  --rknn mike_T128.rknn \
  --T 128 \
  --text "In 2025, multiple reports and investigations confirmed that U.S. Immigration and Customs Enforcement (ICE) detained at least 170 U.S. citizens during immigration enforcement operations." \
  --outdir out_rknn \
  --warmup 1 --runs 1
```

--------------------- T = 256 -----------------------------

```bash
python3 bench_optispeech_rknn_cli.py \
  --rknn mike_T256.rknn \
  --T 256 \
  --text "In 2025, multiple reports and investigations confirmed that U.S. Immigration and Customs Enforcement (ICE) detained at least 170 U.S. citizens during immigration enforcement operations." \
  --outdir out_rknn \
  --warmup 1 --runs 1
```

### Model Training Guidle LJSpeech (Linux x86)

Read this discussion: https://github.com/mush42/optispeech/issues/2

Goto the repo root directory:

```bash
cd optispeech
mkdir -p datasets && cd datasets
wget -O LJSpeech-1.1.tar.bz2 https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2
```

Covert ljspeech datasets to Hydra fortmat:

```bash
cd ..
python3 scripts/split_ljspeech_for_optispeech.py
python3 scripts/link_ljspeech_wavs_for_optispeech.py
python3 scripts/fix_ljspeech_metadata.py
```

Run the preprocess_dataset. If this process is failed with NVIDIA GeForce GTX 1650 (4GB VRAM)
Then we need to filter out the LJSpeech raw data before executing preprocess_dataset script instead.

```bash
rm -rf data/LJSpeech-1.1
python3 -m optispeech.tools.preprocess_dataset \
  --format ljspeech \
  -w 1 -b 1 \
  ljspeech \
  datasets/LJSpeech-1.1_optispeech \
  data/LJSpeech-1.1
```

For NVIDIA GeForce GTX 1650 (4GB VRAM) need to filter out utterances with audio more than a number (for ex. 6.0s) to avoid CUDA OOM:

```bash
python3 scripts/filter_ljspeech_by_audio_len.py --root data/LJSpeech-1.1 --max_s 6.0 --sr 22050
```

Backup and rename new train.txt val.txt:

```bash
mv data/LJSpeech-1.1/train.txt data/LJSpeech-1.1/train_orig.txt
mv data/LJSpeech-1.1/val.txt data/LJSpeech-1.1/val_orig.txt

mv data/LJSpeech-1.1/train.filter.txt data/LJSpeech-1.1/train.txt
mv data/LJSpeech-1.1/val.filter.txt data/LJSpeech-1.1/val.txt
```

Must run data statistics before training:

```bash
python3 -m optispeech.tools.generate_data_statistics ljspeech
```

Start training:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:24,garbage_collection_threshold:0.8

python3 -m optispeech.train experiment=ljspeech \
  data.train_filelist_path="data/LJSpeech-1.1/train.txt" \
  data.valid_filelist_path="data/LJSpeech-1.1/val.txt" \
  data.batch_size=1 \
  data.num_workers=1 \
  data.pin_memory=true \
  model.train_args.gradient_accumulate_batches=64 \
  trainer.accelerator=gpu trainer.devices=1 trainer.precision=16-mixed \
  +trainer.num_sanity_val_steps=0 \
  +trainer.limit_val_batches=0.0 \
  +trainer.max_steps=300000 \
  model.generator.segment_size=2 \
  model.train_args.evaluate_utmos=false \
  model.train_args.evaluate_pesq=false \
  model.train_args.evaluate_periodicity=false \
  callbacks.model_checkpoint.every_n_epochs=4 \
  callbacks.model_checkpoint.save_last=true
```

Parameter meaning:

```
1) export PYTORCH_CUDA_ALLOC_CONF=...

This controls PyTorch’s CUDA memory allocator behavior (to reduce fragmentation and OOM spikes).

    • expandable_segments:True
    Lets the allocator use growable memory segments instead of many fixed chunks. This often reduces fragmentation and “OOM even though free memory exists”.

    • max_split_size_mb:24
    Limits how big a memory block PyTorch is allowed to split into smaller blocks.
    Smaller value (like 24 MB) can reduce fragmentation in some workloads, but sometimes can make allocation slower.
    If you still see fragmentation/OOM, you try values like 32, 64, 128 depending on behavior.

    • garbage_collection_threshold:0.8
    Controls how aggressively the allocator reclaims cached blocks.
    0.8 means: when memory pressure is high (roughly 80% utilization), it will start freeing cached blocks sooner, helping avoid sudden OOM.

2) data.batch_size=1

This is the micro-batch size per training step (per GPU).

    • batch_size=1 means each forward/backward pass uses 1 sample at a time, which is the lowest VRAM setting.
    If you want “effective batch size” bigger than 1, you use gradient accumulation (like model.train_args.gradient_accumulate_batches=64).
    Effective batch size ≈ batch_size * gradient_accumulate_batches * num_gpus
    So 1 * 64 * 1 = 64 effective batch (but slower).

3) data.num_workers=1

This is the number of CPU worker processes used by the PyTorch DataLoader to load data in parallel.

    • Higher num_workers ⇒ faster data loading, but more CPU/RAM usage and sometimes more instability (especially on some systems).

    • num_workers=1 is the safest, most stable option (but can be slower).

    This does not directly reduce GPU VRAM, but it can help avoid system memory pressure / stalls.


4) model.generator.segment_size=8

This is a big VRAM lever.

OptiSpeech trains on audio/feature segments (chunks). segment_size controls the chunk length used in training (not the dataset’s full utterance length).

    • Smaller segment_size ⇒ shorter chunks ⇒ smaller tensors (mel, alignment, conv/transformer activations) ⇒ much lower VRAM.

    • But smaller segments can:

        • slow convergence

        • reduce audio quality early on

        • make training noisier

In your earlier config you used 16; setting 8 is even more VRAM-friendly.

Practical summary for your low-VRAM case

        • batch_size=1 + segment_size=8 are the main VRAM reducers.

        • PYTORCH_CUDA_ALLOC_CONF=... reduces OOM caused by fragmentation, especially near the end of epochs / validation / checkpointing.

        • num_workers=1 is mostly for stability and avoiding CPU/RAM pressure.
```

Run this to see the resolved config for your experiment:

```bash
python3 -m optispeech.train experiment=ljspeech --cfg job --resolve
```

### Model Training Guidle Mike (Linux x86)

Goto the repo root directory:

```bash
cd optispeech
mkdir -p datasets
cp hfc_en-US_M.zip datasets
cd datasets
unzip hfc_en-US_M.zip
cd ..
```

Covert hfc_en-US_M datasets to Hydra fortmat:


```bash
rm -rf datasets/hi-fi-captain_optispeech

python3 scripts/mike_convert_hifi_captain_to_optispeech.py \
  --wav_dir  datasets/hi-fi-captain/en-US/male/wav/train_parallel \
  --text_file datasets/hi-fi-captain/en-US/male/text/train_parallel.txt \
  --out_dir  datasets/hi-fi-captain_optispeech \
  --val_count 500 \
  --seed 1234

python3 scripts/mike_link_wavs.py \
  --out_root datasets/hi-fi-captain_optispeech \
  --wav_src datasets/hi-fi-captain/en-US/male/wav/train_parallel \
  --mode symlink
```

Run the preprocess_dataset. If this process is failed with NVIDIA GeForce GTX 1650 (4GB VRAM)
Then we need to filter out the raw data before executing preprocess_dataset script instead.

```bash
python3 scripts/mike_filter_metadata_by_wav_duration.py \
  --root datasets/hi-fi-captain_optispeech \
  --max_s 5.0 \
  --backup
```

```bash
rm -rf data/mike

python3 -m optispeech.tools.preprocess_dataset \
  mike \
  datasets/hi-fi-captain_optispeech \
  data/mike \
  --format ljspeech \
  -w 4 -b 1
```

Must run data statistics before training:

```bash
python3 -m optispeech.tools.generate_data_statistics mike
```

Start training:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:24,garbage_collection_threshold:0.8

python3 -m optispeech.train experiment=mike-lightspeech \
  data.train_filelist_path="data/mike/train.txt" \
  data.valid_filelist_path="data/mike/val.txt" \
  data.batch_size=1 \
  data.num_workers=1 \
  data.pin_memory=true \
  model.train_args.gradient_accumulate_batches=16 \
  trainer.accelerator=gpu trainer.devices=1 trainer.precision=16-mixed \
  +trainer.num_sanity_val_steps=0 \
  +trainer.limit_val_batches=0.0 \
  +trainer.max_steps=300000 \
  model.generator.segment_size=1 \
  model.train_args.evaluate_utmos=false \
  model.train_args.evaluate_pesq=false \
  model.train_args.evaluate_periodicity=false \
  callbacks.model_checkpoint.every_n_epochs=10 \
  callbacks.model_checkpoint.save_last=true
```  