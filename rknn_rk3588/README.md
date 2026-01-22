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

### Model Training Guidle (Linux x86)

Goto the repo root directory:

```bash
cd optispeech
mkdir -p datasets && cd datasets
wget -O LJSpeech-1.1.tar.bz2 https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2
```

Resample the wav files to 24K:

```bash
cd ..
python3 scripts/resample_ljspeech_to_24k.py
```

Covert ljspeech datasets to Hydra fortmat:

```bash
rm -rf data/hfc_female-en_us/input
python3 scripts/convert_ljspeech_to_optispeech_input.py \
  --ljspeech_dir datasets/LJSpeech-1.1 \
  --out_input_dir data/hfc_female-en_us/input \
  --val_size 500
```

Run the preprocess_dataset:

```bash
rm -rf data/hfc_female-en_us/output
python3 -m optispeech.tools.preprocess_dataset \
  --format ljspeech \
  -w 4 -b 1 \
  hfc_female-en_us \
  data/hfc_female-en_us/input \
  data/hfc_female-en_us/output
```

For NVIDIA GeForce GTX 1650 (4GB VRAM) need to limit the datasets to avoid CUDA OOM:

```bash
python3 scripts/make_safe_filelists.py \
  --train_in data/hfc_female-en_us/output/train.txt \
  --val_in   data/hfc_female-en_us/output/val.txt \
  --train_out data/hfc_female-en_us/output/train.safe.txt \
  --val_out   data/hfc_female-en_us/output/val.safe.txt \
  --min_s 0.2 --max_s 6.0 --sr 24000 \
  --max_mel_frames 360 --max_phoneme 160
```

Start training with a limited 300000 steps:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:24,garbage_collection_threshold:0.8

python3 -m optispeech.train experiment=hfc_female-en_us \
  run_name=opti_hfc_female_gpu \
  data.train_filelist_path=data/hfc_female-en_us/output/train.safe.txt \
  data.valid_filelist_path=data/hfc_female-en_us/output/val.safe.txt \
  data.batch_size=1 \
  data.num_workers=4 \
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
  callbacks.model_checkpoint.save_last=true 
```

Run this to see the resolved config for your experiment:

```bash
python3 -m optispeech.train experiment=hfc_female-en_us --cfg job --resolve
```  