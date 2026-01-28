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
git checkout henry_training
pip install -e .
```

Install some extra dependencies:

```bash
pip install transformers
pip install onnxruntime soundfile numpy
```

### Model Training Guidle (hfc_female-en_us on Linux x86)

Read this discussion: https://github.com/mush42/optispeech/issues/2
OptiSpeech training procedue:


1. Preprocess the dataset

```bash
python -m optispeech.tools.preprocess_dataset \
    <my_dataset_config> \
    <my_dataset_directory> \
    <processed_dataset_directory>
  --output "datasets/hfc_female-en_us-dataset"
```

2. Generate dataset statistics and update the config with them:

```bash
python -m optispeech.tools.generate_data_statistics <my_dataset_config>
```

3. Start training:

```bash
python -m optispeech.train \
    experiment="<my_experiment_config>" \
    model.train_args.evaluate_utmos=false \
    data.batch_size=32 \
    data.num_workers=8 \
    data.train_filelist_path="<processed_dataset_directory>/train.txt" \
    data.valid_filelist_path="<processed_dataset_directory>/val.txt" \
    callbacks.model_checkpoint.every_n_epochs=5  \
    paths.log_dir="<logs_directory>"
```


Download hfc_en-US_F.zip from this URL https://ast-astrec.nict.go.jp/en/release/hi-fi-captain/


Goto the repo root directory:

```bash
cd optispeech
mkdir -p datasets && cd datasets
unzip -q hfc_en-US_F.zip
```

In result we have folder "datasets/hi-fi-captain/en-US/female", next prepare the dataset following the Hydra format:

```bash
cd optispeech
python3 scripts/prepare_hfc_datasets.py \
  --train_text "datasets/hi-fi-captain/en-US/female/text/train_parallel.txt" \
  --train_wav "datasets/hi-fi-captain/en-US/female/wav/train_parallel" \
  --val_text "datasets/hi-fi-captain/en-US/female/text/eval.txt" \
  --val_wav "datasets/hi-fi-captain/en-US/female/wav/eval" \
  --output "datasets/hfc_female-en_us-dataset"
```

Then we have folder name hfc_female-en_us-dataset in Hydra format:

```bash
hfc_female-en_us-dataset
├── train
│   ├── metadata.csv
│   └── wav
└── val
    ├── metadata.csv
    └── wav
```

Preprocess the dataset

```bash
cd optispeech

!rm -rf data/emily
!python3 -m optispeech.tools.preprocess_dataset \
    --format ljspeech \
    emily \
    datasets/hfc_female-en_us-dataset \
    data/emily
```

When done we have a preprocessed data under data/emily folder:

```bash
data/emily
├── data
├── train.txt
└── val.txt
```

Generate dataset statistics:

```bash
cd optispeech
python3 -m optispeech.tools.generate_data_statistics emily
```

When completed it generate data statistics in a json file, update the data statistics into "optispeech/configs/data/emily.yaml"
The content is something like:

```bash
data_statistics:
  pitch_min: 1e-06
  pitch_max: 622.776062
  pitch_mean: 168.61853
  pitch_std: 108.118469
  energy_min: 0.052268
  energy_max: 457.190826
  energy_mean: 76.248787
  energy_std: 66.543098
  mel_mean: -4.363451
  mel_std: 2.479436
```

Start training:


1. Firs training:


```bash
python3 -m optispeech.train \
    experiment="emily" \
    ++data.train_filelist_path="data/emily/train.txt" \
    ++data.valid_filelist_path="data/emily/val.txt" \
    ++data.batch_size=8 \
    ++data.num_workers=1 \
    ++model.train_args.evaluate_utmos=false \
    ++model.train_args.evaluate_pesq=false \
    ++trainer.max_steps=300000 \
    ++callbacks.model_checkpoint.every_n_epochs=2 \
    ++callbacks.model_checkpoint.save_last=true
```

2. Second or more times training:

```bash
python3 -m optispeech.train \
    experiment="emily" \
    ++data.train_filelist_path="data/emily/train.txt" \
    ++data.valid_filelist_path="data/emily/val.txt" \
    ++data.batch_size=8 \
    ++data.num_workers=1 \
    ++model.train_args.evaluate_utmos=false \
    ++model.train_args.evaluate_pesq=false \
    ++trainer.max_steps=300000 \
    ++callbacks.model_checkpoint.every_n_epochs=2 \
    ++callbacks.model_checkpoint.save_last=true \
    ckpt_path="logs/train/emily/runs/2026-01-27_11-16-03/checkpoints/last.ckpt"
```

### Tensor Dashboard Monitoring

Run tensorboard to open a webserver to easier monitor the training progress:

```bash
cd optispeech
pip install tensorboard
tensorboard --logdir=logs --port=6006
```

On web browser open http://localhost:6006
   