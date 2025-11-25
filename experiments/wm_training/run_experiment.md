# How to run these experiments

This document explains how to set up the environment, download the dataset/checkpoint, and run the training scripts for our experiments.

## 1. Clone the repository

```bash
git clone https://github.com/rbalestr-lab/stable-worldmodel.git
cd stable-worldmodel
```

## 2. Create a conda environment and install the package

Python 3.10 recommended. Create and activate a conda environment, then install the package in editable mode.

```bash
conda create --name swm python=3.10 -y
conda activate swm
pip install -e .
```

## 3. (Optional) Change the cache directory

By default, checkpoints and datasets are stored under `~/.stable-worldmodel/`. To change this behavior, add the following to your `~/.bashrc` or shell profile (replace the path):

```bash
export STABLEWM_HOME="/path/to/stable-worldmodel-cache"
```

After editing your profile, reload it (e.g. `source ~/.bashrc`) or open a new terminal.

## 4. Download the PushT dataset and the dino-wm ckpt

Install the google drive downloader package:

```bash
pip install gdown
```

Download the files using their Drive IDs and extract them to your chosen cache directory (or the default `~/.stable-worldmodel/`). Replace `/path/to/stable-worldmodel-cachedir` with your actual cache path.

```bash
gdown --id 11sWn9i2XT8n9wjry84zKmenr08EakaEH
gdown --id 1TBGB30GBQzzN8cdpQJsOCQEpkrPga7pM

# Example: extract into the cache directory
tar --use-compress-program=unzstd -xvf dataset_train.tar.zst -C /path/to/stable-worldmodel-cachedir
tar --use-compress-program=unzstd -xvf dataset_val.tar.zst -C /path/to/stable-worldmodel-cachedir
```

Notes:

- The repository expects the dataset and checkpoints to be accessible from the cache directory defined by `STABLEWM_HOME` (default `~/.stable-worldmodel/`).
- If the downloaded filenames differ, adjust the `tar` commands accordingly.

## 5. Run training experiments

**Important: Make sure you are on the `experiments` branch before running the experiments:**

```bash
git checkout experiments
```

Run the training script from the repository root. Below are several example commands for different backbones. Adjust `backbone` and `output_model_name` as needed.

```bash
python experiments/wm_training/run.py --config-name=pusht.yaml --multirun "backbone=glob(*)" launcher=your_name
```
