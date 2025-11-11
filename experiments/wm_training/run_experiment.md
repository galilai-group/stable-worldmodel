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
gdown --id 1f-3y5kWVKwtqH5YGNPhDP-BzSQnomZnV
gdown --id 1je3nRgGERd-U_4dHyNcwMpJW66w8_YBQ

# Example: extract into the cache directory
tar --use-compress-program=unzstd -xvf dataset_train.tar.zst -C /path/to/stable-worldmodel-cachedir
tar --use-compress-program=unzstd -xvf dataset_val.tar.zst -C /path/to/stable-worldmodel-cachedir
```

Notes:

- The repository expects the dataset and checkpoints to be accessible from the cache directory defined by `STABLEWM_HOME` (default `~/.stable-worldmodel/`).
- If the downloaded filenames differ, adjust the `tar` commands accordingly.

## 5. Run training experiments

Run the training script from the repository root. Below are several example commands for different backbones. Adjust `backbone` and `output_model_name` as needed.

```bash
# ResNet example
python experiments/wm_training/run.py backbone=microsoft/resnet-50 output_model_name=resnet50

# ViT example
python experiments/wm_training/run.py backbone=google/vit-base-patch16-224 output_model_name=vit_base

# DINO v1 examples
python experiments/wm_training/run.py backbone=facebook/dino-vits16 output_model_name=dino_vits16

# DINOv2
python experiments/wm_training/run.py backbone=facebook/dinov2-small output_model_name=dinov2_small

# DINOv3 (example)
python experiments/wm_training/run.py backbone=facebook/dinov3-vits16-pretrain-lvd1689m output_model_name=dinov3_vits16

# MAE
python experiments/wm_training/run.py backbone=facebook/vit-mae-base output_model_name=vit_mae_base

# IJEPA
python experiments/wm_training/run.py backbone=facebook/ijepa_vith14_22k output_model_name=ijepa_vith14_22k

# CLIP (example)
python experiments/wm_training/run.py backbone=timm/vit_base_patch32_clip_224.metaclip_400m output_model_name=metaclip_vit_base
```
