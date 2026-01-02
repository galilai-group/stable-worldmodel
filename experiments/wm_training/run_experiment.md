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

You might to set the below variables to avoid bugs with stable-pretraining
```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
```

---------------

# PushT
Run the training script from the repository root. Below are several example commands for different backbones.

**all backbones**:
```bash
python experiments/wm_training/run.py --config-name=pusht.yaml --multirun "backbone=glob(*)" launcher=your_name
```

**dinov2 encoder scaling**:
```bash
python experiments/wm_training/run.py --config-name=pusht.yaml --multirun backbone=dinov2_small,dinov2_base,dinov2_large,dinov2_giant launcher=your_name
```

**predictor scaling**:
```bash
python experiments/wm_training/run.py --config-name=pusht.yaml --multirun predictor=tiny,small,base,large launcher=your_name
```
>>>>>>>>>>>>>.

**data all variations**:
```bash
python experiments/wm_training/run.py --config-name=pusht.yaml --multirun backbone=dinov2_small dataset_name=pusht_weak_100_variation_all launcher=your_name
```

**data scaling**:
```bash
python experiments/wm_training/run.py --config-name=pusht.yaml --multirun backbone=dinov2_small dataset_name=pusht_expert_train,pusht_weak_100_variation_all subset_prop=0.1,0.5 launcher=your_name
```

**interaction quality interpolation**:
```bash
python experiments/wm_training/run.py --config-name=pusht.yaml --multirun backbone=dinov2_small dataset_name=pusht_weak_100,pusht_weak_300 launcher=your_name
```

**quality interpolation**:
```bash
python experiments/wm_training/run.py --config-name=pusht.yaml --multirun backbone=dinov2_small dataset_name=pusht_expert_train injected_dataset.names="[pusht_weak_100]" injected_dataset.proportions="[0.95],[0.9],[0.8],[0.5],[0.2]" launcher=your_name -m
```

**variation interpolation**:
```bash
python experiments/wm_training/run.py --config-name=tworoom.yaml --multirun backbone=dinov2_small dataset_name=pusht_expert_train injected_dataset.names="[pusht_weak_100_variation_all]" injected_dataset.proportions="[0.5],[0.1],[0.01]" launcher=your_name -m
```

--------------

# TwoRoom
Run the training script from the repository root. Below are several example commands for different backbones.

**all backbones**:
```bash
python experiments/wm_training/run.py --config-name=tworoom.yaml --multirun "backbone=glob(*)" launcher=your_name
```

**dinov2 encoder scaling**:
```bash
python experiments/wm_training/run.py --config-name=tworoom.yaml --multirun backbone=dinov2_small,dinov2_base,dinov2_large,dinov2_giant launcher=your_name
```

**predictor scaling**:
```bash
python experiments/wm_training/run.py --config-name=tworoom.yaml --multirun predictor=tiny,small,base,large launcher=your_name
```

**data all variations**:
```bash
python experiments/wm_training/run.py --config-name=tworoom.yaml --multirun backbone=dinov2_small dataset_name=tworoom_noisy_variation_all launcher=your_name
```

**data scaling**:
```bash
python experiments/wm_training/run.py --config-name=tworoom.yaml --multirun backbone=dinov2_small dataset_name=tworoom_noisy_weak,tworoom_noisy_variation_all subset_prop=0.1,0.5 launcher=your_name
```

**interaction quality interpolation**:
```bash
python experiments/wm_training/run.py --config-name=tworoom.yaml --multirun backbone=dinov2_small dataset_name=tworoom_noisy_weak,tworoom_random launcher=your_name
```

**quality interpolation**:
```bash
python experiments/wm_training/run.py --config-name=tworoom.yaml --multirun backbone=dinov2_small dataset_name=tworoom_noisy injected_dataset.names="[tworoom_random]" injected_dataset.proportions="[0.95],[0.9],[0.8],[0.5],[0.2]" launcher=your_name
```

**variation interpolation**:
```bash
python experiments/wm_training/run.py --config-name=tworoom.yaml --multirun backbone=dinov2_small dataset_name=tworoom_noisy injected_dataset.names="[tworoom_noisy_variation_all]" injected_dataset.proportions="[0.5],[0.1],[0.01]" launcher=your_name
```


# Cube
Run the training script from the repository root. Below are several example commands for different backbones.

**all backbones**:
```bash
python experiments/wm_training/run.py --config-name=cube.yaml --multirun "backbone=glob(*)" launcher=your_name
```

**dinov2 encoder scaling**:
```bash
python experiments/wm_training/run.py --config-name=cube.yaml --multirun backbone=dinov2_small,dinov2_base,dinov2_large,dinov2_giant launcher=your_name
```

**predictor scaling**:
```bash
python experiments/wm_training/run.py --config-name=cube.yaml --multirun predictor=tiny,small,base,large launcher=your_name
```

**data all variations**:
```bash
python experiments/wm_training/run.py --config-name=cube.yaml --multirun backbone=dinov2_small dataset_name=ogb_cube_oracle_variation_all launcher=your_name
```

**data scaling**:
```bash
python experiments/wm_training/run.py --config-name=cube.yaml --multirun backbone=dinov2_small dataset_name=ogb_cube_oracle,ogb_cube_oracle_variation_all subset_prop=0.1,0.5 launcher=your_name
```

**variation interpolation**:
```bash
python experiments/wm_training/run.py --config-name=cube.yaml --multirun backbone=dinov2_small dataset_name=ogb_cube_oracle injected_dataset.names="[ogb_cube_oracle_variation_all]" injected_dataset.proportions="[0.5],[0.1],[0.01]" launcher=your_name -m
```

# Scene
Run the training script from the repository root. Below are several example commands for different backbones.

**all backbones**:
```bash
python experiments/wm_training/run.py --config-name=scene.yaml --multirun "backbone=glob(*)" launcher=your_name
```

**dinov2 encoder scaling**:
```bash
python experiments/wm_training/run.py --config-name=scene.yaml --multirun backbone=dinov2_small,dinov2_base,dinov2_large,dinov2_giant launcher=your_name
```

**predictor scaling**:
```bash
python experiments/wm_training/run.py --config-name=scene.yaml --multirun predictor=tiny,small,base,large launcher=your_name
```

**data all variations**:
```bash
python experiments/wm_training/run.py --config-name=scene.yaml --multirun backbone=dinov2_small dataset_name=ogb_scene_oracle_variation_all launcher=your_name
```

**data scaling**:
```bash
python experiments/wm_training/run.py --config-name=scene.yaml --multirun backbone=dinov2_small dataset_name=ogb_scene_oracle,ogb_scene_oracle_variation_all subset_prop=0.1,0.5 launcher=your_name
```

**variation interpolation**:
```bash
python experiments/wm_training/run.py --config-name=scene.yaml --multirun backbone=dinov2_small dataset_name=ogb_scene_oracle injected_dataset.names="[ogb_scene_oracle_variation_all]" injected_dataset.proportions="[0.5],[0.1],[0.01]" launcher=your_name -m
```
