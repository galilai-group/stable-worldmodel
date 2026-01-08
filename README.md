<div align="center">

# stable-worldmodel

_World Models Research Made Simple_

</div>

</br>
<p align="center">
  <a href="https://github.com/rbalestr-lab/stable-worldmodel/actions/workflows/testing.yaml">
    <img src="https://github.com/rbalestr-lab/stable-worldmodel/actions/workflows/testing.yaml/badge.svg" alt="Test" />
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="license" />
  </a>
  <a>
    <img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="license" />
  </a>
</p>

## Overview

**Stable World Model** provides a streamlined framework for conducting world model research with reproducible data collection, flexible model training, and comprehensive evaluation tools. Built on top of Gymnasium, it offers vectorized environments, domain randomization, and integrated support for multiple planning algorithms.

## Installation

#### Prerequisites

- Python >= 3.10
- CUDA-compatible GPU (recommended for training)

#### Quick Install

Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
# Create conda environment with Python 3.10 and required build tools
conda create -n stable-worldmodel python=3.10 swig bazel ffmpeg=7 -c conda-forge -y

# Activate the environment
conda activate stable-worldmodel

# Install uv
pip install uv

# Clone and install
git clone https://github.com/rbalestr-lab/stable-worldmodel.git
cd stable-worldmodel
uv pip install -e .
```

Using pip:

```bash
git clone https://github.com/rbalestr-lab/stable-worldmodel.git
cd stable-worldmodel
pip install -e .
```

#### Optional: Robocasa install
For robot manipulation environments (RoboCasa, RoboSuite), you need to manually install these dependencies from source, as they cannot be installed via pip alone.

1. **Install RoboSuite** (use the `robocasa-dev` branch):
   ```bash
   git clone git@github.com:ARISE-Initiative/robosuite.git
   cd robosuite
   git checkout robocasa-dev
   uv pip install -e .
   cd ..
   ```

2. **Install RoboCasa**:
   ```bash
   git clone https://github.com/Basile-Terv/robocasa.git
   cd robocasa
   uv pip install -e .

   # If you encounter issues with numba/numpy, run:
   # conda install -c numba numba=0.56.4 -y
   ```

3. **Download RoboCasa assets and setup**:
   ```bash
   python robocasa/scripts/download_kitchen_assets.py   # Caution: Assets are around 5GB
   python robocasa/scripts/setup_macros.py              # Set up system variables
   ```

4. **Download RoboCasa data**:
   ```bash
   # downloads all human datasets with images
   python -m robocasa.scripts.download_datasets --ds_types human_im

   # lite download: download human datasets without images
   python -m robocasa.scripts.download_datasets --ds_types human_raw

   # downloads all MimicGen datasets with images
   python -m robocasa.scripts.download_datasets --ds_types mg_im

   cd ..
   ```

5. **Convert RoboCasa data for stable-worldmodel**:

   After downloading the raw HDF5 data, convert it to the format used by `VideoDataset`:

   ```bash
   # Convert specific tasks (raw data from ~/robocasa/datasets/, output to $STABLEWM_HOME/robocasa/)
   python scripts/convert_robocasa_hdf5.py \
       --task_names PnPCounterToCab PnPCounterToSink \
       --mode video

   # Convert a small subset for testing
   python scripts/convert_robocasa_hdf5.py \
       --task_names PnPCounterToCab \
       --filter_first_episodes 5
   ```

   The converted dataset will be saved to `$STABLEWM_HOME/robocasa/` by default.

#### Development Installation

For contributors and researchers developing new features:

```bash
uv pip install -e ".[dev,docs]"
```

This includes testing tools (`pytest`, `coverage`) and documentation generators (`sphinx`).

#### Optional: Robocasa Installation

follow the instructions of the public robocasa and download the assets at `STABLEWM_HOME`.




## Architecture

```
stable_worldmodel/
├── envs/                   # Gymnasium environments
│   ├── pusht.py
│   ├── simple_point_maze.py
│   ├── two_room.py
│   └── ogbench_cube.py
├── solver/                 # Planning algorithms
│   ├── cem.py               # Cross-Entropy Method
│   ├── mppi.py              # Model Predictive Path Integral
│   ├── gd.py                # Gradient Descent
│   └── nevergrad.py         # Nevergrad
├── wm/                     # World model architectures
│   ├── dinowm.py            # DINO World Model
│   ├── dreamer.py           # Dreamer
│   └── tdmpc.py             # Temporal Difference MPC
├── policy.py
├── spaces.py               # Extended Gymnasium spaces
├── world.py
├── data.py
└── utils.py
```

## Testing

We maintain high test coverage to ensure reliability:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=stable_worldmodel --cov-report=term-missing
```

## Simple Install

### Setup
```bash
mkdir -p ~/scratch/datasets/
cd ~/scratch/
git clone https://github.com/rbalestr-lab/stable-worldmodel
cd stable-worldmodel
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv --python=3.10.4
source .venv/bin/activate
git fetch --all
git pull
git checkout experiments
uv pip install -e .
uv pip install gdown wandb hydra-submitit-launcher datasets==4.1.1
echo 'alias swm="cd ~/scratch/stable-worldmodel && source .venv/bin/activate"' >> ~/.bashrc
echo 'export HF_HOME=~/scratch/datasets/hf/' >> ~/.bashrc
echo 'export STABLEWM_HOME=~/scratch/datasets/' >> ~/.bashrc
source ~/.bashrc
```
### OGBench - Scene
```bash
gdown --folder 1t4rlIxLL1mdZHDeqofInBI9tIghoyXnc && mv dataset/* . && rmdir dataset && ls *.tar.zst | xargs -n1 tar --zstd -xf && rm *.tar.zst
```

### OGBench - Cube
```bash
gdown --folder 1Yz1FtA0xQEPZ_zH8RaAPpmqAWHxZqKob && mv dataset/* . && rmdir dataset && ls *.tar.zst | xargs -n1 tar --zstd -xf && rm *.tar.zst
```

### TwoRoom
```bash
gdown --folder 1OaFonvWWVnzabL_RPICoblB3rylZL0pI && mv dataset/* . && rmdir dataset && ls *.tar.zst | xargs -n1 tar --zstd -xf && rm *.tar.zst
```

### PushT
```bash
gdown --folder 1M7PfMRzoSujcUkqZxEfwjzGBIpRMdl88 && mv dataset/* . && rmdir dataset && ls *.tar.zst | xargs -n1 tar --zstd -xf && rm *.tar.zst
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
