<h1 align="center">stable-worldmodel</h1>

<p align="center"><i>A platform for reproducible world model research and evaluation.</i></p>

<p align="center">
  <a href="https://galilai-group.github.io/stable-worldmodel/"><img alt="Documentation" src="https://img.shields.io/badge/Docs-blue.svg"/></a>
  <a href="https://github.com/galilai-group/stable-worldmodel"><img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/galilai-group/stable-worldmodel/tests.yaml?label=Tests"/></a>
  <a href="https://pypi.python.org/pypi/stable-worldmodel/#history"><img alt="PyPI" src="https://img.shields.io/pypi/v/stable-worldmodel.svg"/></a>
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"/></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"/></a>
</p>

<p align="center">
  <a href="#installation"><b>Installation</b></a> ·
  <a href="#quick-start"><b>Quick Start</b></a> ·
  <a href="#environments"><b>Environments</b></a> ·
  <a href="#solvers-and-baselines"><b>Solvers & Baselines</b></a> ·
  <a href="https://galilai-group.github.io/stable-worldmodel/"><b>Documentation</b></a> ·
  <a href="#citation"><b>Citation</b></a>
</p>

---

`stable-worldmodel` provides a single, unified interface for the three stages of world model research — **collecting data**, **training**, and **evaluating with model-predictive control** — across a large suite of standardized environments. It ships with reference implementations of common baselines and planning solvers so research code can stay focused on the contribution that matters: the model and the objective.

## Installation

From PyPI:

```bash
pip install stable-worldmodel
```

From source (development):

```bash
git clone https://github.com/galilai-group/stable-worldmodel
cd stable-worldmodel
uv venv --python=3.10 && source .venv/bin/activate
uv sync --all-extras --group dev
```

Datasets and checkpoints are stored under `$STABLEWM_HOME` (defaults to `~/.stable_worldmodel/`). Override the variable to point at your preferred storage location.

> The library is in active development. APIs may change between minor versions.

## Quick Start

```python
import stable_worldmodel as swm
from stable_worldmodel.policy import WorldModelPolicy, PlanConfig
from stable_worldmodel.solver import CEMSolver

# 1. Collect a dataset
world = swm.World("swm/PushT-v1", num_envs=8)
world.set_policy(your_expert_policy)
world.collect("data/pusht_demo.lance", episodes=100, seed=0)

# 2. Load it and train your world model (format is autodetected)
dataset = swm.data.load_dataset("data/pusht_demo.lance", num_steps=16)
world_model = ...  # your model

# 3. Evaluate with model-predictive control
solver = CEMSolver(model=world_model, num_samples=300)
policy = WorldModelPolicy(solver=solver, config=PlanConfig(horizon=10))

world.set_policy(policy)
results = world.evaluate(episodes=50)
print(f"Success Rate: {results['success_rate']:.1f}%")
```

Reference implementations are provided in [`scripts/train/`](scripts/train): [`prejepa.py`](scripts/train/prejepa.py) reproduces [DINO-WM](https://arxiv.org/abs/2411.04983), and [`gcivl.py`](scripts/train/gcivl.py) implements several [goal-conditioned RL baselines](https://arxiv.org/abs/2410.20092).

<p align="center">
  <img src="docs/assets/dinowm-gpu-usage.png" alt="GPU utilization comparison" width="60%">
  <br>
  <em>GPU utilization for DINO-WM trained on Push-T with a DINOv2-Small backbone.</em>
</p>

## Data Formats

Recording, loading, and conversion all go through a small **format registry**. Pick the backend that matches your trade-off, or [register your own](https://galilai-group.github.io/stable-worldmodel/api/dataset/#registering-a-custom-format).

| Format    | On-disk layout                                  | Best for                                       |
|-----------|-------------------------------------------------|------------------------------------------------|
| `lance`   | LanceDB table (episode-contiguous flat rows)    | default — append-friendly, fast indexed reads  |
| `hdf5`    | single `.h5` file (one dataset per column)      | portable single-file artifact                  |
| `folder`  | `.npz` columns + one JPEG per step              | inspection, partial-key streaming              |
| `video`   | `.npz` columns + one MP4 per episode (`decord`) | long episodes, compact image storage           |
| `lerobot` | `lerobot://<repo_id>` (read-only adapter)       | training/eval directly on LeRobot Hub datasets |

```python
world.collect("data/pusht.lance", episodes=100)                  # default: lance
world.collect("data/pusht_video", episodes=100, format="video")  # mp4 episodes

ds = swm.data.load_dataset("data/pusht.lance", num_steps=16)     # autodetect
swm.data.convert("data/pusht.lance", "data/pusht_video",
                 dest_format="video", fps=30)                    # one-shot migration
```

Every writer accepts a `mode` kwarg (`'append'` (default), `'overwrite'`, `'error'`); re-running `world.collect` extends the existing dataset rather than failing.

## Environments

<div align="center">

<table>
<tr>
<td align="center"><img src="docs/assets/ballincup.gif" width="120"/><br><img src="docs/assets/ballincup_var.gif" width="120"/></td>
<td align="center"><img src="docs/assets/cartpole.gif"  width="120"/><br><img src="docs/assets/cartpole_var.gif"  width="120"/></td>
<td align="center"><img src="docs/assets/cheetah.gif"   width="120"/><br><img src="docs/assets/cheetah_var.gif"   width="120"/></td>
<td align="center"><img src="docs/assets/finger.gif"    width="120"/><br><img src="docs/assets/finger_var.gif"    width="120"/></td>
<td align="center"><img src="docs/assets/hopper.gif"    width="120"/><br><img src="docs/assets/hopper_var.gif"    width="120"/></td>
</tr>
</table>

<table>
<tr>
<td align="center"><img src="docs/assets/pendulum.gif"  width="120"/><br><img src="docs/assets/pendulum_var.gif"  width="120"/></td>
<td align="center"><img src="docs/assets/quadruped.gif" width="120"/><br><img src="docs/assets/quadruped_var.gif" width="120"/></td>
<td align="center"><img src="docs/assets/reacher.gif"   width="120"/><br><img src="docs/assets/reacher_var.gif"   width="120"/></td>
<td align="center"><img src="docs/assets/walker.gif"    width="120"/><br><img src="docs/assets/walker_var.gif"    width="120"/></td>
</tr>
</table>

<table>
<tr>
<td align="center"><img src="docs/assets/cartpole_control.gif"     width="120"/><br><img src="docs/assets/cartpole_control_var.gif"     width="120"/></td>
<td align="center"><img src="docs/assets/mountain_car_control.gif" width="120"/><br><img src="docs/assets/mountain_car_control_var.gif" width="120"/></td>
<td align="center"><img src="docs/assets/acrobot_control.gif"      width="120"/><br><img src="docs/assets/acrobot_control_var.gif"      width="120"/></td>
<td align="center"><img src="docs/assets/pendulum_control.gif"     width="120"/><br><img src="docs/assets/pendulum_control_var.gif"     width="120"/></td>
</tr>
</table>

<table>
<tr>
<td align="center"><img src="docs/assets/cube.gif"    width="120"/><br><img src="docs/assets/cube_fov.gif"    width="120"/></td>
<td align="center"><img src="docs/assets/scene.gif"   width="120"/><br><img src="docs/assets/scene_fov.gif"   width="120"/></td>
<td align="center"><img src="docs/assets/pusht.gif"   width="120"/><br><img src="docs/assets/pusht_fov.gif"   width="120"/></td>
<td align="center"><img src="docs/assets/tworoom.gif" width="120"/><br><img src="docs/assets/tworoom_fov.gif" width="120"/></td>
</tr>
</table>

<table>
<tr>
<td align="center"><img src="docs/assets/fetch_reach.gif"        width="120"/><br><img src="docs/assets/fetch_reach_var.gif"        width="120"/></td>
<td align="center"><img src="docs/assets/fetch_push.gif"         width="120"/><br><img src="docs/assets/fetch_push_var.gif"         width="120"/></td>
<td align="center"><img src="docs/assets/fetch_slide.gif"        width="120"/><br><img src="docs/assets/fetch_slide_var.gif"        width="120"/></td>
<td align="center"><img src="docs/assets/fetch_pickandplace.gif" width="120"/><br><img src="docs/assets/fetch_pickandplace_var.gif" width="120"/></td>
<td align="center"><img src="docs/assets/craftax_classic.gif"    width="120"/><br><img src="docs/assets/craftax.gif"                width="120"/></td>
</tr>
</table>

<em>Top row: default appearance &nbsp;·&nbsp; Bottom row: visual factor of variation</em>

</div>

Environments are pulled from the [DeepMind Control Suite](https://github.com/google-deepmind/dm_control), [Gymnasium classic control](https://gymnasium.farama.org/environments/classic_control/), [OGBench](https://github.com/seohongpark/ogbench), [Craftax](https://github.com/MichaelTMatthews/Craftax), the [Arcade Learning Environment](https://ale.farama.org/) (100+ Atari games), and classical world model benchmarks ([Two-Room](https://arxiv.org/abs/2411.04983), [PushT](https://arxiv.org/abs/2303.04137)). Each ships with both visual and physical factors of variation for robustness studies. Adding a new environment only requires conforming to the [Gymnasium](https://gymnasium.farama.org/) interface.

<div align="center">

| [Environment ID](https://github.com/galilai-group/stable-worldmodel/tree/main/stable_worldmodel/envs) |  # FoV  |
|------------------------------|---------|
| swm/PushT-v1                 | 16      |
| swm/TwoRoom-v1               | 17      |
| swm/OGBCube-v0               | 11      |
| swm/OGBScene-v0              | 12      |
| swm/HumanoidDMControl-v0     | 7       |
| swm/CheetahDMControl-v0      | 7       |
| swm/HopperDMControl-v0       | 7       |
| swm/ReacherDMControl-v0      | 8       |
| swm/WalkerDMControl-v0       | 8       |
| swm/AcrobotDMControl-v0      | 8       |
| swm/PendulumDMControl-v0     | 6       |
| swm/CartpoleDMControl-v0     | 6       |
| swm/BallInCupDMControl-v0    | 9       |
| swm/FingerDMControl-v0       | 10      |
| swm/ManipulatorDMControl-v0  | 8       |
| swm/QuadrupedDMControl-v0    | 7       |
| swm/CartPoleControl-v1       | 10      |
| swm/MountainCarControl-v0    | 5       |
| swm/MountainCarContinuousControl-v0 | 4 |
| swm/AcrobotControl-v1        | 11      |
| swm/PendulumControl-v1       | 9       |
| swm/FetchReach-v3            | 8       |
| swm/FetchPush-v3             | 11      |
| swm/FetchSlide-v3            | 11      |
| swm/FetchPickAndPlace-v3     | 11      |
| swm/CraftaxClassicPixels-v1  | —       |
| swm/CraftaxClassicSymbolic-v1| —       |
| swm/CraftaxPixels-v1         | —       |
| swm/CraftaxSymbolic-v1       | —       |
| [ALE/* (100+ Atari games)](https://ale.farama.org/) | — |

</div>

## Solvers and Baselines

<div align="center">

| [Solver](https://github.com/galilai-group/stable-worldmodel/tree/main/stable_worldmodel/solver) | Type            |
|---------------------------------------|-----------------|
| Cross-Entropy Method (CEM)            | Sampling        |
| Improved CEM (iCEM)                   | Sampling        |
| Model Predictive Path Integral (MPPI) | Sampling        |
| Predictive Sampling                   | Sampling        |
| Gradient Descent (SGD, Adam)          | Gradient        |
| Projected Gradient Descent (PGD)      | Gradient        |
| Augmented Lagrangian                  | Constrained Opt |

| [Baseline](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) | Type              |
|----------|-------------------|
| DINO-WM  | JEPA              |
| PLDM     | JEPA              |
| LeWM     | JEPA              |
| GCBC     | Behaviour Cloning |
| GCIVL    | RL                |
| GCIQL    | RL                |

</div>

## Command-Line Interface

After installation, the `swm` command is available for inspecting datasets, environments, and checkpoints without writing code:

```bash
swm datasets                    # list cached datasets
swm inspect pusht_expert_train  # inspect a specific dataset
swm envs                        # list all registered environments
swm fovs PushT-v1               # show factors of variation for an environment
swm checkpoints                 # list available model checkpoints
```

## Documentation

The full documentation lives at [galilai-group.github.io/stable-worldmodel](https://galilai-group.github.io/stable-worldmodel/), with API references, tutorials, and guides.

## Built on `stable-worldmodel`

- **[C-JEPA](https://hazel-heejeong-nam.github.io/cjepa/)**
- **[LeWM](https://le-wm.github.io/)**

## Citation

```bibtex
@misc{maes_lelidec2026swm-1,
  title  = {stable-worldmodel-v1: Reproducible World Modeling Research and Evaluation},
  author = {Lucas Maes and Quentin Le Lidec and Dan Haramati and
            Nassim Massaudi and Damien Scieur and Yann LeCun and
            Randall Balestriero},
  year   = {2026},
  eprint = {2602.08968},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url    = {https://arxiv.org/abs/2602.08968},
}
```

## Questions

Open an [issue](https://github.com/galilai-group/stable-worldmodel/issues) — happy to help.
