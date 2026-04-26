<h1 align="center">
    <p> <b>stable-worldmodel</b></p>
</h1>

<h2 align="center">
    <p><i>World model research made simple. From data collection to training and evaluation.</i></p>
</h2>

<div align="center" style="line-height: 1;">
  <a href="https://galilai-group.github.io/stable-worldmodel/" target="_blank" style="margin: 2px;"><img alt="Documentation" src="https://img.shields.io/badge/Docs-blue.svg" style="display: inline-block; vertical-align: middle;"/></a>
  <a href="https://github.com/galilai-group/stable-worldmodel" target="_blank" style="margin: 2px;"><img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/galilai-group/stable-worldmodel/tests.yaml?label=Tests" style="display: inline-block; vertical-align: middle;"/></a>
  <a href="https://pypi.python.org/pypi/stable-worldmodel/#history" target="_blank" style="margin: 2px;"><img alt="PyPI" src="https://img.shields.io/pypi/v/stable-worldmodel.svg" style="display: inline-block; vertical-align: middle;"/></a>
  <a href="https://pytorch.org/get-started/locally/" target="_blank" style="margin: 2px;"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
  <a href="https://github.com/astral-sh/ruff" target="_blank" style="margin: 2px;"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" style="display: inline-block; vertical-align: middle;"/></a>
</div>

<p align="center">
  <a href="#quick-example"><b>Quick Example</b></a> | <a href="#supported-environments"><b>Environments</b></a> | <a href="#installing-stable-worldmodel"><b>Installation</b></a> | <a href="https://galilai-group.github.io/stable-worldmodel/"><b>Documentation</b></a> | <a href="#contributing"><b>Contributing</b></a> | <a href="#citation"><b>Citation</b></a>
</p>



## Quick Example

```python
import stable_worldmodel as swm
from stable_worldmodel.policy import WorldModelPolicy, PlanConfig
from stable_worldmodel.solver import CEMSolver

# collect a dataset (default writer is hdf5; pass format='video' for MP4 episodes)
world = swm.World('swm/PushT-v1', num_envs=8)
world.set_policy(your_expert_policy)
world.collect('data/pusht_demo.h5', episodes=100, seed=0)

# load dataset and train your world model — format is autodetected
dataset = swm.data.load_dataset('data/pusht_demo.h5', num_steps=16)
world_model = ...  # your world-model

# evaluate with model predictive control
solver = CEMSolver(model=world_model, num_samples=300)
policy = WorldModelPolicy(solver=solver, config=PlanConfig(horizon=10))

world.set_policy(policy)
results = world.evaluate(episodes=50)
print(f"Success Rate: {results['success_rate']:.1f}%")
```

stable-worldmodel eases reproducibility by already implementing several baselines: [`scripts/train/prejepa.py`](scripts/train/prejepa.py) reproduces results from the [DINO-WM paper](https://arxiv.org/abs/2411.04983) and [`scripts/train/gcivl.py`](scripts/train/gcivl.py) implements several [goal-conditioned RL algorithms](https://arxiv.org/abs/2410.20092).
To foster research in MPC for world models, several planning solvers are already implemented, including zeroth-order ([CEM](stable_worldmodel/solver/cem.py), [MPPI](stable_worldmodel/solver/mppi.py)), gradient-based ([GradientSolver](stable_worldmodel/solver/gd.py), [PGD](stable_worldmodel/solver/discrete_solvers.py)), and constrained gradient approaches ([LagrangianSolver](stable_worldmodel/solver/lagrangian.py)).

### Data Formats

All datasets — recording, loading, and conversion — go through a small
**format registry**. Pick a backend that suits the trade-off you care about,
or [register your own](https://galilai-group.github.io/stable-worldmodel/api/dataset/#registering-a-custom-format):

| Format     | On-disk layout                                  | Best for                                          |
|------------|-------------------------------------------------|---------------------------------------------------|
| `hdf5`     | single `.h5` file (one dataset per column)      | training — fast indexed reads, low CPU overhead   |
| `folder`   | `.npz` columns + one JPEG per step              | inspection, partial-key streaming                 |
| `video`    | `.npz` columns + one MP4 per episode (`decord`) | long episodes, compact image storage              |
| `lerobot`  | `lerobot://<repo_id>` (read-only adapter)       | training/eval directly on LeRobot Hub datasets    |

```python
import stable_worldmodel as swm

world.collect('data/pusht.h5', episodes=100)                    # default: hdf5
world.collect('data/pusht_video', episodes=100, format='video') # mp4 episodes

ds = swm.data.load_dataset('data/pusht.h5', num_steps=16)       # autodetect
swm.data.convert('data/pusht.h5', 'data/pusht_video',
                 dest_format='video', fps=30)                   # one-shot migration
```

HDF5 ensures high-performance data loading, reduces CPU bottlenecks, and
improves overall GPU utilization; MP4 is convenient for visualization and
keeps long-horizon datasets small.

<p align="center">
  <img src="docs/assets/dinowm-gpu-usage.png" alt="GPU utilization comparison" width="60%">
  <br>
  <em>GPU utilization for DINO-WM trained on Push-T with a DINOv2-Small backbone.</em>
</p>

See the full documentation [here](https://galilai-group.github.io/stable-worldmodel/).

## Supported Environments

<div align="center">

<table>
<tr>
<td align="center"><img src="docs/assets/ballincup.gif"   width="120"/><br><img src="docs/assets/ballincup_var.gif"   width="120"/></td>
<td align="center"><img src="docs/assets/cartpole.gif"    width="120"/><br><img src="docs/assets/cartpole_var.gif"    width="120"/></td>
<td align="center"><img src="docs/assets/cheetah.gif"     width="120"/><br><img src="docs/assets/cheetah_var.gif"     width="120"/></td>
<td align="center"><img src="docs/assets/finger.gif"      width="120"/><br><img src="docs/assets/finger_var.gif"      width="120"/></td>
<td align="center"><img src="docs/assets/hopper.gif"      width="120"/><br><img src="docs/assets/hopper_var.gif"      width="120"/></td>
<td align="center"><img src="docs/assets/pendulum.gif"    width="120"/><br><img src="docs/assets/pendulum_var.gif"    width="120"/></td>
</tr>
</table>

<table>
<tr>
<td align="center"><img src="docs/assets/quadruped.gif"   width="120"/><br><img src="docs/assets/quadruped_var.gif"   width="120"/></td>
<td align="center"><img src="docs/assets/reacher.gif"     width="120"/><br><img src="docs/assets/reacher_var.gif"     width="120"/></td>
<td align="center"><img src="docs/assets/walker.gif"      width="120"/><br><img src="docs/assets/walker_var.gif"      width="120"/></td>
<td align="center"><img src="docs/assets/cube.gif"        width="120"/><br><img src="docs/assets/cube_fov.gif"        width="120"/></td>
<td align="center"><img src="docs/assets/scene.gif"       width="120"/><br><img src="docs/assets/scene_fov.gif"       width="120"/></td>
<td align="center"><img src="docs/assets/pusht.gif"       width="120"/><br><img src="docs/assets/pusht_fov.gif"       width="120"/></td>
</tr>
</table>

<table>
<tr>
<td align="center"><img src="docs/assets/tworoom.gif"            width="120"/><br><img src="docs/assets/tworoom_fov.gif"            width="120"/></td>
<td align="center"><img src="docs/assets/fetch_reach.gif"        width="120"/><br><img src="docs/assets/fetch_reach_var.gif"        width="120"/></td>
<td align="center"><img src="docs/assets/fetch_push.gif"         width="120"/><br><img src="docs/assets/fetch_push_var.gif"         width="120"/></td>
<td align="center"><img src="docs/assets/fetch_slide.gif"        width="120"/><br><img src="docs/assets/fetch_slide_var.gif"        width="120"/></td>
<td align="center"><img src="docs/assets/fetch_pickandplace.gif" width="120"/><br><img src="docs/assets/fetch_pickandplace_var.gif" width="120"/></td>
<td align="center"><img src="docs/assets/craftax_classic.gif"    width="120"/><br><img src="docs/assets/craftax.gif"                  width="120"/></td>
</tr>
</table>

</div>

<p align="center"><em>Top row: default appearance &nbsp;·&nbsp; Bottom row: visual factor of variation</em></p>

stable-worldmodel supports a large collection of environments from the [DeepMind Control Suite](https://github.com/google-deepmind/dm_control), [OGBench](https://github.com/seohongpark/ogbench), [Craftax](https://github.com/MichaelTMatthews/Craftax), and classical world model benchmarks such as [Two-Room](https://arxiv.org/abs/2411.04983) and [PushT](https://arxiv.org/abs/2303.04137).

Each environment includes visual and physical factor variations to evaluate robustness and generalization. New environments can easily be added to stable-worldmodel as they only need to follow the [Gymnasium](https://gymnasium.farama.org/) interface.

<div align="center">

<table>
<tr>
<td valign="top">

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
| swm/FetchReach-v3            | 8       |
| swm/FetchPush-v3             | 11      |
| swm/FetchSlide-v3            | 11      |
| swm/FetchPickAndPlace-v3     | 11      |
| swm/CraftaxClassicPixels-v1  | —       |
| swm/CraftaxClassicSymbolic-v1| —       |
| swm/CraftaxPixels-v1         | —       |
| swm/CraftaxSymbolic-v1       | —       |

</td>
<td valign="top">

| [Solver](https://github.com/galilai-group/stable-worldmodel/tree/main/stable_worldmodel/solver) | Type |
|----------|---|
| Cross-Entropy Method (CEM)| Sampling |
| Improved CEM (iCEM) | Sampling |
| Model Predictive Path Integral (MPPI) | Sampling |
| Gradient Descent (SGD, Adam) | Gradient |
| Projected Gradient Descent (PGD) | Gradient |
| Augmented Lagrangian | Constrained Opt |

<br>

| [Baselines](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) | Type |
|-------------------|----|
| DINO-WM           |JEPA|
| PLDM              |JEPA|
| LeWM              |JEPA|
| GCBC              |Behaviour Cloning|
| GCIVL             |RL|
| GCIQL             |RL|

</td>

</td>
</tr>
</table>

</div>




## CLI

After installation, the `swm` command-line tool is available to inspect your datasets, environments, and checkpoints without writing any code:

```bash
# list cached datasets
swm datasets

# inspect a specific dataset
swm inspect pusht_expert_train

# list all registered environments
swm envs

# show factors of variation for an environment
swm fovs PushT-v1

# list available model checkpoints
swm checkpoints
```

## Installing stable-worldmodel

stable-worldmodel is available on PyPI and can be installed with:

```bash
pip install stable-worldmodel
```

> **Note:** The library is still in active development.

### Install from Source

To set up a development environment from source:

```bash
git clone https://github.com/galilai-group/stable-worldmodel
cd stable-worldmodel/
uv venv --python=3.10
source .venv/bin/activate
uv sync --all-extras --group dev
```

> **Note:** All datasets and models will be saved in the `$STABLEWM_HOME` environment variable. By default this is `~/.stable_worldmodel/`. Adapt this directory according to your storage needs.


### Questions

If you have a question, please [file an issue](https://github.com/galilai-group/stable-worldmodel/issues).


## Works Based on stable-worldmodel

The following research projects have been built on top of stable-worldmodel:

- **[C-JEPA](https://hazel-heejeong-nam.github.io/cjepa/)**
- **[LeWM](https://le-wm.github.io/)**

## Citation

```bibtex
@misc{maes_lelidec2026swm-1,
      title={stable-worldmodel-v1: Reproducible World Modeling Research and Evaluation},
      author = {Lucas Maes and Quentin Le Lidec and Dan Haramati and
                Nassim Massaudi and Damien Scieur and Yann LeCun and
                Randall Balestriero},
      year={2026},
      eprint={2602.08968},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.08968},
}
```
