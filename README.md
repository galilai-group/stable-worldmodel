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
from stable_worldmodel.data import HDF5Dataset
from stable_worldmodel.policy import WorldModelPolicy, PlanConfig
from stable_worldmodel.solver import CEMSolver

# collect a dataset
world = swm.World('swm/PushT-v1', num_envs=8)
world.set_policy(your_expert_policy)
world.record_dataset(dataset_name='pusht_demo', episodes=100)

# load dataset and train your world model
dataset = HDF5Dataset(name='pusht_demo', num_steps=16)
world_model = ...  # your world-model

# evaluate with model predictive control
solver = CEMSolver(model=world_model, num_samples=300)
policy = WorldModelPolicy(solver=solver, config=PlanConfig(horizon=10))

world.set_policy(policy)
results = world.evaluate(episodes=50)
print(f"Success Rate: {results['success_rate']:.1f}%")
```

stable-worldmodel eases reproducibility by already implementing several baselines: [`scripts/train/prejepa.py`](scripts/train/prejepa.py) reproduces results from the [DINO-WM paper](https://arxiv.org/abs/2411.04983) and [`scripts/train/gcivl.py`](scripts/train/gcivl.py) implements several [goal-conditioned RL algorithms](https://arxiv.org/abs/2410.20092).
To foster research in MPC for world models, several planning solvers are already implemented, including zeroth-order ([CEM](stable_worldmodel/solver/cem.py),[MPPI](stable_worldmodel/solver/mppi.py)), as well as [gradient-based](stable_worldmodel/solver/gd.py) approaches.

### Efficiency

We support multiple dataset formats to optimize efficiency: MP4 enables fast and convenient visualization, while HDF5 ensures high-performance data loading, reduces CPU bottlenecks, and improves overall GPU utilization.

<p align="center">
  <img src="docs/assets/dinowm-gpu-usage.png" alt="GPU utilization comparison" width="60%">
  <br>
  <em>GPU utilization for DINO-WM trained on Push-T with a DINOv2-Small backbone.</em>
</p>

See the full documentation [here](https://galilai-group.github.io/stable-worldmodel/).

## Supported Environments

<p align="center">
  <img src="docs/assets/envs_grid_1.gif" alt="Environments Grid 1">
  <br>
  <img src="docs/assets/envs_grid_2.gif" alt="Environments Grid 2">
</p>

stable-worldmodel supports a large collection of environments from the [DeepMind Control Suite](https://github.com/google-deepmind/dm_control), [OGBench](https://github.com/seohongpark/ogbench), and classical world model benchmarks such as [Two-Room](https://arxiv.org/abs/2411.04983) and [PushT](https://arxiv.org/abs/2303.04137).

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

</td>
<td valign="top">

| [Solver](https://github.com/galilai-group/stable-worldmodel/tree/main/stable_worldmodel/solver) | Type |
|----------|---|
| Cross-Entropy Method (CEM)| Sampling |
| Model Predictive Path Integral (MPPI) | Sampling |
| Gradient Descent (SGD, Adam) | Gradient |
| Projected Gradient Descent (PGD) | Gradient |

<br>

| [Baselines](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) | Type |
|-------------------|----|
| DINO-WM           |JEPA|
| GCBC              |Behaviour Cloning|
| HILP              |RL|
| GCIVL             |RL|
| GCIQL             |RL|

</td>

</td>
</tr>
</table>

</div>




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

> **Note:** All datasets and models will be saved in the `$STABLEWM_HOME` environment variable. By default this is `~/.stable-wm/`. Adapt this directory according to your storage needs.


### Questions

If you have a question, please [file an issue](https://github.com/galilai-group/stable-worldmodel/issues).


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
