[![Documentation](https://img.shields.io/badge/Docs-blue.svg)](https://galilai-group.github.io/stable-worldmodel/)
![Tests](https://img.shields.io/github/actions/workflow/status/galilai-group/stable-worldmodel/tests.yaml?label=Tests)
[![PyPI](https://img.shields.io/pypi/v/stable-worldmodel.svg)](https://pypi.python.org/pypi/stable-worldmodel/#history)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# stable-worldmodel

World model research made simple. From data collection to training and evaluation.

```bash
pip install stable-worldmodel
```

> **Note:** The library is still in active development.

See the full documentation at [here](https://galilai-group.github.io/stable-worldmodel/).


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

## Supported Environments

| Environment | Normal | Variations |
|:-----------:|:------:|:----------:|
| **Cheetah** | <img src="docs/assets/cheetah.gif" width="200"> | <img src="docs/assets/cheetah_var.gif" width="200"> |
| **Hopper** | <img src="docs/assets/hopper.gif" width="200"> | <img src="docs/assets/hopper_var.gif" width="200"> |
| **Walker** | <img src="docs/assets/walker.gif" width="200"> | <img src="docs/assets/walker_var.gif" width="200"> |
| **Quadruped** | <img src="docs/assets/quadruped.gif" width="200"> | <img src="docs/assets/quadruped_var.gif" width="200"> |
| **Reacher** | <img src="docs/assets/reacher.gif" width="200"> | <img src="docs/assets/reacher_var.gif" width="200"> |
| **Pendulum** | <img src="docs/assets/pendulum.gif" width="200"> | <img src="docs/assets/pendulum_var.gif" width="200"> |
| **Cartpole** | <img src="docs/assets/cartpole.gif" width="200"> | <img src="docs/assets/cartpole_var.gif" width="200"> |
| **Ball in Cup** | <img src="docs/assets/ballincup.gif" width="200"> | <img src="docs/assets/ballincup_var.gif" width="200"> |
| **Finger** | <img src="docs/assets/finger.gif" width="200"> | <img src="docs/assets/finger_var.gif" width="200"> |
| **Push-T** | <img src="docs/assets/pusht.gif" width="200"> | <img src="docs/assets/pusht_fov.gif" width="200"> |
| **Two-Room** | <img src="docs/assets/tworoom.gif" width="200"> | <img src="docs/assets/tworoom_fov.gif" width="200"> |
| **OGB Cube** | <img src="docs/assets/cube.gif" width="200"> | <img src="docs/assets/cube_fov.gif" width="200"> |
| **OGB Scene** | <img src="docs/assets/scene.gif" width="200"> | <img src="docs/assets/scene_fov.gif" width="200"> |

## Contributing

Setup your codebase:

```bash
uv venv --python=3.10
source .venv/bin/activate
uv sync --all-extras --group dev
```

## Questions

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