[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Testing](https://github.com/rbalestr-lab/stable-worldmodel/actions/workflows/testing.yaml/badge.svg)](https://github.com/rbalestr-lab/stable-worldmodel/actions/workflows/testing.yaml)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

<p align="center">
  <img src="./assets/stable-worldmodel-logo.png" alt="stable-worldmodel logo" width="500px"/>
</p>

A stable library for world model research and evaluation that provides unified interfaces for data collection, model training, and policy evaluation.

## Features

- 🧑‍🔬 **Controlled Factors of Variation**: Manage and track environmental factors with extended Gymnasium spaces
- 🎯 **Complete Solver Support**: Multiple planning algorithms (CEM, Gradient Descent, MPPI, Random)
- ✅ **High Test Coverage**: Comprehensive test suite ensuring reliability and correctness

## Installation

### Quick Start

1. **Install uv** (fast Python package manager):

```bash
pip install uv
```

2. **Clone and install the package**:

```bash
git clone https://github.com/rbalestr-lab/stable-worldmodel.git
cd stable-worldmodel
uv pip install -e .
```

### Development Installation

For development with testing and documentation tools:

```bash
uv pip install -e . --group dev --group doc
```

## Quick Example

```python
import stable_worldmodel as swm
import torch

# Create environment
world = swm.World(
    "swm/SimplePointMaze-v0",
    num_envs=7,
    image_shape=(224, 224),
    render_mode="rgb_array",
)

# Collect training data
world.set_policy(swm.policy.RandomPolicy())
world.record_dataset("simple-pointmaze", episodes=10, seed=2347)

# Train world model
swm.pretraining(
    "scripts/train/dummy.py",
    "++dump_object=True dataset_name=simple-pointmaze output_model_name=dummy_test"
)

# Load and evaluate
action_dim = world.envs.single_action_space.shape[0]
world_model = swm.wm.DummyWorldModel((224, 224, 3), action_dim)
solver = swm.solver.RandomSolver(
    horizon=5,
    action_dim=action_dim,
    cost_fn=torch.nn.functional.mse_loss
)
policy = swm.policy.WorldModelPolicy(
    world_model, solver,
    horizon=10, action_block=5, receding_horizon=5
)
world.set_policy(policy)

results = world.evaluate(episodes=2, seed=2347)
print(results)
```

## Project Structure

```
stable_worldmodel/
├── envs/          # Custom Gymnasium environments
├── solver/        # Planning algorithms (CEM, GD, MPPI, Random)
├── wm/            # World model implementations
├── tests/         # Test suite
├── policy.py      # Policy implementations
├── spaces.py      # Extended Gymnasium spaces with state tracking
├── world.py       # Main World interface
└── utils.py       # Utility functions
```

## Testing

Run tests with coverage:

```bash
pytest --cov=stable_worldmodel --cov-report=term-missing
```

## Contributors

- [Randall Balestriero](https://github.com/RandallBalestriero)
- [Dan Haramati](https://github.com/DanHrmti)
- [Lucas Maes](https://github.com/lucas-maes)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

@inproceedings{tbd,
title = "TBD",
author = "",
booktitle = "",
}
