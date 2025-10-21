````markdown
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Testing](https://github.com/rbalestr-lab/stable-worldmodel/actions/workflows/testing.yaml/badge.svg)](https://github.com/rbalestr-lab/stable-worldmodel/actions/workflows/testing.yaml)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

<p align="center">
  <img src="./assets/stable-worldmodel-logo.png" alt="stable-worldmodel logo" width="500px"/>
</p>

<p align="center">
  <strong>A unified library for world model research, training, and evaluation</strong>
</p>

---

**Stable World Model** provides a streamlined framework for conducting world model research with reproducible data collection, flexible model training, and comprehensive evaluation tools. Built on top of Gymnasium, it offers vectorized environments, domain randomization, and integrated support for multiple planning algorithms.

## ✨ Key Features

### 🎯 **Complete Research Pipeline**

- **Data Collection**: Vectorized episode recording with automatic sharding and variation tracking
- **Model Training**: Seamless integration with popular world model architectures (DINO-WM, Dreamer, TDMPC)
- **Policy Evaluation**: Built-in metrics and video generation for performance analysis

### 🔬 **Domain Randomization**

- **Variation Spaces**: Extended Gymnasium spaces for controlled factor manipulation
- **Automatic Tracking**: Record and replay episodes with specific variation configurations
- **State Queries**: Inspect and manipulate environment factors programmatically

### 🚀 **Planning & Optimization**

Multiple solver implementations for model-based planning:

- **CEM** (Cross-Entropy Method)
- **MPPI** (Model Predictive Path Integral)
- **Gradient Descent** with automatic differentiation
- **Nevergrad** integration for advanced optimization
- **Random Shooting** baseline

### ⚡ **Performance & Reliability**

- Vectorized environments for efficient data collection
- High test coverage ensuring stability
- GPU-accelerated policy rollouts
- Configurable parallelization

## 📦 Installation

### Prerequisites

- Python ≥ 3.10
- CUDA-compatible GPU (recommended for training)

### Quick Install

Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
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

### Development Installation

For contributors and researchers developing new features:

```bash
uv pip install -e ".[dev,docs]"
```

This includes testing tools (`pytest`, `coverage`) and documentation generators (`sphinx`).

## 🚀 Quick Start

### Basic Workflow

```python
import stable_worldmodel as swm

# 1. Create a vectorized world
world = swm.World(
    "swm/PushT-v1",
    num_envs=4,
    image_shape=(224, 224),
    render_mode="rgb_array",
)

# 2. Collect training data
world.set_policy(swm.policy.RandomPolicy())
world.record_dataset(
    "pusht-demos",
    episodes=100,
    seed=42,
    options=None,
)

# 3. Train a world model
swm.pretraining(
    "scripts/train/dinowm.py",
    dataset_name="pusht-demos",
    output_model_name="pusht-wm",
    dump_object=True,
)

# 4. Load model and create planning policy
model = swm.policy.AutoCostModel("pusht-wm").to("cuda")
config = swm.PlanConfig(
    horizon=10,
    receding_horizon=5,
    action_block=5
)
solver = swm.solver.CEMSolver(
    model,
    num_samples=500,
    n_steps=5,
    topk=50,
    device="cuda"
)
policy = swm.policy.WorldModelPolicy(
    solver=solver,
    config=config
)

# 5. Evaluate performance
world.set_policy(policy)
results = world.evaluate(episodes=50, seed=42)
print(f"Success Rate: {results['success_rate']:.2%}")
```

### Working with Variations

```python
import stable_worldmodel as swm

world = swm.World("swm/SimplePointMaze-v0", num_envs=8)

# Check available variations
print("Available variations:", world.single_variation_space.names())
# Output: ['walls.number', 'walls.shape', 'walls.positions']

# Record data with specific variations
world.record_dataset(
    "maze-varied",
    episodes=200,
    options={
        "variation": ("walls.number", "walls.shape")
    }
)

# Query dataset variation info
info = swm.data.dataset_info("maze-varied")
print(info["variation"])
```

### Video Recording

```python
# Record video during live rollout
world.record_video(
    "./videos",
    seed=42,
    options={"variation": ["all"]}
)

# Generate video from saved dataset
world.record_video_from_dataset(
    "./videos",
    dataset_name="pusht-demos",
    episode_idx=[0, 1, 2],
)
```

## 🏗️ Architecture

```
stable_worldmodel/
├── envs/          # Gymnasium environments
│   ├── pusht.py          # Push-T manipulation task
│   ├── simple_point_maze.py  # Configurable maze navigation
│   ├── two_room.py       # Two-room exploration task
│   └── ogbench_cube.py   # OGBench cube manipulation
├── solver/        # Planning algorithms
│   ├── cem.py            # Cross-Entropy Method
│   ├── mppi.py           # Model Predictive Path Integral
│   ├── gd.py             # Gradient Descent
│   └── nevergrad.py      # Nevergrad integration
├── wm/            # World model architectures
│   ├── dinowm.py         # DINO World Model
│   ├── dreamer.py        # Dreamer
│   └── tdmpc.py          # Temporal Difference MPC
├── policy.py      # Policy implementations
├── spaces.py      # Extended Gymnasium spaces
├── world.py       # Main World orchestration
├── data.py        # Data management utilities
└── utils.py       # Training and helper utilities
```

## 📚 Documentation

Comprehensive documentation is available at [link-to-docs] (or locally in `docs/build/html/`).

### Key Topics

- [Quick Start Guide](docs/source/introduction/quickstart.rst)
- [Creating Custom Worlds](docs/source/tutorials/new-world.rst)
- [Training World Models](docs/source/tutorials/dinowm.rst)
- [API Reference](docs/source/api/)

Build documentation locally:

```bash
cd docs
make html
```

## 🧪 Testing

We maintain high test coverage to ensure reliability:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=stable_worldmodel --cov-report=term-missing

# Run specific test categories
pytest -m unit          # Fast unit tests
pytest -m integration   # Integration tests
pytest -m gpu          # GPU-dependent tests
```

## 🌍 Available Environments

| Environment          | Description                   | Action Space    | Observation   | Variations          |
| -------------------- | ----------------------------- | --------------- | ------------- | ------------------- |
| `PushT-v1`           | Push T-shaped block to target | Continuous (2D) | RGB + Proprio | Block shape, color  |
| `SimplePointMaze-v0` | Navigate maze with point mass | Continuous (2D) | RGB           | Wall config, layout |
| `TwoRoom-v0`         | Two-room navigation task      | Continuous (2D) | RGB           | Room colors, sizes  |
| `OGBCube-v0`         | Cube manipulation (OGBench)   | Continuous (3D) | RGB + State   | Cube properties     |

## 🛠️ CLI Tools

The package includes command-line tools for common operations:

```bash
# List cached datasets
swm list-datasets

# Show dataset information
swm dataset-info pusht-demos

# List available worlds
swm list-worlds

# Get world metadata
swm world-info swm/PushT-v1

# Delete cached data
swm delete-dataset old-data
swm delete-model outdated-model
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork and clone the repository
2. Install development dependencies: `uv pip install -e ".[dev,docs]"`
3. Create a feature branch: `git checkout -b feature-name`
4. Make changes and add tests
5. Run tests: `pytest`
6. Format code: `ruff format .`
7. Submit a pull request

### Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Format code
ruff format .

# Run linter
ruff check .

# Auto-fix issues
ruff check --fix .
```

## 📖 Citation

If you use Stable World Model in your research, please cite:

```bibtex
@software{stable_worldmodel2024,
  title = {Stable World Model: A Unified Framework for World Model Research},
  author = {Balestriero, Randall and Maes, Lucas and Haramati, Dan},
  year = {2024},
  url = {https://github.com/rbalestr-lab/stable-worldmodel}
}
```

## 👥 Contributors

- [Randall Balestriero](https://github.com/RandallBalestriero)
- [Lucas Maes](https://github.com/lucas-maes)
- [Dan Haramati](https://github.com/DanHrmti)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with:

- [Gymnasium](https://gymnasium.farama.org/) - Reinforcement learning environments
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Lightning](https://lightning.ai/) - Training infrastructure
- [Stable Pretraining](https://github.com/rbalestr-lab/stable-pretraining) - SSL training utilities

## 📬 Contact

For questions, issues, or collaboration opportunities:

- Open an [issue](https://github.com/rbalestr-lab/stable-worldmodel/issues)
- Start a [discussion](https://github.com/rbalestr-lab/stable-worldmodel/discussions)

---

<p align="center">
  Made with ❤️ by the Stable World Model team
</p>
````
