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

#### Development Installation

For contributors and researchers developing new features:

```bash
uv pip install -e ".[dev,docs]"
```

This includes testing tools (`pytest`, `coverage`) and documentation generators (`sphinx`).

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
