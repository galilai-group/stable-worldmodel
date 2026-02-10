---
title: Stable World-Model
summary: World Model Research Made Simple
sidebar_title: Home
---

!!! danger ""
    **The library is still in active development!**

Stable World-Model is an open-source library to conduct world model research.  You can install `stable-worldmodel` directly from PyPI:

=== "uv"

        :::bash
        uv add stable-worldmodel

=== "pip"

        :::bash
        pip install stable-worldmodel

=== "uv (all dependencies)"

        :::bash
        uv add stable-worldmodel --all-extras

=== "pip (all dependencies)"

        :::bash
        pip install stable-worldmodel[env, train]


!!! note ""
    ⚠️ The base installation does not include environment (`env`) or training (`train`) dependencies. Install them separately or use the "all dependencies" option above if you need to run simulations or train models.

A **world model** is a learned simulator that predicts how an environment evolves in response to actions, enabling agents to plan by imagining future outcomes. Stable World-Model provides a unified research ecosystem that simplifies the entire pipeline: from data collection to model training and evaluation.

**Why another library?** World models have recently gained a lot of attention from the community. However, each new article re-implements over and over the same baselines, evaluation protocols, and data processing logic. We took that as an opportunity to provide a clean, documented, and tested library that researchers can trust for evaluation or training. More than just re-implementation, stable-worldmodel provides a complete ecosystem for world model research, from data collection to evaluation. We also extended the range of test-beds by providing researchers with a lean and simple API to fully customize the environments in which agents operate: from colors, to shapes, to physics properties. Everything is customizable, allowing for easy continual learning, out-of-distribution, or zero-shot robustness evaluation.

## Install
---

Set up a ready-to-go development environment to contribute to the library:

```bash
git clone https://github.com/galilai-group/stable-worldmodel
cd stable-worldmodel/
uv venv --python=3.10
source .venv/bin/activate
uv sync --all-extras --group dev
```

!!! warning ""
    All datasets and models will be saved in the `$STABLEWM_HOME` environment variable.
    By default the corresponding location is `~/.stable-wm/`. We encourage every user to adapt that directory according to their need and storage.

## Example
---

Here is a quick start example: collect a dataset and perform an evaluation.

```python
import stable_worldmodel as swm
from stable_worldmodel.data import HDF5Dataset
from stable_worldmodel.policy import WorldModelPolicy, PlanConfig
from stable_worldmodel.solver import CEMSolver


world = swm.World('swm/PushT-v1', num_envs=8)
world.set_policy(your_expert_policy)

world.record_dataset(dataset_name='pusht_demo',
                     episodes=100,
                     seed=0,
                     options={"variation":["all"],
                })

# ... train your world model with pusht_demo...
world_model = ... # your world-model implementing get_cost

# evaluation
dataset = HDF5Dataset(
    name='pusht_demo',
    frameskip=1,
    num_steps=16,
    keys_to_load=['pixels', 'action', 'state']
)

# model predictive control
solver = CEMSolver(model=world_model, num_samples=300, device='cuda')
policy = WorldModelPolicy(
    solver=solver,
    config=PlanConfig(horizon=10, receding_horizon=5)
)

world.set_policy(policy)
results = world.evaluate(episodes=50, seed=0)

print(f"Success Rate: {results['success_rate']:.1f}%")

```

See the [Quick Start Guide](quick_start.md) for detailed explanations of each component.



## Next Steps
---

After you have installed stable-worldmodel, try the [Quick Start Guide](quick_start.md). You can also explore other parts of the documentation:

| | |
|---|---|
| **[Environments](envs/pusht.md)** | Explore the included environments: PushT, TwoRoom, OGBench, DMControl, and more. |
| **[API Reference](api/world.md)** | Detailed documentation for World, Policy, Solver, Dataset, and other modules. |

## Citation

If you wish to cite our [pre-print](https://arxiv.org/abs/2602.08968):

```bibtex
@misc{maes_lelidec2026swm-1,
      title={stable-worldmodel-v1: Reproducible World Modeling Research and Evaluation}, 
      author={Lucas Maes and Quentin Le Lidec and Dan Haramati and Nassim Massaudi and Damien Scieur and Yann LeCun and Randall Balestriero},
      year={2026},
      eprint={2602.08968},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.08968}, 
}
```