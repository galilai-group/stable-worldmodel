# stable-worldmodel

[![Documentation](https://img.shields.io/badge/Documentation-blue.svg)](TBA)
[![LeaderBoard](https://img.shields.io/badge/Benchmarks-blue.svg)](TBA)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![WandB](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/site)

## Complete Example

<details>
<summary>End-2-End training</summary>

This example demonstrates the key features of `stable-worldmodel`: dictionary-structured data, unified forward function, and rich monitoring through callbacks.

```python
import stable_worldmodel as swm
import torch

# create world
world = swm.World(
    "swm/SimplePointMaze-v0",
    num_envs=7,
    image_shape=(224, 224),
    render_mode="rgb_array",
)

# collect data for pre-training
world.set_policy(swm.policy.RandomPolicy())
world.record_dataset("simple-pointmaze", episodes=10, seed=2347)
world.record_video("./", seed=2347)

# pre-train world model
swm.pretraining("scripts/train/dummy.py", "++dump_object=True dataset_name=simple-pointmaze output_model_name=dummy_test")

# evaluate world model
action_dim = world.envs.single_action_space.shape[0]
cost_fn = torch.nn.functional.mse_loss
world_model = swm.wm.DummyWorldModel((224, 224, 3), action_dim)
solver = swm.solver.RandomSolver(horizon=5, action_dim=action_dim, cost_fn=cost_fn)
policy = swm.policy.WorldModelPolicy(world_model, solver, horizon=10, action_block=5, receding_horizon=5)
world.set_policy(policy)

spt_module = torch.load(swm.data.get_cache_dir()+"/dummy_test_object.ckpt", weights_only=False)
world_model = spt_module.model
results = world.evaluate(episodes=2, seed=2347)
```
</details>


## Contributors

Core contributors (in order of joining the project):
- [Randall Balestriero](https://github.com/RandallBalestriero)
- [Dan Haramati](https://github.com/DanHrmti)
- [Lucas Maes](https://github.com/lucas-maes)

## dino-wm (temporary)

### ckpt

```bash
wget https://osf.io/xvzs4/download -O ckpt.zip && unzip ckpt.zip && rm ckpt.zip
```

### dataset

```bash
wget https://osf.io/k2d8w/download -O dinowm_pushT.zip && unzip dinowm_pushT.zip && rm dinowm_pushT.zip
```

## Installation

Follow the below isntruction (if you are not using conda, adjust accordingly)

1. create conda env
  ```
  conda create -n xeno python=3.10
  ```

2. activate your env and install uv (faster package manager)
  ```
  conda activate xeno
  pip3 install uv
  ```
3. install our package
  ```
  uv pip install -e .
  ```
4. install MUJOCO
  ```
  wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
  tar -xvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
  ```
  and add to your `.bashrc`
  ```
  export MUJOCO_HOME="$HOME/.mujoco/mujoco210"
export PATH="$MUJOCO_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MUJOCO_HOME/bin:$LD_LIBRARY_PATH"
```



