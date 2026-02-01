title: World
summary: A unified interface for orchestrating vectorized environments, managing policy interactions, and handling data collection (HDF5/Video) and evaluation pipelines.
---

The `World` class is the central entry point for managing vectorized environments in `stable_worldmodel`. It handles synchronization, preprocessing (resizing, stacking), and interaction with policies.

/// tab | Basic Usage
```python
from stable_worldmodel import World
from stable_worldmodel.policy import RandomPolicy

# 1. Initialize the World with 4 parallel environments
world = World(
    env_name="swm/PushT-v1",
    num_envs=4,
    image_shape=(64, 64),
    history_size=1
)

# 2. Set a policy (e.g., Random)
world.set_policy(RandomPolicy())

# 3. Reset and step
world.reset()
for _ in range(100):
    world.step()
    # Access current states/infos
    # world.infos["pixels"] -> (4, 3, 64, 64)
```
///

/// tab | Recording Video
```python
from stable_worldmodel import World
from stable_worldmodel.policy import RandomPolicy

world = World(
    env_name="swm/PushT-v1",
    num_envs=1,
    image_shape=(64, 64)
)
world.set_policy(RandomPolicy())

# Record a 500-step video
world.record_video(
    video_path="./videos",
    max_steps=500,
    viewname="pixels"
)
```
///

/// tab | Recording Dataset
```python
from stable_worldmodel import World
from stable_worldmodel.policy import RandomPolicy

world = World(
    env_name="swm/PushT-v1",
    num_envs=4,  # Collect 4 episodes in parallel
    image_shape=(64, 64)
)
world.set_policy(RandomPolicy())

# Record 50 episodes to a .h5 dataset
world.record_dataset(
    dataset_name="pusht_random",
    episodes=50,
    cache_dir="./data"
)
# Result: ./data/pusht_random.h5
```
///

/// tab | Evaluation
```python
from stable_worldmodel import World
from stable_worldmodel.data import HDF5Dataset
from stable_worldmodel.policy import RandomPolicy # or your trained policy

# 1. Load a dataset for initial states
dataset = HDF5Dataset("pusht_random", cache_dir="./data")

# 2. Setup World
world = World(env_name="swm/PushT-v1", num_envs=4, image_shape=(64, 64))
world.set_policy(RandomPolicy())

# 3. Evaluate starting from dataset states
results = world.evaluate_from_dataset(
    dataset=dataset,
    episodes_idx=[0, 1, 2, 3],  # Episodes to test on
    start_steps=[0, 0, 0, 0],   # Start from beginning
    goal_offset_steps=50,       # Goal is state at t=50
    eval_budget=100             # Max steps to reach goal
)

print(f"Success Rate: {results['success_rate']}%")
```
///

!!! tip "Performance"
    The `World` class uses a custom `SyncWorld` vectorized environment for synchronized execution, ensuring deterministic and batched stepping across multiple environments.

/// tab | Per-Environment Options
The `World` class supports passing different options to each environment during reset, enabling per-environment variations and configurations:

```python
from stable_worldmodel import World

world = World(
    env_name="swm/PushT-v1",
    num_envs=3,
    image_shape=(64, 64)
)

# Different variations for each environment
per_env_options = [
    {"variation": ["agent.color"], "variation_values": {"agent.color": [255, 0, 0]}},
    {"variation": ["agent.color"], "variation_values": {"agent.color": [0, 255, 0]}},
    {"variation": ["agent.color"], "variation_values": {"agent.color": [0, 0, 255]}},
]

world.reset(options=per_env_options)
```

This is useful for:

- **Domain randomization**: Different visual variations per environment
- **Curriculum learning**: Different difficulty levels per environment
- **Parallel evaluation**: Testing multiple configurations simultaneously
///

::: stable_worldmodel.World
    options:
        heading_level: 2
        members: false
        show_source: false

## **[ Recording ]**

::: stable_worldmodel.World.record_dataset
::: stable_worldmodel.World.record_video

## **[ Evaluation ]**

::: stable_worldmodel.World.evaluate_from_dataset
::: stable_worldmodel.World.evaluate

## **[ Environment ]**

::: stable_worldmodel.World.reset
::: stable_worldmodel.World.step
::: stable_worldmodel.World.close
::: stable_worldmodel.World.set_policy

## **[ Properties ]**

::: stable_worldmodel.World.num_envs
::: stable_worldmodel.World.observation_space
::: stable_worldmodel.World.action_space
::: stable_worldmodel.World.variation_space
::: stable_worldmodel.World.single_variation_space
::: stable_worldmodel.World.single_action_space
::: stable_worldmodel.World.single_observation_space

