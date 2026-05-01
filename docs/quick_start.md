title: Quick Start Guide
summary: Get started with stable-worldmodel in minutes
sidebar_title: Quickstart
---

## World

The `World` class is the main interface for interacting with environments. It wraps a pool of vectorized Gymnasium environments (`EnvPool`) and a preprocessing pipeline (`MegaWrapper`), and provides a unified API for data collection, evaluation, and planning.

```python
import stable_worldmodel as swm

world = swm.World('swm/PushT-v1', num_envs=4, image_shape=(64, 64))
```

Unlike the traditional Gymnasium interface that returns `(obs, reward, terminated, truncated, info)` from `step()`, `World` adopts a simpler philosophy: all data produced by the environments is stored in a single dictionary accessible via `world.infos`. You drive the simulation by attaching a policy and calling `collect()` or `evaluate()` — the internal rollout loop steps the envs, refreshes `world.infos`, and invokes the policy for you.

```python
world.set_policy(policy)
world.reset(seed=0)

# Array/tensor values carry a leading time dim of 1 after the env dim.
print(world.infos['pixels'].shape)    # (num_envs, 1, H, W, C)
print(world.infos['state'].shape)     # (num_envs, 1, state_dim)
```

See the [World API](api/world.md) for all available parameters and methods.

!!! info ""
    Just want the environments? All environments are self-contained and follow the standard [Gymnasium](https://gymnasium.farama.org/) API. Simply import the library to register them.

        :::python                                                                                                            
        import gymnasium as gym                                                                                              
        import stable_worldmodel  # registers all environments                                                               
                                                                                                                            
        env = gym.make('swm/PushT-v1', render_mode='rgb_array')   

Use the CLI to list all available environments:

```bash
swm envs
```

### Factors of Variation (FoV)

A key feature of stable-worldmodel is the **variation space**, which defines all customizable properties of an environment. This enables domain randomization, continual learning experiments, and out-of-distribution evaluation by controlling factors like colors, shapes, sizes, and physics properties.

Each environment exposes a `variation_space` that describes what can be varied:

```python
# View the variation space structure (exposed on the env pool)
print(world.envs.single_variation_space.to_str())
```

For example, in PushT this might show:

```
agent:
    color: Box(0, 255, (3,), uint8)
    scale: Box(20.0, 60.0, (), float32)
    shape: Discrete(3)
    angle: Box(-6.28, 6.28, (), float32)
    start_position: Box([64. 64.], [448. 448.], (2,), float32)
block:
    color: Box(0, 255, (3,), uint8)
    scale: Box(20.0, 60.0, (), float32)
    ...
background:
    color: Box(0, 255, (3,), uint8)
```

Use the CLI to inspect the factors of variation for a specific environment:

```bash
swm fovs swm/PushT-v1
```

#### Controlling Variations at Reset

Each environment has a set of **default variations** that are randomized at every reset (e.g., starting positions of agents and objects). Other factors like colors and shapes remain fixed to a default value unless explicitly specified. To randomize additional factors, pass a `variation` option to `reset()`:

```python
# Randomize agent and block colors at each reset
world.reset(seed=0, options={'variation': ['agent.color', 'block.color']})

# Randomize everything
world.reset(seed=0, options={'variation': ['all']})

# Randomize all agent properties
world.reset(seed=0, options={'variation': ['agent']})
```

#### Setting Specific Values

You can also set exact values for variations:

```python
import numpy as np

world.reset(seed=0, options={
    'variation': ['agent.color', 'background.color'],
    'variation_values': {
        'agent.color': np.array([255, 0, 0], dtype=np.uint8),      # Red agent
        'background.color': np.array([0, 0, 0], dtype=np.uint8),   # Black background
    }
})
```

This is particularly useful for:

- **Domain randomization**: Train robust models by randomizing visual properties
- **Continual learning**: Gradually shift environment properties over training
- **OOD evaluation**: Test generalization to unseen colors, sizes, or configurations
- **Reproducibility**: Record and replay exact environment configurations from datasets

### Policy for World Interactions

A `World` requires a policy to determine actions. During `collect()` / `evaluate()`, the internal rollout loop calls `policy.get_action(world.infos)` on every step.

Any custom policy must implement the `get_action` method:

```python
class MyPolicy:
    def get_action(self, info: dict) -> np.ndarray:
        """
        Args:
            info: dict with all information collected from the environments e.g:
                  - 'pixels': current observation images (num_envs, C, H, W)
                  - 'goal': goal images if goal_conditioned (num_envs, C, H, W)
                  - 'state': low-dimensional state if available

        Returns:
            actions: Array of shape (num_envs, action_dim)
        """
        return actions
```

The simplest option is the built-in random policy:

```python
# Attach a random policy
policy = swm.policy.RandomPolicy(seed=42)
world.set_policy(policy)
```

For model-based control or planning, use `WorldModelPolicy` which wraps a [solver](api/solver.md) and a trained world model to plan actions:

```python
from stable_worldmodel.policy import WorldModelPolicy, PlanConfig
from stable_worldmodel.solver import CEMSolver

# Configure planning parameters
config = PlanConfig(
    horizon=10,             # Planning horizon
    receding_horizon=5,     # Steps to execute before replanning
    action_block=1,         # Action repeat / frame skip
    warm_start=True         # Reuse previous plan as initialization
)

world_model = ...           # load your model

# Create a planning policy with a solver and your trained model
policy = WorldModelPolicy(
    solver=CEMSolver(model=world_model, num_samples=300),
    config=config
)
world.set_policy(policy)
```

The `world_model` can be any object that implements the `get_cost` method:

```python
class MyWorldModel:
    def get_cost(self, info: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            info: Dictionary with all data (pixels, goal, state, etc.)
            action_candidates: Candidate action sequences (num_envs, num_samples, horizon, action_dim)

        Returns:
            costs: Cost per sample per env (num_envs, num_samples, 1). Lower is better.
        """
        return costs
```

The `WorldModelPolicy` internally calls the solver to optimize action sequences using the world model's cost predictions. See [Model-Based Planning](#model-based-planning) for a complete example.

## Dataset

Stable World-Model provides a small, pluggable data layer for recording and
loading episode datasets. Recording, loading, and conversion all go through
the same **format registry**, so the same code records to a Lance table or
to a folder of MP4 episodes — only the `format=` flag changes. See
the [Dataset API](api/dataset.md) for the full reference; the rest of this
section covers the everyday workflow.

### Recording a Dataset

Use `world.collect()` to roll out episodes and dump their trajectories. Each
info key becomes a column in the resulting file. The default writer is
`lance`; pass `format='hdf5'`, `'video'`, or `'folder'` to switch backends.

```python
world = swm.World('swm/PushT-v1', num_envs=8, image_shape=(224, 224))
policy = swm.policy.RandomPolicy(seed=42)  # can be your JEPA or RL policy
world.set_policy(policy)

# Lance table (default — recommended for training).
world.collect('data/pusht_random.lance', episodes=100, seed=0)

# Re-running the same call extends the table (mode='append' is the default
# for every writer); pass mode='overwrite' to start fresh.
world.collect('data/pusht_random.lance', episodes=50, seed=1)   # +50 episodes

# Or a directory with one MP4 per episode (compact, easy to inspect).
world.collect('data/pusht_random_video', episodes=100, seed=0,
              format='video')
```

!!! info "Expert Policies"
    Some environments come with a built-in weak expert policy for data collection. These policies are not optimal but provide better coverage than random actions. Check the environment documentation to see if an expert policy is available.

### Loading a Dataset

`swm.data.load_dataset()` resolves a name to a local path and dispatches to
the matching reader. It accepts a local path, a HuggingFace repo
(`<user>/<repo>`, downloaded to `$STABLEWM_HOME/datasets/`), or a
scheme-prefixed identifier such as `lerobot://lerobot/pusht`. `frameskip`
controls the stride between frames; `num_steps` is the sequence length
returned per sample.

```python
import stable_worldmodel as swm

dataset = swm.data.load_dataset(
    'data/pusht_random.lance',                    # autodetected as lance
    frameskip=1,
    num_steps=4,
    keys_to_load=['pixels', 'action', 'state'],
)

sample = dataset[0]
print(sample['pixels'].shape)   # (4, 3, H, W)
print(sample['action'].shape)   # (4, action_dim)
```

The same call works for the other formats:

```python
# Folder / Video (autodetected from directory layout)
swm.data.load_dataset('data/pusht_random_video', num_steps=4)

# LeRobot Hub dataset
swm.data.load_dataset(
    'lerobot://lerobot/pusht',
    primary_camera_key='observation.images.top',  # → `pixels`
    num_steps=4,
    keys_to_load=['pixels', 'action', 'proprio', 'ep_idx', 'step_idx'],
)
```

!!! info "LeRobot Support"
    LeRobot support is read-only and requires Python 3.12+. Install with `pip install 'stable-worldmodel[format]'`.

The returned dataset is compatible with PyTorch `DataLoader` for batched training.

### Converting Between Formats

`swm.data.convert()` migrates a dataset from one format to another, episode
by episode. Source format is autodetected; pass `dest_format` and any
writer kwargs:

```python
swm.data.convert(
    'data/pusht_random.lance',        # source
    'data/pusht_random_video',        # destination
    dest_format='video',
    fps=30,
)
```

Use the CLI to list all available datasets, or inspect a specific one:

```bash
swm datasets
swm inspect pusht_expert_train
```

### Recording Videos

To visualize a policy's behavior, pass `video=<dir>` to `evaluate()`. One mp4 per episode is written into the target directory — useful for debugging, qualitative evaluation, or generating figures for papers.

```python
world.evaluate(episodes=10, seed=0, video='./videos/')
```

### Recording Videos from Dataset

Use `record_video_from_dataset()` to replay and export stored episodes as MP4 videos. This is useful for visualizing collected data without re-running the environment.

```python
from stable_worldmodel.utils import record_video_from_dataset

record_video_from_dataset(
    video_path='./videos/',
    dataset=dataset,
    episode_idx=[0, 1, 2],  # Episodes to export
    max_steps=500,
    fps=30,
    viewname='pixels'       # Can also be a list: ['pixels', 'goal']
)
```

## Evaluation

`world.evaluate()` has two modes of operation, selected by whether you pass a `dataset`. Both measure how well a policy reaches a goal state, but they differ in how start and goal are sampled. **We recommend the dataset-driven mode** as it guarantees solvable tasks and enables fair comparisons across methods.

### Dataset-driven Evaluation (Recommended)

When you pass `dataset=...`, each env is assigned one dataset episode. The env starts at `start_steps[i]` of that episode and targets the state `goal_offset` steps later. Because the goal is taken from a completed trajectory, the task is **solvable within the eval budget** — the original trajectory already reached that state. The run ends when every env has either succeeded or exhausted its budget (`reset_mode='wait'` by default here).

```python
results = world.evaluate(
    dataset=dataset,
    episodes_idx=[0, 1, 2, 3],      # Which episodes to sample from
    start_steps=[0, 10, 20, 30],    # Starting timestep within each episode
    goal_offset=50,                 # Goal = state at start_step + 50
    eval_budget=100,                # Max env steps per episode
)
```

For example, if `start_steps[i]=10` and `goal_offset=50`, env `i` starts at timestep 10 of the trajectory and must reach the state that was at timestep 60. Since the original trajectory completed this transition, we know the task is achievable.

!!! note ""
    The length of `episodes_idx` and `start_steps` must match `num_envs`, as each environment evaluates one configuration in parallel.

### Episodic Evaluation

Without a dataset, `evaluate()` runs `episodes` episodes with randomly sampled goals and auto-resets terminated envs (`reset_mode='auto'` by default).

```python
world = swm.World('swm/PushT-v1', num_envs=4, image_shape=(224, 224))
world.set_policy(my_policy)

results = world.evaluate(
    episodes=50,        # Total episodes to evaluate
    seed=0,             # Seed for reproducibility
)

print(f"Success Rate: {results['success_rate']:.1f}%")
print(f"Episode Successes: {results['episode_successes']}")  # Per-episode results
```

The returned dictionary contains:

- `success_rate`: Overall success percentage
- `episode_successes`: Per-episode success flags (bool/uint array)
- `seeds`: Per-episode reset seeds

## Model-Based Planning

Use a trained world model for planning with solvers like CEM or MPPI:

```python
from stable_worldmodel.solver import CEMSolver
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy

# Create solver with your trained model
solver = CEMSolver(
    model=cost_model,       # Your trained world model
    num_samples=300,
    n_steps=30,
    device='cuda'
)

# Configure planning
config = PlanConfig(
    horizon=10,
    receding_horizon=5,
    action_block=1,
    warm_start=True
)

# Create planning policy
policy = WorldModelPolicy(solver=solver, config=config)

# Evaluate
world = swm.World('swm/PushT-v1', num_envs=1, image_shape=(224, 224))
world.set_policy(policy)
results = world.evaluate(episodes=50, seed=0)
```

## And then?

You should have a look at the [baselines](baselines.md) we implemented, tested, and benchmarked.

Explore the [Environments](envs/pusht.md) documentation for detailed information on each task, or dive into the [API Reference](api/world.md) for complete method signatures.
