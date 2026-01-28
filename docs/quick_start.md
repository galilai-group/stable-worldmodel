title: Quick Start Guide
summary: Get started with stable-worldmodel in minutes
sidebar_title: Quickstart
---

## World

The `World` class is the main interface for interacting with environments. It wraps vectorized Gymnasium environments and provides a unified API for data collection, evaluation, and planning.

```python
import stable_worldmodel as swm

world = swm.World(env='swm/PushT-v1', num_envs=4)
```

Unlike the traditional Gymnasium interface that returns `(obs, reward, terminated, truncated, info)` from `step()`, `World` adopts a simpler philosophy: all data produced by the environments is stored in a single dictionary accessible via `world.infos`. The `step()` and `reset()` methods operate on all environments simultaneously and update this dictionary in place.

```python
world.reset(seed=0)
world.step()

# Access all environment data through world.infos
print(world.infos['pixels'].shape)    # (num_envs, C, H, W)
print(world.infos['reward'])          # (num_envs,)
print(world.infos['terminated'])      # (num_envs,)
```

See the [World API](api/world.md) for all available parameters and methods.

!!! info ""
    Just want the environments? All environments are self-contained and follow the standard [Gymnasium](https://gymnasium.farama.org/) API. Simply import the library to register them.

        :::python                                                                                                            
        import gymnasium as gym                                                                                              
        import stable_worldmodel  # registers all environments                                                               
                                                                                                                            
        env = gym.make('swm/PushT-v1', render_mode='rgb_array')   


### Factors of Variation (FoV)

A key feature of stable-worldmodel is the **variation space**, which defines all customizable properties of an environment. This enables domain randomization, continual learning experiments, and out-of-distribution evaluation by controlling factors like colors, shapes, sizes, and physics properties.

Each environment exposes a `variation_space` that describes what can be varied:

```python
# View the variation space structure
print(world.single_variation_space.to_str())
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

A `World` requires a policy to determine actions. When you call `world.step()`, it internally calls `policy.get_action(info)` to obtain the actions for all environments.

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

Stable World-Model provides utilities for recording and loading episode datasets in HDF5 format.

### Recording a Dataset

Use `world.record_dataset()` to collect episodes and save them in HDF5 format. The dataset is saved to `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). This is useful for collecting expert demonstrations, random exploration data, or rollouts from a trained policy.

```python
world = swm.World('swm/PushT-v1', num_envs=8, image_shape=(224, 224))
policy = swm.policy.RandomPolicy(seed=42) # can be your JEPA or RL Policy
world.set_policy(policy)

# Record 100 episodes to HDF5
world.record_dataset(
    dataset_name='pusht_random',
    episodes=100,
    seed=0
)
```

!!! info "Expert Policies"
    Some environments come with a built-in weak expert policy for data collection. These policies are not optimal but provide better coverage than random actions. Check the environment documentation to see if an expert policy is available.

### Loading a Dataset

Load recorded datasets using `HDF5Dataset`. The `frameskip` parameter controls the stride between frames, and `num_steps` sets the sequence length returned per sample. This makes it easy to train models on temporal sequences of observations and actions.

```python
from stable_worldmodel.data import HDF5Dataset

dataset = HDF5Dataset(
    name='pusht_random',
    frameskip=1,  # stride between frames
    num_steps=4,  # sequence length
    keys_to_load=['pixels', 'action', 'state']
)

# Access samples
sample = dataset[0]
print(sample['pixels'].shape)   # (4, 3, H, W)
print(sample['action'].shape)   # (4, action_dim)
```

The dataset is compatible with PyTorch `DataLoader` for batched training.

### Recording Videos

Use `world.record_video()` to visualize the behavior of the current world policy. This is helpful for debugging, qualitative evaluation, or generating figures for papers.

```python
world.record_video(
    video_path='./videos/',
    max_steps=500,
    fps=30,
    seed=0
)
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

Stable World-Model provides two ways to evaluate world models: `evaluate()` and `evaluate_from_dataset()`. Both methods measure how well a policy can reach a goal state, but they differ in how the goal is sampled. **We recommend using `evaluate_from_dataset()`** as it guarantees solvable tasks and enables fair comparisons across methods.

### Offline Evaluation with `evaluate_from_dataset()` (Recommended)

Use `world.evaluate_from_dataset()` to evaluate from starting states taken from an existing dataset. This method:

1. Takes a **starting state** from a trajectory in the dataset (`start_steps`)
2. Sets the **goal** as the state `goal_offset_steps` ahead in that same trajectory

Because the goal is taken from a completed trajectory, this guarantees the problem is **solvable within the given budget**—the original trajectory already reached that state. This makes evaluation fairer and more interpretable.

```python
results = world.evaluate_from_dataset(
    dataset=dataset,
    episodes_idx=[0, 1, 2, 3],      # Which episodes to sample from
    start_steps=[0, 10, 20, 30],    # Starting timestep within each episode
    goal_offset_steps=50,           # Goal = state at start_step + 50
    eval_budget=100,                # Max budget (2x the current temp dist)
)
```

For example, if `start_steps=10` and `goal_offset_steps=50`, the agent starts at timestep 10 of the trajectory and must reach the state that was at timestep 60. Since the original trajectory completed this transition, we know the task is achievable.

!!! note ""
    The length of `episodes_idx` and `start_steps` must match `num_envs`, as each environment evaluates one configuration in parallel.

### Online Evaluation with `evaluate()`

Use `world.evaluate()` to evaluate your policy with randomly sampled goals. At each episode, the environment samples a random goal state that the agent must reach.

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

The `evaluate()` method runs episodes in parallel across `num_envs` environments, automatically handling episode resets and aggregating results. The returned dictionary contains:

- `success_rate`: Overall success percentage
- `episode_successes`: Array of per-episode success (0 or 1)
- `episode_count`: Total episodes completed
- Any custom metrics specified in `eval_keys`

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