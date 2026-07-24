title: Policy
summary: Agent policies for interacting with environments
---

Policies determine the actions taken by agents in the environment. `stable_worldmodel` provides base classes and implementations for random, expert, and model-based policies.

/// tab | Random Policy
A simple policy that samples actions uniformly from the environment's action space.

```python
from stable_worldmodel.policy import RandomPolicy

# Create a random policy
policy = RandomPolicy(seed=42)

# Attach to a world/env later
# world.set_policy(policy)
```
///

/// tab | World Model Policy
A policy that uses a `Solver` (like CEM or MPPI) and a World Model to plan actions.

```python
from stable_worldmodel.policy import WorldModelPolicy, PlanConfig
from stable_worldmodel.planning import CEMSolver

# 1. Define Planning Configuration
cfg = PlanConfig(
    horizon=10,
    receding_horizon=1,
    action_block=1
)

# 2. Instantiate a Solver
solver = CEMSolver(cost=world_model) # Or MPPISolver, GradientSolver, etc.

# 3. Create the Policy
policy = WorldModelPolicy(
    solver=solver,
    config=cfg
)
```
///

/// tab | Feed-Forward Policy
A policy that uses a neural network model for direct action prediction via a single forward pass. Useful for imitation learning policies like Goal-Conditioned Behavioral Cloning (GCBC).

```python
from stable_worldmodel.policy import FeedForwardPolicy
from stable_worldmodel.wm.utils import load_pretrained

# 1. Load a pre-trained model with a get_action method
model = load_pretrained("path/to/checkpoint")

# 2. Create the Policy
policy = FeedForwardPolicy(
    model=model,
    process={"action": action_scaler},  # Optional preprocessors
    transform={"pixels": image_transform}  # Optional transforms
)
```
///

!!! note "Protocol"
    All policies must implement the `get_action(obs, **kwargs)` method. The `World` class automatically calls `set_env()` when a policy is attached.

!!! note "Planning with observation history"
    With `PlanConfig(history_len > 1)`, `WorldModelPolicy` keeps a per-env [`HistoryBuffer`](buffer.md) over its `history_keys` (default `('pixels',)`) plus the executed actions. At each replan the solver's info dict carries the frames at the last `history_len` block boundaries (`pixels` gains a real time dim) and the executed action blocks between them under `action_history` (solver space, one flattened block of `action_block` env actions per step). Candidates remain strictly future — see the [rollout contract](planning.md#the-rollout-contract). Markovian `Costable` models (TD-MPC2) only support the default `history_len=1`.

!!! warning "Episode-start warm-up feeds synthetic context"
    For the first `(history_len - 1) * action_block` env steps of each episode there is not enough real history yet, and the world model receives **fake repeated frames**: the missing context slots are filled with copies of the episode's first frame, with zero action blocks between them (as if the env had been stationary before the episode began). All context is real after that window. See the [buffer warm-up docs](buffer.md#warm-up) for the rationale.

::: stable_worldmodel.policy.PlanConfig
    options:
        heading_level: 2
        members: false
        show_source: false

::: stable_worldmodel.policy.BasePolicy
    options:
        heading_level: 2
        members: false
        show_source: false

::: stable_worldmodel.policy.BasePolicy.get_action

::: stable_worldmodel.policy.BasePolicy.set_env

::: stable_worldmodel.policy.BasePolicy._prepare_info

::: stable_worldmodel.policy.RandomPolicy
    options:
        heading_level: 2
        members: false
        show_source: false

::: stable_worldmodel.policy.RandomPolicy.get_action

::: stable_worldmodel.policy.ExpertPolicy
    options:
        heading_level: 2
        members: false
        show_source: false

::: stable_worldmodel.policy.ExpertPolicy.get_action

::: stable_worldmodel.policy.FeedForwardPolicy
    options:
        heading_level: 2
        members: false
        show_source: false

::: stable_worldmodel.policy.FeedForwardPolicy.get_action

::: stable_worldmodel.policy.WorldModelPolicy
    options:
        heading_level: 2
        members: false
        show_source: false

::: stable_worldmodel.policy.WorldModelPolicy.get_action


## **[ Utils ]**

Load a pretrained checkpoint (a folder with `weights.pt` + `config.json`) with
[`load_pretrained`](../guides/checkpoints.md). It reconstructs the model from
`config.json` and loads the weights — the model is returned directly, ready to
pass to `FeedForwardPolicy` / `WorldModelPolicy`.

::: stable_worldmodel.wm.utils.load_pretrained
    options:
        heading_level: 3
        show_source: false

Use the CLI to list available model checkpoints:

```bash
swm checkpoints
swm checkpoints pusht  # filter by name
```
