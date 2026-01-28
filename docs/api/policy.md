title: Policy
summary: Agent policies for interacting with environments
---

# Policy Management

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
from stable_worldmodel.solver.random import RandomSolver

# 1. Define Planning Configuration
cfg = PlanConfig(
    horizon=10,
    receding_horizon=1,
    action_block=1
)

# 2. Instantiate a Solver
solver = RandomSolver() # Or CEMSolver, MPPI, etc.

# 3. Create the Policy
policy = WorldModelPolicy(
    solver=solver,
    config=cfg
)
```
///

!!! note "Protocol"
    All policies must implement the `get_action(obs, **kwargs)` method. The `World` class automatically calls `set_env()` when a policy is attached.

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

::: stable_worldmodel.policy.WorldModelPolicy
    options:
        heading_level: 2
        members: false
        show_source: false

::: stable_worldmodel.policy.WorldModelPolicy.get_action


## **[ Utils ]**

::: stable_worldmodel.policy.AutoCostModel
    options:
        heading_level: 3
        show_source: false
