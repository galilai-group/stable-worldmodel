---
title: Planning
summary: Composable objectives and cost evaluators for planning
---

Solvers optimize action sequences against a [`Costable`][stable_worldmodel.planning.Costable] — anything exposing `get_cost(info_dict, action_candidates)`. Some world models implement it natively (e.g. TD-MPC2); for the others, this module provides the glue: a [`ShootingCostEvaluator`][stable_worldmodel.planning.ShootingCostEvaluator] composes any model exposing the [`Dynamics`][stable_worldmodel.planning.Dynamics] surface (`encode`/`rollout`) with a swappable [`Objective`][stable_worldmodel.planning.Objective], so changing the planning cost never requires subclassing the world model.

## **[ Quick Tour ]**

```python
import stable_worldmodel as swm
from stable_worldmodel.planning import (
    CEMSolver,
    ControlPenalty,
    GoalMSE,
    ShootingCostEvaluator,
    WeightedSum,
)

model = swm.wm.utils.load_pretrained('lewm/pusht')

# 1. Single-term cost: last-step MSE to the goal embedding
cost = ShootingCostEvaluator(model, GoalMSE())

# 2. Multi-term cost: goal distance + action magnitude penalty
cost = ShootingCostEvaluator(
    model,
    WeightedSum([(1.0, GoalMSE()), (0.1, ControlPenalty())]),
)

# 3. Plug into any solver — the evaluator duck-types as a Costable
solver = CEMSolver(cost=cost, n_steps=30, num_samples=300, topk=30)
config = swm.PlanConfig(horizon=10, receding_horizon=1, action_block=1)
policy = swm.policy.WorldModelPolicy(solver=solver, config=config)
```

To plan under inequality constraints, pass objectives as `constraints=` — the
evaluator then exposes `get_constraints` and satisfies the
[`Constrainable`][stable_worldmodel.planning.Constrainable] protocol that
[`LagrangianSolver`](solver.md#example-constrained-planning-with-lagrangiansolver)
feature-detects:

```python
cost = ShootingCostEvaluator(
    model,
    GoalMSE(),
    constraints=[ControlPenalty()],  # satisfied when <= 0
)
```

### Writing a custom objective

An objective is any callable mapping a populated `info_dict` to a
per-candidate cost of shape `(B, S)`. The evaluator rolls candidates out
first, so the `info_dict` already holds the rollout outputs (e.g.
`predicted_emb`) plus the raw candidates under `action_candidates`:

```python
import torch.nn as nn


class SmoothnessPenalty(nn.Module):
    """Penalizes large action changes between consecutive steps."""

    def forward(self, info_dict: dict) -> torch.Tensor:
        actions = info_dict['action_candidates']  # (B, S, H, action_dim)
        deltas = actions[..., 1:, :] - actions[..., :-1, :]
        return deltas.pow(2).sum(dim=tuple(range(2, actions.ndim)))
```

## **[ Evaluator ]**

::: stable_worldmodel.planning.ShootingCostEvaluator
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.planning.ShootingCostEvaluator.get_cost

::: stable_worldmodel.planning.ShootingCostEvaluator.criterion

::: stable_worldmodel.planning.default_goal_encode

## **[ Objectives ]**

::: stable_worldmodel.planning.GoalMSE
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.planning.ControlPenalty
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.planning.WeightedSum
    options:
        heading_level: 3
        members: false
        show_source: false

## **[ Protocols ]**

The structural contracts live in `stable_worldmodel.protocols` and are
re-exported from `stable_worldmodel.planning`. They are `Protocol` classes:
nothing subclasses them, anything with the right methods satisfies them.

::: stable_worldmodel.planning.Costable
    options:
        heading_level: 3
        show_source: false

::: stable_worldmodel.planning.Constrainable
    options:
        heading_level: 3
        show_source: false

::: stable_worldmodel.planning.Dynamics
    options:
        heading_level: 3
        show_source: false

::: stable_worldmodel.planning.Objective
    options:
        heading_level: 3
        show_source: false
