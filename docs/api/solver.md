---
title: Solver
summary: Model-based planning solvers for action optimization
---

## **[ Base Class ]**

::: stable_worldmodel.solver.Solver
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.solver.Solver.configure

::: stable_worldmodel.solver.Solver.solve

::: stable_worldmodel.solver.Solver.action_dim
::: stable_worldmodel.solver.Solver.n_envs
::: stable_worldmodel.solver.Solver.horizon

## **[ Implementations ]**

::: stable_worldmodel.solver.CEMSolver
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.solver.CEMSolver.configure

::: stable_worldmodel.solver.CEMSolver.solve

::: stable_worldmodel.solver.ICEMSolver
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.solver.ICEMSolver.configure

::: stable_worldmodel.solver.ICEMSolver.solve

::: stable_worldmodel.solver.MPPISolver
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.solver.MPPISolver.configure

::: stable_worldmodel.solver.MPPISolver.solve

::: stable_worldmodel.solver.PredictiveSamplingSolver
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.solver.PredictiveSamplingSolver.configure

::: stable_worldmodel.solver.PredictiveSamplingSolver.solve

::: stable_worldmodel.solver.GradientSolver
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.solver.GradientSolver.configure

::: stable_worldmodel.solver.GradientSolver.solve

::: stable_worldmodel.solver.PGDSolver
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.solver.PGDSolver.configure

::: stable_worldmodel.solver.PGDSolver.solve

::: stable_worldmodel.solver.LagrangianSolver
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.solver.LagrangianSolver.configure

::: stable_worldmodel.solver.LagrangianSolver.solve

## **[ Callbacks ]**

Solvers accept a `callbacks=[...]` list of [`Callback`][stable_worldmodel.solver.callbacks.Callback]
objects. Each callback fires once per inner-loop step and accumulates a
per-batch buffer; final histories are returned in `outputs['callbacks']`,
keyed by `cb.output_key` (defaults to the class name).

```python
from stable_worldmodel.solver import GradientSolver
from stable_worldmodel.solver.callbacks import (
    BestCostRecorder, GradNormRecorder, ActionNormRecorder,
)

solver = GradientSolver(
    model=model, n_steps=20, num_samples=8,
    callbacks=[
        BestCostRecorder(),                # mean over envs (default)
        GradNormRecorder(reduction='none'), # one entry per env
        ActionNormRecorder(reduction='sum'),
    ],
)
solver.configure(action_space=action_space, n_envs=4, config=config)
out = solver.solve(info_dict)

# out['callbacks']['BestCostRecorder']  -> list[list[float]]   (batches x steps)
# out['callbacks']['GradNormRecorder']  -> list[list[list[float]]]
```

### Reduction modes

Every callback accepts `reduction ∈ {'mean', 'sum', 'none'}`. Reduction is
applied across the env axis only; within-sample reductions (e.g. min over
samples for `BestCostRecorder`) are intrinsic to each metric.

| Mode     | Output per step                            |
|----------|--------------------------------------------|
| `'mean'` | scalar (default)                           |
| `'sum'`  | scalar                                     |
| `'none'` | `list[float]` — one value per env in batch |

### Available callbacks

| Callback | Solver(s) | Records |
|---|---|---|
| `BestCostRecorder` | any | min cost over samples |
| `MeanCostRecorder` | any | mean cost over samples |
| `GradNormRecorder` | GD | L2 norm of action gradient (optional `per_step` for per-horizon-step values) |
| `ActionNormRecorder` | GD | L2 norm of action tensor |
| `EliteCostRecorder` | CEM, iCEM | dict of elite cost stats (mean/min/max) |
| `VarNormRecorder` | CEM, iCEM | mean variance of action distribution |
| `MeanShiftRecorder` | CEM, iCEM | L2 distance between consecutive means |
| `EliteSpreadRecorder` | CEM, iCEM | within-elite std (top-k diversity) |

### Writing a custom callback

Subclass [`Callback`][stable_worldmodel.solver.callbacks.Callback] and
implement `compute(**state)`. Pull the tensors you need from `state` and
call `self._reduce(per_env_tensor)` to honour the reduction mode.

```python
from stable_worldmodel.solver.callbacks import Callback

class CostRangeRecorder(Callback):
    """Records per-env (max - min) cost across the sample population."""

    def compute(self, **state):
        costs = state['costs'].detach()           # (B, N)
        per_env = costs.max(dim=1).values - costs.min(dim=1).values
        return self._reduce(per_env)
```

State keys passed by each solver:

- **GD**: `step`, `params`, `cost`, `costs`
- **CEM**: `step`, `candidates`, `costs`, `topk_vals`, `topk_inds`,
  `topk_candidates`, `mean`, `var`, `prev_mean`, `prev_var`
- **iCEM**: same as CEM plus `action_low`, `action_high`

::: stable_worldmodel.solver.callbacks.Callback
    options:
        heading_level: 3
        members:
            - reset
            - start_batch
            - end_solve
            - compute
            - output_key

::: stable_worldmodel.solver.callbacks.BestCostRecorder
::: stable_worldmodel.solver.callbacks.MeanCostRecorder
::: stable_worldmodel.solver.callbacks.GradNormRecorder
::: stable_worldmodel.solver.callbacks.ActionNormRecorder
::: stable_worldmodel.solver.callbacks.EliteCostRecorder
::: stable_worldmodel.solver.callbacks.VarNormRecorder
::: stable_worldmodel.solver.callbacks.MeanShiftRecorder
::: stable_worldmodel.solver.callbacks.EliteSpreadRecorder

## **[ Example: Constrained Planning with LagrangianSolver ]**

The `LagrangianSolver` extends gradient-based planning to handle **inequality
constraints** of the form `g(a) ≤ 0`. It uses the augmented Lagrangian method:
dual variables (λ) are maintained per environment and updated via dual ascent
after each inner optimisation loop, while a quadratic penalty term (controlled
by `rho`) enforces feasibility.

```python
import dataclasses
import torch
import gymnasium as gym
import numpy as np
from stable_worldmodel.solver import LagrangianSolver
from stable_worldmodel.policy import PlanConfig


# ── 1. Define a world model with cost and optional constraints ──────────────

class MyModel(torch.nn.Module):
    """Minimal example: cost is MSE to a goal; two inequality constraints."""

    def get_cost(self, info_dict, action_candidates):
        # action_candidates: (B, S, H, D)
        # returns:           (B, S)
        goal = torch.zeros(action_candidates.shape[-1])
        return (action_candidates.mean(dim=2) - goal).pow(2).mean(dim=-1)

    def get_constraints(self, info_dict, action_candidates):
        # returns: (B, S, C)  — violated when > 0
        # g0: action L2 norm <= 1
        g0 = action_candidates.norm(dim=-1).mean(dim=2) - 1.0
        # g1: first action dimension <= 0.5
        g1 = action_candidates[..., 0].mean(dim=2) - 0.5
        return torch.stack([g0, g1], dim=-1)


# ── 2. Build and configure the solver ──────────────────────────────────────

model = MyModel()

solver = LagrangianSolver(
    model=model,
    n_steps=30,            # inner gradient steps per outer iteration
    n_outer_steps=10,      # dual-ascent (outer) iterations
    num_samples=8,         # parallel action candidates per env
    rho_init=1.0,          # initial quadratic penalty coefficient
    rho_scale=2.0,         # rho doubles each outer step
    rho_max=1e4,
    persist_multipliers=True,  # warm-start λ across planning calls
    optimizer_kwargs={"lr": 0.05},
)

action_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                              shape=(1, 4), dtype=np.float32)
config = PlanConfig(horizon=10, receding_horizon=1, action_block=1)
solver.configure(action_space=action_space, n_envs=2, config=config)


# ── 3. Solve ────────────────────────────────────────────────────────────────

info_dict = {"obs": torch.zeros(2, 4)}  # current env observations
out = solver.solve(info_dict)

print(out["actions"].shape)        # (2, 10, 4)  — best action per env
print(out["lambdas"])              # (2, 2)       — dual variables
print(out["constraint_violation"]) # mean ReLU(g) across samples


# ── 4. Receding-horizon planning (warm start) ───────────────────────────────

# Execute the first step, shift the plan, re-plan
executed_steps = 1
remaining = out["actions"][:, executed_steps:, :]   # (2, 9, 4)
out2 = solver.solve(info_dict, init_action=remaining)
```

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `n_steps` | — | Inner gradient steps per outer iteration |
| `n_outer_steps` | `5` | Dual-ascent iterations |
| `rho_init` | `1.0` | Initial quadratic penalty weight |
| `rho_scale` | `2.0` | Multiplicative growth for `rho` each outer step |
| `rho_max` | `1e4` | Upper bound on `rho` |
| `persist_multipliers` | `True` | Keep λ across `solve()` calls (warm start) |
| `num_samples` | `1` | Parallel candidate trajectories per environment |
| `action_noise` | `0.0` | Gaussian noise injected each inner step |

### Constraint protocol

Your model must implement `get_constraints(info_dict, action_candidates) -> Tensor`
returning shape `(B, S, C)`.  A constraint is **satisfied** when its value is ≤ 0.

To enforce an **equality** `h(a) = 0`, add two constraints: `h(a) ≤ 0` and
`-h(a) ≤ 0`.
