---
title: Solver
summary: Model-based planning solvers — design, conventions, and implementations
---

## **[ Overview ]**

A **solver** is the optimisation engine of a model-based planner. Given a
*world model* that scores action sequences, a solver searches the action
space for a low-cost plan. `stable-worldmodel` ships a small family of
solvers (CEM, iCEM, MPPI, gradient descent, projected gradient,
predictive sampling, augmented Lagrangian, categorical CEM) and a
`CompositeSolver` that composes them across `Dict` action spaces.

All solvers expose the same minimal public surface — the
[`Solver`][stable_worldmodel.solver.Solver] protocol — so policies and
training loops can swap optimisers without changing surrounding code:

```python
solver.configure(action_space=action_space, n_envs=n_envs, config=plan_cfg)
out = solver.solve(info_dict, init_action=warm_start)
actions = out["actions"]   # (n_envs, horizon, action_dim)
```

## **[ Design Philosophy ]**

The package is built around four ideas. They are worth reading once
before subclassing or composing solvers — most subtleties downstream
follow from them.

### 1. One outer loop, four hooks

Every search algorithm in this package is structured the same way:

1. **`init_state`** — build an opaque per-env search state from
   (optionally) a warm-start.
2. **`propose`** — sample candidate action plans from the current state.
3. **evaluate** — `model.get_cost` scores all candidates in one batched
   call.
4. **`update`** — refit the state (mean/variance, distribution, action
   parameters, dual variables, …) from the costs.
5. **`finalize`** — read out the final plan from the converged state.

[`BaseSolver`][stable_worldmodel.solver.BaseSolver] owns the outer
iteration loop, env-batching, info-dict expansion, callback firing, and
state plumbing. Subclasses implement only the four algorithm-specific
hooks. The common loop is roughly:

```python
state = self.init_state(n_envs, init_action)
for step in range(n_steps):
    candidates = self.propose(state)              # (B, N, H, D)
    costs       = model.get_cost(infos, candidates)  # (B, N)
    state, payload = self.update(state, candidates, costs)
    fire_callbacks(step, candidates, costs, **payload)
actions = self.finalize(state)
```

This is the **template-method pattern**: the parent owns the *sequence*,
the child owns the *steps*. Two consequences:

- A new solver is typically 30–80 lines (see CEM, MPPI, predictive
  sampling).
- Cross-cutting concerns — env-batching, callbacks, dtype/device,
  warm-starts — are written once and reused everywhere.

### 2. Protocol over inheritance at the edge

Inside the package, solvers subclass `BaseSolver`. At the *boundary*
(the policy, the training loop, user code) they are seen through the
[`Solver`][stable_worldmodel.solver.Solver] **runtime-checkable
Protocol**:

```python
@runtime_checkable
class Solver(Protocol):
    def configure(self, *, action_space, n_envs, config) -> None: ...
    def solve(self, info_dict, init_action=None) -> dict: ...
    @property
    def action_dim(self) -> int: ...
    @property
    def n_envs(self) -> int: ...
    @property
    def horizon(self) -> int: ...
```

A class is a valid `Solver` as soon as it implements this surface — no
inheritance required. This is what lets `GradientSolver` and
`PredictiveSamplingSolver` (which do not inherit from `BaseSolver`) and
external user-written solvers all be drop-in compatible.

Use `isinstance(obj, Solver)` to validate at boundaries; use
`BaseSolver` as the default starting point when writing a new one.

### 3. Composition for structured action spaces

Real environments often have heterogeneous action spaces — a continuous
gripper command alongside a discrete mode switch, for example. Rather
than baking this into every solver,
[`CompositeSolver`][stable_worldmodel.solver.CompositeSolver] holds one
solver per sub-key and runs **a single shared outer loop**:

- Each child proposes candidates for its own component.
- The joint dict of candidates is passed through **one** call to
  `model.get_cost`.
- Every child refits from the same cost tensor.

The cost is the only coupling, but it is enough — children see each
other's choices in the score they refit against.

### 4. Callbacks for diagnostics, not control flow

Callbacks observe the loop; they never steer it. Each callback receives
the per-step state (`step`, `candidates`, `costs`, plus solver-specific
payload like `topk_vals`, `mean`, `var`, `params`), reduces it to a
scalar (or list) per env, and accumulates a history that is returned in
`outputs["callbacks"]`. This keeps `solve()` clean, makes solvers
unit-testable without instrumentation, and lets users add metrics
(`GradNormRecorder`, `EliteSpreadRecorder`, custom ones) without
touching solver code.

## **[ Mental Model: Tensor Conventions ]**

Solvers are vectorised over three dimensions. Internalising the layout
makes the rest of the API obvious.

| Symbol | Meaning                                          |
|--------|--------------------------------------------------|
| `B`    | env-batch (subset of `n_envs` processed at once) |
| `N`    | `num_samples` — candidate trajectories per env   |
| `H`    | `horizon` — planning horizon in timesteps        |
| `D`    | `action_dim × action_block` — flat per-step action |

Shapes you will see:

| Object                                | Shape                | Notes                                      |
|---------------------------------------|----------------------|--------------------------------------------|
| Candidate plans (continuous solvers)  | `(B, N, H, D)`       | Tensor                                     |
| Candidate plans (composite)           | `dict[str, (B,N,H,Dk)]` | One tensor per sub-action key           |
| Per-candidate cost                    | `(B, N)`             | Returned by `model.get_cost`               |
| Final actions                         | `(n_envs, H, D)`     | After `finalize`, on CPU                   |
| Expanded `info_dict` value            | `(B, N, ...)`        | Each entry broadcast over the sample axis  |

Two conventions worth highlighting:

- **`action_block`** lets a single planning step emit several consecutive
  environment actions. The flat per-step dim is `action_dim × action_block`;
  the planner treats the block as one decision, the executor unrolls it.
- **Warm-starts** (`init_action`) may supply only a *prefix* of the horizon.
  The base classes pad the remainder with zeros, so receding-horizon
  control naturally reuses the previous plan minus the executed step.

## **[ Solver Lifecycle ]**

A solver is used in three phases:

1. **Construct** — algorithm hyperparameters only (`n_steps`,
   `num_samples`, learning rate, noise scale, …). At this point the
   solver knows nothing about the environment.
2. **Configure** — bind to the environment: `action_space`, `n_envs`,
   `config` (a `PlanConfig`-like object exposing `horizon` and
   `action_block`). After `configure`, properties like `action_dim`,
   `horizon`, `n_envs` are valid.
3. **Solve** — call `solve(info_dict, init_action=...)` repeatedly. The
   solver may keep state across calls (e.g. `LagrangianSolver` warm-starts
   its dual variables; CEM's mean is implicitly carried via
   `init_action`).

```python
# 1. Construct
solver = CEMSolver(model=world_model, n_steps=20, num_samples=128, topk=16)

# 2. Configure (once per environment)
solver.configure(action_space=env.action_space, n_envs=8, config=plan_cfg)

# 3. Solve (every control step)
for obs in rollout:
    out  = solver.solve({"obs": obs}, init_action=prev_plan)
    plan = out["actions"]
```

## **[ Output Contract ]**

`solve()` returns a dict. Every solver guarantees:

| Key         | Shape / Type                                | Always present |
|-------------|---------------------------------------------|:---:|
| `actions`   | `(n_envs, H, D)` tensor on CPU              | yes |
| `costs`     | `list[float]` of length `n_envs`            | yes for `BaseSolver` subclasses |
| `callbacks` | `dict[output_key, list[list[Any]]]`         | only if callbacks were attached |

Solvers are free to add their own keys via `extra_outputs(state)`
(`mean`, `var`, `probs`, `lambdas`, `constraint_violation`, …). See the
implementation tables below for what each one returns.

## **[ Solver Protocol ]**

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

## **[ BaseSolver ]**

The recommended starting point for a new sampling-based solver. It owns
the outer loop, env-batching, callback firing, and state plumbing.
Subclasses implement four hooks; everything else is inherited.

::: stable_worldmodel.solver.BaseSolver
    options:
        heading_level: 3
        members:
            - configure
            - solve
            - init_state
            - propose
            - update
            - finalize
            - step
            - extra_outputs
            - action_block
            - dtype
            - horizon
            - n_envs
        show_source: false

### Writing a custom solver

The minimal recipe: subclass `BaseSolver`, implement the four hooks,
and you get the outer loop, env-batching, info-dict expansion, dtype
handling, and callbacks for free.

```python
import numpy as np
import torch
from stable_worldmodel.solver import BaseSolver


class RandomShootingSolver(BaseSolver):
    """Baseline: sample candidates around the current best plan, keep argmin.

    State is the current best plan ``(B, H, D)``. Each iteration samples
    ``num_samples`` Gaussian perturbations around it and replaces the
    state with the cheapest candidate.
    """

    def __init__(self, model, n_steps, num_samples, noise=1.0, **kw):
        super().__init__(model=model, n_steps=n_steps,
                         num_samples=num_samples, **kw)
        self.noise = noise

    def configure(self, *, action_space, n_envs, config):
        super().configure(action_space=action_space, n_envs=n_envs, config=config)
        # Convention: vector Box spaces have a leading "envs" dim.
        self._action_dim = int(np.prod(action_space.shape[1:]))

    @property
    def action_dim(self):
        return self._action_dim * self.action_block

    # --- four hooks ---

    def init_state(self, n_envs, init=None):
        if init is None:
            init = torch.zeros(n_envs, self.horizon, self.action_dim,
                               dtype=self.dtype, device=self.device)
        return init

    def propose(self, state):
        # state: (B, H, D) -> candidates: (B, N, H, D)
        eps = torch.randn(state.shape[0], self.num_samples, *state.shape[1:],
                          device=self.device, dtype=self.dtype) * self.noise
        cands = state.unsqueeze(1) + eps
        cands[:, 0] = state                      # always keep current best
        return cands

    def update(self, state, candidates, costs):
        best = costs.argmin(dim=1)               # (B,)
        idx  = torch.arange(state.shape[0], device=self.device)
        new_state = candidates[idx, best]
        return new_state, {}                     # no extra callback payload

    def finalize(self, state):
        return state.detach().cpu()
```

That is a complete, working solver. It is callable
(`solver(info_dict)`), batches over environments, accepts callbacks,
respects warm-starts, and integrates with `CompositeSolver`.

The `update` hook returns `(new_state, payload)`. `payload` is merged
with `step`, `candidates`, `costs` and forwarded to every callback —
this is how solver-specific diagnostics (`mean`, `var`, `topk_vals`,
`probs`, `params`, …) flow out without polluting the loop.

## **[ Implementations ]**

| Solver | Action space | Gradient required | Strengths |
|--------|--------------|:-:|-----------|
| `PredictiveSamplingSolver` | Box       | no  | One-shot, real-time control, tiny code path |
| `CEMSolver`                | Box       | no  | Robust on smooth costs, well-understood |
| `ICEMSolver`               | Box       | no  | Sample-efficient CEM (colored noise + elite memory) |
| `MPPISolver`               | Box       | no  | Soft-weighted refit, good on noisy costs |
| `GradientSolver`           | Box       | yes | Direct backprop through the world model |
| `LagrangianSolver`         | Box       | yes | Inequality-constrained planning (`g(a) ≤ 0`) |
| `PGDSolver`                | Discrete  | yes | Projected gradient on the simplex |
| `CategoricalCEMSolver`     | Discrete  | no  | Gradient-free CEM over categorical distributions |

Reference docs for each implementation follow below.

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

::: stable_worldmodel.solver.CategoricalCEMSolver
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.solver.CategoricalCEMSolver.configure

::: stable_worldmodel.solver.CategoricalCEMSolver.solve

::: stable_worldmodel.solver.LagrangianSolver
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.solver.LagrangianSolver.configure

::: stable_worldmodel.solver.LagrangianSolver.solve

## **[ Composite Solvers ]**

[`CompositeSolver`][stable_worldmodel.solver.CompositeSolver] composes
one `BaseSolver` per key of a `gym.spaces.Dict` action space and runs a
single shared outer loop. Each child proposes candidates for its
component; the joint dict is evaluated through **one** call to
`model.get_cost`; each child refits from the same cost tensor.

This keeps coupled action components synchronised: the discrete mode
switch sees the continuous gripper command's effect in the cost, and
vice versa.

### Constraints

- All children must share `num_samples` so candidates align under one
  joint cost evaluation.
- `n_steps`, `batch_size`, and the planning `config` are taken from the
  first child; per-child `n_steps` is ignored — the composite owns the
  outer loop.
- Callbacks attach to the **composite**, not the children. Payload keys
  are namespaced by sub-key: `"<child>.mean"`, `"<child>.topk_vals"`, …
- The model's `get_cost(info_dict, action_candidates)` receives
  `action_candidates` as a `dict[str, Tensor]` rather than a single
  tensor.

### Example

```python
import gymnasium as gym
from stable_worldmodel.solver import (
    CEMSolver, CategoricalCEMSolver, CompositeSolver,
)
from stable_worldmodel.policy import PlanConfig

# Hybrid action space: continuous arm command + discrete gripper mode.
action_space = gym.spaces.Dict({
    "arm":     gym.spaces.Box(low=-1, high=1, shape=(1, 6)),
    "gripper": gym.spaces.Discrete(3),
})

solver = CompositeSolver(
    sub_solvers={
        "arm":     CEMSolver(model=model, n_steps=20, num_samples=128, topk=16),
        "gripper": CategoricalCEMSolver(model=model, n_steps=20,
                                        num_samples=128, topk=16),
    },
)

solver.configure(action_space=action_space, n_envs=4,
                 config=PlanConfig(horizon=8, action_block=1))

out = solver.solve({"obs": obs})
out["actions"]              # dict: {"arm": (...), "gripper": (...)}
out["arm.mean"]             # CEM extras, namespaced
out["gripper.probs"]        # CategoricalCEM extras, namespaced
```

::: stable_worldmodel.solver.CompositeSolver
    options:
        heading_level: 3
        members:
            - configure
            - init_state
            - propose
            - update
            - finalize
            - extra_outputs
        show_source: false

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
        BestCostRecorder(),                 # mean over envs (default)
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
- **CategoricalCEM**: `step`, `candidates`, `costs`, `topk_vals`, `topk_inds`,
  `topk_candidates`, `probs`, `prev_probs`

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


# 1. Define a world model with cost and optional constraints

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


# 2. Build and configure the solver

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


# 3. Solve

info_dict = {"obs": torch.zeros(2, 4)}  # current env observations
out = solver.solve(info_dict)

print(out["actions"].shape)        # (2, 10, 4)  — best action per env
print(out["lambdas"])              # (2, 2)       — dual variables
print(out["constraint_violation"]) # mean ReLU(g) across samples


# 4. Receding-horizon planning (warm start)

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

## **[ Example: Discrete Planning with CategoricalCEMSolver ]**

`CategoricalCEMSolver` is the discrete-action analogue of `CEMSolver`. Instead
of fitting a Gaussian per timestep, it maintains a **categorical distribution**
over the `Discrete(K)` action space and refits it from the empirical
frequencies of top-K elite trajectories. Sampling uses the Gumbel-max trick
(seeded via the solver's `torch.Generator`) and candidates are passed to
`model.get_cost` as one-hot tensors — the same layout used by `PGDSolver`, so
discrete world models work unchanged.

```python
import torch
import gymnasium as gym
from stable_worldmodel.solver import CategoricalCEMSolver
from stable_worldmodel.policy import PlanConfig


# 1. World model: cost defined over one-hot candidates

class DiscreteModel(torch.nn.Module):
    """Cost is minimized by selecting category 2 at every position."""

    def get_cost(self, info_dict, action_candidates):
        # action_candidates: (B, N, H, action_block * K) one-hot floats
        # returns:          (B, N)
        K = 4
        ab = action_candidates.shape[-1] // K
        c = action_candidates.reshape(*action_candidates.shape[:-1], ab, K)
        return -c[..., 2].sum(dim=(-1, -2))


# 2. Build and configure the solver

solver = CategoricalCEMSolver(
    model=DiscreteModel(),
    n_steps=20,        # CEM iterations
    num_samples=128,   # candidates per iteration
    topk=16,           # elite count
    smoothing=0.01,    # Laplace floor — prevents premature collapse
    alpha=0.1,         # EMA momentum on probs (0 = full overwrite)
    seed=0,
)

action_space = gym.spaces.Discrete(4)
config = PlanConfig(horizon=8, receding_horizon=4, action_block=1)
solver.configure(action_space=action_space, n_envs=2, config=config)


# 3. Solve

info_dict = {"obs": torch.zeros(2, 4)}
out = solver.solve(info_dict)

print(out["actions"].shape)     # (2, 8, 1)  — discrete indices, argmax of probs
print(out["probs"][0].shape)    # (2, 8, 1, 4) — final categorical distribution
print(out["costs"])             # mean elite cost per env
```

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `num_samples` | `300` | Candidate trajectories sampled per iteration |
| `n_steps` | `30` | CEM iterations |
| `topk` | `30` | Elite count for refit |
| `smoothing` | `0.0` | Laplace smoothing on refit probs (avoids collapse) |
| `alpha` | `0.0` | EMA momentum: `probs ← α · prev + (1−α) · new` |
| `batch_size` | `1` | Envs processed per outer batch |

### Output layout

| Key | Shape | Meaning |
|---|---|---|
| `actions` | `(n_envs, horizon, action_block)` | argmax of final probs (int64) |
| `probs` | `[(n_envs, horizon, action_block, K)]` | final categorical distribution |
| `costs` | `list[float]` of length `n_envs` | mean elite cost on the last iteration |
| `callbacks` | `dict[str, list[list[Any]]]` | per-callback history (if any) |

### Choosing between `PGDSolver` and `CategoricalCEMSolver`

Both target `Discrete(K)` action spaces.

- **`PGDSolver`** does projected gradient descent on simplex-valued action
  variables. Requires a **differentiable** `model.get_cost` and benefits from
  smooth cost landscapes.
- **`CategoricalCEMSolver`** is **gradient-free**. Use when the cost is
  non-differentiable (discrete simulators, ranking losses, learned classifiers
  used as oracles) or when PGD gets stuck in local minima.
