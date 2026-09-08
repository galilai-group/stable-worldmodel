---
title: Wire Harness
summary: Single-mover sequential target reaching on a planar table (MuJoCo)
---

## Description

A planar manipulation task: one mover (a platform sliding in `x`/`y`) must
visit a sequence of five target positions on a 6.72 m × 3.84 m table. Targets
are reached one at a time — the episode advances to the next target as soon as
the mover is inside the goal radius, and terminates once the whole sequence is
done.

The environment simulates the mover in MuJoCo (0.25 ms physics timestep, 60 Hz
control) and exposes the world-model signals `stable_worldmodel` expects:
`state` / `goal_state` vectors and a rendered `goal` image in every info dict,
plus `_set_state` / `_set_goal_state` hooks so `World.evaluate` can replay
arbitrary dataset starts and goals.

**Success criteria** — two regimes:

- *Rollout* (data collection, RL): each target counts when the mover is within
  `goal_radius` (0.1 m); the episode terminates after the last target.
- *Dataset-driven evaluation*: `_set_goal_state` installs a single goal; the
  episode terminates (= success) when the mover reaches it inside the budget.

```python
import stable_worldmodel as swm
world = swm.World('swm/WireHarness-v0', num_envs=4, image_shape=(176, 320),
                  camera_view='top', add_pixels=True)
```

## Environment Specs

| Property | Value |
|----------|-------|
| Action Space | `Box(-1, 1, shape=(2,))` — mover velocity `[vx, vy]`, scaled by `vel` |
| Observation Space | `Box(-inf, inf, shape=(4,))` — `[dx, dy, dist_norm, angle_norm]` to the active target |
| Reward | Dense: `10 · (prev_dist − dist) − 0.01`, `+10` per target reached |
| Episode Length | 3 600 steps (60 s × 60 Hz) or until all targets are reached |
| Render Size | 640×352 (offscreen MuJoCo renderer; `camera_view='top'` or `'oblique'`) |
| Physics | MuJoCo, 0.25 ms timestep, 60 Hz control |
| Targets | 5 positions, visited in sequence |

### Fixed Geometry Constants

All static configuration lives as module constants in
`stable_worldmodel/envs/wire_harness/env_basic.py` (no separate config file):

| Constant | Value | Description |
|----------|-------|-------------|
| `X_MIN..X_MAX` | 0 – 6.72 m | Table width |
| `Y_MIN..Y_MAX` | 0 – 3.84 m | Table height |
| `goal_radius` | 0.1 m | Success radius for the active target |
| `vel` | 2.0 | Joint velocity factor |
| `simend` | 60 s | Episode wall-clock cap |
| `MOVER_START` | `[4.0, 3.0]` | Default start position |
| `DEFAULT_TARGETS` | 5 × `(x, y)` | Default target sequence |

Actions are masked at the table boundary: a velocity component pointing out of
the workspace is zeroed before it reaches the physics.

### Observation Details

The observation is relative to the **currently active** target, so a single
policy generalizes across the sequence:

| Index | Description |
|-------|-------------|
| 0 | `dx` — signed x-offset to the active target |
| 1 | `dy` — signed y-offset to the active target |
| 2 | `dist_norm` — normalized distance to the active target |
| 3 | `angle_norm` — normalized bearing to the active target |

### Reward

| Term | Value |
|------|-------|
| Step cost | −0.01 |
| Progress | +10 · (distance reduction toward the active target) |
| Target reached | +10 (advances to the next target; terminates after the last) |

### Info Dictionary

The `info` dict returned by `step()` and `reset()` contains:

| Key | Description |
|-----|-------------|
| `state` | Mover position `(x, y)`, shape `(2,)` |
| `goal_state` | Active target `(x, y)`, shape `(2,)` — or the eval goal, if one is set |
| `goal` | Goal image (mover rendered at the active target) — when rendering is on |
| `sim_step` | Control steps taken this episode (`step()` only) |
| `dist_to_target` | Distance to the active target (`step()` only) |
| `current_target_idx` | Index of the active target (`step()` only) |
| `targets_reached` | Number of targets completed (`step()` only) |

## Variation Space

| Factor | Type | Description |
|--------|------|-------------|
| `mover.start_position` | Box([0, 0], [6.72, 3.84]) | Episode start position |
| `mover.target_positions` | Box, shape (5, 2) | The full target sequence |

Both are resampled on every `reset()` unless the caller passes
`options={'variation': ...}` or `options={'variation_values': ...}`. The
variation space is the single source of truth: `reset()` writes it, reads it
back, and teleports the physics to match — so `info['state']` always equals the
sampled `start_position`.

Dataset-driven evaluation bypasses sampling and injects arbitrary starts/goals
via `_set_state` / `_set_goal_state` (the callables in
`scripts/plan/eval_wm.py`). `_set_goal_state` is transient — the next `reset()`
restores the sequential targets.

## Datasets

| Name | Policy | Contents |
|------|--------|----------|
| `wireharness_expert` | Single-mover SAC expert | 1 000 rollouts, 176×320 top-view frames |

```bash
# config: scripts/data/config/wireharness.yaml
python scripts/data/collect_wireharness.py num_traj=1000
```

The 176×320 render shape keeps the native 640×352 aspect ratio (no squashing);
`scripts/plan/eval_wm.py` reads it back from `world_image_shape` so evaluation
frames match the training distribution.

## Expert Policy

A SAC expert (stable-baselines3, VecNormalize observation statistics) loaded
from an explicit checkpoint + normalizer path:

```python
from stable_worldmodel.envs.wire_harness import ExpertPolicy

policy = ExpertPolicy(
    ckpt_path='~/.stable_worldmodel/checkpoints/wire_harness/best_one_mover_sac/best_model.zip',
    vec_normalize_path='~/.stable_worldmodel/checkpoints/wire_harness/best_one_mover_sac/vec_normalize.pkl',
    device='cuda',
)
world.set_policy(policy)
```

## Benchmark

Dataset-driven closed-loop evaluation (`scripts/plan/eval_wm.py`, config
`scripts/plan/config/wireharness_basic.yaml`): starts and goals are replayed
from `wireharness_expert.lance`, CEM solver (300 samples, horizon 10, receding
horizon 5, action block 5).

Identical setup for both policies — **50 episodes, goal offset 75 steps,
400-step budget, seed 42**:

| Policy | Successes | Success Rate |
|--------|-----------|--------------|
| Random | 8 / 50 | 16 % |
| LeWM (epoch 27) | 34 / 50 | **68 %** |

### Random-policy budget sweep

The random baseline scales with how far ahead the goal sits, which sets the
difficulty floor for the table above:

| Episodes | Goal offset | Budget | Success rate |
|----------|-------------|--------|--------------|
| 1 | 5 | 100 | 100 % |
| 5 | 25 | 200 | 40 % |
| 5 | 75 | 200 | 0 % |
| 50 | 75 | 400 | 16 % |

### Longer-horizon LeWM evaluation

LeWM epoch 27 was also evaluated with a longer goal offset and execution
budget. The run completed on **50 episodes**, with **goal offset 250 steps**
and a **1,000-step evaluation budget**:

| Policy | Successes | Failures | Success Rate |
|--------|-----------|----------|--------------|
| LeWM (epoch 27) | 19 / 50 | 31 / 50 | 38 % |

This is a harder setting than the 75-step benchmark above, so the results
should not be compared directly unless the random baseline is run with the
same offset and budget.

```bash
# random baseline
python scripts/plan/eval_wm.py --config-name wireharness_basic \
    eval.num_eval=50 eval.goal_offset_steps=75 eval.eval_budget=400

# world model
python scripts/plan/eval_wm.py --config-name wireharness_basic \
    policy=<lewm_checkpoint> \
    eval.num_eval=50 eval.goal_offset_steps=75 eval.eval_budget=400
```
