---
title: Meta-World
summary: Sawyer robot manipulation tasks (MT10) with domain randomization
sidebar_title: Meta-World
---

## Description

[Meta-World](https://metaworld.farama.org/) is a benchmark of robotic
manipulation tasks built on a simulated Sawyer arm. stable-worldmodel wraps the
ten tasks of the canonical **MT10** suite and adds a variation space on top of
each one, so the same task can be rendered and simulated under controlled
changes to colors, lighting, object physics, and target placement. This makes
the suite useful for out-of-distribution and zero-shot robustness evaluation of
world models.

Each task uses the same 4-DoF end-effector control (three Cartesian deltas plus
a gripper command) and the same Sawyer arm, so a single policy or world model
can be evaluated across all of them.

!!! note "Installation"
    Meta-World requires Python 3.10 or 3.11. It is installed with the `env`
    extra on supported interpreters:

        :::bash
        pip install 'stable-worldmodel[env]'

    On Python 3.12 the dependency is skipped, and creating a Meta-World
    environment raises an informative `ImportError`.

## Tasks

| Environment id | Meta-World task |
|---|---|
| `swm/MetaWorldReach-v0` | `reach-v3` |
| `swm/MetaWorldPush-v0` | `push-v3` |
| `swm/MetaWorldPickPlace-v0` | `pick-place-v3` |
| `swm/MetaWorldDoorOpen-v0` | `door-open-v3` |
| `swm/MetaWorldDrawerOpen-v0` | `drawer-open-v3` |
| `swm/MetaWorldDrawerClose-v0` | `drawer-close-v3` |
| `swm/MetaWorldButtonPressTopdown-v0` | `button-press-topdown-v3` |
| `swm/MetaWorldPegInsertSide-v0` | `peg-insert-side-v3` |
| `swm/MetaWorldWindowOpen-v0` | `window-open-v3` |
| `swm/MetaWorldWindowClose-v0` | `window-close-v3` |

```python
import stable_worldmodel as swm

world = swm.World('swm/MetaWorldPush-v0', num_envs=4, image_shape=(224, 224))
```

## Environment Specs

| Property | Value |
|----------|-------|
| Action Space | `Box(-1, 1, shape=(4,))`: end-effector xyz deltas and gripper |
| Observation Space | `Box(shape=(39,))`: current and previous proprioceptive and object state, with the goal in the last 3 entries |
| Reward | Dense Meta-World shaped reward |
| Episode Length | 500 steps |
| Render Size | Configurable via `image_shape` (square) |

### Info Dictionary

Every step exposes the standard stable-worldmodel keys alongside Meta-World's
own task metrics:

- `state`: the full 39-d observation vector.
- `proprio`: the hand position and gripper opening (first 4 entries).
- `goal_state`: the task target position (`_target_pos`).
- `success`: `1.0` once the task's success criterion is met, else `0.0`.
- `env_name`: the underlying Meta-World task id.
- Meta-World extras such as `near_object`, `grasp_success`, and
  `obj_to_target`.

## Variation Space

The variation space is assembled per task from the contents of that task's
MuJoCo model, so each environment advertises only the factors it can honor.

| Group | Key | Available on |
|-------|-----|--------------|
| `table` | `color` | all tasks |
| `background` | `color` | all tasks |
| `light` | `intensity` | all tasks |
| `rendering` | `transparent_arm` | all tasks |
| `object` | `color` | tasks with a recolorable task object |
| `object` | `mass`, `friction`, `scale`, `position` | tasks with a free object (reach, push, pick-place) |
| `goal` | `position` | tasks with a target site |

Inspect the exact space for a given task from the CLI:

```bash
swm fovs swm/MetaWorldPush-v0
```

### Default Variations

By default, resets randomize the visual factors only, which keeps the task that
Meta-World samples solvable:

```python
# Visual domain randomization (the default set)
world.reset(seed=0, options={'variation': [
    'table.color', 'object.color', 'background.color', 'light.intensity',
]})

# Randomize everything the task supports, including physics and target
world.reset(seed=0, options={'variation': ['all']})

# Set exact values, for reproducible OOD configurations
import numpy as np
world.reset(seed=0, options={
    'variation': ['object.color', 'object.mass'],
    'variation_values': {
        'object.color': np.array([0.1, 0.6, 0.9]),
        'object.mass': np.array([3.0]),
    },
})
```

## Datasets

Collect demonstrations or random-policy rollouts like any other environment:

```python
world.set_policy(swm.policy.RandomPolicy(seed=0))
world.collect('data/metaworld_push.lance', episodes=100, seed=0,
              options={'variation': ['all']})
```

## References

- Yu et al., *Meta-World: A Benchmark and Evaluation for Multi-Task and Meta
  Reinforcement Learning*, CoRL 2019.
- [Meta-World documentation](https://metaworld.farama.org/)
