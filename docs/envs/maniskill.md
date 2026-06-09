---
title: ManiSkill3 (Franka + SIMPLER Bridge)
summary: GPU-parallel SAPIEN manipulation tasks — Franka Panda table-top + SIMPLER/real2sim Bridge (WidowX) digital twins.
external_links:
    arxiv: https://arxiv.org/abs/2410.00425
    github: https://github.com/haosulab/ManiSkill
    docs: https://maniskill.readthedocs.io/en/latest/tasks/index.html
---

## Description

Manipulation environments from [ManiSkill3](https://github.com/haosulab/ManiSkill) (built on the
[SAPIEN](https://sapien.ucsd.edu/) engine), wrapped as standard SWM environments. Two families are
registered, both stationary arms sharing one gym contract:

- **Franka Panda table-top manipulation** — the general ManiSkill3 suite (`swm/MS*`).
- **SIMPLER / real2sim Bridge digital twins** — the WidowX tasks from the
  [SIMPLER](https://simpler-env.github.io/) benchmark, ported into ManiSkill3 (`swm/Simpler*`).

Only the simulator/environment layer is integrated. SIMPLER's policy stack (RT-1/Octo/OpenVLA) and
its real-vs-sim metrics are **not** — `World.evaluate` evaluates policies via the task's native
success detector (mapped onto `terminated`).

!!! warning "GPU + Vulkan required"
    ManiSkill/SAPIEN need an **NVIDIA GPU with both CUDA and Vulkan** working (rendering uses
    Vulkan). A100/H100 work provided the box has the NVIDIA Vulkan driver installed — the common
    headless failure is CUDA-present-but-Vulkan-missing. Pre-flight check:
    `nvidia-smi` and `vulkaninfo | head` (must list the NVIDIA GPU).

## Installation

```bash
pip install "stable-worldmodel[maniskill]"   # or: uv sync --extra maniskill
```

The dependency is GPU-only and is intentionally **not** part of `[all]`. The import is lazy, so
`import stable_worldmodel` works (and the envs register) on machines without ManiSkill installed —
they only fail when you actually instantiate one.

```python
import stable_worldmodel as swm

# Franka Panda manipulation
world = swm.World('swm/MSPickCube-v0', num_envs=4, image_shape=(224, 224))

# SIMPLER Bridge (WidowX) digital twin
world = swm.World('swm/SimplerCarrotOnPlate-v0', num_envs=4, image_shape=(224, 224))
```

!!! note "First run downloads assets"
    The Panda cube tasks need no extra assets. The Bridge tasks pull scene + WidowX
    robot assets on first use (public Hugging Face / GitHub downloads, no token):
    ```bash
    uv run python -m mani_skill.utils.download_asset bridge_v2_real2sim -y
    ```
    The WidowX robot URDF downloads on first `gym.make` (auto-prompts; pre-fetch by
    answering `y`). The Bridge tasks only support `obs_mode='rgb+segmentation'`, which
    is already set per-task in `TASK_SPECS`.

### Available Environments

**Franka Panda table-top manipulation**

| Environment ID | ManiSkill Task | Task Objective |
|----------------|----------------|----------------|
| `swm/MSPickCube-v0` | `PickCube-v1` | Grasp a cube and lift it to a target height |
| `swm/MSPushCube-v0` | `PushCube-v1` | Push a cube to a goal region |
| `swm/MSPullCube-v0` | `PullCube-v1` | Pull a cube to a goal region |
| `swm/MSPokeCube-v0` | `PokeCube-v1` | Poke a cube with a tool |
| `swm/MSStackCube-v0` | `StackCube-v1` | Stack one cube on another |
| `swm/MSLiftPegUpright-v0` | `LiftPegUpright-v1` | Lift a peg to an upright pose |
| `swm/MSPegInsertionSide-v0` | `PegInsertionSide-v1` | Insert a peg into a side slot |
| `swm/MSPlugCharger-v0` | `PlugCharger-v1` | Plug a charger into a socket |
| `swm/MSPickSingleYCB-v0` | `PickSingleYCB-v1` | Grasp a randomized YCB object |
| `swm/MSRollBall-v0` | `RollBall-v1` | Roll a ball to a goal |

**SIMPLER / real2sim Bridge digital twins (WidowX)**

| Environment ID | ManiSkill Task | Task Objective |
|----------------|----------------|----------------|
| `swm/SimplerCarrotOnPlate-v0` | `PutCarrotOnPlateInScene-v1` | Put the carrot on the plate |
| `swm/SimplerSpoonOnTowel-v0` | `PutSpoonOnTableClothInScene-v1` | Put the spoon on the towel |
| `swm/SimplerStackCube-v0` | `StackGreenCubeOnYellowCubeBakedTexInScene-v1` | Stack the green cube on the yellow cube |
| `swm/SimplerEggplantInBasket-v0` | `PutEggplantInBasketScene-v1` | Put the eggplant in the basket |

---

## Environment Specs

| Property | Value |
|----------|-------|
| Action Space | Per task, read from the underlying env. Default control mode is task-native; pass `control_mode='pd_ee_delta_pose'` for a uniform 7-D `[Δxyz, Δrot, gripper]` action. |
| Observation Space | `Box(-inf, inf, shape=(proprio_dim,))` — flattened agent proprioception |
| Reward | Native ManiSkill task reward |
| Success | Native ManiSkill success detector, mapped onto `terminated` |
| Render Size | Configurable via `resolution=224` on init |
| Physics / Render | SAPIEN (CUDA + Vulkan) |

### Info Dictionary

The `info` dict returned by `reset()` and `step()` follows the standard SWM convention:

| Key | Type | Description |
|-----|------|-------------|
| `env_name` | `str` | The ManiSkill task id (e.g. `PickCube-v1`) |
| `proprio` | `(proprio_dim,)` | Flattened agent state (`obs['agent']`: qpos/qvel/...) |
| `state` | `(state_dim,)` | `proprio` plus task `extra` (tcp/object poses) where present |
| `success` | `bool` | Native task success at this step |
| `instruction` | `str` | Language instruction (Bridge tasks; absent otherwise) |

### Variation Space

The variation space is currently empty (`swm_spaces.Dict({})`). SIMPLER's distribution-shift knobs
(lighting, camera, background, distractors, textures) are a planned follow-up.

## Adding a robot or task

Environments are declared in `stable_worldmodel/envs/maniskill/tasks.py` as `TASK_SPECS` — a flat
list where each entry maps a `swm/...` id to a ManiSkill `task_id` plus optional overrides. The
wrapper is robot-agnostic, so **adding a robot or task is a one-line entry, no code changes**:

```python
TASK_SPECS = [
    # New task on its default robot
    dict(id='swm/MSPushCube-v0', task_id='PushCube-v1'),

    # Same task on a different embodiment (ManiSkill swaps via robot_uids;
    # the wrapper picks up the new action_space automatically)
    dict(id='swm/MSPickCubeWidowX-v0', task_id='PickCube-v1', robot_uids='widowx'),

    # Robot-specific task variant
    dict(id='swm/MSPickCubeSO100-v0', task_id='PickCubeSO100-v1'),

    # Uniform 7-D end-effector action across arms (for a shared policy)
    dict(id='swm/MSPickCubeEE-v0', task_id='PickCube-v1', control_mode='pd_ee_delta_pose'),
]
```

Any key besides `id` is passed to `ManiSkillWrapper` (`robot_uids`, `control_mode`, `camera_name`,
`obs_mode`, ...).
