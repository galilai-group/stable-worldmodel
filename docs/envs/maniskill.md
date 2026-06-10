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

!!! note "Assets download automatically on first use"
    The Panda cube tasks need no extra assets. The Bridge tasks pull scene + WidowX
    robot assets the first time you make them (public Hugging Face / GitHub downloads,
    no token). The wrapper sets `MS_SKIP_ASSET_DOWNLOAD_PROMPT=1` so this happens
    automatically — no separate command, and it works headless/non-interactively. To
    be prompted instead, set `MS_SKIP_ASSET_DOWNLOAD_PROMPT=0`. To pre-fetch
    explicitly: `python -m mani_skill.utils.download_asset bridge_v2_real2sim -y`.

    The Bridge tasks only support `obs_mode='rgb+segmentation'`, already set per-task
    in `TASK_SPECS`.

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

The env exposes a `variation_space` of visual Factors of Variation, aligned with SIMPLER's
distribution-shift axes. These are sampled each reset (or set explicitly via
`reset(options={'variation': [...]})` / `options={'variation_values': {...}}`) and recorded in
`info['variation.<key>']`. The set is restricted to factors **verified to change the rendered
frame** through the wrapper (applied by `_apply_variations` using ManiSkill's own SAPIEN APIs):

| Factor | Type | Default | SIMPLER axis |
|--------|------|---------|--------------|
| `light.intensity` | `Box(0.3, 1.0, (1,))` | `0.7` | lighting (scene ambient) |
| `camera.angle_delta` | `Box(-10, 10, (1, 2))` | `[[0, 0]]` (azimuth/elevation°) | camera poses |
| `object.color` | `Box(0, 1, (3,))` | `[0.8, 0.1, 0.1]` | object appearance |
| `rendering.transparent_arm` | `Discrete(2)` | `0` | arm texture |

`DEFAULT_VARIATIONS` (resampled each reset) = `light.intensity`, `camera.angle_delta`,
`object.color`.

Object/robot pose is **not** a settable factor — ManiSkill already randomizes object placement per
episode (seed-driven). **Follow-ups:** table/background texture (those surfaces are textured, so
`set_base_color` is a no-op — needs texture swap or `BaseDigitalTwinEnv` greenscreen); distractors.

### Success rate

**Demo replay (env + success detection).** There's no trained policy bundled, so to confirm the env
and its success detection report real successes (not just 0% from a random policy), replay ManiSkill's
official demonstrations through the wrapper:

```bash
python -m mani_skill.utils.download_demo PickCube-v1
python scripts/examples/maniskill_demo_replay.py \
    --swm-id swm/MSPickCube-v0 --task PickCube-v1 --episodes 20
```

The script restores each demo's recorded initial `env_state` and replays its actions through the
wrapper, reporting `success_rate` (**100%** on PickCube). Reproduction uses the initial state, not the
seed — batched-GPU demos aren't seed-reproducible in a single env.

**World-model MPC.** A LeWM world model trained on a collected PickCube dataset and evaluated with
goal-conditioned MPC reached **2% (1/50)** goal-reaching success on `swm/MSPickCube-v0` (random-policy
data, 30 epochs, goal threshold 2.0 on the flat sim state) — a real but low baseline. The full path:

```bash
python scripts/data/collect_maniskill.py                  # 1. collect a dataset
python scripts/train/lewm.py data=maniskill \             # 2. train LeWM
    output_model_name=lewm_mspickcube wandb.enabled=false
python scripts/plan/eval_wm.py --config-name maniskill    # 3. goal-conditioned MPC eval
```

Step 3 does not run on raw `collect_maniskill.py` output: `scripts/plan/config/maniskill.yaml` is a
**template** you point at a prepared dataset in the benchmark layout (`episode_idx`/`step_idx`/`ep_len`/
`ep_offset` + `pixels` HWC + `action`/`proprio`/`state` = flat sim state) — the same format as PushT's
provided `pusht_expert_train.h5`, supplied separately rather than read straight from `World.collect`.
The checkpoint's `config.json` must also be the model block (`save_pretrained(config_key='model')`;
`lewm.py` otherwise saves the full cfg).

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
