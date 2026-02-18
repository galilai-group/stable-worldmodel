---
title: OGBench
summary: 3D robotic manipulation tasks from OGBench
external_links:
    arxiv: https://arxiv.org/pdf/2410.20092
    github: https://github.com/seohongpark/ogbench
---

<!-- ![ogbench](assets/env/ogbench/cube.gif) -->

## Description

A suite of 3D robotic environments built on [OGBench](https://github.com/seohongpark/ogbench) and [MuJoCo](https://mujoco.org/). The environments feature a UR5e robotic arm with a Robotiq gripper performing various manipulation tasks including pick-and-place, stacking, and complex scene interactions. It also includes a suite of navigation environments.

Three environment families are available:

- **Cube**: Pure cube manipulation tasks with 1-8 colored cubes
- **Scene**: Complex scene with cubes, buttons, a drawer, and a window
- **PointMaze**: 2D maze navigation with point-mass, ant, or humanoid agents

```python
import stable_worldmodel as swm

# Cube environment - set env_type to: single, double, triple, quadruple, or octuple
world = swm.World('swm/OGBCube-v0', num_envs=4, env_type='single')

# Scene environment (cube + drawer + window + buttons)
world = swm.World('swm/OGBScene-v0', num_envs=4)

# PointMaze environment - point agent in a large maze, pixel observations
world = swm.World('swm/OGBPointMaze-v0', num_envs=4)
```

---

## Cube Environment

![cube](../assets/cube.gif)


Manipulation tasks involving 1-8 colored cubes that must be moved to target positions. Tasks include pick-and-place, swapping positions, and multi-cube stacking.

**Success criteria**: All cubes must be within 4cm of their target positions.

### Environment Specs

| Property | Value |
|----------|-------|
| Action Space | `Box(-1, 1, shape=(7,))` — 6 joint velocities + gripper |
| Observation Space | Pixels `(224, 224, 3)` or state vector |
| Reward | Number of cubes at target minus total cubes |
| Episode Length | 200 steps |
| Render Size | 224×224 (configurable) |
| Physics | MuJoCo |

### Environment Types

| `env_type` | Cubes |
|------------|-------|
| `single` | 1 |
| `double` | 2 |
| `triple` | 3 |
| `quadruple` | 4 |
| `octuple` | 8 |

### Info Dictionary

| Key | Description |
|-----|-------------|
| `goal` | Goal observation (state or pixels) |
| `success` | Whether all cubes are at targets |
| `proprio/*` | Robot proprioception (joint pos/vel, gripper state) |
| `privileged/block_{i}_pos` | Position of cube i |
| `privileged/block_{i}_quat` | Orientation of cube i |

### Variation Space


![cube_fov](../assets/cube_fov.gif)


| Factor | Type | Description |
|--------|------|-------------|
| `cube.color` | Box(0, 1, shape=(N, 3)) | RGB colors for each cube |
| `cube.size` | Box(0.01, 0.03, shape=(N,)) | Cube half-extents |
| `agent.color` | Box(0, 1, shape=(3,)) | Robot arm color |
| `floor.color` | Box(0, 1, shape=(2, 3)) | Checkerboard floor colors |
| `camera.angle_delta` | Box(-10, 10, shape=(1, 2)) | Camera yaw/pitch offsets |
| `light.intensity` | Box(0, 1, shape=(1,)) | Scene lighting intensity |

---

## Scene Environment

![scene](../assets/scene.gif)


A complex manipulation scene with a single cube, two lock buttons, a drawer, and a window. Tasks require interacting with multiple objects in sequence (e.g., press button to unlock drawer, then open drawer).

**Success criteria**: All objects must be in their target configurations (cube position, button states, drawer position, window position).

### Environment Specs

| Property | Value |
|----------|-------|
| Action Space | `Box(-1, 1, shape=(7,))` — 6 joint velocities + gripper |
| Observation Space | Pixels `(224, 224, 3)` or state vector |
| Episode Length | 200 steps |
| Environment ID | `swm/OGBScene-v0` |

### Scene Objects

| Object | Description |
|--------|-------------|
| Cube | Single cube to manipulate |
| Buttons (2) | Lock/unlock buttons for drawer and window |
| Drawer | Slidable drawer (locked by button 0) |
| Window | Slidable window (locked by button 1) |

### Tasks

1. **Open**: Open both drawer and window
2. **Unlock and Lock**: Toggle locks and close drawer/window
3. **Rearrange Medium**: Move cube and toggle some objects
4. **Put in Drawer**: Open drawer and place cube inside
5. **Rearrange Hard**: Complex multi-step rearrangement

### Variation Space

![scene_fov](../assets/scene_fov.gif)

| Factor | Type | Description |
|--------|------|-------------|
| `cube.color` | Box(0, 1, shape=(1, 3)) | Cube RGB color |
| `cube.size` | Box(0.01, 0.03, shape=(1,)) | Cube half-extents |
| `lock_color` | Box(0, 1, shape=(2, 3)) | Colors for locked buttons/handles |
| `agent.color` | Box(0, 1, shape=(3,)) | Robot arm color |
| `floor.color` | Box(0, 1, shape=(2, 3)) | Checkerboard floor colors |
| `camera.angle_delta` | Box(-5, 5, shape=(1, 2)) | Camera yaw/pitch offsets |
| `light.intensity` | Box(0, 1, shape=(1,)) | Scene lighting intensity |

---

## Multiview Mode

The Cube and Scene environments support rendering from multiple camera angles simultaneously:

```python
# Enable multiview rendering
world = swm.World('swm/OGBCube-v0', num_envs=4, env_type='double', multiview=True)

# Returns dict with 'front_pixels' and 'side_pixels' keys
obs = world.reset()
```

## Expert Policy

The Cube and Scene environments include a built-in expert policy for data collection:

```python
from stable_worldmodel.envs.ogbench import ExpertPolicy

policy = ExpertPolicy()
world.set_policy(policy)
```

---

## PointMaze Environment

2D maze navigation tasks where an agent must reach a goal position (or push a ball to a goal). Built on OGBench's `locomaze` suite, wrapping it with a `variation_space` for visual domain randomization.

**Success criteria**: Agent (or ball) is within a threshold distance of the goal position.

### Environment Specs

| Property | Value |
|----------|-------|
| Action Space | Depends on agent: `Box(-1, 1, shape=(2,))` for point, `Box(-1, 1, shape=(8,))` for ant |
| Observation Space | Pixels `(64, 64, 3)` or state vector |
| Render Size | 64×64 (configurable) |
| Environment ID | `swm/OGBPointMaze-v0` |
| Physics | MuJoCo |

### Configuration

```python
# Default: point agent, large maze, pixel observations
world = swm.World('swm/OGBPointMaze-v0', num_envs=4)

# Custom configuration
world = swm.World(
    'swm/OGBPointMaze-v0',
    num_envs=4,
    loco_env_type='point',   # 'point', 'ant', or 'humanoid'
    maze_env_type='maze',    # 'maze' (navigate) or 'ball' (push ball to goal)
    maze_type='large',       # 'arena', 'medium', 'large', 'giant', 'teleport'
    ob_type='pixels',        # 'pixels' or 'states'
)
```

### Agent Types

| `loco_env_type` | Description |
|-----------------|-------------|
| `point` | Simple 2D point-mass agent |
| `ant` | Quadruped ant |
| `humanoid` | Humanoid |

### Maze Layouts

| `maze_type` | Description |
|-------------|-------------|
| `arena` | Open arena with no walls |
| `medium` | Medium-sized maze |
| `large` | Large maze (default) |
| `giant` | Very large maze |
| `teleport` | Maze with teleportation portals |

### Maze Task Types

| `maze_env_type` | Description |
|-----------------|-------------|
| `maze` | Navigate agent to goal position |
| `ball` | Push a ball to the goal position |

### Variation Space

| Factor | Type | Description |
|--------|------|-------------|
| `agent.color` | Box(0, 1, shape=(3,)) | Agent sphere RGB color |
| `wall.color` | Box(0, 1, shape=(3,)) | Maze wall material RGB color |
| `floor.color` | Box(0, 1, shape=(2, 3)) | Checker floor texture colors (rgb1 / rgb2) |
| `floor.pixel_encoding` | Discrete(2) | Toggle OGBench's position-encoding floor gradient (pixels mode only) |
| `light.intensity` | Box(0, 1, shape=(1,)) | Global directional light diffuse intensity |

> **Note**: In `ob_type='pixels'` mode, `floor.color` only takes effect when `floor.pixel_encoding=0`. When enabled (`=1`), OGBench overwrites the floor with a position-encoding gradient (R ← row, G ← column, B = 128) to help the agent infer its location.

---

## Datasets

| Name | Episodes | Environment |
|------|----------|-------------|
| `ogbench_cube_single` | 1000 | Cube-Single |
| `ogbench_cube_double` | 1000 | Cube-Double |
| `ogbench_scene` | 1000 | Scene |
