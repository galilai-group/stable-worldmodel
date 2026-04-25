---
title: Gymnasium Robotics
summary: 3D contact-rich manipulation suites from Gymnasium Robotics.
external_links:
    arxiv: https://arxiv.org/abs/1802.09464
    github: https://github.com/Farama-Foundation/Gymnasium-Robotics
---

## Description

A collection of 3D contact-rich manipulation environments built using the [Gymnasium Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics) API. The environments use the [MuJoCo](https://mujoco.org/) physics engine to simulate diverse robotic kinematics, explicitly tracking continuous-control primitives like reaching, pushing, sliding, and pick-and-place dynamics.

```python
import stable_worldmodel as swm

# Reach environment (pure gripper kinematics)
world = swm.World('swm/FetchReach-v3', num_envs=4, image_shape=(224, 224))

# Contact-rich block manipulation environments
world = swm.World('swm/FetchPush-v3', num_envs=4, image_shape=(224, 224))
world = swm.World('swm/FetchSlide-v3', num_envs=4, image_shape=(224, 224))
world = swm.World('swm/FetchPickAndPlace-v3', num_envs=4, image_shape=(224, 224))
```

### Available Environments

| Environment | Environment ID | Task Objective |
|-------------|---------------|----------------|
| [Fetch Reach](#fetch-manipulation-suite) | `swm/FetchReach-v3` | Move the gripper to a coordinate in mid-air |
| [Fetch Push](#fetch-manipulation-suite) | `swm/FetchPush-v3` | Push a block to a stationary coordinate |
| [Fetch Slide](#fetch-manipulation-suite) | `swm/FetchSlide-v3` | Strike a block so it slides across a slippery table |
| [Fetch Pick And Place](#fetch-manipulation-suite) | `swm/FetchPickAndPlace-v3` | Grasp a block and lift it to a mid-air coordinate |

---

## Fetch Manipulation Suite

<div style="display: flex; gap: 10px; margin-bottom: 20px;">
  <img src="../../assets/fetch_push.gif" alt="fetch push" style="width: 24%; object-fit: contain;">
  <img src="../../assets/fetch_slide.gif" alt="fetch slide" style="width: 24%; object-fit: contain;">
  <img src="../../assets/fetch_pickandplace.gif" alt="fetch pick and place" style="width: 24%; object-fit: contain;">
  <img src="../../assets/fetch_reach.gif" alt="fetch reach" style="width: 24%; object-fit: contain;">
</div>

An agent controls a 7-DoF Fetch robotic arm. The agent manipulates explicit Cartesian coordinates to move the gripper and actuate the fingers to interact with the environment.

**Success criteria**: The gripper (in *Reach*) or the physical block (in *Push, Slide, Pick And Place*) must be moved within a short euclidean distance threshold of the goal state coordinates.

### Environment Specs

| Property | Value |
|----------|-------|
| Action Space | `Box(-1, 1, shape=(4,))` — 3D Cartesian velocity + Gripper control |
| Observation Space (Reach) | `Box(-inf, inf, shape=(13,))` — Flattened `observation` + `desired_goal` arrays |
| Observation Space (Push/Slide/Pick) | `Box(-inf, inf, shape=(28,))` — Flattened `observation` + `desired_goal` arrays |
| Reward | Standard sparse/dense task rewards depending on upstream configurations |
| Episode Length | 50 steps (default) |
| Render Size | Configurable via `resolution=224` on init |
| Physics | MuJoCo |

### Task Types

| `env_name` | Manipulated Objects | Task Dynamics |
|------------|---------------------|---------------|
| `FetchReach-v3` | None | Pure kinematic arm movement |
| `FetchPush-v3` | 1 Block | Surface contact physics |
| `FetchSlide-v3` | 1 Block | Momentum and low-friction sliding |
| `FetchPickAndPlace-v3` | 1 Block | Grasping and mid-air suspension |

### Info Dictionary

The `info` dict returned by `step()` and `reset()` is strictly formatted to standard SWM API integrations:

| Key | Shape (Push/Pick/Slide) | Shape (Reach) | Description |
|-----|-------------------------|---------------|-------------|
| `env_name` | `str` | `str` | The ID of the environment (e.g. `FetchPush-v4`) |
| `state` | `(28,)` | `(13,)` | Full flattened state array natively concatenating `observation` + `desired_goal` |
| `proprio` | `(25,)` | `(10,)` | Isolated agent internal state (gripper poses, joint velocities, object physics) |
| `goal_state` | `(3,)` | `(3,)` | The 3D Cartesian XYZ target coordinates for the objective |

### Variation Space

The environment supports extensive domain randomization across both visual textures and explicitly intercepting internal MuJoCo physics states.

| Factor | Type | Description |
|--------|------|-------------|
| `table.color` | RGBBox | Table surface color (default: `[0.3, 0.3, 0.3]`) |
| `object.color` | RGBBox | Manipulated block color (default: `[0.8, 0.1, 0.1]`) |
| `background.color` | RGBBox | Studio/Skybox background color (default: `[0.1, 0.1, 0.1]`) |
| `light.intensity` | Box(0.0, 1.0) | Diffuse scene lighting intensity (default: `0.7`) |
| `camera.angle_delta` | Box(-10.0, 10.0, shape=(1, 2)) | Azimuth/elevation camera perturbations (degrees) |
| `agent.start_position` | Box([1.25, 0.6], [1.45, 0.9]) | Starting 2D (x,y) coordinates for the initial gripper spawn targeting |
| `block.start_position` | Box([1.15, 0.6], [1.45, 0.9]) | Explicit 2D (x,y) override intercepting initial qpos spawning |
| `block.angle` | Box(-π, π) | Explicit Z-rotation override intercepting initial qpos quaternions |
| `block.mass` | Box(0.01, 50.0) | Mass of the manipulated block (kg); default inherits the MuJoCo model's `body_mass` |
| `goal.start_position` | Box([1.15, 0.6, 0.424], [1.45, 0.9, 0.424]) | Explicit XYZ override redefining visual and reward goal markers |
| `rendering.transparent_arm` | Discrete(2) | When set to `1`, natively lowers the PyMuJoCo alpha channels mapping to the robot body, making the robot translucent (see-through) |

> `block.start_position`, `block.angle`, and `block.mass` are only present in environments with a manipulated object (`FetchPush-v3`, `FetchSlide-v3`, `FetchPickAndPlace-v3`). `FetchReach-v3` exposes the remaining 8 factors.

#### Default Configuration

By default, the following visual factors are natively tracked and continuously randomized upon resetting if no variations are specified:

- `table.color`
- `object.color`
- `light.intensity`
- `background.color`
- `camera.angle_delta`

#### Overriding Physics and Randomization

To randomize physical spawn properties or enable fully deterministic data generation workflows, simply pass the bounding properties via the `options` array natively supported by the Gymnasium wrapper during environment reset:

```python
# Randomize environment visuals AND strictly dictate starting block positions
obs, info = world.reset(options={
    'variation': [
        'table.color', 'object.color', 'background.color', 
        'agent.start_position', 'block.start_position'
    ]
})

# Or bypass randomization altogether and strictly force perfect coordinate reproducibility:
obs, info = world.reset(options={
    'variation_values': {
        'block.start_position': [1.3, 0.7],
        'goal.start_position': [1.3, 0.7, 0.4247]
    }
})
```
