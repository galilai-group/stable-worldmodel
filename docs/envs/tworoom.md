---
title: Two-Room
summary: A 2D navigation task through doorways
external_links:
    arxiv: https://arxiv.org/abs/2411.04983
---

![pusht](../assets/tworoom.gif)

## Description

A 2D navigation task where a circular agent must reach a target position in another room by navigating through doorways. The environment uses PyTorch-based rendering and collision detection with a wall dividing the space into two rooms connected by one or more doors.

The agent starts in one room and must navigate to the target in the other room. The task requires planning a path through the door openings rather than simple point-to-point navigation.

**Success criteria**: The episode terminates when the agent is within a distance threshold of the target (4.5 × scale factor).

```python
import stable_worldmodel as swm
world = swm.World('swm/TwoRoom-v1', num_envs=4, image_shape=(128, 128))
```

## Environment Specs

| Property | Value |
|----------|-------|
| Action Space | `Box(-1, 1, shape=(2,))` — 2D velocity direction |
| Observation Space | `Box(0, img_size, shape=(10,))` — state vector |
| Reward | 0 (sparse) |
| Episode Length | Until target reached or timeout |
| Render Size | 224×224 (configurable) |
| Physics | Torch-based, 10 Hz control |

### Observation Details

The observation is a flat state vector of shape `(10,)`:

| Index | Description |
|-------|-------------|
| 0-1 | Agent position (x, y) |
| 2-3 | Target position (x, y) |
| 4-9 | Door center positions (up to 3 doors × 2 coords) |

### Info Dictionary

The `info` dict returned by `step()` and `reset()` contains:

| Key | Description |
|-----|-------------|
| `target` | Target image (3, H, W) — agent rendered at target position |
| `pos_agent` | Current agent position as numpy array |
| `pos_target` | Target position as numpy array |
| `target_pos` | Target position from variation space |
| `state` | Agent position tensor |
| `distance_to_target` | Euclidean distance to target |

## Variation Space

![tworoom_fov](../assets/tworoom_fov.gif)


The environment supports extensive customization through the variation space:

| Factor | Type | Description |
|--------|------|-------------|
| `agent.color` | RGBBox | Agent color (default: red) |
| `agent.radius` | Box | Agent radius in pixels (1-2× dot_std) |
| `agent.position` | Box | Starting position (left room by default) |
| `agent.speed` | Box | Movement speed (0.5-3.0× scale) |
| `target.color` | RGBBox | Target color (default: green) |
| `target.radius` | Box | Target radius in pixels |
| `target.position` | Box | Target position (right room by default) |
| `wall.color` | RGBBox | Wall color (default: black) |
| `wall.thickness` | Discrete | Wall thickness (2-10× scale) |
| `wall.axis` | Discrete(2) | 0: horizontal, 1: vertical |
| `wall.border_color` | RGBBox | Border color (default: black) |
| `door.color` | RGBBox | Door color (default: white) |
| `door.number` | Discrete(1, 3) | Number of doors |
| `door.size` | MultiDiscrete | Half-extent size of each door |
| `door.position` | MultiDiscrete | Center position of each door along wall |
| `background.color` | RGBBox | Background color (default: white) |
| `rendering.render_target` | Discrete(2) | Whether to render target (0: no, 1: yes) |

### Default Variations

By default, these factors are randomized at each reset:

- `agent.position`
- `target.position`

To randomize additional factors:

```python
# Randomize colors for domain randomization
world.reset(options={'variation': ['agent.color', 'target.color', 'background.color']})

# Randomize everything
world.reset(options={'variation': ['all']})
```

## Datasets

| Name | Episodes | Policy | Download |
|------|----------|--------|----------|
| `tworoom_expert` | 1000 | Weak Expert | — |

## Expert Policy

This environment includes a built-in weak expert policy for data collection:

```python
from stable_worldmodel.envs.two_room import ExpertPolicy

policy = ExpertPolicy()
world.set_policy(policy)
```
