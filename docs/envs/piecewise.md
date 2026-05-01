---
title: Piecewise (Piecewise)
summary: A 2D navigation task with piecewise zone-dependent dynamics
---

## Description

A 2D navigation task where a circular agent must reach a target position in a single open room bounded by 4 border walls. The room is virtually divided into a `grid_n × grid_n` grid of zones, each applying an additive bias vector to the agent's motion.

This piecewise dynamics design makes the environment challenging for world models: the agent must learn that the same action produces different outcomes depending on its current zone.

**Success criteria**: The episode terminates when the agent is within 16 pixels of the target.

```python
import stable_worldmodel as swm
world = swm.World('swm/Piecewise-v0', num_envs=4, image_shape=(224, 224), grid_n=2)
```

## Environment Specs

| Property | Value |
|----------|-------|
| Action Space | `Box(-1, 1, shape=(2,))` — 2D velocity direction |
| Observation Space | `Box(0, 224, shape=(4,))` — state vector |
| Reward | 0 (sparse) |
| Episode Length | Until target reached or timeout |
| Render Size | 224×224 (fixed) |
| Physics | Torch-based, 10 Hz control |

### Fixed Geometry Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `IMG_SIZE` | 224 | Image dimensions in pixels |
| `BORDER_SIZE` | 14 | Border/wall thickness in pixels |

### Motion Model

At each step, the agent position is updated as:

```
pos_next = pos + action * speed + bias[zone]
```

where `bias[zone]` is the additive bias vector for the zone the agent currently occupies.

### Observation Details

The observation is a flat state vector of shape `(4,)`:

| Index | Description |
|-------|-------------|
| 0-1 | Agent position (x, y) |
| 2-3 | Target position (x, y) |

### Info Dictionary

The `info` dict returned by `step()` and `reset()` contains:

| Key | Description |
|-----|-------------|
| `env_name` | `'Piecewise'` |
| `proprio` | Agent position as numpy array |
| `state` | Agent position as numpy array |
| `goal_state` | Target position as numpy array |
| `distance_to_target` | Euclidean distance to target |

## Variation Space

The environment supports extensive customization through the variation space:

| Factor | Type | Description |
|--------|------|-------------|
| `agent.color` | RGBBox | Agent color (default: red) |
| `agent.radius` | Box(7, 14) | Agent radius in pixels |
| `agent.position` | Box | Starting position |
| `agent.speed` | Box(1.75, 10.5) | Movement speed in pixels/step |
| `target.color` | RGBBox | Target color (default: green) |
| `target.radius` | Box(7, 14) | Target radius in pixels |
| `target.position` | Box | Target position |
| `background.color` | RGBBox | Background color (default: white) |
| `border.color` | RGBBox | Border/wall color (default: black) |
| `zones.bias_i` | Box(-5, 5, shape=(2,)) | Additive bias vector for zone `i` |
| `zones.color_i` | RGBBox | Background color for zone `i` (pastel, evenly spaced hues) |
| `rendering.render_target` | Discrete(2) | Whether to render the target dot (0: no, 1: yes) |
| `rendering.render_zones` | Discrete(2) | Whether to color zones (0: no, 1: yes) |
| `rendering.render_bias_field` | Discrete(2) | Whether to overlay bias vector field (0: no, 1: yes) |
| `task.min_steps` | Discrete(15, 100) | Minimum steps required to reach target |

Zone indices follow row-major order: zone `i = row * grid_n + col`.

### Default Variations

By default, these factors are randomized at each reset:

- `agent.position`
- `target.position`

To randomize additional factors:

```python
# Randomize zone biases for piecewise dynamics diversity
world.reset(options={'variation': ['zones.bias_0', 'zones.bias_1', 'zones.bias_2', 'zones.bias_3']})

# Randomize everything
world.reset(options={'variation': ['all']})
```

## Expert Policy

This environment includes a built-in analytical expert policy that inverts the motion equation to go towards the goal:

```python
from stable_worldmodel.envs.piecewise.expert_policy import ExpertPolicy

policy = ExpertPolicy(action_noise=0.0, action_repeat_prob=0.0)
world.set_policy(policy)
```

| Parameter | Description |
|-----------|-------------|
| `action_noise` | Std of Gaussian noise added to actions (default: 0.0) |
| `action_repeat_prob` | Probability of repeating the previous action (default: 0.0) |

## Data Collection

```bash
python scripts/data/collect_piecewise.py
python scripts/data/collect_piecewise.py --grid-n 3 --output /tmp/piecewise
python scripts/data/collect_piecewise.py --bias-scale 2.0 --horizon 400
python scripts/data/collect_piecewise.py --no-render-zones --no-render-bias
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--output` | `/tmp/piecewise_video` | Directory to save videos |
| `--grid-n` | `2` | Grid size (`grid_n × grid_n` zones) |
| `--horizon` | `300` | Max steps per episode |
| `--bias-scale` | `4.0` | Magnitude of per-zone bias vectors |
| `--image-size` | `224` | Output frame size in pixels |
| `--fps` | `15` | Video frame rate |
| `--seed` | `42` | Random seed |
| `--no-render-zones` | `False` | Disable zone background coloring |
| `--no-render-bias` | `False` | Disable bias vector-field overlay |