title: Adding New Environment
summary: Register a Gymnasium environment with pixels, state, goals, and factors of variation.
sidebar_title: New Environment
---

This tutorial shows how to add a custom environment to
`stable_worldmodel`. The library builds on Gymnasium, so the environment
itself is an ordinary `gymnasium.Env`. The extra requirements are:

- `render()` must return an RGB array when `World(add_pixels=True)` is used.
- Useful data should be returned in `info` (`state`, `goal`, `goal_state`,
  task metrics, and so on).
- Optional factors of variation should be exposed as `variation_space`.

## Create a minimal environment

Save this as `line_reach_env.py`:

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_worldmodel import spaces as swm_spaces


class LineReachEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}

    def __init__(self, render_mode='rgb_array'):
        self.render_mode = render_mode
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.variation_space = swm_spaces.Dict(
            {
                'agent': swm_spaces.Dict(
                    {
                        'start_position': swm_spaces.Box(
                            low=np.array([-0.9], dtype=np.float32),
                            high=np.array([0.9], dtype=np.float32),
                            init_value=np.array([0.0], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        'color': swm_spaces.RGBBox(
                            init_value=np.array(
                                [40, 120, 255],
                                dtype=np.uint8,
                            ),
                        ),
                    }
                ),
                'goal': swm_spaces.Dict(
                    {
                        'position': swm_spaces.Box(
                            low=np.array([-0.9], dtype=np.float32),
                            high=np.array([0.9], dtype=np.float32),
                            init_value=np.array([0.75], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        'color': swm_spaces.RGBBox(
                            init_value=np.array(
                                [255, 80, 80],
                                dtype=np.uint8,
                            ),
                        ),
                    }
                ),
            }
        )

        self.position = 0.0
        self.goal = 0.75

    def _obs(self):
        return np.array([self.position, self.goal], dtype=np.float32)

    def _info(self):
        return {
            'state': self._obs(),
            'goal_state': np.array([self.goal], dtype=np.float32),
            'goal': self._render(show_agent=False),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        swm_spaces.reset_variation_space(
            self.variation_space,
            seed=seed,
            options=options,
            default_variations=(
                'agent.start_position',
                'goal.position',
            ),
        )
        self.position = float(
            self.variation_space['agent']['start_position'].value[0]
        )
        self.goal = float(self.variation_space['goal']['position'].value[0])
        return self._obs(), self._info()

    def step(self, action):
        action = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        self.position = float(np.clip(self.position + 0.05 * action, -1.0, 1.0))

        distance = abs(self.position - self.goal)
        terminated = distance < 0.03
        truncated = False
        reward = 1.0 if terminated else -distance

        return self._obs(), reward, terminated, truncated, self._info()

    def render(self):
        return self._render(show_agent=True)

    def _render(self, show_agent=True):
        canvas = np.full((64, 64, 3), 245, dtype=np.uint8)
        canvas[31:33, 4:60] = 30

        def xcoord(value):
            return int(np.interp(value, [-1.0, 1.0], [4, 59]))

        gx = xcoord(self.goal)
        goal_color = self.variation_space['goal']['color'].value
        canvas[20:44, max(0, gx - 1) : min(64, gx + 2)] = goal_color

        if show_agent:
            ax = xcoord(self.position)
            agent_color = self.variation_space['agent']['color'].value
            canvas[26:38, max(0, ax - 3) : min(64, ax + 4)] = agent_color

        return canvas
```

The environment is still a normal Gymnasium environment. You can test it
directly:

```python
env = LineReachEnv()
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
frame = env.render()
print(obs, reward, frame.shape)
```

## Register it

For a one-file experiment, register the class in the same process before you
construct `World`:

```python
import stable_worldmodel as swm
from stable_worldmodel.envs import register

from line_reach_env import LineReachEnv


register(
    id='swm/LineReach-v0',
    entry_point=LineReachEnv,
)

world = swm.World(
    'swm/LineReach-v0',
    num_envs=4,
    image_shape=(64, 64),
    max_episode_steps=50,
)
```

For a reusable package, put the registration in your package import path and
use a string entry point:

```python
from stable_worldmodel.envs import register

register(
    id='swm/LineReach-v0',
    entry_point='my_project.envs.line_reach:LineReachEnv',
)
```

Make sure the module that calls `register(...)` is imported before creating
the environment. The `swm envs` CLI can only list custom environments that
are registered during CLI startup, so package-level registration is preferable
for shared environments.

## Use it through World

```python
import stable_worldmodel as swm
from stable_worldmodel.policy import RandomPolicy

world = swm.World(
    'swm/LineReach-v0',
    num_envs=8,
    image_shape=(64, 64),
    max_episode_steps=50,
)

world.set_policy(RandomPolicy(seed=0))
world.reset(
    seed=0,
    options={
        'variation': [
            'agent.start_position',
            'goal.position',
            'agent.color',
        ],
    },
)

print(world.infos['pixels'].shape)  # (8, 1, 64, 64, 3)
print(world.infos['state'].shape)   # (8, 1, 2)
print(world.envs.single_variation_space.to_str())
world.close()
```

`World` wraps the raw environment with `MegaWrapper`. That wrapper renders
pixels, resizes them, moves observations and step metadata into `info`, and
stacks values across the vectorized environments.

## Collect a dataset

```python
from pathlib import Path
import os

import stable_worldmodel as swm


root = Path(os.environ.get('STABLEWM_HOME', Path.home() / '.stable_worldmodel'))
dataset_path = root / 'datasets' / 'line_reach_random.lance'

world = swm.World(
    'swm/LineReach-v0',
    num_envs=8,
    image_shape=(64, 64),
    max_episode_steps=50,
)
world.set_policy(swm.policy.RandomPolicy(seed=0))

world.collect(
    dataset_path,
    episodes=100,
    seed=0,
    options={'variation': ['all']},
)
world.close()
```

Now inspect and load it like any built-in dataset:

```bash
swm inspect line_reach_random
```

```python
dataset = swm.data.load_dataset(
    'line_reach_random.lance',
    num_steps=8,
    keys_to_load=['pixels', 'action', 'state', 'goal_state'],
)
```

## State-only environments

If your environment has no meaningful renderer, skip pixel rendering:

```python
world = swm.World(
    'swm/LineReach-v0',
    num_envs=8,
    add_pixels=False,
)
```

In that mode `pixels` is not added, video recording is unavailable, and the
raw observation is lifted into `world.infos['observation']` unless your env
already returns a dict observation.

## Environment checklist

- `reset()` calls `super().reset(seed=seed)` so Gymnasium seeds
  `self.np_random`.
- `action_space` and `observation_space` have stable shapes and dtypes.
- `render()` returns `uint8` RGB with shape `(H, W, 3)`.
- `info` contains every signal you want to record or train on.
- Goal-conditioned tasks include both a visual goal (`goal`) and a compact
  goal signal (`goal_state`) when possible.
- Variation spaces use `stable_worldmodel.spaces` and are reset with
  `reset_variation_space(...)`.
- Exact variation values are written to datasets only for watched variation
  keys passed in `options['variation']`.
