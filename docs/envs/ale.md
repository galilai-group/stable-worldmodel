---
title: ALE (Atari)
summary: The Arcade Learning Environment — 100+ classic Atari 2600 games
external_links:
    docs: https://ale.farama.org/
    github: https://github.com/Farama-Foundation/Arcade-Learning-Environment
---

## Description

The [Arcade Learning Environment](https://ale.farama.org/) (ALE) wraps the Stella Atari 2600 emulator and exposes 100+ games (Breakout, Pong, Space Invaders, Ms. Pac-Man, …) as gymnasium envs. Importing `stable_worldmodel` registers every game under its standard `ALE/<Game>-v5` id, so any title from the official catalogue is available via `gym.make`.

For the full game list, action sets, and per-game details, see the upstream docs: <https://ale.farama.org/environments/complete_list/>.

```python
import stable_worldmodel as swm
from stable_worldmodel.policy import RandomPolicy

world = swm.World(
    'ALE/Breakout-v5',
    num_envs=1,
    image_shape=(224, 160),
    goal_conditioned=False,
    render_mode='rgb_array',
)
world.set_policy(RandomPolicy(seed=0))
world.evaluate(episodes=1, seed=0, video='./videos/breakout')
```

## Install

`ale-py` ships with the `env` extra:

```bash
pip install 'stable-worldmodel[env]'
```

If it's missing at import time, a warning is emitted and `ALE/*` ids are not registered.

## Notes

- Pass `render_mode='rgb_array'` so `MegaWrapper` can grab frames via `env.render()`.
- ALE actions are discrete (`Discrete(n)`); `n` varies per game.
- Atari frames are 210×160. Pick an `image_shape` whose dims are divisible by 16 (e.g. `(224, 160)`) to avoid imageio's macro-block-size warning when writing mp4s.
- These envs are not goal-conditioned — pass `goal_conditioned=False` to `swm.World`.
