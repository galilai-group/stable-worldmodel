"""Atari wrappers using ``ale_py.vector_env.AtariVectorEnv``.

All standard Atari preprocessing (noop reset, fire reset, episodic life,
reward clipping, frame skip, resize) is handled natively by the ALE C++
vector environment.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


def _parse_game(env_name: str) -> str:
    """Convert e.g. ``ALE/Breakout-v5`` or ``Breakout`` to ``breakout``."""
    name = env_name.split('/')[-1].split('-')[0]
    # Insert underscores before uppercase letters and lowercase
    chars = []
    for i, c in enumerate(name):
        if i > 0 and c.isupper() and name[i - 1].islower():
            chars.append('_')
        chars.append(c.lower())
    return ''.join(chars)


class AtariEnvAdapter:
    """Wraps ``AtariVectorEnv(num_envs=1)`` with a single-env interface."""

    def __init__(self, vector_env):
        self._venv = vector_env

    @property
    def unwrapped(self):
        return self._venv

    @property
    def action_space(self):
        return self._venv.single_action_space

    @property
    def observation_space(self):
        return self._venv.single_observation_space

    def reset(self, **kwargs):
        obs, info = self._venv.reset(**kwargs)
        return obs[0, 0], info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._venv.step(
            np.asarray([action])
        )
        return (
            obs[0, 0],
            float(reward[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            info,
        )

    def close(self):
        self._venv.close()

    def seed(self, seed=None):
        self._venv.reset(seed=seed)


def make_atari_env(
    env_name,
    seed=0,
    frameskip=4,
    max_episode_steps=27000,
    clip_reward=True,
    episodic_life=True,
    noop_max=30,
    fire_reset=True,
    repeat_action_probability=0.25,
    img_size=64,
):
    """Create an Atari environment via ``ale_py.vector_env.AtariVectorEnv``.

    All preprocessing (noop reset, fire reset, episodic life, reward clipping,
    frame skip, resize) is handled natively by the ALE C++ implementation.
    Returns a single-env adapter (not a raw ``VectorEnv``).
    """
    from ale_py.vector_env import AtariVectorEnv

    game = _parse_game(env_name)

    venv = AtariVectorEnv(
        game=game,
        num_envs=1,
        frameskip=frameskip,
        grayscale=False,
        stack_num=1,
        img_height=img_size,
        img_width=img_size,
        maxpool=False,
        noop_max=noop_max,
        use_fire_reset=fire_reset,
        episodic_life=episodic_life,
        reward_clipping=clip_reward,
        max_num_frames_per_episode=max_episode_steps * frameskip,
        repeat_action_probability=repeat_action_probability,
        full_action_space=True,
    )
    env = AtariEnvAdapter(venv)
    env.reset(seed=seed)
    return env


__all__ = [
    'AtariEnvAdapter',
    'make_atari_env',
]
