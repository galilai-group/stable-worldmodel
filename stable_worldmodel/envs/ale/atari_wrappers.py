"""Atari 100k protocol wrappers.

These follow the standard protocol from "Leveraging Procedural Generation to
Benchmark Reinforcement Learning" (PLANet, 2020) used by the DIAMOND paper.
"""

import gymnasium as gym
import numpy as np


class NoopResetEnv(gym.Wrapper):
    """Perform a random number of no-op actions at reset to add stochasticity."""

    def __init__(self, env, max_noop=30):
        super().__init__(env)
        self.max_noop = max_noop
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.max_noop + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, _ = self.env.step(0)
            if terminated or truncated:
                obs, _ = self.env.reset(**kwargs)
                break
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """Make terminal when a life is lost, so the agent learns to survive."""

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_lives = self.env.unwrapped.ale.lives()
        if 0 < new_lives < self.lives:
            terminated = True
        self.lives = new_lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Press FIRE on reset for games that require it (e.g. Breakout, Space Invaders)."""

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            obs, _ = self.env.reset(**kwargs)
        return obs, info


class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to {-1, 0, 1}."""

    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return np.sign(reward)


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
):
    """Create an Atari 100k protocol environment.

    Applies the standard wrappers used by the DIAMOND paper for evaluation
    on the Atari 100k benchmark.
    """
    import ale_py

    gym.register_envs(ale_py)

    env = gym.make(
        env_name,
        obs_type='rgb',
        frameskip=frameskip,
        mode=0,
        difficulty=0,
        repeat_action_probability=repeat_action_probability,
        full_action_space=True,
        max_episode_steps=max_episode_steps,
    )

    if noop_max > 0:
        env = NoopResetEnv(env, max_noop=noop_max)

    if fire_reset:
        env = FireResetEnv(env)

    if episodic_life:
        env = EpisodicLifeEnv(env)

    if clip_reward:
        env = ClipRewardEnv(env)

    env.reset(seed=seed)
    return env


__all__ = [
    'NoopResetEnv',
    'EpisodicLifeEnv',
    'FireResetEnv',
    'ClipRewardEnv',
    'make_atari_env',
]
