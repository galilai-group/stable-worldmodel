import os

import gymnasium as gym
import numpy as np


os.environ.setdefault('MUJOCO_GL', 'egl')


def _flatten_obs(obs):
    if isinstance(obs, dict):
        parts = []
        for k in sorted(obs.keys()):
            parts.append(_flatten_obs(obs[k]))
        if not parts:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(parts, axis=0, dtype=np.float32)
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    return arr


class ManiSkillWrapper(gym.Env):

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}

    def __init__(
        self,
        task_id: str,
        obs_mode: str = 'state',
        control_mode: str | None = None,
        render_mode: str = 'rgb_array',
        sim_backend: str = 'cpu',
        seed: int | None = None,
        **make_kwargs,
    ):
        import mani_skill.envs

        self._task_id = task_id
        kwargs = dict(
            obs_mode=obs_mode,
            render_mode=render_mode,
            sim_backend=sim_backend,
            num_envs=1,
        )
        if control_mode is not None:
            kwargs['control_mode'] = control_mode
        kwargs.update(make_kwargs)

        self.env = gym.make(task_id, **kwargs)

        obs, _ = self.env.reset(seed=seed)
        flat = _flatten_obs(self._to_numpy(obs))
        self.observation_space = gym.spaces.Box(
            low=np.full(flat.shape, -np.inf, dtype=np.float32),
            high=np.full(flat.shape, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

        act_space = self.env.action_space
        self._action_keys: list[str] | None = None
        self._action_splits: list[int] | None = None
        if isinstance(act_space, gym.spaces.Dict):
            self._action_keys = list(act_space.spaces.keys())
            lows, highs, sizes = [], [], []
            for k in self._action_keys:
                sub = act_space.spaces[k]
                sub_low = np.asarray(sub.low, dtype=np.float32)
                sub_high = np.asarray(sub.high, dtype=np.float32)
                if sub_low.ndim == 2:
                    sub_low, sub_high = sub_low[0], sub_high[0]
                lows.append(sub_low)
                highs.append(sub_high)
                sizes.append(sub_low.size)
            low = np.concatenate(lows)
            high = np.concatenate(highs)
            self._action_splits = list(np.cumsum(sizes)[:-1])
        elif hasattr(act_space, 'low') and act_space.low.ndim == 2:
            low = np.asarray(act_space.low[0], dtype=np.float32)
            high = np.asarray(act_space.high[0], dtype=np.float32)
        else:
            low = np.asarray(act_space.low, dtype=np.float32)
            high = np.asarray(act_space.high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )

        self._cumulative_reward = 0.0
        self.variation_space = None
        self.env_name = self.__class__.__name__.replace('Wrapper', '')

    @property
    def unwrapped(self):
        return self

    @staticmethod
    def _to_numpy(x):
        import torch

        if isinstance(x, dict):
            return {k: ManiSkillWrapper._to_numpy(v) for k, v in x.items()}
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim > 0 and x.shape[0] == 1:
            x = x[0]
        return x

    @staticmethod
    def _scalarize(x):
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x).reshape(-1)
        return float(x[0]) if x.size else float('nan')

    @property
    def info(self):
        return {
            'env_name': self.env_name,
            'success': float('nan'),
            'score': self._cumulative_reward,
        }

    def reset(self, seed=None, options=None):
        self._cumulative_reward = 0.0
        obs, info = self.env.reset(seed=seed, options=options)
        obs = _flatten_obs(self._to_numpy(obs))
        out_info = self.info
        success = info.get('success') if isinstance(info, dict) else None
        if success is not None:
            out_info['success'] = float(self._scalarize(success))
        return obs, out_info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self._action_keys is not None:
            parts = np.split(action, self._action_splits)
            action = {
                k: parts[i].reshape(1, -1)
                for i, k in enumerate(self._action_keys)
            }
        else:
            action = action.reshape(1, -1)
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self._scalarize(reward)
        terminated = bool(self._scalarize(terminated))
        truncated = bool(self._scalarize(truncated))
        self._cumulative_reward += reward

        out_info = self.info
        success = info.get('success') if isinstance(info, dict) else None
        if success is not None:
            out_info['success'] = float(self._scalarize(success))

        obs = _flatten_obs(self._to_numpy(obs))
        return obs, reward, terminated, truncated, out_info

    def render(self, width: int = 224, height: int = 224, camera_id=None):
        frame = self.env.render()
        frame = self._to_numpy(frame)
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def close(self):
        self.env.close()
