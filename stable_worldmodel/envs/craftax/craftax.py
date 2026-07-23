"""Wrapper to convert Craftax envs to Gymnasium envs."""

import gymnasium as gym
import jax
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name
from gymnasium import spaces


class CraftaxWrapper(gym.Env):
    """Convert a Craftax (gymnax-style) env to a Gymnasium Env."""

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}

    def __init__(
        self,
        env_name: str,
        obs_shape: tuple[int, ...],
        seed: int = 0,
        backend: str | None = None,
        render_mode: str | None = None,
    ):
        self._env = make_craftax_env_from_name(env_name, auto_reset=False)
        self._params = self._env.default_params
        self._is_pixels = 'Pixels' in env_name
        self.render_mode = render_mode
        self.backend = backend
        self._state = None
        self._last_obs = None
        self._last_info = None
        self._key = jax.random.PRNGKey(seed)

        # Bounds hardcoded to [0, 1] (matches all current Craftax variants).
        # Shape is hardcoded by each subclass — Craftax-Pixels-v1's declared
        # observation_space has H and W swapped, so we cannot trust it.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Discrete(self._env.num_actions)

        env = self._env
        params = self._params

        def _reset(key):
            key1, key2 = jax.random.split(key)
            obs, state = env.reset(key2, params)
            return obs, state, key1

        def _step(key, state, action):
            key1, key2 = jax.random.split(key)
            obs, state, reward, done, info = env.step(
                key2, state, action, params
            )
            return obs, state, reward, done, info, key1

        self._reset_fn = jax.jit(_reset, backend=self.backend)
        self._step_fn = jax.jit(_step, backend=self.backend)

        self.variation_space = None
        self.env_name = (
            self.__class__.__name__.replace('CraftaxWrapper', '') or env_name
        )

    @property
    def unwrapped(self):
        return self

    @property
    def craftax_env(self):
        """Access the underlying Craftax env explicitly."""
        return self._env

    @property
    def info(self):
        if self._last_info is None:
            return {'env_name': self.env_name}
        return {
            'env_name': self.env_name,
            **{k: np.asarray(v) for k, v in self._last_info.items()},
        }

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        obs, self._state, self._key = self._reset_fn(self._key)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._last_info = None
        return self._last_obs, self.info

    def step(self, action):
        obs, self._state, reward, done, info, self._key = self._step_fn(
            self._key, self._state, int(action)
        )
        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._last_info = info
        return (
            self._last_obs,
            float(np.asarray(reward)),
            bool(np.asarray(done)),
            False,
            self.info,
        )

    def render(self):
        if self.render_mode != 'rgb_array':
            raise NotImplementedError(
                f'render_mode={self.render_mode!r} is not supported; '
                "use 'rgb_array'."
            )
        if not self._is_pixels:
            raise NotImplementedError(
                'render is only supported for Pixels variants; symbolic '
                'envs do not produce a frame.'
            )
        if self._last_obs is None:
            raise RuntimeError('must call reset or step before rendering')
        return (np.clip(self._last_obs, 0.0, 1.0) * 255.0).astype(np.uint8)

    def close(self):
        pass


class CraftaxPixelsWrapper(CraftaxWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            env_name='Craftax-Pixels-v1', obs_shape=(130, 110, 3), **kwargs
        )


class CraftaxSymbolicWrapper(CraftaxWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            env_name='Craftax-Symbolic-v1', obs_shape=(8268,), **kwargs
        )


class CraftaxClassicPixelsWrapper(CraftaxWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            env_name='Craftax-Classic-Pixels-v1',
            obs_shape=(63, 63, 3),
            **kwargs,
        )


class CraftaxClassicSymbolicWrapper(CraftaxWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            env_name='Craftax-Classic-Symbolic-v1',
            obs_shape=(1345,),
            **kwargs,
        )
