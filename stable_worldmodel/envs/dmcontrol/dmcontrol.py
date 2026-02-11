# wrapper adapted from https://github.com/nicklashansen/newt/blob/main/tdmpc2/envs/dmcontrol.py

import gymnasium as gym
import mujoco  # noqa: F401
import numpy as np
from stable_worldmodel import spaces as swm_spaces


def get_obs_shape(env):
    obs_shp = []
    for v in env.observation_spec().values():
        try:
            shp = np.prod(v.shape)
        except Exception:
            shp = 1
        obs_shp.append(shp)
    return (int(np.sum(obs_shp)),)


class DMControlWrapper(gym.Env):
    def __init__(self, env, domain):
        self.env = env
        self.camera_id = 2 if domain == "quadruped" else 0
        obs_shape = get_obs_shape(env)
        action_shape = env.action_spec().shape
        self.observation_space = gym.spaces.Box(
            low=np.full(obs_shape, -np.inf, dtype=np.float32),
            high=np.full(obs_shape, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.full(action_shape, env.action_spec().minimum),
            high=np.full(action_shape, env.action_spec().maximum),
            dtype=env.action_spec().dtype,
        )
        self.action_spec_dtype = env.action_spec().dtype
        self._cumulative_reward = 0
        self.action_repeat = 2
        self.variation_space = None

    @property
    def unwrapped(self):
        return self

    @property
    def dmc_env(self):
        """Access the underlying dm_control env explicitly."""
        return self.env

    @property
    def info(self):
        return {
            "success": float("nan"),
            "qpos": np.copy(self.env.physics.data.qpos),
            "qvel": np.copy(self.env.physics.data.qvel),
            "score": self._cumulative_reward / 1000,
        }

    def _obs_to_array(self, obs):
        return np.concatenate([v.flatten() for v in obs.values()], dtype=np.float32)

    def reset(self, seed=None, options=None):
        options = options or {}
        swm_spaces.reset_variation_space(
            self.variation_space,
            seed,
            options,
        )

        self._mjcf_model = self.modify_mjcf_model(self._mjcf_model)
        if self._dirty:
            self.compile_model(seed=seed, environment_kwargs={})

        self._cumulative_reward = 0
        time_step = self.env.reset()
        obs = time_step.observation
        if "state" in options and options["state"] is not None:
            state = np.asarray(options["state"])
            assert state.ndim == 1, "State option must be a 1D array!"
            nq = self.env.physics.model.nq
            nv = self.env.physics.model.nv
            assert state.shape[0] == nq + nv, (
                f"State option must have shape ({nq + nv},)!"
            )
            self.set_state(state[:nq], state[nq:])
            obs = self.env.task.get_observation(self.env.physics)
        return self._obs_to_array(obs), self.info

    def step(self, action):
        reward = 0
        action = action.astype(self.action_spec_dtype)
        for _ in range(self.action_repeat):
            step = self.env.step(action)
            reward += step.reward or 0
        self._cumulative_reward += reward

        return (
            self._obs_to_array(step.observation),
            reward,
            False,
            False,
            self.info,
        )

    def set_state(self, qpos, qvel):
        """Reset the environment to a specific state."""

        assert qpos.shape == (self.env.physics.model.nq,) and qvel.shape == (
            self.env.physics.model.nv,
        )
        self.env.physics.data.qpos[:] = np.copy(qpos)
        self.env.physics.data.qvel[:] = np.copy(qvel)
        if self.env.physics.model.na == 0:
            self.env.physics.data.act[:] = None
        self.env.physics.forward()

    def render(self, width=224, height=224, camera_id=None):
        return self.env.physics.render(height, width, camera_id or self.camera_id)

    def close(self):
        self.env.close()

    def compile_model(self, seed=None, environment_kwargs=None):
        raise NotImplementedError

    def modify_mjcf_model(self, mjcf_model):
        raise NotImplementedError

    def mark_dirty(self):
        """Mark the environment as dirty, requiring recompilation of the model."""
        self._dirty = True
