# wrapper adapted from https://github.com/nicklashansen/newt/blob/main/tdmpc2/envs/dmcontrol.py

from collections.abc import Sequence

import gymnasium as gym
import mujoco
import numpy as np


# from envs.tasks import cartpole, cheetah, walker, hopper, ball_in_cup, pendulum, fish, giraffe, spinner, jumper
# from dm_control import suite
# from dm_control.suite import common
# from dm_control import mjcf
# suite._DOMAINS['giraffe'] = giraffe
# suite._DOMAINS['spinner'] = spinner
# suite._DOMAINS['jumper'] = jumper
# suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
# suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)
# from dm_control.suite.wrappers import action_scale

# from envs.wrappers.pixels import Pixels


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
            "score": self._cumulative_reward / 1000,
        }

    def _obs_to_array(self, obs):
        return np.concatenate([v.flatten() for v in obs.values()], dtype=np.float32)

    def reset(self, seed=None, options=None):
        options = options or {}
        variations = options.get("variation", [])
        if not isinstance(variations, Sequence):
            raise ValueError("variation option must be a Sequence containing variations names to sample")

        assert self.variation_space is not None, "Variation space must be defined to apply variations!"
        self.variation_space.reset()
        self.variation_space.update(variations)

        if options is not None and "variation_values" in options:
            self.variation_space.set_value(options["variation_values"])

        assert self.variation_space.check(debug=True), "Variation values must be within variation space!"

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
            assert state.shape[0] == nq + nv, f"State option must have shape ({nq + nv},)!"
            self.set_state(state[:nq], state[nq:])
            obs = self.env.task.get_observation(self.env.physics)
        return self._obs_to_array(obs), self.info

    def step(self, action):
        reward = 0
        action = action.astype(self.action_spec_dtype)
        for _ in range(self.action_repeat):
            step = self.env.step(action)
            reward += step.reward
        self._cumulative_reward += reward
        return self._obs_to_array(step.observation), reward, False, False, self.info

    def set_state(self, qpos, qvel):
        """Reset the environment to a specific state."""
        assert qpos.shape == (self.env.physics.model.nq,) and qvel.shape == (self.env.physics.model.nv,)
        self.env.physics.data.qpos[:] = np.copy(qpos)
        self.env.physics.data.qvel[:] = np.copy(qvel)
        if self.env.physics.model.na == 0:
            self.env.physics.data.act[:] = None
        mujoco.mj_forward(self.env.physics.model, self.env.physics.data)

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


# def make_env(cfg):
# 	"""
# 	Make DMControl environment.
# 	Adapted from https://github.com/facebookresearch/drqv2
# 	"""
# 	domain, task = cfg.task.replace('-', '_').split('_', 1)
# 	domain = dict(cup='ball_in_cup', pointmass='point_mass').get(domain, domain)
# 	if (domain, task) not in suite.ALL_TASKS:
# 		raise ValueError('Unknown task:', task)
# 	assert cfg.obs in {'state', 'rgb'}, 'This task only supports state and rgb observations.'
# 	env = suite.load(domain,
# 					 task,
# 					 task_kwargs={'random': cfg.seed},
# 					 visualize_reward=False)
# 	env = action_scale.Wrapper(env, minimum=-1., maximum=1.)
# 	env = DMControlWrapper(env, domain)
# 	# if cfg.obs == 'rgb':
# 	# 	env = Pixels(env, cfg)
# 	return env

# if __name__ == '__main__':
# 	# Quick smoke test: load humanoid walk and wrap it.
# 	env = suite.load('humanoid', 'walk', task_kwargs={'random': 0}, visualize_reward=False)
# 	env = action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)
# 	env = DMControlWrapper(env, 'humanoid')
# 	obs, info = env.reset()
# 	print('obs shape:', obs.shape)
# 	print('info:', info)
# 	env.close()
