from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np


class PettingZooParallelWrapper(gym.Env):
    """Adapt a PettingZoo ParallelEnv to the ``World``/``EnvPool`` contract.

    One wrapped PettingZoo environment is treated as one SWM environment.
    PettingZoo agents are exposed as stable per-agent info columns such as
    ``observation.player_0``, ``action.player_0`` and ``reward.player_0``.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        env: Any,
        *,
        reward_aggregation: str | Callable[[dict[Any, float]], float] = 'sum',
    ) -> None:
        super().__init__()
        self.env = env
        self.possible_agents = list(env.possible_agents)
        if not self.possible_agents:
            raise ValueError('PettingZoo env must define possible_agents.')

        self._agent_to_key = {
            agent: _agent_key(agent) for agent in self.possible_agents
        }
        if len(set(self._agent_to_key.values())) != len(self._agent_to_key):
            raise ValueError(
                'PettingZoo agent names must be unique after string '
                f'conversion: {self._agent_to_key!r}'
            )
        self._key_to_agent = {v: k for k, v in self._agent_to_key.items()}
        self.reward_aggregation = reward_aggregation
        self._step_counter = 0
        self._id = 0
        self._last_actions = {
            agent: _empty_action(env.action_space(agent))
            for agent in self.possible_agents
        }

        self.action_space = gym.spaces.Dict(
            {
                self._agent_to_key[agent]: env.action_space(agent)
                for agent in self.possible_agents
            }
        )
        self.observation_space = gym.spaces.Dict(
            {
                self._agent_to_key[agent]: env.observation_space(agent)
                for agent in self.possible_agents
            }
        )
        self.metadata = getattr(env, 'metadata', self.metadata)

    @property
    def agents(self) -> list[Any]:
        return list(getattr(self.env, 'agents', []))

    @property
    def unwrapped(self):
        return getattr(self.env, 'unwrapped', self.env)

    def reset(self, *, seed=None, options=None):
        self._step_counter = 0
        observations, infos = self.env.reset(seed=seed, options=options)
        self._id = _episode_id(self.env)
        self._last_actions = {
            agent: _empty_action(self.env.action_space(agent))
            for agent in self.possible_agents
        }
        info = self._build_info(
            observations=observations,
            rewards={},
            terminations={},
            truncations={},
            infos=infos,
            env_reward=np.nan,
            env_terminated=False,
            env_truncated=False,
        )
        return observations, info

    def step(self, action):
        actions = self._to_pettingzoo_actions(action)
        observations, rewards, terminations, truncations, infos = (
            self.env.step(actions)
        )
        self._step_counter += 1

        for agent, value in actions.items():
            self._last_actions[agent] = value

        env_reward = self._aggregate_reward(rewards)
        env_done = _env_done(self.env, terminations, truncations)
        env_terminated = env_done and any(
            bool(v) for v in terminations.values()
        )
        env_truncated = (
            env_done
            and not env_terminated
            and any(bool(v) for v in truncations.values())
        )
        info = self._build_info(
            observations=observations,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            infos=infos,
            env_reward=env_reward,
            env_terminated=env_terminated,
            env_truncated=env_truncated,
        )
        return observations, env_reward, env_terminated, env_truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def _to_pettingzoo_actions(self, action) -> dict[Any, Any]:
        if not isinstance(action, dict):
            raise TypeError(
                'PettingZooParallelWrapper.step expects a dict keyed by agent.'
            )
        actions = {}
        live_agents = set(self.agents)
        for key, value in action.items():
            agent = self._key_to_agent.get(key, key)
            if agent in live_agents:
                actions[agent] = value
        return actions

    def _aggregate_reward(self, rewards: dict[Any, float]) -> float:
        if callable(self.reward_aggregation):
            return float(self.reward_aggregation(rewards))
        values = [float(v) for v in rewards.values()]
        if not values:
            return 0.0
        if self.reward_aggregation == 'sum':
            return float(np.sum(values))
        if self.reward_aggregation == 'mean':
            return float(np.mean(values))
        raise ValueError(
            "reward_aggregation must be 'sum', 'mean', or a callable."
        )

    def _build_info(
        self,
        *,
        observations: dict[Any, Any],
        rewards: dict[Any, float],
        terminations: dict[Any, bool],
        truncations: dict[Any, bool],
        infos: dict[Any, dict],
        env_reward: float,
        env_terminated: bool,
        env_truncated: bool,
    ) -> dict[str, Any]:
        live_agents = set(self.agents)
        info: dict[str, Any] = {
            'agent_mask': np.array(
                [agent in live_agents for agent in self.possible_agents],
                dtype=bool,
            ),
            'reward': env_reward,
            'terminated': bool(env_terminated),
            'truncated': bool(env_truncated),
            'step_idx': self._step_counter,
            'id': self._id,
        }

        for agent in self.possible_agents:
            key = self._agent_to_key[agent]
            obs = observations.get(
                agent, _empty_observation(self.env.observation_space(agent))
            )
            _write_space_value(
                info,
                f'observation.{key}',
                obs,
                self.env.observation_space(agent),
                _empty_observation,
            )
            _write_space_value(
                info,
                f'action.{key}',
                self._last_actions[agent],
                self.env.action_space(agent),
                _empty_action,
            )
            info[f'reward.{key}'] = float(rewards.get(agent, np.nan))
            info[f'terminated.{key}'] = bool(terminations.get(agent, False))
            info[f'truncated.{key}'] = bool(truncations.get(agent, False))

            for sub_key, value in (infos.get(agent) or {}).items():
                if _is_stackable(value):
                    info[f'info.{key}.{sub_key}'] = value

        return info


def _agent_key(agent: Any) -> str:
    return str(agent).replace('.', '_')


class PettingZooAECWrapper(gym.Env):
    """Adapt a PettingZoo AECEnv to the ``World``/``EnvPool`` contract.

    One SWM step advances one selected PettingZoo agent turn. The current
    agent is exposed through ``current_agent`` and ``current_agent_idx``.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        env: Any,
        *,
        reward_aggregation: str | Callable[[dict[Any, float]], float] = 'sum',
    ) -> None:
        super().__init__()
        self.env = env
        self.possible_agents = list(env.possible_agents)
        if not self.possible_agents:
            raise ValueError('PettingZoo env must define possible_agents.')

        self._agent_to_key = {
            agent: _agent_key(agent) for agent in self.possible_agents
        }
        if len(set(self._agent_to_key.values())) != len(self._agent_to_key):
            raise ValueError(
                'PettingZoo agent names must be unique after string '
                f'conversion: {self._agent_to_key!r}'
            )
        self._key_to_agent = {v: k for k, v in self._agent_to_key.items()}
        self._agent_indices = {
            agent: i for i, agent in enumerate(self.possible_agents)
        }
        self.reward_aggregation = reward_aggregation
        self._step_counter = 0
        self._id = 0
        self._current_agent = self.possible_agents[0]
        self._last_actions = {
            agent: _empty_action(env.action_space(agent))
            for agent in self.possible_agents
        }
        self._last_observations = {
            agent: _empty_observation(env.observation_space(agent))
            for agent in self.possible_agents
        }
        self._last_infos = {agent: {} for agent in self.possible_agents}

        self.action_space = gym.spaces.Dict(
            {
                self._agent_to_key[agent]: env.action_space(agent)
                for agent in self.possible_agents
            }
        )
        self.observation_space = gym.spaces.Dict(
            {
                self._agent_to_key[agent]: env.observation_space(agent)
                for agent in self.possible_agents
            }
        )
        self.metadata = getattr(env, 'metadata', self.metadata)

    @property
    def agents(self) -> list[Any]:
        return list(getattr(self.env, 'agents', []))

    @property
    def unwrapped(self):
        return getattr(self.env, 'unwrapped', self.env)

    def reset(self, *, seed=None, options=None):
        self._step_counter = 0
        self.env.reset(seed=seed, options=options)
        self._id = _episode_id(self.env)
        self._last_actions = {
            agent: _empty_action(self.env.action_space(agent))
            for agent in self.possible_agents
        }
        self._last_observations = {
            agent: _empty_observation(self.env.observation_space(agent))
            for agent in self.possible_agents
        }
        self._last_infos = {agent: {} for agent in self.possible_agents}
        self._capture_current_agent()

        rewards = _agent_attr_dict(self.env, 'rewards')
        terminations = _agent_attr_dict(self.env, 'terminations')
        truncations = _agent_attr_dict(self.env, 'truncations')
        info = self._build_info(
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            env_reward=np.nan,
            env_terminated=False,
            env_truncated=False,
        )
        return self._last_observations, info

    def step(self, action):
        if not self.agents:
            info = self._build_info(
                rewards={},
                terminations={},
                truncations={},
                env_reward=0.0,
                env_terminated=True,
                env_truncated=False,
            )
            return self._last_observations, 0.0, True, False, info

        agent = self._current_agent
        actions = self._to_agent_actions(action)
        agent_action = actions.get(agent)
        if agent_action is None:
            agent_action = _empty_action(self.env.action_space(agent))
        self.env.step(agent_action)
        self._step_counter += 1
        self._last_actions[agent] = agent_action

        rewards = _agent_attr_dict(self.env, 'rewards')
        terminations = _agent_attr_dict(self.env, 'terminations')
        truncations = _agent_attr_dict(self.env, 'truncations')
        env_reward = self._aggregate_reward(rewards)
        env_done = not self.agents
        env_terminated = env_done and any(
            bool(v) for v in terminations.values()
        )
        env_truncated = (
            env_done
            and not env_terminated
            and any(bool(v) for v in truncations.values())
        )
        self._capture_current_agent()
        info = self._build_info(
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            env_reward=env_reward,
            env_terminated=env_terminated,
            env_truncated=env_truncated,
        )
        return (
            self._last_observations,
            env_reward,
            env_terminated,
            env_truncated,
            info,
        )

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def _capture_current_agent(self) -> None:
        if not self.agents:
            return
        agent = getattr(self.env, 'agent_selection', self.agents[0])
        observation, reward, termination, truncation, info = self.env.last()
        self._current_agent = agent
        self._last_observations[agent] = observation
        self._last_infos[agent] = info or {}
        rewards = _agent_attr_dict(self.env, 'rewards')
        rewards[agent] = reward
        terminations = _agent_attr_dict(self.env, 'terminations')
        terminations[agent] = termination
        truncations = _agent_attr_dict(self.env, 'truncations')
        truncations[agent] = truncation

    def _to_agent_actions(self, action) -> dict[Any, Any]:
        if isinstance(action, dict):
            return {
                self._key_to_agent.get(key, key): value
                for key, value in action.items()
            }
        return {self._current_agent: action}

    def _aggregate_reward(self, rewards: dict[Any, float]) -> float:
        if callable(self.reward_aggregation):
            return float(self.reward_aggregation(rewards))
        values = [float(v) for v in rewards.values()]
        if not values:
            return 0.0
        if self.reward_aggregation == 'sum':
            return float(np.sum(values))
        if self.reward_aggregation == 'mean':
            return float(np.mean(values))
        raise ValueError(
            "reward_aggregation must be 'sum', 'mean', or a callable."
        )

    def _build_info(
        self,
        *,
        rewards: dict[Any, float],
        terminations: dict[Any, bool],
        truncations: dict[Any, bool],
        env_reward: float,
        env_terminated: bool,
        env_truncated: bool,
    ) -> dict[str, Any]:
        live_agents = set(self.agents)
        current_agent = self._current_agent if self.agents else None
        info: dict[str, Any] = {
            'agent_mask': np.array(
                [agent in live_agents for agent in self.possible_agents],
                dtype=bool,
            ),
            'current_agent_idx': (
                self._agent_indices[current_agent]
                if current_agent is not None
                else -1
            ),
            'current_agent': (
                self._agent_to_key[current_agent]
                if current_agent is not None
                else ''
            ),
            'reward': env_reward,
            'terminated': bool(env_terminated),
            'truncated': bool(env_truncated),
            'step_idx': self._step_counter,
            'id': self._id,
        }

        for agent in self.possible_agents:
            key = self._agent_to_key[agent]
            _write_space_value(
                info,
                f'observation.{key}',
                self._last_observations[agent],
                self.env.observation_space(agent),
                _empty_observation,
            )
            _write_space_value(
                info,
                f'action.{key}',
                self._last_actions[agent],
                self.env.action_space(agent),
                _empty_action,
            )
            info[f'reward.{key}'] = float(rewards.get(agent, 0.0))
            info[f'terminated.{key}'] = bool(terminations.get(agent, False))
            info[f'truncated.{key}'] = bool(truncations.get(agent, False))

            for sub_key, value in self._last_infos.get(agent, {}).items():
                if _is_stackable(value):
                    info[f'info.{key}.{sub_key}'] = value

        return info


def _episode_id(env: Any) -> int:
    rng = getattr(env, 'np_random', None)
    if rng is None:
        rng = getattr(getattr(env, 'unwrapped', None), 'np_random', None)
    if rng is not None:
        max_int = np.iinfo(np.int64).max
        if hasattr(rng, 'integers'):
            return int(rng.integers(0, max_int))
        if hasattr(rng, 'randint'):
            return int(rng.randint(0, max_int))
    return int(np.random.default_rng().integers(0, np.iinfo(np.int64).max))


def _empty_observation(space: gym.Space) -> np.ndarray:
    if isinstance(space, gym.spaces.Dict):
        return {
            key: _empty_observation(subspace)
            for key, subspace in space.spaces.items()
        }
    if isinstance(space, gym.spaces.Tuple):
        return tuple(_empty_observation(subspace) for subspace in space.spaces)
    if isinstance(space, gym.spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)
    if isinstance(space, gym.spaces.Discrete):
        return np.array(-1, dtype=np.int64)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return np.full(space.nvec.shape, -1, dtype=np.int64)
    if isinstance(space, gym.spaces.MultiBinary):
        return np.full(space.shape, -1, dtype=np.int8)
    sample = space.sample()
    return np.zeros_like(_as_array(sample))


def _empty_action(space: gym.Space) -> np.ndarray:
    if isinstance(space, gym.spaces.Dict):
        return {
            key: _empty_action(subspace)
            for key, subspace in space.spaces.items()
        }
    if isinstance(space, gym.spaces.Tuple):
        return tuple(_empty_action(subspace) for subspace in space.spaces)
    if isinstance(space, gym.spaces.Box):
        if np.issubdtype(space.dtype, np.floating):
            return np.full(space.shape, np.nan, dtype=space.dtype)
        return np.zeros(space.shape, dtype=space.dtype)
    if isinstance(space, gym.spaces.Discrete):
        return np.array(-1, dtype=np.int64)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return np.full(space.nvec.shape, -1, dtype=np.int64)
    if isinstance(space, gym.spaces.MultiBinary):
        return np.full(space.shape, -1, dtype=np.int8)
    sample = _as_array(space.sample())
    if np.issubdtype(sample.dtype, np.floating):
        return np.full_like(sample, np.nan)
    return np.zeros_like(sample)


def _as_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.copy()
    return np.asarray(value)


def _is_stackable(value: Any) -> bool:
    return isinstance(value, (np.ndarray, bool, int, float, np.number))


def _write_space_value(
    info: dict[str, Any],
    prefix: str,
    value: Any,
    space: gym.Space,
    empty_value: Callable[[gym.Space], Any],
) -> None:
    if isinstance(space, gym.spaces.Dict):
        value = value if isinstance(value, dict) else empty_value(space)
        for key, subspace in space.spaces.items():
            sub_value = value.get(key, empty_value(subspace))
            _write_space_value(
                info,
                f'{prefix}.{_agent_key(key)}',
                sub_value,
                subspace,
                empty_value,
            )
        return
    if isinstance(space, gym.spaces.Tuple):
        if not isinstance(value, tuple | list):
            value = empty_value(space)
        for i, subspace in enumerate(space.spaces):
            sub_value = value[i] if i < len(value) else empty_value(subspace)
            _write_space_value(
                info, f'{prefix}.{i}', sub_value, subspace, empty_value
            )
        return
    info[prefix] = _as_array(value)


def _env_done(
    env: Any,
    terminations: dict[Any, bool],
    truncations: dict[Any, bool],
) -> bool:
    live = list(getattr(env, 'agents', []))
    if not live:
        return True
    return all(
        bool(terminations.get(agent, False))
        or bool(truncations.get(agent, False))
        for agent in live
    )


def _agent_attr_dict(env: Any, name: str) -> dict[Any, Any]:
    value = getattr(env, name, {})
    return dict(value) if value is not None else {}
