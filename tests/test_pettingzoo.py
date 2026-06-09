import gymnasium as gym
import numpy as np
import pytest

from stable_worldmodel import MultiAgentRandomPolicy
from stable_worldmodel.world import EnvPool, World
from stable_worldmodel.wrapper import (
    PettingZooAECWrapper,
    PettingZooParallelWrapper,
)


class TinyParallelEnv:
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, max_cycles: int = 2):
        self.possible_agents = ['player_0', 'player_1']
        self.agents = []
        self.max_cycles = max_cycles
        self.cycles = 0
        self.np_random = np.random.default_rng()
        self.action_log = []
        self.observation_spaces = {
            agent: gym.spaces.Dict(
                {
                    'obs': gym.spaces.Box(
                        low=0, high=10, shape=(1,), dtype=np.float32
                    ),
                    'action_mask': gym.spaces.MultiBinary(2),
                }
            )
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: gym.spaces.Discrete(2) for agent in self.possible_agents
        }

    @property
    def unwrapped(self):
        return self

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self.cycles = 0
        self.action_log.clear()
        return self._observations(), self._infos()

    def step(self, actions):
        assert set(actions) <= set(self.agents)
        self.action_log.append({k: int(v) for k, v in actions.items()})
        self.cycles += 1
        rewards = {agent: float(actions[agent]) for agent in self.agents}
        terminations = {
            agent: self.cycles >= self.max_cycles for agent in self.agents
        }
        truncations = {agent: False for agent in self.agents}
        if all(terminations.values()):
            observations = {}
            infos = {
                agent: {'score': float(self.cycles)} for agent in self.agents
            }
            self.agents = []
        else:
            observations = self._observations()
            infos = self._infos()
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def close(self):
        pass

    def _observations(self):
        return {
            agent: {
                'obs': np.array([self.cycles], dtype=np.float32),
                'action_mask': np.ones(2, dtype=np.int8),
            }
            for agent in self.agents
        }

    def _infos(self):
        return {
            agent: {'score': np.array([self.cycles], dtype=np.float32)}
            for agent in self.agents
        }


class TinyAECEnv:
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, max_cycles: int = 4):
        self.possible_agents = ['player_0', 'player_1']
        self.agents = []
        self.max_cycles = max_cycles
        self.agent_selection = None
        self.turns = 0
        self.np_random = np.random.default_rng()
        self.rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}
        self.action_log = []
        self.observation_spaces = {
            agent: gym.spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: gym.spaces.Discrete(2) for agent in self.possible_agents
        }

    @property
    def unwrapped(self):
        return self

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self.agent_selection = self.agents[0]
        self.turns = 0
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {
            agent: {'turns': np.array([0], dtype=np.float32)}
            for agent in self.possible_agents
        }
        self.action_log.clear()

    def last(self):
        agent = self.agent_selection
        return (
            np.array([self.turns], dtype=np.float32),
            self.rewards.get(agent, 0.0),
            self.terminations.get(agent, False),
            self.truncations.get(agent, False),
            self.infos.get(agent, {}),
        )

    def step(self, action):
        agent = self.agent_selection
        self.action_log.append((agent, int(action)))
        self.rewards = {a: 0.0 for a in self.possible_agents}
        self.rewards[agent] = float(action)
        self.turns += 1
        if self.turns >= self.max_cycles:
            self.terminations = {a: True for a in self.possible_agents}
            self.agents = []
            self.agent_selection = None
            return
        self.agent_selection = self.agents[
            self.turns % len(self.possible_agents)
        ]
        self.infos[self.agent_selection] = {
            'turns': np.array([self.turns], dtype=np.float32)
        }

    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def close(self):
        pass


class DeadTurnAECEnv(TinyAECEnv):
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.terminations['player_0'] = True

    def step(self, action):
        agent = self.agent_selection
        self.action_log.append((agent, action))
        assert action is None
        self.agents = ['player_1']
        self.agent_selection = 'player_1'
        self.rewards = {a: 0.0 for a in self.possible_agents}
        self.terminations = {'player_0': True, 'player_1': False}
        self.truncations = {a: False for a in self.possible_agents}
        self.infos['player_1'] = {
            'turns': np.array([self.turns], dtype=np.float32)
        }


class MemoryWriter:
    def __init__(self):
        self.episodes = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def write_episode(self, ep_data):
        self.episodes.append(ep_data)

    def write_episodes(self, episodes):
        self.episodes.extend(list(episodes))


def test_parallel_wrapper_flattens_agent_columns():
    env = PettingZooParallelWrapper(TinyParallelEnv(max_cycles=2))

    _, info = env.reset(seed=123)

    assert env.action_space.shape == (2,)
    assert isinstance(env.action_space.sample(), np.ndarray)
    np.testing.assert_array_equal(info['agent_mask'], [True, True])
    np.testing.assert_array_equal(info['observation.player_0.obs'], [0.0])
    np.testing.assert_array_equal(
        info['observation.player_0.action_mask'], [1, 1]
    )
    assert info['action.player_0'] == -1
    assert np.isnan(info['reward'])
    assert not info['terminated']

    _, reward, terminated, truncated, info = env.step(
        np.array([1, 0], dtype=np.int64)
    )

    assert reward == 1.0
    assert not terminated
    assert not truncated
    assert info['action.player_0'] == 1
    assert info['reward.player_0'] == 1.0
    assert env.env.action_log[-1] == {'player_0': 1, 'player_1': 0}
    env.close()


def test_env_pool_steps_batched_pettingzoo_ndarray_actions():
    pool = EnvPool(
        [
            lambda: PettingZooParallelWrapper(TinyParallelEnv(max_cycles=2))
            for _ in range(2)
        ]
    )
    pool.reset(seed=0)

    actions = np.array([[1, 0], [0, 1]], dtype=np.int64)
    _, rewards, terminateds, truncateds, infos = pool.step(actions)

    np.testing.assert_array_equal(rewards, [1.0, 1.0])
    np.testing.assert_array_equal(terminateds, [False, False])
    np.testing.assert_array_equal(truncateds, [False, False])
    assert pool.envs[0].env.action_log[-1] == {'player_0': 1, 'player_1': 0}
    assert pool.envs[1].env.action_log[-1] == {'player_0': 0, 'player_1': 1}
    np.testing.assert_array_equal(infos['action.player_0'][:, 0], [1, 0])
    pool.close()


def test_world_from_pettingzoo_collects_parallel_episodes():
    world = World.from_pettingzoo(
        lambda: TinyParallelEnv(max_cycles=2), num_envs=2
    )
    writer = MemoryWriter()
    world.set_policy(MultiAgentRandomPolicy(seed=0))

    world.collect(writer=writer, episodes=3, progress=False)

    assert len(writer.episodes) == 3
    for episode in writer.episodes:
        assert set(
            [
                'agent_mask',
                'observation.player_0.obs',
                'observation.player_0.action_mask',
                'action.player_0',
                'reward.player_0',
                'terminated.player_0',
            ]
        ) <= set(episode)
        assert len(episode['reward.player_0']) == 2
        assert all(
            not isinstance(value, dict)
            for values in episode.values()
            for value in values
        )
    world.close()


def test_aec_wrapper_advances_one_agent_turn_per_step():
    env = PettingZooAECWrapper(TinyAECEnv(max_cycles=3))

    _, info = env.reset(seed=123)

    assert info['current_agent'] == 'player_0'
    assert info['current_agent_idx'] == 0
    np.testing.assert_array_equal(info['agent_mask'], [True, True])
    np.testing.assert_array_equal(info['observation.player_0'], [0.0])
    assert info['action.player_0'] == -1

    _, reward, terminated, truncated, info = env.step(
        np.array([1, 0], dtype=np.int64)
    )

    assert reward == 1.0
    assert not terminated
    assert not truncated
    assert env.env.action_log[-1] == ('player_0', 1)
    assert info['current_agent'] == 'player_1'
    assert info['action.player_0'] == 1
    np.testing.assert_array_equal(info['observation.player_1'], [1.0])
    env.close()


def test_aec_wrapper_passes_none_for_dead_agent_turn():
    env = PettingZooAECWrapper(DeadTurnAECEnv())

    _, info = env.reset(seed=123)

    assert info['current_agent'] == 'player_0'
    assert info['terminated.player_0']

    _, _, terminated, truncated, info = env.step(
        np.array([1, 0], dtype=np.int64)
    )

    assert not terminated
    assert not truncated
    assert env.env.action_log[-1] == ('player_0', None)
    assert info['current_agent'] == 'player_1'
    env.close()


def test_world_from_pettingzoo_collects_aec_episodes():
    world = World.from_pettingzoo(
        lambda: TinyAECEnv(max_cycles=4), num_envs=2, api='aec'
    )
    writer = MemoryWriter()
    world.set_policy(MultiAgentRandomPolicy(seed=0))

    world.collect(writer=writer, episodes=3, progress=False)

    assert len(writer.episodes) == 3
    for episode in writer.episodes:
        assert set(
            [
                'agent_mask',
                'current_agent_idx',
                'observation.player_0',
                'action.player_0',
                'reward.player_0',
                'terminated.player_0',
            ]
        ) <= set(episode)
        assert len(episode['reward.player_0']) == 4
    world.close()


def test_world_from_pettingzoo_rejects_unknown_api():
    with pytest.raises(ValueError, match='api'):
        World.from_pettingzoo(lambda: TinyParallelEnv(), num_envs=1, api='bad')
