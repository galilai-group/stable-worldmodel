"""Action repetition in the Piecewise expert, including real collection resets."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from stable_worldmodel import World
from stable_worldmodel.envs.piecewise.expert_policy import (
    ExpertPolicy,
    _get_zone,
)
from stable_worldmodel.envs.piecewise.piecewise_env import PiecewiseEnv


@pytest.fixture(autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def raw_env():
    env = PiecewiseEnv()
    env.reset(seed=0)
    yield env
    env.close()


@pytest.fixture
def world():
    env = World('swm/Piecewise-v0', num_envs=2, add_pixels=False)
    env.reset(seed=0)
    yield env
    env.close()


def observation(direction, num_envs=None, step_idx=None):
    """Use a far diagonal goal so every action component clips to +-1.

    Speed is at most 10.5 and each bias component is at most 4, so
    (1000 - 4) / 10.5 > 1 whatever the zone and variation values.
    """
    state = np.array([112.0, 112.0], dtype=np.float32)
    goal = state + 1000 * np.asarray(direction, dtype=np.float32)
    if num_envs is not None:
        state = np.broadcast_to(state, (num_envs, 1, 2)).copy()
        goal = np.broadcast_to(goal, (num_envs, 1, 2)).copy()
    info = {'state': state, 'goal_state': goal}
    if step_idx is not None:
        info['step_idx'] = np.asarray(step_idx)
    return info


def unclipped_observation(env, action):
    """Place the goal so the noise-free expert action is ``action``."""
    state = np.array([60.0, 60.0], dtype=np.float32)
    speed = float(env.variation_space['agent']['speed'].value.item())
    zone = _get_zone(state, env.grid_n, env.BORDER_SIZE, env.IMG_SIZE)
    bias = env.variation_space['zones'][f'bias_{zone}'].value
    goal = state + np.asarray(action, dtype=np.float32) * speed + bias
    return {'state': state, 'goal_state': goal.astype(np.float32)}


@pytest.mark.parametrize('probability', [-0.1, 1.1, np.nan, np.inf])
def test_invalid_probability(probability):
    with pytest.raises(ValueError, match='action_repeat_prob'):
        ExpertPolicy(action_repeat_prob=probability)


@pytest.mark.parametrize('wrapped', [False, True])
def test_single_env_repeats_and_set_env_clears_history(raw_env, wrapped):
    env = gym.wrappers.TimeLimit(raw_env, 10) if wrapped else raw_env
    policy = ExpertPolicy(action_repeat_prob=1, seed=0)
    policy.set_env(env)
    first = policy.get_action(observation([1, 1]))
    np.testing.assert_array_equal(first, [1, 1])
    np.testing.assert_array_equal(
        policy.get_action(observation([-1, 1])), first
    )
    np.testing.assert_array_equal(
        policy.get_action(observation([-1, 1], step_idx=0)), [-1, 1]
    )
    np.testing.assert_array_equal(
        policy.get_action(observation([-1, -1], step_idx=1)), [-1, 1]
    )

    env.reset(seed=1)
    policy.set_env(env)
    np.testing.assert_array_equal(
        policy.get_action(observation([1, -1])), [1, -1]
    )


def test_world_reset_chooses_fresh_action():
    world = World('swm/Piecewise-v0', num_envs=1, add_pixels=False)
    try:
        policy = ExpertPolicy(action_repeat_prob=1, seed=0)
        world.set_policy(policy)
        world.reset(
            seed=0,
            options={
                'state': np.array([60.0, 60.0]),
                'target_state': np.array([180.0, 180.0]),
            },
        )
        np.testing.assert_array_equal(policy.get_action(world.infos), [[1, 1]])

        world.reset(
            seed=1,
            options={
                'state': np.array([180.0, 180.0]),
                'target_state': np.array([60.0, 60.0]),
            },
        )
        np.testing.assert_array_equal(world.infos['step_idx'], [[0]])
        fresh = ExpertPolicy(action_repeat_prob=1, seed=0)
        fresh.set_env(world.envs)
        expected = fresh.get_action(world.infos)
        np.testing.assert_array_equal(expected, [[-1, -1]])
        np.testing.assert_array_equal(policy.get_action(world.infos), expected)
    finally:
        world.close()


def test_vector_reset_only_clears_reset_environment(world):
    policy = ExpertPolicy(action_repeat_prob=1, seed=0)
    world.set_policy(policy)
    first = policy.get_action(observation([1, 1], 2, [[0], [0]]))
    np.testing.assert_array_equal(first, [[1, 1], [1, 1]])
    mixed = policy.get_action(observation([-1, 1], 2, [[0], [3]]))
    np.testing.assert_array_equal(mixed, [[-1, 1], [1, 1]])
    np.testing.assert_array_equal(
        policy.get_action(observation([-1, -1], 2, [[1], [4]])), mixed
    )


def test_set_env_to_larger_world_clears_history(world):
    policy = ExpertPolicy(action_repeat_prob=0.5, seed=0)
    world.set_policy(policy)
    policy.get_action(observation([1, 1], 2, [[1], [1]]))

    larger = World('swm/Piecewise-v0', num_envs=3, add_pixels=False)
    try:
        larger.set_policy(policy)
        larger.reset(seed=1)
        action = policy.get_action(observation([-1, 1], 3, [[1], [1], [1]]))
    finally:
        larger.close()
    assert action.shape == (3, 2)
    np.testing.assert_array_equal(action, [[-1, 1]] * 3)


def test_set_seed_clears_history_and_replays_noise(raw_env):
    policy = ExpertPolicy(action_repeat_prob=1, action_noise=0.2, seed=9)
    policy.set_env(raw_env)
    info = unclipped_observation(raw_env, [0.2, -0.2])
    first = policy.get_action(info)
    # The noise is visible: the action is not clipped and not noise free.
    assert np.all(np.abs(first) < 1)
    assert not np.allclose(first, [0.2, -0.2], atol=1e-3)
    policy.set_seed(9)
    np.testing.assert_array_equal(policy.get_action(info), first)

    policy.set_seed(10)
    fresh = ExpertPolicy(action_repeat_prob=1, action_noise=0.2, seed=10)
    fresh.set_env(raw_env)
    expected = fresh.get_action(info)
    assert not np.array_equal(expected, first)
    np.testing.assert_array_equal(policy.get_action(info), expected)


def test_repeat_caches_copy_of_clipped_action(raw_env):
    policy = ExpertPolicy(action_repeat_prob=1, seed=0)
    policy.set_env(raw_env)
    first = policy.get_action(observation([1, -1]))
    np.testing.assert_array_equal(first, [1, -1])
    first[:] = 0  # A caller may reuse its returned array as a work buffer.
    repeated = policy.get_action(observation([-1, 1]))
    np.testing.assert_array_equal(repeated, [1, -1])
    assert repeated.dtype == np.float32


def _collect_first_actions(tmp_path, probability, h5py):
    world = World(
        'swm/Piecewise-v0', num_envs=1, add_pixels=False, max_episode_steps=3
    )
    policy = ExpertPolicy(action_repeat_prob=probability, seed=0)
    world.set_policy(policy)
    path = tmp_path / f'piecewise_{probability}.h5'
    try:
        world.collect(
            path=path, format='hdf5', episodes=3, seed=0, progress=False
        )
    finally:
        world.close()
    with h5py.File(path) as data:
        ep_len = data['ep_len'][:]
        actions = data['action'][:]
    assert len(ep_len) == 3
    # Episodes can end early when the agent reaches the target.
    starts = np.concatenate([[0], np.cumsum(ep_len)[:-1]])
    episodes = [actions[s : s + n] for s, n in zip(starts, ep_len)]
    return np.stack([episode[0] for episode in episodes]), episodes


def test_world_collect_starts_each_episode_fresh(tmp_path):
    # The 'hdf5' format needs both h5py and hdf5plugin.
    pytest.importorskip('stable_worldmodel.data.formats.hdf5')
    import h5py

    no_repeat, _ = _collect_first_actions(tmp_path, 0, h5py)
    always_repeat, episodes = _collect_first_actions(tmp_path, 1, h5py)

    # Precondition: the episodes need different first actions.
    assert not np.all(no_repeat == no_repeat[0])
    np.testing.assert_array_equal(always_repeat, no_repeat)
    for episode in episodes:
        # Collection stores a NaN action beside the terminal observation.
        stepped = episode[~np.isnan(episode).any(axis=1)]
        np.testing.assert_array_equal(
            stepped, np.broadcast_to(stepped[0], stepped.shape)
        )
