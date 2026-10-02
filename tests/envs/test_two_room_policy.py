"""Action repetition in the TwoRoom expert, including real collection resets."""

import gymnasium as gym
import numpy as np
import pytest
import torch

pytest.importorskip('pygame')
pytest.importorskip('shapely')

from stable_worldmodel import World
from stable_worldmodel.envs.two_room import ExpertPolicy, TwoRoomEnv


@pytest.fixture(autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def raw_env():
    env = TwoRoomEnv()
    env.reset(seed=0, options={'variation': []})
    yield env
    env.close()


@pytest.fixture
def world():
    env = World('swm/TwoRoom-v1', num_envs=2, add_pixels=False)
    env.reset(seed=0, options={'variation': []})
    yield env
    env.close()


def observation(direction, num_envs=None, step_idx=None):
    """Keep all waypoints in the left room so expected directions are exact."""
    state = np.array([60.0, 60.0], dtype=np.float32)
    goal = state + 20 * np.asarray(direction, dtype=np.float32)
    if num_envs is not None:
        state = np.broadcast_to(state, (num_envs, 1, 2)).copy()
        goal = np.broadcast_to(goal, (num_envs, 1, 2)).copy()
    info = {'state': state, 'goal_state': goal}
    if step_idx is not None:
        info['step_idx'] = np.asarray(step_idx)
    return info


@pytest.mark.parametrize('probability', [-0.1, 1.1, np.nan, np.inf])
def test_invalid_probability(probability):
    with pytest.raises(ValueError, match='action_repeat_prob'):
        ExpertPolicy(action_repeat_prob=probability)


@pytest.mark.parametrize('wrapped', [False, True])
def test_single_env_repeats_and_set_env_clears_history(raw_env, wrapped):
    env = gym.wrappers.TimeLimit(raw_env, 10) if wrapped else raw_env
    policy = ExpertPolicy(action_repeat_prob=1, seed=0)
    policy.set_env(env)
    east = policy.get_action(observation([1, 0]))
    np.testing.assert_array_equal(east, [1, 0])
    np.testing.assert_array_equal(policy.get_action(observation([0, 1])), east)
    np.testing.assert_array_equal(
        policy.get_action(observation([0, 1], step_idx=0)), [0, 1]
    )
    np.testing.assert_array_equal(
        policy.get_action(observation([-1, 0], step_idx=1)), [0, 1]
    )

    env.reset(seed=1)
    policy.set_env(env)
    np.testing.assert_array_equal(
        policy.get_action(observation([-1, 0])), [-1, 0]
    )


def test_zero_probability_preserves_noise_rng_sequence(raw_env):
    policy = ExpertPolicy(action_noise=0.3, seed=8)
    policy.set_env(raw_env)
    expected_rng = np.random.default_rng(8)
    for _ in range(4):
        expected = np.array([1, 0], dtype=np.float32)
        expected += expected_rng.normal(0, 0.3, size=(2,)).astype(np.float32)
        expected = np.clip(expected, -1, 1)
        np.testing.assert_array_equal(
            policy.get_action(observation([1, 0])), expected
        )


def test_repeat_caches_copy_of_clipped_action(raw_env):
    policy = ExpertPolicy(action_repeat_prob=1, action_noise=10, seed=0)
    policy.set_env(raw_env)
    first = policy.get_action(observation([1, 0]))
    np.testing.assert_array_equal(first, [1, -1])
    first[:] = 0  # A caller may reuse its returned array as a work buffer.
    repeated = policy.get_action(observation([0, 1]))
    np.testing.assert_array_equal(repeated, [1, -1])
    assert repeated.dtype == np.float32


def test_set_seed_clears_history_and_replays_noise(raw_env):
    policy = ExpertPolicy(action_repeat_prob=1, action_noise=0.2, seed=9)
    policy.set_env(raw_env)
    first = policy.get_action(observation([1, 0]))
    policy.set_seed(9)
    np.testing.assert_array_equal(
        policy.get_action(observation([1, 0])), first
    )
    policy.set_seed(10)
    fresh = ExpertPolicy(action_repeat_prob=1, action_noise=0.2, seed=10)
    fresh.set_env(raw_env)
    np.testing.assert_array_equal(
        policy.get_action(observation([0, 1])),
        fresh.get_action(observation([0, 1])),
    )


def test_vector_reset_only_clears_reset_environment(world):
    policy = ExpertPolicy(action_repeat_prob=1, seed=0)
    world.set_policy(policy)
    east = policy.get_action(observation([1, 0], 2, [[0], [0]]))
    np.testing.assert_array_equal(east, [[1, 0], [1, 0]])
    mixed = policy.get_action(observation([0, 1], 2, [[0], [3]]))
    np.testing.assert_array_equal(mixed, [[0, 1], [1, 0]])
    np.testing.assert_array_equal(
        policy.get_action(observation([-1, 0], 2, [[1], [4]])), mixed
    )


def test_seeded_repeat_decisions_are_per_env_and_keep_last_output(world):
    policy = ExpertPolicy(action_repeat_prob=0.5, seed=0)
    world.set_policy(policy)
    policy.get_action(observation([1, 0], 2, [[0], [0]]))
    # Seed 0 draws 0.637 then 0.270: only the second environment repeats.
    second = policy.get_action(observation([0, 1], 2, [[1], [1]]))
    np.testing.assert_array_equal(second, [[0, 1], [1, 0]])
    # The next two draws are below 0.5. Repeat each env's last *output*.
    third = policy.get_action(observation([-1, 0], 2, [[2], [2]]))
    np.testing.assert_array_equal(third, second)


@pytest.mark.parametrize('probability', [0, 1])
def test_world_collect_repeats_within_episode_only(tmp_path, probability):
    # The 'hdf5' format needs both h5py and hdf5plugin.
    pytest.importorskip('stable_worldmodel.data.formats.hdf5')
    import h5py

    world = World(
        'swm/TwoRoom-v1', num_envs=1, add_pixels=False, max_episode_steps=3
    )
    policy = ExpertPolicy(
        action_repeat_prob=probability, action_noise=0.2, seed=17
    )
    world.set_policy(policy)
    path = tmp_path / 'two_room.h5'
    try:
        world.collect(
            path=path,
            format='hdf5',
            episodes=2,
            seed=0,
            options={
                'variation': [],
                'state': np.array([60.0, 60.0]),
                'target_state': np.array([60.0, 180.0]),
            },
            progress=False,
        )
    finally:
        world.close()

    with h5py.File(path) as data:
        np.testing.assert_array_equal(data['ep_len'][:], [4, 4])
        # Collection stores the next action beside its starting observation.
        # The terminal observation has no next action and gets a NaN row.
        actions = data['action'][:].reshape(2, 4, 2)[:, :-1]
        for episode in actions:
            if probability == 1:
                np.testing.assert_array_equal(
                    episode, np.broadcast_to(episode[0], episode.shape)
                )
            else:
                assert not np.array_equal(episode[0], episode[1])
        assert not np.array_equal(actions[0, 0], actions[1, 0])
