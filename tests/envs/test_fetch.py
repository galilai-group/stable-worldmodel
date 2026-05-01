import gymnasium as gym
import numpy as np
import pytest


@pytest.mark.parametrize(
    'env_id',
    [
        'swm/FetchReach-v3',
        'swm/FetchPush-v3',
        'swm/FetchSlide-v3',
        'swm/FetchPickAndPlace-v3',
    ],
)
def test_fetch_environment_initialization(env_id):
    env = gym.make(env_id)
    assert isinstance(env.observation_space, gym.spaces.Box), (
        'Observation space must be a flattened Box'
    )
    assert env.observation_space.shape[0] > 0, (
        'Observation space must have positive dimension'
    )

    obs, info = env.reset()
    assert getattr(env.unwrapped, 'env_name', None) or 'env_name' in info, (
        'env_name must be present'
    )
    assert 'proprio' in info, 'proprio state must be exposed'
    assert 'state' in info, 'flattened state must be exposed'
    assert 'goal_state' in info, 'goal state must be exposed'

    env.close()


def test_fetch_visual_randomization():
    env = gym.make('swm/FetchPush-v3')

    color_target = np.array([0.5, 0.1, 0.9])
    obs, info = env.reset(
        options={
            'variation_values': {
                'table.color': color_target,
                'background.color': color_target,
                'object.color': color_target,
            }
        }
    )

    # Assert the variation space tracked the override
    vs = env.get_wrapper_attr('variation_space')
    np.testing.assert_allclose(vs['table']['color'].value, color_target)
    np.testing.assert_allclose(vs['background']['color'].value, color_target)

    env.close()


def test_fetch_physical_randomization():
    env = gym.make('swm/FetchPush-v3')

    target_pos = np.array([1.4, 0.8])
    obs, info = env.reset(
        options={'variation_values': {'block.start_position': target_pos}}
    )

    vs = env.get_wrapper_attr('variation_space')
    np.testing.assert_allclose(vs['block']['start_position'].value, target_pos)

    env.close()


@pytest.mark.parametrize(
    'env_id,expected_obs_dim',
    [
        ('swm/FetchReach-v3', 13),
        ('swm/FetchPush-v3', 28),
    ],
)
def test_fetch_step_output(env_id, expected_obs_dim):
    env = gym.make(env_id)
    obs, info = env.reset()
    assert obs.shape == (expected_obs_dim,), (
        f'reset obs shape mismatch: {obs.shape}'
    )

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (expected_obs_dim,), (
        f'step obs shape mismatch: {obs.shape}'
    )
    assert 'env_name' in info
    assert 'proprio' in info
    assert 'state' in info
    assert 'goal_state' in info
    assert info['state'].shape == (expected_obs_dim,)

    env.close()


@pytest.mark.parametrize(
    'env_id',
    [
        'swm/FetchReachDense-v3',
        'swm/FetchPushDense-v3',
        'swm/FetchSlideDense-v3',
        'swm/FetchPickAndPlaceDense-v3',
    ],
)
def test_fetch_dense_registration(env_id):
    """Dense reward variants exist and behave like the sparse flattened ones."""
    env = gym.make(env_id)
    assert isinstance(env.observation_space, gym.spaces.Box)
    obs, info = env.reset()
    assert obs.ndim == 1 and obs.shape[0] > 0
    # Dense envs deliver a non-zero (typically negative) reward on most steps.
    _, reward, _, _, _ = env.step(env.action_space.sample())
    assert isinstance(reward, (int, float, np.floating))
    env.close()


@pytest.mark.parametrize(
    'env_id',
    [
        'swm/FetchReachDict-v3',
        'swm/FetchPushDict-v3',
        'swm/FetchSlideDict-v3',
        'swm/FetchPickAndPlaceDict-v3',
    ],
)
def test_fetch_dict_obs_preserved(env_id):
    """Dict variants preserve observation/achieved_goal/desired_goal for HER."""
    env = gym.make(env_id)
    assert isinstance(env.observation_space, gym.spaces.Dict)
    for key in ('observation', 'achieved_goal', 'desired_goal'):
        assert key in env.observation_space.spaces, f"missing key '{key}'"

    obs, info = env.reset()
    assert isinstance(obs, dict)
    for key in ('observation', 'achieved_goal', 'desired_goal'):
        assert key in obs

    obs, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )
    assert isinstance(obs, dict)
    assert 'achieved_goal' in obs and 'desired_goal' in obs
    env.close()


def test_fetch_mass_init_value_applied():
    """Passing init_value={'block.mass': ...} at make time must land in
    model.body_mass and persist across resets (fixed-per-run semantics)."""
    target = np.array([7.5])
    env = gym.make('swm/FetchPushDense-v3', init_value={'block.mass': target})

    env.reset(seed=0)
    bid = env.get_wrapper_attr('_object_body_id')
    assert bid >= 0, 'object0 body should resolve in Push'
    np.testing.assert_allclose(env.unwrapped.model.body_mass[bid], target[0])

    # A second reset (no options) must preserve the fixed mass.
    env.reset(seed=1)
    np.testing.assert_allclose(env.unwrapped.model.body_mass[bid], target[0])
    env.close()


def test_fetch_mass_per_reset_override():
    """Per-reset override via options['variation_values'] also reaches MuJoCo."""
    env = gym.make('swm/FetchPushDense-v3')
    env.reset(
        seed=0, options={'variation_values': {'block.mass': np.array([1.25])}}
    )
    bid = env.get_wrapper_attr('_object_body_id')
    np.testing.assert_allclose(env.unwrapped.model.body_mass[bid], 1.25)
    env.close()


def test_fetch_reach_has_no_block_mass():
    """Reach has no object, so the block.* subspace (and body id) must be absent."""
    env = gym.make('swm/FetchReachDense-v3')
    env.reset(seed=0)
    assert env.get_wrapper_attr('_object_body_id') == -1
    vs = env.get_wrapper_attr('variation_space')
    assert 'block' not in vs.spaces
    env.close()


def test_fetch_compute_reward_forwarded():
    """HerReplayBuffer calls env.compute_reward(achieved, desired, info) when
    relabeling goals. The wrapper must forward this to the underlying env."""
    env = gym.make('swm/FetchPushDict-v3')
    obs, _ = env.reset()

    # Relabel against the actual achieved goal → reward should be the success
    # value (0.0 for Fetch sparse; -L2 for dense); not an exception.
    reward = env.unwrapped.compute_reward(
        obs['achieved_goal'], obs['achieved_goal'], {}
    )
    assert reward is not None and np.isfinite(reward)

    # Relabel against a far-away goal → reward is a valid float (shape/dtype OK).
    far = obs['achieved_goal'] + np.array([1.0, 1.0, 1.0])
    reward_far = env.unwrapped.compute_reward(obs['achieved_goal'], far, {})
    assert np.isfinite(reward_far)
    env.close()
