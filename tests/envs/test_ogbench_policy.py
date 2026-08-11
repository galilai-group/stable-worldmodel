"""Tests for OGBench environment expert policy."""

from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest

from stable_worldmodel.world.world import _slice_policy_info


pytest.importorskip('ogbench')

from stable_worldmodel.envs.ogbench import ExpertPolicy  # noqa: E402


NUM_ENVS = 4
ACTION_DIM = 5
MASK = np.array([False, True, False, True])


################################
## ExpertPolicy Tests         ##
################################


class MockCubeEnv:
    """Only what the policy reads off an env: action space and cube count."""

    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(ACTION_DIM,), dtype=np.float32
        )
        self._env_type = 'single'  # read by _get_cube_stack_prob

    @property
    def unwrapped(self):
        return self


class MarkerOracle:
    """Oracle whose action identifies the environment it was asked for."""

    def __init__(self, env_idx):
        self.env_idx = env_idx
        self.done = False

    def reset(self, *args, **kwargs):
        pass

    def select_action(self, *args, **kwargs):
        # Inside [-1, 1]: the policy clips, which would erase larger tags.
        return np.full(ACTION_DIM, self.env_idx / 10.0, dtype=np.float32)


@pytest.fixture
def masked_policy():
    """ExpertPolicy over mock envs, with every oracle replaced by a marker."""
    envs = [MockCubeEnv() for _ in range(NUM_ENVS)]

    vec_env = MagicMock()
    vec_env.spec = None
    vec_env.envs = envs
    vec_env.num_envs = NUM_ENVS
    vec_env.action_space = gym.spaces.Box(
        low=-1, high=1, shape=(NUM_ENVS, ACTION_DIM), dtype=np.float32
    )

    policy = ExpertPolicy(
        policy_type='markov_oracle',
        action_noise=0.0,
        p_random_action=0.0,
        seed=0,
    )
    # set_env is skipped: it builds real oracles against a live MuJoCo model.
    policy.env = vec_env
    policy._p_stack = np.zeros(NUM_ENVS)
    policy._xi = np.zeros(NUM_ENVS)
    policy._agents = [None] * NUM_ENVS
    policy._oracle_agents = {
        'cube': [MarkerOracle(i) for i in range(NUM_ENVS)]
    }
    return policy


def _info(num_rows):
    """Minimal info dict; `step_idx == 0` makes the policy adopt the oracles."""
    return {
        'step_idx': np.zeros((num_rows, 1), dtype=np.int64),
        'privileged/target_task': np.array([['cube']] * num_rows),
    }


def test_expert_policy_get_action_with_env_mask_returns_selected_rows(
    masked_policy,
):
    """Each returned row must come from its own environment."""
    indices = np.flatnonzero(MASK)
    ready = _slice_policy_info(
        masked_policy, _info(NUM_ENVS), indices, NUM_ENVS
    )

    actions = masked_policy.get_action(ready, env_mask=MASK)

    assert actions.shape == (len(indices), ACTION_DIM)
    for row, env_idx in enumerate(indices):
        np.testing.assert_allclose(actions[row], env_idx / 10.0, atol=1e-6)


def test_expert_policy_get_action_without_mask_acts_for_every_env(
    masked_policy,
):
    """`env_mask=None` keeps the original full-width behaviour."""
    actions = masked_policy.get_action(_info(NUM_ENVS))

    assert actions.shape == (NUM_ENVS, ACTION_DIM)
    for env_idx in range(NUM_ENVS):
        np.testing.assert_allclose(actions[env_idx], env_idx / 10.0, atol=1e-6)


def test_expert_policy_keeps_per_env_state_on_a_masked_call(masked_policy):
    """Per-env buffers stay keyed by absolute index, not by row."""
    indices = np.flatnonzero(MASK)
    ready = _slice_policy_info(
        masked_policy, _info(NUM_ENVS), indices, NUM_ENVS
    )

    masked_policy.get_action(ready, env_mask=MASK)

    assert len(masked_policy._agents) == NUM_ENVS
    for env_idx in indices:
        assert masked_policy._agents[env_idx].env_idx == env_idx
    # Envs outside the mask were left untouched.
    for env_idx in set(range(NUM_ENVS)) - set(indices.tolist()):
        assert masked_policy._agents[env_idx] is None
