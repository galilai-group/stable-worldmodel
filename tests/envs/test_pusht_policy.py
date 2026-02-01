"""Tests for PushT environment expert policy."""

from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest

from stable_worldmodel.envs.pusht.expert_policy import WeakPolicy


################################
## WeakPolicy Tests           ##
################################


@pytest.fixture
def mock_single_env():
    """Create a mock single environment for WeakPolicy testing."""
    mock_env = MagicMock(spec=gym.Env)
    mock_env.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,))
    mock_env.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
    mock_env.spec = MagicMock()
    mock_env.spec.id = "swm/PushT-v0"
    return mock_env


@pytest.fixture
def mock_vectorized_env():
    """Create a mock vectorized environment for WeakPolicy testing."""
    mock_env = MagicMock()
    mock_env.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, 4))
    mock_env.action_space = gym.spaces.Box(low=-1, high=1, shape=(3, 2))
    mock_env.spec = None  # Vectorized envs typically don't have spec

    # Create mock sub-environments with spec
    mock_sub_envs = []
    for _ in range(3):
        sub_env = MagicMock()
        sub_env.spec = MagicMock()
        sub_env.spec.id = "swm/PushT-v0"
        mock_sub_envs.append(sub_env)

    mock_env.envs = mock_sub_envs
    return mock_env


def test_weak_policy_init_default():
    """Test WeakPolicy initialization with default values."""
    policy = WeakPolicy()
    assert policy.dist_constraint == 100
    assert policy.discrete is False


def test_weak_policy_init_custom_constraint():
    """Test WeakPolicy initialization with custom dist_constraint."""
    policy = WeakPolicy(dist_constraint=50)
    assert policy.dist_constraint == 50


def test_weak_policy_init_invalid_constraint():
    """Test WeakPolicy raises assertion for non-positive dist_constraint."""
    with pytest.raises(AssertionError):
        WeakPolicy(dist_constraint=0)

    with pytest.raises(AssertionError):
        WeakPolicy(dist_constraint=-10)


def test_weak_policy_set_env_single_env(mock_single_env):
    """Test WeakPolicy.set_env with a single environment that has spec."""
    policy = WeakPolicy()
    policy.set_env(mock_single_env)

    assert policy.env is mock_single_env
    assert policy.discrete is False


def test_weak_policy_set_env_discrete_env():
    """Test WeakPolicy.set_env correctly detects discrete action space."""
    mock_env = MagicMock(spec=gym.Env)
    mock_env.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,))
    mock_env.action_space = gym.spaces.Discrete(9)
    mock_env.spec = MagicMock()
    mock_env.spec.id = "swm/PushTDiscrete-v0"

    policy = WeakPolicy()
    policy.set_env(mock_env)

    assert policy.discrete is True


def test_weak_policy_set_env_vectorized_env(mock_vectorized_env):
    """Test WeakPolicy.set_env with a vectorized environment (spec is None at top level)."""
    policy = WeakPolicy()
    policy.set_env(mock_vectorized_env)

    assert policy.env is mock_vectorized_env
    assert policy.discrete is False


def test_weak_policy_set_env_vectorized_env_discrete():
    """Test WeakPolicy.set_env with vectorized env having discrete action space."""
    mock_env = MagicMock()
    mock_env.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, 4))
    mock_env.action_space = gym.spaces.MultiDiscrete([9, 9, 9])
    mock_env.spec = None

    mock_sub_env = MagicMock()
    mock_sub_env.spec = MagicMock()
    mock_sub_env.spec.id = "swm/PushTDiscrete-v0"
    mock_env.envs = [mock_sub_env, mock_sub_env, mock_sub_env]

    policy = WeakPolicy()
    policy.set_env(mock_env)

    assert policy.discrete is True


def test_weak_policy_set_env_no_spec_no_envs():
    """Test WeakPolicy.set_env raises assertion when no spec is found."""
    mock_env = MagicMock()
    mock_env.spec = None
    mock_env.envs = None

    policy = WeakPolicy()
    with pytest.raises(AssertionError):
        policy.set_env(mock_env)


def test_weak_policy_set_env_empty_envs_list():
    """Test WeakPolicy.set_env raises assertion when envs list is empty."""
    mock_env = MagicMock()
    mock_env.spec = None
    mock_env.envs = []

    policy = WeakPolicy()
    with pytest.raises(AssertionError):
        policy.set_env(mock_env)


def test_weak_policy_set_env_wrong_environment():
    """Test WeakPolicy.set_env raises assertion for non-PushT environment."""
    mock_env = MagicMock(spec=gym.Env)
    mock_env.spec = MagicMock()
    mock_env.spec.id = "CartPole-v1"

    policy = WeakPolicy()
    with pytest.raises(AssertionError):
        policy.set_env(mock_env)


def test_weak_policy_set_env_vectorized_wrong_environment():
    """Test WeakPolicy.set_env raises assertion for vectorized non-PushT environment."""
    mock_env = MagicMock()
    mock_env.spec = None

    mock_sub_env = MagicMock()
    mock_sub_env.spec = MagicMock()
    mock_sub_env.spec.id = "CartPole-v1"
    mock_env.envs = [mock_sub_env]

    policy = WeakPolicy()
    with pytest.raises(AssertionError):
        policy.set_env(mock_env)


def test_weak_policy_set_env_sub_env_no_spec():
    """Test WeakPolicy.set_env raises assertion when sub-env has no spec."""
    mock_env = MagicMock()
    mock_env.spec = None

    mock_sub_env = MagicMock()
    mock_sub_env.spec = None
    mock_env.envs = [mock_sub_env]

    policy = WeakPolicy()
    with pytest.raises(AssertionError):
        policy.set_env(mock_env)
