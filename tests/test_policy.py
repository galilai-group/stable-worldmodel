"""Tests for policy module."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from gymnasium import spaces as gym_spaces

from stable_worldmodel.policy import (
    BasePolicy,
    ExpertPolicy,
    PlanConfig,
    RandomPolicy,
)


###########################
## PlanConfig Tests      ##
###########################


def test_plan_config_properties():
    """Test PlanConfig dataclass properties."""
    config = PlanConfig(horizon=10, receding_horizon=5)
    assert config.horizon == 10
    assert config.receding_horizon == 5
    assert config.history_len == 1
    assert config.action_block == 1
    assert config.warm_start is True
    assert config.plan_len == 10  # horizon * action_block


def test_plan_config_with_action_block():
    """Test PlanConfig with custom action_block."""
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=2)
    assert config.plan_len == 20


def test_plan_config_frozen():
    """Test that PlanConfig is immutable."""
    config = PlanConfig(horizon=10, receding_horizon=5)
    with pytest.raises(Exception):  # FrozenInstanceError
        config.horizon = 20


###########################
## BasePolicy Tests      ##
###########################


def test_base_policy_init():
    """Test BasePolicy initialization."""
    policy = BasePolicy()
    assert policy.env is None
    assert policy.type == "base"


def test_base_policy_kwargs():
    """Test BasePolicy with kwargs."""
    policy = BasePolicy(custom_arg="value", another=42)
    assert policy.custom_arg == "value"
    assert policy.another == 42


def test_base_policy_get_action_not_implemented():
    """Test that BasePolicy.get_action raises NotImplementedError."""
    policy = BasePolicy()
    with pytest.raises(NotImplementedError):
        policy.get_action({})


def test_base_policy_set_env():
    """Test BasePolicy.set_env method."""
    policy = BasePolicy()
    mock_env = MagicMock()
    policy.set_env(mock_env)
    assert policy.env is mock_env


###########################
## RandomPolicy Tests    ##
###########################


def test_random_policy_init():
    """Test RandomPolicy initialization."""
    policy = RandomPolicy()
    assert policy.type == "random"
    assert policy.seed is None


def test_random_policy_with_seed():
    """Test RandomPolicy with seed."""
    policy = RandomPolicy(seed=42)
    assert policy.seed == 42


def test_random_policy_get_action():
    """Test RandomPolicy.get_action method."""
    policy = RandomPolicy()
    mock_env = MagicMock()
    mock_env.action_space.sample.return_value = np.array([0.5, 0.5])
    policy.set_env(mock_env)

    action = policy.get_action({})
    mock_env.action_space.sample.assert_called_once()
    np.testing.assert_array_equal(action, np.array([0.5, 0.5]))


def test_random_policy_set_seed():
    """Test RandomPolicy.set_seed method."""
    policy = RandomPolicy(seed=42)
    mock_env = MagicMock()
    policy.set_env(mock_env)
    policy.set_seed(123)
    mock_env.action_space.seed.assert_called_once_with(123)


def test_random_policy_set_seed_no_env():
    """Test RandomPolicy.set_seed when env is None."""
    policy = RandomPolicy(seed=42)
    # Should not raise
    policy.set_seed(123)


###########################
## ExpertPolicy Tests    ##
###########################


def test_expert_policy_init():
    """Test ExpertPolicy initialization."""
    policy = ExpertPolicy()
    assert policy.type == "expert"


def test_expert_policy_get_action():
    """Test ExpertPolicy.get_action method returns None (placeholder)."""
    policy = ExpertPolicy()
    result = policy.get_action({}, goal_obs={})
    assert result is None
