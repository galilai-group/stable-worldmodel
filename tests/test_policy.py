"""Tests for policy module."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from gymnasium import spaces as gym_spaces

from stable_worldmodel.policy import (
    AutoActionableModel,
    AutoCostModel,
    BasePolicy,
    ExpertPolicy,
    FeedForwardPolicy,
    PlanConfig,
    RandomPolicy,
    WorldModelPolicy,
    _load_model_with_attribute,
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


###########################
## _prepare_info Tests   ##
###########################


class MockTransformable:
    """Mock class implementing Transformable protocol."""

    def __init__(self, scale: float = 2.0):
        self.scale = scale

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x / self.scale


def test_prepare_info_basic():
    """Test _prepare_info with basic numpy array conversion."""
    policy = BasePolicy()
    info = {"state": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    result = policy._prepare_info(info)
    assert torch.is_tensor(result["state"])
    torch.testing.assert_close(result["state"], torch.tensor([1.0, 2.0, 3.0]))


def test_prepare_info_with_process():
    """Test _prepare_info with process dict."""
    policy = BasePolicy()
    policy.process = {"state": MockTransformable(scale=2.0)}
    info = {"state": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
    result = policy._prepare_info(info)
    expected = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    torch.testing.assert_close(result["state"], expected)


def test_prepare_info_with_process_3d():
    """Test _prepare_info with process dict and 3D array (flattening)."""
    policy = BasePolicy()
    policy.process = {"state": MockTransformable(scale=2.0)}
    # Shape: (batch=2, time=2, features=3)
    info = {
        "state": np.array(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
            dtype=np.float32,
        )
    }
    result = policy._prepare_info(info)
    # Should preserve shape after processing
    assert result["state"].shape == (2, 2, 3)


def test_prepare_info_process_non_numpy_raises():
    """Test _prepare_info raises ValueError for non-numpy in process."""
    policy = BasePolicy()
    policy.process = {"state": MockTransformable()}
    info = {"state": torch.tensor([1.0, 2.0])}  # Tensor instead of numpy
    with pytest.raises(ValueError, match="Expected numpy array"):
        policy._prepare_info(info)


def test_prepare_info_string_dtype_not_converted():
    """Test _prepare_info doesn't convert string arrays to tensor."""
    policy = BasePolicy()
    info = {"name": np.array(["test", "name"])}
    result = policy._prepare_info(info)
    # String arrays should not be converted
    assert isinstance(result["name"], np.ndarray)


def test_prepare_info_non_numpy_passthrough():
    """Test _prepare_info passes through non-numpy types without process."""
    policy = BasePolicy()
    info = {"tensor": torch.tensor([1.0, 2.0]), "scalar": 42}
    result = policy._prepare_info(info)
    assert torch.is_tensor(result["tensor"])
    assert result["scalar"] == 42


###########################
## FeedForwardPolicy     ##
###########################


class MockActionableModel(torch.nn.Module):
    """Mock model implementing Actionable protocol."""

    def __init__(self, action_dim: int = 2):
        super().__init__()
        self.linear = torch.nn.Linear(4, action_dim)
        self.action_dim = action_dim

    def get_action(self, info: dict) -> torch.Tensor:
        # Return fixed action for testing
        batch_size = info.get("pixels", info.get("goal")).shape[0]
        return torch.zeros(batch_size, self.action_dim)


def test_feedforward_policy_init():
    """Test FeedForwardPolicy initialization."""
    model = MockActionableModel()
    policy = FeedForwardPolicy(model=model)
    assert policy.type == "feed_forward"
    assert policy.model is model
    assert policy.process == {}
    assert policy.transform == {}


def test_feedforward_policy_init_with_process_transform():
    """Test FeedForwardPolicy initialization with process and transform."""
    model = MockActionableModel()
    process = {"action": MockTransformable()}
    transform = {"pixels": lambda x: x}
    policy = FeedForwardPolicy(model=model, process=process, transform=transform)
    assert policy.process is process
    assert policy.transform is transform


def test_feedforward_policy_get_action():
    """Test FeedForwardPolicy.get_action method."""
    model = MockActionableModel(action_dim=2)
    policy = FeedForwardPolicy(model=model)

    mock_env = MagicMock()
    mock_env.action_space.shape = (2,)
    policy.set_env(mock_env)

    info = {
        "pixels": np.random.rand(1, 64, 64, 3).astype(np.float32),
        "goal": np.random.rand(1, 64, 64, 3).astype(np.float32),
    }
    action = policy.get_action(info)
    assert isinstance(action, np.ndarray)
    assert action.shape == (1, 2)


def test_feedforward_policy_get_action_with_process():
    """Test FeedForwardPolicy.get_action with action post-processing."""
    model = MockActionableModel(action_dim=2)
    process = {"action": MockTransformable(scale=2.0)}
    policy = FeedForwardPolicy(model=model, process=process)

    mock_env = MagicMock()
    mock_env.action_space.shape = (2,)
    policy.set_env(mock_env)

    info = {
        "pixels": np.random.rand(1, 64, 64, 3).astype(np.float32),
        "goal": np.random.rand(1, 64, 64, 3).astype(np.float32),
    }
    action = policy.get_action(info)
    # Action should be inverse transformed (divided by 2)
    np.testing.assert_array_equal(action, np.zeros((1, 2)))


def test_feedforward_policy_no_env_raises():
    """Test FeedForwardPolicy env is None by default."""
    model = MockActionableModel()
    policy = FeedForwardPolicy(model=model)
    assert policy.env is None


def test_feedforward_policy_no_goal_raises():
    """Test FeedForwardPolicy.get_action raises without goal."""
    model = MockActionableModel()
    policy = FeedForwardPolicy(model=model)
    mock_env = MagicMock()
    policy.set_env(mock_env)
    with pytest.raises(AssertionError, match="'goal' must be provided"):
        policy.get_action({"pixels": np.array([1.0])})


###########################
## WorldModelPolicy      ##
###########################


class MockSolver:
    """Mock Solver for testing implementing Solver protocol."""

    def __init__(self):
        self.configured = False
        self._action_space = None
        self._n_envs = 1
        self._config = None

    def configure(self, *, action_space, n_envs, config):
        self.configured = True
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config

    @property
    def action_dim(self) -> int:
        return self._action_space.shape[0] * self._config.action_block

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def solve(self, info_dict, init_action=None):
        action_dim = self._action_space.shape[0]
        return {"actions": torch.zeros(self._n_envs, self._config.horizon, action_dim)}

    def __call__(self, info_dict, init_action=None):
        return self.solve(info_dict, init_action)


def test_worldmodel_policy_init():
    """Test WorldModelPolicy initialization."""
    solver = MockSolver()
    config = PlanConfig(horizon=10, receding_horizon=2, action_block=2)
    policy = WorldModelPolicy(solver=solver, config=config)

    assert policy.type == "world_model"
    assert policy.solver is solver
    assert policy.cfg is config
    assert policy.process == {}
    assert policy.transform == {}


def test_worldmodel_policy_flatten_receding_horizon():
    """Test WorldModelPolicy.flatten_receding_horizon property."""
    solver = MockSolver()
    config = PlanConfig(horizon=10, receding_horizon=2, action_block=3)
    policy = WorldModelPolicy(solver=solver, config=config)
    assert policy.flatten_receding_horizon == 6  # 2 * 3


def test_worldmodel_policy_set_env():
    """Test WorldModelPolicy.set_env configures solver."""
    solver = MockSolver()
    config = PlanConfig(horizon=10, receding_horizon=2)
    policy = WorldModelPolicy(solver=solver, config=config)

    mock_env = MagicMock()
    mock_env.num_envs = 4
    mock_env.action_space = gym_spaces.Box(low=-1, high=1, shape=(2,))
    policy.set_env(mock_env)

    assert solver.configured
    assert solver.n_envs == 4
    assert policy._action_buffer is not None


def test_worldmodel_policy_set_env_no_num_envs():
    """Test WorldModelPolicy.set_env with env without num_envs."""
    solver = MockSolver()
    config = PlanConfig(horizon=10, receding_horizon=2)
    policy = WorldModelPolicy(solver=solver, config=config)

    mock_env = MagicMock(spec=[])  # No num_envs attribute
    mock_env.action_space = gym_spaces.Box(low=-1, high=1, shape=(2,))
    policy.set_env(mock_env)

    assert solver.n_envs == 1  # Default


def test_worldmodel_policy_get_action():
    """Test WorldModelPolicy.get_action method."""
    solver = MockSolver()
    config = PlanConfig(horizon=10, receding_horizon=2, action_block=1)
    policy = WorldModelPolicy(solver=solver, config=config)

    mock_env = MagicMock()
    mock_env.num_envs = 1
    mock_env.action_space = gym_spaces.Box(low=-1, high=1, shape=(2,))
    policy.set_env(mock_env)

    info = {
        "pixels": np.random.rand(1, 1, 64, 64, 3).astype(np.float32),
        "goal": np.random.rand(1, 1, 64, 64, 3).astype(np.float32),
    }
    action = policy.get_action(info)
    assert isinstance(action, np.ndarray)
    assert action.shape == (2,)


def test_worldmodel_policy_get_action_uses_buffer():
    """Test WorldModelPolicy.get_action uses action buffer on subsequent calls."""
    solver = MockSolver()
    config = PlanConfig(horizon=10, receding_horizon=2, action_block=1)
    policy = WorldModelPolicy(solver=solver, config=config)

    mock_env = MagicMock()
    mock_env.num_envs = 1
    mock_env.action_space = gym_spaces.Box(low=-1, high=1, shape=(2,))
    policy.set_env(mock_env)

    info = {
        "pixels": np.random.rand(1, 1, 64, 64, 3).astype(np.float32),
        "goal": np.random.rand(1, 1, 64, 64, 3).astype(np.float32),
    }

    # First call should plan
    action1 = policy.get_action(info)
    # Buffer should have receding_horizon - 1 actions left
    assert len(policy._action_buffer) == 1

    # Second call should use buffer (no new planning)
    action2 = policy.get_action(info)
    assert len(policy._action_buffer) == 0


def test_worldmodel_policy_get_action_with_process():
    """Test WorldModelPolicy.get_action with action post-processing."""
    solver = MockSolver()
    config = PlanConfig(horizon=10, receding_horizon=1, action_block=1)
    process = {"action": MockTransformable(scale=2.0)}
    policy = WorldModelPolicy(solver=solver, config=config, process=process)

    mock_env = MagicMock()
    mock_env.num_envs = 1
    mock_env.action_space = gym_spaces.Box(low=-1, high=1, shape=(2,))
    policy.set_env(mock_env)

    info = {
        "pixels": np.random.rand(1, 1, 64, 64, 3).astype(np.float32),
        "goal": np.random.rand(1, 1, 64, 64, 3).astype(np.float32),
    }
    action = policy.get_action(info)
    # Action should be inverse transformed
    np.testing.assert_array_equal(action, np.zeros(2))


def test_worldmodel_policy_no_env_raises():
    """Test WorldModelPolicy.get_action fails without set_env called."""
    solver = MockSolver()
    config = PlanConfig(horizon=10, receding_horizon=2)
    policy = WorldModelPolicy(solver=solver, config=config)
    with pytest.raises(TypeError):
        policy.get_action({"pixels": np.array([1.0]), "goal": np.array([1.0])})


def test_worldmodel_policy_no_pixels_raises():
    """Test WorldModelPolicy.get_action raises without pixels."""
    solver = MockSolver()
    config = PlanConfig(horizon=10, receding_horizon=2)
    policy = WorldModelPolicy(solver=solver, config=config)
    mock_env = MagicMock()
    policy.set_env(mock_env)
    with pytest.raises(AssertionError, match="'pixels' must be provided"):
        policy.get_action({"goal": np.array([1.0])})


def test_worldmodel_policy_no_goal_raises():
    """Test WorldModelPolicy.get_action raises without goal."""
    solver = MockSolver()
    config = PlanConfig(horizon=10, receding_horizon=2)
    policy = WorldModelPolicy(solver=solver, config=config)
    mock_env = MagicMock()
    policy.set_env(mock_env)
    with pytest.raises(AssertionError, match="'goal' must be provided"):
        policy.get_action({"pixels": np.array([1.0])})


def test_worldmodel_policy_warm_start():
    """Test WorldModelPolicy warm start feature."""
    solver = MockSolver()
    config = PlanConfig(horizon=10, receding_horizon=2, action_block=1, warm_start=True)
    policy = WorldModelPolicy(solver=solver, config=config)

    mock_env = MagicMock()
    mock_env.num_envs = 1
    mock_env.action_space = gym_spaces.Box(low=-1, high=1, shape=(2,))
    policy.set_env(mock_env)

    info = {
        "pixels": np.random.rand(1, 1, 64, 64, 3).astype(np.float32),
        "goal": np.random.rand(1, 1, 64, 64, 3).astype(np.float32),
    }

    # First call triggers planning
    policy.get_action(info)
    # After first plan, _next_init should be set (warm start)
    assert policy._next_init is not None


def test_worldmodel_policy_no_warm_start():
    """Test WorldModelPolicy without warm start."""
    solver = MockSolver()
    config = PlanConfig(
        horizon=10, receding_horizon=2, action_block=1, warm_start=False
    )
    policy = WorldModelPolicy(solver=solver, config=config)

    mock_env = MagicMock()
    mock_env.num_envs = 1
    mock_env.action_space = gym_spaces.Box(low=-1, high=1, shape=(2,))
    policy.set_env(mock_env)

    info = {
        "pixels": np.random.rand(1, 1, 64, 64, 3).astype(np.float32),
        "goal": np.random.rand(1, 1, 64, 64, 3).astype(np.float32),
    }

    policy.get_action(info)
    # Without warm start, _next_init should be None
    assert policy._next_init is None


###########################
## Auto Loading Tests    ##
###########################


class MockModuleWithGetAction(torch.nn.Module):
    """Mock module with get_action method."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def get_action(self, info):
        return torch.zeros(2)


class MockModuleWithGetCost(torch.nn.Module):
    """Mock module with get_cost method."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def get_cost(self, info):
        return torch.zeros(1)


class MockParentModule(torch.nn.Module):
    """Mock parent module containing child with attribute."""

    def __init__(self, child):
        super().__init__()
        self.encoder = torch.nn.Linear(10, 10)
        self.child = child


@pytest.fixture
def mock_checkpoint_dir(tmp_path):
    """Create a temporary checkpoint directory."""
    return tmp_path


def test_load_model_with_attribute_direct(mock_checkpoint_dir):
    """Test _load_model_with_attribute finds attribute directly from directory."""
    model = MockModuleWithGetAction()
    ckpt_path = mock_checkpoint_dir / "direct_object.ckpt"
    torch.save(model, ckpt_path)

    # Pass directory - it will find the *_object.ckpt file
    result = _load_model_with_attribute(str(mock_checkpoint_dir), "get_action")
    assert hasattr(result, "get_action")


def test_load_model_with_attribute_nested(mock_checkpoint_dir):
    """Test _load_model_with_attribute finds nested attribute."""
    child = MockModuleWithGetAction()
    parent = MockParentModule(child)
    # Create a subdirectory for this test
    nested_dir = mock_checkpoint_dir / "nested_test"
    nested_dir.mkdir()
    ckpt_path = nested_dir / "model_object.ckpt"
    torch.save(parent, ckpt_path)

    result = _load_model_with_attribute(str(nested_dir), "get_action")
    assert hasattr(result, "get_action")


def test_load_model_with_attribute_from_dir(mock_checkpoint_dir):
    """Test _load_model_with_attribute loads from directory."""
    model = MockModuleWithGetAction()
    subdir = mock_checkpoint_dir / "from_dir"
    subdir.mkdir()
    ckpt_path = subdir / "model_object.ckpt"
    torch.save(model, ckpt_path)

    result = _load_model_with_attribute(str(subdir), "get_action")
    assert hasattr(result, "get_action")


def test_load_model_with_attribute_not_found(mock_checkpoint_dir):
    """Test _load_model_with_attribute raises when attribute not found."""
    model = torch.nn.Linear(4, 2)  # No get_action
    subdir = mock_checkpoint_dir / "no_attr"
    subdir.mkdir()
    ckpt_path = subdir / "test_object.ckpt"
    torch.save(model, ckpt_path)

    with pytest.raises(RuntimeError, match="No module with 'get_action' found"):
        _load_model_with_attribute(str(subdir), "get_action")


def test_load_model_with_attribute_cache_dir(mock_checkpoint_dir):
    """Test _load_model_with_attribute uses cache_dir."""
    model = MockModuleWithGetAction()
    run_name = "my_model"
    run_dir = mock_checkpoint_dir / run_name
    run_dir.mkdir()
    ckpt_path = run_dir / "epoch_0_object.ckpt"
    torch.save(model, ckpt_path)

    result = _load_model_with_attribute(
        run_name, "get_action", cache_dir=mock_checkpoint_dir
    )
    assert hasattr(result, "get_action")


def test_auto_actionable_model(mock_checkpoint_dir):
    """Test AutoActionableModel function."""
    model = MockModuleWithGetAction()
    subdir = mock_checkpoint_dir / "actionable"
    subdir.mkdir()
    ckpt_path = subdir / "test_object.ckpt"
    torch.save(model, ckpt_path)

    result = AutoActionableModel(str(subdir))
    assert hasattr(result, "get_action")


def test_auto_cost_model(mock_checkpoint_dir):
    """Test AutoCostModel function."""
    model = MockModuleWithGetCost()
    subdir = mock_checkpoint_dir / "costable"
    subdir.mkdir()
    ckpt_path = subdir / "test_object.ckpt"
    torch.save(model, ckpt_path)

    result = AutoCostModel(str(subdir))
    assert hasattr(result, "get_cost")
