"""Tests for RandomSolver class."""

from unittest.mock import MagicMock

import numpy as np
import torch
from gymnasium import spaces as gym_spaces

from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver.random import RandomSolver


###########################
## Initialization Tests  ##
###########################


def test_random_solver_init():
    """Test RandomSolver initialization creates unconfigured instance."""
    solver = RandomSolver()
    assert solver._configured is False
    assert solver._action_space is None
    assert solver._n_envs is None
    assert solver._action_dim is None
    assert solver._config is None


def test_random_solver_properties_before_configure():
    """Test that properties return None before configuration."""
    solver = RandomSolver()
    assert solver.n_envs is None
    # action_dim and horizon will raise AttributeError since config is None


###########################
## Configuration Tests   ##
###########################


def test_random_solver_configure_box_action_space():
    """Test configuration with continuous Box action space."""
    solver = RandomSolver()
    action_space = gym_spaces.Box(low=-1, high=1, shape=(4, 3), dtype=np.float32)
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    assert solver._configured is True
    assert solver._action_space == action_space
    assert solver._n_envs == 2
    assert solver._config == config
    assert solver._action_dim == 3  # shape[1:]


def test_random_solver_configure_discrete_action_space():
    """Test configuration with discrete action space."""
    solver = RandomSolver()
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=8, receding_horizon=4, action_block=2)

    solver.configure(action_space=action_space, n_envs=3, config=config)

    assert solver._configured is True
    assert solver.n_envs == 3
    assert solver._action_dim == 1


def test_random_solver_configure_multi_discrete_action_space():
    """Test configuration with multi-discrete action space."""
    solver = RandomSolver()
    action_space = gym_spaces.MultiDiscrete([5, 3, 2])
    config = PlanConfig(horizon=8, receding_horizon=4, action_block=2)

    solver.configure(action_space=action_space, n_envs=3, config=config)

    assert solver._configured is True
    assert solver.n_envs == 3
    assert solver._action_dim == 1  # Empty shape[1:] = 1


def test_random_solver_configure_multi_env():
    """Test configuration with multiple environments."""
    solver = RandomSolver()
    action_space = gym_spaces.Box(low=-1, high=1, shape=(10, 2), dtype=np.float32)
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=16, config=config)

    assert solver.n_envs == 16


def test_random_solver_properties_after_configure():
    """Test that properties work correctly after configuration."""
    solver = RandomSolver()
    action_space = gym_spaces.Box(low=-1, high=1, shape=(4, 3), dtype=np.float32)
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=2)

    solver.configure(action_space=action_space, n_envs=4, config=config)

    assert solver.n_envs == 4
    assert solver.action_dim == 6  # 3 * 2 (base_dim * action_block)
    assert solver.horizon == 10


def test_random_solver_action_dim_with_action_block():
    """Test action_dim calculation with different action_block values."""
    solver = RandomSolver()
    action_space = gym_spaces.Box(low=-1, high=1, shape=(4, 5), dtype=np.float32)

    # Test action_block = 1
    config1 = PlanConfig(horizon=10, receding_horizon=5, action_block=1)
    solver.configure(action_space=action_space, n_envs=1, config=config1)
    assert solver.action_dim == 5

    # Test action_block = 3
    solver2 = RandomSolver()
    config2 = PlanConfig(horizon=10, receding_horizon=5, action_block=3)
    solver2.configure(action_space=action_space, n_envs=1, config=config2)
    assert solver2.action_dim == 15  # 5 * 3


###########################
## Solve Method Tests    ##
###########################


def test_random_solver_solve_full_horizon():
    """Test solving generates full action sequence."""
    solver = RandomSolver()
    action_space = gym_spaces.Box(low=-1, high=1, shape=(4, 3), dtype=np.float32)
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=1)

    solver.configure(action_space=action_space, n_envs=4, config=config)
    result = solver.solve(info_dict={})

    assert "actions" in result
    actions = result["actions"]
    assert isinstance(actions, torch.Tensor)
    assert actions.shape == (4, 10, 3)  # (n_envs, horizon, action_dim)


def test_random_solver_solve_with_action_block():
    """Test solving with action blocking."""
    solver = RandomSolver()
    action_space = gym_spaces.Box(low=-1, high=1, shape=(2, 2), dtype=np.float32)
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=3)

    solver.configure(action_space=action_space, n_envs=2, config=config)
    result = solver.solve(info_dict={})

    actions = result["actions"]
    assert actions.shape == (2, 5, 6)  # (n_envs, horizon, action_dim=2*3)


def test_random_solver_solve_single_env():
    """Test solving with a single environment."""
    solver = RandomSolver()
    action_space = gym_spaces.Box(low=-1, high=1, shape=(1, 4), dtype=np.float32)
    config = PlanConfig(horizon=7, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=1, config=config)
    result = solver.solve(info_dict={})

    actions = result["actions"]
    assert actions.shape == (1, 7, 4)


def test_random_solver_solve_ignores_info_dict():
    """Test that solve ignores info_dict content."""
    solver = RandomSolver()
    action_space = gym_spaces.Box(low=-1, high=1, shape=(2, 3), dtype=np.float32)
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    # Should produce same shape regardless of info_dict content
    result1 = solver.solve(info_dict={})
    result2 = solver.solve(info_dict={"state": torch.randn(2, 10), "obs": None})

    assert result1["actions"].shape == result2["actions"].shape == (2, 5, 3)


def test_random_solver_solve_samples_from_action_space():
    """Test that solve actually calls action_space.sample()."""
    solver = RandomSolver()
    action_space = MagicMock()
    action_space.shape = (2, 3)
    action_space.sample.return_value = np.random.randn(2, 3)

    config = PlanConfig(horizon=4, receding_horizon=2, action_block=2)
    solver.configure(action_space=action_space, n_envs=2, config=config)

    solver.solve(info_dict={})

    # Should call sample 4 * 2 = 8 times (horizon * action_block)
    assert action_space.sample.call_count == 8


###########################
## Warm-Start Tests      ##
###########################


def test_random_solver_solve_with_init_action():
    """Test warm-starting with partial action sequence."""
    solver = RandomSolver()
    action_space = gym_spaces.Box(low=-1, high=1, shape=(2, 3), dtype=np.float32)
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    # Provide first 3 steps
    init_action = torch.randn(2, 3, 3)
    result = solver.solve(info_dict={}, init_action=init_action)

    actions = result["actions"]
    assert actions.shape == (2, 10, 3)
    # First 3 steps should be from init_action
    assert torch.allclose(actions[:, :3, :], init_action)


def test_random_solver_solve_warm_start_fills_remaining():
    """Test that warm-start only samples remaining actions."""
    solver = RandomSolver()
    action_space = MagicMock()
    action_space.shape = (2, 4)
    action_space.sample.return_value = np.random.randn(2, 4)

    config = PlanConfig(horizon=10, receding_horizon=5, action_block=1)
    solver.configure(action_space=action_space, n_envs=2, config=config)

    # Provide 7 steps, need 3 more
    init_action = torch.randn(2, 7, 4)
    result = solver.solve(info_dict={}, init_action=init_action)

    # Should only sample 3 times (10 - 7)
    assert action_space.sample.call_count == 3
    assert result["actions"].shape == (2, 10, 4)


def test_random_solver_solve_warm_start_full_horizon():
    """Test warm-starting with complete action sequence."""
    solver = RandomSolver()
    action_space = MagicMock()
    action_space.shape = (2, 3)
    action_space.sample.return_value = np.random.randn(2, 3)

    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)
    solver.configure(action_space=action_space, n_envs=2, config=config)

    # Provide all 5 steps
    init_action = torch.randn(2, 5, 3)
    result = solver.solve(info_dict={}, init_action=init_action)

    # Should not sample anything
    assert action_space.sample.call_count == 0
    assert torch.equal(result["actions"], init_action)


def test_random_solver_solve_warm_start_with_action_block():
    """Test warm-starting with action blocking."""
    solver = RandomSolver()
    action_space = MagicMock()
    action_space.shape = (2, 2)
    action_space.sample.return_value = np.random.randn(2, 2)

    config = PlanConfig(horizon=6, receding_horizon=3, action_block=2)
    solver.configure(action_space=action_space, n_envs=2, config=config)

    # Provide 4 steps, need 2 more with action_block=2
    init_action = torch.randn(2, 4, 4)  # action_dim = 2 * 2
    result = solver.solve(info_dict={}, init_action=init_action)

    # Should sample 4 times (2 remaining * 2 action_block)
    assert action_space.sample.call_count == 4
    assert result["actions"].shape == (2, 6, 4)


###########################
## Callable Tests        ##
###########################


def test_random_solver_callable():
    """Test that solver is callable via __call__."""
    solver = RandomSolver()
    action_space = gym_spaces.Box(low=-1, high=1, shape=(2, 3), dtype=np.float32)
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    # Test both calling methods produce same shape
    result1 = solver(info_dict={})
    result2 = solver.solve(info_dict={})

    assert result1["actions"].shape == result2["actions"].shape == (2, 5, 3)


def test_random_solver_callable_with_kwargs():
    """Test callable interface with keyword arguments."""
    solver = RandomSolver()
    action_space = gym_spaces.Box(low=-1, high=1, shape=(2, 3), dtype=np.float32)
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    init_action = torch.randn(2, 2, 3)
    result = solver(info_dict={}, init_action=init_action)

    assert result["actions"].shape == (2, 5, 3)
    assert torch.allclose(result["actions"][:, :2, :], init_action)


###########################
## Edge Cases & Errors   ##
###########################


def test_random_solver_solve_empty_init_action():
    """Test solving with empty init_action tensor."""
    solver = RandomSolver()
    action_space = gym_spaces.Box(low=-1, high=1, shape=(2, 3), dtype=np.float32)
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    # Empty tensor (0 steps)
    init_action = torch.zeros((2, 0, 3))
    result = solver.solve(info_dict={}, init_action=init_action)

    assert result["actions"].shape == (2, 5, 3)


def test_random_solver_horizon_1():
    """Test solver with horizon of 1."""
    solver = RandomSolver()
    action_space = gym_spaces.Box(low=-1, high=1, shape=(2, 3), dtype=np.float32)
    config = PlanConfig(horizon=1, receding_horizon=1, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)
    result = solver.solve(info_dict={})

    assert result["actions"].shape == (2, 1, 3)


def test_random_solver_large_horizon():
    """Test solver with large horizon."""
    solver = RandomSolver()
    action_space = gym_spaces.Box(low=-1, high=1, shape=(2, 3), dtype=np.float32)
    config = PlanConfig(horizon=100, receding_horizon=50, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)
    result = solver.solve(info_dict={})

    assert result["actions"].shape == (2, 100, 3)


def test_random_solver_many_envs():
    """Test solver with many parallel environments."""
    solver = RandomSolver()
    action_space = gym_spaces.Box(low=-1, high=1, shape=(64, 4), dtype=np.float32)
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=1)

    solver.configure(action_space=action_space, n_envs=64, config=config)
    result = solver.solve(info_dict={})

    assert result["actions"].shape == (64, 10, 4)


def test_random_solver_multidimensional_action():
    """Test solver with multi-dimensional action space."""
    solver = RandomSolver()
    action_space = gym_spaces.Box(low=-1, high=1, shape=(2, 3, 4, 5), dtype=np.float32)
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)
    result = solver.solve(info_dict={})

    # action_dim should be product of shape[1:] = 3*4*5 = 60
    assert result["actions"].shape == (2, 5, 60)


###########################
## Integration Tests     ##
###########################


def test_random_solver_deterministic_with_seed():
    """Test that results are reproducible with numpy seed."""
    solver1 = RandomSolver()
    solver2 = RandomSolver()
    action_space = gym_spaces.Box(low=-1, high=1, shape=(2, 3), dtype=np.float32)
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    solver1.configure(action_space=action_space, n_envs=2, config=config)
    solver2.configure(action_space=action_space, n_envs=2, config=config)

    # Set seed and sample
    action_space.seed(42)
    result1 = solver1.solve(info_dict={})

    # Reset seed and sample again
    action_space.seed(42)
    result2 = solver2.solve(info_dict={})

    # Results should be identical
    assert torch.allclose(result1["actions"], result2["actions"])


def test_random_solver_multiple_solves():
    """Test multiple solve calls produce different results."""
    solver = RandomSolver()
    action_space = gym_spaces.Box(low=-1, high=1, shape=(2, 3), dtype=np.float32)
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    result1 = solver.solve(info_dict={})
    result2 = solver.solve(info_dict={})

    # Results should be different (with very high probability)
    assert not torch.allclose(result1["actions"], result2["actions"])


def test_random_solver_receding_horizon_pattern():
    """Test typical receding horizon planning pattern."""
    solver = RandomSolver()
    action_space = gym_spaces.Box(low=-1, high=1, shape=(4, 2), dtype=np.float32)
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=1)

    solver.configure(action_space=action_space, n_envs=4, config=config)

    # First planning step
    result1 = solver.solve(info_dict={})
    actions1 = result1["actions"]
    assert actions1.shape == (4, 10, 2)

    # Execute first 5 steps, keep remaining 5
    remaining = actions1[:, 5:, :]
    assert remaining.shape == (4, 5, 2)

    # Second planning step with warm-start
    result2 = solver.solve(info_dict={}, init_action=remaining)
    actions2 = result2["actions"]
    assert actions2.shape == (4, 10, 2)

    # First 5 steps of second plan should match remaining from first plan
    assert torch.allclose(actions2[:, :5, :], remaining)


def test_random_solver_respects_action_space_bounds():
    """Test that sampled actions respect action space bounds."""
    solver = RandomSolver()
    low, high = -2.0, 3.0
    action_space = gym_spaces.Box(low=low, high=high, shape=(4, 5), dtype=np.float32)
    config = PlanConfig(horizon=20, receding_horizon=10, action_block=1)

    solver.configure(action_space=action_space, n_envs=4, config=config)
    result = solver.solve(info_dict={})

    actions = result["actions"]
    # Check all actions are within bounds
    assert torch.all(actions >= low)
    assert torch.all(actions <= high)
