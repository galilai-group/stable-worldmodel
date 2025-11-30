"""Tests for RandomSolver class."""

import numpy as np
import torch
from gymnasium import spaces as gym_spaces

from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver.discrete_solvers import PGDSolver


class DummyCostModel:
    """Simple Costable implementation for tests."""

    def get_cost(
        self,
        info_dict: dict,
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        w = torch.randn_like(action_candidates)
        # Quadratic cost: sum over horizon and action dims
        cost = (action_candidates - w).pow(2).sum(dim=(-1, -2))
        # cost = action_candidates.pow(2).sum(dim=(-1, -2))

        # shape: (batch_envs, num_samples)
        return cost


###########################
## Initialization Tests  ##
###########################


def test_pgd_solver_init():
    """Test PGDSolver initialization creates unconfigured instance."""
    dummy_model = DummyCostModel()

    solver = PGDSolver(model=dummy_model, n_steps=10)
    assert solver._action_simplex_dim is None
    assert solver._n_envs is None
    assert solver._action_dim is None
    assert solver._config is None
    assert solver._configured is False


def test_pgd_solver_properties_before_configure():
    """Test that properties return None before configuration."""
    dummy_model = DummyCostModel()

    solver = PGDSolver(model=dummy_model, n_steps=10)
    assert solver.n_envs is None
    # action_dim and horizon will raise AttributeError since config is None


###########################
## Configuration Tests   ##
###########################


def test_pgd_solver_configure_discrete_action_space():
    """Test configuration with discrete action space."""
    dummy_model = DummyCostModel()
    solver = PGDSolver(model=dummy_model, n_steps=10)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=8, receding_horizon=4, action_block=2)

    solver.configure(action_space=action_space, n_envs=3, config=config)

    assert solver._configured is True
    assert solver._action_space is action_space
    assert solver.n_envs == 3
    assert solver._action_dim == 1
    assert solver._action_simplex_dim == 5


def test_pgd_solver_configure_multi_env():
    """Test configuration with multiple environments."""
    dummy_model = DummyCostModel()
    solver = PGDSolver(model=dummy_model, n_steps=10)
    action_space = gym_spaces.Discrete(10)
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=16, config=config)

    assert solver.n_envs == 16


def test_pgd_solver_properties_after_configure():
    """Test that properties work correctly after configuration."""
    dummy_model = DummyCostModel()
    solver = PGDSolver(model=dummy_model, n_steps=10)
    # action_space = gym_spaces.Box(low=-1, high=1, shape=(4, 3), dtype=np.float32)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=2)

    solver.configure(action_space=action_space, n_envs=4, config=config)

    assert solver.n_envs == 4
    assert solver.action_dim == 2  # 1 * 2 (base_dim * action_block)
    assert solver.action_simplex_dim == 10  # 5 * 2
    assert solver.horizon == 10


def test_pgd_solver_action_dim_with_action_block():
    """Test action_dim calculation with different action_block values."""
    dummy_model = DummyCostModel()
    solver = PGDSolver(model=dummy_model, n_steps=10)
    action_space = gym_spaces.Discrete(5)

    # Test action_block = 1
    config1 = PlanConfig(horizon=10, receding_horizon=5, action_block=1)
    solver.configure(action_space=action_space, n_envs=1, config=config1)
    assert solver.action_dim == 1  # 1 * 1 (base_dim * action_block)
    assert solver.action_simplex_dim == 5  # 5 * 1

    # Test action_block = 3
    solver2 = PGDSolver(model=dummy_model, n_steps=10)
    config2 = PlanConfig(horizon=10, receding_horizon=5, action_block=3)
    solver2.configure(action_space=action_space, n_envs=1, config=config2)
    assert solver2.action_dim == 3  # 1 * 3 (base_dim * action_block)
    assert solver2.action_simplex_dim == 15  # 5 * 3


###########################
## Solve Method Tests    ##
###########################


def test_pgd_solver_solve_full_horizon():
    """Test solving generates full action sequence."""
    dummy_model = DummyCostModel()
    solver = PGDSolver(model=dummy_model, n_steps=10)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=1)

    solver.configure(action_space=action_space, n_envs=4, config=config)
    result = solver.solve(info_dict={})

    assert "actions" in result
    actions = result["actions"]
    assert isinstance(actions, torch.Tensor)
    assert actions.shape == (4, 10, 1)  # (n_envs, horizon, action_dim)


def test_pgd_solver_solve_with_action_block():
    """Test solving with action blocking."""
    dummy_model = DummyCostModel()
    solver = PGDSolver(model=dummy_model, n_steps=10)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=3)

    solver.configure(action_space=action_space, n_envs=2, config=config)
    result = solver.solve(info_dict={})

    actions = result["actions"]
    assert actions.shape == (2, 5, 3)  # (n_envs, horizon, action_dim=3*1)


def test_pgd_solver_solve_single_env():
    """Test solving with a single environment."""
    solver = PGDSolver(model=DummyCostModel(), n_steps=10)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=7, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=1, config=config)
    result = solver.solve(info_dict={})

    actions = result["actions"]
    assert actions.shape == (1, 7, 1)


def test_pgd_solver_solve_ignores_info_dict():
    """Test that solve ignores info_dict content."""
    solver = PGDSolver(model=DummyCostModel(), n_steps=10)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=3)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    # Should produce same shape regardless of info_dict content
    result1 = solver.solve(info_dict={})
    result2 = solver.solve(info_dict={"state": torch.randn(2, 10), "obs": None})
    results3 = solver.solve(info_dict={"state": np.random.randn(2, 10), "obs": None})

    assert result1["actions"].shape == result2["actions"].shape == results3["actions"].shape == (2, 5, 3)


###########################
## Warm-Start Tests      ##
###########################


def test_pgd_solver_with_init_action():
    """Test warm-starting with partial action sequence."""
    solver = PGDSolver(model=DummyCostModel(), n_steps=0, var_scale=0.0)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    # Provide first 3 steps
    init_action = torch.randn(2, 3, 5)
    solver.init_action(init_action)
    actions = solver.init.squeeze(1)

    assert torch.allclose(actions[:, :3, :], init_action)


def test_pgd_solver_warm_start_full_horizon():
    """Test warm-starting with complete action sequence."""
    solver = PGDSolver(model=DummyCostModel(), n_steps=0, var_scale=0.0)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    # Provide all 5 steps
    init_action = torch.randn(2, 5, 5)
    solver.init_action(init_action)
    actions = solver.init.squeeze(1)

    assert torch.allclose(actions, init_action)


def test_pgd_solver_warm_start_with_action_block():
    """Test warm-starting with action blocking."""
    solver = PGDSolver(model=DummyCostModel(), n_steps=0)
    action_space = gym_spaces.Discrete(5)

    config = PlanConfig(horizon=6, receding_horizon=3, action_block=3)
    solver.configure(action_space=action_space, n_envs=2, config=config)
    init_action = torch.randn(2, 6, 15)
    solver.init_action(init_action)

    actions = solver.init.squeeze(1)
    assert torch.allclose(actions, init_action)


def test_pgd_solver_warm_start_with_var_scale():
    """Test warm-starting with var_scale."""
    solver = PGDSolver(model=DummyCostModel(), n_steps=0, var_scale=0.1, num_samples=2)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=6, receding_horizon=3, action_block=3)
    solver.configure(action_space=action_space, n_envs=2, config=config)
    init_action = torch.randn(2, 6, 15)
    solver.init_action(init_action)
    actions = solver.init[:, 0, :, :]
    actions_noised = solver.init[:, 1, :, :]

    # actions should be the same
    assert torch.allclose(actions, init_action)
    # actions_noised should be different from init_action
    assert not torch.allclose(actions_noised, init_action)


def test_pgd_solver_warm_start_with_from_scalar():
    """Test warm-starting with from_scalar."""
    solver = PGDSolver(model=DummyCostModel(), n_steps=0, var_scale=0.0)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=6, receding_horizon=3, action_block=3)
    solver.configure(action_space=action_space, n_envs=2, config=config)
    init_action = torch.randint(0, 5, (2, 6, 3))
    solver.init_action(init_action, from_scalar=True)
    actions = solver.init[:, 0, :, :]

    # correct shape
    assert actions.shape == (2, 6, 15)
    # correct values
    actions_scalar = solver._factor_action_block(actions).argmax(dim=-1)
    assert torch.allclose(actions_scalar, init_action)


###########################
## Callable Tests        ##
###########################


def test_pgd_solver_callable():
    """Test that solver is callable via __call__."""
    solver = PGDSolver(model=DummyCostModel(), n_steps=10)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    # Test both calling methods produce same shape
    result1 = solver(info_dict={})
    result2 = solver.solve(info_dict={})

    assert result1["actions"].shape == result2["actions"].shape == (2, 5, 1)


def test_pgd_solver_callable_with_kwargs():
    """Test callable interface with keyword arguments."""
    solver = PGDSolver(model=DummyCostModel(), n_steps=10)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    init_action = torch.randn(2, 2, 5)
    result = solver(info_dict={}, init_action=init_action)

    assert result["actions"].shape == (2, 5, 1)


###########################
## Edge Cases & Errors   ##
###########################


def test_pgd_solver_solve_empty_init_action():
    """Test solving with empty init_action tensor."""
    solver = PGDSolver(model=DummyCostModel(), n_steps=10)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    # Empty tensor (0 steps)
    init_action = torch.zeros((2, 0, 5))
    result = solver.solve(info_dict={}, init_action=init_action)

    assert result["actions"].shape == (2, 5, 1)


def test_pgd_solver_horizon_1():
    """Test solver with horizon of 1."""
    solver = PGDSolver(model=DummyCostModel(), n_steps=10, action_noise=0.1)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=1, receding_horizon=1, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)
    result = solver.solve(info_dict={})

    assert result["actions"].shape == (2, 1, 1)


def test_pgd_solver_large_horizon():
    """Test solver with large horizon."""
    solver = PGDSolver(model=DummyCostModel(), n_steps=10)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=100, receding_horizon=50, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)
    result = solver.solve(info_dict={})

    assert result["actions"].shape == (2, 100, 1)


def test_pgd_solver_many_envs():
    """Test solver with many parallel environments."""
    solver = PGDSolver(model=DummyCostModel(), n_steps=10)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=1)

    solver.configure(action_space=action_space, n_envs=64, config=config)
    result = solver.solve(info_dict={})

    assert result["actions"].shape == (64, 10, 1)


###########################
## Integration Tests     ##
###########################


def test_pgd_solver_deterministic_with_seed():
    """Test that results are reproducible with seeds."""
    solver1 = PGDSolver(model=DummyCostModel(), n_steps=10)
    solver2 = PGDSolver(model=DummyCostModel(), n_steps=10)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=2)

    solver1.configure(action_space=action_space, n_envs=2, config=config)
    solver2.configure(action_space=action_space, n_envs=2, config=config)

    # First run
    torch.manual_seed(42)
    np.random.seed(42)
    action_space.seed(42)
    result1 = solver1.solve(info_dict={})

    # Second run – reset seeds to the same values
    torch.manual_seed(42)
    np.random.seed(42)
    action_space.seed(42)
    result2 = solver2.solve(info_dict={})

    assert torch.allclose(result1["actions"], result2["actions"])


def test_pgd_solver_multiple_solves():
    """Test multiple solve calls produce different results."""
    solver = PGDSolver(model=DummyCostModel(), n_steps=5)
    action_space = gym_spaces.Discrete(10)
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    result1 = solver.solve(info_dict={})
    result2 = solver.solve(info_dict={})

    # Results should be different (with very high probability)
    assert not torch.allclose(result1["actions"], result2["actions"])


def test_pgd_solver_receding_horizon_pattern():
    """Test typical receding horizon planning pattern."""
    solver = PGDSolver(model=DummyCostModel(), n_steps=10)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=1)

    solver.configure(action_space=action_space, n_envs=4, config=config)

    # First planning step
    result1 = solver.solve(info_dict={})
    actions1 = result1["actions"]
    assert actions1.shape == (4, 10, 1)

    # Execute first 5 steps, keep remaining 5
    remaining = actions1[:, 5:, :]
    assert remaining.shape == (4, 5, 1)

    # Second planning step with warm-start
    result2 = solver.solve(info_dict={}, init_action=remaining, from_scalar=True)
    actions2 = result2["actions"]
    assert actions2.shape == (4, 10, 1)


def test_pgd_solver_respects_action_space_bounds():
    """Test that sampled actions have only allowed discrete values 0, 1, 2, 3, or 4."""
    solver = PGDSolver(model=DummyCostModel(), n_steps=10)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=20, receding_horizon=10, action_block=1)

    solver.configure(action_space=action_space, n_envs=4, config=config)
    result = solver.solve(info_dict={})

    actions = result["actions"]
    # Check that all actions are in {0, 1, 2, 3, 4}
    allowed = set(range(action_space.n))
    actions_np = actions.cpu().numpy()
    assert set(actions_np.flatten()).issubset(allowed)
