"""Tests for RandomSolver class."""

import torch
from gymnasium import spaces as gym_spaces

from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver.cat_cem import CategoricalCEMSolver


class DummyCostModel:
    """Simple Costable implementation for tests."""

    def get_cost(
        self,
        info_dict: dict,
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        # w = torch.randn_like(action_candidates)
        w = torch.zeros_like(action_candidates)
        # Quadratic cost: sum over horizon and action dims
        cost = (action_candidates - w).pow(2).sum(dim=(-1, -2))

        # shape: (batch_envs, num_samples)
        return cost


###########################
## Configuration Tests   ##
###########################


def test_cat_cem_solver_configure_discrete_action_space():
    """Test CategoricalCEMSolver configuration with discrete action space."""
    solver = CategoricalCEMSolver(model=DummyCostModel())
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=8, receding_horizon=4, action_block=2)

    solver.configure(action_space=action_space, n_envs=3, config=config)

    assert solver._configured is True
    assert solver.n_envs == 3
    assert solver._action_dim == 5


def test_cat_cem_solver_configure_multi_discrete_action_space():
    """Test CategoricalCEMSolver configuration with multi-discrete action space."""
    solver = CategoricalCEMSolver(model=DummyCostModel())
    action_space = gym_spaces.MultiDiscrete([5, 3, 2])
    config = PlanConfig(horizon=8, receding_horizon=4, action_block=2)

    solver.configure(action_space=action_space, n_envs=3, config=config)

    assert solver._configured is True
    assert solver.n_envs == 3
    assert solver._action_dim == 10  # Empty shape[1:] = 1


def test_cat_cem_solver_configure_multi_env():
    """Test CategoricalCEMSolver configuration with multiple environments."""
    solver = CategoricalCEMSolver(model=DummyCostModel())
    action_space = gym_spaces.MultiDiscrete([2, 3])
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=16, config=config)

    assert solver.n_envs == 16


def test_cat_cem_solver_properties_after_configure():
    """Test that properties work correctly after configuration."""
    solver = CategoricalCEMSolver(model=DummyCostModel(), independence=True)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=2)

    solver.configure(action_space=action_space, n_envs=4, config=config)

    assert solver.n_envs == 4
    assert solver.action_dim == 10
    assert solver.horizon == 10

    # check logits
    logits = solver.init_action_distrib()
    assert logits.shape == (4, 10, 2, 5)
    assert torch.allclose(logits, torch.zeros_like(logits))

    solver = CategoricalCEMSolver(model=DummyCostModel(), independence=False)
    solver.configure(action_space=action_space, n_envs=4, config=config)

    logits = solver.init_action_distrib()
    assert logits.shape == (4, 10, 2, 5)
    assert torch.allclose(logits, torch.zeros_like(logits))


###########################
## Solve Method Tests    ##
###########################


def test_cat_cem_solver_solve_full_horizon():
    """Test solving generates full action sequence."""
    solver = CategoricalCEMSolver(model=DummyCostModel())
    action_space = gym_spaces.MultiDiscrete([2, 3])
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=1)

    solver.configure(action_space=action_space, n_envs=4, config=config)
    result = solver.solve(info_dict={})

    assert "actions" in result
    actions = result["actions"]
    assert isinstance(actions, torch.Tensor)
    assert actions.shape == (4, 10, 2)  # (n_envs, horizon, action_dim)


def test_cat_cem_solver_solve_with_action_block():
    """Test solving with action blocking."""
    solver = CategoricalCEMSolver(model=DummyCostModel())
    action_space = gym_spaces.MultiDiscrete([2, 3])
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=3)

    solver.configure(action_space=action_space, n_envs=2, config=config)
    result = solver.solve(info_dict={})

    actions = result["actions"]
    assert actions.shape == (2, 5, 3 * 2)


def test_cat_cem_solver_solve_single_env():
    """Test solving with a single environment."""
    solver = CategoricalCEMSolver(model=DummyCostModel())
    action_space = gym_spaces.MultiDiscrete([2, 3])
    config = PlanConfig(horizon=7, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=1, config=config)
    result = solver.solve(info_dict={})

    actions = result["actions"]
    assert actions.shape == (1, 7, 2)


def test_cat_cem_solver_solve_ignores_info_dict():
    """Test that solve ignores info_dict content."""
    solver = CategoricalCEMSolver(model=DummyCostModel())
    action_space = gym_spaces.MultiDiscrete([2, 3])
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    # Should produce same shape regardless of info_dict content
    result1 = solver.solve(info_dict={})
    result2 = solver.solve(info_dict={"state": torch.randn(2, 10)})

    assert result1["actions"].shape == result2["actions"].shape == (2, 5, 2)


###########################
## Warm-Start Tests      ##
###########################


def test_cat_cem_solver_configure_with_init_action_independence():
    """Test warm-starting with partial action sequence."""
    horizon = 10
    receding_horizon = 5
    action_block = 2
    n_envs = 2
    n = 5
    solver = CategoricalCEMSolver(model=DummyCostModel(), independence=True)
    action_space = gym_spaces.Discrete(n)
    config = PlanConfig(horizon=horizon, receding_horizon=receding_horizon, action_block=action_block)

    solver.configure(action_space=action_space, n_envs=n_envs, config=config)

    # Provide first 5 steps
    init_action = torch.randint(0, n, size=(n_envs, horizon - receding_horizon, action_block))

    logits = solver.init_action_distrib(init_action)
    assert logits.shape == (n_envs, horizon, action_block, n)
    assert torch.allclose(
        logits[:, horizon - receding_horizon :, :, :], torch.zeros_like(logits[:, horizon - receding_horizon :, :, :])
    )
    assert not torch.allclose(
        logits[:, : horizon - receding_horizon, :, :], torch.zeros_like(logits[:, : horizon - receding_horizon, :, :])
    )

    action_space = gym_spaces.MultiDiscrete([6, 7, 8])
    solver.configure(action_space=action_space, n_envs=n_envs, config=config)

    init_action_1 = torch.randint(0, 6, size=(n_envs, horizon - receding_horizon, action_block))
    init_action_2 = torch.randint(0, 7, size=(n_envs, horizon - receding_horizon, action_block))
    init_action_3 = torch.randint(0, 8, size=(n_envs, horizon - receding_horizon, action_block))
    init_action = torch.stack([init_action_1, init_action_2, init_action_3], dim=-1)
    logits = solver.init_action_distrib(init_action)

    assert logits.shape == (n_envs, horizon, action_block, 6 + 7 + 8)
    assert solver._action_dim == 6 + 7 + 8
    assert torch.allclose(
        logits[:, horizon - receding_horizon :, :, :], torch.zeros_like(logits[:, horizon - receding_horizon :, :, :])
    )
    assert not torch.allclose(
        logits[:, : horizon - receding_horizon, :, :], torch.zeros_like(logits[:, : horizon - receding_horizon, :, :])
    )


def test_cat_cem_solver_configure_with_init_action_coupled():
    """Test warm-starting with partial action sequence."""
    horizon = 10
    receding_horizon = 5
    action_block = 2
    n_envs = 3
    n = 5
    solver = CategoricalCEMSolver(model=DummyCostModel(), independence=False)
    action_space = gym_spaces.Discrete(n)
    config = PlanConfig(horizon=horizon, receding_horizon=receding_horizon, action_block=action_block)

    solver.configure(action_space=action_space, n_envs=n_envs, config=config)

    # Provide first 5 steps
    init_action = torch.randint(0, n, size=(n_envs, horizon - receding_horizon, action_block))

    logits = solver.init_action_distrib(init_action)
    assert logits.shape == (n_envs, horizon, action_block, n)
    assert torch.allclose(
        logits[:, horizon - receding_horizon :, :, :], torch.zeros_like(logits[:, horizon - receding_horizon :, :, :])
    )
    assert not torch.allclose(
        logits[:, : horizon - receding_horizon, :, :], torch.zeros_like(logits[:, : horizon - receding_horizon, :, :])
    )

    solver = CategoricalCEMSolver(model=DummyCostModel(), independence=False)
    action_space = gym_spaces.MultiDiscrete([2, 3])
    solver.configure(action_space=action_space, n_envs=n_envs, config=config)

    init_action_1 = torch.randint(0, 2, size=(n_envs, horizon - receding_horizon, action_block))
    init_action_2 = torch.randint(0, 3, size=(n_envs, horizon - receding_horizon, action_block))

    init_action = torch.stack([init_action_1, init_action_2], dim=-1)

    logits = solver.init_action_distrib(init_action)
    assert logits.shape == (n_envs, horizon, action_block, 2 * 3)
    assert solver._action_dim == 2 + 3
    assert torch.allclose(
        logits[:, horizon - receding_horizon :, :, :], torch.zeros_like(logits[:, horizon - receding_horizon :, :, :])
    )
    assert not torch.allclose(
        logits[:, : horizon - receding_horizon, :, :], torch.zeros_like(logits[:, : horizon - receding_horizon, :, :])
    )


###########################
## Callable Tests        ##
###########################


def test_cat_cem_solver_callable():
    """Test that solver is callable via __call__."""
    solver = CategoricalCEMSolver(model=DummyCostModel())
    action_space = gym_spaces.MultiDiscrete([2, 3])
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    # Test both calling methods produce same shape
    result1 = solver(info_dict={})
    result2 = solver.solve(info_dict={})

    assert result1["actions"].shape == result2["actions"].shape == (2, 5, 2)


###########################
## Edge Cases & Errors   ##
###########################


def test_cat_cem_solver_horizon_1():
    """Test solver with horizon of 1."""
    solver = CategoricalCEMSolver(model=DummyCostModel())
    action_space = gym_spaces.MultiDiscrete([2, 3])
    config = PlanConfig(horizon=1, receding_horizon=1, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)
    result = solver.solve(info_dict={})

    assert result["actions"].shape == (2, 1, 2)


def test_cat_cem_solver_large_horizon():
    """Test solver with large horizon."""
    solver = CategoricalCEMSolver(model=DummyCostModel())
    action_space = gym_spaces.MultiDiscrete([2, 3])
    config = PlanConfig(horizon=100, receding_horizon=50, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)
    result = solver.solve(info_dict={})

    assert result["actions"].shape == (2, 100, 2)


def test_cat_cem_solver_many_envs():
    """Test solver with many parallel environments."""
    solver = CategoricalCEMSolver(model=DummyCostModel())
    action_space = gym_spaces.MultiDiscrete([2, 3])
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=1)

    solver.configure(action_space=action_space, n_envs=64, config=config)
    result = solver.solve(info_dict={})

    assert result["actions"].shape == (64, 10, 2)


def test_cat_cem_solver_multidimensional_action():
    """Test solver with multi-dimensional action space."""
    solver = CategoricalCEMSolver(model=DummyCostModel())
    action_space = gym_spaces.MultiDiscrete([2, 3])
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)
    result = solver.solve(info_dict={})

    # action_dim should be product of shape[1:] = 2 + 3 = 5
    assert result["actions"].shape == (2, 5, 2)


###########################
## Integration Tests     ##
###########################


def test_cat_cem_solver_deterministic_with_seed():
    """Test that results are reproducible with numpy seed."""
    solver1 = CategoricalCEMSolver(model=DummyCostModel())
    solver2 = CategoricalCEMSolver(model=DummyCostModel())
    action_space = gym_spaces.MultiDiscrete([2, 3])
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    solver1.configure(action_space=action_space, n_envs=2, config=config)
    solver2.configure(action_space=action_space, n_envs=2, config=config)

    # Set seed and sample
    action_space.seed(42)
    result1 = solver1.solve(info_dict={})

    # Reset seed and sample again
    action_space.seed(42)
    result2 = solver2.solve(info_dict={})

    print(result1["actions"])
    print(result2["actions"])

    # Results should be identical
    assert torch.allclose(result1["actions"], result2["actions"])


def test_cat_cem_solver_multiple_solves():
    """Test multiple solve calls produce different results."""
    solver = CategoricalCEMSolver(model=DummyCostModel())
    action_space = gym_spaces.MultiDiscrete([2, 3])
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    result1 = solver.solve(info_dict={})
    result2 = solver.solve(info_dict={})

    # Results should be different (with very high probability)
    assert not torch.allclose(result1["actions"], result2["actions"])


def test_cat_cem_solver_receding_horizon_pattern():
    """Test typical receding horizon planning pattern."""
    solver = CategoricalCEMSolver(model=DummyCostModel(), independence=False)
    action_space = gym_spaces.MultiDiscrete([2, 3])
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


def test_cat_cem_solver_sample_indices_factorized():
    """Test that the solver samples indices from the action distribution."""
    solver = CategoricalCEMSolver(model=DummyCostModel(), independence=True, num_samples=1)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=1)
    solver.configure(action_space=action_space, n_envs=1, config=config)

    logits = solver.init_action_distrib()
    sample = solver._sample_factorized_indices(logits, 1)
    onehot = solver._indices_to_onehot_concat(sample)

    assert sample.shape == (1, 1, 10, 1, 1)
    assert onehot.shape == (1, 1, 10, 1, 5)

    n_envs = 4
    horizon = 10
    receding_horizon = 5
    action_block = 3
    num_samples = 5

    action_space = gym_spaces.MultiDiscrete([2, 3])
    solver = CategoricalCEMSolver(model=DummyCostModel(), independence=True, num_samples=num_samples)
    config = PlanConfig(horizon=horizon, receding_horizon=receding_horizon, action_block=action_block)
    solver.configure(action_space=action_space, n_envs=n_envs, config=config)

    init_action_1 = torch.randint(0, 2, size=(n_envs, horizon - receding_horizon, action_block))
    init_action_2 = torch.randint(0, 3, size=(n_envs, horizon - receding_horizon, action_block))
    init_action = torch.stack([init_action_1, init_action_2], dim=-1)

    logits = solver.init_action_distrib(init_action)
    sample = solver._sample_factorized_indices(logits, n_envs)
    onehot = solver._indices_to_onehot_concat(sample)

    assert sample.shape == (n_envs, num_samples, horizon, action_block, 2)
    assert onehot.shape == (n_envs, num_samples, horizon, action_block, 5)


def test_cat_cem_solver_sample_indices_joint():
    """Test that the solver samples indices from the action distribution."""
    solver = CategoricalCEMSolver(model=DummyCostModel(), independence=False, num_samples=1)
    action_space = gym_spaces.MultiDiscrete([2, 3])
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=1)
    solver.configure(action_space=action_space, n_envs=1, config=config)

    logits = solver.init_action_distrib()
    joint, comp = solver._sample_joint_indices(logits, 1)
    onehot = solver._indices_to_onehot_concat(comp)

    assert joint.shape == (1, 1, 10, 1)
    assert comp.shape == (1, 1, 10, 1, 2)
    assert onehot.shape == (1, 1, 10, 1, 5)

    n_envs = 4
    horizon = 10
    receding_horizon = 5
    action_block = 3
    num_samples = 5

    action_space = gym_spaces.MultiDiscrete([2, 3])
    solver = CategoricalCEMSolver(model=DummyCostModel(), independence=False, num_samples=num_samples)
    config = PlanConfig(horizon=horizon, receding_horizon=receding_horizon, action_block=action_block)
    solver.configure(action_space=action_space, n_envs=n_envs, config=config)

    init_action_1 = torch.randint(0, 2, size=(n_envs, horizon - receding_horizon, action_block))
    init_action_2 = torch.randint(0, 3, size=(n_envs, horizon - receding_horizon, action_block))
    init_action = torch.stack([init_action_1, init_action_2], dim=-1)
    logits = solver.init_action_distrib(init_action)
    joint, comp = solver._sample_joint_indices(logits, n_envs)
    onehot = solver._indices_to_onehot_concat(comp)

    assert joint.shape == (n_envs, num_samples, horizon, action_block)
    assert comp.shape == (n_envs, num_samples, horizon, action_block, 2)
    assert onehot.shape == (n_envs, num_samples, horizon, action_block, 5)


def test_cat_cem_solver_update_logits_factorized():
    """Test that the solver updates logits using elite samples (factorized case)."""
    solver = CategoricalCEMSolver(model=DummyCostModel(), independence=True, num_samples=2)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=1)
    solver.configure(action_space=action_space, n_envs=1, config=config)

    logits = solver.init_action_distrib()
    sample = solver._sample_factorized_indices(logits, 1)
    new_logits = solver._update_factorized_from_elites_idx(sample)

    assert new_logits.shape == (1, 10, 1, 5)


def test_cat_cem_solver_update_logits_joint():
    """Test that the solver updates logits using elite samples (joint case)."""
    solver = CategoricalCEMSolver(model=DummyCostModel(), independence=False, num_samples=2)
    action_space = gym_spaces.MultiDiscrete([2, 3])
    config = PlanConfig(horizon=10, receding_horizon=5, action_block=1)
    solver.configure(action_space=action_space, n_envs=1, config=config)

    logits = solver.init_action_distrib()
    joint, comp = solver._sample_joint_indices(logits, 1)
    new_logits = solver._update_joint_from_elites_jointidx(joint)

    assert new_logits.shape == (1, 10, 1, 6)


if __name__ == "__main__":
    test_cat_cem_solver_deterministic_with_seed()
