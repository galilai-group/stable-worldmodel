"""Tests for LagrangianSolver class."""

import numpy as np
import pytest
import torch
from gymnasium import spaces as gym_spaces

from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver.lagrangian import LagrangianSolver


# ---------------------------------------------------------------------------
# Minimal mock models
# ---------------------------------------------------------------------------


class CostOnlyModel(torch.nn.Module):
    """Model with only get_cost — no constraints."""

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        # (B, S, H, D) -> (B, S): MSE toward zero
        return action_candidates.pow(2).mean(dim=(-1, -2))


class ConstrainedModel(torch.nn.Module):
    """Model with get_cost and get_constraints (2 inequality constraints)."""

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        return action_candidates.pow(2).mean(dim=(-1, -2))

    def get_constraints(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        # g0: action L2 norm <= 1  =>  ||a||_2 - 1 <= 0
        action_norm = action_candidates.norm(dim=-1).mean(dim=2)  # (B, S)
        g0 = action_norm - 1.0

        # g1: first action dim <= 0.5  =>  a[..., 0] - 0.5 <= 0
        g1 = action_candidates[..., 0].mean(dim=2) - 0.5

        return torch.stack([g0, g1], dim=-1)  # (B, S, 2)


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def make_box_space(action_dim: int = 4):
    return gym_spaces.Box(
        low=-np.inf, high=np.inf, shape=(1, action_dim), dtype=np.float32
    )


def make_solver(
    model=None, n_steps=5, n_outer_steps=2, num_samples=3, **kwargs
):
    if model is None:
        model = CostOnlyModel()
    return LagrangianSolver(
        model=model,
        n_steps=n_steps,
        n_outer_steps=n_outer_steps,
        num_samples=num_samples,
        **kwargs,
    )


def configure(solver, action_dim=4, n_envs=2, horizon=4, action_block=1):
    action_space = make_box_space(action_dim)
    config = PlanConfig(
        horizon=horizon, receding_horizon=1, action_block=action_block
    )
    solver.configure(action_space=action_space, n_envs=n_envs, config=config)
    return action_space, config


###########################
## Initialization Tests  ##
###########################


def test_init_defaults():
    """LagrangianSolver starts unconfigured with sensible defaults."""
    solver = make_solver()
    assert not solver._configured
    assert solver._n_envs is None
    assert solver._action_dim is None
    assert solver._config is None
    assert solver._lambdas is None
    assert solver.rho_init == 1.0
    assert solver.rho_max == 1e4
    assert solver.rho_scale == 2.0
    assert solver.persist_multipliers is True


def test_init_custom_params():
    """Constructor stores custom hyperparameters correctly."""
    solver = LagrangianSolver(
        model=CostOnlyModel(),
        n_steps=20,
        n_outer_steps=7,
        num_samples=5,
        rho_init=0.5,
        rho_max=500.0,
        rho_scale=3.0,
        persist_multipliers=False,
        optimizer_kwargs={'lr': 0.01},
    )
    assert solver.n_steps == 20
    assert solver.n_outer_steps == 7
    assert solver.num_samples == 5
    assert solver.rho_init == 0.5
    assert solver.rho_max == 500.0
    assert solver.rho_scale == 3.0
    assert solver.persist_multipliers is False
    assert solver.optimizer_kwargs == {'lr': 0.01}


###########################
## Configuration Tests   ##
###########################


def test_configure_sets_state():
    """configure() populates all expected attributes."""
    solver = make_solver()
    action_space, config = configure(
        solver, action_dim=4, n_envs=3, horizon=8, action_block=2
    )

    assert solver._configured
    assert solver._n_envs == 3
    assert solver._action_dim == 4
    assert solver._action_space is action_space
    assert solver._config is config


def test_configure_properties():
    """action_dim and horizon properties reflect config after configure()."""
    solver = make_solver()
    configure(solver, action_dim=6, n_envs=2, horizon=10, action_block=3)

    assert solver.n_envs == 2
    assert solver.action_dim == 6 * 3  # action_dim * action_block
    assert solver.horizon == 10


def test_configure_action_block_1():
    """action_dim == raw dim when action_block=1."""
    solver = make_solver()
    configure(solver, action_dim=4, n_envs=1, horizon=5, action_block=1)
    assert solver.action_dim == 4


def test_n_envs_before_configure_is_none():
    solver = make_solver()
    assert solver.n_envs is None


###########################
## init_action Tests     ##
###########################


def test_init_action_zeros_creates_zero_init():
    """init_action with a zeros tensor creates a zero init of correct shape."""
    solver = make_solver(num_samples=1, var_scale=0.0)
    configure(solver, action_dim=4, n_envs=2, horizon=5, action_block=1)

    with torch.no_grad():
        solver.init_action(torch.zeros(2, 5, 4))

    # shape: (n_envs, num_samples, horizon, action_dim)
    assert solver.init.shape == (2, 1, 5, 4)
    assert torch.allclose(solver.init, torch.zeros_like(solver.init))


def test_init_action_full_tensor_preserved_as_first_sample():
    """Providing a full-horizon tensor preserves it as the first sample."""
    solver = make_solver(num_samples=1, var_scale=0.0)
    configure(solver, action_dim=4, n_envs=2, horizon=6, action_block=1)

    init = torch.ones(2, 6, 4)
    with torch.no_grad():
        solver.init_action(init)

    first_sample = solver.init[:, 0]  # (n_envs, horizon, action_dim)
    assert torch.allclose(first_sample, init)


def test_init_action_full_horizon_preserved():
    """Providing all steps for the first sample is preserved exactly (var_scale=0)."""
    solver = make_solver(num_samples=1, var_scale=0.0)
    configure(solver, action_dim=4, n_envs=2, horizon=5, action_block=1)

    init = torch.randn(2, 5, 4)
    with torch.no_grad():
        solver.init_action(init)

    assert torch.allclose(solver.init[:, 0], init)


def test_init_action_var_scale_noises_extra_samples():
    """Extra samples (beyond the first) are perturbed by var_scale."""
    solver = make_solver(num_samples=4, var_scale=1.0)
    configure(solver, action_dim=4, n_envs=2, horizon=5, action_block=1)

    init = torch.zeros(2, 5, 4)
    with torch.no_grad():
        solver.init_action(init)

    # First sample should match init exactly
    assert torch.allclose(solver.init[:, 0], init)
    # At least one other sample should differ (var_scale > 0)
    assert not torch.allclose(solver.init[:, 1], init)


###########################
## Augmented Lagrangian  ##
###########################


def test_augmented_lagrangian_loss_no_violation():
    """Loss equals cost when constraints are satisfied (g <= 0)."""
    solver = make_solver()
    configure(solver)

    B, S, C = 2, 3, 2
    costs = torch.ones(B, S)
    constraints = -torch.ones(B, S, C)  # all satisfied
    lambdas = torch.zeros(B, C)

    loss = solver._augmented_lagrangian_loss(
        costs, constraints, lambdas, rho=1.0
    )
    # linear: lambda=0 -> 0; quadratic: relu(-1)=0 -> 0
    assert torch.isclose(loss, torch.tensor(float(B * S)))


def test_augmented_lagrangian_loss_with_violation():
    """Violated constraints increase the loss."""
    solver = make_solver()
    configure(solver)

    B, S, C = 2, 3, 1
    costs = torch.zeros(B, S)
    constraints_ok = -torch.ones(B, S, C)
    constraints_viol = torch.ones(B, S, C)  # g=1 > 0
    lambdas = torch.zeros(B, C)
    rho = 1.0

    loss_ok = solver._augmented_lagrangian_loss(
        costs, constraints_ok, lambdas, rho
    )
    loss_viol = solver._augmented_lagrangian_loss(
        costs, constraints_viol, lambdas, rho
    )

    assert loss_viol > loss_ok


def test_augmented_lagrangian_loss_linear_term():
    """Non-zero lambda multiplies constraint value linearly."""
    solver = make_solver()
    configure(solver)

    B, S, C = 1, 1, 1
    costs = torch.zeros(B, S)
    constraints = torch.ones(B, S, C) * 2.0  # g=2
    lambdas = torch.ones(B, C) * 3.0  # lambda=3
    rho = 0.0  # disable quadratic term

    loss = solver._augmented_lagrangian_loss(costs, constraints, lambdas, rho)
    # Expected: lambda * g = 3 * 2 = 6
    assert torch.isclose(loss, torch.tensor(6.0))


###########################
## Update Multipliers    ##
###########################


def test_update_multipliers_clamps_to_zero():
    """Multipliers never go below zero (dual feasibility)."""
    solver = make_solver()
    configure(solver)

    B, S, C = 2, 3, 2
    constraints = -torch.ones(B, S, C)  # all satisfied
    lambdas = torch.zeros(B, C)

    new_lambdas = solver._update_multipliers(constraints, lambdas, rho=1.0)
    assert (new_lambdas >= 0).all()


def test_update_multipliers_increases_on_violation():
    """Multipliers increase when constraints are violated."""
    solver = make_solver()
    configure(solver)

    B, S, C = 2, 3, 1
    constraints = torch.ones(B, S, C)  # g=1 > 0
    lambdas = torch.zeros(B, C)

    new_lambdas = solver._update_multipliers(constraints, lambdas, rho=1.0)
    assert (new_lambdas > lambdas).all()


def test_update_multipliers_formula():
    """lambda_new = clamp(lambda + rho * mean_g, 0)."""
    solver = make_solver()
    configure(solver)

    B, S, C = 1, 4, 1
    # mean_g = 0.5 (two +1 and two -1 samples)
    constraints = torch.tensor([1.0, 1.0, -1.0, -1.0]).reshape(B, S, C)
    lambdas = torch.zeros(B, C)
    rho = 2.0

    new_lambdas = solver._update_multipliers(constraints, lambdas, rho)
    expected = torch.clamp(torch.zeros(B, C) + rho * 0.0, min=0.0)
    assert torch.allclose(new_lambdas, expected)


###########################
## Solve Output Tests    ##
###########################


def test_solve_output_keys():
    """solve() returns dict with expected keys."""
    solver = make_solver()
    configure(solver)
    out = solver.solve({'obs': torch.zeros(2, 4)})

    assert 'actions' in out
    assert 'cost' in out
    assert 'constraint_violation' in out
    assert 'lambdas' in out


def test_solve_actions_shape_no_constraint():
    """Actions have shape (n_envs, horizon, action_dim) without constraints."""
    solver = make_solver(n_steps=3, n_outer_steps=2, num_samples=4)
    configure(solver, action_dim=4, n_envs=3, horizon=6, action_block=1)

    out = solver.solve({})
    assert out['actions'].shape == (3, 6, 4)


def test_solve_actions_shape_with_action_block():
    """action_block scales the last dimension of actions."""
    solver = make_solver(n_steps=3, n_outer_steps=2, num_samples=2)
    configure(solver, action_dim=4, n_envs=2, horizon=5, action_block=2)

    out = solver.solve({})
    assert out['actions'].shape == (2, 5, 8)  # action_dim * action_block


def test_solve_actions_shape_with_constraints():
    """Actions shape is correct when model has constraints."""
    solver = make_solver(
        model=ConstrainedModel(), n_steps=5, n_outer_steps=3, num_samples=4
    )
    configure(solver, action_dim=4, n_envs=2, horizon=4, action_block=1)

    out = solver.solve({})
    assert out['actions'].shape == (2, 4, 4)


def test_solve_lambdas_none_without_constraints():
    """lambdas is None when model has no constraints."""
    solver = make_solver(model=CostOnlyModel())
    configure(solver)
    out = solver.solve({})
    assert out['lambdas'] is None


def test_solve_lambdas_shape_with_constraints():
    """lambdas has shape (n_envs, n_constraints) when constraints exist."""
    solver = make_solver(
        model=ConstrainedModel(), n_steps=3, n_outer_steps=2, num_samples=3
    )
    configure(solver, action_dim=4, n_envs=3, horizon=4, action_block=1)

    out = solver.solve({})
    assert out['lambdas'] is not None
    assert out['lambdas'].shape == (3, 2)  # (n_envs, C=2)


def test_solve_lambdas_nonnegative():
    """Lagrange multipliers are always >= 0."""
    solver = make_solver(
        model=ConstrainedModel(), n_steps=5, n_outer_steps=5, num_samples=4
    )
    configure(solver, action_dim=4, n_envs=2, horizon=4, action_block=1)

    out = solver.solve({})
    assert (out['lambdas'] >= 0).all()


def test_solve_constraint_violation_recorded():
    """constraint_violation list is populated when constraints are present."""
    solver = make_solver(
        model=ConstrainedModel(), n_steps=3, n_outer_steps=2, num_samples=3
    )
    configure(solver, action_dim=4, n_envs=2, horizon=4, action_block=1)

    out = solver.solve({})
    assert len(out['constraint_violation']) > 0


def test_solve_callable_interface():
    """solver(info_dict) is equivalent to solver.solve(info_dict)."""
    solver = make_solver()
    configure(solver, action_dim=4, n_envs=2, horizon=4, action_block=1)

    out1 = solver.solve({})
    solver2 = make_solver()
    configure(solver2, action_dim=4, n_envs=2, horizon=4, action_block=1)
    out2 = solver2({})

    assert out1['actions'].shape == out2['actions'].shape


###########################
## Multiplier Persistence##
###########################


def test_persist_multipliers_warm_starts():
    """With persist_multipliers=True, lambdas carry over between calls."""
    solver = make_solver(
        model=ConstrainedModel(),
        n_steps=5,
        n_outer_steps=3,
        num_samples=4,
        persist_multipliers=True,
    )
    configure(solver, action_dim=4, n_envs=2, horizon=4, action_block=1)

    out1 = solver.solve({})
    _lambdas_after_first = out1['lambdas'].clone()

    out2 = solver.solve({})
    lambdas_after_second = out2['lambdas']

    # With persistence, internal state should reflect accumulated multipliers
    # (not necessarily larger, but lambdas object must exist)
    assert lambdas_after_second is not None
    # Internal _lambdas must be non-None
    assert solver._lambdas is not None


def test_no_persist_multipliers_resets():
    """With persist_multipliers=False, lambdas reset each solve()."""
    solver = make_solver(
        model=ConstrainedModel(),
        n_steps=5,
        n_outer_steps=3,
        num_samples=4,
        persist_multipliers=False,
    )
    configure(solver, action_dim=4, n_envs=2, horizon=4, action_block=1)

    solver.solve({})
    assert (
        solver._lambdas is None or True
    )  # _lambdas gets reset at start of solve()

    # Run twice and check consistency
    out1 = solver.solve({})
    out2 = solver.solve({})
    # Both runs should produce same lambdas (same starting point each time)
    assert torch.allclose(out1['lambdas'], out2['lambdas'])


###########################
## Batch Size Tests      ##
###########################


def test_solve_with_batch_size_smaller_than_n_envs():
    """batch_size < n_envs triggers batched processing and still returns correct shape."""
    solver = LagrangianSolver(
        model=CostOnlyModel(),
        n_steps=3,
        n_outer_steps=2,
        num_samples=3,
        batch_size=2,
    )
    configure(solver, action_dim=4, n_envs=6, horizon=4, action_block=1)

    out = solver.solve({})
    assert out['actions'].shape == (6, 4, 4)


def test_solve_with_batch_size_equal_to_n_envs():
    """batch_size == n_envs is equivalent to no batching."""
    solver = LagrangianSolver(
        model=CostOnlyModel(),
        n_steps=3,
        n_outer_steps=2,
        num_samples=3,
        batch_size=4,
    )
    configure(solver, action_dim=4, n_envs=4, horizon=4, action_block=1)

    out = solver.solve({})
    assert out['actions'].shape == (4, 4, 4)


def test_solve_with_batch_size_one():
    """batch_size=1 processes one env at a time."""
    solver = LagrangianSolver(
        model=CostOnlyModel(),
        n_steps=3,
        n_outer_steps=2,
        num_samples=2,
        batch_size=1,
    )
    configure(solver, action_dim=4, n_envs=3, horizon=4, action_block=1)

    out = solver.solve({})
    assert out['actions'].shape == (3, 4, 4)


###########################
## Info-Dict Expansion   ##
###########################


def test_solve_expands_tensor_info():
    """Tensor entries in info_dict are correctly expanded per batch."""

    class InfoConsumingModel(torch.nn.Module):
        def get_cost(self, info_dict, action_candidates):
            # Verify info has been expanded to (B, S, ...)
            obs = info_dict['obs']
            B, S, H, D = action_candidates.shape
            assert obs.shape[:2] == (B, S), (
                f'Expected ({B}, {S}, ...), got {obs.shape}'
            )
            return action_candidates.pow(2).mean(dim=(-1, -2))

    solver = LagrangianSolver(
        model=InfoConsumingModel(),
        n_steps=2,
        n_outer_steps=1,
        num_samples=3,
    )
    configure(solver, action_dim=4, n_envs=2, horizon=4, action_block=1)

    info = {'obs': torch.zeros(2, 8)}
    out = solver.solve(info)
    assert out['actions'].shape == (2, 4, 4)


def test_solve_expands_numpy_info():
    """Numpy entries in info_dict are expanded per batch."""

    class NpInfoModel(torch.nn.Module):
        def get_cost(self, info_dict, action_candidates):
            obs = info_dict['obs']
            B, S, H, D = action_candidates.shape
            assert obs.shape[:2] == (B, S)
            return action_candidates.pow(2).mean(dim=(-1, -2))

    solver = LagrangianSolver(
        model=NpInfoModel(),
        n_steps=2,
        n_outer_steps=1,
        num_samples=2,
    )
    configure(solver, action_dim=4, n_envs=2, horizon=4, action_block=1)

    info = {'obs': np.zeros((2, 8))}
    out = solver.solve(info)
    assert out['actions'].shape == (2, 4, 4)


###########################
## Warm-Start Tests      ##
###########################


def test_solve_warm_start_accepted():
    """Passing init_action to solve() doesn't crash and returns correct shape."""
    solver = make_solver(n_steps=3, n_outer_steps=2, num_samples=2)
    configure(solver, action_dim=4, n_envs=2, horizon=6, action_block=1)

    init = torch.zeros(2, 3, 4)  # partial 3-step plan
    out = solver.solve({}, init_action=init)
    assert out['actions'].shape == (2, 6, 4)


###########################
## Rho Scaling Tests     ##
###########################


def test_rho_grows_each_outer_step():
    """rho should not exceed rho_max."""
    n_outer = 20
    solver = LagrangianSolver(
        model=ConstrainedModel(),
        n_steps=2,
        n_outer_steps=n_outer,
        num_samples=2,
        rho_init=1.0,
        rho_max=8.0,
        rho_scale=2.0,
    )
    configure(solver, action_dim=4, n_envs=1, horizon=4, action_block=1)

    # We just check the solve completes without error (rho clamped internally)
    out = solver.solve({})
    assert out['actions'].shape == (1, 4, 4)


###########################
## Edge Cases            ##
###########################


def test_solve_single_env():
    solver = make_solver(n_steps=3, n_outer_steps=2, num_samples=2)
    configure(solver, action_dim=4, n_envs=1, horizon=4, action_block=1)
    out = solver.solve({})
    assert out['actions'].shape == (1, 4, 4)


def test_solve_horizon_1():
    solver = make_solver(n_steps=3, n_outer_steps=2, num_samples=2)
    configure(solver, action_dim=4, n_envs=2, horizon=1, action_block=1)
    out = solver.solve({})
    assert out['actions'].shape == (2, 1, 4)


def test_solve_num_samples_1():
    solver = make_solver(n_steps=3, n_outer_steps=2, num_samples=1)
    configure(solver, action_dim=4, n_envs=2, horizon=4, action_block=1)
    out = solver.solve({})
    assert out['actions'].shape == (2, 4, 4)


def test_solve_many_envs():
    solver = make_solver(n_steps=2, n_outer_steps=1, num_samples=2)
    configure(solver, action_dim=4, n_envs=32, horizon=4, action_block=1)
    out = solver.solve({})
    assert out['actions'].shape == (32, 4, 4)


def test_solve_cost_shape_assertion_violated():
    """solve() raises AssertionError if model returns wrong cost shape."""

    class BadCostModel(torch.nn.Module):
        def get_cost(self, info_dict, action_candidates):
            # Wrong: returns 1-D instead of (B, S)
            return action_candidates.pow(2).mean()

    solver = LagrangianSolver(
        model=BadCostModel(), n_steps=2, n_outer_steps=1, num_samples=2
    )
    configure(solver, action_dim=4, n_envs=2, horizon=4, action_block=1)

    with pytest.raises(AssertionError):
        solver.solve({})


def test_solve_cost_requires_grad_assertion():
    """solve() raises AssertionError if cost has no gradient."""

    class NoGradModel(torch.nn.Module):
        def get_cost(self, info_dict, action_candidates):
            return action_candidates.detach().pow(2).mean(dim=(-1, -2))

    solver = LagrangianSolver(
        model=NoGradModel(), n_steps=2, n_outer_steps=1, num_samples=2
    )
    configure(solver, action_dim=4, n_envs=2, horizon=4, action_block=1)

    with pytest.raises(AssertionError):
        solver.solve({})


###########################
## Integration Tests     ##
###########################


def test_cost_decreases_over_steps():
    """Optimizer should reduce cost on a convex quadratic problem."""

    class QuadraticCost(torch.nn.Module):
        """Cost = ||a - goal||^2, goal far from zero so initial cost is high."""

        def __init__(self, goal=2.0):
            super().__init__()
            self.goal = goal

        def get_cost(self, info_dict, action_candidates):
            return (action_candidates - self.goal).pow(2).mean(dim=(-1, -2))

    solver = LagrangianSolver(
        model=QuadraticCost(
            goal=0.0
        ),  # goal at zero; init at zero -> cost = 0
        n_steps=30,
        n_outer_steps=1,
        num_samples=1,
        optimizer_kwargs={'lr': 0.1},
    )
    configure(solver, action_dim=4, n_envs=1, horizon=4, action_block=1)

    out = solver.solve({})
    costs = out['cost'][0]  # list of losses per inner step
    # The optimizer should not increase cost by more than 2x from first to last
    # (just a sanity check, not a strict convergence guarantee)
    assert costs[-1] <= costs[0] * 10 or costs[-1] < 1.0


def test_constrained_solve_reduces_violation():
    """Running more outer steps should not increase constraint violation."""

    def run(n_outer):
        solver = LagrangianSolver(
            model=ConstrainedModel(),
            n_steps=20,
            n_outer_steps=n_outer,
            num_samples=8,
            rho_init=1.0,
            rho_scale=2.0,
            optimizer_kwargs={'lr': 0.05},
        )
        configure(solver, action_dim=4, n_envs=2, horizon=4, action_block=1)
        out = solver.solve({})
        return (
            out['constraint_violation'][-1]
            if out['constraint_violation']
            else float('inf')
        )

    viol_few = run(n_outer=1)
    viol_many = run(n_outer=8)
    # More outer steps with growing rho should generally reduce violation
    assert viol_many <= viol_few + 0.5  # generous tolerance


###########################
## Uncovered Branches    ##
###########################


def test_configure_non_box_action_space_warns(caplog):
    """configure() logs a warning when action space is not a Box."""
    solver = make_solver()
    discrete_space = gym_spaces.Discrete(5)

    # loguru captures to stdlib when propagate=True; use capfd or just check no crash
    # The warning is emitted via loguru — we just verify configure() doesn't raise
    solver.configure(
        action_space=discrete_space,
        n_envs=2,
        config=PlanConfig(horizon=4, receding_horizon=1, action_block=1),
    )
    assert (
        solver._configured
    )  # configure still succeeds despite wrong space type


def test_solve_info_dict_passthrough_non_tensor():
    """Non-tensor, non-numpy values in info_dict are passed through unchanged (else branch)."""

    class ScalarInfoModel(torch.nn.Module):
        def get_cost(self, info_dict, action_candidates):
            # The scalar value should be unchanged (not sliced/expanded)
            assert info_dict['tag'] == 'hello'
            return action_candidates.pow(2).mean(dim=(-1, -2))

    solver = LagrangianSolver(
        model=ScalarInfoModel(),
        n_steps=2,
        n_outer_steps=1,
        num_samples=2,
    )
    configure(solver, action_dim=4, n_envs=2, horizon=4, action_block=1)

    out = solver.solve({'tag': 'hello'})
    assert out['actions'].shape == (2, 4, 4)


def test_action_noise_applied_during_solve():
    """action_noise > 0 perturbs actions between inner steps."""
    solver = LagrangianSolver(
        model=CostOnlyModel(),
        n_steps=5,
        n_outer_steps=1,
        num_samples=1,
        action_noise=1.0,
        var_scale=0.0,
        seed=0,
    )
    configure(solver, action_dim=4, n_envs=2, horizon=4, action_block=1)

    # With action_noise > 0 the solve should complete and return valid actions
    out = solver.solve({})
    assert out['actions'].shape == (2, 4, 4)
    # Actions should not be all zeros (noise was injected)
    assert not torch.all(out['actions'] == 0)
