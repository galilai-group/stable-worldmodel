"""Tests for CategoricalCEMSolver."""

import numpy as np
import pytest
import torch
from gymnasium import spaces as gym_spaces

from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver.callbacks import (
    BestCostRecorder,
    MeanCostRecorder,
)
from stable_worldmodel.solver.categorical_cem import CategoricalCEMSolver


class DummyCostModel:
    """Random quadratic cost over one-hot candidates."""

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        # action_candidates: (B, N, H, action_simplex_dim) one-hot
        return action_candidates.pow(2).sum(dim=(-1, -2))


class FavorCategoryCostModel:
    """Cost is minimized by selecting a target category at every position."""

    def __init__(self, target: int = 2, base_simplex_dim: int = 4) -> None:
        self.target = target
        self.K = base_simplex_dim

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        # candidates: (B, N, H, action_block * K) one-hot
        ab = action_candidates.shape[-1] // self.K
        c = action_candidates.reshape(
            *action_candidates.shape[:-1], ab, self.K
        )
        # Negative mass on target → minimized by picking target everywhere
        return -c[..., self.target].sum(dim=(-1, -2))


###########################
## Initialization Tests  ##
###########################


def test_categorical_cem_solver_init():
    """Test CategoricalCEMSolver initialization."""
    model = DummyCostModel()
    solver = CategoricalCEMSolver(
        model=model, n_steps=10, num_samples=64, topk=8
    )
    assert solver.model is model
    assert solver.n_steps == 10
    assert solver.num_samples == 64
    assert solver.topk == 8
    assert solver.smoothing == 0.0
    assert solver.alpha == 0.0


def test_categorical_cem_solver_init_with_options():
    """Test initialization with smoothing and alpha."""
    solver = CategoricalCEMSolver(
        model=DummyCostModel(),
        n_steps=5,
        num_samples=32,
        topk=4,
        smoothing=0.05,
        alpha=0.3,
    )
    assert solver.smoothing == 0.05
    assert solver.alpha == 0.3


###########################
## Configuration Tests   ##
###########################


def test_categorical_cem_configure_discrete():
    """Configure with a Discrete action space."""
    solver = CategoricalCEMSolver(model=DummyCostModel(), n_steps=5)
    action_space = gym_spaces.Discrete(7)
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    assert solver._configured is True
    assert solver.n_envs == 2
    assert solver.base_simplex_dim == 7
    assert solver.action_simplex_dim == 7
    assert solver.action_block == 1
    assert solver.horizon == 5


def test_categorical_cem_configure_with_action_block():
    """action_simplex_dim should multiply by action_block."""
    solver = CategoricalCEMSolver(model=DummyCostModel(), n_steps=5)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=8, receding_horizon=4, action_block=3)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    assert solver.base_simplex_dim == 5
    assert solver.action_block == 3
    assert solver.action_simplex_dim == 15


def test_categorical_cem_configure_rejects_box():
    """Configure should assert on non-Discrete spaces."""
    solver = CategoricalCEMSolver(model=DummyCostModel(), n_steps=5)
    action_space = gym_spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    with pytest.raises(AssertionError):
        solver.configure(action_space=action_space, n_envs=1, config=config)


###########################
## init_probs Tests      ##
###########################


def test_categorical_cem_init_probs_uniform():
    """init_probs returns uniform distribution of correct shape."""
    solver = CategoricalCEMSolver(model=DummyCostModel(), n_steps=5)
    action_space = gym_spaces.Discrete(4)
    config = PlanConfig(horizon=6, receding_horizon=3, action_block=2)
    solver.configure(action_space=action_space, n_envs=3, config=config)

    probs = solver.init_probs(3)

    assert probs.shape == (3, 6, 2, 4)
    assert torch.allclose(probs, torch.full_like(probs, 0.25))
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3, 6, 2))


###########################
## Solve Method Tests    ##
###########################


def test_categorical_cem_solve_shape():
    """Solve produces actions of shape (n_envs, horizon, action_block)."""
    solver = CategoricalCEMSolver(
        model=DummyCostModel(), n_steps=3, num_samples=16, topk=4
    )
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=4, receding_horizon=2, action_block=1)
    solver.configure(action_space=action_space, n_envs=3, config=config)

    info_dict = {'state': torch.zeros(3, 2)}
    out = solver.solve(info_dict)

    assert 'actions' in out
    assert out['actions'].shape == (3, 4, 1)
    assert out['actions'].dtype == torch.int64
    assert len(out['costs']) == 3


def test_categorical_cem_solve_action_block_shape():
    """Solve handles action_block > 1 in output shape."""
    solver = CategoricalCEMSolver(
        model=DummyCostModel(), n_steps=2, num_samples=8, topk=2
    )
    action_space = gym_spaces.Discrete(4)
    config = PlanConfig(horizon=3, receding_horizon=2, action_block=3)
    solver.configure(action_space=action_space, n_envs=2, config=config)

    out = solver.solve({'state': torch.zeros(2, 2)})

    assert out['actions'].shape == (2, 3, 3)


def test_categorical_cem_solve_actions_in_bounds():
    """All action indices must be in [0, base_simplex_dim)."""
    solver = CategoricalCEMSolver(
        model=DummyCostModel(),
        n_steps=3,
        num_samples=16,
        topk=4,
        batch_size=2,
    )
    action_space = gym_spaces.Discrete(6)
    config = PlanConfig(horizon=8, receding_horizon=4, action_block=2)
    solver.configure(action_space=action_space, n_envs=4, config=config)

    out = solver.solve({'state': torch.zeros(4, 2)})

    actions_np = out['actions'].cpu().numpy()
    assert actions_np.min() >= 0
    assert actions_np.max() < action_space.n


def test_categorical_cem_probs_sum_to_one():
    """Output probs at each (env, t, block) sum to 1."""
    solver = CategoricalCEMSolver(
        model=DummyCostModel(), n_steps=3, num_samples=16, topk=4
    )
    action_space = gym_spaces.Discrete(4)
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=2)
    solver.configure(action_space=action_space, n_envs=2, config=config)

    out = solver.solve({'state': torch.zeros(2, 2)})

    probs = out['probs'][0]
    assert probs.shape == (2, 5, 2, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 5, 2), atol=1e-5)


def test_categorical_cem_solve_batched():
    """batch_size smaller than n_envs still produces full output."""
    solver = CategoricalCEMSolver(
        model=DummyCostModel(),
        n_steps=2,
        num_samples=8,
        topk=2,
        batch_size=2,
    )
    action_space = gym_spaces.Discrete(3)
    config = PlanConfig(horizon=4, receding_horizon=2, action_block=1)
    solver.configure(action_space=action_space, n_envs=5, config=config)

    out = solver.solve({'state': torch.zeros(5, 2)})

    assert out['actions'].shape == (5, 4, 1)
    assert len(out['costs']) == 5


###########################
## Convergence Tests     ##
###########################


def test_categorical_cem_converges_to_optimal_category():
    """With a clear cost signal, solver should pick the target category."""
    target = 2
    K = 4
    solver = CategoricalCEMSolver(
        model=FavorCategoryCostModel(target=target, base_simplex_dim=K),
        n_steps=15,
        num_samples=64,
        topk=8,
    )
    action_space = gym_spaces.Discrete(K)
    config = PlanConfig(horizon=4, receding_horizon=2, action_block=1)
    solver.configure(action_space=action_space, n_envs=2, config=config)

    out = solver.solve({'state': torch.zeros(2, 2)})

    # Every output action should be the target.
    assert (out['actions'] == target).all()
    # Final elite cost = -horizon (one mass per timestep on target).
    assert all(c == pytest.approx(-config.horizon) for c in out['costs'])


def test_categorical_cem_smoothing_keeps_distribution_full_support():
    """With smoothing > 0, no probability collapses fully to 0."""
    solver = CategoricalCEMSolver(
        model=FavorCategoryCostModel(target=1, base_simplex_dim=3),
        n_steps=10,
        num_samples=32,
        topk=4,
        smoothing=0.1,
    )
    action_space = gym_spaces.Discrete(3)
    config = PlanConfig(horizon=3, receding_horizon=2, action_block=1)
    solver.configure(action_space=action_space, n_envs=1, config=config)

    out = solver.solve({'state': torch.zeros(1, 2)})

    probs = out['probs'][0]
    assert (probs > 0).all()


###########################
## Reproducibility Tests ##
###########################


def test_categorical_cem_deterministic_with_seed():
    """Same seed → same actions."""
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=4, receding_horizon=2, action_block=2)

    def run(seed: int) -> torch.Tensor:
        solver = CategoricalCEMSolver(
            model=DummyCostModel(),
            n_steps=4,
            num_samples=16,
            topk=4,
            seed=seed,
        )
        solver.configure(action_space=action_space, n_envs=3, config=config)
        return solver.solve({'state': torch.zeros(3, 2)})['actions']

    a1 = run(42)
    a2 = run(42)
    a3 = run(123)

    assert torch.equal(a1, a2)
    assert not torch.equal(a1, a3)


###########################
## Callable / Callbacks  ##
###########################


def test_categorical_cem_callable():
    """__call__ forwards to solve."""
    solver = CategoricalCEMSolver(
        model=DummyCostModel(), n_steps=2, num_samples=8, topk=2
    )
    action_space = gym_spaces.Discrete(4)
    config = PlanConfig(horizon=3, receding_horizon=2, action_block=1)
    solver.configure(action_space=action_space, n_envs=2, config=config)

    out1 = solver({'state': torch.zeros(2, 2)})
    out2 = solver.solve({'state': torch.zeros(2, 2)})

    assert out1['actions'].shape == out2['actions'].shape == (2, 3, 1)


def test_categorical_cem_callbacks_record_history():
    """Callbacks accumulate per-step history across batches."""
    cbs = [BestCostRecorder(), MeanCostRecorder()]
    solver = CategoricalCEMSolver(
        model=DummyCostModel(),
        n_steps=4,
        num_samples=16,
        topk=4,
        batch_size=2,
        callbacks=cbs,
    )
    action_space = gym_spaces.Discrete(3)
    config = PlanConfig(horizon=3, receding_horizon=2, action_block=1)
    solver.configure(action_space=action_space, n_envs=4, config=config)

    out = solver.solve({'state': torch.zeros(4, 2)})

    assert 'callbacks' in out
    history = out['callbacks'][BestCostRecorder().output_key]
    # 4 envs / batch_size 2 = 2 batches; each batch logs n_steps=4 entries.
    assert len(history) == 2
    assert all(len(batch) == 4 for batch in history)


###########################
## Edge Cases            ##
###########################


def test_categorical_cem_horizon_1():
    """Solver with horizon=1."""
    solver = CategoricalCEMSolver(
        model=DummyCostModel(), n_steps=2, num_samples=8, topk=2
    )
    action_space = gym_spaces.Discrete(3)
    config = PlanConfig(horizon=1, receding_horizon=1, action_block=1)
    solver.configure(action_space=action_space, n_envs=2, config=config)

    out = solver.solve({'state': torch.zeros(2, 2)})

    assert out['actions'].shape == (2, 1, 1)


def test_categorical_cem_topk_equals_num_samples():
    """topk == num_samples is a valid (degenerate) configuration."""
    solver = CategoricalCEMSolver(
        model=DummyCostModel(), n_steps=2, num_samples=8, topk=8
    )
    action_space = gym_spaces.Discrete(3)
    config = PlanConfig(horizon=2, receding_horizon=1, action_block=1)
    solver.configure(action_space=action_space, n_envs=1, config=config)

    out = solver.solve({'state': torch.zeros(1, 2)})

    assert out['actions'].shape == (1, 2, 1)
