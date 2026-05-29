"""Tests for solver callbacks."""

import numpy as np
import pytest
import torch
from gymnasium import spaces as gym_spaces

from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver.callbacks import (
    ActionNormRecorder,
    BestCostRecorder,
    Callback,
    EliteCostRecorder,
    EliteSpreadRecorder,
    GradNormRecorder,
    MeanCostRecorder,
    MeanShiftRecorder,
    VarNormRecorder,
)
from stable_worldmodel.solver.cem import CEMSolver
from stable_worldmodel.solver.gd import GradientSolver
from stable_worldmodel.solver.icem import ICEMSolver


class DummyCostModel:
    """Quadratic cost model used across solver tests."""

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        return action_candidates.pow(2).sum(dim=(-1, -2))


###########################
## Base class            ##
###########################


def test_callback_invalid_reduction():
    with pytest.raises(ValueError, match='reduction'):
        BestCostRecorder(reduction='median')  # type: ignore[arg-type]


def test_callback_history_structure():
    """reset/start_batch/end_solve produce list[list[...]] (batches x steps)."""
    cb = BestCostRecorder()
    cb.reset()
    cb.start_batch()
    cb(step=0, costs=torch.tensor([[1.0, 0.5]]))
    cb(step=1, costs=torch.tensor([[0.3, 0.2]]))
    cb.start_batch()
    cb(step=0, costs=torch.tensor([[2.0, 1.0]]))
    cb.end_solve()
    assert len(cb.history) == 2
    assert len(cb.history[0]) == 2
    assert len(cb.history[1]) == 1


def test_callback_reset_clears_history():
    cb = MeanCostRecorder()
    cb.reset()
    cb.start_batch()
    cb(step=0, costs=torch.ones(1, 3))
    cb.end_solve()
    assert cb.history
    cb.reset()
    assert cb.history == []
    assert cb._current == []


def test_callback_output_key_default():
    assert BestCostRecorder().output_key == 'BestCostRecorder'


def test_callback_output_key_override():
    class Custom(Callback):
        name = 'my_metric'

        def compute(self, **state):  # type: ignore[no-untyped-def]
            return 1.0

    assert Custom().output_key == 'my_metric'


def test_callback_compute_not_implemented():
    cb = Callback()
    cb.start_batch()
    with pytest.raises(NotImplementedError):
        cb(step=0)


###########################
## Cost recorders        ##
###########################


@pytest.fixture
def costs_3x3():
    # (B=3 envs, N=3 samples)
    return torch.tensor([[1.0, 2.0, 0.5], [0.3, 0.9, 1.5], [4.0, 0.7, 2.0]])


def test_best_cost_recorder_mean(costs_3x3):
    cb = BestCostRecorder(reduction='mean')
    cb.reset()
    cb.start_batch()
    cb(step=0, costs=costs_3x3)
    cb.end_solve()
    # per-env mins: [0.5, 0.3, 0.7] -> mean = 0.5
    assert cb.history[0][0] == pytest.approx(0.5, abs=1e-6)


def test_best_cost_recorder_sum(costs_3x3):
    cb = BestCostRecorder(reduction='sum')
    cb.reset()
    cb.start_batch()
    cb(step=0, costs=costs_3x3)
    cb.end_solve()
    assert cb.history[0][0] == pytest.approx(1.5, abs=1e-6)


def test_best_cost_recorder_none(costs_3x3):
    cb = BestCostRecorder(reduction='none')
    cb.reset()
    cb.start_batch()
    cb(step=0, costs=costs_3x3)
    cb.end_solve()
    per_env = cb.history[0][0]
    assert isinstance(per_env, list)
    assert len(per_env) == 3
    assert per_env[0] == pytest.approx(0.5, abs=1e-6)
    assert per_env[1] == pytest.approx(0.3, abs=1e-6)
    assert per_env[2] == pytest.approx(0.7, abs=1e-6)


def test_mean_cost_recorder(costs_3x3):
    cb = MeanCostRecorder(reduction='none')
    cb.reset()
    cb.start_batch()
    cb(step=0, costs=costs_3x3)
    cb.end_solve()
    per_env = cb.history[0][0]
    assert per_env[0] == pytest.approx(costs_3x3[0].mean().item(), abs=1e-6)


###########################
## GD recorders          ##
###########################


def test_grad_norm_recorder_no_grad():
    cb = GradNormRecorder()
    cb.reset()
    cb.start_batch()
    p = torch.randn(2, 3, 4, 5)  # no grad
    cb(step=0, params=p)
    cb.end_solve()
    assert cb.history[0][0] == 0.0


def test_grad_norm_recorder_with_grad():
    p = torch.randn(2, 3, 4, 5, requires_grad=True)
    (p**2).sum().backward()
    # grad at p = 2p; norm per (B, N) sample = ||2p|| over (H*D)
    expected_per_env = p.grad.detach().flatten(2).norm(dim=-1).mean(dim=-1)

    for r, expected in [
        ('mean', expected_per_env.mean().item()),
        ('sum', expected_per_env.sum().item()),
    ]:
        cb = GradNormRecorder(reduction=r)  # type: ignore[arg-type]
        cb.reset()
        cb.start_batch()
        cb(step=0, params=p)
        cb.end_solve()
        assert cb.history[0][0] == pytest.approx(expected, rel=1e-5)

    cb = GradNormRecorder(reduction='none')
    cb.reset()
    cb.start_batch()
    cb(step=0, params=p)
    cb.end_solve()
    out = cb.history[0][0]
    assert isinstance(out, list) and len(out) == 2


def test_action_norm_recorder():
    p = torch.randn(2, 3, 4, 5)
    expected_per_env = p.flatten(2).norm(dim=-1).mean(dim=-1)
    cb = ActionNormRecorder(reduction='mean')
    cb.reset()
    cb.start_batch()
    cb(step=0, params=p)
    cb.end_solve()
    assert cb.history[0][0] == pytest.approx(
        expected_per_env.mean().item(), rel=1e-5
    )


###########################
## CEM recorders         ##
###########################


def test_elite_cost_recorder_returns_dict():
    v = torch.tensor([[1.0, 2.0, 0.5], [0.3, 0.9, 1.5]])  # (B=2, K=3)
    cb = EliteCostRecorder(reduction='mean')
    cb.reset()
    cb.start_batch()
    cb(step=0, topk_vals=v)
    cb.end_solve()
    entry = cb.history[0][0]
    assert set(entry.keys()) == {'mean', 'min', 'max'}
    assert entry['min'] == pytest.approx(0.4, abs=1e-6)  # mean of [0.5, 0.3]
    assert entry['max'] == pytest.approx(1.75, abs=1e-6)  # mean of [2.0, 1.5]


def test_elite_cost_recorder_none_per_env():
    v = torch.tensor([[1.0, 2.0, 0.5], [0.3, 0.9, 1.5]])
    cb = EliteCostRecorder(reduction='none')
    cb.reset()
    cb.start_batch()
    cb(step=0, topk_vals=v)
    cb.end_solve()
    entry = cb.history[0][0]
    assert isinstance(entry['min'], list) and len(entry['min']) == 2
    assert entry['min'][0] == pytest.approx(0.5, abs=1e-6)
    assert entry['min'][1] == pytest.approx(0.3, abs=1e-6)


def test_var_norm_recorder():
    var = torch.full((2, 4, 3), 0.25)  # (B, H, D)
    cb = VarNormRecorder(reduction='mean')
    cb.reset()
    cb.start_batch()
    cb(step=0, var=var)
    cb.end_solve()
    assert cb.history[0][0] == pytest.approx(0.25, abs=1e-6)


def test_mean_shift_recorder_first_step_returns_none():
    """No prev_mean on first step => callback skips append."""
    cb = MeanShiftRecorder()
    cb.reset()
    cb.start_batch()
    cb(step=0, mean=torch.zeros(2, 4, 3), prev_mean=None)
    cb.end_solve()
    assert cb.history == [] or cb.history == [[]]


def test_mean_shift_recorder_value():
    mean = torch.full((2, 4, 3), 0.5)
    prev = torch.zeros(2, 4, 3)
    # ||mean - prev|| flattened = sqrt(12 * 0.25) = sqrt(3) ~= 1.732
    cb = MeanShiftRecorder(reduction='mean')
    cb.reset()
    cb.start_batch()
    cb(step=0, mean=mean, prev_mean=prev)
    cb.end_solve()
    assert cb.history[0][0] == pytest.approx(np.sqrt(3.0), rel=1e-5)


def test_elite_spread_recorder():
    # All elites identical => spread is 0
    topk = torch.zeros(2, 5, 4, 3)
    cb = EliteSpreadRecorder(reduction='mean')
    cb.reset()
    cb.start_batch()
    cb(step=0, topk_candidates=topk)
    cb.end_solve()
    assert cb.history[0][0] == pytest.approx(0.0, abs=1e-6)


###########################
## Solver integration    ##
###########################


def _gd_solver(callbacks):
    solver = GradientSolver(
        model=DummyCostModel(),
        n_steps=3,
        num_samples=4,
        batch_size=2,
        callbacks=callbacks,
    )
    action_space = gym_spaces.Box(
        low=-1, high=1, shape=(2, 2), dtype=np.float32
    )
    config = PlanConfig(horizon=3, receding_horizon=2)
    solver.configure(action_space=action_space, n_envs=2, config=config)
    return solver


def _cem_solver(callbacks):
    solver = CEMSolver(
        model=DummyCostModel(),
        n_steps=3,
        num_samples=20,
        batch_size=2,
        topk=5,
        callbacks=callbacks,
    )
    action_space = gym_spaces.Box(
        low=-1, high=1, shape=(2, 2), dtype=np.float32
    )
    config = PlanConfig(horizon=3, receding_horizon=2)
    solver.configure(action_space=action_space, n_envs=2, config=config)
    return solver


def _icem_solver(callbacks):
    solver = ICEMSolver(
        model=DummyCostModel(),
        n_steps=3,
        num_samples=20,
        batch_size=2,
        topk=5,
        callbacks=callbacks,
    )
    action_space = gym_spaces.Box(
        low=-1, high=1, shape=(2, 2), dtype=np.float32
    )
    config = PlanConfig(horizon=3, receding_horizon=2)
    solver.configure(action_space=action_space, n_envs=2, config=config)
    return solver


def test_gd_solver_with_callbacks():
    cbs = [BestCostRecorder(), GradNormRecorder(), ActionNormRecorder()]
    solver = _gd_solver(cbs)
    info = {'pixels': torch.randn(2, 1, 3, 8, 8)}
    out = solver(info)

    assert 'callbacks' in out
    keys = set(out['callbacks'].keys())
    assert keys == {
        'BestCostRecorder',
        'GradNormRecorder',
        'ActionNormRecorder',
    }
    # 2 envs / batch_size 2 => 1 batch, 3 steps each
    for k in keys:
        assert len(out['callbacks'][k]) == 1
        assert len(out['callbacks'][k][0]) == 3


def test_gd_solver_no_callbacks_omits_key():
    solver = _gd_solver(None)
    info = {'pixels': torch.randn(2, 1, 3, 8, 8)}
    out = solver(info)
    assert 'callbacks' not in out


def test_cem_solver_with_callbacks():
    cbs = [
        BestCostRecorder(),
        EliteCostRecorder(),
        VarNormRecorder(),
        MeanShiftRecorder(),
        EliteSpreadRecorder(),
    ]
    solver = _cem_solver(cbs)
    info = {'pixels': torch.randn(2, 1, 3, 8, 8)}
    out = solver(info)

    assert 'callbacks' in out
    assert set(out['callbacks'].keys()) == {
        'BestCostRecorder',
        'EliteCostRecorder',
        'VarNormRecorder',
        'MeanShiftRecorder',
        'EliteSpreadRecorder',
    }
    assert len(out['callbacks']['BestCostRecorder'][0]) == 3
    # MeanShift always has a prev_mean (captured before each topk update)
    assert len(out['callbacks']['MeanShiftRecorder'][0]) == 3


def test_icem_solver_with_callbacks():
    cbs = [BestCostRecorder(), VarNormRecorder(), MeanShiftRecorder()]
    solver = _icem_solver(cbs)
    info = {'pixels': torch.randn(2, 1, 3, 8, 8)}
    out = solver(info)

    assert 'callbacks' in out
    assert set(out['callbacks'].keys()) == {
        'BestCostRecorder',
        'VarNormRecorder',
        'MeanShiftRecorder',
    }
    assert len(out['callbacks']['BestCostRecorder'][0]) == 3


def test_solver_callbacks_reset_between_solves():
    """Calling solve() twice should not accumulate history across calls."""
    cb = BestCostRecorder()
    solver = _cem_solver([cb])
    info = {'pixels': torch.randn(2, 1, 3, 8, 8)}
    solver(info)
    history_1 = list(cb.history)
    solver(info)
    history_2 = cb.history
    assert len(history_1) == len(history_2)  # not accumulating
