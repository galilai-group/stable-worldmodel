"""Tests for the pluggable planning cost layer (ShootingCostEvaluator / Objective).

Covers:
  * parity — ShootingCostEvaluator(model, GoalMSE()) reproduces the monolithic
    get_cost the world models used to implement inline;
  * feature-detected constraints and end-to-end solving through CEMSolver and
    LagrangianSolver (the previously-unreachable constraint path).

The objectives themselves (GoalMSE / ControlPenalty / WeightedSum) are tested
directly in ``test_objective.py``.
"""

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces as gym_spaces

from stable_worldmodel.planning import (
    ControlPenalty,
    GoalMSE,
    ShootingCostEvaluator,
)
from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.planning.solver.cem import CEMSolver
from stable_worldmodel.planning.solver.lagrangian import LagrangianSolver

# CEM-like dimensions; H == T keeps the fake rollout a plain projection.
B, S, T, D, H, A = 2, 3, 2, 5, 2, 4


class FakeLeWM(torch.nn.Module):
    """Deterministic stand-in exposing the LeWM Dynamics + criterion surface.

    ``reference_get_cost`` is a verbatim copy of the *old* monolithic
    ``LeWM.get_cost`` — the parity target the ShootingCostEvaluator seam must match.
    """

    def __init__(self, dim: int = D) -> None:
        super().__init__()
        self.dim = dim
        # Real parameter so ShootingCostEvaluator.parameters() -> dtype inference works.
        self.proj = torch.nn.Linear(A, dim)

    def encode(self, info: dict) -> dict:
        pixels = info['pixels']  # (B, T, C, Hh, Ww)
        emb = pixels.flatten(start_dim=2).mean(dim=-1)  # (B, T)
        info['emb'] = (
            emb.unsqueeze(-1).expand(*emb.shape, self.dim).contiguous()
        )
        return info

    def rollout(self, info: dict, action_candidates: torch.Tensor) -> dict:
        # (B, S, H, A) -> (B, S, T, D), differentiable in the actions.
        info['predicted_emb'] = self.proj(action_candidates)
        return info

    def criterion(self, info: dict) -> torch.Tensor:
        """Verbatim copy of LeWM.criterion — the parity target for GoalMSE."""
        pred_emb = info['predicted_emb']  # (B, S, T, D)
        goal_emb = info['goal_emb']  # (B, T, D)
        goal_emb = goal_emb[:, None, -1:, :].expand_as(pred_emb)
        return F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction='none',
        ).sum(dim=tuple(range(2, pred_emb.ndim)))

    def reference_get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """The old monolithic get_cost (goal-encode + rollout + criterion)."""
        assert 'goal' in info_dict, 'goal not in info_dict'
        if 'goal_emb' not in info_dict:
            goal = {
                k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)
            }
            goal['pixels'] = goal['goal']
            for k in list(info_dict):
                if k.startswith('goal_'):
                    goal[k[len('goal_') :]] = goal.pop(k)
            goal.pop('action')
            goal = self.encode(goal)
            info_dict['goal_emb'] = goal['emb']
        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)


def _make_info_dict():
    return {
        'pixels': torch.randn(B, S, T, 3, 8, 8),
        'goal': torch.randn(B, S, T, 3, 8, 8),
        'action': torch.randn(B, S, H, A),
    }


def _clone(info):
    return {
        k: (v.clone() if torch.is_tensor(v) else v) for k, v in info.items()
    }


def _make_action_candidates():
    return torch.randn(B, S, H, A)


# ---------------------------------------------------------------------------
# Parity: the seam reproduces the old monolithic get_cost
# ---------------------------------------------------------------------------


def test_cost_evaluator_matches_monolithic_get_cost():
    """ShootingCostEvaluator(model, GoalMSE()) == old inline get_cost, bit-for-bit."""
    torch.manual_seed(0)
    model = FakeLeWM()
    info = _make_info_dict()
    ac = _make_action_candidates()

    reference = model.reference_get_cost(_clone(info), ac)
    seam = ShootingCostEvaluator(model, GoalMSE()).get_cost(_clone(info), ac)

    assert seam.shape == (B, S)
    torch.testing.assert_close(seam, reference)


def test_cost_evaluator_respects_preinjected_goal_emb():
    """A pre-populated goal_emb is reused; encode is not called again."""
    torch.manual_seed(0)
    model = FakeLeWM()
    info = _make_info_dict()
    info['goal_emb'] = torch.randn(B, T, D)
    ac = _make_action_candidates()

    def _fail(_):
        raise AssertionError('encode must not run when goal_emb is present')

    model.encode = _fail  # type: ignore[method-assign]
    seam = ShootingCostEvaluator(model, GoalMSE()).get_cost(_clone(info), ac)
    assert seam.shape == (B, S)


def test_encode_goal_none_skips_goal_encoding():
    """encode_goal=None skips goal handling entirely (no 'goal' required)."""
    torch.manual_seed(0)
    model = FakeLeWM()
    ac = _make_action_candidates()

    cost = ShootingCostEvaluator(
        model, ControlPenalty(), encode_goal=None
    ).get_cost({}, ac)
    assert cost.shape == (B, S)


def test_goal_encode_drops_action_history():
    """Executed past actions are planning context, not goal content: the
    goal-encode dict must never carry ``action_history``."""
    torch.manual_seed(0)
    model = FakeLeWM()
    info = _make_info_dict()
    info['action_history'] = torch.randn(B, S, T - 1, A)
    ac = _make_action_candidates()

    seen = []
    orig_encode = model.encode

    def spy(goal_dict):
        seen.append(set(goal_dict))
        return orig_encode(goal_dict)

    model.encode = spy  # type: ignore[method-assign]
    cost = ShootingCostEvaluator(model, GoalMSE()).get_cost(_clone(info), ac)
    assert cost.shape == (B, S)
    assert seen and all('action_history' not in keys for keys in seen)


# ---------------------------------------------------------------------------
# Constraints: feature-detected via attribute presence
# ---------------------------------------------------------------------------


def test_get_constraints_absent_without_constraints():
    model = FakeLeWM()
    assert 'get_constraints' not in vars(
        ShootingCostEvaluator(model, GoalMSE())
    )


def test_get_constraints_present_with_constraints():
    model = FakeLeWM()
    ev = ShootingCostEvaluator(
        model, GoalMSE(), constraints=[ControlPenalty()]
    )
    assert 'get_constraints' in vars(ev)


def test_get_constraints_stacks_terms():
    torch.manual_seed(0)
    model = FakeLeWM()
    ac = _make_action_candidates()
    ev = ShootingCostEvaluator(
        model,
        ControlPenalty(),
        constraints=[ControlPenalty(), ControlPenalty()],
        encode_goal=None,
    )
    constraints = ev.get_constraints({}, ac)
    assert constraints.shape == (B, S, 2)


# ---------------------------------------------------------------------------
# Integration: solve end-to-end through the unmodified solvers
# ---------------------------------------------------------------------------


def _configure(solver, n_envs=2, horizon=H, action_dim=A):
    action_space = gym_spaces.Box(
        low=-np.inf, high=np.inf, shape=(1, action_dim), dtype=np.float32
    )
    config = PlanConfig(horizon=horizon, receding_horizon=1, action_block=1)
    solver.configure(action_space=action_space, n_envs=n_envs, config=config)


def test_cem_solves_through_cost_evaluator():
    """CEMSolver treats ShootingCostEvaluator as a plain Costable and returns actions."""
    torch.manual_seed(0)
    model = FakeLeWM()
    evaluator = ShootingCostEvaluator(
        model, ControlPenalty(), encode_goal=None
    )

    solver = CEMSolver(
        cost=evaluator, num_samples=32, n_steps=3, topk=8, seed=0
    )
    _configure(solver, n_envs=2)

    out = solver.solve({'dummy': torch.zeros(2, 1)})
    assert out['actions'].shape == (2, H, A)


def test_lagrangian_solves_with_cost_evaluator_constraints():
    """The evaluator feeds constraints into the previously-unreachable path."""
    torch.manual_seed(0)
    model = FakeLeWM()

    class MeanActionBudget:
        """g = mean(action) - 0.5 <= 0 — a differentiable constraint term."""

        def __call__(self, info_dict):
            a = info_dict['action_candidates']
            return a.mean(dim=(2, 3)) - 0.5

    evaluator = ShootingCostEvaluator(
        model,
        ControlPenalty(),
        constraints=[MeanActionBudget()],
        encode_goal=None,
    )

    solver = LagrangianSolver(
        cost=evaluator, n_steps=4, n_outer_steps=2, num_samples=4, seed=0
    )
    _configure(solver, n_envs=2)

    out = solver.solve({})
    assert out['actions'].shape == (2, H, A)
    assert out['lambdas'] is not None
    assert out['lambdas'].shape == (2, 1)
    assert (out['lambdas'] >= 0).all()
