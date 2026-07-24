"""Tests for the pluggable planning objectives.

Direct-behavior tests for the concrete
:class:`~stable_worldmodel.protocols.Objective` implementations in
:mod:`stable_worldmodel.planning.objective` (GoalMSE / ControlPenalty /
WeightedSum). Their use inside the ``ShootingCostEvaluator`` seam is covered in
``test_evaluator.py``.
"""

import pytest
import torch
import torch.nn.functional as F

from stable_worldmodel.planning import (
    ControlPenalty,
    GoalMSE,
    WeightedSum,
)

# CEM-like dimensions.
B, S, T, D, H, A = 2, 3, 2, 5, 2, 4


def test_goal_mse_formula():
    pred = torch.randn(B, S, T, D)
    goal = torch.randn(B, T, D)
    out = GoalMSE()({'predicted_emb': pred, 'goal_emb': goal})

    goal_b = goal[:, None, -1:, :].expand_as(pred)
    expected = F.mse_loss(
        pred[..., -1:, :], goal_b[..., -1:, :], reduction='none'
    ).sum(dim=tuple(range(2, pred.ndim)))
    torch.testing.assert_close(out, expected)


def test_control_penalty_formula():
    ac = torch.randn(B, S, H, A)
    out = ControlPenalty()({'action_candidates': ac})
    torch.testing.assert_close(out, ac.pow(2).sum(dim=(2, 3)))


def test_weighted_sum_combines_terms():
    ac = torch.randn(B, S, H, A)
    info = {'action_candidates': ac}
    combined = WeightedSum([(2.0, ControlPenalty()), (0.5, ControlPenalty())])
    torch.testing.assert_close(combined(info), 2.5 * ControlPenalty()(info))


def test_weighted_sum_requires_terms():
    with pytest.raises(ValueError):
        WeightedSum([])
