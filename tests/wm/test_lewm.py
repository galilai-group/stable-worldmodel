"""Tests for the LeWM Dynamics surface + ShootingCostEvaluator planning seam.

LeWM no longer exposes ``get_cost``; planning goes through
``stable_worldmodel.planning.ShootingCostEvaluator(model, GoalMSE())``. These tests
guard that LeWM still satisfies the ``Dynamics`` protocol and that goal encoding
happens once (as ``(B, T, D)``) and is reused when pre-injected. The bit-for-bit
parity of ``GoalMSE`` with the old ``criterion`` lives in
``tests/planning/test_evaluator.py``.
"""

import torch

from stable_worldmodel.planning import ShootingCostEvaluator, GoalMSE
from stable_worldmodel.protocols import Dynamics
from stable_worldmodel.wm.lewm.lewm import LeWM

# CEM-like dimensions
B, S, T, D, H, A = 2, 3, 2, 5, 4, 2


def _make_info_dict():
    """Return a CEM-expanded info_dict mimicking what a solver passes to get_cost."""
    return {
        'pixels': torch.randn(B, S, T, 3, 8, 8),
        'goal': torch.randn(B, S, T, 3, 8, 8),
        'action': torch.randn(B, S, H, A),
    }


def _make_action_candidates():
    return torch.randn(B, S, H, A)


def _bare_model():
    """Bypass LeWM.__init__; we only need the encode/rollout surface."""
    return object.__new__(LeWM)


def test_lewm_satisfies_dynamics_protocol():
    """LeWM exposes encode/rollout, so it is a Dynamics a ShootingCostEvaluator can wrap."""
    assert isinstance(_bare_model(), Dynamics)


def test_lewm_no_longer_exposes_get_cost():
    """Cost now lives in the ShootingCostEvaluator seam, not on the model."""
    assert not hasattr(_bare_model(), 'get_cost')


def test_cost_evaluator_encodes_goal_once_as_3d():
    """ShootingCostEvaluator(LeWM, GoalMSE) encodes the goal once and stores it (B, T, D).

    get_cost strips the S axis with v[:, 0] before encoding; GoalMSE then
    re-inserts it via goal_emb[:, None, -1:, :].expand_as(pred_emb).
    """
    torch.manual_seed(0)
    model = _bare_model()
    info_dict = _make_info_dict()
    action_candidates = _make_action_candidates()

    encode_calls = []

    def mock_encode(goal_dict):
        encode_calls.append(1)
        assert goal_dict['pixels'].shape == (B, T, 3, 8, 8), (
            f'encode received wrong pixels shape: {goal_dict["pixels"].shape}'
        )
        return {'emb': torch.randn(B, T, D)}

    def mock_rollout(info, ac):
        info['predicted_emb'] = torch.randn(B, S, T, D)
        return info

    model.encode = mock_encode
    model.rollout = mock_rollout

    cost = ShootingCostEvaluator(model, GoalMSE()).get_cost(
        info_dict, action_candidates
    )

    assert len(encode_calls) == 1, 'encode should be called exactly once'
    assert cost.shape == (B, S), (
        f'cost shape: expected ({B},{S}), got {cost.shape}'
    )


def test_cost_evaluator_respects_preinjected_goal_emb():
    """A pre-populated goal_emb is reused; encode is not called again."""
    torch.manual_seed(0)
    model = _bare_model()
    info_dict = _make_info_dict()
    info_dict['goal_emb'] = torch.randn(B, T, D)
    action_candidates = _make_action_candidates()

    def encode_must_not_be_called(_):
        raise AssertionError(
            'encode must not be called when goal_emb is already in info_dict'
        )

    def mock_rollout(info, ac):
        info['predicted_emb'] = torch.randn(B, S, T, D)
        return info

    model.encode = encode_must_not_be_called
    model.rollout = mock_rollout

    cost = ShootingCostEvaluator(model, GoalMSE()).get_cost(
        info_dict, action_candidates
    )
    assert cost.shape == (B, S)
