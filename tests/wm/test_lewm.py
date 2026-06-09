"""Regression tests for LeWM.get_cost goal_emb shape handling (PR1 G1/G2).

Shape contract (matching upstream commit 1986aae):
  get_cost stores goal_emb as (B, T, D)
  criterion broadcasts via goal_emb[:, None, -1:, :].expand_as(pred_emb)
"""

import torch

from stable_worldmodel.wm.lewm.lewm import LeWM

# CEM-like dimensions
B, S, T, D, H, A = 2, 3, 2, 5, 4, 2


def _make_info_dict():
    """Return a CEM-expanded info_dict mimicking what CEMSolver passes to get_cost."""
    return {
        'pixels': torch.randn(B, S, T, 3, 8, 8),
        'goal': torch.randn(B, S, T, 3, 8, 8),
        'action': torch.randn(B, S, H, A),
    }


def _make_action_candidates():
    return torch.randn(B, S, H, A)


def _bare_model():
    """Bypass LeWM.__init__; we only need get_cost to run."""
    return object.__new__(LeWM)


# ---------------------------------------------------------------------------
# G1: get_cost must store goal_emb as (B, T, D) for criterion to broadcast
# ---------------------------------------------------------------------------


def test_lewm_get_cost_stores_goal_emb_for_criterion_broadcast():
    """
    get_cost encodes the goal once (stripping the S axis with v[:, 0]) and
    stores goal['emb'] as (B, T, D).  criterion then inserts the S axis itself
    via goal_emb[:, None, -1:, :].expand_as(pred_emb).

    Before upstream commit 1986aae this was broken end-to-end (3D → 4D mismatch
    in the old expand_as path).  After 1986aae LeWM.criterion was fixed; our
    branch regression test guards that get_cost still delivers the 3D tensor
    that the fixed criterion expects.
    """
    torch.manual_seed(0)
    model = _bare_model()
    info_dict = _make_info_dict()
    action_candidates = _make_action_candidates()

    encode_call_count = []

    def mock_encode(goal_dict):
        encode_call_count.append(1)
        # get_cost strips the S axis with v[:, 0], so pixels must be (B, T, ...)
        assert goal_dict['pixels'].shape == (B, T, 3, 8, 8), (
            f'encode received wrong pixels shape: {goal_dict["pixels"].shape}'
        )
        return {'emb': torch.randn(B, T, D)}

    def mock_rollout(info, ac):
        info['predicted_emb'] = torch.randn(B, S, T, D)
        return info

    def mock_criterion(info):
        pred_emb = info['predicted_emb']  # (B, S, T, D)
        goal_emb = info['goal_emb']  # (B, T, D)

        assert goal_emb.shape == (B, T, D), (
            f'goal_emb shape mismatch: expected ({B},{T},{D}), '
            f'got {tuple(goal_emb.shape)}'
        )
        # verify the broadcast the real criterion will perform works
        broadcast_goal = goal_emb[:, None, -1:, :].expand_as(pred_emb)
        assert broadcast_goal.shape == pred_emb.shape

        return torch.zeros(B, S)

    model.encode = mock_encode
    model.rollout = mock_rollout
    model.criterion = mock_criterion

    cost = LeWM.get_cost(model, info_dict, action_candidates)

    assert len(encode_call_count) == 1, 'encode should be called exactly once'
    assert cost.shape == (B, S), (
        f'cost shape: expected ({B},{S}), got {cost.shape}'
    )


# ---------------------------------------------------------------------------
# G2 symmetry: pre-injected goal_emb must not be overwritten
# ---------------------------------------------------------------------------


def test_lewm_get_cost_respects_preinjected_goal_emb():
    """
    When goal_emb is already present in info_dict, encode must not be called
    and the stored tensor must pass through to criterion unchanged.
    Pre-injected goal_emb follows the same (B, T, D) contract.
    """
    torch.manual_seed(0)
    model = _bare_model()
    info_dict = _make_info_dict()
    preinjected = torch.randn(B, T, D)
    info_dict['goal_emb'] = preinjected
    action_candidates = _make_action_candidates()

    def encode_must_not_be_called(_):
        raise AssertionError(
            'LeWM.encode must not be called when goal_emb is already in info_dict'
        )

    def mock_rollout(info, ac):
        info['predicted_emb'] = torch.randn(B, S, T, D)
        return info

    def mock_criterion(info):
        assert info['goal_emb'] is preinjected, (
            'pre-injected goal_emb tensor was replaced instead of reused'
        )
        return torch.zeros(B, S)

    model.encode = encode_must_not_be_called
    model.rollout = mock_rollout
    model.criterion = mock_criterion

    cost = LeWM.get_cost(model, info_dict, action_candidates)
    assert cost.shape == (B, S)
