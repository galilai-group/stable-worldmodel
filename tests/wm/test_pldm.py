"""Regression tests for PLDM.get_cost goal_emb shape handling (PR1 G1/G2).

Shape contract (matching upstream LeWM contract from commit 1986aae):
  get_cost stores goal_emb as (B, T, D)
  criterion broadcasts via goal_emb[:, None, -1:, :].expand_as(pred_emb)
"""

import torch

from stable_worldmodel.wm.pldm.pldm import PLDM

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
    """Bypass PLDM.__init__; we only need get_cost to run."""
    return object.__new__(PLDM)


# ---------------------------------------------------------------------------
# G1: get_cost must store goal_emb as (B, T, D) for criterion to broadcast
# ---------------------------------------------------------------------------


def test_pldm_get_cost_stores_goal_emb_for_criterion_broadcast():
    """
    PLDM.get_cost encodes the goal once (stripping the S axis with v[:, 0])
    and stores goal['emb'] as (B, T, D).  PLDM.criterion then inserts the S
    axis itself via goal_emb[:, None, -1:, :].expand_as(pred_emb), matching
    the LeWM contract introduced upstream in commit 1986aae.

    Before this PR: PLDM.get_cost expanded to (B, S, T, D) while
    PLDM.criterion tried expand_as on that 4D tensor against a differently
    shaped pred_emb — a shape mismatch.
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

    cost = PLDM.get_cost(model, info_dict, action_candidates)

    assert len(encode_call_count) == 1, 'encode should be called exactly once'
    assert cost.shape == (B, S), (
        f'cost shape: expected ({B},{S}), got {cost.shape}'
    )


# ---------------------------------------------------------------------------
# G2: missing guard — encode must not be called when goal_emb is pre-injected
# ---------------------------------------------------------------------------


def test_pldm_get_cost_respects_preinjected_goal_emb():
    """
    Before this PR, PLDM.get_cost had no 'if goal_emb not in info_dict' guard,
    so it always re-encoded the goal and overwrote any pre-injected value.

    After this PR, get_cost must skip encoding entirely and leave the
    pre-injected (B, T, D) tensor untouched.
    """
    torch.manual_seed(0)
    model = _bare_model()
    info_dict = _make_info_dict()
    preinjected = torch.randn(B, T, D)
    info_dict['goal_emb'] = preinjected
    action_candidates = _make_action_candidates()

    def encode_must_not_be_called(_):
        raise AssertionError(
            'PLDM.encode must not be called when goal_emb is already in info_dict'
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

    cost = PLDM.get_cost(model, info_dict, action_candidates)
    assert cost.shape == (B, S)
