"""Tests for the PLDM Dynamics surface + ShootingCostEvaluator planning seam.

PLDM no longer exposes ``get_cost``; planning goes through
``stable_worldmodel.planning.ShootingCostEvaluator(model, GoalMSE())``. These tests
guard that PLDM still satisfies the ``Dynamics`` protocol and that goal encoding
happens once (as ``(B, T, D)``) and is reused when pre-injected. The bit-for-bit
parity of ``GoalMSE`` with the old ``criterion`` lives in
``tests/planning/test_evaluator.py``.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stable_worldmodel.planning import ShootingCostEvaluator, GoalMSE
from stable_worldmodel.protocols import Dynamics
from stable_worldmodel.wm.pldm.pldm import PLDM

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
    """Bypass PLDM.__init__; we only need the encode/rollout surface."""
    return object.__new__(PLDM)


def test_pldm_satisfies_dynamics_protocol():
    """PLDM exposes encode/rollout, so it is a Dynamics a ShootingCostEvaluator can wrap."""
    assert isinstance(_bare_model(), Dynamics)


def test_pldm_no_longer_exposes_get_cost():
    """Cost now lives in the ShootingCostEvaluator seam, not on the model."""
    assert not hasattr(_bare_model(), 'get_cost')


def test_cost_evaluator_encodes_goal_once_as_3d():
    """ShootingCostEvaluator(PLDM, GoalMSE) encodes the goal once and stores it (B, T, D)."""
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


###############################################
## Real rollout: history / candidate pairing ##
###############################################

# Small dims for the real-rollout tests. The toy predictor adds action
# embeddings to frame embeddings, so action dim == embedding dim.
RB, RS, RD = 2, 3, 4


class _StubViT(nn.Module):
    """ViT-like stub: cls token = first RD flattened pixel values."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, pixels, interpolate_pos_encoding=False):
        self.calls += 1
        bt = pixels.shape[0]
        feats = pixels.reshape(bt, -1)[:, :RD]
        return SimpleNamespace(
            last_hidden_state=torch.stack([feats, feats], dim=1)
        )


class _CumsumPredictor(nn.Module):
    """Causal toy predictor: output[t] = sum_{k<=t} (emb[k] + act[k])."""

    def __init__(self, num_frames=3):
        super().__init__()
        self.num_frames = num_frames

    def forward(self, emb, act_emb):
        return (emb + act_emb).cumsum(dim=1)


class _RecordingIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return x


def _toy_model():
    return PLDM(
        encoder=_StubViT(),
        predictor=_CumsumPredictor(),
        action_encoder=_RecordingIdentity(),
    )


def _encode_frames(pixels):
    """What _StubViT + identity projector yield per context frame."""
    b, t = pixels.shape[:2]
    return pixels.reshape(b * t, -1)[:, :RD].reshape(b, t, RD)


def _reference_rollout_legacy(emb_init, candidates, HS):
    """Hand-rolled pre-change PLDM rollout semantics at H=1."""
    emb = emb_init.clone()
    act = candidates[:, :1]
    for t in range(candidates.size(1) - 1):
        pred = (emb[:, -HS:] + act[:, -HS:]).cumsum(dim=1)[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
        act = torch.cat([act, candidates[:, t + 1 : t + 2]], dim=1)
    pred = (emb[:, -HS:] + act[:, -HS:]).cumsum(dim=1)[:, -1:]
    return torch.cat([emb, pred], dim=1)


def test_rollout_h1_matches_legacy_semantics():
    """With a single context frame the new contract is bit-identical to the
    old one (first candidate = action paired with the current frame)."""
    torch.manual_seed(0)
    model = _toy_model()
    info = {'pixels': torch.randn(RB, RS, 1, 3, 8, 8)}
    candidates = torch.randn(RB, RS, 5, RD)

    out = model.rollout(dict(info), candidates)

    emb0 = _encode_frames(info['pixels'][:, 0])  # (RB, 1, RD)
    emb0 = emb0.unsqueeze(1).expand(RB, RS, 1, RD).reshape(RB * RS, 1, RD)
    expected = _reference_rollout_legacy(
        emb0, candidates.reshape(RB * RS, 5, RD), HS=3
    )
    torch.testing.assert_close(
        out['predicted_emb'].reshape(RB * RS, -1, RD), expected
    )
    torch.testing.assert_close(out['action'], candidates[:, :, :1])


def test_rollout_consumes_past_actions_in_order():
    """The rollout's action sequence must be cat([action_history, candidates])."""
    torch.manual_seed(0)
    model = _toy_model()
    info = {'pixels': torch.randn(RB, RS, 3, 3, 8, 8)}
    past = torch.randn(RB, RS, 2, RD)
    candidates = torch.randn(RB, RS, 4, RD)
    info['action_history'] = past

    out = model.rollout(info, candidates)

    # last action_encoder call sees the full executed+candidate sequence
    seq = model.action_encoder.inputs[-1]  # (BS, H-1+T, A)
    expected = torch.cat([past, candidates], dim=2).reshape(RB * RS, 6, RD)
    torch.testing.assert_close(seq, expected)
    torch.testing.assert_close(
        out['action'], torch.cat([past, candidates[:, :, :1]], dim=2)
    )


def test_rollout_output_length_and_past_sensitivity():
    """Output holds H context + T predicted frames, and predictions react
    to the executed past actions (they are inputs, not dead weight)."""
    torch.manual_seed(0)
    model = _toy_model()
    pixels = torch.randn(RB, RS, 3, 3, 8, 8)
    candidates = torch.randn(RB, RS, 4, RD)

    info_a = {'pixels': pixels, 'action_history': torch.zeros(RB, RS, 2, RD)}
    out_a = model.rollout(info_a, candidates)['predicted_emb']

    info_b = {'pixels': pixels, 'action_history': torch.ones(RB, RS, 2, RD)}
    out_b = model.rollout(info_b, candidates)['predicted_emb']

    assert out_a.shape == (RB, RS, 3 + 4, RD)
    ctx = _encode_frames(pixels[:, 0])  # (RB, 3, RD)
    torch.testing.assert_close(
        out_a[:, :, :3], ctx.unsqueeze(1).expand(RB, RS, 3, RD)
    )
    assert not torch.allclose(out_a[:, :, 3:], out_b[:, :, 3:])


def test_rollout_rejects_mismatched_action_history():
    model = _toy_model()
    info = {
        'pixels': torch.randn(RB, RS, 3, 3, 8, 8),
        'action_history': torch.randn(RB, RS, 1, RD),  # needs H-1 = 2
    }
    with pytest.raises(AssertionError, match='action_history'):
        model.rollout(info, torch.randn(RB, RS, 4, RD))


def test_rollout_rejects_multiframe_pixels_without_action_history():
    """H > 1 pixels without executed past actions must fail loudly rather
    than silently pairing context frames with optimizer candidates."""
    model = _toy_model()
    info = {'pixels': torch.randn(RB, RS, 3, 3, 8, 8)}
    with pytest.raises(AssertionError, match='action_history'):
        model.rollout(info, torch.randn(RB, RS, 4, RD))
