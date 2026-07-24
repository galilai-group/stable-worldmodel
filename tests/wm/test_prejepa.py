"""Tests for the PreJEPA rollout contract.

Candidates are strictly future; executed past action blocks arrive via
``info['action_history']`` and are injected into the context-frame
embeddings (``replace_action_in_embedding``) in place of the old
convention that consumed the first ``n_obs`` optimizer candidates as
past actions.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stable_worldmodel.wm.prejepa.prejepa import PreJEPA

# Small dims: batch, samples, patches, pixel-embedding dim, action dim
RB, RS, RP, RDP, RA = 2, 3, 2, 4, 2


class _StubBackbone(nn.Module):
    """ViT-like stub: every token = first RDP flattened pixel values."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, pixels, interpolate_pos_encoding=False):
        self.calls += 1
        bt = pixels.shape[0]
        feats = pixels.reshape(bt, -1)[:, :RDP]
        # cls + RP patches; _encode_image drops the cls token
        hidden = torch.stack([feats] * (1 + RP), dim=1)
        return SimpleNamespace(last_hidden_state=hidden)


class _FlatCumsum(nn.Module):
    """Causal toy predictor over the flattened (t p) sequence dim."""

    def forward(self, x):
        return x.cumsum(dim=1)


class _ActionEnc(nn.Module):
    emb_dim = RA

    def forward(self, x):
        return x


def _toy_model():
    return PreJEPA(
        encoder=_StubBackbone(),
        predictor=_FlatCumsum(),
        extra_encoders={'action': _ActionEnc()},
        history_size=3,
    )


def _rollout_info(n_obs, id_val=7, step_idx=0):
    return {
        'pixels': torch.randn(RB, RS, n_obs, 3, 8, 8),
        'id': torch.full((RB, RS, 1), id_val, dtype=torch.int64),
        'step_idx': torch.full((RB, RS, 1), step_idx, dtype=torch.int64),
    }


def test_rollout_consumes_past_actions():
    """Context frames get the executed blocks + first candidate injected."""
    torch.manual_seed(0)
    model = _toy_model()
    info = _rollout_info(n_obs=3)
    past = torch.randn(RB, RS, 2, RA)
    candidates = torch.randn(RB, RS, 4, RA)
    info['action_history'] = past

    injected = []
    orig = model.replace_action_in_embedding

    def spy(embedding, act):
        injected.append(act.detach().clone())
        return orig(embedding, act)

    model.replace_action_in_embedding = spy
    out = model.rollout(info, candidates)

    expected_ctx = torch.cat([past, candidates[:, :, :1]], dim=2)
    torch.testing.assert_close(injected[0], expected_ctx)
    torch.testing.assert_close(out['action'], expected_ctx)
    # remaining injections are the future candidates, one per step
    assert len(injected) == 1 + (candidates.size(2) - 1)


def test_rollout_output_length():
    """Output holds n_obs context + T predicted latent states."""
    torch.manual_seed(0)
    model = _toy_model()
    info = _rollout_info(n_obs=3)
    info['action_history'] = torch.randn(RB, RS, 2, RA)
    candidates = torch.randn(RB, RS, 4, RA)

    out = model.rollout(info, candidates)

    assert out['predicted_embedding'].shape == (
        RB,
        RS,
        3 + 4,
        RP,
        RDP + RA,
    )


def test_rollout_h1_without_action_history():
    """Single-frame context needs no action_history; the first candidate
    is the action paired with the current frame (legacy behavior)."""
    torch.manual_seed(0)
    model = _toy_model()
    info = _rollout_info(n_obs=1)
    candidates = torch.randn(RB, RS, 5, RA)

    out = model.rollout(info, candidates)

    assert out['predicted_embedding'].shape == (
        RB,
        RS,
        1 + 5,
        RP,
        RDP + RA,
    )
    torch.testing.assert_close(out['action'], candidates[:, :, :1])


def test_rollout_rejects_mismatched_action_history():
    model = _toy_model()
    info = _rollout_info(n_obs=3)
    info['action_history'] = torch.randn(RB, RS, 1, RA)  # needs n-1 = 2
    with pytest.raises(AssertionError, match='action_history'):
        model.rollout(info, torch.randn(RB, RS, 4, RA))


def test_rollout_rejects_multiframe_pixels_without_action_history():
    model = _toy_model()
    info = _rollout_info(n_obs=3)
    with pytest.raises(AssertionError, match='action_history'):
        model.rollout(info, torch.randn(RB, RS, 4, RA))


def test_cache_reused_within_step_with_history():
    """The (id, step_idx) context-encoding cache stays valid under n>1:
    two rollouts at the same step encode pixels exactly once."""
    torch.manual_seed(0)
    model = _toy_model()
    pixels = torch.randn(RB, RS, 3, 3, 8, 8)
    past = torch.randn(RB, RS, 2, RA)

    for _ in range(2):
        info = _rollout_info(n_obs=3)
        info['pixels'] = pixels
        info['action_history'] = past
        model.rollout(info, torch.randn(RB, RS, 4, RA))

    assert model.backbone.calls == 1
