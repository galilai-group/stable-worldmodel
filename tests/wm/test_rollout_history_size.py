"""Regression tests for issue #225: rollout() must respect predictor.num_frames
instead of a hardcoded history_size default.
"""

import pytest
import torch
import torch.nn as nn

from stable_worldmodel.wm.lewm.lewm import LeWM
from stable_worldmodel.wm.pldm.pldm import PLDM


class _FakeEncoderOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, pixels, interpolate_pos_encoding=False):
        bt = pixels.shape[0]
        return _FakeEncoderOutput(
            torch.zeros(bt, 1, self.dim, dtype=self.dummy.dtype)
        )


class SpyPredictor(nn.Module):
    """Predictor stand-in that records the time-dim of every forward call."""

    def __init__(self, num_frames, dim):
        super().__init__()
        self.num_frames = num_frames
        self.dim = dim
        self.window_sizes: list[int] = []

    def forward(self, emb, act_emb):
        self.window_sizes.append(emb.shape[1])
        return emb.clone()


class LinearActionEncoder(nn.Module):
    def __init__(self, action_dim, emb_dim):
        super().__init__()
        self.lin = nn.Linear(action_dim, emb_dim)

    def forward(self, x):
        return self.lin(x)


def _build(cls, num_frames, dim=8, action_dim=2):
    model = cls(
        encoder=FakeEncoder(dim),
        predictor=SpyPredictor(num_frames=num_frames, dim=dim),
        action_encoder=LinearActionEncoder(action_dim, dim),
    )
    model.eval()
    return model


def _make_inputs(num_frames, action_dim=2, future_steps=2):
    B, S = 1, 1
    T_hist = num_frames
    T_total = T_hist + future_steps
    info = {'pixels': torch.zeros(B, S, T_hist, 3, 4, 4)}
    actions = torch.zeros(B, S, T_total, action_dim)
    return info, actions


@pytest.mark.parametrize('cls', [LeWM, PLDM])
def test_rollout_respects_predictor_num_frames(cls):
    """Without explicit history_size, rollout must use predictor.num_frames."""
    num_frames = 5
    model = _build(cls, num_frames=num_frames)
    info, actions = _make_inputs(num_frames=num_frames)

    with torch.no_grad():
        model.rollout(info, actions)

    windows = model.predictor.window_sizes
    assert windows, 'predict() was never called'
    assert all(w == num_frames for w in windows), (
        f'rollout used the wrong context window. saw {windows}, '
        f'expected all == predictor.num_frames ({num_frames}).'
    )


@pytest.mark.parametrize('cls', [LeWM, PLDM])
def test_rollout_explicit_history_size_overrides(cls):
    """Explicit history_size= still wins over predictor.num_frames (back-compat)."""
    num_frames = 5
    override = 2
    model = _build(cls, num_frames=num_frames)
    info, actions = _make_inputs(num_frames=num_frames)

    with torch.no_grad():
        model.rollout(info, actions, history_size=override)

    windows = model.predictor.window_sizes
    assert windows, 'predict() was never called'
    assert all(w == override for w in windows), (
        f'explicit history_size override was ignored. saw {windows}, '
        f'expected all == {override}.'
    )
