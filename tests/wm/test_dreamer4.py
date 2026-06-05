"""Integration test for DreamerV4WM: forward pass, loss, backward, rollout."""

import torch
import torch.nn as nn

from stable_worldmodel.wm.dreamer4.module import (
    Dynamics,
    dynamics_loss,
    make_schedule,
    sample_next_frame,
)
from stable_worldmodel.wm.dreamer4.dreamer4 import DreamerV4WM


# ---------------------------------------------------------------------------
# Minimal stub encoder that mimics stable_pretraining ViT output
# ---------------------------------------------------------------------------


class _FakeOutput:
    def __init__(self, hidden):
        self.last_hidden_state = hidden  # (B, seq_len, D)


class _FakeEncoder(nn.Module):
    def __init__(self, emb_dim: int = 32):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj = nn.Linear(3, emb_dim)  # 3 = C (input channels)

    def forward(self, x, **kwargs):
        x_flat = x.mean(dim=(-2, -1))  # (B*T, C) — spatial mean
        cls_vec = self.proj(x_flat)  # (B*T, emb_dim)
        hidden = cls_vec.unsqueeze(1).expand(-1, 4, -1)  # (B*T, 4, emb_dim)
        return _FakeOutput(hidden)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EMB_DIM = 32
D_SPATIAL = 16
ACTION_DIM = 3
K_MAX = 4  # small for fast tests
B, T = 4, 6


def make_model():
    encoder = _FakeEncoder(emb_dim=EMB_DIM)
    projector = nn.Linear(EMB_DIM, D_SPATIAL)
    dynamics = Dynamics(
        d_model=32,
        d_spatial=D_SPATIAL,
        n_spatial=1,
        n_register=2,
        n_heads=2,
        depth=2,
        k_max=K_MAX,
        action_dim=ACTION_DIM,
    )
    return DreamerV4WM(
        encoder=encoder,
        dynamics=dynamics,
        projector=projector,
        k_max=K_MAX,
        eval_K=1,
    )


def make_batch():
    return {
        'pixels': torch.randn(B, T, 3, 16, 16),
        'action': torch.randn(B, T, ACTION_DIM),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_encode():
    model = make_model()
    batch = make_batch()
    out = model.encode(batch)
    assert out['emb'].shape == (B, T, D_SPATIAL)
    assert out['packed_z'].shape == (B, T, 1, D_SPATIAL)


def test_dynamics_forward():
    dynamics = Dynamics(
        d_model=32,
        d_spatial=D_SPATIAL,
        n_spatial=1,
        n_register=2,
        n_heads=2,
        depth=2,
        k_max=K_MAX,
        action_dim=ACTION_DIM,
    )
    z = torch.randn(B, T, 1, D_SPATIAL)
    actions = torch.randn(B, T, ACTION_DIM)
    step_idx = torch.zeros(B, T, dtype=torch.long)
    sig_idx = torch.zeros(B, T, dtype=torch.long)
    out = dynamics(actions, step_idx, sig_idx, z)
    assert out.shape == (B, T, 1, D_SPATIAL)


def test_dynamics_loss_forward_backward():
    dynamics = Dynamics(
        d_model=32,
        d_spatial=D_SPATIAL,
        n_spatial=1,
        n_register=2,
        n_heads=2,
        depth=2,
        k_max=K_MAX,
        action_dim=ACTION_DIM,
    )
    z = torch.randn(B, T, 1, D_SPATIAL)
    actions = torch.randn(B, T, ACTION_DIM)

    loss, aux = dynamics_loss(dynamics, z1=z, actions=actions, k_max=K_MAX)
    assert torch.isfinite(loss)
    assert set(aux.keys()) == {
        'flow_mse',
        'bootstrap_mse',
        'loss_emp',
        'loss_self',
        'sigma_mean',
    }
    loss.backward()  # must not raise


def test_dynamics_loss_with_bootstrap():
    dynamics = Dynamics(
        d_model=32,
        d_spatial=D_SPATIAL,
        n_spatial=1,
        n_register=2,
        n_heads=2,
        depth=2,
        k_max=K_MAX,
        action_dim=ACTION_DIM,
    )
    # override zero-init on flow_head so predictions differ per call
    nn.init.normal_(dynamics.flow_head.weight, std=0.1)
    nn.init.normal_(dynamics.flow_head.bias, std=0.1)

    z = torch.randn(8, T, 1, D_SPATIAL)
    actions = torch.randn(8, T, ACTION_DIM)
    loss, aux = dynamics_loss(
        dynamics,
        z1=z,
        actions=actions,
        k_max=K_MAX,
        bootstrap_start=0,
        global_step=0,
        self_fraction=0.25,
    )
    assert aux['bootstrap_mse'].item() > 0, 'bootstrap should be nonzero'
    loss.backward()


def test_sample_next_frame():
    dynamics = Dynamics(
        d_model=32,
        d_spatial=D_SPATIAL,
        n_spatial=1,
        n_register=2,
        n_heads=2,
        depth=2,
        k_max=K_MAX,
        action_dim=ACTION_DIM,
    )
    past = torch.randn(B, T, 1, D_SPATIAL)
    actions = torch.randn(B, T, ACTION_DIM)  # same T, will be padded for T+1

    for K in [1, 2, K_MAX]:
        sched = make_schedule(k_max=K_MAX, K=K)
        z_next = sample_next_frame(
            dynamics,
            past_packed=past,
            sched=sched,
            actions=actions,
            k_max=K_MAX,
        )
        assert z_next.shape == (B, 1, D_SPATIAL), (
            f'K={K}: bad shape {z_next.shape}'
        )


def test_full_training_step():
    """Simulates one training iteration end-to-end on CPU."""
    model = make_model()
    batch = make_batch()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # encode (frozen encoder path: detach packed_z before dynamics loss)
    out = model.encode(batch)
    packed_z = out['packed_z']
    actions = batch['action']

    loss, aux = dynamics_loss(
        model.dynamics,
        z1=packed_z.detach(),
        actions=actions,
        k_max=K_MAX,
    )
    assert torch.isfinite(loss), f'loss is not finite: {loss}'
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f'\ntraining step OK: loss={loss.item():.4f}')


def test_rollout_shape():
    """Rollout returns (B, S, T, D_SPATIAL) with correct shape."""
    model = make_model()
    model.eval()

    H = 3  # history
    n_future = 4
    S = 2  # MPC samples

    info = {'pixels': torch.randn(B, H, 3, 16, 16)}
    action_candidates = torch.randn(B, S, H + n_future + 1, ACTION_DIM)

    with torch.no_grad():
        info = model.rollout(info, action_candidates, history_size=H)

    pred = info['predicted_emb']
    assert pred.shape[0] == B
    assert pred.shape[1] == S
    assert pred.shape[3] == D_SPATIAL


def test_no_nan_in_rollout():
    model = make_model()
    model.eval()
    H, S = 2, 3
    info = {'pixels': torch.randn(B, H, 3, 16, 16)}
    action_candidates = torch.randn(B, S, H + 3, ACTION_DIM)
    with torch.no_grad():
        info = model.rollout(info, action_candidates, history_size=H)
    assert not torch.isnan(info['predicted_emb']).any(), (
        'NaN in rollout output'
    )
