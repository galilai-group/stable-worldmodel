import numpy as np
import pytest
import torch

from stable_worldmodel.wm.tdmpc2 import (
    TDMPC2,
    shift_reward_to_action_alignment,
    tdmpc2_forward,
    uses_collect_reward_convention,
)


class _Config(dict):
    __getattr__ = dict.__getitem__


class _ForwardContext:
    def __init__(self, model):
        self.model = model

    def log_dict(self, *_args, **_kwargs):
        pass


def _make_config():
    wm = _Config(
        encoding={'observation': 8},
        horizon=3,
        mlp_dim=16,
        enc_dim=16,
        simnorm_dim=8,
        num_q=2,
        rho=0.5,
        tau=0.01,
        consistency_coef=20.0,
        reward_coef=0.1,
        value_coef=0.1,
        discount=0.99,
        entropy_coef=1e-4,
        num_bins=11,
        vmin=-6,
        vmax=2,
    )
    return _Config(
        action_dim=2,
        extra_dims={'observation': 3},
        wm=wm,
    )


def _make_batch(cfg):
    generator = torch.Generator().manual_seed(7)
    batch_size = 8
    num_steps = cfg.wm.horizon + 1
    return {
        'observation': torch.randn(
            batch_size, num_steps, 3, generator=generator
        ),
        'action': torch.randn(
            batch_size,
            num_steps,
            cfg.action_dim,
            generator=generator,
        ).tanh(),
        'reward': torch.randn(batch_size, num_steps, generator=generator),
    }


def _target_q_state(model):
    return torch.cat(
        [p.detach().flatten() for p in model.target_qs.parameters()]
    ).clone()


def test_validation_forward_keeps_running_scale_frozen():
    cfg = _make_config()
    model = TDMPC2(cfg)
    scale_before = model.scale.value.clone()
    target_q_before = _target_q_state(model)

    output = tdmpc2_forward(
        _ForwardContext(model),
        _make_batch(cfg),
        stage='validate',
        cfg=cfg,
    )

    assert torch.equal(model.scale.value, scale_before)
    assert torch.equal(_target_q_state(model), target_q_before)
    assert torch.isfinite(output['loss'])


# 'fit' is what stable_pretraining.Module.training_step passes; 'train' is
# what the online loop in scripts/expert/tdmpc2_online.py passes.
@pytest.mark.parametrize('stage', ['train', 'fit'])
def test_training_forward_updates_running_scale(stage):
    cfg = _make_config()
    model = TDMPC2(cfg)
    scale_before = model.scale.value.clone()
    target_q_before = _target_q_state(model)

    tdmpc2_forward(
        _ForwardContext(model),
        _make_batch(cfg),
        stage=stage,
        cfg=cfg,
    )

    assert not torch.equal(model.scale.value, scale_before)
    assert not torch.equal(_target_q_state(model), target_q_before)


def test_collect_reward_convention_detection():
    # World.collect layout: reward[t] = r(s_{t-1}, a_{t-1}), NaN on the
    # first row of every episode.
    reward = np.array([np.nan, 1.0, 2.0, np.nan, 4.0, 5.0])
    offsets = np.array([0, 3])
    assert uses_collect_reward_convention(reward, offsets)

    # Online-buffer layout: real reward on every episode's first row.
    aligned = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert not uses_collect_reward_convention(aligned, offsets)

    # Partial NaN only (not the collect fingerprint) must not trigger.
    partial = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    assert not uses_collect_reward_convention(partial, offsets)

    # 2D reward column (N, dim).
    reward_2d = np.stack([reward, reward], axis=-1)
    assert uses_collect_reward_convention(reward_2d, offsets)


def test_shift_reward_to_action_alignment():
    collect = torch.tensor([[np.nan], [1.0], [2.0], [3.0]])
    shifted = shift_reward_to_action_alignment(collect)
    # Row t now holds the reward stored on row t+1, i.e. r(s_t, a_t);
    # the last row holds its own finite value and no NaN survives.
    # The trailing size-1 dim is dropped (Lance stores scalars as (T, 1)).
    assert shifted.shape == (4,)
    assert torch.equal(shifted, torch.tensor([1.0, 2.0, 3.0, 3.0]))
    assert torch.isfinite(shifted).all()

    # Multi-dim rewards keep their trailing dims.
    multi = torch.tensor([[np.nan, 10.0], [1.0, 11.0], [2.0, 12.0]])
    assert torch.equal(
        shift_reward_to_action_alignment(multi),
        torch.tensor([[1.0, 11.0], [2.0, 12.0], [2.0, 12.0]]),
    )
    # Single-row window is returned unchanged.
    single = torch.tensor([np.nan])
    out = shift_reward_to_action_alignment(single)
    assert out.shape == single.shape
    assert torch.allclose(out, single, equal_nan=True)


def test_collect_convention_reward_breaks_loss_until_realigned():
    """Red/green for #357: a batch with World.collect rewards (NaN on the
    episode's first row, reward one step behind the action) makes the loss
    NaN; shifting rewards back into action alignment restores a finite loss."""
    cfg = _make_config()
    model = TDMPC2(cfg)
    batch = _make_batch(cfg)
    batch_size = batch['reward'].shape[0]

    # Emulate World.collect: reward[t] = reward of row t-1, NaN on row 0.
    collect = batch['reward'].clone()
    collect = torch.cat(
        [torch.full((batch_size, 1), float('nan')), collect[:, :-1]], dim=1
    )
    batch['reward'] = collect

    output = tdmpc2_forward(
        _ForwardContext(model), batch, stage='validate', cfg=cfg
    )
    assert not torch.isfinite(output['loss'])

    # Fix: shift rewards one row back within each window (the fix applied by
    # scripts/train/tdmpc2.py for collect-recorded datasets, per sample).
    batch['reward'] = torch.stack(
        [
            shift_reward_to_action_alignment(collect[b])
            for b in range(collect.shape[0])
        ]
    )
    output = tdmpc2_forward(
        _ForwardContext(model), batch, stage='validate', cfg=cfg
    )
    assert torch.isfinite(output['loss'])
