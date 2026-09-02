import pytest
import torch

from stable_worldmodel.wm.tdmpc2 import TDMPC2, tdmpc2_forward


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
