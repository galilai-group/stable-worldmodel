"""Tests for the PreJEPA rollout contract.

Candidates are strictly future; executed past action blocks arrive via
``info['action_history']`` and are injected into the context-frame
embeddings (``replace_action_in_embedding``) in place of the old
convention that consumed the first ``n_obs`` optimizer candidates as
past actions.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stable_worldmodel.planning import (
    GoalMSE,
    ShootingCostEvaluator,
    WeightedSum,
    split_goal_encode,
)
from stable_worldmodel.protocols import Dynamics
from stable_worldmodel.wm.prejepa.prejepa import PreJEPA

# tests/wm/test_prejepa.py -> repo root, so file loads don't depend on the cwd
REPO_ROOT = Path(__file__).resolve().parents[2]

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

    assert out['predicted_emb'].shape == (
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

    assert out['predicted_emb'].shape == (
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


##################################################
## Planning seam: ShootingCostEvaluator parity  ##
##################################################

# PreJEPA no longer exposes get_cost. Its latent concatenates pixels + one
# embedding per extra encoder, so the goal is built by split_goal_encode and
# scored with one GoalMSE per source combined by WeightedSum. These tests pin
# that composition to the numbers the deleted PreJEPA.get_cost produced.

RPR = 3  # proprio embedding dim


class _ExtraEnc(nn.Module):
    """Identity extra encoder; ``emb_dim`` is read when injecting actions."""

    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, x):
        return x


class _MixingCumsum(nn.Module):
    """Causal over the sequence *and* mixing across features, so injected
    actions actually reach the pixel dims (an elementwise predictor would
    make the parity check vacuous)."""

    def __init__(self, dim=RDP + RPR + RA):
        super().__init__()
        gen = torch.Generator().manual_seed(3)
        self.mix = nn.Parameter(torch.randn(dim, dim, generator=gen) * 0.3)

    def forward(self, x):
        return x.cumsum(dim=1) @ self.mix


def _split_model():
    return PreJEPA(
        encoder=_StubBackbone(),
        predictor=_MixingCumsum(),
        extra_encoders=nn.ModuleDict(
            {'proprio': _ExtraEnc(RPR), 'action': _ExtraEnc(RA)}
        ),
        history_size=3,
    )


def _split_info(n_obs=3, n_goal=1):
    torch.manual_seed(1)
    return {
        'pixels': torch.randn(RB, RS, n_obs, 3, 8, 8),
        'goal': torch.randn(RB, RS, n_goal, 3, 8, 8),
        'proprio': torch.randn(RB, RS, n_obs, RPR),
        'goal_proprio': torch.randn(RB, RS, n_goal, RPR),
        'action': torch.randn(RB, RS, n_obs, RA),
        'action_history': torch.randn(RB, RS, n_obs - 1, RA),
        'id': torch.full((RB, RS, 1), 7, dtype=torch.int64),
        'step_idx': torch.zeros((RB, RS, 1), dtype=torch.int64),
    }


def _clone(info):
    return {
        k: (v.clone() if torch.is_tensor(v) else v) for k, v in info.items()
    }


def _prejepa_objective():
    """The composition that reproduces the deleted PreJEPA.criterion: one
    GoalMSE per non-action source, mean-reduced, extras before pixels."""
    return WeightedSum(
        [
            (
                1.0,
                GoalMSE(
                    pred_key='predicted_proprio_emb',
                    goal_key='proprio_goal_emb',
                    reduction='mean',
                ),
            ),
            (
                1.0,
                GoalMSE(
                    pred_key='predicted_pixels_emb',
                    goal_key='pixels_goal_emb',
                    reduction='mean',
                ),
            ),
        ]
    )


def _legacy_get_cost(model, info_dict, action_candidates):
    """Verbatim copy of the goal-encoding branch + criterion deleted from
    ``PreJEPA.get_cost`` — the parity target for the evaluator seam."""
    import torch.nn.functional as F

    emb_keys = [k for k in model.extra_encoders.keys() if k != 'action']

    goal_info_dict = {
        k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)
    }
    goal_info_dict = model.encode(
        goal_info_dict,
        target='goal_emb',
        pixels_key='goal',
        prefix='goal_',
        emb_keys=emb_keys,
    )
    for key in (
        'goal_emb',
        'pixels_goal_emb',
        *(f'{k}_goal_emb' for k in emb_keys),
    ):
        emb = goal_info_dict[key]
        info_dict[key] = emb.unsqueeze(1).expand(
            -1, action_candidates.shape[1], *([-1] * (emb.ndim - 1))
        )

    info_dict = model.rollout(info_dict, action_candidates)

    cost = 0.0
    for key in emb_keys + ['pixels']:
        preds = info_dict[f'predicted_{key}_emb']
        goal = info_dict[f'{key}_goal_emb']
        cost = cost + F.mse_loss(
            preds[:, :, -1:], goal, reduction='none'
        ).mean(dim=tuple(range(2, preds.ndim)))
    return cost


def test_prejepa_satisfies_dynamics_protocol():
    """PreJEPA exposes encode/rollout, so a ShootingCostEvaluator can wrap it."""
    assert isinstance(object.__new__(PreJEPA), Dynamics)


def test_prejepa_no_longer_exposes_get_cost():
    """Cost now lives in the evaluator seam, not on the model."""
    assert not hasattr(object.__new__(PreJEPA), 'get_cost')
    assert not hasattr(object.__new__(PreJEPA), 'criterion')


def test_composition_matches_legacy_get_cost_bitwise():
    """WeightedSum of per-source GoalMSE == the deleted PreJEPA.get_cost,
    bit-for-bit. This is the guarantee that migrating the dinowm checkpoints
    to the seam does not move any number."""
    torch.manual_seed(0)
    candidates = torch.randn(RB, RS, 4, RA)
    info = _split_info()

    legacy = _legacy_get_cost(_split_model(), _clone(info), candidates.clone())
    seam = ShootingCostEvaluator(
        _split_model(), _prejepa_objective()
    ).get_cost(_clone(info), candidates.clone())

    # guard against a vacuous comparison: the cost must actually discriminate
    assert len({round(v, 3) for v in legacy[0].tolist()}) > 1
    torch.testing.assert_close(legacy, seam, rtol=0, atol=0)


def test_shipped_objective_configs_match_legacy_get_cost():
    """The yaml the plan scripts select must reproduce the legacy cost too —
    the composition is only correct if its term order and reduction are."""
    import hydra
    from omegaconf import OmegaConf

    torch.manual_seed(0)
    candidates = torch.randn(RB, RS, 4, RA)
    info = _split_info()

    cfg = OmegaConf.load(
        REPO_ROOT
        / 'scripts/plan/config/objective/goal_mse_pixels_proprio.yaml'
    )
    objective = hydra.utils.instantiate(cfg)

    legacy = _legacy_get_cost(_split_model(), _clone(info), candidates.clone())
    seam = ShootingCostEvaluator(_split_model(), objective).get_cost(
        _clone(info), candidates.clone()
    )
    torch.testing.assert_close(legacy, seam, rtol=0, atol=0)


def test_default_goal_encode_dispatches_on_split_latent():
    """A split-latent model needs no explicit encode_goal: the default
    dispatches to split_goal_encode on the presence of extra_encoders."""
    torch.manual_seed(0)
    candidates = torch.randn(RB, RS, 4, RA)
    info = _split_info()

    explicit = ShootingCostEvaluator(
        _split_model(), _prejepa_objective(), encode_goal=split_goal_encode
    ).get_cost(_clone(info), candidates.clone())
    dispatched = ShootingCostEvaluator(
        _split_model(), _prejepa_objective()
    ).get_cost(_clone(info), candidates.clone())

    torch.testing.assert_close(explicit, dispatched, rtol=0, atol=0)


def test_weights_trade_the_sources_off():
    """WeightedSum's coefficients now provide the per-source weighting."""
    torch.manual_seed(0)
    candidates = torch.randn(RB, RS, 4, RA)
    info = _split_info()

    def cost_of(objective):
        return ShootingCostEvaluator(_split_model(), objective).get_cost(
            _clone(info), candidates.clone()
        )

    proprio = GoalMSE(
        pred_key='predicted_proprio_emb',
        goal_key='proprio_goal_emb',
        reduction='mean',
    )
    pixels = GoalMSE(
        pred_key='predicted_pixels_emb',
        goal_key='pixels_goal_emb',
        reduction='mean',
    )

    unit = cost_of(WeightedSum([(1.0, proprio), (1.0, pixels)]))
    weighted = cost_of(WeightedSum([(0.25, proprio), (2.0, pixels)]))
    only_pixels = cost_of(WeightedSum([(1.0, pixels)]))
    only_proprio = cost_of(WeightedSum([(1.0, proprio)]))

    torch.testing.assert_close(unit, only_proprio + only_pixels)
    torch.testing.assert_close(
        weighted, 0.25 * only_proprio + 2.0 * only_pixels
    )
    assert not torch.allclose(weighted, unit)


def test_goal_mse_reduction_is_validated():
    with pytest.raises(ValueError, match="'sum' or 'mean'"):
        GoalMSE(reduction='average')


def test_multi_frame_goal_is_the_one_documented_divergence():
    """The legacy criterion averaged over *every* goal frame; GoalMSE compares
    against the last one. They coincide for the single-frame goals the eval
    pipeline produces, and only then — pinned here so a multi-frame goal
    cannot change costs unnoticed."""
    torch.manual_seed(0)
    candidates = torch.randn(RB, RS, 4, RA)

    single = _split_info(n_goal=1)
    torch.testing.assert_close(
        _legacy_get_cost(_split_model(), _clone(single), candidates.clone()),
        ShootingCostEvaluator(_split_model(), _prejepa_objective()).get_cost(
            _clone(single), candidates.clone()
        ),
        rtol=0,
        atol=0,
    )

    multi = _split_info(n_goal=3)
    legacy = _legacy_get_cost(
        _split_model(), _clone(multi), candidates.clone()
    )
    seam = ShootingCostEvaluator(
        _split_model(), _prejepa_objective()
    ).get_cost(_clone(multi), candidates.clone())
    assert not torch.allclose(legacy, seam)


def test_extra_encoder_inputs_must_match_the_pixel_time_dim():
    """Regression: proprio has to be stacked over the same context frames as
    pixels (WorldModelPolicy.history_keys must cover every extra encoder),
    otherwise the fused concat is malformed."""
    model = _split_model()
    info = {
        'pixels': torch.randn(RB, 3, 3, 8, 8),
        'proprio': torch.randn(RB, 1, RPR),  # 1 frame vs 3 pixel frames
        'action': torch.randn(RB, 3, RA),
    }
    with pytest.raises(RuntimeError, match='Sizes of tensors must match'):
        model.encode(info)
