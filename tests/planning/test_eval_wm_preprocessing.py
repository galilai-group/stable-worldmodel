"""Regression tests for world-model evaluation preprocessing."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.plan.eval_wm import get_policy_preprocessing
from stable_worldmodel.data import ZScoreScaler


def _cfg():
    return SimpleNamespace(
        dataset=SimpleNamespace(keys_to_cache=['action', 'proprio', 'state'])
    )


class _NoStatsDataset:
    def get_col_data(self, _column):
        raise AssertionError(
            'evaluation statistics must not be read for modern checkpoints'
        )


def test_saved_preprocessing_does_not_read_evaluation_statistics(monkeypatch):
    saved = {
        'proprio': ZScoreScaler(mean=[100.0], std=[10.0]),
        'action': ZScoreScaler(mean=[20.0], std=[5.0]),
    }
    monkeypatch.setattr(
        'stable_worldmodel.wm.utils.load_preprocessing',
        lambda _name: saved,
    )

    process = get_policy_preprocessing(
        _cfg(), 'run/weights.pt', _NoStatsDataset()
    )

    np.testing.assert_allclose(
        process['proprio'].transform(np.array([[110.0]])),
        [[1.0]],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        process['action'].inverse_transform(np.array([[0.0]])),
        [[20.0]],
        rtol=0,
        atol=0,
    )
    assert process['goal_proprio'] is process['proprio']
    assert 'state' not in process
    assert 'goal_action' not in process


def test_legacy_checkpoint_keeps_explicit_eval_fit_fallback(monkeypatch):
    monkeypatch.setattr(
        'stable_worldmodel.wm.utils.load_preprocessing',
        lambda _name: None,
    )

    values = {
        'state': np.array([[0.0], [2.0]], dtype=np.float64),
        'proprio': np.array([[4.0], [8.0]], dtype=np.float64),
        'action': np.array([[10.0], [20.0]], dtype=np.float64),
    }
    dataset = SimpleNamespace(get_col_data=lambda column: values[column])

    with pytest.warns(RuntimeWarning, match='no saved training preprocessing'):
        process = get_policy_preprocessing(
            _cfg(), 'legacy/weights.pt', dataset
        )

    np.testing.assert_allclose(
        process['state'].transform(np.array([[1.0]])),
        [[0.0]],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        process['action'].inverse_transform(np.array([[0.0]])),
        [[15.0]],
        rtol=0,
        atol=0,
    )
    assert process['goal_state'] is process['state']
    assert process['goal_proprio'] is process['proprio']


def test_checkpoint_artifact_roundtrip_drives_eval(tmp_path, monkeypatch):
    import torch

    from stable_worldmodel.wm.utils import save_pretrained

    monkeypatch.setenv('STABLEWM_HOME', str(tmp_path))

    state = ZScoreScaler(eps=1e-6).fit(
        np.array([[0.0], [2.0], [4.0]], dtype=np.float64)
    )
    action = ZScoreScaler(eps=1e-6).fit(
        np.array([[10.0], [20.0], [30.0]], dtype=np.float64)
    )

    save_pretrained(
        torch.nn.Identity(),
        run_name='modern',
        config={'_target_': 'torch.nn.Identity'},
        cache_dir=tmp_path,
        preprocessing={'state': state, 'action': action},
    )

    cfg = SimpleNamespace(
        dataset=SimpleNamespace(keys_to_cache=['state', 'action'])
    )
    process = get_policy_preprocessing(cfg, 'modern', _NoStatsDataset())

    np.testing.assert_allclose(
        process['state'].transform(np.array([[2.0]])),
        [[0.0]],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        process['action'].inverse_transform(np.array([[0.0]])),
        [[20.0]],
        rtol=0,
        atol=0,
    )
    assert process['goal_state'] is process['state']
