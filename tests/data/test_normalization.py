"""Tests for stable_worldmodel.data.normalization."""

import json
import pickle

import numpy as np
import pytest
import torch

from stable_worldmodel.data.normalization import (
    IdentityScaler,
    PercentileScaler,
    ZScoreScaler,
    deserialize_scaler,
    get_scaler,
    serialize_scaler,
)

# ─── IdentityScaler ───────────────────────────────────────────────────────────


def test_identity_scaler_passthrough_numpy():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    s = IdentityScaler().fit(x)
    np.testing.assert_array_equal(s.transform(x), x)
    np.testing.assert_array_equal(s.inverse_transform(x), x)
    np.testing.assert_array_equal(s(x), x)


def test_identity_scaler_passthrough_torch():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    s = IdentityScaler()
    assert torch.equal(s(x), x)


# ─── ZScoreScaler ─────────────────────────────────────────────────────────────


def test_zscore_fit_stats():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=[1.0, -2.0], scale=[3.0, 0.5], size=(2000, 2))
    s = ZScoreScaler().fit(x)
    np.testing.assert_allclose(s.mean.squeeze(), [1.0, -2.0], atol=0.2)
    np.testing.assert_allclose(s.std.squeeze(), [3.0, 0.5], atol=0.1)


def test_zscore_transform_inverse_roundtrip_numpy():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 4))
    s = ZScoreScaler().fit(x)
    y = s.transform(x)
    np.testing.assert_allclose(s.inverse_transform(y), x, atol=1e-6)


def test_zscore_transform_inverse_roundtrip_torch():
    rng = np.random.default_rng(2)
    x_np = rng.normal(size=(100, 4)).astype(np.float32)
    s = ZScoreScaler().fit(x_np)

    x = torch.from_numpy(x_np)
    y = s.transform(x)
    assert isinstance(y, torch.Tensor)
    torch.testing.assert_close(s.inverse_transform(y), x, atol=1e-5, rtol=1e-5)


def test_zscore_call_returns_float_tensor():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    s = ZScoreScaler().fit(x)
    out = s(x)
    assert out.dtype == torch.float32


def test_zscore_ignores_nan_rows_in_fit():
    x = np.array([[1.0, 2.0], [np.nan, 3.0], [3.0, 4.0]])
    s = ZScoreScaler().fit(x)
    # Only the clean rows [1,2] and [3,4] are used.
    np.testing.assert_allclose(s.mean.squeeze(), [2.0, 3.0])


def test_zscore_eps_avoids_zero_std():
    x = np.ones((10, 3))  # zero std on every dim
    s = ZScoreScaler().fit(x)
    # Should not raise / produce NaN/inf.
    out = s.transform(x)
    assert np.all(np.isfinite(out))


# ─── PercentileScaler ─────────────────────────────────────────────────────────


def test_percentile_fit_bounds():
    x = np.linspace(0, 100, 1001).reshape(-1, 1)
    s = PercentileScaler(low=1.0, high=99.0).fit(x)
    np.testing.assert_allclose(s.q_low, [1.0], atol=0.5)
    np.testing.assert_allclose(s.q_high, [99.0], atol=0.5)


def test_percentile_transform_in_range():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(500, 3))
    s = PercentileScaler().fit(x)
    y = s.transform(x)
    assert y.min() >= -1.0 - 1e-9
    assert y.max() <= 1.0 + 1e-9


def test_percentile_clips_outliers():
    x = np.concatenate([np.linspace(0, 1, 100), [1e6]]).reshape(-1, 1)
    s = PercentileScaler(low=1.0, high=99.0).fit(x)
    y = s.transform(x)
    assert y.max() == pytest.approx(1.0)


def test_percentile_roundtrip_within_bounds():
    rng = np.random.default_rng(4)
    x = rng.uniform(-2, 2, size=(200, 2))
    s = PercentileScaler(low=0.0, high=100.0).fit(x)
    y = s.transform(x)
    np.testing.assert_allclose(s.inverse_transform(y), x, atol=1e-6)


def test_percentile_torch_tensor():
    rng = np.random.default_rng(5)
    x_np = rng.normal(size=(200, 2)).astype(np.float32)
    s = PercentileScaler().fit(x_np)
    x = torch.from_numpy(x_np)
    y = s.transform(x)
    assert isinstance(y, torch.Tensor)
    assert y.min() >= -1.0 - 1e-6
    assert y.max() <= 1.0 + 1e-6


# ─── get_scaler ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    'method, cls',
    [
        ('zscore', ZScoreScaler),
        ('percentile', PercentileScaler),
        ('none', IdentityScaler),
    ],
)
def test_get_scaler_dispatches(method, cls):
    assert isinstance(get_scaler(method), cls)


def test_get_scaler_unknown_method_raises():
    with pytest.raises(ValueError, match='Unknown normalizer method'):
        get_scaler('not-a-real-method')


# ─── Picklability (DataLoader spawn-safe) ─────────────────────────────────────


@pytest.mark.parametrize(
    'scaler',
    [
        IdentityScaler(),
        ZScoreScaler().fit(np.random.default_rng(6).normal(size=(50, 3))),
        PercentileScaler().fit(np.random.default_rng(7).normal(size=(50, 3))),
    ],
)
def test_scaler_round_trips_through_pickle(scaler):
    blob = pickle.dumps(scaler)
    restored = pickle.loads(blob)
    x = np.random.default_rng(8).normal(size=(10, 3))
    np.testing.assert_allclose(restored.transform(x), scaler.transform(x))


# ─── column_normalizer ────────────────────────────────────────────────────────


class _StubDataset:
    """Minimal stand-in exposing get_col_data(col)."""

    def __init__(self, columns):
        self._columns = columns

    def get_col_data(self, col):
        return self._columns[col]


@pytest.fixture
def stub_dataset():
    rng = np.random.default_rng(9)
    return _StubDataset(
        {'action': rng.normal(size=(200, 4)).astype(np.float32)}
    )


def test_column_normalizer_default_is_zscore(stub_dataset):
    pytest.importorskip('stable_pretraining')
    from stable_worldmodel.data import column_normalizer

    t = column_normalizer(stub_dataset, 'action', 'action')
    assert isinstance(t.lambd, ZScoreScaler)


def test_column_normalizer_percentile(stub_dataset):
    pytest.importorskip('stable_pretraining')
    from stable_worldmodel.data import column_normalizer

    t = column_normalizer(
        stub_dataset, 'action', 'action', method='percentile'
    )
    assert isinstance(t.lambd, PercentileScaler)
    assert t.lambd.q_low is not None and t.lambd.q_high is not None


def test_column_normalizer_none_returns_identity(stub_dataset):
    pytest.importorskip('stable_pretraining')
    from stable_worldmodel.data import column_normalizer

    t = column_normalizer(stub_dataset, 'action', 'action', method='none')
    assert isinstance(t.lambd, IdentityScaler)


def test_column_normalizer_applies_to_sample(stub_dataset):
    pytest.importorskip('stable_pretraining')
    from stable_worldmodel.data import column_normalizer

    t = column_normalizer(stub_dataset, 'action', 'action_norm')
    sample = {
        'action': torch.from_numpy(stub_dataset.get_col_data('action')[:5])
    }
    out = t(sample)
    assert 'action_norm' in out
    # mean should be ~0 after z-score on a sample drawn from the fitted dist.
    assert abs(float(out['action_norm'].mean())) < 1.0


#  JSON-safe serialization


def test_identity_scaler_serialization_roundtrip():
    scaler = IdentityScaler()
    spec = serialize_scaler(scaler)

    assert spec == {'method': 'none'}

    restored = deserialize_scaler(spec)
    assert isinstance(restored, IdentityScaler)

    x = np.random.default_rng(10).normal(size=(8, 3))
    np.testing.assert_array_equal(restored.transform(x), x)


def test_zscore_scaler_serialization_roundtrip():
    rng = np.random.default_rng(11)
    x = rng.normal(loc=[3.0, -2.0], scale=[2.0, 4.0], size=(100, 2)).astype(
        np.float32
    )

    scaler = ZScoreScaler(eps=1e-6).fit(x)
    spec = serialize_scaler(scaler)

    assert spec['method'] == 'zscore'
    assert spec['eps'] == 1e-6
    assert spec['mean_dtype'] == 'float32'
    assert spec['std_dtype'] == 'float32'
    assert isinstance(spec['mean'], list)
    assert isinstance(spec['std'], list)

    restored = deserialize_scaler(spec)

    np.testing.assert_array_equal(restored.mean, scaler.mean)
    np.testing.assert_array_equal(restored.std, scaler.std)
    assert restored.mean.dtype == scaler.mean.dtype
    assert restored.std.dtype == scaler.std.dtype
    assert restored.eps == scaler.eps

    probe = rng.normal(size=(10, 2)).astype(np.float32)
    assert restored.transform(probe).dtype == scaler.transform(probe).dtype
    np.testing.assert_allclose(
        restored.transform(probe),
        scaler.transform(probe),
        rtol=0,
        atol=0,
    )


def test_percentile_scaler_serialization_roundtrip():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(200, 3)).astype(np.float32)

    scaler = PercentileScaler(
        low=5.0,
        high=95.0,
        eps=1e-7,
    ).fit(x)

    spec = serialize_scaler(scaler)

    assert spec['method'] == 'percentile'
    assert spec['low'] == 5.0
    assert spec['high'] == 95.0
    assert spec['eps'] == 1e-7
    assert spec['q_low_dtype'] == str(scaler.q_low.dtype)
    assert spec['q_high_dtype'] == str(scaler.q_high.dtype)
    assert isinstance(spec['q_low'], list)
    assert isinstance(spec['q_high'], list)

    restored = deserialize_scaler(spec)

    np.testing.assert_array_equal(restored.q_low, scaler.q_low)
    np.testing.assert_array_equal(restored.q_high, scaler.q_high)
    assert restored.q_low.dtype == scaler.q_low.dtype
    assert restored.q_high.dtype == scaler.q_high.dtype
    assert restored.low == scaler.low
    assert restored.high == scaler.high
    assert restored.eps == scaler.eps

    probe = rng.normal(size=(10, 3)).astype(np.float32)
    assert restored.transform(probe).dtype == scaler.transform(probe).dtype
    np.testing.assert_allclose(
        restored.transform(probe),
        scaler.transform(probe),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    'scaler',
    [
        IdentityScaler(),
        ZScoreScaler().fit(np.random.default_rng(13).normal(size=(20, 2))),
        PercentileScaler().fit(np.random.default_rng(14).normal(size=(20, 2))),
    ],
)
def test_serialized_scaler_is_json_safe(scaler):
    spec = serialize_scaler(scaler)

    # Must not rely on pickle, NumPy-specific JSON encoders, or OmegaConf.
    encoded = json.dumps(spec)
    decoded = json.loads(encoded)

    assert decoded == spec


def test_deserialize_scaler_rejects_unknown_method():
    with pytest.raises(ValueError, match='Unknown serialized scaler method'):
        deserialize_scaler({'method': 'definitely-not-a-scaler'})


def test_serialize_scaler_rejects_unknown_type():
    with pytest.raises(TypeError, match='Unsupported scaler type'):
        serialize_scaler(object())
