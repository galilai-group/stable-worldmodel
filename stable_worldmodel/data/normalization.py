"""Picklable, sklearn-style scalers for per-column normalization.

Each scaler implements the :class:`~stable_worldmodel.policy.Transformable`
protocol (``transform`` / ``inverse_transform``) plus the sklearn pattern
(``fit`` / ``fit_transform``), and additionally exposes ``__call__`` so the
fitted scaler drops directly into
:class:`stable_pretraining.data.transforms.WrapTorchTransform`. Inputs may
be NumPy arrays or Torch tensors — the output type matches the input.

All scalers are picklable end-to-end (no closures, no lambdas), which is
required when the dataset is consumed by a DataLoader using the ``spawn``
start method.
"""

import numpy as np
import torch


__all__ = [
    'IdentityScaler',
    'PercentileScaler',
    'ZScoreScaler',
    'get_scaler',
]


def _as_like(a, b, x):
    """Cast bounds ``a``/``b`` to the dtype/device of ``x``."""
    if isinstance(x, torch.Tensor):
        a = torch.as_tensor(a, dtype=x.dtype, device=x.device)
        b = torch.as_tensor(b, dtype=x.dtype, device=x.device)
    else:
        a = np.asarray(a)
        b = np.asarray(b)
    return a, b


def _to_numpy_2d(X):
    """Detach, NaN-filter (per-row), and reshape to ``(N, D)``."""
    arr = (
        X.detach().cpu().numpy()
        if isinstance(X, torch.Tensor)
        else np.asarray(X)
    )
    arr = arr.reshape(-1, arr.shape[-1])
    return arr[~np.isnan(arr).any(axis=1)]


class IdentityScaler:
    """No-op scaler. Use for columns that should pass through unchanged."""

    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X

    def fit_transform(self, X):
        return X

    def __call__(self, X):
        return X


class ZScoreScaler:
    """Per-dim z-score scaler: ``(x - mean) / std``.

    Stats are stored as numpy arrays so the scaler pickles without dragging
    torch tensors across process boundaries.
    """

    def __init__(self, mean=None, std=None, eps: float = 1e-8):
        self.mean = np.asarray(mean) if mean is not None else None
        self.std = np.asarray(std) if std is not None else None
        self.eps = eps

    def fit(self, X):
        data = _to_numpy_2d(X)
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True)
        return self

    def transform(self, X):
        mean, std = _as_like(self.mean, self.std, X)
        if isinstance(X, torch.Tensor):
            return (X - mean) / std.clamp(min=self.eps)
        return (X - mean) / np.maximum(std, self.eps)

    def inverse_transform(self, X):
        mean, std = _as_like(self.mean, self.std, X)
        return X * std + mean

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def __call__(self, X):
        out = self.transform(X)
        return out.float() if isinstance(out, torch.Tensor) else out


class PercentileScaler:
    """Per-dim percentile scaler: maps to ``[-1, 1]`` using ``q_low``/``q_high``
    and clips. Robust to outliers compared to z-score."""

    def __init__(
        self,
        low: float = 1.0,
        high: float = 99.0,
        q_low=None,
        q_high=None,
        eps: float = 1e-8,
    ):
        self.low = low
        self.high = high
        self.q_low = np.asarray(q_low) if q_low is not None else None
        self.q_high = np.asarray(q_high) if q_high is not None else None
        self.eps = eps

    def fit(self, X):
        data = _to_numpy_2d(X)
        self.q_low = np.percentile(data, self.low, axis=0)
        self.q_high = np.percentile(data, self.high, axis=0)
        return self

    def transform(self, X):
        q_low, q_high = _as_like(self.q_low, self.q_high, X)
        if isinstance(X, torch.Tensor):
            scale = (q_high - q_low).clamp(min=self.eps)
            return (2 * (X - q_low) / scale - 1).clamp(-1, 1)
        scale = np.maximum(q_high - q_low, self.eps)
        return np.clip(2 * (X - q_low) / scale - 1, -1, 1)

    def inverse_transform(self, X):
        q_low, q_high = _as_like(self.q_low, self.q_high, X)
        return (X + 1) * (q_high - q_low) / 2 + q_low

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def __call__(self, X):
        out = self.transform(X)
        return out.float() if isinstance(out, torch.Tensor) else out


_SCALERS = {
    'zscore': ZScoreScaler,
    'percentile': PercentileScaler,
    'none': IdentityScaler,
}


def get_scaler(method: str = 'zscore', **kwargs):
    """Return an unfitted scaler by method name.

    Args:
        method: One of ``'zscore'``, ``'percentile'``, ``'none'``.
        **kwargs: Forwarded to the scaler constructor.

    Raises:
        ValueError: If ``method`` is not registered.
    """
    if method not in _SCALERS:
        raise ValueError(
            f'Unknown normalizer method: {method!r}. '
            f'Expected one of {list(_SCALERS)}.'
        )
    return _SCALERS[method](**kwargs)
