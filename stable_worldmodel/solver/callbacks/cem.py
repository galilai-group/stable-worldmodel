"""Callbacks for CEM and iCEM solvers."""

from typing import Any

import torch

from .common import Callback


class EliteCostRecorder(Callback):
    """Per-step elite cost stats (mean, min, max), per env."""

    def compute(self, **state: Any) -> dict[str, float | list[float]]:
        v: torch.Tensor = state['topk_vals'].detach()
        return {
            'mean': self._reduce(v.mean(dim=1)),
            'min': self._reduce(v.min(dim=1).values),
            'max': self._reduce(v.max(dim=1).values),
        }


class VarNormRecorder(Callback):
    """Per-step mean variance of the action distribution (per env)."""

    def compute(self, **state: Any) -> float | list[float]:
        var: torch.Tensor = state['var']
        per_env = var.detach().flatten(1).mean(dim=-1)
        return self._reduce(per_env)


class MeanShiftRecorder(Callback):
    """Per-step L2 distance between consecutive distribution means (per env)."""

    def compute(self, **state: Any) -> float | list[float] | None:
        prev_mean: torch.Tensor | None = state.get('prev_mean')
        if prev_mean is None:
            return None
        mean: torch.Tensor = state['mean']
        per_env = (mean - prev_mean).detach().flatten(1).norm(dim=-1)
        return self._reduce(per_env)


class EliteSpreadRecorder(Callback):
    """Per-step within-elite std (diversity of the top-k elites, per env)."""

    def compute(self, **state: Any) -> float | list[float]:
        topk_candidates: torch.Tensor = state['topk_candidates']
        per_env = topk_candidates.detach().std(dim=1).flatten(1).mean(dim=-1)
        return self._reduce(per_env)
