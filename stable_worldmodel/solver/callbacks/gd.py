"""Callbacks specific to the gradient descent solver."""

from typing import Any

import torch

from .common import Callback, Reduction


class GradNormRecorder(Callback):
    """Per-step L2 norm of the action gradient (per env, mean over samples).

    With ``per_step=True``, returns a list of length H with one grad
    norm per horizon step instead of reducing over the horizon dim.
    """

    def __init__(
        self, reduction: Reduction = 'mean', per_step: bool = False
    ) -> None:
        super().__init__(reduction=reduction)
        self.per_step = per_step

    def compute(self, **state: Any) -> Any:
        params: torch.Tensor = state['params']
        g = params.grad
        if self.per_step:
            H = params.shape[2]
            if g is None:
                return [0.0] * H
            # (B, N, H, D) -> norm over D -> (B, N, H) -> mean over N -> (B, H)
            per_env_per_step = g.detach().norm(dim=-1).mean(dim=1)
            return [self._reduce(per_env_per_step[..., h]) for h in range(H)]
        if g is None:
            return 0.0
        per_env = g.detach().flatten(2).norm(dim=-1).mean(dim=-1)
        return self._reduce(per_env)


class ActionNormRecorder(Callback):
    """Per-step L2 norm of the action tensor (per env, mean over samples)."""

    def compute(self, **state: Any) -> float | list[float]:
        params: torch.Tensor = state['params']
        per_env = params.detach().flatten(2).norm(dim=-1).mean(dim=-1)
        return self._reduce(per_env)
