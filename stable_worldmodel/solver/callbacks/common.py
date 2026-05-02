"""Base callback class and solver-agnostic recorders."""

from typing import Any, Literal

import torch


Reduction = Literal['mean', 'sum', 'none']


class Callback:
    """Base class for solver iteration callbacks.

    Subclasses compute a per-env metric and call ``self._reduce(...)`` to
    apply the configured reduction across envs. ``history`` is
    ``list[list[Any]]`` (batches x steps), matching the shape of
    ``outputs['cost']`` in the gradient solver.
    """

    name: str | None = None

    def __init__(self, reduction: Reduction = 'mean') -> None:
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}"
            )
        self.reduction = reduction
        self.history: list[list[Any]] = []
        self._current: list[Any] = []

    def _reduce(self, x: torch.Tensor) -> float | list[float]:
        x = x.detach()
        if self.reduction == 'mean':
            return x.mean().item()
        if self.reduction == 'sum':
            return x.sum().item()
        return x.cpu().tolist()

    def reset(self) -> None:
        self.history = []
        self._current = []

    def start_batch(self) -> None:
        if self._current:
            self.history.append(self._current)
        self._current = []

    def end_solve(self) -> None:
        if self._current:
            self.history.append(self._current)
            self._current = []

    def __call__(self, **state: Any) -> None:
        value = self.compute(**state)
        if value is not None:
            self._current.append(value)

    def compute(self, **state: Any) -> Any:
        raise NotImplementedError

    @property
    def output_key(self) -> str:
        return self.name or self.__class__.__name__


class BestCostRecorder(Callback):
    """Per-step minimum cost over the sample population (per env)."""

    def compute(self, **state: Any) -> float | list[float]:
        costs: torch.Tensor = state['costs']
        return self._reduce(costs.min(dim=1).values)


class MeanCostRecorder(Callback):
    """Per-step mean cost over the sample population (per env)."""

    def compute(self, **state: Any) -> float | list[float]:
        costs: torch.Tensor = state['costs']
        return self._reduce(costs.mean(dim=1))
