"""Model Predictive Path Integral solver for model-based planning."""

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging

from .base import BaseSolver
from .solver import Costable


class MPPISolver(BaseSolver):
    """Model Predictive Path Integral solver for action optimization.

    Args:
        model: World model implementing the Costable protocol.
        batch_size: Number of environments to process in parallel.
        num_samples: Number of action candidates to sample per iteration.
        var_scale: Initial variance scale for action noise.
        n_steps: Number of MPPI iterations.
        topk: Number of elite samples for weighted averaging (None = use all).
        temperature: Temperature parameter for softmax weighting.
        device: Device for tensor computations.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        model: Costable,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1.0,
        n_steps: int = 30,
        topk: int | None = 30,
        temperature: float = 0.5,
        device: str | torch.device = 'cpu',
        seed: int = 1234,
    ) -> None:
        super().__init__(
            model=model,
            n_steps=n_steps,
            batch_size=batch_size,
            num_samples=num_samples,
            device=device,
            callbacks=None,
        )
        self.topk = topk
        self.var_scale = var_scale
        self.temperature = temperature
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)

    def configure(
        self, *, action_space: gym.Space, n_envs: int, config: Any
    ) -> None:
        super().configure(
            action_space=action_space, n_envs=n_envs, config=config
        )
        shape = action_space.shape
        self._action_dim = (
            int(np.prod(shape[1:])) if len(shape) > 1 else int(np.prod(shape))
        )
        if not isinstance(action_space, Box):
            logging.warning(
                f'Action space is discrete, got {type(action_space)}. '
                'MPPISolver may not work as expected.'
            )

    @property
    def action_dim(self) -> int:
        return self._action_dim * self.action_block

    def init_action_distrib(
        self, n_envs: int, actions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize the action distribution parameters (mean and variance)."""
        var = self.var_scale * torch.ones(
            [n_envs, self.horizon, self.action_dim], dtype=self.dtype
        )
        mean = (
            torch.zeros([n_envs, 0, self.action_dim], dtype=self.dtype)
            if actions is None
            else actions
        )
        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            device = mean.device
            new_mean = torch.zeros(
                [n_envs, remaining, self.action_dim], dtype=self.dtype
            )
            mean = torch.cat([mean, new_mean], dim=1).to(device)
        return mean.to(self.device), var.to(self.device)

    # === BaseSolver hooks ===

    def init_state(
        self, n_envs: int, init: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        mean, var = self.init_action_distrib(n_envs, init)
        return {'mean': mean, 'var': var}

    def propose(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        bs = state['mean'].shape[0]
        noise = torch.randn(
            bs,
            self.num_samples,
            self.horizon,
            self.action_dim,
            generator=self.torch_gen,
            device=self.device,
            dtype=self.dtype,
        )
        cands = state['mean'].unsqueeze(1) + noise * state['var'].unsqueeze(1)
        cands[:, 0] = state['mean']
        return cands

    def update(
        self,
        state: dict[str, torch.Tensor],
        candidates: torch.Tensor,
        costs: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if self.topk is not None and self.topk < self.num_samples:
            topk_vals, topk_inds = torch.topk(
                costs, k=self.topk, dim=1, largest=False
            )
            bs = candidates.shape[0]
            batch_indices = (
                torch.arange(bs, device=self.device)
                .unsqueeze(1)
                .expand(-1, self.topk)
            )
            relevant_cands = candidates[batch_indices, topk_inds]
            relevant_costs = topk_vals
        else:
            relevant_cands = candidates
            relevant_costs = costs

        # Stabilized softmax weighting
        min_cost = relevant_costs.min(dim=1, keepdim=True)[0]
        weights = torch.softmax(
            -(relevant_costs - min_cost) / self.temperature, dim=1
        )
        weighted = (weights.unsqueeze(-1).unsqueeze(-1) * relevant_cands).sum(
            dim=1
        )

        # MPPI updates the mean only; var is held fixed (matches original).
        return {'mean': weighted, 'var': state['var']}, {}

    def finalize(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return state['mean'].detach().cpu()

    def extra_outputs(self, state: dict[str, torch.Tensor]) -> dict[str, Any]:
        return {
            'mean': [state['mean'].detach().cpu()],
            'var': [state['var'].detach().cpu()],
        }

    def _cost_summary(self, payload: dict, costs: torch.Tensor) -> list[float]:
        # MPPI's "final cost" = mean over the elite (or full) sample set.
        if self.topk is not None and self.topk < self.num_samples:
            topk_vals = costs.topk(k=self.topk, dim=1, largest=False).values
            return topk_vals.mean(dim=1).cpu().tolist()
        return costs.mean(dim=1).cpu().tolist()
