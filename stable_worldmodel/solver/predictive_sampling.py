"""Predictive Sampling solver for model-based planning.

Reference: Howell et al., "Predictive Sampling: Real-time Behaviour Synthesis
with MuJoCo", 2022.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging

from .base import BaseSolver
from .solver import Costable


class PredictiveSamplingSolver(BaseSolver):
    """Predictive Sampling solver for action optimization.

    At each step, perturb the nominal action sequence with Gaussian noise,
    evaluate ``num_samples`` candidates, and replace the nominal with the
    argmin. The first candidate is always the unperturbed nominal so the
    nominal never gets worse.

    Args:
        model: World model implementing the Costable protocol.
        batch_size: Number of environments to process in parallel.
        num_samples: Number of action candidates to sample.
        n_steps: Number of refinement iterations (default 1, matching the
            original single-shot algorithm).
        noise_scale: Standard deviation of additive Gaussian noise.
        device: Device for tensor computations.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        model: Costable,
        batch_size: int = 1,
        num_samples: int = 300,
        n_steps: int = 1,
        noise_scale: float = 1.0,
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
        self.noise_scale = noise_scale
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
                'PredictiveSamplingSolver may not work as expected.'
            )

    @property
    def action_dim(self) -> int:
        return self._action_dim * self.action_block

    def init_nominal(
        self, n_envs: int, actions: torch.Tensor | None = None
    ) -> torch.Tensor:
        nominal = (
            torch.zeros([n_envs, 0, self.action_dim], dtype=self.dtype)
            if actions is None
            else actions
        )
        remaining = self.horizon - nominal.shape[1]
        if remaining > 0:
            device = nominal.device
            pad = torch.zeros(
                [n_envs, remaining, self.action_dim], dtype=self.dtype
            )
            nominal = torch.cat([nominal, pad], dim=1).to(device)
        return nominal.to(self.device)

    # === BaseSolver hooks ===

    def init_state(
        self, n_envs: int, init: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        return {'nominal': self.init_nominal(n_envs, init)}

    def propose(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        bs = state['nominal'].shape[0]
        noise = torch.randn(
            bs,
            self.num_samples,
            self.horizon,
            self.action_dim,
            generator=self.torch_gen,
            device=self.device,
            dtype=self.dtype,
        )
        cands = state['nominal'].unsqueeze(1) + noise * self.noise_scale
        cands[:, 0] = state['nominal']
        return cands

    def update(
        self,
        state: dict[str, torch.Tensor],
        candidates: torch.Tensor,
        costs: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        bs = candidates.shape[0]
        best_idx = costs.argmin(dim=1)
        batch_indices = torch.arange(bs, device=self.device)
        best_cands = candidates[batch_indices, best_idx]
        best_costs = costs[batch_indices, best_idx]
        return {'nominal': best_cands}, {'best_costs': best_costs}

    def finalize(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return state['nominal'].detach().cpu()

    def _cost_summary(self, payload: dict, costs: torch.Tensor) -> list[float]:
        # Output 'costs' lists the best-per-env cost from the final step.
        return payload['best_costs'].cpu().tolist()
