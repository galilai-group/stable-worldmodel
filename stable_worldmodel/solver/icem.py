"""Improved Cross Entropy Method (iCEM) solver for model-based planning."""

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging

from .base import BaseSolver
from .callbacks import Callback
from .solver import Costable


class ICEMSolver(BaseSolver):
    """Improved Cross Entropy Method (iCEM) solver with colored noise and elite retention.

    [1] Pinneri et al., "Sample-efficient Cross-Entropy Method for Real-time Planning",
    Conference on Robot Learning, 2020.

    Args:
        model: World model implementing the Costable protocol.
        batch_size: Number of environments to process in parallel.
        num_samples: Number of action candidates to sample per iteration.
        var_scale: Initial variance scale for the action distribution.
        n_steps: Number of CEM iterations.
        topk: Number of elite samples to keep for distribution update.
        noise_beta: Colored noise exponent (0 = white).
        alpha: Momentum for mean/std EMA update.
        n_elite_keep: Number of elites carried from previous iteration.
        return_mean: If False, return best single trajectory instead of mean.
        device: Device for tensor computations.
        seed: Random seed for reproducibility.
        callbacks: Optional list of callbacks.
    """

    def __init__(
        self,
        model: Costable,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1,
        n_steps: int = 30,
        topk: int = 30,
        noise_beta: float = 2.0,
        alpha: float = 0.1,
        n_elite_keep: int = 5,
        return_mean: bool = True,
        device: str | torch.device = 'cpu',
        seed: int = 1234,
        callbacks: list[Callback] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            n_steps=n_steps,
            batch_size=batch_size,
            num_samples=num_samples,
            device=device,
            callbacks=callbacks,
        )
        self.var_scale = var_scale
        self.topk = topk
        self.noise_beta = noise_beta
        self.alpha = alpha
        self.n_elite_keep = n_elite_keep
        self.return_mean = return_mean
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

        if isinstance(action_space, Box):
            # candidates have last-dim action_dim * action_block; tile bounds
            # so clamp broadcasts over the flattened block.
            self._action_low = torch.tensor(
                action_space.low[0], device=self.device, dtype=self.dtype
            ).repeat(self.action_block)
            self._action_high = torch.tensor(
                action_space.high[0], device=self.device, dtype=self.dtype
            ).repeat(self.action_block)
        else:
            logging.warning(
                f'Action space is discrete, got {type(action_space)}. '
                'ICEMSolver may not work as expected.'
            )
            self._action_low = None
            self._action_high = None

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
        return mean, var

    def _colored_noise(self, bs: int) -> torch.Tensor:
        """Sample colored noise of shape (bs, num_samples, horizon, action_dim)."""
        noise_shape = (bs, self.num_samples, self.action_dim, self.horizon)
        if self.horizon <= 1:
            noise = torch.randn(
                noise_shape,
                generator=self.torch_gen,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            freqs = torch.fft.rfftfreq(self.horizon, device=self.device).to(
                self.dtype
            )
            freqs[0] = 1.0
            scale = freqs.pow(-self.noise_beta / 2)
            scale[0] = scale[1]
            white = torch.randn(
                noise_shape,
                generator=self.torch_gen,
                device=self.device,
                dtype=self.dtype,
            )
            fft = torch.fft.rfft(white, dim=-1)
            colored = torch.fft.irfft(fft * scale, n=self.horizon, dim=-1)
            std = colored.std(dim=-1, keepdim=True).clamp(min=1e-8)
            noise = colored / std
        return noise.transpose(-1, -2)

    # === BaseSolver hooks ===

    def init_state(
        self, n_envs: int, init: torch.Tensor | None = None
    ) -> dict[str, Any]:
        mean, var = self.init_action_distrib(n_envs, init)
        return {
            'mean': mean.to(self.device),
            'var': var.to(self.device),
            'prev_elites': None,
        }

    def propose(self, state: dict[str, Any]) -> torch.Tensor:
        bs = state['mean'].shape[0]
        noise = self._colored_noise(bs)
        cands = noise * state['var'].unsqueeze(1) + state['mean'].unsqueeze(1)
        cands[:, 0] = state['mean']

        prev = state['prev_elites']
        if prev is not None:
            n_inject = min(self.n_elite_keep, prev.shape[1])
            cands[:, 1 : 1 + n_inject] = prev[:, :n_inject]

        if self._action_low is not None:
            cands = cands.clamp(self._action_low, self._action_high)
        return cands

    def update(
        self,
        state: dict[str, Any],
        candidates: torch.Tensor,
        costs: torch.Tensor,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        topk_vals, topk_inds = torch.topk(
            costs, k=self.topk, dim=1, largest=False
        )
        bs = candidates.shape[0]
        batch_indices = (
            torch.arange(bs, device=self.device)
            .unsqueeze(1)
            .expand(-1, self.topk)
        )
        topk_candidates = candidates[batch_indices, topk_inds]

        elite_mean = topk_candidates.mean(dim=1)
        elite_var = topk_candidates.std(dim=1)
        new_mean = self.alpha * state['mean'] + (1 - self.alpha) * elite_mean
        new_var = self.alpha * state['var'] + (1 - self.alpha) * elite_var

        # Stash the best trajectory in case return_mean=False.
        new_state = {
            'mean': new_mean,
            'var': new_var,
            'prev_elites': topk_candidates,
            'best_traj': topk_candidates[:, 0],
        }
        payload = {
            'topk_vals': topk_vals,
            'topk_inds': topk_inds,
            'topk_candidates': topk_candidates,
            'mean': new_mean,
            'var': new_var,
            'prev_mean': state['mean'],
            'prev_var': state['var'],
            'action_low': self._action_low,
            'action_high': self._action_high,
        }
        return new_state, payload

    def finalize(self, state: dict[str, Any]) -> torch.Tensor:
        chosen = (
            state['mean']
            if self.return_mean
            else state.get('best_traj', state['mean'])
        )
        return chosen.detach().cpu()

    def extra_outputs(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            'mean': [state['mean'].detach().cpu()],
            'var': [state['var'].detach().cpu()],
        }

    def _slice_state(
        self, state: dict[str, Any], start: int, end: int
    ) -> dict[str, Any]:
        # prev_elites/best_traj are absent at first slice; preserve None.
        return {
            'mean': state['mean'][start:end],
            'var': state['var'][start:end],
            'prev_elites': None,
        }

    def _write_back_state(
        self,
        state: dict[str, Any],
        bstate: dict[str, Any],
        start: int,
        end: int,
    ) -> None:
        if self.return_mean:
            state['mean'][start:end] = bstate['mean']
        else:
            state['mean'][start:end] = bstate.get('best_traj', bstate['mean'])
        state['var'][start:end] = bstate['var']
