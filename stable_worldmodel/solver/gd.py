"""Gradient-based solver for model-based planning."""

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging

from .base import BaseSolver
from .callbacks import Callback
from .solver import Costable


class GradientSolver(BaseSolver):
    """Gradient-based solver using backpropagation through the world model.

    Args:
        model: World model implementing the Costable protocol.
        n_steps: Number of gradient descent iterations.
        batch_size: Number of environments to process in parallel.
        var_scale: Initial variance scale for action perturbations.
        num_samples: Number of action samples to optimize in parallel.
        action_noise: Noise added to actions during optimization.
        device: Device for tensor computations.
        seed: Random seed for reproducibility.
        optimizer_cls: PyTorch optimizer class to use.
        optimizer_kwargs: Keyword arguments for the optimizer.
        grad_clip: Optional max-norm clip on the action gradient.
        callbacks: Optional list of callbacks.
    """

    def __init__(
        self,
        model: Costable,
        n_steps: int,
        batch_size: int | None = None,
        var_scale: float = 1,
        num_samples: int = 1,
        action_noise: float = 0.0,
        device: str | torch.device = 'cpu',
        seed: int = 1234,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.SGD,
        optimizer_kwargs: dict | None = None,
        grad_clip: float | None = None,
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
        self.action_noise = action_noise
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = (
            optimizer_kwargs if optimizer_kwargs is not None else {'lr': 1.0}
        )
        self.grad_clip = grad_clip
        self.init: torch.Tensor | None = None

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
                'GradientSolver may not work as expected.'
            )

    @property
    def action_dim(self) -> int:
        return self._action_dim * self.action_block

    def init_action(
        self, n_envs: int, actions: torch.Tensor | None = None
    ) -> None:
        """Initialize ``self.init`` with shape ``(n_envs, num_samples, horizon, action_dim)``."""
        if actions is None:
            actions = torch.zeros(
                (n_envs, 0, self.action_dim), dtype=self.dtype
            )
        remaining = self.horizon - actions.shape[1]
        if remaining > 0:
            new_actions = torch.zeros(
                n_envs, remaining, self.action_dim, dtype=self.dtype
            )
            actions = torch.cat([actions, new_actions], dim=1).to(self.device)
        actions = actions.unsqueeze(1).repeat_interleave(
            self.num_samples, dim=1
        )
        actions[:, 1:] += (
            torch.randn(
                actions[:, 1:].shape,
                generator=self.torch_gen,
                device=self.device,
                dtype=self.dtype,
            )
            * self.var_scale
        )
        self.init = actions

    # === BaseSolver hooks ===

    def init_state(
        self, n_envs: int, init: torch.Tensor | None = None
    ) -> dict[str, Any]:
        with torch.no_grad():
            self.init_action(n_envs, init)
        # `best` holds the per-env best trajectory (filled in at write-back).
        best = torch.zeros(
            n_envs, self.horizon, self.action_dim, dtype=self.dtype
        )
        return {'init': self.init, 'best': best}

    def _slice_state(
        self, state: dict[str, Any], start: int, end: int
    ) -> dict[str, Any]:
        params = state['init'][start:end].clone().detach()
        params.requires_grad = True
        return {
            'params': params,
            'optim': self.optimizer_cls([params], **self.optimizer_kwargs),
            'last_costs': None,
        }

    def _write_back_state(
        self,
        state: dict[str, Any],
        bstate: dict[str, Any],
        start: int,
        end: int,
    ) -> None:
        with torch.no_grad():
            state['init'][start:end] = bstate['params']
            costs = bstate['last_costs']
            if costs is not None:
                top_idx = costs.argmin(dim=1)
                bs = bstate['params'].shape[0]
                idx = torch.arange(bs, device=bstate['params'].device)
                state['best'][start:end] = bstate['params'][idx, top_idx].cpu()

    def propose(self, state: dict[str, Any]) -> torch.Tensor:
        return state['params']

    def update(
        self,
        state: dict[str, Any],
        candidates: torch.Tensor,
        costs: torch.Tensor,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not costs.requires_grad:
            raise RuntimeError('Cost must require grad for GradientSolver.')

        # Zero before backward, not after step, so candidates.grad survives
        # until callbacks fire (BaseSolver runs them after update returns).
        state['optim'].zero_grad(set_to_none=True)
        cost = costs.sum()
        cost.backward()

        payload = {'params': candidates, 'cost': cost}

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(candidates, self.grad_clip)

        state['optim'].step()

        if self.action_noise > 0:
            with torch.no_grad():
                candidates.data += (
                    torch.randn(
                        candidates.shape,
                        generator=self.torch_gen,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    * self.action_noise
                )

        state['last_costs'] = costs.detach()
        return state, payload

    def finalize(self, state: dict[str, Any]) -> torch.Tensor:
        return state['best']

    def _cost_summary(self, payload: dict, costs: torch.Tensor) -> list[float]:
        return costs.detach().min(dim=1).values.cpu().tolist()
