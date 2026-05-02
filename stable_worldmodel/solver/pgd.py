"""Projected Gradient Descent solver for discrete action spaces."""

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete

from .base import BaseSolver
from .solver import Costable


class PGDSolver(BaseSolver):
    """Projected Gradient Descent solver for discrete action optimization.

    Args:
        model: World model implementing the Costable protocol.
        n_steps: Number of gradient descent iterations.
        batch_size: Number of environments to process in parallel.
        var_scale: Initial variance scale for action perturbations.
        num_samples: Number of action samples to optimize in parallel.
        action_noise: Noise added to actions during optimization.
        device: Device for tensor computations.
        seed: Random seed for reproducibility.
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
    ) -> None:
        super().__init__(
            model=model,
            n_steps=n_steps,
            batch_size=batch_size,
            num_samples=num_samples,
            device=device,
            callbacks=None,
        )
        self.var_scale = var_scale
        self.action_noise = action_noise
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)

        self._action_simplex_dim = None
        self.init: torch.Tensor | None = None
        self._from_scalar_pending: bool = False

    def configure(
        self, *, action_space: gym.Space, n_envs: int, config: Any
    ) -> None:
        assert isinstance(action_space, Discrete), (
            f'Action space must be discrete, got {type(action_space)}'
        )
        super().configure(
            action_space=action_space, n_envs=n_envs, config=config
        )
        shape = action_space.shape
        self._action_dim = (
            int(np.prod(shape[1:]))
            if len(shape) > 1
            else int(np.prod(shape) or 1)
        )
        self._action_simplex_dim = int(action_space.n)

    @property
    def action_dim(self) -> int:
        return self._action_dim * self.action_block

    @property
    def action_simplex_dim(self) -> int:
        return self._action_simplex_dim * self.action_block

    def init_action(
        self, actions: torch.Tensor | None = None, from_scalar: bool = False
    ) -> None:
        """Initialize ``self.init`` to ``(n_envs, num_samples, horizon, action_simplex_dim)``."""
        if actions is None:
            actions = torch.zeros((self._n_envs, 0, self.action_simplex_dim))
        elif from_scalar:
            actions = torch.nn.functional.one_hot(
                actions, num_classes=self._action_simplex_dim
            ).to(torch.float32)
            actions = actions.reshape(
                *actions.shape[:-2], self.action_simplex_dim
            )
            assert (
                actions.shape[0] == self._n_envs
                and actions.shape[1] <= self.horizon
                and actions.shape[2] == self.action_simplex_dim
            )

        remaining = self.horizon - actions.shape[1]
        if remaining > 0:
            new_actions = torch.zeros(
                self._n_envs, remaining, self.action_simplex_dim
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
            )
            * self.var_scale
        )
        self.init = actions

    def _factor_action_block(self, actions: torch.Tensor) -> torch.Tensor:
        original_shape = actions.shape
        return actions.reshape(
            *original_shape[:-1], self.action_block, self._action_simplex_dim
        )

    def _project_action_simplex(self, actions: torch.Tensor) -> torch.Tensor:
        original_shape = actions.shape
        s = self._factor_action_block(actions).reshape(
            -1, self._action_simplex_dim
        )
        mu, _ = torch.sort(s, descending=True, dim=-1)
        cumulative = mu.cumsum(dim=-1)
        d = s.size(-1)
        indices = torch.arange(1, d + 1, device=s.device, dtype=s.dtype)
        threshold = (cumulative - 1) / indices
        cond = (mu > threshold).to(torch.int32)
        rho = cond.cumsum(dim=-1)
        valid_rho = rho * cond
        rho_max = valid_rho.max(dim=-1, keepdim=True)[0]
        rho_min = torch.clamp(rho_max, min=1)
        psi = (cumulative.gather(-1, rho_min - 1) - 1) / rho_min
        return torch.clamp(s - psi, min=0.0).reshape(original_shape)

    # === BaseSolver hooks ===

    def solve(
        self,
        info_dict: dict,
        init_action: torch.Tensor | None = None,
        from_scalar: bool = False,
    ) -> dict:
        # Stash from_scalar so init_state can pick it up via the BaseSolver loop.
        self._from_scalar_pending = from_scalar
        try:
            return super().solve(info_dict, init_action=init_action)
        finally:
            self._from_scalar_pending = False

    def init_state(
        self, n_envs: int, init: torch.Tensor | None = None
    ) -> dict[str, Any]:
        with torch.no_grad():
            self.init_action(init, from_scalar=self._from_scalar_pending)
        best = torch.zeros(
            n_envs, self.horizon, self.action_block, dtype=torch.long
        )
        return {'init': self.init, 'best': best}

    def _slice_state(
        self, state: dict[str, Any], start: int, end: int
    ) -> dict[str, Any]:
        params = state['init'][start:end].clone().detach()
        params.requires_grad = True
        return {
            'params': params,
            'optim': torch.optim.SGD([params], lr=1.0),
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
                top_one_hot = bstate['params'][idx, top_idx]
                state['best'][start:end] = (
                    self._factor_action_block(top_one_hot).argmax(dim=-1).cpu()
                )

    def propose(self, state: dict[str, Any]) -> torch.Tensor:
        return state['params']

    def update(
        self,
        state: dict[str, Any],
        candidates: torch.Tensor,
        costs: torch.Tensor,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not costs.requires_grad:
            raise RuntimeError('Cost must require grad for PGDSolver.')

        cost = costs.sum()
        cost.backward()
        state['optim'].step()
        state['optim'].zero_grad(set_to_none=True)

        if self.action_noise > 0:
            with torch.no_grad():
                candidates.data += (
                    torch.randn(
                        candidates.shape,
                        generator=self.torch_gen,
                        device=candidates.device,
                    )
                    * self.action_noise
                )

        with torch.no_grad():
            candidates.copy_(self._project_action_simplex(candidates))

        state['last_costs'] = costs.detach()
        return state, {'params': candidates, 'cost': cost}

    def finalize(self, state: dict[str, Any]) -> torch.Tensor:
        return state['best']

    def _cost_summary(self, payload: dict, costs: torch.Tensor) -> list[float]:
        return costs.detach().min(dim=1).values.cpu().tolist()
