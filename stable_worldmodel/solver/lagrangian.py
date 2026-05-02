"""Lagrangian solver for stable world model."""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.spaces import Box
from loguru import logger as logging

from .base import BaseSolver
from .solver import Costable


class LagrangianSolver(BaseSolver):
    """Augmented Lagrangian solver with nested primal/dual updates.

    L = cost + Σ_i λ_i * g_i + Σ_i ρ * max(0, g_i)^2

    Args:
        model: Cost model. If it implements ``get_constraints(infos, actions)
            -> (B, S, C)``, the solver enforces constraints (satisfied when
            ``constraint <= 0``).
        n_steps: Inner gradient descent steps per outer iteration.
        n_outer_steps: Number of dual ascent (outer) iterations.
        batch_size: Number of environments to process in parallel.
        num_samples: Number of action samples to optimize in parallel.
        var_scale: Initial variance scale for action perturbations.
        action_noise: Per-step Gaussian action noise std.
        rho_init: Initial quadratic penalty coefficient.
        rho_max: Cap on the quadratic penalty.
        rho_scale: Multiplicative growth factor for rho per outer step.
        persist_multipliers: Warm-start λ across solve() calls.
        device: Device for tensor computations.
        seed: Random seed for reproducibility.
        optimizer_cls: PyTorch optimizer class.
        optimizer_kwargs: Keyword arguments for the optimizer.
    """

    def __init__(
        self,
        model: Costable,
        n_steps: int,
        n_outer_steps: int = 5,
        batch_size: int | None = None,
        num_samples: int = 1,
        var_scale: float = 1.0,
        action_noise: float = 0.0,
        rho_init: float = 1.0,
        rho_max: float = 1e4,
        rho_scale: float = 2.0,
        persist_multipliers: bool = True,
        device: str | torch.device = 'cpu',
        seed: int = 1234,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            model=model,
            n_steps=n_steps,
            batch_size=batch_size,
            num_samples=num_samples,
            device=device,
            callbacks=None,
        )
        self.n_outer_steps = n_outer_steps
        self.var_scale = var_scale
        self.action_noise = action_noise
        self.rho_init = rho_init
        self.rho_max = rho_max
        self.rho_scale = rho_scale
        self.persist_multipliers = persist_multipliers
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = (
            optimizer_kwargs if optimizer_kwargs is not None else {'lr': 1.0}
        )
        self._lambdas: torch.Tensor | None = None
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
                'LagrangianSolver may not work as expected.'
            )

    @property
    def action_dim(self) -> int:
        return self._action_dim * self.action_block

    def init_action(self, actions: torch.Tensor | None = None) -> None:
        """Initialize ``self.init`` to ``(n_envs, num_samples, horizon, action_dim)``."""
        if actions is None:
            actions = torch.zeros((self._n_envs, 0, self.action_dim))
        remaining = self.horizon - actions.shape[1]
        if remaining > 0:
            new_actions = torch.zeros(self._n_envs, remaining, self.action_dim)
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

    def _init_multipliers(self, num_constraints: int) -> None:
        self._lambdas = torch.zeros(
            self._n_envs, num_constraints, device=self.device
        )

    def _augmented_lagrangian_loss(
        self,
        costs: torch.Tensor,
        constraints: torch.Tensor,
        lambdas_batch: torch.Tensor,
        rho: float,
    ) -> torch.Tensor:
        linear_penalty = (lambdas_batch.unsqueeze(1) * constraints).sum(dim=-1)
        quadratic_penalty = rho * F.relu(constraints).pow(2).sum(dim=-1)
        return (costs + linear_penalty + quadratic_penalty).sum()

    def _update_multipliers(
        self,
        constraints: torch.Tensor,
        lambdas_batch: torch.Tensor,
        rho: float,
    ) -> torch.Tensor:
        mean_g = constraints.mean(dim=1)
        return torch.clamp(lambdas_batch + rho * mean_g, min=0.0)

    # === BaseSolver hooks (unused — solve overrides the loop) ===

    def init_state(self, n_envs: int, init=None) -> None:  # pragma: no cover
        raise NotImplementedError(
            'LagrangianSolver overrides solve(); hooks are unused.'
        )

    def propose(self, state):  # pragma: no cover
        raise NotImplementedError

    def update(self, state, candidates, costs):  # pragma: no cover
        raise NotImplementedError

    def finalize(self, state):  # pragma: no cover
        raise NotImplementedError

    # === Custom outer/inner loop ===

    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        start_time = time.time()
        outputs: dict = {
            'cost': [],
            'constraint_violation': [],
            'actions': None,
            'lambdas': None,
        }

        with torch.no_grad():
            self.init_action(init_action)

        if not self.persist_multipliers:
            self._lambdas = None

        bs = self.batch_size if self.batch_size is not None else self._n_envs
        total_envs = self._n_envs
        batch_top_actions: list[torch.Tensor] = []

        for start in range(0, total_envs, bs):
            end = min(start + bs, total_envs)
            current_bs = end - start

            batch_init = self.init[start:end].clone().detach()
            batch_init.requires_grad = True

            expanded_infos = self._expand_infos(
                info_dict, start, end, current_bs
            )
            for k, v in expanded_infos.items():
                if torch.is_tensor(v):
                    expanded_infos[k] = v.to(self.device)

            rho = self.rho_init
            batch_cost_history: list[float] = []
            costs = None
            final_constraints = None
            lambdas_batch = None

            for _outer in range(self.n_outer_steps):
                optim = self.optimizer_cls(
                    [batch_init], **self.optimizer_kwargs
                )

                for _step in range(self.n_steps):
                    costs = self.model.get_cost(
                        expanded_infos.copy(), batch_init
                    )
                    constraints = (
                        self.model.get_constraints(
                            expanded_infos.copy(), batch_init
                        )
                        if hasattr(self.model, 'get_constraints')
                        else None
                    )

                    assert isinstance(costs, torch.Tensor), (
                        f'Got {type(costs)} cost, expect torch.Tensor'
                    )
                    assert costs.ndim == 2 and costs.shape == (
                        current_bs,
                        self.num_samples,
                    ), (
                        f'Cost should be of shape ({current_bs}, '
                        f'{self.num_samples}), got {costs.shape}'
                    )
                    assert costs.requires_grad, (
                        'Cost must requires_grad for LagrangianSolver.'
                    )

                    if constraints is not None:
                        assert constraints.ndim == 3 and constraints.shape[
                            :2
                        ] == (current_bs, self.num_samples), (
                            f'Constraints should be of shape ({current_bs}, '
                            f'{self.num_samples}, C), got {constraints.shape}'
                        )
                        if self._lambdas is None:
                            self._init_multipliers(constraints.shape[-1])
                        lambdas_batch = self._lambdas[start:end]
                        loss = self._augmented_lagrangian_loss(
                            costs, constraints, lambdas_batch, rho
                        )
                    else:
                        loss = costs.sum()

                    loss.backward()
                    optim.step()
                    optim.zero_grad(set_to_none=True)

                    if self.action_noise > 0:
                        batch_init.data += (
                            torch.randn(
                                batch_init.shape, generator=self.torch_gen
                            )
                            * self.action_noise
                        )

                    batch_cost_history.append(loss.item())

                if constraints is not None:
                    with torch.no_grad():
                        final_constraints = self.model.get_constraints(
                            expanded_infos.copy(), batch_init
                        )
                        lambdas_batch = self._update_multipliers(
                            final_constraints, lambdas_batch, rho
                        )
                        self._lambdas[start:end] = lambdas_batch
                        rho = min(self.rho_max, rho * self.rho_scale)

                with torch.no_grad():
                    mean_cost = costs.mean().item()
                    if constraints is not None:
                        viol = F.relu(final_constraints).mean(dim=(0, 1))
                        lam = lambdas_batch.mean(dim=0)
                        viol_str = ', '.join(f'{v:.4f}' for v in viol.tolist())
                        lam_str = ', '.join(f'{lv:.4f}' for lv in lam.tolist())
                        print(
                            f'  [outer {_outer + 1}/{self.n_outer_steps}] '
                            f'cost={mean_cost:.4f} | '
                            f'constraint_viol=[{viol_str}] | '
                            f'lambdas=[{lam_str}] | '
                            f'rho={rho:.4f}'
                        )
                    else:
                        print(
                            f'  [outer {_outer + 1}/{self.n_outer_steps}] '
                            f'cost={mean_cost:.4f}'
                        )

            outputs['cost'].append(batch_cost_history)
            if final_constraints is not None:
                outputs['constraint_violation'].append(
                    F.relu(final_constraints).mean().item()
                )

            with torch.no_grad():
                self.init[start:end] = batch_init

            top_idx = torch.argsort(costs, dim=1)[:, 0]
            batch_indices = torch.arange(current_bs)
            batch_top_actions.append(
                batch_init[batch_indices, top_idx].detach().cpu()
            )

        outputs['actions'] = torch.cat(batch_top_actions, dim=0)
        outputs['lambdas'] = (
            self._lambdas.cpu() if self._lambdas is not None else None
        )

        constraint_info = ''
        if outputs['constraint_violation']:
            mean_viol = np.mean(outputs['constraint_violation'])
            constraint_info = f' | constraint_violation={mean_viol:.4f}'
        print(
            f'LagrangianSolver.solve completed in '
            f'{time.time() - start_time:.4f} seconds{constraint_info}.'
        )
        return outputs
