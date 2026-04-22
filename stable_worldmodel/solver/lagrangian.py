"""Lagrangian solver for stable world model."""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.spaces import Box
from loguru import logger as logging

from stable_worldmodel.protocols import Costable
from stable_worldmodel.solver.utils import prepare_init_action


class LagrangianSolver(torch.nn.Module):
    """Lagrangian solver for stable world model.

    get_cost returns the cost tensor (B, S). If the model also implements get_constraints,
    it should return the constraint violations (B, S, C), where C is the number of constraints.
    The constraint_cost should represent the cost of violating the constraints, where the constraint
    is satisfied when constraint_cost <= 0. The Lagrangian solver will optimize the following objective:

    L = cost + sum_{i=1}^C lambda_i * constraint_cost_i + sum_{i=1}^C rho_i * max(0, constraint_cost_i)^2

    If you want to use equality constraint, you can convert it to two inequality constraints. For example, if you want to enforce constraint_cost_i == 0, you can add two constraints: constraint_cost_i <= 0 and -constraint_cost_i <= 0.

    Args:
        model: World model implementing the Costable protocol. Its get_cost() returns
            a plain cost tensor (B, S). If it also has get_constraints(), that method
            returns constraints of shape (B, S, C).
        n_steps: Number of gradient descent steps per outer iteration.
        n_outer_steps: Number of dual ascent (outer) iterations.
        batch_size: Number of environments to process in parallel.
        num_samples: Number of action samples to optimize in parallel.
        var_scale: Initial variance scale for action perturbations.
        action_noise: Noise added to actions during optimization.
        rho_init: Initial penalty coefficient for the quadratic constraint term.
        rho_max: Maximum value of the penalty coefficient.
        rho_scale: Multiplicative growth factor for rho after each outer step.
        persist_multipliers: Whether to warm-start Lagrange multipliers across solve() calls.
        device: Device for tensor computations.
        seed: Random seed for reproducibility.
        optimizer_cls: PyTorch optimizer class to use.
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
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.n_outer_steps = n_outer_steps
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.action_noise = action_noise
        self.rho_init = rho_init
        self.rho_max = rho_max
        self.rho_scale = rho_scale
        self.persist_multipliers = persist_multipliers
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = (
            optimizer_kwargs if optimizer_kwargs is not None else {'lr': 1.0}
        )

        self._configured = False
        self._n_envs = None
        self._action_dim = None
        self._config = None
        self._lambdas: torch.Tensor | None = None  # (n_envs, C)

    def configure(
        self, *, action_space: gym.Space, n_envs: int, config: Any
    ) -> None:
        """Configure the solver with environment specifications."""
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

        if not isinstance(action_space, Box):
            logging.warning(
                f'Action space is discrete, got {type(action_space)}. LagrangianSolver may not work as expected.'
            )

    @property
    def n_envs(self) -> int:
        """Number of parallel environments."""
        return self._n_envs

    @property
    def action_dim(self) -> int:
        """Flattened action dimension including action_block grouping."""
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        """Planning horizon in timesteps."""
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        """Make solver callable, forwarding to solve()."""
        return self.solve(*args, **kwargs)

    def init_action(self, actions: torch.Tensor) -> None:
        """Initialize the action tensor for optimization."""
        assert actions.shape == (self.n_envs, self.horizon, self.action_dim), (
            f'Expected actions shape ({self.n_envs}, {self.horizon}, {self.action_dim}), '
            f'got {tuple(actions.shape)}'
        )
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

        if hasattr(self, 'init'):
            self.init.copy_(actions)
        else:
            self.register_parameter('init', torch.nn.Parameter(actions))

    def _init_multipliers(self, num_constraints: int) -> None:
        """Lazily initialize Lagrange multipliers to zeros."""
        self._lambdas = torch.zeros(
            self._n_envs, num_constraints, device=self.device
        )

    def _augmented_lagrangian_loss(
        self,
        costs: torch.Tensor,  # (B, S)
        constraints: torch.Tensor,  # (B, S, C)
        lambdas_batch: torch.Tensor,  # (B, C)
        rho: float,
    ) -> torch.Tensor:
        """Compute the augmented Lagrangian loss.

        L = cost + Σ_i lambda_i * g_i + Σ_i rho * max(0, g_i)^2
        """
        # lambdas_batch: (B, C) -> (B, 1, C) for broadcasting with constraints (B, S, C)
        linear_penalty = (lambdas_batch.unsqueeze(1) * constraints).sum(
            dim=-1
        )  # (B, S)
        quadratic_penalty = rho * F.relu(constraints).pow(2).sum(
            dim=-1
        )  # (B, S)
        return (costs + linear_penalty + quadratic_penalty).sum()

    def _update_multipliers(
        self,
        constraints: torch.Tensor,  # (B, S, C) — detached, no grad
        lambdas_batch: torch.Tensor,  # (B, C)
        rho: float,
    ) -> torch.Tensor:
        """Dual ascent: lambda_i <- max(0, lambda_i + rho * mean_samples(g_i))."""
        mean_g = constraints.mean(dim=1)  # (B, C)
        return torch.clamp(lambdas_batch + rho * mean_g, min=0.0)

    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        """Solve the planning problem using augmented Lagrangian gradient descent."""
        start_time = time.time()
        outputs: dict = {
            'cost': [],
            'constraint_violation': [],
            'actions': None,
            'lambdas': None,
        }

        with torch.no_grad():
            init_action = prepare_init_action(
                self.model,
                info_dict,
                init_action,
                self.horizon,
                n_envs=self.n_envs,
                action_dim=self.action_dim,
            )
            self.init_action(init_action)

        if not self.persist_multipliers:
            self._lambdas = None

        batch_size = (
            self.batch_size if self.batch_size is not None else self.n_envs
        )
        total_envs = self.n_envs
        batch_top_actions_list = []

        for start_idx in range(0, total_envs, batch_size):
            end_idx = min(start_idx + batch_size, total_envs)
            current_bs = end_idx - start_idx

            batch_init = self.init[start_idx:end_idx].clone().detach()
            batch_init.requires_grad = True

            # Expand info_dict for current batch — same pattern as GradientSolver
            expanded_infos = {}
            for k, v in info_dict.items():
                if torch.is_tensor(v):
                    batch_v = v[start_idx:end_idx]
                    batch_v = batch_v.unsqueeze(1)
                    batch_v = batch_v.expand(
                        current_bs, self.num_samples, *batch_v.shape[2:]
                    )
                elif isinstance(v, np.ndarray):
                    batch_v = v[start_idx:end_idx]
                    batch_v = np.repeat(
                        batch_v[:, None, ...], self.num_samples, axis=1
                    )
                else:
                    batch_v = v
                expanded_infos[k] = batch_v

            rho = self.rho_init
            batch_cost_history = []
            costs = None
            final_constraints = None

            for _outer in range(self.n_outer_steps):
                # Fresh optimizer each outer step — avoids stale momentum after dual ascent
                optim = self.optimizer_cls(
                    [batch_init], **self.optimizer_kwargs
                )

                for _step in range(self.n_steps):
                    current_info = expanded_infos.copy()
                    costs = self.model.get_cost(current_info, batch_init)
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
                        f'Cost should be of shape ({current_bs}, {self.num_samples}), got {costs.shape}'
                    )
                    assert costs.requires_grad, (
                        'Cost must requires_grad for LagrangianSolver.'
                    )

                    if constraints is not None:
                        assert constraints.ndim == 3 and constraints.shape[
                            :2
                        ] == (current_bs, self.num_samples), (
                            f'Constraints should be of shape ({current_bs}, {self.num_samples}, C), got {constraints.shape}'
                        )
                        if self._lambdas is None:
                            self._init_multipliers(constraints.shape[-1])
                        lambdas_batch = self._lambdas[start_idx:end_idx]
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

                # Dual ascent after inner loop converges
                if constraints is not None:
                    with torch.no_grad():
                        final_constraints = self.model.get_constraints(
                            expanded_infos.copy(), batch_init
                        )
                        lambdas_batch = self._update_multipliers(
                            final_constraints, lambdas_batch, rho
                        )
                        self._lambdas[start_idx:end_idx] = lambdas_batch
                        rho = min(self.rho_max, rho * self.rho_scale)

                with torch.no_grad():
                    mean_cost = costs.mean().item()
                    if constraints is not None:
                        viol = F.relu(final_constraints).mean(dim=(0, 1))  # (C,)
                        lam = lambdas_batch.mean(dim=0)  # (C,)
                        viol_str = ', '.join(f'{v:.4f}' for v in viol.tolist())
                        lam_str = ', '.join(f'{l:.4f}' for l in lam.tolist())
                        print(
                            f'  [outer {_outer+1}/{self.n_outer_steps}] '
                            f'cost={mean_cost:.4f} | '
                            f'constraint_viol=[{viol_str}] | '
                            f'lambdas=[{lam_str}] | '
                            f'rho={rho:.4f}'
                        )
                    else:
                        print(
                            f'  [outer {_outer+1}/{self.n_outer_steps}] '
                            f'cost={mean_cost:.4f}'
                        )

            outputs['cost'].append(batch_cost_history)

            if final_constraints is not None:
                outputs['constraint_violation'].append(
                    F.relu(final_constraints).mean().item()
                )

            with torch.no_grad():
                self.init[start_idx:end_idx] = batch_init

            top_idx = torch.argsort(costs, dim=1)[:, 0]
            batch_indices = torch.arange(current_bs)
            top_actions_batch = batch_init[batch_indices, top_idx]
            batch_top_actions_list.append(top_actions_batch.detach().cpu())

        outputs['actions'] = torch.cat(batch_top_actions_list, dim=0)
        outputs['lambdas'] = (
            self._lambdas.cpu() if self._lambdas is not None else None
        )

        constraint_info = ''
        if outputs['constraint_violation']:
            mean_viol = np.mean(outputs['constraint_violation'])
            constraint_info = f' | constraint_violation={mean_viol:.4f}'
        print(
            f'LagrangianSolver.solve completed in {time.time() - start_time:.4f} seconds{constraint_info}.'
        )
        return outputs
