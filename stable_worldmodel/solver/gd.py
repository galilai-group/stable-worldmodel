"""Gradient-based solver for model-based planning."""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging

from stable_worldmodel.solver.utils import prepare_init_action
from .callbacks import Callback
from .solver import Costable


class GradientSolver(torch.nn.Module):
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
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.action_noise = action_noise
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = (
            optimizer_kwargs if optimizer_kwargs is not None else {'lr': 1.0}
        )
        self.grad_clip = grad_clip
        self.callbacks = list(callbacks) if callbacks else []

        try:
            self._dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            self._dtype = torch.float32

        self._configured = False
        self._n_envs = None
        self._action_dim = None
        self._config = None

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
                f'Action space is discrete, got {type(action_space)}. GradientSolver may not work as expected.'
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

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        """Make solver callable, forwarding to solve()."""
        return self.solve(*args, **kwargs)

    def init_action(
        self, n_envs: int, actions: torch.Tensor | None = None
    ) -> None:
        """Initialize the action tensor for optimization."""
        if actions is None:
            actions = torch.zeros(
                (n_envs, 0, self.action_dim), dtype=self.dtype
            )

        # fill remaining action
        remaining = self.horizon - actions.shape[1]

        if remaining > 0:
            new_actions = torch.zeros(
                n_envs, remaining, self.action_dim, dtype=self.dtype
            )
            actions = torch.cat([actions, new_actions], dim=1).to(self.device)


        actions = actions.unsqueeze(1).repeat_interleave(
            self.num_samples, dim=1
        )  # add sample dim
        actions[:, 1:] += (
            torch.randn(
                actions[:, 1:].shape,
                generator=self.torch_gen,
                device=self.device,
                dtype=self.dtype,
            )
            * self.var_scale
        )  # add small noise to all samples except the first one

        # reset actions — re-register when shape differs (batch size may vary across calls)
        if hasattr(self, 'init') and self.init.shape == actions.shape:
            self.init.copy_(actions)
        else:
            if 'init' in self._parameters:
                del self._parameters['init']
            self.register_parameter('init', torch.nn.Parameter(actions))

    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        """Solve the planning problem using gradient descent."""
        start_time = time.time()
        outputs = {
            'cost': [],  # Will store list of cost histories per batch
            'actions': None,
        }

        # Batch size is taken from info_dict so callers can solve for a subset of envs
        total_envs = len(next(iter(info_dict.values())))

        with torch.no_grad():
            init_action = prepare_init_action(
                self.model,
                info_dict,
                init_action,
                self.horizon,
                n_envs=total_envs,
                action_dim=self.action_dim,
            )
            self.init_action(total_envs, init_action)

        for cb in self.callbacks:
            cb.reset()

        # Determine batch size (default to all envs if not specified which can cause memory issues)
        batch_size = (
            self.batch_size if self.batch_size is not None else total_envs
        )

        # Lists to hold results from each batch to be concatenated later
        batch_top_actions_list = []

        # --- Outer Loop: Iterate over batches ---
        for start_idx in range(0, total_envs, batch_size):
            end_idx = min(start_idx + batch_size, total_envs)
            current_bs = end_idx - start_idx

            batch_init = self.init[start_idx:end_idx].clone().detach()
            batch_init.requires_grad = True

            # We initialize the optimizer class passed in __init__ with the kwargs
            optim = self.optimizer_cls([batch_init], **self.optimizer_kwargs)

            expanded_infos = {}
            for k, v in info_dict.items():
                v_batch = v[start_idx:end_idx]
                if torch.is_tensor(v):
                    target_dtype = (
                        self.dtype if v_batch.is_floating_point() else None
                    )
                    v_batch = (
                        v_batch.to(device=self.device, dtype=target_dtype)
                        .unsqueeze(1)
                        .expand(
                            current_bs,
                            self.num_samples,
                            *v_batch.shape[1:],
                        )
                    )
                elif isinstance(v, np.ndarray):
                    v_batch = np.repeat(
                        v_batch[:, None, ...], self.num_samples, axis=1
                    )
                expanded_infos[k] = v_batch

            # Perform Gradient Descent for this batch
            batch_cost_history = []

            for cb in self.callbacks:
                cb.start_batch()

            for step in range(self.n_steps):
                costs = self.model.get_cost(expanded_infos, batch_init)

                assert isinstance(costs, torch.Tensor), (
                    f'Got {type(costs)} cost, expect torch.Tensor'
                )
                assert (
                    costs.ndim == 2
                    and costs.shape[0] == current_bs
                    and costs.shape[1] == self.num_samples
                ), (
                    f'Cost should be of shape ({current_bs}, {self.num_samples}), got {costs.shape}'
                )
                assert costs.requires_grad, (
                    'Cost must requires_grad for GD solver.'
                )

                cost = costs.sum()  # Sum cost for this batch
                cost.backward()

                for cb in self.callbacks:
                    cb(
                        step=step,
                        params=batch_init,
                        cost=cost,
                        costs=costs,
                    )

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(batch_init, self.grad_clip)

                optim.step()
                optim.zero_grad(set_to_none=True)

                # Add noise
                if self.action_noise > 0:
                    batch_init.data += (
                        torch.randn(batch_init.shape, generator=self.torch_gen)
                        * self.action_noise
                    )

                batch_cost_history.append(cost.item())

            # Store cost history for this batch
            outputs['cost'].append(batch_cost_history)

            # Update the global self.init with the optimized batch values
            with torch.no_grad():
                self.init[start_idx:end_idx] = batch_init

            top_idx = torch.argsort(costs, dim=1)[:, 0]
            batch_indices = torch.arange(current_bs)

            top_actions_batch = batch_init[batch_indices, top_idx]
            batch_top_actions_list.append(top_actions_batch.detach().cpu())

        # Concatenate all batch results
        outputs['actions'] = torch.cat(batch_top_actions_list, dim=0)

        if self.callbacks:
            outputs['callbacks'] = {}
            for cb in self.callbacks:
                cb.end_solve()
                outputs['callbacks'][cb.output_key] = cb.history

        end_time = time.time()
        print(
            f'GradientSolver.solve completed in {end_time - start_time:.4f} seconds.'
        )

        return outputs
