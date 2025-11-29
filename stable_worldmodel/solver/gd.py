import time

import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging

from .solver import Costable


class GDSolver(torch.nn.Module):
    """Gradient Descent Solver."""

    def __init__(
        self,
        model: Costable,
        n_steps: int,
        batch_size: int | None = None,
        var_scale: float = 1,
        num_samples: int = 1,
        action_noise: float = 0.0,
        device="cpu",
        seed: int = 1234,
    ):
        """Gradient Descent Method Solver.
        Args:
            model (Costable): The world model used to compute costs.
            n_steps (int): Number of gradient descent steps.
            batch_size (int | None): Batch size for processing environments. If None, process all envs at once.
            var_scale (float): Scale of the initial action variance in the samples.
            num_samples (int): Number of initial action samples to optimize.
            action_noise (float): Standard deviation of noise added to actions during optimization.
            device (str): Device to run the solver on.
            seed (int): Random seed for reproducibility.
        """
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.action_noise = action_noise
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)

        self._configured = False
        self._n_envs = None
        self._action_dim = None
        self._config = None

    def configure(self, *, action_space, n_envs: int, config) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

        # warning if action space is discrete
        if not isinstance(action_space, Box):
            logging.warning(f"Action space is discrete, got {type(action_space)}. GDSolver may not work as expected.")

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.solve(*args, **kwargs)

    def init_action(self, actions=None):
        """Initialize the action tensor for the solver.

        set self.init - initial action sequences (n_envs, horizon, action_dim)
        """
        if actions is None:
            actions = torch.zeros((self._n_envs, 0, self.action_dim))

        # fill remaining action
        remaining = self.horizon - actions.shape[1]

        if remaining > 0:
            new_actions = torch.zeros(self._n_envs, remaining, self.action_dim)
            actions = torch.cat([actions, new_actions], dim=1).to(self.device)

        actions = actions.unsqueeze(1).repeat_interleave(self.num_samples, dim=1)  # add sample dim
        actions[:, 1:] += (
            torch.randn(actions[:, 1:].shape, generator=self.torch_gen, device=self.device) * self.var_scale
        )  # add small noise to all samples except the first one

        # reset actions
        if hasattr(self, "init"):
            self.init.copy_(actions)
        else:
            self.register_parameter("init", torch.nn.Parameter(actions))

    def solve(self, info_dict, init_action=None) -> dict:
        """Solve the planning optimization problem using gradient descent with batch processing."""
        start_time = time.time()
        outputs = {
            "cost": [],  # Will store list of cost histories per batch
            "actions": None,
        }

        with torch.no_grad():
            self.init_action(init_action)

        # Determine batch size (default to all envs if not specified which can cause memory issues)
        batch_size = self.batch_size if self.batch_size is not None else self.n_envs
        total_envs = self.n_envs

        # Lists to hold results from each batch to be concatenated later
        batch_top_actions_list = []

        # --- Outer Loop: Iterate over batches ---
        for start_idx in range(0, total_envs, batch_size):
            end_idx = min(start_idx + batch_size, total_envs)
            current_bs = end_idx - start_idx

            batch_init = self.init[start_idx:end_idx].clone().detach()
            batch_init.requires_grad = True

            optim = torch.optim.SGD([batch_init], lr=1.0)

            # Prepare Batch Infos
            # Slice the input info_dict and then expand dimensions
            expanded_infos = {}
            for k, v in info_dict.items():
                # Slice the data for the current batch indices
                # Assumes input data dim 0 corresponds to n_envs
                if torch.is_tensor(v):
                    batch_v = v[start_idx:end_idx]
                    batch_v = batch_v.unsqueeze(1)
                    batch_v = batch_v.expand(current_bs, self.num_samples, *batch_v.shape[2:])
                elif isinstance(v, np.ndarray):
                    batch_v = v[start_idx:end_idx]
                    batch_v = np.repeat(batch_v[:, None, ...], self.num_samples, axis=1)
                expanded_infos[k] = batch_v

            # Perform Gradient Descent for this batch
            batch_cost_history = []

            for step in range(self.n_steps):
                current_info = expanded_infos.copy()

                # Calculate cost using the batch parameter
                costs = self.model.get_cost(current_info, batch_init)

                assert isinstance(costs, torch.Tensor), f"Got {type(costs)} cost, expect torch.Tensor"
                assert costs.ndim == 2 and costs.shape[0] == current_bs and costs.shape[1] == self.num_samples, (
                    f"Cost should be of shape ({current_bs}, {self.num_samples}), got {costs.shape}"
                )
                assert costs.requires_grad, "Cost must requires_grad for GD solver."

                cost = costs.sum()  # Sum cost for this batch
                cost.backward()
                optim.step()
                optim.zero_grad(set_to_none=True)

                # Add noise
                if self.action_noise > 0:
                    batch_init.data += torch.randn(batch_init.shape, generator=self.torch_gen) * self.action_noise

                batch_cost_history.append(cost.item())

            # Store cost history for this batch
            outputs["cost"].append(batch_cost_history)

            # Update the global self.init with the optimized batch values
            with torch.no_grad():
                self.init[start_idx:end_idx] = batch_init

            top_idx = torch.argsort(costs, dim=1)[:, 0]
            batch_indices = torch.arange(current_bs)

            top_actions_batch = batch_init[batch_indices, top_idx]
            batch_top_actions_list.append(top_actions_batch.detach().cpu())

        # Concatenate all batch results
        outputs["actions"] = torch.cat(batch_top_actions_list, dim=0)
        end_time = time.time()
        print(f"GDSolver.solve completed in {end_time - start_time:.4f} seconds.")

        return outputs
