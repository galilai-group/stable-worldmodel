import time

import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging

from .solver import Costable


class MPPISolver:
    """Model Predictive Path Integral Solver.

    proposed in https://arxiv.org/abs/1509.01149
    algorithm from: https://acdslab.github.io/mppi-generic-website/docs/mppi.html

    Note:
        The original MPPI compute the cost as a summation of costs along the trajectory.
        Here, we use the final cost only, which should be updated in future updates.
    """

    def __init__(
        self,
        model: Costable,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1.0,
        n_steps: int = 30,
        topk: int = 30,
        temperature: float = 0.5,
        device="cpu",
        seed: int = 1234,
    ):
        self.model = model
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.topk = topk
        self.var_scale = var_scale
        self.n_steps = n_steps
        self.temperature = temperature
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)

    def configure(self, *, action_space, n_envs: int, config) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

        # warning if action space is discrete
        if not isinstance(action_space, Box):
            logging.warning(
                f"Action space is discrete, got {type(action_space)}. MPPISolver may not work as expected."
            )

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

    def init_action_distrib(self, actions=None):
        """Initialize the action distribution params (mu, sigma) given the initial condition.

        Args:
            actions (n_envs, T, action_dim): initial actions, T <= horizon
        """
        var = self.var_scale * torch.ones([self.n_envs, self.horizon, self.action_dim])
        mean = torch.zeros([self.n_envs, 0, self.action_dim]) if actions is None else actions

        # -- fill remaining actions with random sample
        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            device = mean.device
            new_mean = torch.zeros([self.n_envs, remaining, self.action_dim])
            mean = torch.cat([mean, new_mean], dim=1).to(device)

        return mean, var

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None) -> dict:
        start_time = time.time()
        outputs = {
            "costs": [],
            "mean": [],
            "var": [],
        }

        # -- initialize the action distribution globally
        mean, var = self.init_action_distrib(init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)

        total_envs = self.n_envs

        # --- Iterate over batches ---
        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx

            # Slice Distribution Parameters for current batch
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]

            # Expand Info Dict for current batch (Same as CEM)
            expanded_infos = {}
            for k, v in info_dict.items():
                v_batch = v[start_idx:end_idx]
                if torch.is_tensor(v):
                    # Add sample dim: (batch, 1, ...)
                    v_batch = v_batch.unsqueeze(1)
                    # Expand: (batch, num_samples, ...)
                    v_batch = v_batch.expand(current_bs, self.num_samples, *v_batch.shape[2:])
                elif isinstance(v, np.ndarray):
                    v_batch = np.repeat(v_batch[:, None, ...], self.num_samples, axis=1)
                expanded_infos[k] = v_batch

            # Optimization Loop
            final_batch_cost = None

            for step in range(self.n_steps):
                # Sample noise: (Batch, Num_Samples, Horizon, Dim)
                noise = torch.randn(
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                )

                # MPPI Logic: candidates = mean + noise * sigma
                candidates = batch_mean.unsqueeze(1) + noise * batch_var.unsqueeze(1)

                # Force the first sample to be the current mean (Zero noise)
                candidates[:, 0] = batch_mean

                # Evaluate candidates
                costs = self.model.get_cost(expanded_infos, candidates)

                assert isinstance(costs, torch.Tensor), f"Expected cost to be a torch.Tensor, got {type(costs)}"
                assert costs.ndim == 2 and costs.shape[0] == current_bs and costs.shape[1] == self.num_samples, (
                    f"Expected cost to be of shape ({current_bs}, {self.num_samples}), got {costs.shape}"
                )

                # Select Elites (Optional, based on topk)
                if self.topk is not None and self.topk < self.num_samples:
                    # topk_vals: (Batch, K), topk_inds: (Batch, K)
                    topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)

                    # Gather Top-K Candidates
                    batch_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, self.topk)
                    # (Batch, K, Horizon, Dim)
                    relevant_candidates = candidates[batch_indices, topk_inds]
                    relevant_costs = topk_vals
                else:
                    relevant_candidates = candidates
                    relevant_costs = costs

                # MPPI Weighting: Softmax(-cost / temperature)
                # Stabilize softmax by subtracting min cost
                min_cost = relevant_costs.min(dim=1, keepdim=True)[0]
                scaled_costs = relevant_costs - min_cost
                weights = torch.softmax(-scaled_costs / self.temperature, dim=1)  # (Batch, K)

                # Update Mean: weighted sum of candidates
                # Reshape weights for broadcasting: (Batch, K, 1, 1)
                weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)
                batch_mean = (weights_expanded * relevant_candidates).sum(dim=1)

                # Store average cost of the utilized samples for logging
                final_batch_cost = relevant_costs.mean(dim=1).cpu().tolist()

            # Write results back to global storage
            mean[start_idx:end_idx] = batch_mean
            # We do not update var in standard MPPI

            # Store history/metadata
            outputs["costs"].extend(final_batch_cost)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]

        print(f"MPPI solve time: {time.time() - start_time:.4f} seconds")
        return outputs
