"""Predictive Sampling solver for model-based planning.

Reference: Howell et al., "Predictive Sampling: Real-time Behaviour Synthesis
with MuJoCo", 2022.
"""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging

from .solver import Costable


class PredictiveSamplingSolver:
    """Predictive Sampling solver for action optimization.

    Args:
        model: World model implementing the Costable protocol.
        batch_size: Number of environments to process in parallel.
        num_samples: Number of action candidates to sample.
        noise_scale: Standard deviation of additive Gaussian noise.
        device: Device for tensor computations.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        model: Costable,
        batch_size: int = 1,
        num_samples: int = 300,
        noise_scale: float = 1.0,
        device: str | torch.device = 'cpu',
        seed: int = 1234,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.noise_scale = noise_scale
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)
        try:
            self._dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            self._dtype = torch.float32

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
                f'Action space is discrete, got {type(action_space)}. PredictiveSamplingSolver may not work as expected.'
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

    def init_nominal(
        self, n_envs: int, actions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Initialize the nominal action sequence."""
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

        return nominal

    @torch.inference_mode()
    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        """Solve the planning problem using Predictive Sampling."""
        start_time = time.time()
        outputs: dict[str, Any] = {'costs': []}

        total_envs = len(next(iter(info_dict.values())))

        nominal = self.init_nominal(total_envs, init_action).to(self.device)

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx

            batch_nominal = nominal[start_idx:end_idx]

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
                            current_bs, self.num_samples, *v_batch.shape[1:]
                        )
                    )
                elif isinstance(v, np.ndarray):
                    v_batch = np.repeat(
                        v_batch[:, None, ...], self.num_samples, axis=1
                    )
                expanded_infos[k] = v_batch

            # Sample noise: (Batch, Num_Samples, Horizon, Dim)
            noise = torch.randn(
                current_bs,
                self.num_samples,
                self.horizon,
                self.action_dim,
                generator=self.torch_gen,
                device=self.device,
                dtype=self.dtype,
            )

            # Candidates = nominal + noise * sigma
            candidates = batch_nominal.unsqueeze(1) + noise * self.noise_scale

            # Force the first sample to be the nominal (zero noise) so the
            # result is never worse than the warm-start.
            candidates[:, 0] = batch_nominal

            costs = self.model.get_cost(expanded_infos, candidates)

            assert isinstance(costs, torch.Tensor), (
                f'Expected cost to be a torch.Tensor, got {type(costs)}'
            )
            assert (
                costs.ndim == 2
                and costs.shape[0] == current_bs
                and costs.shape[1] == self.num_samples
            ), (
                f'Expected cost to be of shape ({current_bs}, {self.num_samples}), got {costs.shape}'
            )

            # Pick argmin per env
            best_idx = costs.argmin(dim=1)
            batch_indices = torch.arange(current_bs, device=self.device)
            best_candidates = candidates[batch_indices, best_idx]
            best_costs = costs[batch_indices, best_idx]

            nominal[start_idx:end_idx] = best_candidates
            outputs['costs'].extend(best_costs.cpu().tolist())

        outputs['actions'] = nominal.detach().cpu()

        print(
            f'Predictive Sampling solve time: {time.time() - start_time:.4f} seconds'
        )
        return outputs
