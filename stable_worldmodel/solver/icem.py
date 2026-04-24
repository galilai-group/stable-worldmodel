"""Improved Cross Entropy Method (iCEM) solver for model-based planning."""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging

from .solver import Costable


class ICEMSolver:
    """Improved Cross Entropy Method (iCEM) solver with colored noise and elite retention.
    iCEM improves the sample efficiency over standard CEM and was introduced by
    [1] for real-time planning.

    Args:
        model: World model implementing the Costable protocol.
        batch_size: Number of environments to process in parallel.
        num_samples: Number of action candidates to sample per iteration.
        var_scale: Initial variance scale for the action distribution.
        n_steps: Number of CEM iterations.
        topk: Number of elite samples to keep for distribution update.
        noise_beta: Colored noise exponent. 0 = white (standard CEM), >0 = more low-frequency noise.
        alpha: Momentum for mean/std EMA update.
        n_elite_keep: Number of elites carried from previous iteration.
        return_mean: If False, return best single trajectory instead of mean.
        device: Device for tensor computations.
        seed: Random seed for reproducibility.
        
    [1] C. Pinneri, S. Sawant, S. Blaes, J. Achterhold, J. Stueckler, M. Rolinek and
    G, Martius, Georg. "Sample-efficient Cross-Entropy Method for Real-time Planning".
    Conference on Robot Learning, 2020.
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
        device: str | torch.device = "cpu",
        seed: int = 1234,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.var_scale = var_scale
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.noise_beta = noise_beta
        self.alpha = alpha
        self.n_elite_keep = n_elite_keep
        self.return_mean = return_mean
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        """Configure the solver with environment specifications."""
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

        if isinstance(action_space, Box):
            self._action_low = torch.tensor(action_space.low[0], device=self.device, dtype=torch.float32)
            self._action_high = torch.tensor(action_space.high[0], device=self.device, dtype=torch.float32)
        else:
            logging.warning(f"Action space is discrete, got {type(action_space)}. ICEMSolver may not work as expected.")
            self._action_low = None
            self._action_high = None

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

    def init_action_distrib(
        self, n_envs: int, actions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize the action distribution parameters (mean and variance)."""
        var = self.var_scale * torch.ones([n_envs, self.horizon, self.action_dim])
        mean = torch.zeros([n_envs, 0, self.action_dim]) if actions is None else actions

        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            device = mean.device
            new_mean = torch.zeros([n_envs, remaining, self.action_dim])
            mean = torch.cat([mean, new_mean], dim=1).to(device)

        return mean, var

    @torch.inference_mode()
    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        """Solve the planning problem using improved Cross Entropy Method."""
        start_time = time.time()
        outputs = {
            "costs": [],
            "mean": [],
            "var": [],
        }

        # Batch size is taken from info_dict so callers can solve for a subset of envs
        total_envs = len(next(iter(info_dict.values())))

        mean, var = self.init_action_distrib(total_envs, init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx

            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]

            expanded_infos = {}
            for k, v in info_dict.items():
                v_batch = v[start_idx:end_idx]
                if torch.is_tensor(v):
                    v_batch = v_batch.unsqueeze(1)
                    v_batch = v_batch.expand(current_bs, self.num_samples, *v_batch.shape[2:])
                elif isinstance(v, np.ndarray):
                    v_batch = np.repeat(v_batch[:, None, ...], self.num_samples, axis=1)
                expanded_infos[k] = v_batch

            prev_topk_candidates = None
            batch_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, self.topk)

            # Precompute FFT scale for colored noise
            noise_shape = (current_bs, self.num_samples, self.action_dim, self.horizon)
            freqs = torch.fft.rfftfreq(self.horizon, device=self.device)
            freqs[0] = 1.0
            noise_scale = freqs.pow(-self.noise_beta / 2)
            noise_scale[0] = noise_scale[1]

            for step in range(self.n_steps):
                # Colored noise: generate with temporal axis last, then transpose
                if self.horizon <= 1:
                    noise = torch.randn(noise_shape, generator=self.torch_gen, device=self.device)
                else:
                    white = torch.randn(noise_shape, generator=self.torch_gen, device=self.device)
                    fft = torch.fft.rfft(white, dim=-1)
                    colored = torch.fft.irfft(fft * noise_scale, n=self.horizon, dim=-1)
                    std = colored.std(dim=-1, keepdim=True).clamp(min=1e-8)
                    noise = colored / std
                noise = noise.transpose(-1, -2)  # -> (bs, num_samples, horizon, action_dim)

                candidates = noise * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                candidates[:, 0] = batch_mean

                # Inject previous elites
                if prev_topk_candidates is not None:
                    n_inject = min(self.n_elite_keep, prev_topk_candidates.shape[1])
                    candidates[:, 1:1 + n_inject] = prev_topk_candidates[:, :n_inject]

                # Clip to action bounds
                if self._action_low is not None:
                    candidates = candidates.clamp(self._action_low, self._action_high)

                current_info = expanded_infos.copy()
                costs = self.model.get_cost(current_info, candidates)

                assert isinstance(costs, torch.Tensor), f"Expected cost to be a torch.Tensor, got {type(costs)}"
                assert costs.ndim == 2 and costs.shape[0] == current_bs and costs.shape[1] == self.num_samples, (
                    f"Expected cost to be of shape ({current_bs}, {self.num_samples}), got {costs.shape}"
                )

                topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
                topk_candidates = candidates[batch_indices, topk_inds]

                prev_topk_candidates = topk_candidates

                # Momentum update
                elite_mean = topk_candidates.mean(dim=1)
                elite_var = topk_candidates.std(dim=1)
                batch_mean = self.alpha * batch_mean + (1 - self.alpha) * elite_mean
                batch_var = self.alpha * batch_var + (1 - self.alpha) * elite_var

            final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()

            if self.return_mean:
                mean[start_idx:end_idx] = batch_mean
            else:
                mean[start_idx:end_idx] = topk_candidates[:, 0]

            var[start_idx:end_idx] = batch_var

            outputs["costs"].extend(final_batch_cost)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]

        print(f"iCEM solve time: {time.time() - start_time:.4f} seconds")
        return outputs
