"""Categorical Cross Entropy Method solver for discrete action spaces."""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete

from .callbacks import Callback
from .solver import Costable


class CategoricalCEMSolver:
    """Cross Entropy Method solver for discrete action optimization.

    Maintains a per-timestep categorical distribution over discrete actions,
    samples candidate trajectories via Gumbel-max, and refits the distribution
    from the top-K elites' empirical frequencies.

    Args:
        model: World model implementing the Costable protocol.
        batch_size: Number of environments to process in parallel.
        num_samples: Number of action candidates to sample per iteration.
        n_steps: Number of CEM iterations.
        topk: Number of elite samples to keep for distribution update.
        smoothing: Laplace smoothing added to refit probs to avoid collapse.
        alpha: Momentum for probs EMA update (0 = full overwrite).
        device: Device for tensor computations.
        seed: Random seed for reproducibility.
        callbacks: Optional list of callbacks.
    """

    def __init__(
        self,
        model: Costable,
        batch_size: int = 1,
        num_samples: int = 300,
        n_steps: int = 30,
        topk: int = 30,
        smoothing: float = 0.0,
        alpha: float = 0.0,
        device: str | torch.device = 'cpu',
        seed: int = 1234,
        callbacks: list[Callback] | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.smoothing = smoothing
        self.alpha = alpha
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)
        self.callbacks = list(callbacks) if callbacks else []
        try:
            self._dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            self._dtype = torch.float32

    def configure(
        self, *, action_space: gym.Space, n_envs: int, config: Any
    ) -> None:
        """Configure the solver with environment specifications."""
        assert isinstance(action_space, Discrete), (
            f'Action space must be Discrete, got {type(action_space)}'
        )
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._base_simplex_dim = int(action_space.n)
        self._configured = True

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_block(self) -> int:
        return self._config.action_block

    @property
    def base_simplex_dim(self) -> int:
        """Number of categories per action position."""
        return self._base_simplex_dim

    @property
    def action_simplex_dim(self) -> int:
        """Flattened simplex dim including action_block grouping."""
        return self._base_simplex_dim * self.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def init_probs(self, n_envs: int) -> torch.Tensor:
        """Initialize uniform categorical probabilities.

        Shape: (n_envs, horizon, action_block, base_simplex_dim).
        """
        K = self._base_simplex_dim
        return torch.full(
            (n_envs, self.horizon, self.action_block, K),
            1.0 / K,
            dtype=self.dtype,
            device=self.device,
        )

    def _sample_indices(self, probs: torch.Tensor) -> torch.Tensor:
        """Gumbel-max sample of categorical indices.

        Args:
            probs: shape (B, H, action_block, K).

        Returns:
            indices: shape (B, num_samples, H, action_block).
        """
        bs, H, ab, K = probs.shape
        log_probs = probs.clamp_min(1e-10).log()
        log_probs = log_probs.unsqueeze(1).expand(
            bs, self.num_samples, H, ab, K
        )
        u = torch.rand(
            log_probs.shape,
            generator=self.torch_gen,
            device=self.device,
            dtype=self.dtype,
        ).clamp_min(1e-10)
        gumbel = -(-u.log()).log()
        return (log_probs + gumbel).argmax(dim=-1)

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: Any = None) -> dict:
        """Solve the planning problem using Categorical CEM.

        ``init_action`` is accepted for API parity with other solvers but is
        ignored; probs are always initialized uniform.
        """
        del init_action
        start_time = time.time()
        outputs: dict = {'costs': [], 'probs': []}

        total_envs = len(next(iter(info_dict.values())))
        probs = self.init_probs(total_envs)

        for cb in self.callbacks:
            cb.reset()

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_probs = probs[start_idx:end_idx]

            expanded_infos: dict = {}
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

            for cb in self.callbacks:
                cb.start_batch()

            final_batch_cost = None
            for step in range(self.n_steps):
                # Sample indices: (B, N, H, action_block)
                indices = self._sample_indices(batch_probs)

                # Force first sample to argmax of current probs (analog of CEM's "current mean")
                indices[:, 0] = batch_probs.argmax(dim=-1)

                # One-hot for cost: (B, N, H, action_block, K) -> flatten last two
                one_hot = torch.nn.functional.one_hot(
                    indices, num_classes=self._base_simplex_dim
                ).to(self.dtype)
                candidates = one_hot.reshape(
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_simplex_dim,
                )

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

                topk_vals, topk_inds = torch.topk(
                    costs, k=self.topk, dim=1, largest=False
                )

                batch_indices = (
                    torch.arange(current_bs, device=self.device)
                    .unsqueeze(1)
                    .expand(-1, self.topk)
                )
                # (B, K_topk, H, action_block, K_simplex)
                topk_one_hot = one_hot[batch_indices, topk_inds]

                # Refit: empirical frequencies over the elite set
                new_probs = topk_one_hot.mean(dim=1)

                if self.smoothing > 0:
                    new_probs = new_probs + self.smoothing
                    new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)

                prev_probs = batch_probs
                if self.alpha > 0:
                    batch_probs = (
                        self.alpha * batch_probs + (1 - self.alpha) * new_probs
                    )
                else:
                    batch_probs = new_probs

                for cb in self.callbacks:
                    cb(
                        step=step,
                        candidates=candidates,
                        costs=costs,
                        topk_vals=topk_vals,
                        topk_inds=topk_inds,
                        topk_candidates=topk_one_hot,
                        probs=batch_probs,
                        prev_probs=prev_probs,
                    )

                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()

            probs[start_idx:end_idx] = batch_probs
            outputs['costs'].extend(final_batch_cost)

        # Output discrete actions: argmax of final probs.
        # Shape (n_envs, horizon, action_block) — matches PGDSolver convention.
        actions = probs.argmax(dim=-1)

        outputs['actions'] = actions.detach().cpu()
        outputs['probs'] = [probs.detach().cpu()]

        if self.callbacks:
            outputs['callbacks'] = {}
            for cb in self.callbacks:
                cb.end_solve()
                outputs['callbacks'][cb.output_key] = cb.history

        print(
            f'Categorical CEM solve time: {time.time() - start_time:.4f} seconds'
        )
        return outputs
