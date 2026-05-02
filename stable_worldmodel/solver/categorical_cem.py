"""Categorical Cross Entropy Method solver for discrete action spaces."""

from typing import Any

import gymnasium as gym
import torch
from gymnasium.spaces import Discrete

from .base import BaseSolver
from .callbacks import Callback
from .solver import Costable


class CategoricalCEMSolver(BaseSolver):
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
        super().__init__(
            model=model,
            n_steps=n_steps,
            batch_size=batch_size,
            num_samples=num_samples,
            device=device,
            callbacks=callbacks,
        )
        self.topk = topk
        self.smoothing = smoothing
        self.alpha = alpha
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)

    def configure(
        self, *, action_space: gym.Space, n_envs: int, config: Any
    ) -> None:
        assert isinstance(action_space, Discrete), (
            f'Action space must be Discrete, got {type(action_space)}'
        )
        super().configure(
            action_space=action_space, n_envs=n_envs, config=config
        )
        self._base_simplex_dim = int(action_space.n)

    @property
    def base_simplex_dim(self) -> int:
        """Number of categories per action position."""
        return self._base_simplex_dim

    @property
    def action_simplex_dim(self) -> int:
        """Flattened simplex dim including action_block grouping."""
        return self._base_simplex_dim * self.action_block

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

    # === BaseSolver hooks ===

    def init_state(
        self, n_envs: int, init: Any | None = None
    ) -> dict[str, torch.Tensor]:
        # init_action accepted for parity but ignored — probs always uniform.
        del init
        return {'probs': self.init_probs(n_envs)}

    def propose(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        probs = state['probs']
        bs = probs.shape[0]
        indices = self._sample_indices(probs)
        # Force first sample to argmax of current probs (analog of CEM mean).
        indices[:, 0] = probs.argmax(dim=-1)
        one_hot = torch.nn.functional.one_hot(
            indices, num_classes=self._base_simplex_dim
        ).to(self.dtype)
        return one_hot.reshape(
            bs, self.num_samples, self.horizon, self.action_simplex_dim
        )

    def update(
        self,
        state: dict[str, torch.Tensor],
        candidates: torch.Tensor,
        costs: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        topk_vals, topk_inds = torch.topk(
            costs, k=self.topk, dim=1, largest=False
        )
        bs = candidates.shape[0]
        batch_indices = (
            torch.arange(bs, device=self.device)
            .unsqueeze(1)
            .expand(-1, self.topk)
        )
        # (B, N, H, action_block, K) one-hot view of the flat candidates.
        one_hot = candidates.reshape(
            bs,
            self.num_samples,
            self.horizon,
            self.action_block,
            self._base_simplex_dim,
        )
        topk_one_hot = one_hot[batch_indices, topk_inds]

        new_probs = topk_one_hot.mean(dim=1)
        if self.smoothing > 0:
            new_probs = new_probs + self.smoothing
            new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)

        prev_probs = state['probs']
        if self.alpha > 0:
            new_probs = self.alpha * prev_probs + (1 - self.alpha) * new_probs

        payload = {
            'topk_vals': topk_vals,
            'topk_inds': topk_inds,
            'topk_candidates': topk_one_hot,
            'probs': new_probs,
            'prev_probs': prev_probs,
        }
        return {'probs': new_probs}, payload

    def finalize(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        # (n_envs, horizon, action_block) — argmax of final probs.
        return state['probs'].argmax(dim=-1).detach().cpu()

    def extra_outputs(self, state: dict[str, torch.Tensor]) -> dict[str, Any]:
        return {'probs': [state['probs'].detach().cpu()]}
