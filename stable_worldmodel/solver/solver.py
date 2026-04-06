from typing import Any, Protocol, runtime_checkable

import gymnasium as gym
import torch


class Costable(Protocol):
    """Protocol for world model cost functions."""

    def criterion(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Compute the cost criterion for action candidates.

        Args:
            info_dict: Dictionary containing environment state information.
            action_candidates: Tensor of proposed actions.

        Returns:
            A tensor of cost values for each action candidate.
        """
        ...

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """Compute cost for given action candidates based on info dictionary.

        Args:
            info_dict: Dictionary containing environment state information.
            action_candidates: Tensor of proposed actions.

        Returns:
            A tensor of cost values for each action candidate.
        """
        ...


@runtime_checkable
class Solver(Protocol):
    """Protocol for model-based planning solvers."""

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        """Configure the solver with environment and planning specifications.

        Args:
            action_space: The action space of the environment.
            n_envs: Number of parallel environments.
            config: Planning configuration object.
        """
        ...

    @property
    def action_dim(self) -> int:
        """Flattened action dimension including action_block grouping."""
        ...

    @property
    def n_envs(self) -> int:
        """Number of parallel environments being planned for."""
        ...

    @property
    def horizon(self) -> int:
        """Planning horizon length in timesteps."""
        ...

    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        """Solve the planning optimization problem to find optimal actions.

        Args:
            info_dict: Dictionary containing environment state information.
            init_action: Optional initial action sequence to warm-start the solver.

        Returns:
            Dictionary containing optimized actions and other solver-specific info.
        """
        ...
        
@runtime_checkable
class Rollable(Protocol):
    """Protocol for world models that expose a differentiable rollout method.

    Required by :class:`~stable_worldmodel.solver.grasp.GRASPSolver` to perform
    single-step latent-space predictions during the joint state–action
    optimisation loop.

    Implementations must accept an ``info_dict`` whose ``'emb'`` key holds the
    current latent state and write the predicted next state(s) into
    ``info_dict['predicted_emb']``.
    """

    def rollout(self, info_dict: dict, action_sequence: torch.Tensor) -> dict:
        """Roll the world model forward for one or more steps from a latent state.

        Args:
            info_dict: Must contain ``'emb': (B, D)`` – the current latent
                embedding.  Any additional keys are forwarded unchanged.
            action_sequence: ``(B, S, T, action_dim)`` action candidates where
                *B* is batch size, *S* is the number of parallel samples, and
                *T* is the number of timesteps to roll out.

        Returns:
            The modified ``info_dict`` with ``'predicted_emb': (B, S, T, D)``
            written in place.
        """
        ...

