from typing import Protocol, runtime_checkable
import numpy as np
import torch


class Costable(Protocol):
    """Protocol for world model cost functions."""

    def criterion(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Compute the cost criterion for action candidates.

        Args:
            info_dict: Dictionary containing environment state information.
            action_candidates: Tensor of proposed actions.

        Returns:
            A tensor of cost values for each action candidate.
        """
        ...

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover
        """Compute cost for given action candidates based on info dictionary.

        Args:
            info_dict: Dictionary containing environment state information.
            action_candidates: Tensor of proposed actions.

        Returns:
            A tensor of cost values for each action candidate.
        """
        ...


class Transformable(Protocol):
    """Protocol for reversible data transformations (e.g., normalizers, scalers)."""

    def transform(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Apply preprocessing to input data.

        Args:
            x: Input data as a numpy array.

        Returns:
            Preprocessed data as a numpy array.
        """
        ...

    def inverse_transform(
        self, x: np.ndarray
    ) -> np.ndarray:  # pragma: no cover
        """Reverse the preprocessing transformation.

        Args:
            x: Preprocessed data as a numpy array.

        Returns:
            Original data as a numpy array.
        """
        ...


@runtime_checkable
class Actionable(Protocol):
    """Protocol for model action computation."""

    def get_action(
        self,
        info: dict,
        horizon: int = 1,
        prefix_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:  # pragma: no cover
        """Compute action(s) from observation and goal.

        Args:
            info: Dictionary containing environment state information.
            horizon: Number of actions to return. When 1 (default), returns a
                single action of shape (..., action_dim). When > 1, returns an
                action sequence of shape (..., horizon, action_dim).
            prefix_actions: Optional warm-start actions of shape
                ``(..., t, action_dim)`` with ``t < horizon`` that are applied
                first to advance the latent state before the actor is rolled
                out for ``horizon`` steps.

        Returns:
            A tensor of actions with shape (..., action_dim) if horizon == 1,
            or (..., horizon, action_dim) if horizon > 1.
        """
        ...
