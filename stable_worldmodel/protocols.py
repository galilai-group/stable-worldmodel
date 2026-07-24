from typing import Protocol, runtime_checkable
import numpy as np
import torch


@runtime_checkable
class Costable(Protocol):
    """Protocol for the cost surface planning solvers consume.

    This is the structural "has ``get_cost``" contract every solver types
    against. It is polymorphic across implementations: a
    :class:`~stable_worldmodel.planning.ShootingCostEvaluator` (a world model composed
    with an :class:`Objective`), as well as models that expose ``get_cost``
    natively (e.g. TD-MPC2, prejepa), all satisfy it.
    """

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


@runtime_checkable
class Constrainable(Protocol):
    """Protocol for the (optional) constraint surface of a cost object.

    A cost object exposes ``get_constraints`` when planning under inequality
    constraints. ``LagrangianSolver`` feature-detects it via
    ``isinstance(cost, Constrainable)``. Following the Lagrangian contract, a
    constraint term ``g_i`` is satisfied when ``g_i <= 0``.
    """

    def get_constraints(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover
        """Compute constraint violations for given action candidates.

        Args:
            info_dict: Dictionary containing environment state information.
            action_candidates: Tensor of proposed actions.

        Returns:
            A tensor of shape ``(B, S, C)`` of per-candidate constraint values,
            where ``C`` is the number of constraints (satisfied when ``<= 0``).
        """
        ...


@runtime_checkable
class Dynamics(Protocol):
    """The dynamics surface a ``ShootingCostEvaluator`` needs from a world model."""

    def encode(self, x: dict) -> dict:  # pragma: no cover
        """Embed raw observations into the model's latent space.

        Args:
            x: Dictionary of observations (e.g. ``pixels``, proprioception).

        Returns:
            The dictionary augmented with latent embeddings (e.g. ``emb``).
        """
        ...

    def rollout(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> dict:  # pragma: no cover
        """Roll candidate action sequences forward through the dynamics.

        Args:
            info_dict: Dictionary containing environment state information.
                ``pixels`` holds ``H`` context frames ``(B, S, H, C, h, w)``;
                when ``H > 1`` the executed action blocks between those
                frames must be provided as ``action_history`` of shape
                ``(B, S, H - 1, action_dim)``.
            action_candidates: Tensor of proposed strictly-future action
                sequences of shape ``(B, S, horizon, action_dim)``.

        Returns:
            The dictionary populated with rollout outputs (e.g.
            ``predicted_emb`` of shape ``(B, S, H + horizon, dim)``, whose
            first ``H`` entries are the encoded context frames).
        """
        ...


@runtime_checkable
class Objective(Protocol):
    """Maps a populated ``info_dict`` to per-candidate cost ``(B, S)``.

    Unlike :class:`Costable`, an ``Objective`` scores an *already-rolled-out*
    ``info_dict`` — it takes no ``action_candidates`` and performs no rollout.
    The ``info_dict`` is expected to already contain rollout outputs (e.g.
    ``predicted_emb``) and any goal/conditioning the objective needs. The
    ``ShootingCostEvaluator`` also stores the raw ``action_candidates`` under the
    ``action_candidates`` key so action-space penalties can read them.
    """

    def __call__(self, info_dict: dict) -> torch.Tensor:  # pragma: no cover
        """Score a rolled-out ``info_dict``.

        Args:
            info_dict: Dictionary already populated with rollout outputs
                (e.g. ``predicted_emb``) and any goal/conditioning keys.

        Returns:
            Per-candidate cost tensor of shape ``(B, S)``.
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
