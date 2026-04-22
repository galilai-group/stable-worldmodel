"""Shared utilities for planning solvers."""

import torch

from stable_worldmodel.protocols import Actionable


def prepare_init_action(
    model,
    info_dict: dict,
    init_action: torch.Tensor | None,
    horizon: int,
    n_envs: int,
    action_dim: int,
) -> torch.Tensor:
    """Extend or generate an initial action sequence to cover the full horizon.

    When the model implements the Actionable protocol, any missing planning
    steps are filled by calling
    ``model.get_action(info_dict, horizon=remaining, prefix_actions=init_action)``,
    so the actor is rolled out from the latent state reached after applying
    the existing warm-start actions.
    When the model is not Actionable, missing steps are zero-padded instead.
    If ``init_action`` already covers the full horizon it is returned unchanged.

    Args:
        model: The solver's world model.
        info_dict: Current observation dict with shape ``(n_envs, ...)``.
        init_action: Optional previous plan of shape ``(n_envs, t, action_dim)``
            where ``t <= horizon``.
        horizon: Full planning horizon expected by the solver.
        n_envs: Number of parallel environments.
        action_dim: Flattened action dimension.

    Returns:
        Action tensor of shape ``(n_envs, horizon, action_dim)``.
    """
    if init_action is not None:
        assert init_action.shape[0] == n_envs, (
            f'init_action batch size {init_action.shape[0]} != n_envs {n_envs}'
        )
        assert init_action.shape[2] == action_dim, (
            f'init_action action_dim {init_action.shape[2]} != action_dim {action_dim}'
        )

    n_prev = init_action.shape[1] if init_action is not None else 0
    remaining = horizon - n_prev

    if remaining <= 0:
        return init_action

    if not isinstance(model, Actionable):
        device = init_action.device if init_action is not None else 'cpu'
        tail = torch.zeros(n_envs, remaining, action_dim, device=device)
        if init_action is not None:
            return torch.cat([init_action, tail], dim=1)
        return tail

    with torch.no_grad():
        tail = model.get_action(
            info_dict, horizon=remaining, prefix_actions=init_action
        )
        # tail: (n_envs, remaining, action_dim)

    if init_action is not None:
        return torch.cat([init_action.to(tail.device), tail], dim=1)
    return tail
