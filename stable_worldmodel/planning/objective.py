"""Pluggable planning objectives, decoupled from the world model.

An :class:`~stable_worldmodel.protocols.Objective` scores a populated
``info_dict`` (after the world model has rolled out candidate actions) and
returns a per-candidate cost tensor of shape ``(B, S)``. Objectives are plain,
composable objects: swapping or combining costs no longer requires subclassing
the world model. The ``Objective`` protocol itself lives in
:mod:`stable_worldmodel.protocols`; this module holds concrete objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_worldmodel.protocols import Objective


class GoalMSE(nn.Module):
    """Last-step MSE between predicted and goal embeddings.

    Reads ``predicted_emb`` ``(B, S, T-1, dim)`` and ``goal_emb``
    ``(B, T, dim)``; returns per-candidate cost ``(B, S)``.

    Behavior-preserving extraction of ``LeWM.criterion`` (reproduces it
    bit-for-bit), so an existing model migrates to the ``ShootingCostEvaluator`` seam
    without changing results.
    """

    def __init__(
        self, pred_key: str = 'predicted_emb', goal_key: str = 'goal_emb'
    ) -> None:
        super().__init__()
        self.pred_key = pred_key
        self.goal_key = goal_key

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred_emb = info_dict[self.pred_key]
        goal_emb = info_dict[self.goal_key][:, None, -1:, :].expand_as(
            pred_emb
        )
        return F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction='none',
        ).sum(dim=tuple(range(2, pred_emb.ndim)))


class ControlPenalty(nn.Module):
    """L2 penalty on the action candidates themselves.

    Reads ``action_candidates`` (shape ``(B, S, H, action_dim)``) from the
    ``info_dict`` — the ``ShootingCostEvaluator`` stores them there before scoring —
    and returns a per-candidate cost ``(B, S)``. Demonstrates a cost the world
    model never had to know about.
    """

    def __init__(self, action_key: str = 'action_candidates') -> None:
        super().__init__()
        self.action_key = action_key

    def forward(self, info_dict: dict) -> torch.Tensor:
        actions = info_dict[self.action_key]
        return actions.pow(2).sum(dim=tuple(range(2, actions.ndim)))


class WeightedSum(nn.Module):
    """Linear combination of objectives: ``sum(w * term(info_dict))``.

    Lets us assemble multi-term costs (goal distance + control penalty +
    constraints) without touching the model or the solver.
    """

    def __init__(self, terms: list[tuple[float, Objective]]) -> None:
        super().__init__()
        if not terms:
            raise ValueError('WeightedSum requires at least one term')
        self.weights = [weight for weight, _ in terms]
        self.objectives = nn.ModuleList(term for _, term in terms)

    def forward(self, info_dict: dict) -> torch.Tensor:
        total = None
        for weight, term in zip(self.weights, self.objectives):
            value = weight * term(info_dict)
            total = value if total is None else total + value
        return total
