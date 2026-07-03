"""Compose a world model with a pluggable objective into a ``Costable``.

``ShootingCostEvaluator`` is the *single-shooting* cost strategy: it rolls the
dynamics out over the full action sequence in one shot, then scores the
resulting trajectory. It splits the world model's monolithic ``get_cost`` into
its three concerns — goal **encode**, dynamics **rollout**, and the
**objective** — letting the objective be swapped freely. Because it exposes
``criterion`` and ``get_cost``, the unmodified solvers consume it exactly like
a world model. The ``Dynamics`` and ``Objective`` protocols it composes live in
:mod:`stable_worldmodel.protocols`.


"""

from collections.abc import Callable

import torch

from stable_worldmodel.protocols import Dynamics, Objective


def default_goal_encode(model: Dynamics, info_dict: dict) -> torch.Tensor:
    """Encode the goal embedding from an ``info_dict``.

    Behavior-preserving extraction of the goal-encoding branch in
    ``LeWM.get_cost``. Override via ``ShootingCostEvaluator(encode_goal=...)`` for
    models that construct their goal differently.
    """
    assert 'goal' in info_dict, 'goal not in info_dict'

    goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
    goal['pixels'] = goal['goal']

    for k in info_dict:
        if k.startswith('goal_'):
            goal[k[len('goal_') :]] = goal.pop(k)

    goal.pop('action')
    goal = model.encode(goal)
    return goal['emb']


class ShootingCostEvaluator(torch.nn.Module):
    """Single-shooting adapter that makes ``(model, objective)`` a ``Costable``.

    Subclasses ``torch.nn.Module`` and registers the world model as a submodule,
    so ``parameters()`` reaches the real model and solvers (CEM/GD) infer the
    candidate dtype from it instead of falling back to float32.

    The actor warm-start tail-fill — a solver calling ``model.get_action`` for
    ``Actionable`` models such as tdmpc2 — is not forwarded through the wrapper;
    it is moot for the ``LeWM``-style models this targets. Add ``get_action``
    delegation if an ``Actionable`` model is ever wrapped.

    Args:
        model: World model providing ``encode``/``rollout`` (the ``Dynamics``
            surface). Registered as a submodule so ``parameters()`` reaches it.
        objective: The cost to apply to the rolled-out ``info_dict``.
        constraints: Optional ``Objective`` terms reused as constraints, each
            returning ``(B, S)`` under the ``LagrangianSolver`` convention
            that a term is satisfied when ``<= 0``. When given,
            ``get_constraints`` is exposed and stacks them into ``(B, S, C)``;
            otherwise the attribute is absent, so a solver probing for it sees
            no constraints.
        encode_goal: Optional ``fn(model, info_dict) -> goal_emb`` to override
            the default goal encoding. Set to ``None`` on the call to skip goal
            encoding entirely (e.g. when the caller pre-populates ``goal_emb``).
    """

    def __init__(
        self,
        model: Dynamics,
        objective: Objective,
        constraints: list[Objective] | None = None,
        encode_goal: Callable[[Dynamics, dict], torch.Tensor]
        | None = default_goal_encode,
    ) -> None:
        super().__init__()
        self.model = model
        self.objective = objective
        self.constraints = constraints
        self.encode_goal = encode_goal
        if constraints:
            self.get_constraints = self._get_constraints

    def _rollout(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> dict:
        """Encode goal (if needed) and roll out candidates into ``info_dict``.

        Shared by ``get_cost`` and ``get_constraints``: the solver calls those
        on separate ``info_dict`` copies, so each must encode + roll out itself.
        """
        if self.encode_goal is not None and 'goal_emb' not in info_dict:
            info_dict['goal_emb'] = self.encode_goal(self.model, info_dict)

        info_dict = self.model.rollout(info_dict, action_candidates)
        info_dict['action_candidates'] = action_candidates
        return info_dict

    def criterion(
        self,
        info_dict: dict,
        action_candidates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score an already-rolled-out ``info_dict`` with the objective."""
        if action_candidates is not None:
            info_dict['action_candidates'] = action_candidates
        return self.objective(info_dict)

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Encode goal (if needed), roll out candidates, then score them."""
        info_dict = self._rollout(info_dict, action_candidates)
        return self.objective(info_dict)

    def _get_constraints(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Roll out candidates and stack constraint terms into ``(B, S, C)``.

        Bound to ``get_constraints`` in ``__init__`` only when constraints are
        configured. Follows the ``LagrangianSolver`` contract: ``g_i <= 0``
        means constraint ``i`` is satisfied.
        """
        info_dict = self._rollout(info_dict, action_candidates)
        return torch.stack(
            [term(info_dict) for term in self.constraints], dim=-1
        )
