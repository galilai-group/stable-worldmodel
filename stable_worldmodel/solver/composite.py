"""Composite solver for ``Dict`` action spaces.

Wraps one ``BaseSolver`` per sub-action-key and runs a single shared
outer loop. Each step the children propose candidates for their own
component, the joint dict of candidates is evaluated through one call to
``model.get_cost``, and each child refits from the resulting cost
tensor. This keeps the optimization synchronized so coupled action
components see each other's choices in the cost.
"""

from typing import Any

import gymnasium as gym
import torch

from .base import BaseSolver


class CompositeSolver(BaseSolver):
    """Compose multiple solvers over a ``gym.spaces.Dict`` action space.

    Args:
        sub_solvers: Mapping from action-space key to a ``BaseSolver``.
        device: Device for tensor computations.

    Notes:
        - All children must share the same ``num_samples`` so candidates
          line up under one joint cost evaluation.
        - ``n_steps``, ``batch_size``, and the planning ``config`` come
          from the first child; children's per-instance ``n_steps`` is
          ignored (the composite owns the outer loop).
        - Callbacks attach to the composite (not its children) and fire
          with namespaced payload keys ``"<child>.<key>"``.
    """

    def __init__(
        self,
        sub_solvers: dict[str, BaseSolver],
        device: str | torch.device = 'cpu',
        callbacks: list | None = None,
    ) -> None:
        if not sub_solvers:
            raise ValueError('CompositeSolver requires at least one child')
        first = next(iter(sub_solvers.values()))
        super().__init__(
            model=first.model,
            n_steps=first.n_steps,
            batch_size=first.batch_size,
            num_samples=first.num_samples,
            device=device,
            callbacks=callbacks,
        )
        self.subs = dict(sub_solvers)

    def configure(
        self, *, action_space: gym.Space, n_envs: int, config: Any
    ) -> None:
        if not isinstance(action_space, gym.spaces.Dict):
            raise TypeError(
                'CompositeSolver requires a Dict action space, '
                f'got {type(action_space).__name__}'
            )
        super().configure(
            action_space=action_space, n_envs=n_envs, config=config
        )
        missing = set(self.subs) - set(action_space.spaces)
        if missing:
            raise KeyError(
                f'CompositeSolver children {sorted(missing)} have no '
                f'matching key in action_space {sorted(action_space.spaces)}'
            )
        for k, sub in self.subs.items():
            sub.configure(
                action_space=action_space[k], n_envs=n_envs, config=config
            )
        ns = {k: s.num_samples for k, s in self.subs.items()}
        if len(set(ns.values())) > 1:
            raise ValueError(
                f'CompositeSolver children must share num_samples; got {ns}'
            )

    # === BaseSolver hooks ===

    def init_state(
        self, n_envs: int, init: dict | None = None
    ) -> dict[str, Any]:
        init = init or {}
        return {
            k: s.init_state(n_envs, init.get(k)) for k, s in self.subs.items()
        }

    def propose(self, state: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {k: s.propose(state[k]) for k, s in self.subs.items()}

    def update(
        self,
        state: dict[str, Any],
        candidates: dict[str, torch.Tensor],
        costs: torch.Tensor,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        new_state: dict[str, Any] = {}
        payload: dict[str, Any] = {}
        for k, sub in self.subs.items():
            ns, p = sub.update(state[k], candidates[k], costs)
            new_state[k] = ns
            for pk, pv in p.items():
                payload[f'{k}.{pk}'] = pv
        return new_state, payload

    def finalize(self, state: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {k: s.finalize(state[k]) for k, s in self.subs.items()}

    def extra_outputs(self, state: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, s in self.subs.items():
            for ek, ev in s.extra_outputs(state[k]).items():
                out[f'{k}.{ek}'] = ev
        return out

    # === Composite-aware overrides for state plumbing ===

    def _slice_state(
        self, state: dict[str, Any], start: int, end: int
    ) -> dict[str, Any]:
        return {
            k: self.subs[k]._slice_state(state[k], start, end) for k in state
        }

    def _write_back_state(
        self,
        state: dict[str, Any],
        bstate: dict[str, Any],
        start: int,
        end: int,
    ) -> None:
        for k in state:
            self.subs[k]._write_back_state(state[k], bstate[k], start, end)
