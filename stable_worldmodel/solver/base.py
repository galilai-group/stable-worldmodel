"""Template-method base class for planning solvers.

Subclasses define the search algorithm by overriding hooks
(``init_state``, ``propose``, ``update``, ``finalize``); ``BaseSolver``
owns the outer iteration loop, batching, info-dict expansion, and the
callback lifecycle. ``CompositeSolver`` reuses the same loop by
delegating each hook to per-key children.
"""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from .callbacks import Callback


_State = Any
_Candidates = torch.Tensor | dict[str, torch.Tensor]


class BaseSolver:
    """Template-method base for planning solvers.

    Subclasses implement four hooks:
      - ``init_state(n_envs, init)`` -> opaque state
      - ``propose(state)`` -> candidates (Tensor or dict of Tensors)
      - ``update(state, candidates, costs)`` -> ``(new_state, payload)``
        where ``payload`` is a dict of extra kwargs forwarded to callbacks
      - ``finalize(state)`` -> the final action plan

    Override ``step(state, infos)`` if a solver needs custom inner logic
    that does not fit the propose -> evaluate -> update split.
    """

    def __init__(
        self,
        model: Any,
        n_steps: int,
        batch_size: int | None = None,
        num_samples: int = 1,
        device: str | torch.device = 'cpu',
        callbacks: list[Callback] | None = None,
    ) -> None:
        """Store solver hyperparameters and infer the model dtype."""
        self.model = model
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.device = device
        self.callbacks = list(callbacks) if callbacks else []
        try:
            self._dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            self._dtype = torch.float32
        self._configured = False
        self._n_envs: int | None = None
        self._action_dim: int | None = None
        self._config: Any | None = None
        self._action_space: gym.Space | None = None

    def configure(
        self, *, action_space: gym.Space, n_envs: int, config: Any
    ) -> None:
        """Bind environment-dependent settings before ``solve`` is called."""
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._configured = True

    @property
    def n_envs(self) -> int:
        """Number of parallel environments this solver was configured for."""
        return self._n_envs

    @property
    def horizon(self) -> int:
        """Planning horizon (number of action steps per plan)."""
        return self._config.horizon

    @property
    def action_block(self) -> int:
        """Number of consecutive actions grouped together per planning step."""
        return self._config.action_block

    @property
    def dtype(self) -> torch.dtype:
        """Floating-point dtype used to cast tensors before model calls."""
        return self._dtype

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        """Alias for ``solve`` so the solver can be called like a function."""
        return self.solve(*args, **kwargs)

    def init_state(self, n_envs: int, init: Any | None = None) -> _State:
        """Build the initial search state for a batch of ``n_envs`` envs."""
        raise NotImplementedError

    def propose(self, state: _State) -> _Candidates:
        """Sample candidate action plans from the current search state."""
        raise NotImplementedError

    def update(
        self,
        state: _State,
        candidates: _Candidates,
        costs: torch.Tensor,
    ) -> tuple[_State, dict[str, Any]]:
        """Return ``(new_state, callback_payload)``.

        ``callback_payload`` is merged with ``step``, ``candidates``, and
        ``costs`` and forwarded as kwargs to every callback.
        """
        raise NotImplementedError

    def finalize(self, state: _State) -> Any:
        """Extract the final action plan from the converged search state."""
        raise NotImplementedError

    def step(
        self, state: _State, infos: dict
    ) -> tuple[_State, dict[str, Any], _Candidates, torch.Tensor]:
        """Run one propose -> evaluate -> update iteration of the search loop."""
        cands = self.propose(state)
        costs = self.model.get_cost(infos, cands)
        self._validate_costs(costs, cands)
        new_state, payload = self.update(state, cands, costs)
        return new_state, payload, cands, costs

    def extra_outputs(self, state: _State) -> dict[str, Any]:
        """Solver-specific extras to merge into the final outputs dict."""
        return {}

    def solve(self, info_dict: dict, init_action: Any | None = None) -> dict:
        """Drive the full planning loop over batches and return actions plus diagnostics."""
        start = time.time()
        total_envs = (
            len(next(iter(info_dict.values()))) if info_dict else self._n_envs
        )
        state = self.init_state(total_envs, init_action)

        for cb in self.callbacks:
            cb.reset()

        outputs: dict[str, Any] = {'costs': []}
        bs = self.batch_size if self.batch_size is not None else total_envs

        for s in range(0, total_envs, bs):
            e = min(s + bs, total_envs)
            current_bs = e - s
            bstate = self._slice_state(state, s, e)
            infos = self._expand_infos(info_dict, s, e, current_bs)

            for cb in self.callbacks:
                cb.start_batch()

            final_summary = None
            for step_i in range(self.n_steps):
                bstate, payload, cands, costs = self.step(bstate, infos)
                self._fire_callbacks(step_i, payload, cands, costs)
                final_summary = self._cost_summary(payload, costs)

            self._write_back_state(state, bstate, s, e)
            if final_summary is not None:
                outputs['costs'].extend(final_summary)

        outputs['actions'] = self.finalize(state)
        outputs.update(self.extra_outputs(state))

        if self.callbacks:
            outputs['callbacks'] = {}
            for cb in self.callbacks:
                cb.end_solve()
                outputs['callbacks'][cb.output_key] = cb.history

        print(
            f'{type(self).__name__} solve time: {time.time() - start:.4f} seconds'
        )
        return outputs

    def _slice_state(self, state: _State, start: int, end: int) -> _State:
        """Return the ``[start:end]`` env-slice of a tensor or dict-of-tensors state."""
        if isinstance(state, dict):
            return {
                k: (v[start:end] if torch.is_tensor(v) else v)
                for k, v in state.items()
            }
        if torch.is_tensor(state):
            return state[start:end]
        return state

    def _write_back_state(
        self, state: _State, bstate: _State, start: int, end: int
    ) -> None:
        """Copy a per-batch state slice back into the full-environment state in place."""
        if isinstance(state, dict) and isinstance(bstate, dict):
            for k in state:
                if torch.is_tensor(state[k]) and torch.is_tensor(
                    bstate.get(k)
                ):
                    state[k][start:end] = bstate[k]
        elif torch.is_tensor(state) and torch.is_tensor(bstate):
            state[start:end] = bstate

    def _expand_infos(
        self, info_dict: dict, start: int, end: int, current_bs: int
    ) -> dict:
        """Slice infos to the batch and broadcast each entry to ``num_samples`` copies."""
        N = self.num_samples
        out: dict = {}
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                v_b = v[start:end]
                target_dtype = self.dtype if v_b.is_floating_point() else None
                v_b = (
                    v_b.to(device=self.device, dtype=target_dtype)
                    .unsqueeze(1)
                    .expand(current_bs, N, *v_b.shape[1:])
                )
            elif isinstance(v, np.ndarray):
                v_b = np.repeat(v[start:end, None, ...], N, axis=1)
            else:
                v_b = v
            out[k] = v_b
        return out

    def _validate_costs(
        self, costs: torch.Tensor, candidates: _Candidates
    ) -> None:
        """Check that ``costs`` is a ``(B, N)`` tensor matching the candidate count."""
        if not isinstance(costs, torch.Tensor):
            raise TypeError(
                f'Cost must be torch.Tensor, got {type(costs).__name__}'
            )
        if costs.ndim != 2:
            raise ValueError(
                f'Expected cost shape (B, N), got {tuple(costs.shape)}'
            )
        N = (
            next(iter(candidates.values())).shape[1]
            if isinstance(candidates, dict)
            else candidates.shape[1]
        )
        if costs.shape[1] != N:
            raise ValueError(
                f'Cost N={costs.shape[1]} does not match candidate N={N}'
            )

    def _cost_summary(self, payload: dict, costs: torch.Tensor) -> list[float]:
        """Reduce per-env costs to a single representative scalar for logging."""
        if 'topk_vals' in payload:
            return payload['topk_vals'].mean(dim=1).cpu().tolist()
        return costs.min(dim=1).values.detach().cpu().tolist()

    def _fire_callbacks(
        self,
        step: int,
        payload: dict,
        candidates: _Candidates,
        costs: torch.Tensor,
    ) -> None:
        """Forward the current step's data as kwargs to every registered callback."""
        if not self.callbacks:
            return
        kwargs = {
            'step': step,
            'candidates': candidates,
            'costs': costs,
            **payload,
        }
        for cb in self.callbacks:
            cb(**kwargs)
