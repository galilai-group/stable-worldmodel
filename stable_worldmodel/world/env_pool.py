"""Environment pools for synchronous and asynchronous stepping.

``EnvPool`` is a lightweight replacement for
``gymnasium.vector.SyncVectorEnv`` with two differences tailored to ``World``:

1. ``reset(mask=...)`` and ``step(mask=...)`` can skip individual envs —
   useful for the ``wait`` reset mode where terminated envs freeze until
   every env has finished.
2. The stacked info dict is pre-allocated on the first reset and updated
   in-place afterwards. Tensor/array values are shaped ``(num_envs, 1, ...)``
   so consumers can rely on a ``(batch, time, ...)`` convention without the
   pool re-stacking every step.

``AsyncEnvPool`` keeps the same spaces, environments, and stacked-info layout,
but runs one operation per environment on a thread pool. Callers submit work
for selected environment indices and consume whichever results finish first;
there is no all-environment step barrier.
"""

from __future__ import annotations

from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space


AsyncEnvEventKind = Literal['step', 'reset']
AsyncEnvMask = np.NDArray[np.bool_]

class EnvPool:
    """Batched env runner with selective stepping.

    Args:
        env_fns: List of zero-arg factories, one per env. Each is called
            once and the result is kept for the lifetime of the pool.
    """

    def __init__(self, env_fns: list):
        self.envs = [fn() for fn in env_fns]
        self._single_env = self.envs[0]
        self._stacked_infos: dict[str, Any] | None = None
        self.seeds = np.zeros(len(self.envs), dtype=np.int64)
        # Cache batched spaces — rebuilding them per-access creates a fresh
        # unseeded space each call, so .seed() / .sample() never advances RNG.
        self._action_space = batch_space(
            self._single_env.action_space, len(self.envs)
        )
        self._observation_space = batch_space(
            self._single_env.observation_space, len(self.envs)
        )

    @property
    def num_envs(self) -> int:
        """Number of envs in the pool."""
        return len(self.envs)

    @property
    def action_space(self) -> gym.Space:
        """Batched action space (``batch_space(single_action_space, num_envs)``)."""
        return self._action_space

    @property
    def single_action_space(self) -> gym.Space:
        """Action space of a single env."""
        return self._single_env.action_space

    @property
    def observation_space(self) -> gym.Space:
        """Batched observation space."""
        return self._observation_space

    @property
    def single_observation_space(self) -> gym.Space:
        """Observation space of a single env."""
        return self._single_env.observation_space

    @property
    def variation_space(self):
        """Variation space from the unwrapped env, or ``None`` if not defined."""
        return getattr(self._single_env.unwrapped, 'variation_space', None)

    @property
    def single_variation_space(self):
        """Variation space for a single env (alias of ``variation_space``)."""
        return self.variation_space

    def reset(
        self,
        seed: int | list[int | None] | None = None,
        options: dict | list[dict | None] | None = None,
        mask: np.ndarray | None = None,
    ) -> tuple[None, dict]:
        """Reset envs and return the stacked info dict.

        Args:
            seed: Base int (each env gets ``seed + i``), a per-env list,
                or ``None``.
            options: Shared dict or per-env list.
            mask: If provided, only envs where ``mask[i]`` is truthy are
                reset. Others keep their current state in the stacked
                info buffer.
        """
        seeds = _broadcast_arg(seed, self.num_envs, increment=True)
        opts = _broadcast_arg(options, self.num_envs)

        per_env_infos = [None] * self.num_envs
        for i, env in enumerate(self.envs):
            if mask is not None and not mask[i]:
                continue
            _, per_env_infos[i] = env.reset(seed=seeds[i], options=opts[i])
            if seeds[i] is not None:
                self.seeds[i] = seeds[i]

        if self._stacked_infos is None or mask is None:
            self._stacked_infos = _stack_fresh(per_env_infos)
        else:
            for i, info in enumerate(per_env_infos):
                if info is not None:
                    _write_env_info(self._stacked_infos, i, info)

        return None, self._stacked_infos

    def step(
        self, actions: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[None, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Step envs and return ``(None, rewards, terminateds, truncateds, infos)``.

        Args:
            actions: Array of shape ``(num_envs, ...)`` — one action per env.
            mask: If provided, only envs where ``mask[i]`` is truthy are
                stepped. Masked envs contribute zero reward and ``False``
                termination/truncation, and their slot in the stacked
                info buffer is left unchanged.
        """
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminateds = np.zeros(self.num_envs, dtype=bool)
        truncateds = np.zeros(self.num_envs, dtype=bool)

        for i, env in enumerate(self.envs):
            if mask is not None and not mask[i]:
                continue
            _, rewards[i], terminateds[i], truncateds[i], info = env.step(
                actions[i]
            )
            _write_env_info(self._stacked_infos, i, info)

        return None, rewards, terminateds, truncateds, self._stacked_infos

    def close(self):
        """Close every env in the pool."""
        for env in self.envs:
            env.close()


@dataclass(frozen=True)
class AsyncEnvEvent:
    """Result of one asynchronous operation on one environment.

    ``info`` is the unbatched dictionary returned by the environment. The
    pool also writes it into its shared stacked-info buffers before returning
    the event.
    """

    env_idx: int
    kind: AsyncEnvEventKind
    info: dict
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class _PendingOperation:
    env_idx: int
    kind: AsyncEnvEventKind
    seed: int | None = None


class AsyncEnvPool(EnvPool):
    """Event-driven environment pool with no per-step batch barrier.

    Environments remain in the main process so existing policies can inspect
    ``pool.envs[i]``. Each environment has at most one operation in flight,
    while operations for different environments run concurrently.

    The regular :meth:`reset` and :meth:`step` methods remain available as
    blocking compatibility operations. Asynchronous callers use
    :meth:`submit_step`, :meth:`submit_reset`, and :meth:`wait_ready`.
    """

    def __init__(self, env_fns: list):
        super().__init__(env_fns)
        self._executor = ThreadPoolExecutor(
            max_workers=self.num_envs,
            thread_name_prefix='swm-env',
        )
        self._pending: dict[Future, _PendingOperation] = {}
        self._busy = np.zeros(self.num_envs, dtype=bool)
        self._closed = False

    @property
    def has_pending(self) -> bool:
        """Whether any environment operation is currently in flight."""
        return bool(self._pending)

    @property
    def pending_indices(self) -> np.ndarray:
        """Indices with an operation currently in flight."""
        return np.flatnonzero(self._busy)

    def reset(
        self,
        seed: int | list[int | None] | None = None,
        options: dict | list[dict | None] | None = None,
        mask: np.ndarray | None = None,
    ) -> tuple[None, dict]:
        self._ensure_idle()
        return super().reset(seed=seed, options=options, mask=mask)

    def step(
        self, actions: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[None, np.ndarray, np.ndarray, np.ndarray, dict]:
        self._ensure_idle()
        return super().step(actions, mask=mask)

    def submit_step(self, env_indices, actions: np.ndarray) -> None:
        """Submit one action for every selected environment.

        Args:
            env_indices: Integer indices, or a boolean mask of length
                ``num_envs``.
            actions: Actions aligned with ``env_indices``. Its leading
                dimension must equal the number of selected environments.
        """
        indices = _normalize_indices(env_indices, self.num_envs)
        self._ensure_available(indices)
        actions = np.asarray(actions)
        if actions.ndim == 0:
            actions = actions.reshape(1)
        if len(actions) != len(indices):
            raise ValueError(
                'actions must have one leading entry per environment index; '
                f'got {len(actions)} actions for {len(indices)} indices'
            )

        for row, env_idx in enumerate(indices):
            action = np.array(actions[row], copy=True)
            self._submit(
                int(env_idx),
                'step',
                _step_env,
                self.envs[env_idx],
                action,
            )

    def submit_reset(
        self,
        env_indices,
        seed: int | list[int | None] | np.ndarray | None = None,
        options: dict | list[dict | None] | None = None,
    ) -> None:
        """Submit independent resets for selected environments."""
        indices = _normalize_indices(env_indices, self.num_envs)
        self._ensure_available(indices)
        seeds = _select_args(seed, indices, self.num_envs, increment=True)
        opts = _select_args(options, indices, self.num_envs)

        for env_idx, env_seed, env_options in zip(indices, seeds, opts):
            self._submit(
                int(env_idx),
                'reset',
                _reset_env,
                self.envs[env_idx],
                env_seed,
                env_options,
                seed=env_seed,
            )

    def wait_ready(self, timeout: float | None = None) -> list[AsyncEnvEvent]:
        """Return all operations ready when the first one completes.

        This is the key asynchronous primitive: it waits for *any* pending
        environment, then opportunistically drains the others that have
        already completed without waiting for the slowest environment.
        """
        if not self._pending:
            raise RuntimeError(
                'No asynchronous environment operations pending.'
            )

        done, _ = wait(
            tuple(self._pending),
            timeout=timeout,
            return_when=FIRST_COMPLETED,
        )
        if not done:
            return []

        done.update(future for future in self._pending if future.done())
        completed = sorted(
            ((self._pending[future], future) for future in done),
            key=lambda item: item[0].env_idx,
        )

        events = []
        for operation, future in completed:
            self._pending.pop(future)
            self._busy[operation.env_idx] = False
            try:
                event = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f'Environment {operation.env_idx} failed during '
                    f'{operation.kind}.'
                ) from exc

            if self._stacked_infos is None:
                raise RuntimeError(
                    'AsyncEnvPool must be reset once before asynchronous work.'
                )
            _write_env_info(self._stacked_infos, event.env_idx, event.info)
            if operation.kind == 'reset' and operation.seed is not None:
                self.seeds[event.env_idx] = operation.seed
            events.append(event)

        return events

    def close(self):
        """Finish pending work, stop worker threads, and close every env."""
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()
        self._busy[:] = False
        super().close()

    def _submit(self, env_idx, kind, fn, *args, seed=None) -> None:
        if self._closed:
            raise RuntimeError('AsyncEnvPool is closed.')
        if self._busy[env_idx]:
            raise RuntimeError(
                f'Environment {env_idx} already has an operation in flight.'
            )

        self._busy[env_idx] = True
        try:
            future = self._executor.submit(fn, env_idx, *args)
        except Exception:
            self._busy[env_idx] = False
            raise
        self._pending[future] = _PendingOperation(env_idx, kind, seed)

    def _ensure_idle(self) -> None:
        if self._pending:
            raise RuntimeError(
                'Cannot run a blocking pool operation while asynchronous '
                'operations are pending.'
            )

    def _ensure_available(self, indices: np.ndarray) -> None:
        busy = indices[self._busy[indices]]
        if len(busy):
            raise RuntimeError(
                f'Environments {busy.tolist()} already have operations '
                'in flight.'
            )


def _step_env(env_idx: int, env, action) -> AsyncEnvEvent:
    _, reward, terminated, truncated, info = env.step(action)
    return AsyncEnvEvent(
        env_idx=env_idx,
        kind='step',
        info=info,
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def _reset_env(env_idx: int, env, seed, options) -> AsyncEnvEvent:
    _, info = env.reset(seed=seed, options=options)
    return AsyncEnvEvent(env_idx=env_idx, kind='reset', info=info)


def _broadcast_arg(arg, n: int, increment: bool = False) -> list:
    if arg is None:
        return [None] * n
    if isinstance(arg, list):
        return arg
    if isinstance(arg, np.ndarray):
        return list(arg)
    if increment and isinstance(arg, (int, np.integer)):
        return [arg + i for i in range(n)]
    return [arg] * n


def _normalize_indices(env_indices, n: int) -> np.ndarray:
    indices = np.asarray(env_indices)
    if indices.dtype == bool:
        if indices.shape != (n,):
            raise ValueError(f'boolean env mask must have shape ({n},)')
        indices = np.flatnonzero(indices)
    else:
        indices = np.atleast_1d(indices).astype(np.int64)

    if ((indices < 0) | (indices >= n)).any():
        raise IndexError(f'environment indices must be between 0 and {n - 1}')
    if len(np.unique(indices)) != len(indices):
        raise ValueError('environment indices must be unique')
    return indices


def _select_args(arg, indices: np.ndarray, n: int, increment: bool = False):
    if arg is None:
        return [None] * len(indices)
    if increment and isinstance(arg, int):
        return [arg + int(i) for i in indices]
    if isinstance(arg, (list, np.ndarray)):
        values = list(arg)
        if len(values) == n:
            return [values[i] for i in indices]
        if len(values) == len(indices):
            return values
        raise ValueError(
            f'per-environment argument must have length {n} or {len(indices)}'
        )
    return [arg] * len(indices)


def _stack_fresh(per_env_infos: list[dict]) -> dict[str, Any]:
    """Build stacked info arrays from a full set of per-env infos.

    Tensor/array values get a leading time dim of 1 after the env dim,
    yielding shape (N, 1, ...) so downstream consumers can rely on a
    (batch, time, ...) convention.
    """
    keys = per_env_infos[0].keys()
    stacked = {}
    for k in keys:
        vals = [info[k] for info in per_env_infos]
        first = vals[0]
        if isinstance(first, torch.Tensor):
            stacked[k] = torch.stack(vals).unsqueeze(1)
        elif isinstance(first, np.ndarray):
            stacked[k] = np.stack(vals)[:, None, ...]
        elif isinstance(first, (bool, int, float, np.number)):
            stacked[k] = np.array(vals)[:, None]
        else:
            stacked[k] = [[v] for v in vals]
    return stacked


def _write_env_info(stacked: dict, idx: int, info: dict) -> None:
    """Write a single env's info into pre-allocated stacked arrays in-place."""
    for k, v in info.items():
        if k not in stacked:
            continue
        buf = stacked[k]
        if isinstance(buf, torch.Tensor):
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v, dtype=buf.dtype, device=buf.device)
            buf[idx, 0] = v
        elif isinstance(buf, np.ndarray):
            buf[idx, 0] = v
        elif isinstance(buf, list):
            buf[idx][0] = v
