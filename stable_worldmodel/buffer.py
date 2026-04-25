"""Online history buffer for policies that need past observations."""

from collections import deque
from typing import Any, Iterable

import numpy as np
import torch


class HistoryBuffer:
    """Per-env ring buffer over batched info dicts.

    Inputs are dicts whose tensor/ndarray values have a leading env dim
    of size ``n_envs`` (e.g. ``EnvPool``'s stacked infos with shape
    ``(n_envs, 1, ...)``). Each call to :meth:`append` stores one slice
    per env. :meth:`get` returns up to the last ``n`` entries strided
    by ``action_block`` env steps so history is surfaced at planning
    cadence regardless of frameskip.

    During warm-up (when fewer than ``(n - 1) * action_block + 1``
    entries are available) :meth:`get` returns the maximum number of
    strided entries that fit, so the time dim grows from 1 up to ``n``
    over the first ``(n - 1) * action_block + 1`` env steps. The shape
    is consistent across envs (limited by the smallest buffer).

    Output for a key with per-step shape ``(n_envs, T, ...)`` is
    ``(n_envs, k*T, ...)`` (concatenated along the time dim, oldest →
    newest), where ``k = min(n, max strided entries available)``. For
    a key with per-step shape ``(n_envs,)`` the output is
    ``(n_envs, k)``.

    Keys listed in ``block_keys`` are instead returned as macro-blocks:
    for each strided history point, the ``action_block`` raw entries
    in the window leading up to it are concatenated and flattened.
    With per-env per-step size ``D``, the output is
    ``(n_envs, k, action_block * D)``. Use this for actions when the
    planner operates at macro-action cadence.

    Args:
        n_envs: Number of parallel environments.
        max_len: Maximum entries retained per env (in env steps).
        action_block: Frameskip — stride used when returning history.
        block_keys: Keys to aggregate as macro-blocks (typically
            ``('action',)``).
    """

    def __init__(
        self,
        n_envs: int,
        max_len: int,
        action_block: int = 1,
        block_keys: Iterable[str] = (),
    ) -> None:
        if n_envs <= 0:
            raise ValueError(f'n_envs must be positive, got {n_envs}')
        if max_len <= 0:
            raise ValueError(f'max_len must be positive, got {max_len}')
        if action_block <= 0:
            raise ValueError(
                f'action_block must be positive, got {action_block}'
            )

        self.n_envs = n_envs
        self.max_len = max_len
        self.action_block = action_block
        self.block_keys = frozenset(block_keys)
        self._buffers: list[deque[dict[str, Any]]] = [
            deque(maxlen=max_len) for _ in range(n_envs)
        ]

    def append(self, info_dict: dict[str, Any]) -> None:
        """Append one entry per env from a batched info dict.

        Splits each value along dim 0 (the env dim) and pushes the
        per-env slice onto the corresponding deque.

        Args:
            info_dict: Dict of values each with leading dim ``n_envs``.
        """
        for i in range(self.n_envs):
            entry = {k: _slice_env(v, i) for k, v in info_dict.items()}
            self._buffers[i].append(entry)

    def get(
        self, n: int
    ) -> dict[str, torch.Tensor | np.ndarray] | None:
        """Return up to the last ``n`` entries per env, strided by ``action_block``.

        Entries are returned in chronological order (oldest → newest)
        along the time dim. If buffers don't yet hold enough entries
        for ``n`` strided samples, returns the largest ``k <= n`` that
        fits in every env's buffer. Returns ``None`` only if some env
        is empty.

        Args:
            n: Maximum number of entries per env to return.

        Returns:
            A dict of stacked tensors/arrays with time dim ``k * T``
            where ``k = min(n, (min_len - 1) // action_block + 1)``
            and ``T`` is the per-step time dim of each value. ``None``
            if any env's buffer is empty or ``n <= 0``.
        """
        if n <= 0:
            return None

        min_len = min(len(buf) for buf in self._buffers)
        if min_len == 0:
            return None

        # Block keys require ``k * action_block`` raw entries (full
        # windows), which is one fewer stride point than the standard
        # formula in the worst case. Tighten k for all keys to keep the
        # time dim consistent across the dict.
        if self.block_keys and self.action_block > 1:
            k = min(n, min_len // self.action_block)
        else:
            k = min(n, (min_len - 1) // self.action_block + 1)
        if k <= 0:
            return None

        # newest at offset 0; chronological → reversed indices
        offsets = list(reversed([i * self.action_block for i in range(k)]))
        keys = self._buffers[0][-1].keys()

        out: dict[str, Any] = {}
        for key in keys:
            if key in self.block_keys and self.action_block > 1:
                out[key] = self._get_blocked(key, k)
            else:
                per_env = []
                for buf in self._buffers:
                    entries = [buf[-1 - off][key] for off in offsets]
                    per_env.append(_merge_time(entries))
                out[key] = _stack_envs(per_env)
        return out

    def _get_blocked(self, key: str, k: int) -> Any:
        """Aggregate ``action_block`` raw entries per stride into one block.

        For each of the ``k`` strided history points, gather the
        ``action_block`` raw entries in the window ending at that
        stride point (chronological within the block) and concatenate
        them flat. Output is ``(n_envs, k, action_block * D)`` where
        ``D`` is the flattened per-env per-step size of ``key``.
        """
        ab = self.action_block
        per_env = []
        for buf in self._buffers:
            blocks = []
            for k_idx in reversed(range(k)):
                # entries within block, chronological (oldest → newest)
                inner_offsets = list(
                    reversed(range(k_idx * ab, (k_idx + 1) * ab))
                )
                inner = [buf[-1 - off][key] for off in inner_offsets]
                merged = _merge_time(inner)
                if torch.is_tensor(merged):
                    merged = merged.flatten()
                elif isinstance(merged, np.ndarray):
                    merged = merged.flatten()
                blocks.append(merged)
            per_env.append(_stack_envs(blocks))
        return _stack_envs(per_env)

    def reset(self, env_ids: list[int] | None = None) -> None:
        """Clear buffers for the given envs (all if ``None``)."""
        ids = range(self.n_envs) if env_ids is None else env_ids
        for i in ids:
            self._buffers[i].clear()

    def __len__(self) -> int:
        """Minimum entry count across envs."""
        return min(len(buf) for buf in self._buffers)


def _slice_env(v: Any, i: int) -> Any:
    if torch.is_tensor(v) or isinstance(v, np.ndarray):
        return v[i]
    return v


def _merge_time(xs: list[Any]) -> Any:
    first = xs[0]
    if torch.is_tensor(first):
        if first.ndim == 0:
            return torch.stack(xs, dim=0)
        return torch.cat(xs, dim=0)
    if isinstance(first, np.ndarray):
        if first.ndim == 0:
            return np.stack(xs, axis=0)
        return np.concatenate(xs, axis=0)
    return np.array(xs)


def _stack_envs(xs: list[Any]) -> Any:
    first = xs[0]
    if torch.is_tensor(first):
        return torch.stack(xs, dim=0)
    if isinstance(first, np.ndarray):
        return np.stack(xs, axis=0)
    return np.array(xs)
