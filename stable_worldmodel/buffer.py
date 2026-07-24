"""Online history buffer for policies that need past observations."""

from collections import deque
from typing import Any
from collections.abc import Iterable

import numpy as np
import torch


class HistoryBuffer:
    """Per-env ring buffer over batched info dicts.

    Inputs are dicts whose tensor/ndarray values have a leading env dim
    of size ``n_envs`` (e.g. ``EnvPool``'s stacked infos with shape
    ``(n_envs, 1, ...)``). Each call to :meth:`append` stores one slice
    per env. :meth:`get` returns the last ``n`` entries strided by
    ``action_block`` env steps so history is surfaced at planning
    cadence regardless of frameskip.

    .. warning:: **Warm-up returns synthetic entries.** While an env
        holds fewer than ``(n - 1) * action_block + 1`` entries (the
        first steps after a reset), its history is left-padded with
        **copies of its oldest real entry** — for block keys, with
        **zero blocks**. The consumer (e.g. the world model) therefore
        sees fake repeated frames with zero actions between them, as if
        the env had been stationary before the episode began. Padding
        keeps the time dim at ``n`` for every env so histories stack
        across desynchronized envs without one env's reset truncating
        the others. All entries are real once the env has lived
        ``(n - 1) * action_block + 1`` steps.

    Output for a key with per-step shape ``(n_envs, T, ...)`` is
    ``(n_envs, n*T, ...)`` (concatenated along the time dim, oldest →
    newest). For a key with per-step shape ``(n_envs,)`` the output is
    ``(n_envs, n)``.

    Keys listed in ``block_keys`` are instead returned as ``n - 1``
    macro-blocks: block ``i`` concatenates (flat, chronological) the
    ``action_block`` raw entries recorded after strided frame ``i`` up
    to and including frame ``i + 1``'s entry — the actions executed
    *between* frames ``i`` and ``i + 1`` ("the block leaving frame
    i"), matching the training convention that pairs ``action[t]``
    with ``frame[t]``. With per-env per-step size ``D``, the output is
    ``(n_envs, n - 1, action_block * D)``. Missing blocks are
    zero-padded on the left and NaNs (e.g. the reset entry's action)
    are zeroed. Block keys are omitted when ``n == 1`` (there is no
    "between" with a single frame). Use this for actions when the
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
        self, n: int, env_ids: list[int] | None = None
    ) -> dict[str, torch.Tensor | np.ndarray] | None:
        """Return the last ``n`` strided entries per env, padded in warm-up.

        Entries are returned in chronological order (oldest → newest)
        along the time dim. An env holding fewer than ``n`` strided
        samples is left-padded with **synthetic entries** — copies of
        its oldest real entry (regular keys) or zero blocks
        (``block_keys``) — so the time dim is always ``n`` (``n - 1``
        for block keys) regardless of how full each env's buffer is.
        See the class-level warning: during warm-up the output is
        partially fake.

        Args:
            n: Number of strided entries per env to return.
            env_ids: Env indices to build history for (all if ``None``);
                output rows follow this order.

        Returns:
            A dict of stacked tensors/arrays with time dim ``n * T``
            (``T`` = per-step time dim) for regular keys and ``n - 1``
            for block keys (omitted when ``n == 1``). ``None`` if any
            selected env's buffer is empty or ``n <= 0``.
        """
        if n <= 0:
            return None

        buffers = (
            self._buffers
            if env_ids is None
            else [self._buffers[i] for i in env_ids]
        )
        if any(len(buf) == 0 for buf in buffers):
            return None

        keys = buffers[0][-1].keys()

        out: dict[str, Any] = {}
        for key in keys:
            if key in self.block_keys:
                if n > 1:
                    out[key] = self._get_blocked(key, n, buffers)
                continue
            per_env = []
            for buf in buffers:
                k = min(n, (len(buf) - 1) // self.action_block + 1)
                # newest at offset 0; chronological → reversed indices
                entries = [
                    buf[-1 - off * self.action_block][key]
                    for off in reversed(range(k))
                ]
                entries = [entries[0]] * (n - k) + entries
                per_env.append(_merge_time(entries))
            out[key] = _stack_envs(per_env)
        return out

    def _get_blocked(self, key: str, n: int, buffers) -> Any:
        """``n - 1`` macro-blocks per env: block ``i`` is the flattened
        ``action_block`` raw entries leaving strided frame ``i`` — the
        window ending at frame ``i + 1``'s entry (chronological within
        the block). Left-padded with zero blocks during warm-up; NaNs
        (e.g. the reset entry's action) are zeroed. Output is
        ``(n_envs, n - 1, action_block * D)`` where ``D`` is the
        flattened per-env per-step size of ``key``.
        """
        ab = self.action_block
        per_env = []
        for buf in buffers:
            k = min(n, (len(buf) - 1) // ab + 1)
            blocks = []
            for i in range(k - 1):
                # frame i+1 sits at offset (k - 2 - i) * ab from the
                # newest entry; its block is the ab entries ending there
                start = (k - 2 - i) * ab
                inner = [
                    buf[-1 - off][key]
                    for off in reversed(range(start, start + ab))
                ]
                merged = _merge_time(inner)
                if torch.is_tensor(merged):
                    merged = torch.nan_to_num(merged.flatten(), nan=0.0)
                elif isinstance(merged, np.ndarray):
                    merged = np.nan_to_num(merged.flatten(), nan=0.0)
                blocks.append(merged)
            zero = _zero_block(buf[-1][key], ab)
            blocks = [zero] * (n - 1 - len(blocks)) + blocks
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


def _zero_block(v: Any, action_block: int) -> Any:
    """A flat zero macro-block sized as ``action_block`` copies of ``v``."""
    if torch.is_tensor(v):
        return v.new_zeros(action_block * v.numel())
    v = np.asarray(v)
    return np.zeros(action_block * v.size, dtype=v.dtype)


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
