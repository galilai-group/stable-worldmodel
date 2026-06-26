"""In-memory replay buffer that doubles as a :class:`Dataset` and a
:class:`Writer`.

Drop the buffer into anything that accepts a Writer (e.g. as the writer in
a rollout collection loop) and rollouts populate it directly. The buffer
subclasses :class:`Dataset`, so a ``DataLoader(buffer, ...)`` iterates
fixed-size clips for training. ``buffer.dump(path, format=...)`` persists
current contents through any registered :class:`Format` writer.

Storage is per-column ring arrays of length ``max_steps``, allocated lazily
from the first episode's schema. Whole episodes are evicted FIFO when
adding the next episode would exceed ``max_steps``. Clips never cross
episode boundaries.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Collection, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import Dataset
from .format import get_format


Sampler = Callable[[int, 'ReplayBuffer', int, int], Any]


def _uniform_sampler(
    step: int, buffer: ReplayBuffer, batch_size: int, history_len: int
) -> np.ndarray:
    n = buffer.num_valid_ends(history_len)
    if n == 0:
        raise RuntimeError(
            f'ReplayBuffer.sample: no clips of history_len={history_len} '
            f'available (num_episodes={buffer.num_episodes}, '
            f'num_steps_stored={buffer.num_steps_stored})'
        )
    return np.random.randint(0, n, size=batch_size)


class ReplayBuffer(Dataset):
    """In-memory ring-storage replay buffer.

    Args:
        max_steps: Capacity in steps. Whole episodes are evicted FIFO when
            adding a new episode would exceed this.
        history_len: Steps per clip returned by ``__getitem__`` (the Dataset
            path) and the default for ``sample(...)``. Equivalent to
            Dataset's ``num_steps``.
        frameskip: Stride between samples for non-dense columns within a
            clip. Actions remain dense and are reshaped to
            ``(history_len, frameskip * action_dim)``, matching
            :class:`FolderDataset`.
        sampler: ``fn(step, buffer, batch_size, history_len) -> indices``
            returning flat clip indices in
            ``[0, buffer.num_valid_ends(history_len))``. Default is uniform.
        transform: Optional dict-in / dict-out transform applied per clip
            in the Dataset path (``__getitem__``).
        key_filter: Optional ``fn(ep_data) -> ep_data`` applied to each
            episode in ``write_episode``, returning the subset of columns to
            store. ``None`` (default) stores every column.
        dense_columns: Additional columns to sample at every underlying row.
            ``action`` is always dense and retains its existing flattened
            representation.
    """

    def __init__(
        self,
        max_steps: int,
        history_len: int = 1,
        frameskip: int = 1,
        sampler: Sampler | None = None,
        transform: Callable[[dict], dict] | None = None,
        key_filter: Callable[[dict], dict] | None = None,
        dense_columns: Collection[str] | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError(f'max_steps must be positive, got {max_steps}')
        if history_len <= 0:
            raise ValueError(
                f'history_len must be positive, got {history_len}'
            )
        if frameskip <= 0:
            raise ValueError(f'frameskip must be positive, got {frameskip}')

        self.max_steps = int(max_steps)
        self.history_len = int(history_len)
        self.num_steps = self.history_len
        self.frameskip = int(frameskip)
        self.span = self.num_steps * self.frameskip
        self.transform = transform
        self.key_filter = key_filter
        self.dense_columns = {'action'}
        if dense_columns is not None:
            self.dense_columns.update(
                (dense_columns,)
                if isinstance(dense_columns, str)
                else dense_columns
            )
        self.sampler: Sampler = (
            sampler if sampler is not None else _uniform_sampler
        )

        self._cols: dict[str, np.ndarray] = {}
        self._head: int = 0
        self._size: int = 0
        self._episodes: deque[tuple[int, int]] = deque()
        self._sample_step: int = 0
        self._clip_starts: np.ndarray | None = None
        self._clip_starts_span: int | None = None

    def __enter__(self) -> ReplayBuffer:
        return self

    def __exit__(self, *exc) -> None:
        return None

    def write_episode(self, ep_data: dict) -> None:
        """Append one completed episode, every column must already span the full episode."""
        if not ep_data:
            return
        if self.key_filter is not None:
            ep_data = self.key_filter(ep_data)
            if not ep_data:
                return
        per_step, ep_len = self._coerce_episode(ep_data)
        if ep_len == 0:
            return
        if ep_len > self.max_steps:
            raise ValueError(
                f'ReplayBuffer.write_episode: episode length {ep_len} '
                f'exceeds max_steps={self.max_steps}'
            )
        self._ensure_allocated(per_step)
        self._evict_to_fit(ep_len)
        self._append(per_step, ep_len)

    def write_episodes(self, episodes: Iterable[dict]) -> None:
        for ep in episodes:
            self.write_episode(ep)

    @property
    def column_names(self) -> list[str]:
        return list(self._cols)

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def num_steps_stored(self) -> int:
        return self._size

    @property
    def lengths(self) -> np.ndarray:
        return np.asarray([ln for _, ln in self._episodes], dtype=np.int32)

    @property
    def offsets(self) -> np.ndarray:
        lens = self.lengths
        if len(lens) == 0:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(([0], np.cumsum(lens[:-1]))).astype(np.int64)

    def num_valid_ends(self, history_len: int | None = None) -> int:
        """Number of valid clips of ``history_len`` (the size of the flat
        index space the sampler chooses from)."""
        h = self.history_len if history_len is None else history_len
        span = h * self.frameskip
        return int(self._get_clip_starts(span)[-1])

    def sample(
        self,
        batch_size: int,
        history_len: int | None = None,
        step: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Draw a batch of clips through the configured sampler.

        Args:
            batch_size: Number of clips.
            history_len: Steps per clip. Defaults to the constructor value.
            step: Current "step" passed to the sampler. If ``None`` the
                buffer's auto-incrementing counter is used.

        Returns:
            ``{col: np.ndarray}`` with each array shaped
            ``(batch_size, history_len, ...)``. Raw numpy — no per-clip
            transform is applied here.
        """
        if batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {batch_size}')
        h = self.history_len if history_len is None else history_len
        if h <= 0:
            raise ValueError(f'history_len must be positive, got {h}')

        cur_step = self._consume_step(step)
        flat = self._call_sampler(cur_step, batch_size, h)

        span = h * self.frameskip
        ep_idx, local_start = self._flat_to_clip(flat, span)
        episodes = list(self._episodes)

        clips = [
            self._gather_clip(episodes[int(e)][0], int(s), h)
            for e, s in zip(ep_idx, local_start)
        ]
        return {
            col: np.stack([c[col] for c in clips], axis=0)
            for col in self._cols
        }

    def __len__(self) -> int:
        return self.num_valid_ends(self.history_len)

    def __repr__(self) -> str:
        cols = ', '.join(self.column_names) or '-'
        return (
            f'ReplayBuffer('
            f'episodes={self.num_episodes}, '
            f'steps={self.num_steps_stored}/{self.max_steps}, '
            f'history_len={self.history_len}, '
            f'frameskip={self.frameskip}, '
            f'columns=[{cols}])'
        )

    def __getitem__(self, idx: int) -> dict:
        n = len(self)
        if n == 0:
            raise IndexError('ReplayBuffer is empty')
        if idx < 0:
            idx += n
        if not 0 <= idx < n:
            raise IndexError(idx)

        ep_idx, local_start = self._flat_to_clip(
            np.array([idx], dtype=np.int64), self.span
        )
        ring_start, _ = self._episodes[int(ep_idx[0])]
        clip = self._gather_clip(
            ring_start, int(local_start[0]), self.history_len
        )
        return self.transform(clip) if self.transform is not None else clip

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        if not 0 <= ep_idx < self.num_episodes:
            raise IndexError(ep_idx)
        ring_start, ep_len = self._episodes[ep_idx]
        if not 0 <= start <= end <= ep_len:
            raise IndexError(
                f'slice [{start}:{end}] out of episode of length {ep_len}'
            )
        positions = (ring_start + np.arange(start, end)) % self.max_steps
        return {col: arr[positions] for col, arr in self._cols.items()}

    def load_chunk(
        self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> list[dict]:
        chunk = []
        for ep, s, e in zip(episodes_idx, start, end):
            ep = int(ep)
            s = int(s)
            e = int(e)
            if not 0 <= ep < self.num_episodes:
                raise IndexError(ep)
            ring_start, ep_len = self._episodes[ep]
            if not 0 <= s <= e <= ep_len:
                raise IndexError(
                    f'slice [{s}:{e}] out of episode of length {ep_len}'
                )
            length = e - s
            if length % self.frameskip:
                raise ValueError(
                    'ReplayBuffer.load_chunk: chunk length must be divisible '
                    'by frameskip '
                    f'(length={length}, frameskip={self.frameskip})'
                )
            chunk.append(
                self._gather_clip(ring_start, s, length // self.frameskip)
            )
        return chunk

    def episodes(self) -> Iterable[dict]:
        """Yield current episodes as ``{col: list[np.ndarray]}`` dicts —
        the per-step-list shape that ``World.collect`` produces and any
        registered :class:`Writer` accepts."""
        for ring_start, ep_len in list(self._episodes):
            positions = (ring_start + np.arange(ep_len)) % self.max_steps
            yield {
                col: list(arr[positions]) for col, arr in self._cols.items()
            }

    def dump(
        self,
        path: str | Path,
        format: str,
        mode: str = 'overwrite',
        **kwargs: Any,
    ) -> None:
        """Persist current contents through the registered writer for ``format``."""
        fmt = get_format(format)
        with fmt.open_writer(path, mode=mode, **kwargs) as writer:
            writer.write_episodes(self.episodes())

    def clear(self) -> None:
        """Drop all stored episodes; reuse allocated arrays."""
        self._head = 0
        self._size = 0
        self._episodes.clear()
        self._invalidate_clip_cache()

    def _coerce_episode(
        self, ep_data: dict
    ) -> tuple[dict[str, np.ndarray], int]:
        """Coerce values to numpy and check per-column lengths agree."""
        per_step: dict[str, np.ndarray] = {}
        ep_len: int | None = None
        for col, vals in ep_data.items():
            arr = np.asarray(vals)
            if arr.ndim == 0:
                raise ValueError(
                    f"ReplayBuffer.write_episode: column '{col}' is scalar; "
                    'expected a per-step array'
                )
            if ep_len is None:
                ep_len = arr.shape[0]
            elif arr.shape[0] != ep_len:
                raise ValueError(
                    f"ReplayBuffer.write_episode: column '{col}' has length "
                    f'{arr.shape[0]}, expected {ep_len}'
                )
            per_step[col] = arr
        return per_step, ep_len or 0

    def _ensure_allocated(self, per_step: dict[str, np.ndarray]) -> None:
        """Allocate ring arrays on the first episode; otherwise check schema."""
        if not self._cols:
            for col, arr in per_step.items():
                self._cols[col] = np.empty(
                    (self.max_steps, *arr.shape[1:]), dtype=arr.dtype
                )
            return

        existing = set(self._cols)
        incoming = set(per_step)
        if existing != incoming:
            raise ValueError(
                'ReplayBuffer.write_episode: schema mismatch. '
                f'missing={sorted(existing - incoming)}, '
                f'extra={sorted(incoming - existing)}'
            )
        for col, arr in per_step.items():
            expected = self._cols[col].shape[1:]
            if arr.shape[1:] != expected:
                raise ValueError(
                    f"ReplayBuffer.write_episode: column '{col}' per-step "
                    f'shape {arr.shape[1:]} mismatches expected {expected}'
                )

    def _evict_to_fit(self, ep_len: int) -> None:
        evicted_any = False
        while self._size + ep_len > self.max_steps and self._episodes:
            _, evicted = self._episodes.popleft()
            self._size -= evicted
            evicted_any = True
        if evicted_any:
            self._invalidate_clip_cache()

    def _append(self, per_step: dict[str, np.ndarray], ep_len: int) -> None:
        positions = (self._head + np.arange(ep_len)) % self.max_steps
        for col, arr in per_step.items():
            self._cols[col][positions] = arr
        self._episodes.append((self._head, ep_len))
        self._head = (self._head + ep_len) % self.max_steps
        self._size += ep_len
        self._invalidate_clip_cache()

    def _consume_step(self, step: int | None) -> int:
        if step is not None:
            return int(step)
        cur = self._sample_step
        self._sample_step += 1
        return cur

    def _call_sampler(
        self, step: int, batch_size: int, history_len: int
    ) -> np.ndarray:
        flat = np.asarray(self.sampler(step, self, batch_size, history_len))
        if flat.shape != (batch_size,):
            raise ValueError(
                f'ReplayBuffer.sample: sampler returned shape {flat.shape}, '
                f'expected ({batch_size},)'
            )
        return flat.astype(np.int64, copy=False)

    def _flat_to_clip(
        self, flat: np.ndarray, span: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map flat clip indices to ``(episode_index, local_clip_start)``."""
        starts = self._get_clip_starts(span)
        total = int(starts[-1])
        if flat.size and (flat.min() < 0 or flat.max() >= total):
            raise IndexError(
                f'ReplayBuffer: clip index out of range [0, {total})'
            )
        ep_idx = np.searchsorted(starts[1:], flat, side='right')
        local_clip_start = flat - starts[ep_idx]
        return ep_idx, local_clip_start

    def _get_clip_starts(self, span: int) -> np.ndarray:
        """Lazy cumulative-clip-starts cache, keyed by ``span``.

        Returns an ``(N+1,)`` int64 array where ``starts[i+1] - starts[i]``
        is the number of valid clip starts in episode ``i``. Recomputed only
        when episodes change or when ``span`` differs from the cached one.
        """
        if self._clip_starts is None or self._clip_starts_span != span:
            n = len(self._episodes)
            valid = np.fromiter(
                (max(0, ln - span + 1) for _, ln in self._episodes),
                dtype=np.int64,
                count=n,
            )
            self._clip_starts = np.empty(n + 1, dtype=np.int64)
            self._clip_starts[0] = 0
            np.cumsum(valid, out=self._clip_starts[1:])
            self._clip_starts_span = span
        return self._clip_starts

    def _invalidate_clip_cache(self) -> None:
        self._clip_starts = None
        self._clip_starts_span = None

    def _gather_clip(
        self, ring_start: int, clip_local_start: int, history_len: int
    ) -> dict[str, np.ndarray]:
        """Gather one clip with sparse and dense columns."""
        base = ring_start + clip_local_start
        sparse_idx = (
            base + np.arange(history_len) * self.frameskip
        ) % self.max_steps
        dense_idx = (
            base + np.arange(history_len * self.frameskip)
        ) % self.max_steps

        clip: dict[str, np.ndarray] = {}
        for col, arr in self._cols.items():
            positions = dense_idx if col in self.dense_columns else sparse_idx
            clip[col] = arr[positions]
        return self._reshape_clip(clip, history_len)


def classic_filter(ep_data: dict) -> dict:
    """``key_filter`` keeping only the classic replay-buffer columns and
    dropping the rest: ``pixels`` (the observation), ``action``, ``reward``,
    ``terminated`` and ``truncated``.

    Use as ``ReplayBuffer(..., key_filter=classic_filter)``.
    """
    classic_keys = ('pixels', 'action', 'reward', 'terminated', 'truncated')
    return {k: v for k, v in ep_data.items() if k in classic_keys}


__all__ = ['ReplayBuffer', 'classic_filter']
