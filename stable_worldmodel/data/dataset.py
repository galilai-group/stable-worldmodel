"""Dataset abstractions: the base class plus composition wrappers.

Concrete readers (HDF5, folder, video, LeRobot) live under ``data.formats``.
This module is the cross-cutting layer:

  - :class:`Dataset` — the abstract base shared by every reader.
  - :class:`MergeDataset` — horizontal join (columns from N datasets of equal length).
  - :class:`ConcatDataset` — vertical concat (episodes from N datasets stacked).
  - :class:`GoalDataset` — augments any dataset with a sampled goal observation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch


class Dataset:
    """Base class for episode-based datasets.

    Subclasses fill in ``column_names`` and ``_load_slice``; everything else
    (clip indexing, ``__getitem__``, ``load_chunk``, ``load_episode``) is
    derived here.

    Args:
        lengths: Episode lengths.
        offsets: Episode start offsets in the underlying flat storage.
        frameskip: Stride between observation samples.
        num_steps: Number of observation steps per sample.
        transform: Optional dict-in / dict-out transform applied per sample.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        offsets: np.ndarray,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
    ) -> None:
        self.lengths = lengths
        self.offsets = offsets
        self.frameskip = frameskip
        self.num_steps = num_steps
        self.span = num_steps * frameskip
        self.transform = transform
        # (N, 2) int64 array of (episode, start) pairs. A materialized
        # Python list of tuples costs ~100 bytes per window — for tens of
        # millions of windows that is gigabytes PER DATALOADER WORKER and
        # minutes of pickling at every worker spawn; the array form is
        # ~16 bytes per window and pickles as a single buffer copy.
        # Rows unpack like tuples (``ep, start = clip_indices[idx]``).
        valid = [
            (ep, length - self.span + 1)
            for ep, length in enumerate(lengths)
            if length >= self.span
        ]
        if valid:
            eps = np.repeat(
                np.array([e for e, _ in valid], dtype=np.int64),
                np.array([c for _, c in valid], dtype=np.int64),
            )
            starts = np.concatenate(
                [np.arange(c, dtype=np.int64) for _, c in valid]
            )
            self.clip_indices = np.stack([eps, starts], axis=1)
        else:
            self.clip_indices = np.empty((0, 2), dtype=np.int64)

    @property
    def column_names(self) -> list[str]:
        """Names of the columns stored in the dataset."""
        raise NotImplementedError

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.clip_indices)

    def __getitem__(self, idx: int) -> dict:
        """Load one clip of ``num_steps`` observations (``frameskip`` apart).

        Args:
            idx: Clip index into ``clip_indices``.

        Returns:
            Dict of per-column tensors for the clip; ``action`` is reshaped
            to ``(num_steps, -1)`` so skipped actions stay grouped per step.
        """
        ep_idx, start = self.clip_indices[idx]
        steps = self._load_slice(ep_idx, start, start + self.span)
        if 'action' in steps:
            steps['action'] = steps['action'].reshape(self.num_steps, -1)
        return steps

    def load_chunk(
        self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> list[dict]:
        """Load one step-range per episode, in bulk.

        Args:
            episodes_idx: Episode index for each slice to load.
            start: Start step (inclusive) of each slice, per episode.
            end: End step (exclusive) of each slice, per episode.

        Returns:
            One dict of per-column tensors per requested slice, in order;
            ``action`` is reshaped to ``((end - start) // frameskip, -1)``.
        """
        chunk = []
        for ep, s, e in zip(episodes_idx, start, end):
            steps = self._load_slice(ep, s, e)
            if 'action' in steps:
                steps['action'] = steps['action'].reshape(
                    (e - s) // self.frameskip, -1
                )
            chunk.append(steps)
        return chunk

    def load_episode(self, episode_idx: int) -> dict:
        """Load a full episode as a dict of per-column tensors.

        Args:
            episode_idx: Index of the episode to load.

        Returns:
            Dict of per-column tensors covering every step of the episode.
        """
        return self._load_slice(episode_idx, 0, self.lengths[episode_idx])

    def get_col_data(self, col: str) -> np.ndarray:
        """Return every value of column ``col`` across the whole dataset."""
        raise NotImplementedError

    def get_dim(self, col: str) -> int:
        """Return the per-step dimensionality of column ``col``."""
        raise NotImplementedError

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        """Return all columns for the given flat storage row(s)."""
        raise NotImplementedError

    def merge_col(
        self,
        source: list[str] | str,
        target: str,
        dim: int = -1,
    ) -> None:
        """Concatenate ``source`` column(s) into a new ``target`` column.

        Args:
            source: Column name(s) to combine.
            target: Name of the resulting column.
            dim: Axis along which the source columns are concatenated.
        """
        raise NotImplementedError


class MergeDataset:
    """Merge several datasets of equal length by columns (horizontal join).

    Args:
        datasets: Datasets to merge.
        keys_from_dataset: Per-dataset key lists. If omitted, each dataset
            contributes the columns not yet seen in earlier datasets.
    """

    def __init__(
        self,
        datasets: list[Any],
        keys_from_dataset: list[list[str]] | None = None,
    ) -> None:
        if not datasets:
            raise ValueError('Need at least one dataset')
        self.datasets = datasets
        self._len = len(datasets[0])

        if keys_from_dataset:
            self.keys_map = keys_from_dataset
        else:
            seen: set[str] = set()
            self.keys_map = []
            for ds in datasets:
                keys = [c for c in ds.column_names if c not in seen]
                seen.update(keys)
                self.keys_map.append(keys)

    @property
    def column_names(self) -> list[str]:
        """Union of the columns contributed by each dataset, in order."""
        cols = []
        for keys in self.keys_map:
            cols.extend(keys)
        return cols

    @property
    def lengths(self) -> np.ndarray:
        """Episode lengths, taken from the first dataset."""
        return self.datasets[0].lengths

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict:
        out = {}
        for ds, keys in zip(self.datasets, self.keys_map):
            item = ds[idx]
            for k in keys:
                if k in item:
                    out[k] = item[k]
        return out

    def load_chunk(
        self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> list[dict]:
        """Load slices from every dataset and merge them column-wise."""
        all_chunks = [
            ds.load_chunk(episodes_idx, start, end) for ds in self.datasets
        ]
        merged = []
        for items in zip(*all_chunks):
            combined = {}
            for item in items:
                combined.update(item)
            merged.append(combined)
        return merged

    def get_col_data(self, col: str) -> np.ndarray:
        """Return column ``col`` from the dataset that contributes it."""
        for ds, keys in zip(self.datasets, self.keys_map):
            if col in keys:
                return ds.get_col_data(col)
        raise KeyError(col)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        """Return the given row(s) with columns gathered from all datasets."""
        out = {}
        for ds, keys in zip(self.datasets, self.keys_map):
            data = ds.get_row_data(row_idx)
            for k in keys:
                if k in data:
                    out[k] = data[k]
        return out


class ConcatDataset:
    """Concatenate datasets sequentially (vertical join, more episodes).

    Args:
        datasets: Datasets to concatenate, in order. Episode and clip
            indices of later datasets are shifted past the earlier ones.
    """

    def __init__(self, datasets: list[Any]) -> None:
        if not datasets:
            raise ValueError('Need at least one dataset')
        self.datasets = datasets

        lengths = [len(ds) for ds in datasets]
        self._cum = np.cumsum([0] + lengths)

        ep_counts = [len(ds.lengths) for ds in datasets]
        self._ep_cum = np.cumsum([0] + ep_counts)

    @property
    def column_names(self) -> list[str]:
        """Union of all datasets' columns, first occurrence order."""
        seen = set()
        cols = []
        for ds in self.datasets:
            for c in ds.column_names:
                if c not in seen:
                    seen.add(c)
                    cols.append(c)
        return cols

    def __len__(self) -> int:
        return self._cum[-1]

    def _loc(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += len(self)
        ds_idx = int(np.searchsorted(self._cum[1:], idx, side='right'))
        local_idx = idx - self._cum[ds_idx]
        return ds_idx, local_idx

    def __getitem__(self, idx: int) -> dict:
        ds_idx, local_idx = self._loc(idx)
        return self.datasets[ds_idx][local_idx]

    def __getitems__(self, indices: list[int]) -> list[dict]:
        mapped = [self._loc(idx) for idx in indices]

        # Group by sub-dataset, preserving original positions.
        groups: dict[int, list[tuple[int, int]]] = {}
        for orig_pos, (ds_idx, local_idx) in enumerate(mapped):
            if ds_idx not in groups:
                groups[ds_idx] = []
            groups[ds_idx].append((orig_pos, local_idx))

        results: list[dict | None] = [None] * len(indices)
        for ds_idx, items in groups.items():
            ds = self.datasets[ds_idx]
            orig_positions = [pos for pos, _ in items]
            local_indices = [local_idx for _, local_idx in items]
            if hasattr(ds, '__getitems__'):
                fetched = ds.__getitems__(local_indices)
            else:
                fetched = [ds[i] for i in local_indices]
            for orig_pos, item in zip(orig_positions, fetched):
                results[orig_pos] = item

        return results  # type: ignore[return-value]

    def load_chunk(
        self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> list[dict]:
        """Route each slice to its dataset using global episode indices."""
        episodes_idx = np.asarray(episodes_idx)
        start = np.asarray(start)
        end = np.asarray(end)

        ds_indices = np.searchsorted(
            self._ep_cum[1:], episodes_idx, side='right'
        )
        local_eps = episodes_idx - self._ep_cum[ds_indices]

        results: list[dict | None] = [None] * len(episodes_idx)
        for ds_idx in range(len(self.datasets)):
            mask = ds_indices == ds_idx
            if not np.any(mask):
                continue
            chunks = self.datasets[ds_idx].load_chunk(
                local_eps[mask], start[mask], end[mask]
            )
            for i, chunk in zip(np.where(mask)[0], chunks):
                results[i] = chunk

        return results  # type: ignore[return-value]

    def get_col_data(self, col: str) -> np.ndarray:
        """Concatenate column ``col`` from every dataset that has it."""
        data = []
        for ds in self.datasets:
            if col in ds.column_names:
                data.append(ds.get_col_data(col))
        if not data:
            raise KeyError(col)
        return np.concatenate(data)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        """Return the given global row(s), stacked when a list is given."""
        if isinstance(row_idx, int):
            ds_idx, local_idx = self._loc(row_idx)
            return self.datasets[ds_idx].get_row_data(local_idx)

        results: dict[str, list[Any]] = {}
        for idx in row_idx:
            ds_idx, local_idx = self._loc(idx)
            row = self.datasets[ds_idx].get_row_data(local_idx)
            for k, v in row.items():
                if k not in results:
                    results[k] = []
                results[k].append(v)

        return {k: np.stack(v) for k, v in results.items()}


class GoalDataset:
    """Wrap any dataset to return a sampled goal observation per item.

    Goals are sampled from one of:
      - random state (uniform over all dataset steps)
      - geometric future state in same episode (Geom(1-gamma))
      - uniform future state in same episode
      - current state
    with probabilities (0.3, 0.5, 0.0, 0.2) by default.

    Args:
        dataset: The dataset to wrap.
        goal_probabilities: 4-tuple of probabilities (random,
            geometric_future, uniform_future, current); must sum to 1.
        gamma: Discount for the geometric future sampler; the future offset
            is drawn from ``Geom(1 - gamma)``.
        current_goal_offset: Step offset (in clip steps) defining the
            "current" state used as goal. Defaults to ``dataset.num_steps``.
        goal_keys: Mapping of source column to goal column (e.g.
            ``{'pixels': 'goal_pixels'}``). Defaults to ``pixels`` and
            ``proprio`` when present in the dataset.
        seed: Seed for the goal-sampling RNG.
    """

    def __init__(
        self,
        dataset: Dataset,
        goal_probabilities: tuple[float, float, float, float] = (
            0.3,
            0.5,
            0.0,
            0.2,
        ),
        gamma: float = 0.99,
        current_goal_offset: int | None = None,
        goal_keys: dict[str, str] | None = None,
        seed: int | None = None,
    ):
        self.dataset = dataset
        self.current_goal_offset = (
            current_goal_offset
            if current_goal_offset is not None
            else dataset.num_steps
        )

        if len(goal_probabilities) != 4:
            raise ValueError(
                'goal_probabilities must be a 4-tuple (random, geometric_future, uniform_future, current)'
            )
        if not np.isclose(sum(goal_probabilities), 1.0):
            raise ValueError('goal_probabilities must sum to 1.0')

        self.goal_probabilities = goal_probabilities
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        self.episode_lengths = dataset.lengths
        self.episode_offsets = dataset.offsets

        self._episode_cumlen = np.cumsum(self.episode_lengths)
        self._total_steps = (
            int(self._episode_cumlen[-1]) if len(self._episode_cumlen) else 0
        )

        if goal_keys is None:
            goal_keys = {}
            column_names = dataset.column_names
            if 'pixels' in column_names:
                goal_keys['pixels'] = 'goal_pixels'
            if 'proprio' in column_names:
                goal_keys['proprio'] = 'goal_proprio'
        self.goal_keys = goal_keys

        _, p_geometric_future, p_uniform_future, _ = goal_probabilities
        needs_future_filtering = p_geometric_future > 0 or p_uniform_future > 0

        clip_indices = np.asarray(dataset.clip_indices, dtype=np.int64)
        if needs_future_filtering:
            frameskip = dataset.frameskip
            current_end_offset = (self.current_goal_offset - 1) * frameskip

            ep_lens = np.asarray(self.episode_lengths, dtype=np.int64)
            current_end = clip_indices[:, 1] + current_end_offset
            keep = current_end + frameskip < ep_lens[clip_indices[:, 0]]
            self._clip_indices = clip_indices[keep]
            self._index_mapping = np.flatnonzero(keep)
        else:
            self._clip_indices = clip_indices
            self._index_mapping = np.arange(len(clip_indices), dtype=np.int64)

    @property
    def clip_indices(self):
        """Clip indices, filtered to clips that still have a future frame."""
        return self._clip_indices

    def __len__(self):
        return len(self._clip_indices)

    @property
    def column_names(self):
        """Columns of the wrapped dataset (goal columns are added per item)."""
        return self.dataset.column_names

    def _sample_goal_kind(self) -> str:
        r = self.rng.random()
        p_random, p_geometric_future, p_uniform_future, _ = (
            self.goal_probabilities
        )
        if r < p_random:
            return 'random'
        if r < p_random + p_geometric_future:
            return 'geometric_future'
        if r < p_random + p_geometric_future + p_uniform_future:
            return 'uniform_future'
        return 'current'

    def _sample_random_step(self) -> tuple[int, int]:
        if self._total_steps == 0:
            return 0, 0
        flat_idx = int(self.rng.integers(0, self._total_steps))
        ep_idx = int(
            np.searchsorted(self._episode_cumlen, flat_idx, side='right')
        )
        prev = self._episode_cumlen[ep_idx - 1] if ep_idx > 0 else 0
        local_idx = flat_idx - prev
        return ep_idx, local_idx

    def _sample_geometric_future_step(
        self, ep_idx: int, local_start: int
    ) -> tuple[int, int]:
        frameskip = self.dataset.frameskip
        current_end = local_start + (self.current_goal_offset - 1) * frameskip
        max_steps = (
            self.episode_lengths[ep_idx] - 1 - current_end
        ) // frameskip
        assert max_steps >= 1, f'No future frames available: {max_steps=}'

        p = max(1.0 - self.gamma, 1e-6)
        k = int(self.rng.geometric(p))
        k = min(k, max_steps)
        local_idx = current_end + k * frameskip
        return ep_idx, local_idx

    def _sample_uniform_future_step(
        self, ep_idx: int, local_start: int
    ) -> tuple[int, int]:
        frameskip = self.dataset.frameskip
        current_end = local_start + (self.current_goal_offset - 1) * frameskip
        max_steps = (
            self.episode_lengths[ep_idx] - 1 - current_end
        ) // frameskip
        assert max_steps >= 1, f'No future frames available: {max_steps=}'

        k = int(self.rng.integers(1, max_steps + 1))
        local_idx = current_end + k * frameskip
        return ep_idx, local_idx

    def _get_clip_info(self, idx: int) -> tuple[int, int]:
        ep_idx, start = self._clip_indices[idx]
        return int(ep_idx), int(start)

    def _load_single_step(
        self, ep_idx: int, local_idx: int
    ) -> dict[str, torch.Tensor]:
        return self.dataset._load_slice(ep_idx, local_idx, local_idx + 1)

    def __getitem__(self, idx: int):
        wrapped_idx = self._index_mapping[idx]
        steps = self.dataset[wrapped_idx]

        if not self.goal_keys:
            return steps

        ep_idx, local_start = self._get_clip_info(idx)

        goal_kind = self._sample_goal_kind()
        if goal_kind == 'random':
            goal_ep_idx, goal_local_idx = self._sample_random_step()
        elif goal_kind == 'geometric_future':
            goal_ep_idx, goal_local_idx = self._sample_geometric_future_step(
                ep_idx, local_start
            )
        elif goal_kind == 'uniform_future':
            goal_ep_idx, goal_local_idx = self._sample_uniform_future_step(
                ep_idx, local_start
            )
        else:
            frameskip = self.dataset.frameskip
            goal_local_idx = (
                local_start + (self.current_goal_offset - 1) * frameskip
            )
            goal_ep_idx = ep_idx

        goal_step = self._load_single_step(goal_ep_idx, goal_local_idx)

        for src_key, goal_key in self.goal_keys.items():
            if src_key not in goal_step or src_key not in steps:
                continue
            goal_val = goal_step[src_key]
            if goal_val.ndim == 0:
                goal_val = goal_val.unsqueeze(0)
            steps[goal_key] = goal_val

        return steps


__all__ = [
    'Dataset',
    'MergeDataset',
    'ConcatDataset',
    'GoalDataset',
]
