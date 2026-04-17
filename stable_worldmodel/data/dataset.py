"""Dataset classes for episode-based reinforcement learning data."""

import io
import logging
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from PIL import Image

from stable_worldmodel.data.utils import get_cache_dir


class Dataset:
    """Base class for episode-based datasets.

    Args:
        lengths: Array of episode lengths.
        offsets: Array of episode start offsets in the data.
        frameskip: Number of frames to skip between samples.
        num_steps: Number of steps per sample.
        transform: Optional transform to apply to loaded data.
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
        self.clip_indices = [
            (ep, start)
            for ep, length in enumerate(lengths)
            if length >= self.span
            for start in range(length - self.span + 1)
        ]

    @property
    def column_names(self) -> list[str]:
        raise NotImplementedError

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.clip_indices)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start = self.clip_indices[idx]
        steps = self._load_slice(ep_idx, start, start + self.span)
        if 'action' in steps:
            steps['action'] = steps['action'].reshape(self.num_steps, -1)
        return steps

    def load_chunk(
        self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> list[dict]:
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
        """Load full episode by index."""
        return self._load_slice(episode_idx, 0, self.lengths[episode_idx])

    def get_col_data(self, col: str) -> np.ndarray:
        raise NotImplementedError

    def get_dim(self, col: str) -> int:
        raise NotImplementedError

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        raise NotImplementedError

    def merge_col(
        self,
        source: list[str] | str,
        target: str,
        dim: int = -1,
    ) -> None:
        raise NotImplementedError


class HDF5Dataset(Dataset):
    """Dataset loading from HDF5 file.

    Reads data from a single .h5 file containing all episode data.
    Uses SWMR mode for robust reading while writing.

    Args:
        name: Name of the dataset (filename without extension), or a full path
            when ``file`` is also provided (used as label only).
        frameskip: Number of frames to skip between samples.
        num_steps: Number of steps per sample sequence.
        transform: Optional data transform callable.
        keys_to_load: Specific keys to load (defaults to all except metadata).
        keys_to_cache: Keys to load entirely into memory for faster access.
        cache_dir: Root cache directory. The file is expected at
            ``<cache_dir>/datasets/<name>.h5``. Ignored when ``file`` is given.
        file: Pre-opened file-like object accepted by :func:`h5py.File` (e.g.
            an ``s3fs`` file handle). When provided, ``cache_dir`` is ignored
            and SWMR is disabled (SWMR requires a local path, not a handle).
    """

    def __init__(
        self,
        name: str,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        keys_to_merge: dict[str, list[str] | str] | None = None,
        cache_dir: str | Path | None = None,
        file: Any | None = None,
    ) -> None:
        self._file_handle = file
        if file is not None:
            # name is informational; h5py will open via the handle.
            self.h5_path = Path(name)
        else:
            datasets_dir = get_cache_dir(cache_dir, sub_folder='datasets')
            self.h5_path = Path(datasets_dir, f'{name}.h5')
        self.h5_file: h5py.File | None = None
        self._cache: dict[str, np.ndarray] = {}

        _open_target: Any = self._file_handle if self._file_handle is not None else self.h5_path
        with h5py.File(_open_target, 'r') as f:
            lengths, offsets = f['ep_len'][:], f['ep_offset'][:]
            self._keys = keys_to_load or [
                k for k in f.keys() if k not in ('ep_len', 'ep_offset')
            ]

            for key in keys_to_cache or []:
                self._cache[key] = f[key][:]
                logging.info(f"Cached '{key}' from '{self.h5_path}'")

        super().__init__(lengths, offsets, frameskip, num_steps, transform)

        if keys_to_merge:
            for target, source in keys_to_merge.items():
                self.merge_col(source, target)

    @property
    def column_names(self) -> list[str]:
        return self._keys

    def _open(self) -> None:
        if self.h5_file is None:
            # SWMR requires a real filesystem path; disable it for file handles
            # (e.g. s3fs objects) which h5py accepts but doesn't support SWMR on.
            target: Any = self._file_handle if self._file_handle is not None else self.h5_path
            swmr = self._file_handle is None
            self.h5_file = h5py.File(
                target, 'r', swmr=swmr, rdcc_nbytes=256 * 1024 * 1024
            )

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        self._open()
        g_start, g_end = (
            self.offsets[ep_idx] + start,
            self.offsets[ep_idx] + end,
        )
        steps = {}
        for col in self._keys:
            src = self._cache if col in self._cache else self.h5_file
            data = src[col][g_start:g_end]
            if col != 'action':
                data = data[:: self.frameskip]

            if data.dtype == np.object_ or data.dtype.kind in ('S', 'U'):
                val = data[0] if len(data) > 0 else b''
                steps[col] = val.decode() if isinstance(val, bytes) else val
            else:
                steps[col] = torch.from_numpy(data)
                if data.ndim == 4 and data.shape[-1] in (1, 3):
                    steps[col] = steps[col].permute(0, 3, 1, 2)

        return self.transform(steps) if self.transform else steps

    def _get_col(self, col: str) -> np.ndarray:
        if col in self._cache:
            return self._cache[col]
        self._open()
        return self.h5_file[col][:]

    def get_col_data(self, col: str) -> np.ndarray:
        return self._get_col(col)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        self._open()
        return {col: self.h5_file[col][row_idx] for col in self._keys}

    def merge_col(
        self,
        source: list[str] | str,
        target: str,
        dim: int = -1,
    ) -> None:
        self._open()

        if isinstance(source, str):
            source = [k for k in self.h5_file.keys() if re.match(source, k)]

        merged = np.concatenate([self._get_col(s) for s in source], axis=dim)
        self._cache[target] = merged
        if target not in self._keys:
            self._keys.append(target)
        logging.info(f"Merged columns {source} into '{target}' and cached it")

    def get_dim(self, col: str) -> int:
        data = self.get_col_data(col)
        return np.prod(data.shape[1:]).item() if data.ndim > 1 else 1


class LanceDataset(Dataset):
    """Dataset backed by LanceDB tables.

    The table is expected to store flattened timesteps with two index columns:
    ``episode_idx`` and ``step_idx``. All other columns are treated as data.

    **Why ``keys_to_cache`` is not needed here (unlike HDF5Dataset)**

    :class:`HDF5Dataset` caches columns because HDF5 is a single compressed
    file: every random-access read must locate and decompress the relevant
    chunk, so pulling the whole column into RAM once pays off quickly.

    Lance has a fundamentally different access pattern:

    * Columns are stored in separate fragment files — a projection onto
      ``['action', 'proprio']`` never touches the pixel fragments at all.
    * :meth:`__getitems__` batches the entire batch's row IDs into **one**
      ``Permutation.take()`` call, so Lance already reads a large sequential
      slice from each fragment rather than many tiny random reads.
    * For remote storage (S3/GCS) this reduces round-trips from
      ``batch_size`` to ``1`` per batch — the dominant cost is transfer
      bandwidth, not seek latency.

    Loading entire columns into RAM via ``keys_to_cache`` therefore provides
    negligible throughput gain (<1 % measured on tworoom and pusht) while
    risking **OOM on large datasets** and slowing down startup with a full
    column scan.  Leave ``keys_to_cache`` empty (the default).

    Args:
        uri: LanceDB URI (local path, ``s3://`` bucket, ``hf://`` dataset, ...).
        table_name: Table to open inside the LanceDB database.
        frameskip: Number of raw rows to skip between returned steps.
        num_steps: Length of the returned temporal window.
        transform: Optional transform composed on top of the returned dict.
        keys_to_load: Explicit list of columns to read. Defaults to all
            non-index columns in the table.
        keys_to_cache: Columns to fully materialise into RAM.  **Not
            recommended** — see note above.  Accepted for API parity with
            :class:`HDF5Dataset` and for unusual cases (e.g. a tiny auxiliary
            column accessed millions of times outside the DataLoader hot path).
        keys_to_merge: Optional mapping of target -> source pattern(s) to
            concatenate and expose as a new cached column.
        image_columns: Columns storing encoded images (JPEG/PNG). These are
            decoded per frame and returned as ``(T, C, H, W)`` tensors similar
            to :class:`HDF5Dataset`.
        episode_index_column: Name of the column containing episode indices.
        step_index_column: Name of the column containing per-episode step ids.
        connect_kwargs: Extra kwargs forwarded to :func:`lancedb.connect`. This
            is how credentials for S3/object storage endpoints are supplied.
    """

    _lancedb = None
    _permutation = None
    _pa = None  # pyarrow; populated alongside _lancedb

    def __init__(
        self,
        uri: str,
        table_name: str,
        *,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        keys_to_merge: dict[str, list[str] | str] | None = None,
        image_columns: list[str] | None = None,
        episode_index_column: str = 'episode_idx',
        step_index_column: str = 'step_idx',
        connect_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.uri = uri
        self.table_name = table_name
        self.connect_kwargs = connect_kwargs or {}
        self._index_columns = (episode_index_column, step_index_column)
        self._cache: dict[str, np.ndarray] = {}
        self._perm = None
        self._fetch_columns: list[str] | None = None

        table = self._connect_table()
        self._schema_names = list(table.schema.names)
        available = [
            c for c in self._schema_names if c not in self._index_columns
        ]
        if not available:
            raise ValueError(
                'No data columns found in Lance table. Expected columns '
                'other than episode/step indices.'
            )

        self._keys = keys_to_load or available
        missing = [k for k in self._keys if k not in available]
        if missing:
            raise KeyError(
                f"Columns {missing} missing from Lance table '{table_name}'"
            )

        # Auto-detect image columns from naming convention when not specified:
        # 'pixels' (single camera) or 'pixels_<view>' (multi-camera).
        # Note: dots cannot be used as separators — Lance reserves '.' for
        # struct field access and rejects top-level names containing it.
        default_image_cols = [
            c for c in self._keys if c == 'pixels' or c.startswith('pixels_')
        ]
        requested_images = (
            image_columns if image_columns is not None else default_image_cols
        )
        self.image_columns = {
            col for col in requested_images if col in self._keys
        }

        lengths, offsets = self._compute_episode_structure(table)

        if keys_to_cache:
            logging.warning(
                "LanceDataset: keys_to_cache=%s was provided, but column caching "
                "is not recommended for Lance datasets. __getitems__ already batches "
                "all row fetches into a single take() call, so caching provides "
                "<1%% throughput gain while risking OOM on large datasets. "
                "Remove keys_to_cache from your config to use Lance efficiently.",
                keys_to_cache,
            )
            for key in keys_to_cache:
                if key not in self._keys:
                    raise KeyError(f"Cannot cache missing column '{key}'")
                self._cache[key] = self._load_full_column(table, key)

        self._update_fetch_columns()

        super().__init__(lengths, offsets, frameskip, num_steps, transform)

        if keys_to_merge:
            for target, source in keys_to_merge.items():
                self.merge_col(source, target)

    @property
    def column_names(self) -> list[str]:
        return list(self._keys)

    def __getstate__(self) -> dict:
        state = super().__dict__.copy()
        state['_perm'] = None
        return state

    @classmethod
    def _import_lance(cls) -> None:
        if cls._lancedb is not None:
            return
        try:
            import lancedb
            import pyarrow as pa
            from lancedb.permutation import Permutation
        except ImportError as exc:  # pragma: no cover - exercised in runtime
            raise ImportError(
                'LanceDataset requires the "lancedb" package. '
                'Install with `pip install lancedb` or '
                '`pip install stable-worldmodel[lance]`.'
            ) from exc
        cls._lancedb = lancedb
        cls._permutation = Permutation
        cls._pa = pa

    def _connect_table(self):
        self._import_lance()
        db = self._lancedb.connect(self.uri, **self.connect_kwargs)
        return db.open_table(self.table_name)

    def _compute_episode_structure(self, table) -> tuple[np.ndarray, np.ndarray]:
        """Read episode_idx column and build per-episode (lengths, offsets) arrays.

        **Ordering assumption**: rows must be stored in episode-contiguous order,
        i.e. all rows for episode 0 appear before episode 1, etc.  This is
        guaranteed by :func:`convert_hdf5_to_lance` which writes episodes
        sequentially.

        Note: ``lancedb.compact_files()`` preserves logical row ordering (it
        merges fragments in fragment-ID order), so compaction is safe.  The
        check below catches tables where data was inserted out of order to
        begin with.

        **No memory risk**: only the episode_idx column (int32, ~4 bytes/row)
        is read here; even a 10 M-row dataset needs only ~40 MB for this pass.
        """
        ep_col, _ = self._index_columns
        reader = (
            table.to_lance().scanner(columns=[ep_col]).to_reader()
        )

        chunks: list[np.ndarray] = []
        for batch in reader:
            chunks.append(
                batch.column(batch.schema.get_field_index(ep_col)).to_numpy()
            )

        if not chunks:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        all_ep_ids = np.concatenate(chunks)

        # Verify rows are in episode-contiguous (non-decreasing) order.
        # We check for *decreasing* transitions which indicate a reordering.
        if len(all_ep_ids) > 1 and (np.diff(all_ep_ids) < 0).any():
            raise ValueError(
                f"Lance table '{self.table_name}' at '{self.uri}' has rows that "
                "are NOT in episode-contiguous order (episode_idx is not "
                "non-decreasing).  This likely means data was appended out of "
                "order.  Re-run convert_hdf5_to_lance(..., overwrite=True) to "
                "rebuild the table with the correct row order."
            )

        # Vectorised boundary detection — O(N) numpy, no Python loop over rows.
        boundary_mask = np.diff(all_ep_ids) != 0
        change_positions = np.flatnonzero(boundary_mask) + 1  # first row of each new ep
        offsets = np.concatenate([[0], change_positions]).astype(np.int64)
        lengths = np.diff(np.concatenate([offsets, [len(all_ep_ids)]])).astype(np.int64)

        return lengths, offsets

    def _load_full_column(self, table, key: str) -> np.ndarray:
        data: list[np.ndarray | np.generic | list[Any] | bytes] = []
        reader = table.to_lance().scanner(columns=[key]).to_reader()
        for batch in reader:
            values = self._batch_column_pylist(batch, key)
            if not values:
                continue
            data.append(self._pylist_to_numpy(values, key))

        if not data:
            return np.array([], dtype=np.float32)

        return np.concatenate(data, axis=0)

    def _update_fetch_columns(self) -> None:
        cached = set(self._cache.keys())
        self._fetch_columns = [k for k in self._keys if k not in cached]
        if not self._fetch_columns:
            self._perm = None

    def _ensure_open(self) -> None:
        if not self._fetch_columns:
            return
        if self._perm is None:
            table = self._connect_table()
            self._perm = (
                self._permutation.identity(table)
                .select_columns(self._fetch_columns)
                .with_format('arrow')
            )

    def _fetch_rows(self, rows: list[int]):
        if not self._fetch_columns:
            return None
        self._ensure_open()
        return self._perm.__getitems__(rows)

    def _batch_column_pylist(self, batch, key: str) -> list[Any]:
        idx = batch.schema.get_field_index(key)
        if idx == -1:
            raise KeyError(f"Column '{key}' not found in batch")
        return batch.column(idx).to_pylist()

    def _extract_column(self, batch, key: str):
        """Fast column extraction from an Arrow RecordBatch.

        Avoids the Python-list roundtrip that :meth:`_batch_column_pylist`
        incurs for numeric columns stored as fixed-size lists.

        Returns
        -------
        numpy ndarray
            For fixed-size list (numeric vectors) and scalar numeric columns.
        list
            For binary columns (JPEG/PNG blobs), string columns, and
            variable-length list columns (schema inferred from Python dicts
            or LeRobot-style datasets).
        """
        pa = self._pa
        col_idx = batch.schema.get_field_index(key)
        if col_idx == -1:
            raise KeyError(f"Column '{key}' not found in batch")
        col = batch.column(col_idx)
        col_type = col.type

        if pa.types.is_binary(col_type) or pa.types.is_large_binary(col_type):
            # Image blobs — pylist gives bytes objects required for JPEG decode.
            return col.to_pylist()

        if pa.types.is_fixed_size_list(col_type):
            # Numeric vector columns (e.g. action, proprio stored as
            # pa.list_(pa.float32(), dim)).  Flatten to a contiguous float32
            # buffer and reshape — no Python object creation at all.
            dim = col_type.list_size
            flat = col.flatten()  # pa.Array of length N * dim
            return flat.to_numpy(zero_copy_only=False).reshape(len(col), dim)

        if pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
            return col.to_pylist()

        if pa.types.is_list(col_type):
            # Variable-length list column (e.g. schema inferred from Python dicts,
            # or LeRobot datasets). Fall back to pylist; _pylist_to_numpy will
            # stack into a float32 array.
            return col.to_pylist()

        # Scalar numeric (int, float) — zero-copy numpy view when possible.
        return col.to_numpy(zero_copy_only=False)

    def _pylist_to_numpy(self, values: list[Any], key: str) -> np.ndarray:
        if not values:
            return np.array([], dtype=np.float32)
        first = values[0]

        if isinstance(first, (bytes, bytearray, memoryview)):
            return np.asarray(values, dtype=object)
        if isinstance(first, str):
            return np.asarray(values, dtype=object)
        if isinstance(first, (list, tuple)):
            return np.asarray(values, dtype=np.float32)
        if isinstance(first, (int, float, np.integer, np.floating)):
            dtype = np.float32 if isinstance(first, (float, np.floating)) else np.int64
            return np.asarray(values, dtype=dtype)
        if isinstance(first, np.ndarray):
            return np.stack(values)

        logging.warning(
            f"Column '{key}' produced unrecognized type {type(first)}; "
            'falling back to object array.'
        )
        return np.asarray(values, dtype=object)

    def _decode_image(self, blob: bytes) -> torch.Tensor:
        with Image.open(io.BytesIO(blob)) as img:
            arr = np.array(img.convert('RGB'))
        # from_numpy is fine here: the result is immediately consumed by
        # torch.stack() in _process_batch, which produces a fresh resizable
        # tensor — the non-resizable frame tensors never reach collation.
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _prepare_numeric_tensor(self, data: np.ndarray, downsample: bool) -> torch.Tensor:
        if downsample:
            data = data[:: self.frameskip]
        # torch.tensor() copies into PyTorch-owned resizable storage.
        # torch.from_numpy() would share numpy's memory and return a
        # non-resizable tensor, which breaks DataLoader collation when
        # __getitems__ is defined (collation runs inside the worker and
        # uses storage.resize_() to preallocate the output batch tensor).
        tensor = torch.tensor(data)
        if tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(0, 3, 1, 2)
        return tensor

    def _process_batch(self, ep_idx: int, g_start: int, batch) -> dict:
        """Assemble one training-sample dict from a pre-fetched Arrow sub-batch.

        This is the inner loop shared by :meth:`_load_slice` (single sample)
        and :meth:`__getitems__` (batch of samples).  It never issues any
        Lance / network I/O — all fetching must be done by the caller.

        Parameters
        ----------
        ep_idx:
            Episode index (used only for cache addressing).
        g_start:
            Global row offset into ``self._cache`` arrays
            (``offsets[ep_idx] + local_start``).
        batch:
            Arrow ``RecordBatch`` containing exactly ``self.span`` rows for the
            non-cached columns, or ``None`` when every column is cached.
        """
        g_end = g_start + self.span
        steps: dict[str, Any] = {}

        for col in self._keys:
            if col in self._cache:
                values = self._cache[col][g_start:g_end]
            else:
                if batch is None:
                    raise KeyError(
                        f"Column '{col}' is not in the cache and no Arrow batch "
                        "was provided — this is a bug."
                    )
                # Fast path: avoids Python list roundtrip for numeric columns.
                values = self._extract_column(batch, col)

            if col in self.image_columns:
                # Decode JPEG/PNG blobs; downsample in the temporal axis.
                if isinstance(values, np.ndarray):
                    blobs = values[:: self.frameskip].tolist()
                else:
                    blobs = values[:: self.frameskip]
                frames = [self._decode_image(v) for v in blobs]
                steps[col] = (
                    torch.stack(frames)
                    if frames
                    else torch.empty(0, dtype=torch.uint8)
                )
                continue

            if isinstance(values, np.ndarray):
                # No .copy() needed: _prepare_numeric_tensor uses torch.tensor()
                # which always copies, so there is no aliasing risk with the cache.
                data = values
            else:
                data = self._pylist_to_numpy(values, col)

            if data.dtype == object and data.size > 0:
                first = data.flat[0]
                if isinstance(first, (bytes, bytearray)):
                    steps[col] = first.decode() if isinstance(first, bytes) else first
                    continue
                if isinstance(first, str):
                    steps[col] = first
                    continue

            downsample = col != 'action'
            steps[col] = self._prepare_numeric_tensor(data, downsample)

        return steps

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        g_start = int(self.offsets[ep_idx] + start)
        rows = list(range(g_start, g_start + (end - start)))
        batch = self._fetch_rows(rows)
        steps = self._process_batch(ep_idx, g_start, batch)
        return self.transform(steps) if self.transform else steps

    def __getitems__(self, indices: list[int]) -> list[dict]:
        """Batch-level fetch: one Lance ``take`` for the whole batch.

        PyTorch DataLoader (≥ 2.0) calls this automatically when the method is
        defined, passing all sample indices for the batch at once.  Instead of
        N individual ``Permutation.__getitems__`` calls (one per sample) this
        issues a **single** call with all row IDs flattened together, then
        slices the returned ``RecordBatch`` with zero-copy
        ``RecordBatch.slice(offset, length)`` per sample.

        For local storage the win is modest (one Python call vs. N).  For
        remote storage (S3, GCS) the win is large: one HTTP round-trip fetches
        the data for the entire batch rather than one round-trip per sample.
        """
        # ── 1. Collect all global row IDs for the batch ──────────────────────
        all_rows: list[int] = []
        row_offsets: list[int] = []  # position of each sample's first row in all_rows
        sample_meta: list[tuple[int, int]] = []  # (ep_idx, g_start)

        for idx in indices:
            ep_idx, start = self.clip_indices[idx]
            g_start = int(self.offsets[ep_idx] + start)
            row_offsets.append(len(all_rows))
            all_rows.extend(range(g_start, g_start + self.span))
            sample_meta.append((ep_idx, g_start))

        # ── 2. Single Lance take for all non-cached columns ──────────────────
        big_batch = None
        if self._fetch_columns and all_rows:
            self._ensure_open()

            # ``Permutation.__getitems__`` returns rows in strictly
            # increasing order and drops duplicates.  Build a sorted unique
            # row list and map back to the original order so we can
            # materialize the overlapping windows exactly as requested.
            unique_rows = sorted(set(all_rows))
            unique_batch = self._perm.__getitems__(unique_rows)

            if len(unique_rows) == len(all_rows) and all_rows == unique_rows:
                big_batch = unique_batch
            else:
                row_lookup = {row: idx for idx, row in enumerate(unique_rows)}
                gather = [row_lookup[row] for row in all_rows]
                pa = self._pa
                indices_arr = pa.array(gather, type=pa.int64())
                big_batch = unique_batch.take(indices_arr)

        # ── 3. Slice + assemble each sample from the big batch ───────────────
        results: list[dict] = []
        for i, (ep_idx, g_start) in enumerate(sample_meta):
            # RecordBatch.slice is zero-copy — no data is copied here.
            sub_batch = (
                big_batch.slice(row_offsets[i], self.span)
                if big_batch is not None
                else None
            )
            steps = self._process_batch(ep_idx, g_start, sub_batch)
            if self.transform:
                steps = self.transform(steps)
            if 'action' in steps:
                steps['action'] = steps['action'].reshape(self.num_steps, -1)
            results.append(steps)

        return results

    def get_col_data(self, col: str) -> np.ndarray:
        if col in self._cache:
            return self._cache[col]
        table = self._connect_table()
        data = self._load_full_column(table, col)
        self._cache[col] = data
        self._update_fetch_columns()
        return data

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        if isinstance(row_idx, (list, tuple, np.ndarray)):
            idxs = [int(i) for i in row_idx]
        else:
            idxs = [int(row_idx)]
        batch = self._fetch_rows(idxs)
        out: dict[str, Any] = {}
        for col in self._keys:
            if col in self._cache:
                values = self._cache[col][idxs]
            else:
                if batch is None:
                    raise KeyError(
                        f"Column '{col}' missing from cached columns and fetch columns"
                    )
                values = self._batch_column_pylist(batch, col)
            if isinstance(values, np.ndarray):
                arr = values
            else:
                arr = self._pylist_to_numpy(values, col)
            out[col] = arr
        return out

    def merge_col(
        self,
        source: list[str] | str,
        target: str,
        dim: int = -1,
    ) -> None:
        if isinstance(source, str):
            pattern = re.compile(source)
            cols = [k for k in self._keys if pattern.match(k)]
        else:
            cols = source
        merged = np.concatenate([self.get_col_data(c) for c in cols], axis=dim)
        self._cache[target] = merged
        if target not in self._keys:
            self._keys.append(target)
        self._update_fetch_columns()

    def get_dim(self, col: str) -> int:
        data = self.get_col_data(col)
        return np.prod(data.shape[1:]).item() if data.ndim > 1 else 1

class FolderDataset(Dataset):
    """Dataset loading from folder structure.

    Metadata is stored in .npz files, heavy media (images) can be stored as individual files.

    Args:
        name: Name of the dataset folder.
        frameskip: Number of frames to skip.
        num_steps: Sequence length.
        transform: Optional transform.
        keys_to_load: Specific keys to load.
        folder_keys: Keys that correspond to folders of image files.
        cache_dir: Base directory containing the dataset folder.
    """

    def __init__(
        self,
        name: str,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
        keys_to_load: list[str] | None = None,
        folder_keys: list[str] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.path = (
            Path(cache_dir or get_cache_dir(sub_folder='datasets')) / name
        )
        self.folder_keys = folder_keys or []
        self._cache: dict[str, np.ndarray] = {}

        lengths = np.load(self.path / 'ep_len.npz')['arr_0']
        offsets = np.load(self.path / 'ep_offset.npz')['arr_0']

        if keys_to_load is None:
            keys_to_load = sorted(
                p.stem if p.suffix == '.npz' else p.name
                for p in self.path.iterdir()
                if p.stem not in ('ep_len', 'ep_offset')
            )
        self._keys = keys_to_load

        for key in self._keys:
            if key not in self.folder_keys:
                npz = self.path / f'{key}.npz'
                if npz.exists():
                    self._cache[key] = np.load(npz)['arr_0']
                    logging.info(f"Cached '{key}' from '{npz}'")

        super().__init__(lengths, offsets, frameskip, num_steps, transform)

    @property
    def column_names(self) -> list[str]:
        return self._keys

    def _load_file(self, ep_idx: int, step: int, key: str) -> np.ndarray:
        path = self.path / key / f'ep_{ep_idx}_step_{step}'
        img_path = path.with_suffix('.jpeg')
        if not img_path.exists():
            img_path = path.with_suffix('.jpg')
        return np.array(Image.open(img_path))

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        g_start, g_end = (
            self.offsets[ep_idx] + start,
            self.offsets[ep_idx] + end,
        )
        steps = {}
        for col in self._keys:
            if col in self.folder_keys:
                data = np.stack(
                    [
                        self._load_file(ep_idx, s, col)
                        for s in range(start, end, self.frameskip)
                    ]
                )
            else:
                data = self._cache[col][g_start:g_end]
                if col != 'action':
                    data = data[:: self.frameskip]

            if data.dtype == np.object_ or data.dtype.kind in ('S', 'U'):
                val = data[0] if len(data) > 0 else b''
                steps[col] = val.decode() if isinstance(val, bytes) else val
            else:
                steps[col] = torch.from_numpy(data)
                if data.ndim == 4 and data.shape[-1] in (1, 3):
                    steps[col] = steps[col].permute(0, 3, 1, 2)
        return self.transform(steps) if self.transform else steps

    def get_col_data(self, col: str) -> np.ndarray:
        if col not in self._cache:
            raise KeyError(
                f"'{col}' not in cache (folder keys cannot be retrieved as full array)"
            )
        return self._cache[col]

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        return {
            c: self._cache[c][row_idx] for c in self._keys if c in self._cache
        }


class ImageDataset(FolderDataset):
    """Convenience alias for FolderDataset with image defaults.

    Assumes 'pixels' is a folder of images.
    """

    def __init__(
        self, name: str, image_keys: list[str] | None = None, **kw: Any
    ) -> None:
        super().__init__(name, folder_keys=image_keys or ['pixels'], **kw)


class VideoDataset(FolderDataset):
    """Dataset loading video frames from MP4 files using decord.

    Assumes video files are stored in a folder structure.
    """

    _decord: Any = None  # Lazy-loaded module reference

    def __init__(
        self, name: str, video_keys: list[str] | None = None, **kw: Any
    ) -> None:
        if VideoDataset._decord is None:
            try:
                import decord

                decord.bridge.set_bridge('torch')
                VideoDataset._decord = decord
            except ImportError:
                raise ImportError('VideoDataset requires decord')
        super().__init__(name, folder_keys=video_keys or ['video'], **kw)

    @lru_cache(maxsize=8)
    def _reader(self, ep_idx: int, key: str) -> Any:
        return VideoDataset._decord.VideoReader(
            str(self.path / key / f'ep_{ep_idx}.mp4'), num_threads=1
        )

    def _load_file(self, ep_idx: int, step: int, key: str) -> np.ndarray:
        return self._reader(ep_idx, key)[step].numpy()

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        g_start, g_end = (
            self.offsets[ep_idx] + start,
            self.offsets[ep_idx] + end,
        )
        steps = {}
        for col in self._keys:
            if col in self.folder_keys:
                # Decord efficient batch loading
                frames = self._reader(ep_idx, col).get_batch(
                    list(range(start, end, self.frameskip))
                )
                steps[col] = frames.permute(0, 3, 1, 2)
            else:
                data = self._cache[col][g_start:g_end]
                if col != 'action':
                    data = data[:: self.frameskip]

                if data.dtype == np.object_ or data.dtype.kind in ('S', 'U'):
                    val = data[0] if len(data) > 0 else b''
                    steps[col] = (
                        val.decode() if isinstance(val, bytes) else val
                    )
                else:
                    steps[col] = torch.from_numpy(data)

        return self.transform(steps) if self.transform else steps


class MergeDataset:
    """Merges multiple datasets of same length (horizontal join).

    Combines columns from different datasets (e.g. one dataset has 'pixels',
    another has 'rewards') into a single view.

    Args:
        datasets: List of dataset instances to merge.
        keys_from_dataset: Optional list of keys to take from each dataset.
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
            # Auto-deduplicate: each dataset provides keys not seen in previous datasets
            seen: set[str] = set()
            self.keys_map = []
            for ds in datasets:
                keys = [c for c in ds.column_names if c not in seen]
                seen.update(keys)
                self.keys_map.append(keys)

    @property
    def column_names(self) -> list[str]:
        cols = []
        for keys in self.keys_map:
            cols.extend(keys)
        return cols

    @property
    def lengths(self) -> np.ndarray:
        """Episode lengths from first dataset (all merged datasets share same structure)."""
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
        for ds, keys in zip(self.datasets, self.keys_map):
            if col in keys:
                return ds.get_col_data(col)
        raise KeyError(col)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        out = {}
        for ds, keys in zip(self.datasets, self.keys_map):
            data = ds.get_row_data(row_idx)
            for k in keys:
                if k in data:
                    out[k] = data[k]
        return out


class ConcatDataset:
    """Concatenates multiple datasets (vertical join).

    Combines datasets sequentially to increase the total number of episodes/samples.

    Args:
        datasets: List of datasets to concatenate.
    """

    def __init__(self, datasets: list[Any]) -> None:
        if not datasets:
            raise ValueError('Need at least one dataset')
        self.datasets = datasets

        # Cumulative lengths for index mapping: [0, len(ds0), len(ds0)+len(ds1), ...]
        lengths = [len(ds) for ds in datasets]
        self._cum = np.cumsum([0] + lengths)

        # Cumulative episode counts for load_chunk mapping
        ep_counts = [len(ds.lengths) for ds in datasets]
        self._ep_cum = np.cumsum([0] + ep_counts)

    @property
    def column_names(self) -> list[str]:
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
        """Map global index to (dataset_index, local_index)."""
        if idx < 0:
            idx += len(self)
        ds_idx = int(np.searchsorted(self._cum[1:], idx, side='right'))
        local_idx = idx - self._cum[ds_idx]
        return ds_idx, local_idx

    def __getitem__(self, idx: int) -> dict:
        ds_idx, local_idx = self._loc(idx)
        return self.datasets[ds_idx][local_idx]

    def load_chunk(
        self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> list[dict]:
        episodes_idx = np.asarray(episodes_idx)
        start = np.asarray(start)
        end = np.asarray(end)

        # Map global episode indices to dataset indices
        ds_indices = np.searchsorted(
            self._ep_cum[1:], episodes_idx, side='right'
        )
        local_eps = episodes_idx - self._ep_cum[ds_indices]

        # Group by dataset and collect results
        results: list[dict | None] = [None] * len(episodes_idx)
        for ds_idx in range(len(self.datasets)):
            mask = ds_indices == ds_idx
            if not np.any(mask):
                continue

            chunks = self.datasets[ds_idx].load_chunk(
                local_eps[mask], start[mask], end[mask]
            )

            # Place results back in original order
            for i, chunk in zip(np.where(mask)[0], chunks):
                results[i] = chunk

        return results  # type: ignore[return-value]

    def get_col_data(self, col: str) -> np.ndarray:
        data = []
        for ds in self.datasets:
            if col in ds.column_names:
                data.append(ds.get_col_data(col))
        if not data:
            raise KeyError(col)
        return np.concatenate(data)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        if isinstance(row_idx, int):
            ds_idx, local_idx = self._loc(row_idx)
            return self.datasets[ds_idx].get_row_data(local_idx)

        # Multiple indices: collect and stack results
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
    """
    Dataset wrapper that samples an additional goal observation per item.

    Works with any dataset type (HDF5Dataset, FolderDataset, VideoDataset, etc.)

    Goals are sampled from:
      - random state (uniform over dataset steps)
      - geometric future state in same episode (Geom(1-gamma))
      - uniform future state in same episode (uniform over future steps)
      - current state
    with probabilities (0.3, 0.5, 0.0, 0.2) by default.
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
        """
        Args:
            dataset: Base dataset to wrap.
            goal_probabilities: Tuple of (p_random, p_geometric_future, p_uniform_future, p_current) for goal sampling.
            gamma: Discount factor for geometric future goal sampling.
            current_goal_offset: Number of frames from clip start for "current" goal sampling.
                If None, defaults to num_steps, i.e., last frame of clip.
                When training with history, set this to history_size so "current" means last frame of history.
            goal_keys: Mapping of source observation keys to goal observation keys. If None, defaults to {"pixels": "goal", "proprio": "goal_proprio"}.
            seed: Random seed for goal sampling.
        """
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

        # All Dataset subclasses have lengths and offsets
        self.episode_lengths = dataset.lengths
        self.episode_offsets = dataset.offsets

        self._episode_cumlen = np.cumsum(self.episode_lengths)
        self._total_steps = (
            int(self._episode_cumlen[-1]) if len(self._episode_cumlen) else 0
        )

        # Auto-detect goal keys if not provided
        if goal_keys is None:
            goal_keys = {}
            column_names = dataset.column_names
            if 'pixels' in column_names:
                goal_keys['pixels'] = 'goal_pixels'
            if 'proprio' in column_names:
                goal_keys['proprio'] = 'goal_proprio'
        self.goal_keys = goal_keys

        # Build clip_indices with stricter constraint to ensure at least one future frame
        # for geometric/uniform future goal sampling (only if these modes are used)
        _, p_geometric_future, p_uniform_future, _ = goal_probabilities
        needs_future_filtering = p_geometric_future > 0 or p_uniform_future > 0

        if needs_future_filtering:
            frameskip = dataset.frameskip
            current_end_offset = (self.current_goal_offset - 1) * frameskip

            self._clip_indices = []
            self._index_mapping = []  # Maps our indices to wrapped dataset indices

            for wrapped_idx, (ep, start) in enumerate(dataset.clip_indices):
                current_end = start + current_end_offset
                # Need at least one frame after current_end (i.e., current_end + frameskip < length)
                if current_end + frameskip < self.episode_lengths[ep]:
                    self._clip_indices.append((ep, start))
                    self._index_mapping.append(wrapped_idx)
        else:
            # No future goal sampling, use wrapped dataset's indices directly
            self._clip_indices = list(dataset.clip_indices)
            self._index_mapping = list(range(len(dataset.clip_indices)))

    @property
    def clip_indices(self):
        """Clip indices filtered to ensure at least one future frame is available."""
        return self._clip_indices

    def __len__(self):
        return len(self._clip_indices)

    @property
    def column_names(self):
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
        """Sample random (ep_idx, local_idx) from entire dataset."""
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
        """Sample future (ep_idx, local_idx) from same episode using geometric distribution."""
        frameskip = self.dataset.frameskip
        # The minimum goal index should be the last frame of the history (current state)
        current_end = local_start + (self.current_goal_offset - 1) * frameskip
        max_steps = (
            self.episode_lengths[ep_idx] - 1 - current_end
        ) // frameskip
        # clip_indices filtering guarantees max_steps >= 1
        assert max_steps >= 1, f'No future frames available: {max_steps=}'

        p = max(1.0 - self.gamma, 1e-6)
        k = int(self.rng.geometric(p))
        k = min(k, max_steps)
        local_idx = current_end + k * frameskip
        return ep_idx, local_idx

    def _sample_uniform_future_step(
        self, ep_idx: int, local_start: int
    ) -> tuple[int, int]:
        """Sample future (ep_idx, local_idx) from same episode using uniform distribution."""
        frameskip = self.dataset.frameskip
        # The minimum goal index should be the last frame of the history (current state)
        current_end = local_start + (self.current_goal_offset - 1) * frameskip
        max_steps = (
            self.episode_lengths[ep_idx] - 1 - current_end
        ) // frameskip
        # clip_indices filtering guarantees max_steps >= 1
        assert max_steps >= 1, f'No future frames available: {max_steps=}'

        k = int(self.rng.integers(1, max_steps + 1))
        local_idx = current_end + k * frameskip
        return ep_idx, local_idx

    def _get_clip_info(self, idx: int) -> tuple[int, int]:
        """Returns (episode_idx, local_start) for a given GoalDataset index."""
        return self._clip_indices[idx]

    def _load_single_step(
        self, ep_idx: int, local_idx: int
    ) -> dict[str, torch.Tensor]:
        """Load a single step from episode ep_idx at local index local_idx."""
        return self.dataset._load_slice(ep_idx, local_idx, local_idx + 1)

    def __getitem__(self, idx: int):
        # Get base sample from wrapped dataset using index mapping
        wrapped_idx = self._index_mapping[idx]
        steps = self.dataset[wrapped_idx]

        if not self.goal_keys:
            return steps

        # Get episode and local start for this index
        ep_idx, local_start = self._get_clip_info(idx)

        # Sample goal (transform will be applied via underlying dataset's load_chunk/load_slice)
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
        else:  # current
            # Use current_goal_offset to determine the "current" frame
            frameskip = self.dataset.frameskip
            goal_local_idx = (
                local_start + (self.current_goal_offset - 1) * frameskip
            )
            goal_ep_idx = ep_idx

        # Load goal step
        goal_step = self._load_single_step(goal_ep_idx, goal_local_idx)

        # Add goal observations to steps
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
    'HDF5Dataset',
    'LanceDataset',
    'FolderDataset',
    'ImageDataset',
    'VideoDataset',
    'MergeDataset',
    'ConcatDataset',
    'GoalDataset',
]
