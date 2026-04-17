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
    """Dataset backed by a LanceDB table of flattened timesteps.

    The table carries two index columns (``episode_idx``, ``step_idx``); rows
    must be episode-contiguous (``convert_hdf5_to_lance`` guarantees this).

    On Linux, construction forces the multiprocessing start method to
    ``spawn`` (Lance's tokio pool is not fork-safe).  See
    ``docs/lance_implementation.md`` for the full design notes, including
    why ``keys_to_cache`` is normally unnecessary here.

    Args:
        uri: LanceDB database URI, or a full ``.../foo.lance`` path (table
            name is then inferred from the ``.lance`` stem).
        table_name: Table inside the database; optional when ``uri`` ends
            in ``.lance``.
        frameskip, num_steps, transform, keys_to_load, keys_to_cache,
            keys_to_merge: mirror :class:`HDF5Dataset`.
        image_columns: optional override — every ``pa.binary`` /
            ``pa.large_binary`` column is treated as an encoded image by
            default.
        episode_index_column, step_index_column: index column names.
        connect_kwargs: forwarded to :func:`lancedb.connect` (e.g. S3 creds).
    """

    _lancedb = None
    _permutation = None
    _pa = None  # pyarrow; populated alongside _lancedb
    _fork_warning_emitted = False

    def __init__(
        self,
        uri: str,
        table_name: str | None = None,
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
        uri, table_name = self._resolve_uri_and_table(uri, table_name)
        self.uri = uri
        self.table_name = table_name
        self.connect_kwargs = connect_kwargs or {}
        self._index_columns = (episode_index_column, step_index_column)
        self._cache: dict[str, np.ndarray] = {}
        self._perm = None
        self._fetch_columns: list[str] | None = None

        self._maybe_warn_fork_start_method()
        table = self._connect_table()
        self._schema_names = list(table.schema.names)
        available = [c for c in self._schema_names if c not in self._index_columns]
        if not available:
            raise ValueError('Lance table has no data columns (only index columns).')

        self._keys = keys_to_load or available
        missing = [k for k in self._keys if k not in available]
        if missing:
            raise KeyError(f"Columns {missing} missing from Lance table '{table_name}'")

        # Binary columns → encoded image blobs (what the converter emits).
        pa = self._pa
        binary_cols = {
            f.name for f in table.schema
            if pa.types.is_binary(f.type) or pa.types.is_large_binary(f.type)
        }
        self.image_columns = (
            binary_cols & set(self._keys) if image_columns is None
            else {c for c in image_columns if c in self._keys}
        )

        lengths, offsets = self._compute_episode_structure(table)

        if keys_to_cache:
            logging.warning(
                'LanceDataset: keys_to_cache=%s is unnecessary — __getitems__ '
                'already batches reads; caching risks OOM. Drop it from your config.',
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
        # _perm wraps an unpicklable Rust reader; workers reopen on first use.
        state = self.__dict__.copy()
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

    @staticmethod
    def _resolve_uri_and_table(
        uri: str, table_name: str | None
    ) -> tuple[str, str]:
        """Split ``.../foo.lance`` into ``(parent_uri, foo)`` when ``table_name`` is None."""
        if table_name is not None:
            return uri, table_name

        stripped = uri.rstrip('/')
        if not stripped.lower().endswith('.lance'):
            raise ValueError(
                "LanceDataset: ``table_name`` was not provided and ``uri`` "
                f"({uri!r}) does not end in '.lance'. Either pass "
                "``table_name`` explicitly, or point ``uri`` at the full "
                "table path, e.g. './store/foo.lance'."
            )
        sep = stripped.rfind('/')
        parent, leaf = (stripped[:sep], stripped[sep + 1:]) if sep >= 0 else ('.', stripped)
        inferred_name = leaf[: -len('.lance')]
        if not inferred_name:
            raise ValueError(f'LanceDataset: could not infer table name from uri {uri!r}.')
        return parent, inferred_name

    @classmethod
    def _maybe_warn_fork_start_method(cls) -> None:
        """On Linux, switch ``fork`` → ``spawn`` (Lance is not fork-safe)."""
        if cls._fork_warning_emitted:
            return
        import multiprocessing as mp
        import sys

        cls._fork_warning_emitted = True
        if sys.platform != 'linux':
            return
        current = mp.get_start_method(allow_none=True)
        if current not in (None, 'fork'):
            return
        try:
            mp.set_start_method('spawn', force=True)
            logging.info(
                "LanceDataset: multiprocessing start method set to 'spawn' (was %s).",
                current or 'default (fork)',
            )
        except RuntimeError as exc:
            logging.warning(
                "LanceDataset could not switch multiprocessing to 'spawn' (%s); "
                'DataLoader workers may deadlock. Set it explicitly at startup.',
                exc,
            )

    def _compute_episode_structure(self, table) -> tuple[np.ndarray, np.ndarray]:
        """Scan episode_idx once and return per-episode (lengths, offsets).

        Assumes rows are in episode-contiguous (non-decreasing) order.
        """
        ep_col, _ = self._index_columns
        reader = table.to_lance().scanner(columns=[ep_col]).to_reader()
        chunks = [
            batch.column(batch.schema.get_field_index(ep_col)).to_numpy()
            for batch in reader
        ]
        if not chunks:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        all_ep_ids = np.concatenate(chunks)
        if len(all_ep_ids) > 1 and (np.diff(all_ep_ids) < 0).any():
            raise ValueError(
                f"Lance table '{self.table_name}' at '{self.uri}' is not "
                'episode-contiguous (episode_idx decreases). Rebuild with '
                'convert_hdf5_to_lance(..., overwrite=True).'
            )

        change_positions = np.flatnonzero(np.diff(all_ep_ids) != 0) + 1
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
        """Zero-copy numpy for fixed-size lists / scalars; pylist otherwise."""
        pa = self._pa
        col_idx = batch.schema.get_field_index(key)
        if col_idx == -1:
            raise KeyError(f"Column '{key}' not found in batch")
        col = batch.column(col_idx)
        col_type = col.type

        if pa.types.is_binary(col_type) or pa.types.is_large_binary(col_type):
            return col.to_pylist()
        if pa.types.is_fixed_size_list(col_type):
            dim = col_type.list_size
            flat = col.flatten()
            return flat.to_numpy(zero_copy_only=False).reshape(len(col), dim)
        if pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
            return col.to_pylist()
        if pa.types.is_list(col_type):
            return col.to_pylist()
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
        # from_numpy is safe here — torch.stack() in _process_batch copies
        # each frame into a fresh resizable tensor before collation sees it.
        with Image.open(io.BytesIO(blob)) as img:
            arr = np.array(img.convert('RGB'))
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _prepare_numeric_tensor(self, data: np.ndarray, downsample: bool) -> torch.Tensor:
        # torch.tensor (not from_numpy): collation under __getitems__ calls
        # storage.resize_(), which fails on numpy-backed non-resizable tensors.
        if downsample:
            data = data[:: self.frameskip]
        tensor = torch.tensor(data)
        if tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(0, 3, 1, 2)
        return tensor

    def _process_batch(self, ep_idx: int, g_start: int, batch) -> dict:
        """Assemble one sample dict from a pre-fetched Arrow batch (no I/O)."""
        g_end = g_start + self.span
        steps: dict[str, Any] = {}

        for col in self._keys:
            if col in self._cache:
                values = self._cache[col][g_start:g_end]
            elif batch is None:
                raise KeyError(f"Column '{col}' not cached and no batch provided")
            else:
                values = self._extract_column(batch, col)

            if col in self.image_columns:
                blobs = values[:: self.frameskip]
                if isinstance(blobs, np.ndarray):
                    blobs = blobs.tolist()
                frames = [self._decode_image(v) for v in blobs]
                steps[col] = torch.stack(frames) if frames else torch.empty(0, dtype=torch.uint8)
                continue

            data = values if isinstance(values, np.ndarray) else self._pylist_to_numpy(values, col)

            if data.dtype == object and data.size > 0:
                first = data.flat[0]
                if isinstance(first, (bytes, bytearray)):
                    steps[col] = first.decode() if isinstance(first, bytes) else first
                    continue
                if isinstance(first, str):
                    steps[col] = first
                    continue

            steps[col] = self._prepare_numeric_tensor(data, downsample=col != 'action')

        return steps

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        g_start = int(self.offsets[ep_idx] + start)
        rows = list(range(g_start, g_start + (end - start)))
        batch = self._fetch_rows(rows)
        steps = self._process_batch(ep_idx, g_start, batch)
        return self.transform(steps) if self.transform else steps

    def __getitems__(self, indices: list[int]) -> list[dict]:
        """Batch fetch: one Lance ``take`` for the whole DataLoader batch.

        Called by PyTorch DataLoader ≥ 2.0 when defined, collapsing
        ``batch_size`` HTTP round-trips to one on remote storage.
        """
        all_rows: list[int] = []
        row_offsets: list[int] = []
        sample_meta: list[tuple[int, int]] = []
        for idx in indices:
            ep_idx, start = self.clip_indices[idx]
            g_start = int(self.offsets[ep_idx] + start)
            row_offsets.append(len(all_rows))
            all_rows.extend(range(g_start, g_start + self.span))
            sample_meta.append((ep_idx, g_start))

        big_batch = None
        if self._fetch_columns and all_rows:
            self._ensure_open()
            unique_rows = sorted(set(all_rows))
            unique_batch = self._perm.__getitems__(unique_rows)
            if len(unique_rows) == len(all_rows) and all_rows == unique_rows:
                big_batch = unique_batch
            else:
                row_lookup = {row: i for i, row in enumerate(unique_rows)}
                gather = self._pa.array([row_lookup[r] for r in all_rows], type=self._pa.int64())
                big_batch = unique_batch.take(gather)

        results: list[dict] = []
        for i, (ep_idx, g_start) in enumerate(sample_meta):
            sub_batch = big_batch.slice(row_offsets[i], self.span) if big_batch is not None else None
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
