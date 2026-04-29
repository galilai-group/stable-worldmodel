"""Lance format: LanceDB table with episode-contiguous flat rows.

Image columns (``pixels`` or ``pixels_<view>``) are stored as JPEG blobs in
``pa.binary`` columns. Tabular columns are flattened to fixed-size lists of
float32. Two index columns — ``episode_idx`` and ``step_idx`` — let the reader
recover episode boundaries by scanning a single column.

Lance rejects field names containing ``.`` (it uses dot as a struct-field
path separator). The writer transparently renames ``foo.bar`` → ``foo_bar``;
readers refer to columns by their on-disk (renamed) name.
"""

from __future__ import annotations

import io
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import lancedb
import pyarrow as pa
from lancedb.permutation import Permutation

from stable_worldmodel.data.dataset import Dataset
from stable_worldmodel.data.format import (
    Format,
    register_format,
    validate_write_mode,
)
from stable_worldmodel.data.formats.utils import is_image_column


_DEFAULT_JPEG_QUALITY = 95


def _to_lance_name(name: str) -> str:
    return name.replace('.', '_')


def _is_image_name(name: str) -> bool:
    return name == 'pixels' or name.startswith('pixels_')


def _encode_frame(frame: np.ndarray, jpeg_quality: int) -> bytes:
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 1:
        frame = frame.squeeze(-1)
    buf = io.BytesIO()
    Image.fromarray(frame.astype(np.uint8)).save(
        buf, format='JPEG', quality=jpeg_quality
    )
    return buf.getvalue()


class LanceDataset(Dataset):
    """Reader for a LanceDB table written by :class:`LanceWriter`.

    Args:
        path: Either a ``.lance`` directory path or a database URI.
        table_name: Table inside the database; inferred from a ``.lance``
            path when omitted.
        frameskip, num_steps, transform, keys_to_load, keys_to_cache,
            keys_to_merge: standard ``Dataset`` knobs.
        image_columns: override image-column auto-detection (any
            ``pa.binary`` column is treated as encoded image by default).
        episode_index_column, step_index_column: index column names.
        connect_kwargs: forwarded to :func:`lancedb.connect` (e.g. S3 creds).
    """

    _fork_warning_emitted = False

    def __init__(
        self,
        path: str | Path | None = None,
        table_name: str | None = None,
        *,
        uri: str | None = None,
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
        # Accept either `path=` (preferred, matches other readers) or `uri=`.
        loc = path if path is not None else uri
        if loc is None:
            raise TypeError('LanceDataset requires `path` (or `uri`)')

        resolved_uri, resolved_name = self._resolve_uri_and_table(
            str(loc), table_name
        )
        self.uri = resolved_uri
        self.table_name = resolved_name
        self.connect_kwargs = connect_kwargs or {}
        self._index_columns = (episode_index_column, step_index_column)
        self._cache: dict[str, np.ndarray] = {}
        self._perm = None
        self._fetch_columns: list[str] | None = None

        self._maybe_warn_fork_start_method()
        table = self._connect_table()
        self._schema_names = list(table.schema.names)
        available = [
            c for c in self._schema_names if c not in self._index_columns
        ]
        if not available:
            raise ValueError(
                'Lance table has no data columns (only index columns).'
            )

        self._keys = keys_to_load or available
        missing = [k for k in self._keys if k not in available]
        if missing:
            raise KeyError(
                f"Columns {missing} missing from Lance table '{resolved_name}'"
            )

        binary_cols = {
            f.name
            for f in table.schema
            if pa.types.is_binary(f.type) or pa.types.is_large_binary(f.type)
        }
        self.image_columns = (
            binary_cols & set(self._keys)
            if image_columns is None
            else {c for c in image_columns if c in self._keys}
        )

        lengths, offsets = self._compute_episode_structure(table)

        if keys_to_cache:
            logging.warning(
                'LanceDataset: keys_to_cache=%s is unnecessary — '
                '__getitems__ already batches reads; caching risks OOM.',
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
        state = self.__dict__.copy()
        state['_perm'] = None
        return state

    def _connect_table(self):
        db = lancedb.connect(self.uri, **self.connect_kwargs)
        return db.open_table(self.table_name)

    @staticmethod
    def _resolve_uri_and_table(
        loc: str, table_name: str | None
    ) -> tuple[str, str]:
        if table_name is not None:
            return loc, table_name

        stripped = loc.rstrip('/')
        if stripped.lower().endswith('.lance'):
            sep = stripped.rfind('/')
            parent, leaf = (
                (stripped[:sep], stripped[sep + 1 :])
                if sep >= 0
                else ('.', stripped)
            )
            return parent, leaf[: -len('.lance')]

        # Directory holding a single *.lance subdir.
        p = Path(loc)
        if p.is_dir():
            tables = sorted(p.glob('*.lance'))
            if len(tables) == 1:
                return str(p), tables[0].stem
            if len(tables) > 1:
                raise ValueError(
                    f'Ambiguous Lance dataset: multiple *.lance dirs in {p}. '
                    'Pass `table_name` explicitly.'
                )

        raise ValueError(
            f'LanceDataset: cannot infer table from {loc!r}. Pass '
            '`table_name=` explicitly or point to a `*.lance` directory.'
        )

    @classmethod
    def _maybe_warn_fork_start_method(cls) -> None:
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
                "LanceDataset: multiprocessing start method set to 'spawn' "
                '(was %s).',
                current or 'default (fork)',
            )
        except RuntimeError as exc:
            logging.warning(
                "LanceDataset could not switch multiprocessing to 'spawn' "
                '(%s); DataLoader workers may deadlock.',
                exc,
            )

    def _compute_episode_structure(
        self, table
    ) -> tuple[np.ndarray, np.ndarray]:
        ep_col, _ = self._index_columns
        reader = table.to_lance().scanner(columns=[ep_col]).to_reader()
        chunks = [
            batch.column(batch.schema.get_field_index(ep_col)).to_numpy()
            for batch in reader
        ]
        if not chunks:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        ep_ids = np.concatenate(chunks)
        if len(ep_ids) > 1 and (np.diff(ep_ids) < 0).any():
            raise ValueError(
                f"Lance table '{self.table_name}' at '{self.uri}' is not "
                'episode-contiguous (episode_idx decreases). Rebuild it.'
            )

        change_positions = np.flatnonzero(np.diff(ep_ids) != 0) + 1
        offsets = np.concatenate([[0], change_positions]).astype(np.int64)
        lengths = np.diff(np.concatenate([offsets, [len(ep_ids)]])).astype(
            np.int64
        )
        return lengths, offsets

    def _load_full_column(self, table, key: str) -> np.ndarray:
        data: list[np.ndarray] = []
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
                Permutation.identity(table)
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
        col_idx = batch.schema.get_field_index(key)
        if col_idx == -1:
            raise KeyError(f"Column '{key}' not found in batch")
        col = batch.column(col_idx)
        ctype = col.type
        if pa.types.is_binary(ctype) or pa.types.is_large_binary(ctype):
            return col.to_pylist()
        if pa.types.is_fixed_size_list(ctype):
            dim = ctype.list_size
            flat = col.flatten()
            return flat.to_numpy(zero_copy_only=False).reshape(len(col), dim)
        if pa.types.is_string(ctype) or pa.types.is_large_string(ctype):
            return col.to_pylist()
        if pa.types.is_list(ctype):
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
            dtype = (
                np.float32
                if isinstance(first, (float, np.floating))
                else np.int64
            )
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
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _prepare_numeric_tensor(
        self, data: np.ndarray, downsample: bool
    ) -> torch.Tensor:
        if downsample:
            data = data[:: self.frameskip]
        tensor = torch.tensor(data)
        if tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(0, 3, 1, 2)
        return tensor

    def _process_batch(self, ep_idx: int, g_start: int, batch) -> dict:
        g_end = g_start + self.span
        steps: dict[str, Any] = {}
        for col in self._keys:
            if col in self._cache:
                values = self._cache[col][g_start:g_end]
            elif batch is None:
                raise KeyError(
                    f"Column '{col}' not cached and no batch provided"
                )
            else:
                values = self._extract_column(batch, col)

            if col in self.image_columns:
                blobs = values[:: self.frameskip]
                if isinstance(blobs, np.ndarray):
                    blobs = blobs.tolist()
                frames = [self._decode_image(v) for v in blobs]
                steps[col] = (
                    torch.stack(frames)
                    if frames
                    else torch.empty(0, dtype=torch.uint8)
                )
                continue

            data = (
                values
                if isinstance(values, np.ndarray)
                else self._pylist_to_numpy(values, col)
            )

            if data.dtype == object and data.size > 0:
                first = data.flat[0]
                if isinstance(first, (bytes, bytearray)):
                    steps[col] = (
                        first.decode() if isinstance(first, bytes) else first
                    )
                    continue
                if isinstance(first, str):
                    steps[col] = first
                    continue

            steps[col] = self._prepare_numeric_tensor(
                data, downsample=col != 'action'
            )
        return steps

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        g_start = int(self.offsets[ep_idx] + start)
        rows = list(range(g_start, g_start + (end - start)))
        batch = self._fetch_rows(rows)
        steps = self._process_batch(ep_idx, g_start, batch)
        return self.transform(steps) if self.transform else steps

    def __getitems__(self, indices: list[int]) -> list[dict]:
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
                gather = pa.array(
                    [row_lookup[r] for r in all_rows], type=pa.int64()
                )
                big_batch = unique_batch.take(gather)

        results: list[dict] = []
        for i, (ep_idx, g_start) in enumerate(sample_meta):
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
                        f"Column '{col}' missing from cached and fetch columns"
                    )
                values = self._batch_column_pylist(batch, col)
            arr = (
                values
                if isinstance(values, np.ndarray)
                else self._pylist_to_numpy(values, col)
            )
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


class LanceWriter:
    """Append episodes to a Lance table.

    Layout: ``<path>/<table>.lance/`` (LanceDB stores each table as a
    directory). When ``path`` ends in ``.lance``, the parent is the URI and
    the stem is the table name; otherwise ``table_name`` must be supplied.

    Image columns (``pixels`` / ``pixels_<view>``, or any uint8 HxWxC array)
    are JPEG-encoded into ``pa.binary``. Tabular columns become fixed-size
    lists of float32. Column names with ``.`` are renamed to ``_`` (Lance
    rejects dots in top-level field names).

    Two write paths:
      * :meth:`write_episode` — push one episode; one ``table.add`` per call,
        one Lance version per call. Convenient for tests and one-off writes.
      * :meth:`write_episodes` — pull from a caller-provided iterable and
        stream it through a single ``pa.RecordBatchReader``. The whole
        iterable lands as one Lance version (the iterator pattern from the
        LanceDB docs). Memory stays bounded to one in-flight batch — the
        right shape for large collection sessions like ``World.collect``.
    """

    def __init__(
        self,
        path: str | Path,
        table_name: str | None = None,
        *,
        jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
        connect_kwargs: dict[str, Any] | None = None,
        mode: str = 'append',
    ):
        validate_write_mode(mode)
        loc = str(path).rstrip('/')
        if table_name is None:
            if loc.lower().endswith('.lance'):
                p = Path(loc)
                self.uri = str(p.parent) if str(p.parent) else '.'
                self.table_name = p.stem
            else:
                raise ValueError(
                    'LanceWriter: pass `table_name=` or a path ending in '
                    "'.lance'."
                )
        else:
            self.uri = loc
            self.table_name = table_name

        Path(self.uri).mkdir(parents=True, exist_ok=True)
        self.jpeg_quality = jpeg_quality
        self.connect_kwargs = connect_kwargs or {}
        self.mode = mode

        self._db = None
        self._table = None
        self._initialized = False
        self._appending_existing = False
        self._rename_map: dict[str, str] = {}
        self._image_cols: set[str] = set()
        self._dims: dict[str, int] = {}
        self._schema: pa.Schema | None = None
        self._ep_idx = 0
        self._global_ptr = 0

    def __enter__(self):
        self._db = lancedb.connect(self.uri, **self.connect_kwargs)
        if self.table_name in self._db.list_tables().tables:
            if self.mode == 'error':
                raise FileExistsError(
                    f"Lance table '{self.table_name}' already exists at "
                    f"'{self.uri}'. Pass mode='overwrite' to replace it or "
                    "mode='append' to extend it."
                )
            if self.mode == 'overwrite':
                self._db.drop_table(self.table_name)
            else:
                self._open_existing_for_append()
        return self

    def __exit__(self, *exc):
        self._db = None
        self._table = None

    def write_episode(self, ep_data: dict) -> None:
        if self._db is None:
            raise RuntimeError('LanceWriter used outside of a `with` block')
        self._consume_episodes([ep_data])

    def write_episodes(self, episodes) -> None:
        if self._db is None:
            raise RuntimeError('LanceWriter used outside of a `with` block')
        self._consume_episodes(episodes)

    def _consume_episodes(self, episodes) -> None:
        """Drive a single ``create_table``/``table.add`` from an iterable.

        The schema must be settled before we hand the reader to Lance, so we
        peek the first episode here, run the standard schema-init / append-
        validation hooks against it, then yield it as the first batch and
        stream the remaining episodes through the same generator.
        """
        iterator = iter(episodes)
        try:
            first_ep = next(iterator)
        except StopIteration:
            return

        if not self._initialized:
            self._init_schema(first_ep)
            self._initialized = True
        elif self._appending_existing and not self._rename_map:
            self._validate_episode_against_existing(first_ep)

        def batch_gen():
            yield self._batch_from_episode(first_ep)
            for ep in iterator:
                yield self._batch_from_episode(ep)

        reader = pa.RecordBatchReader.from_batches(self._schema, batch_gen())
        if self._table is None:
            self._table = self._db.create_table(
                self.table_name, data=reader, schema=self._schema
            )
        else:
            self._table.add(reader)

    def _batch_from_episode(self, ep_data: dict) -> pa.RecordBatch:
        ep_len = len(next(iter(ep_data.values())))
        batch = self._build_batch(ep_data, ep_len)
        self._ep_idx += 1
        self._global_ptr += ep_len
        return batch

    def _open_existing_for_append(self) -> None:
        self._table = self._db.open_table(self.table_name)
        schema = self._table.schema
        image_cols: set[str] = set()
        dims: dict[str, int] = {}
        for f in schema:
            if f.name in ('episode_idx', 'step_idx'):
                continue
            if pa.types.is_binary(f.type) or pa.types.is_large_binary(f.type):
                image_cols.add(f.name)
            elif pa.types.is_fixed_size_list(f.type):
                dims[f.name] = f.type.list_size
            else:
                raise ValueError(
                    f"LanceWriter: cannot append to '{self.table_name}' — "
                    f"existing column '{f.name}' has unsupported type "
                    f'{f.type}.'
                )

        existing = self._table.to_lance().to_table(columns=['episode_idx'])
        ep_col = existing.column('episode_idx').to_numpy()
        self._image_cols = image_cols
        self._dims = dims
        self._schema = schema
        self._global_ptr = int(len(ep_col))
        self._ep_idx = int(ep_col.max()) + 1 if self._global_ptr else 0
        self._initialized = True
        self._appending_existing = True

    def _validate_episode_against_existing(self, ep_data: dict) -> None:
        reserved = {'episode_idx', 'step_idx'}
        incoming_to_lance: dict[str, str] = {}
        for col in ep_data:
            lance_name = _to_lance_name(col)
            if lance_name in reserved:
                continue
            if lance_name in incoming_to_lance.values():
                raise ValueError(
                    f'LanceWriter: append failed — incoming columns map to '
                    f"the same Lance name '{lance_name}'."
                )
            incoming_to_lance[col] = lance_name
        lance_to_incoming = {v: k for k, v in incoming_to_lance.items()}

        expected = set(self._image_cols) | set(self._dims)
        incoming = set(lance_to_incoming)
        missing = expected - incoming
        extra = incoming - expected
        if missing or extra:
            raise ValueError(
                f'LanceWriter: append failed — schema mismatch on table '
                f"'{self.table_name}'. Missing columns: {sorted(missing)}; "
                f'unexpected columns: {sorted(extra)}.'
            )

        for lance_name in self._image_cols:
            vals = ep_data[lance_to_incoming[lance_name]]
            if not (_is_image_name(lance_name) or is_image_column(vals)):
                raise ValueError(
                    f"LanceWriter: append failed — column '{lance_name}' is "
                    'image-typed on disk but incoming values are not images.'
                )

        for lance_name, expected_dim in self._dims.items():
            vals = ep_data[lance_to_incoming[lance_name]]
            sample = np.asarray(vals[0])
            actual_dim = int(sample.reshape(-1).shape[0])
            if actual_dim != expected_dim:
                raise ValueError(
                    f"LanceWriter: append failed — column '{lance_name}' "
                    f'dimension mismatch: existing={expected_dim}, '
                    f'incoming={actual_dim}.'
                )

        ordered_lance_names = [
            f.name
            for f in self._schema
            if f.name not in ('episode_idx', 'step_idx')
        ]
        self._rename_map = {
            lance_to_incoming[ln]: ln for ln in ordered_lance_names
        }

    def _init_schema(self, sample_ep: dict) -> None:
        rename_map: dict[str, str] = {}
        image_cols: set[str] = set()
        dims: dict[str, int] = {}
        ordered_cols: list[str] = []

        reserved = {'episode_idx', 'step_idx'}
        dropped = [c for c in sample_ep if _to_lance_name(c) in reserved]
        if dropped:
            logging.warning(
                'LanceWriter: dropping incoming columns %s — names reserved '
                'for the writer-managed index columns.',
                dropped,
            )

        for col, vals in sample_ep.items():
            lance_name = _to_lance_name(col)
            if lance_name in reserved:
                continue
            rename_map[col] = lance_name
            ordered_cols.append(lance_name)

            if _is_image_name(lance_name) or is_image_column(vals):
                image_cols.add(lance_name)
                continue

            sample = np.asarray(vals[0])
            dims[lance_name] = int(sample.reshape(-1).shape[0])

        renamed = {k: v for k, v in rename_map.items() if k != v}
        if renamed:
            logging.info(
                'LanceWriter: renamed columns for Lance compatibility: %s',
                renamed,
            )

        fields = [
            pa.field('episode_idx', pa.int32()),
            pa.field('step_idx', pa.int32()),
        ]
        for col in ordered_cols:
            if col in image_cols:
                fields.append(pa.field(col, pa.binary()))
            else:
                fields.append(pa.field(col, pa.list_(pa.float32(), dims[col])))

        self._rename_map = rename_map
        self._image_cols = image_cols
        self._dims = dims
        self._schema = pa.schema(fields)

    def _build_batch(self, ep_data: dict, ep_len: int) -> pa.RecordBatch:
        episode_idx = np.full(ep_len, self._ep_idx, dtype=np.int32)
        step_idx = np.arange(ep_len, dtype=np.int32)

        arrays: list[pa.Array] = [
            pa.array(episode_idx, type=pa.int32()),
            pa.array(step_idx, type=pa.int32()),
        ]
        for col, lance_name in self._rename_map.items():
            vals = ep_data[col]
            if lance_name in self._image_cols:
                blobs = [
                    _encode_frame(np.asarray(v), self.jpeg_quality)
                    for v in vals
                ]
                arrays.append(pa.array(blobs, type=pa.binary()))
            else:
                dim = self._dims[lance_name]
                flat = np.asarray(vals, dtype=np.float32).reshape(ep_len, dim)
                arrays.append(
                    pa.FixedSizeListArray.from_arrays(
                        pa.array(flat.reshape(-1), type=pa.float32()), dim
                    )
                )

        return pa.record_batch(arrays, schema=self._schema)


@register_format
class Lance(Format):
    name = 'lance'

    @classmethod
    def detect(cls, path) -> bool:
        s = str(path).rstrip('/')
        if s.lower().endswith('.lance'):
            return True
        p = Path(s)
        if p.is_dir():
            return any(p.glob('*.lance'))
        return False

    @classmethod
    def open_reader(cls, path, **kwargs) -> LanceDataset:
        return LanceDataset(path=path, **kwargs)

    @classmethod
    def open_writer(cls, path, **kwargs) -> LanceWriter:
        return LanceWriter(path, **kwargs)


__all__ = ['Lance', 'LanceDataset', 'LanceWriter']
