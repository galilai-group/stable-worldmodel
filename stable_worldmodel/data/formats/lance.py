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


# Optional fast-path: torchvision's libjpeg-turbo-backed batch decoder.
# Falls back to PIL on a thread pool when unavailable or when a blob is
# malformed (decode_jpeg is stricter about JPEG conformance than PIL).
try:
    from torchvision.io import (
        ImageReadMode as _TVImageReadMode,
        decode_jpeg as _tv_decode_jpeg,
    )

    _TV_RGB = _TVImageReadMode.RGB
except (ImportError, AttributeError):
    _tv_decode_jpeg = None
    _TV_RGB = None


_DEFAULT_JPEG_QUALITY = 95
_IMAGE_CODECS = ('raw', 'jpeg', 'both')
_IMAGE_SHAPE_META_KEY = b'image_shape'
_LANCE_COMPRESSION_KEY = b'lance-encoding:compression'
# Storage format 2.2 is required for general-compression metadata to take
# effect on binary columns (auto-LZ4 above 32KB, opt-in zstd via metadata).
_DATA_STORAGE_VERSION = '2.2'
# Default codec for raw image columns. Zstd → ~30-40× on synthetic env
# frames (tworoom), ~1.5-2× on natural images. LZ4 is the fallback
# auto-applied by Lance 2.2 when no codec hint is given.
_DEFAULT_RAW_COMPRESSION = b'zstd'


def _normalize_image_frame(frame: np.ndarray) -> np.ndarray:
    """Normalise a single frame to uint8 (H, W, C) with C in {1, 3, 4}."""
    arr = np.asarray(frame)
    if (
        arr.ndim == 3
        and arr.shape[0] in (1, 3, 4)
        and arr.shape[-1] not in (1, 3, 4)
    ):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


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

        # Auto-detect image columns from schema. Three storage flavours:
        #   * pa.binary / pa.large_binary WITHOUT image_shape metadata —
        #     JPEG blob; decoded via _decode_images on read.
        #   * pa.binary / pa.large_binary WITH image_shape metadata —
        #     raw uint8 bytes, current default; Lance applies general
        #     compression (zstd via field metadata). Decoded via
        #     np.frombuffer + reshape on read.
        #   * pa.fixed_size_list(uint8, H*W*C) WITH image_shape metadata —
        #     legacy raw layout (pre-compression-fix); still readable.
        jpeg_cols: set[str] = set()
        raw_image_shapes: dict[str, tuple[int, int, int]] = {}
        for f in table.schema:
            shape_meta = (f.metadata or {}).get(_IMAGE_SHAPE_META_KEY)
            is_bin = pa.types.is_binary(f.type) or pa.types.is_large_binary(
                f.type
            )
            is_fsl_u8 = pa.types.is_fixed_size_list(f.type) and (
                pa.types.is_uint8(f.type.value_type)
            )
            if shape_meta and (is_bin or is_fsl_u8):
                h, w, c = (int(x) for x in shape_meta.decode().split(','))
                raw_image_shapes[f.name] = (h, w, c)
            elif is_bin:
                jpeg_cols.add(f.name)
        autodetected = jpeg_cols | set(raw_image_shapes)
        self.image_columns = (
            autodetected & set(self._keys)
            if image_columns is None
            else {c for c in image_columns if c in self._keys}
        )
        # Subset of image_columns that are stored as raw uint8 frames.
        self._raw_image_shapes = {
            c: raw_image_shapes[c]
            for c in self.image_columns
            if c in raw_image_shapes
        }

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

    def _decode_images(self, blobs) -> torch.Tensor:
        """Decode a list of JPEG blobs into a stacked ``(N, C, H, W)`` uint8 tensor.

        Fast path: ``torchvision.io.decode_jpeg`` when available —
        libjpeg-turbo with internal SIMD, GIL-released, supports CUDA
        decode if a GPU is present. Single Python call so DataLoader
        workers (which are already process-parallel) don't compete with
        an extra thread pool of our own. Falls back to a sequential PIL
        loop when torchvision is missing or a blob is non-conformant.
        """
        if not blobs:
            return torch.empty(0, dtype=torch.uint8)

        if _tv_decode_jpeg is not None:
            try:
                byte_tensors = [
                    torch.frombuffer(
                        b if isinstance(b, (bytes, bytearray)) else bytes(b),
                        dtype=torch.uint8,
                    )
                    for b in blobs
                ]
                decoded = _tv_decode_jpeg(byte_tensors, mode=_TV_RGB)
                return torch.stack(decoded)
            except (RuntimeError, TypeError):
                pass  # malformed blob — fall through to PIL

        return torch.stack([self._decode_image(b) for b in blobs])

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
                if col in self._raw_image_shapes:
                    h, w, c = self._raw_image_shapes[col]
                    # Two source layouts for raw uint8 image data:
                    #   * list of byte-blobs (pa.large_binary, current
                    #     default — frombuffer per blob, then stack)
                    #   * numpy (T, h*w*c) (legacy fixed_size_list path
                    #     or cached column — straight reshape)
                    if isinstance(values, np.ndarray):
                        arr = values.reshape(-1, h, w, c)
                    else:
                        arr = np.stack(
                            [
                                np.frombuffer(b, dtype=np.uint8).reshape(
                                    h, w, c
                                )
                                for b in values
                            ]
                        )
                    arr = arr[:: self.frameskip]
                    steps[col] = (
                        torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
                    )
                    continue
                blobs = values[:: self.frameskip]
                if isinstance(blobs, np.ndarray):
                    blobs = blobs.tolist()
                steps[col] = self._decode_images(blobs)
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
        image_codec: str = 'jpeg',
    ):
        validate_write_mode(mode)
        if image_codec not in _IMAGE_CODECS:
            raise ValueError(
                f'image_codec must be one of {_IMAGE_CODECS}, got {image_codec!r}'
            )
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
        self.image_codec = image_codec

        self._db = None
        self._table = None
        self._initialized = False
        self._appending_existing = False
        # Source col → list of (lance_name, kind) pairs.
        # kind ∈ {'image_raw', 'image_jpeg', 'tabular'}.
        # Most cols map 1→1; image cols in 'both' mode map 1→2.
        self._col_plan: dict[str, list[tuple[str, str]]] = {}
        # Per Lance column: (h, w, c) for image_raw, dim for tabular, None for image_jpeg.
        self._col_meta: dict[str, Any] = {}
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
        elif self._appending_existing and not self._col_plan:
            self._validate_episode_against_existing(first_ep)

        def batch_gen():
            yield self._batch_from_episode(first_ep)
            for ep in iterator:
                yield self._batch_from_episode(ep)

        reader = pa.RecordBatchReader.from_batches(self._schema, batch_gen())
        if self._table is None:
            self._table = self._db.create_table(
                self.table_name,
                data=reader,
                schema=self._schema,
                data_storage_version=_DATA_STORAGE_VERSION,
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
        # Reverse-engineer per-column metadata from the existing schema so
        # we can append in whatever codec the table was originally written.
        col_meta: dict[str, Any] = {}
        for f in schema:
            if f.name in ('episode_idx', 'step_idx'):
                continue
            shape_meta = (f.metadata or {}).get(_IMAGE_SHAPE_META_KEY)
            is_bin = pa.types.is_binary(f.type) or pa.types.is_large_binary(
                f.type
            )
            is_fsl_u8 = pa.types.is_fixed_size_list(f.type) and (
                pa.types.is_uint8(f.type.value_type)
            )
            if shape_meta and (is_bin or is_fsl_u8):
                h, w, c = (int(x) for x in shape_meta.decode().split(','))
                col_meta[f.name] = ('image_raw', (h, w, c))
            elif is_bin:
                col_meta[f.name] = ('image_jpeg', None)
            elif pa.types.is_fixed_size_list(f.type):
                col_meta[f.name] = ('tabular', f.type.list_size)
            else:
                raise ValueError(
                    f"LanceWriter: cannot append to '{self.table_name}' — "
                    f"existing column '{f.name}' has unsupported type "
                    f'{f.type}.'
                )

        existing = self._table.to_lance().to_table(columns=['episode_idx'])
        ep_col = existing.column('episode_idx').to_numpy()
        self._col_meta = col_meta
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
                    'LanceWriter: append failed — incoming columns map to '
                    f"the same Lance name '{lance_name}'."
                )
            incoming_to_lance[col] = lance_name
        lance_to_incoming = {v: k for k, v in incoming_to_lance.items()}

        # Build the expected primary set: in 'both' mode `<X>_jpeg` is a
        # companion of `<X>` (same source col, written twice), so it
        # doesn't need its own incoming entry. Group lance cols by primary.
        kinds = {ln: kind for ln, (kind, _) in self._col_meta.items()}
        primary_lance: dict[str, list[str]] = {}
        for ln in self._col_meta:
            base = ln
            if kinds[ln] == 'image_jpeg' and ln.endswith('_jpeg'):
                stripped = ln[: -len('_jpeg')]
                if kinds.get(stripped) == 'image_raw':
                    base = stripped
            primary_lance.setdefault(base, []).append(ln)

        incoming_lance = set(incoming_to_lance.values())
        missing = set(primary_lance) - incoming_lance
        extra = incoming_lance - set(primary_lance)
        if missing or extra:
            raise ValueError(
                f'LanceWriter: append failed — schema mismatch on table '
                f"'{self.table_name}'. Missing columns: {sorted(missing)}; "
                f'unexpected columns: {sorted(extra)}.'
            )

        # Per-col type/shape checks.
        for primary, lance_names in primary_lance.items():
            vals = ep_data[lance_to_incoming[primary]]
            for ln in lance_names:
                kind, info = self._col_meta[ln]
                if kind == 'image_raw':
                    sample = _normalize_image_frame(vals[0])
                    if sample.shape != info:
                        raise ValueError(
                            f"LanceWriter: append failed — column '{ln}' "
                            f'image-shape mismatch: existing={info}, '
                            f'incoming={sample.shape}.'
                        )
                elif kind == 'image_jpeg':
                    if not (_is_image_name(ln) or is_image_column(vals)):
                        raise ValueError(
                            f"LanceWriter: append failed — column '{ln}' is "
                            'image-typed on disk but incoming values are '
                            'not images.'
                        )
                else:  # tabular
                    sample = np.asarray(vals[0])
                    if int(sample.reshape(-1).shape[0]) != info:
                        raise ValueError(
                            f"LanceWriter: append failed — column '{ln}' "
                            f'dimension mismatch: existing={info}, '
                            f'incoming={sample.reshape(-1).shape[0]}.'
                        )

        # Build the col plan from the schema's lance-col order.
        plan: dict[str, list[tuple[str, str]]] = {}
        for ln in (
            f.name
            for f in self._schema
            if f.name not in ('episode_idx', 'step_idx')
        ):
            for primary, lance_names in primary_lance.items():
                if ln in lance_names:
                    src = lance_to_incoming[primary]
                    plan.setdefault(src, []).append((ln, kinds[ln]))
                    break
        self._col_plan = plan

    def _init_schema(self, sample_ep: dict) -> None:
        col_plan: dict[str, list[tuple[str, str]]] = {}
        col_meta: dict[str, Any] = {}
        rename_map: dict[str, str] = {}

        reserved = {'episode_idx', 'step_idx'}
        dropped = [c for c in sample_ep if _to_lance_name(c) in reserved]
        if dropped:
            logging.warning(
                'LanceWriter: dropping incoming columns %s — names reserved '
                'for the writer-managed index columns.',
                dropped,
            )

        ordered_lance: list[str] = []
        for col, vals in sample_ep.items():
            lance_name = _to_lance_name(col)
            if lance_name in reserved:
                continue
            rename_map[col] = lance_name
            entries: list[tuple[str, str]] = []

            if _is_image_name(lance_name) or is_image_column(vals):
                sample = _normalize_image_frame(vals[0])
                shape = sample.shape  # (H, W, C)
                if self.image_codec in ('raw', 'both'):
                    entries.append((lance_name, 'image_raw'))
                    col_meta[lance_name] = ('image_raw', shape)
                    ordered_lance.append(lance_name)
                if self.image_codec in ('jpeg', 'both'):
                    jpeg_name = (
                        f'{lance_name}_jpeg'
                        if self.image_codec == 'both'
                        else lance_name
                    )
                    entries.append((jpeg_name, 'image_jpeg'))
                    col_meta[jpeg_name] = ('image_jpeg', None)
                    ordered_lance.append(jpeg_name)
            else:
                sample = np.asarray(vals[0])
                dim = int(sample.reshape(-1).shape[0])
                entries.append((lance_name, 'tabular'))
                col_meta[lance_name] = ('tabular', dim)
                ordered_lance.append(lance_name)

            col_plan[col] = entries

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
        for ln in ordered_lance:
            kind, info = col_meta[ln]
            if kind == 'image_raw':
                h, w, c = info
                # pa.large_binary (not fixed_size_list) so the
                # `lance-encoding:compression` metadata key actually
                # engages — fixed_size_list<uint8> bypasses general
                # compression entirely in Lance v2.
                fields.append(
                    pa.field(
                        ln,
                        pa.large_binary(),
                        metadata={
                            _IMAGE_SHAPE_META_KEY: f'{h},{w},{c}'.encode(),
                            _LANCE_COMPRESSION_KEY: _DEFAULT_RAW_COMPRESSION,
                        },
                    )
                )
            elif kind == 'image_jpeg':
                fields.append(pa.field(ln, pa.binary()))
            else:  # tabular
                fields.append(pa.field(ln, pa.list_(pa.float32(), info)))

        self._col_plan = col_plan
        self._col_meta = col_meta
        self._schema = pa.schema(fields)

    def _build_batch(self, ep_data: dict, ep_len: int) -> pa.RecordBatch:
        episode_idx = np.full(ep_len, self._ep_idx, dtype=np.int32)
        step_idx = np.arange(ep_len, dtype=np.int32)

        arrays: list[pa.Array] = [
            pa.array(episode_idx, type=pa.int32()),
            pa.array(step_idx, type=pa.int32()),
        ]
        # Iterate the schema's lance-col order to ensure arrays line up.
        for f in self._schema:
            if f.name in ('episode_idx', 'step_idx'):
                continue
            kind, info = self._col_meta[f.name]
            # Find the source col that feeds this lance col.
            src_col = next(
                src
                for src, entries in self._col_plan.items()
                if any(ln == f.name for ln, _ in entries)
            )
            vals = ep_data[src_col]
            if kind == 'image_raw':
                # Each frame becomes a contiguous H*W*C uint8 byte blob.
                # Stored as pa.large_binary so Lance applies the
                # general-compression encoding (LZ4/Zstd via metadata).
                blobs = [_normalize_image_frame(v).tobytes() for v in vals]
                arrays.append(pa.array(blobs, type=pa.large_binary()))
            elif kind == 'image_jpeg':
                blobs = [
                    _encode_frame(np.asarray(v), self.jpeg_quality)
                    for v in vals
                ]
                arrays.append(pa.array(blobs, type=pa.binary()))
            else:  # tabular
                flat = np.asarray(vals, dtype=np.float32).reshape(ep_len, info)
                arrays.append(
                    pa.FixedSizeListArray.from_arrays(
                        pa.array(flat.reshape(-1), type=pa.float32()), info
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
