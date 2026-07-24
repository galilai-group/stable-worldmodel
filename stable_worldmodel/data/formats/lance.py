"""Lance format: LanceDB table with episode-contiguous flat rows.

Image columns (``pixels`` or ``pixels_<view>``) are stored as JPEG blobs in
``pa.binary`` columns. Tabular columns are flattened to fixed-size lists of
float32. Two index columns — ``episode_idx`` and ``step_idx`` — let the reader
recover episode boundaries by scanning a single column.

Episode-scoped data (the :data:`~stable_worldmodel.data.EPISODE_DATA_KEY`
entry of an episode dict — values constant within an episode, e.g. a scene
XML) is stored in a ``<table>_episodes`` side table with one row per
episode: ``episode_idx`` plus one column per key (str → ``pa.string``,
bytes → ``pa.large_binary``, scalars → their arrow type, numeric arrays →
variable-length ``pa.list_(float32)``). Per-step string columns are not
supported: every binary frames column is an encoded image.

Lance rejects field names containing ``.`` (it uses dot as a struct-field
path separator). The writer transparently renames ``foo.bar`` → ``foo_bar``;
readers refer to columns by their on-disk (renamed) name.
"""

from __future__ import annotations

import io
import logging
import os
import re
import warnings
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
    split_episode_data,
    validate_write_mode,
)
from stable_worldmodel.data.formats.utils import is_image_column

# JPEG blobs from Lance are immutable `bytes`; we wrap them in a
# torch tensor view (zero-copy) and hand straight to torchvision's
# decoder, which returns a fresh tensor for the output. The warning
# is correct in general (writes via the view would corrupt the
# source) but doesn't apply since downstream is read-only.
warnings.filterwarnings(
    'ignore',
    message='The given buffer is not writable',
    category=UserWarning,
)


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


def _settle_episode_keys(
    settled: tuple[str, ...] | None,
    ep_extra: dict,
    *,
    appending: bool,
    writer_name: str,
    table_name: str,
    episodes_table_name: str,
) -> tuple[str, ...]:
    """Enforce a fixed episode-data key set per table; return it settled.

    The first episode of a fresh table fixes the key set (possibly empty);
    every later episode — including across append sessions, where the caller
    seeds ``settled`` from the on-disk side table — must match it exactly.
    """
    keys = tuple(sorted(ep_extra))
    if settled is None:
        return keys
    if keys == settled:
        return settled
    if appending and not settled:
        raise ValueError(
            f"{writer_name}: table '{table_name}' has no episode-data side "
            f"table ('{episodes_table_name}') but the incoming episode "
            f'carries episode data {sorted(keys)}. Regenerate the dataset '
            'in the episode-data layout before appending.'
        )
    raise ValueError(
        f'{writer_name}: episode-data key mismatch — every episode of '
        f"table '{table_name}' must carry the same keys. Expected "
        f'{sorted(settled)}, got {sorted(keys)}.'
    )


def _episode_schema(row: dict) -> pa.Schema:
    """Infer the ``<table>_episodes`` side-table schema from one row."""
    fields = [pa.field('episode_idx', pa.int32())]
    for key in sorted(k for k in row if k != 'episode_idx'):
        v = row[key]
        if isinstance(v, str):
            ptype = pa.string()
        elif isinstance(v, (bytes, bytearray, memoryview)):
            ptype = pa.large_binary()
        elif isinstance(v, (bool, np.bool_)):
            ptype = pa.bool_()
        elif isinstance(v, (int, np.integer)):
            ptype = pa.int64()
        elif isinstance(v, (float, np.floating)):
            ptype = pa.float64()
        elif isinstance(v, (np.ndarray, torch.Tensor, list, tuple)):
            ptype = pa.list_(pa.float32())
        else:
            raise TypeError(
                'unsupported episode-data type '
                f"{type(v).__name__} for key '{key}' (expected str, "
                'bytes, scalar, or numeric array).'
            )
        fields.append(pa.field(key, ptype))
    return pa.schema(fields)


def _episode_value_array(field: pa.Field, vals: list) -> pa.Array:
    if pa.types.is_string(field.type):
        vals = [
            v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
            for v in vals
        ]
    elif pa.types.is_large_binary(field.type):
        vals = [v.encode() if isinstance(v, str) else bytes(v) for v in vals]
    elif pa.types.is_list(field.type):
        vals = [
            np.asarray(
                v.detach().cpu() if isinstance(v, torch.Tensor) else v,
                dtype=np.float32,
            ).reshape(-1)
            for v in vals
        ]
    return pa.array(vals, type=field.type)


def _episode_rows_table(
    rows: list[dict], schema: pa.Schema | None
) -> tuple[pa.Table, pa.Schema]:
    """Build the side-table batch for ``rows``, inferring the schema from
    the first row when none is settled yet."""
    if schema is None:
        schema = _episode_schema(rows[0])
    arrays = [
        _episode_value_array(field, [row[field.name] for row in rows])
        for field in schema
    ]
    return pa.Table.from_batches(
        [pa.record_batch(arrays, schema=schema)]
    ), schema


_MP_START_FORCED = False

# Imported once by the forkserver so DataLoader workers fork from a warm
# interpreter instead of re-importing this stack per worker (~15-30 s each
# from a network-mounted env; the whole tree is >5 min cold).
# INVARIANT: nothing on these modules' import path may import lance/lancedb
# or start runtimes, threads holding locks, or CUDA at import time. In
# particular the lance Tokio runtime must only ever be created inside each
# worker, after the fork — that is the whole reason forkserver is used. Do
# NOT add `lancedb`, `stable_worldmodel.data`, `stable_pretraining.data`, or
# anything else that imports lance at module level.
# Every module here must also import cleanly or raise *ImportError* when
# absent: the stdlib forkserver swallows only ImportError from preload
# modules, so any other exception at import time kills the server and makes
# every worker start fail with BrokenPipeError. Modules that load native
# libraries (and can raise RuntimeError/OSError) go in
# _FORKSERVER_PRELOAD_OPTIONAL instead, which is probed before use.
_FORKSERVER_PRELOAD = (
    'numpy',
    'PIL',
    'pyarrow',
    'torch',
    'torchvision',
    'lightning',
    'transformers',
    'datasets',
    'cv2',
    'stable_pretraining',
    'stable_worldmodel',
)

# Media decoders/encoders used by the lance-video path. They dlopen FFmpeg
# shared libraries at import and raise RuntimeError/OSError (not ImportError)
# when those are missing or ABI-incompatible — exactly what the forkserver
# will not tolerate. We import them in the parent first and preload only the
# ones that actually load, so a broken/absent FFmpeg degrades to "not
# preloaded" instead of crashing every DataLoader worker.
_FORKSERVER_PRELOAD_OPTIONAL = (
    'imageio',
    'torchcodec',
)


def _force_forkserver() -> None:
    """Switch Linux multiprocessing to a fork-safe start method for lancedb.

    lance runs an internal async (Tokio) runtime; forking a process that has
    already started it leaves the child with a half-initialized runtime that
    can deadlock. ``forkserver`` launches a clean helper interpreter (with no
    lance runtime) and forks DataLoader workers from *that*, so each worker
    opens lance after the fork and gets its own runtime — the start method
    lancedb itself recommends. Worker startup is also cheaper than ``spawn``
    (a fork of the clean server vs a full re-exec + re-import per worker).
    Falls back to ``spawn`` where forkserver is unavailable.

    The server preloads :data:`_FORKSERVER_PRELOAD` (heavy, fork-benign
    modules only — never lance itself) so each worker forks with the
    expensive imports already in ``sys.modules``. Modules absent from the
    environment raise ImportError and are silently skipped by the stdlib
    forkserver; :data:`_FORKSERVER_PRELOAD_OPTIONAL` modules (which can fail
    with RuntimeError/OSError) are probed here and only preloaded when they
    load, since a non-ImportError in the server would kill every worker.
    """
    import importlib
    import logging
    import multiprocessing as mp
    import sys

    import torch

    global _MP_START_FORCED
    if _MP_START_FORCED or sys.platform != 'linux':
        _MP_START_FORCED = True
        return
    _MP_START_FORCED = True

    if mp.get_start_method(allow_none=True) in (None, 'fork'):
        methods = mp.get_all_start_methods()
        target = 'forkserver' if 'forkserver' in methods else 'spawn'
        try:
            mp.set_start_method(target, force=True)
        except RuntimeError as exc:
            logging.warning('Could not switch to %s (%s)', target, exc)
    if mp.get_start_method(allow_none=True) == 'forkserver':
        preload = list(_FORKSERVER_PRELOAD)
        for modname in _FORKSERVER_PRELOAD_OPTIONAL:
            try:
                importlib.import_module(modname)
            except Exception as exc:
                logging.warning(
                    'Skipping forkserver preload of %s (%s)', modname, exc
                )
            else:
                preload.append(modname)
        try:
            mp.set_forkserver_preload(preload)
        except Exception as exc:
            logging.warning('Could not set forkserver preload (%s)', exc)
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except RuntimeError:
        pass


class LanceDataset(Dataset):
    """Reader for a LanceDB table written by :class:`LanceWriter`.

    Args:
        path: Either a ``.lance`` directory path or a database URI.
        table_name: Table inside the database; inferred from a ``.lance``
            path when omitted.
        frameskip: Standard ``Dataset`` knob.
        num_steps: Standard ``Dataset`` knob.
        transform: Standard ``Dataset`` knob.
        keys_to_load: Standard ``Dataset`` knob.
        keys_to_cache: Standard ``Dataset`` knob.
        keys_to_merge: Standard ``Dataset`` knob.
        image_columns: override image-column auto-detection (every
            ``pa.binary`` column is treated as an encoded image by default).
        episode_index_column: episode index column name.
        step_index_column: step index column name.
        connect_kwargs: forwarded to :func:`lancedb.connect` (e.g. S3 creds).
    """

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
        self._episodes_table_name = f'{resolved_name}_episodes'
        self.connect_kwargs = connect_kwargs or {}
        self._index_columns = (episode_index_column, step_index_column)
        self._cache: dict[str, np.ndarray] = {}
        self._perm = None
        self._fetch_columns: list[str] | None = None

        _force_forkserver()
        db = lancedb.connect(self.uri, **self.connect_kwargs)
        table = db.open_table(self.table_name)

        legacy_strings = [
            f.name
            for f in table.schema
            if f.name not in self._index_columns
            and (
                pa.types.is_string(f.type) or pa.types.is_large_string(f.type)
            )
        ]
        if legacy_strings:
            raise ValueError(
                f"Lance table '{self.table_name}' at '{self.uri}' contains "
                f'per-step string columns {legacy_strings} (legacy layout). '
                'Per-step strings are no longer supported: episode-scoped '
                f"values now live in a '{self._episodes_table_name}' side "
                'table. Regenerate the dataset in the episode-data layout.'
            )

        self._episode_columns: list[str] = []
        self._episode_data_cache: dict[str, list] | None = None
        if self._episodes_table_name in db.list_tables().tables:
            ep_schema = db.open_table(self._episodes_table_name).schema
            self._episode_columns = [
                n for n in ep_schema.names if n != 'episode_idx'
            ]

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
                'LanceDataset: keys_to_cache=%s is not required — Lance '
                'has efficient random access via batched __getitems__.',
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

    @property
    def episode_column_names(self) -> list[str]:
        return list(self._episode_columns)

    def get_episode_data(
        self, episodes_idx: np.ndarray | list[int] | None = None
    ) -> dict[str, list]:
        if not self._episode_columns:
            return {}
        if self._episode_data_cache is None:
            self._episode_data_cache = self._load_episode_table()
        if episodes_idx is None:
            idxs: Any = range(len(self.lengths))
        else:
            idxs = [
                int(i)
                for i in np.asarray(episodes_idx, dtype=np.int64).reshape(-1)
            ]
        return {
            k: [vals[i] for i in idxs]
            for k, vals in self._episode_data_cache.items()
        }

    def _load_episode_table(self) -> dict[str, list]:
        db = lancedb.connect(self.uri, **self.connect_kwargs)
        tbl = db.open_table(self._episodes_table_name).to_lance().to_table()
        n = len(self.lengths)
        ep_ids = tbl.column('episode_idx').to_numpy(zero_copy_only=False)
        if tbl.num_rows != n or not np.array_equal(
            np.sort(ep_ids), np.arange(n)
        ):
            raise ValueError(
                f"Lance episode-data table '{self._episodes_table_name}' at "
                f"'{self.uri}' has {tbl.num_rows} rows for {n} episodes — "
                'the dataset is inconsistent (partial write?). Rebuild it.'
            )
        order = np.argsort(ep_ids)
        out: dict[str, list] = {}
        for name in self._episode_columns:
            col = tbl.column(name)
            raw = col.to_pylist()
            out[name] = [
                self._decode_episode_value(raw[int(i)], col.type)
                for i in order
            ]
        return out

    @staticmethod
    def _decode_episode_value(value, ptype) -> Any:
        if value is None:
            return None
        if pa.types.is_list(ptype) or pa.types.is_fixed_size_list(ptype):
            return np.asarray(value, dtype=np.float32)
        if pa.types.is_binary(ptype) or pa.types.is_large_binary(ptype):
            return bytes(value)
        return value

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state['_perm'] = None
        # Episode blobs (e.g. scene XMLs) can be large; workers reload the
        # tiny side table lazily instead of shipping the cache in the pickle.
        state['_episode_data_cache'] = None
        # spt.Module sets `dataset._trainer = trainer` on every dataset to
        # inject `global_step` / `current_epoch` into samples. The trainer
        # transitively reaches `train_dataloader._iterator` (a
        # `_MultiProcessingDataLoaderIter`, which raises NotImplementedError
        # on pickle). Drop the back-reference so worker-process pickling
        # doesn't traverse into it; workers see a stale snapshot of trainer
        # state anyway, so a missing trainer is fine.
        state['_trainer'] = None
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

        # Directory holding a single *.lance subdir. `_episodes` /
        # `_videos` siblings are companion tables, never the frames table.
        p = Path(loc)
        if p.is_dir():
            tables = sorted(
                t
                for t in p.glob('*.lance')
                if not t.stem.endswith(('_episodes', '_videos'))
            )
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

    def _process_batch(
        self,
        ep_idx: int,
        g_start: int,
        batch,
        g_end: int | None = None,
        decoded_images: dict | None = None,
    ) -> dict:
        if g_end is None:
            g_end = g_start + self.span
        decoded_images = decoded_images or {}
        steps = {}
        for col in self._keys:
            if col in decoded_images:
                steps[col] = decoded_images[col]
            else:
                steps[col] = self._process_col(col, batch, g_start, g_end)
        return steps

    def _process_col(self, col: str, batch, g_start: int, g_end: int) -> Any:
        """Decode a single window column into its tensor / scalar value.

        Split out from :meth:`_process_batch` so subclasses (e.g. the
        video-blob reader) can intercept select columns while reusing the
        tabular / JPEG decode path verbatim.
        """
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
            return self._decode_images(blobs)

        data = (
            values
            if isinstance(values, np.ndarray)
            else self._pylist_to_numpy(values, col)
        )
        return self._prepare_numeric_tensor(data, downsample=col != 'action')

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        g_start = int(self.offsets[ep_idx] + start)
        rows = list(range(g_start, g_start + (end - start)))
        batch = self._fetch_rows(rows)
        steps = self._process_batch(
            ep_idx, g_start, batch, g_end=g_start + (end - start)
        )
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
        unique_pos: dict[int, int] | None = None
        decoded_images: dict[str, torch.Tensor] = {}
        if self._fetch_columns and all_rows:
            self._ensure_open()
            unique_rows = sorted(set(all_rows))
            unique_batch = self._perm.__getitems__(unique_rows)
            unique_pos = {row: i for i, row in enumerate(unique_rows)}
            if len(unique_rows) == len(all_rows) and all_rows == unique_rows:
                big_batch = unique_batch
            else:
                gather = pa.array(
                    [unique_pos[r] for r in all_rows], type=pa.int64()
                )
                big_batch = unique_batch.take(gather)

            # Decode each fetched image column once over the deduped rows;
            # overlapping windows then gather shared frames instead of
            # re-decoding the same blob per window.
            for col in self.image_columns:
                if col in self._cache:
                    continue
                blobs = self._extract_column(unique_batch, col)
                if isinstance(blobs, np.ndarray):
                    blobs = blobs.tolist()
                decoded_images[col] = self._decode_images(blobs)

        results: list[dict] = []
        for i, (ep_idx, g_start) in enumerate(sample_meta):
            sub_batch = (
                big_batch.slice(row_offsets[i], self.span)
                if big_batch is not None
                else None
            )
            if decoded_images:
                window_rows = range(
                    g_start, g_start + self.span, self.frameskip
                )
                gather_idx = [unique_pos[r] for r in window_rows]
                steps = self._process_batch(
                    ep_idx,
                    g_start,
                    sub_batch,
                    decoded_images={
                        col: frames[gather_idx]
                        for col, frames in decoded_images.items()
                    },
                )
            else:
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
    are JPEG-encoded into ``pa.binary``. Other tabular columns become
    fixed-size lists of float32; per-step string columns are dropped with a
    warning. Column names with ``.`` are renamed to ``_`` (Lance rejects
    dots in top-level field names).

    Episode-scoped data (the ``EPISODE_DATA_KEY`` entry of an episode dict)
    goes to a ``<table>_episodes`` side table, one row per episode, flushed
    after each frames write commits. The key set must be identical for every
    episode of the table — including across append sessions.

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

        self._episodes_table_name = f'{self.table_name}_episodes'
        self._ep_keys: tuple[str, ...] | None = None
        self._ep_schema: pa.Schema | None = None
        self._ep_table = None
        self._pending_ep_rows: list[dict] = []

    def __enter__(self):
        self._db = lancedb.connect(self.uri, **self.connect_kwargs)
        tables = self._db.list_tables().tables
        if self.table_name in tables:
            if self.mode == 'error':
                raise FileExistsError(
                    f"Lance table '{self.table_name}' already exists at "
                    f"'{self.uri}'. Pass mode='overwrite' to replace it or "
                    "mode='append' to extend it."
                )
            if self.mode == 'overwrite':
                self._db.drop_table(self.table_name)
                if self._episodes_table_name in tables:
                    self._db.drop_table(self._episodes_table_name)
            else:
                self._open_existing_for_append()
        elif self._episodes_table_name in tables:
            # Frames table gone but its side table survived (e.g. an
            # interrupted overwrite): the orphan can only desync the new
            # session, so drop it.
            logging.warning(
                "LanceWriter: dropping orphaned episode-data table '%s' "
                '(no frames table found).',
                self._episodes_table_name,
            )
            self._db.drop_table(self._episodes_table_name)
        return self

    def __exit__(self, *exc):
        self._db = None
        self._table = None
        self._ep_table = None
        self._pending_ep_rows.clear()

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

        first_steps, first_extra = split_episode_data(first_ep)

        # Validate eagerly: raised here it stays a clean ValueError instead
        # of resurfacing wrapped in a RuntimeError once Lance consumes the
        # batch reader (`_batch_from_episode` re-checks each episode).
        self._check_episode_keys(first_extra)

        if not self._initialized:
            self._init_schema(first_steps)
            self._initialized = True
        elif self._appending_existing and not self._rename_map:
            self._validate_episode_against_existing(first_steps)

        def batch_gen():
            yield self._batch_from_episode(first_steps, first_extra)
            for ep in iterator:
                yield self._batch_from_episode(*split_episode_data(ep))

        reader = pa.RecordBatchReader.from_batches(self._schema, batch_gen())
        try:
            if self._table is None:
                self._table = self._db.create_table(
                    self.table_name, data=reader, schema=self._schema
                )
            else:
                self._table.add(reader)
        except BaseException:
            # The frames write did not commit; drop the side-table rows
            # queued for it so a retry cannot double-write them.
            self._pending_ep_rows.clear()
            raise
        self._flush_episode_rows()

    def _batch_from_episode(
        self, per_step: dict, ep_extra: dict
    ) -> pa.RecordBatch:
        self._check_episode_keys(ep_extra)
        ep_len = len(next(iter(per_step.values())))
        batch = self._build_batch(per_step, ep_len)
        if ep_extra:
            self._pending_ep_rows.append(
                {'episode_idx': self._ep_idx, **ep_extra}
            )
        self._ep_idx += 1
        self._global_ptr += ep_len
        return batch

    def _check_episode_keys(self, ep_extra: dict) -> None:
        self._ep_keys = _settle_episode_keys(
            self._ep_keys,
            ep_extra,
            appending=self._appending_existing,
            writer_name='LanceWriter',
            table_name=self.table_name,
            episodes_table_name=self._episodes_table_name,
        )

    def _flush_episode_rows(self) -> None:
        if not self._pending_ep_rows:
            return
        rows, self._pending_ep_rows = self._pending_ep_rows, []
        tbl, self._ep_schema = _episode_rows_table(rows, self._ep_schema)
        if self._ep_table is not None:
            self._ep_table.add(tbl)
        elif self._episodes_table_name in self._db.list_tables().tables:
            self._ep_table = self._db.open_table(self._episodes_table_name)
            self._ep_table.add(tbl)
        else:
            self._ep_table = self._db.create_table(
                self._episodes_table_name, data=tbl, schema=self._ep_schema
            )

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
            elif pa.types.is_string(f.type) or pa.types.is_large_string(
                f.type
            ):
                raise ValueError(
                    f"LanceWriter: cannot append to '{self.table_name}' — "
                    f"column '{f.name}' is a per-step string column (legacy "
                    'layout). Per-step strings are no longer supported: '
                    'episode-scoped values now live in a '
                    f"'{self._episodes_table_name}' side table. Regenerate "
                    'the dataset in the episode-data layout.'
                )
            elif pa.types.is_fixed_size_list(f.type):
                dims[f.name] = f.type.list_size
            else:
                raise ValueError(
                    f"LanceWriter: cannot append to '{self.table_name}' — "
                    f"existing column '{f.name}' has unsupported type "
                    f'{f.type}.'
                )

        if self._episodes_table_name in self._db.list_tables().tables:
            self._ep_table = self._db.open_table(self._episodes_table_name)
            self._ep_schema = self._ep_table.schema
            self._ep_keys = tuple(
                sorted(n for n in self._ep_schema.names if n != 'episode_idx')
            )
        else:
            self._ep_keys = ()

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
        skipped_strings: list[str] = []
        for col, vals in ep_data.items():
            lance_name = _to_lance_name(col)
            if lance_name in reserved:
                continue
            is_image = _is_image_name(lance_name) or is_image_column(vals)
            if not is_image:
                kind = np.asarray(vals[0]).dtype.kind
                if kind in 'US':
                    skipped_strings.append(col)
                    continue
                if kind not in 'biuf':
                    continue
            if lance_name in incoming_to_lance.values():
                raise ValueError(
                    f'LanceWriter: append failed — incoming columns map to '
                    f"the same Lance name '{lance_name}'."
                )
            incoming_to_lance[col] = lance_name
        lance_to_incoming = {v: k for k, v in incoming_to_lance.items()}

        if skipped_strings:
            logging.warning(
                'LanceWriter: dropping per-step string columns %s — '
                'per-step strings are not supported; put episode-constant '
                'values in the episode data instead.',
                skipped_strings,
            )

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

        dropped_strings: list[str] = []
        dropped_non_numeric: list[str] = []
        for col, vals in sample_ep.items():
            lance_name = _to_lance_name(col)
            if lance_name in reserved:
                continue

            is_image = _is_image_name(lance_name) or is_image_column(vals)
            if not is_image:
                sample = np.asarray(vals[0])
                if sample.dtype.kind in 'US':
                    dropped_strings.append(col)
                    continue
                if sample.dtype.kind not in 'biuf':
                    dropped_non_numeric.append(col)
                    continue

            rename_map[col] = lance_name
            ordered_cols.append(lance_name)
            if is_image:
                image_cols.add(lance_name)
            else:
                dims[lance_name] = int(sample.reshape(-1).shape[0])

        if dropped_strings:
            logging.warning(
                'LanceWriter: dropping per-step string columns %s — '
                'per-step strings are not supported; put episode-constant '
                'values in the episode data instead.',
                dropped_strings,
            )
        if dropped_non_numeric:
            logging.warning(
                'LanceWriter: dropping non-numeric columns %s — values are '
                'not convertible to float32.',
                dropped_non_numeric,
            )

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
    supports_episode_data = True

    @classmethod
    def detect(cls, path) -> bool:
        s = str(path).rstrip('/')
        if s.lower().endswith('.lance'):
            return True
        p = Path(s)
        if p.is_dir():
            # A `*_videos.lance` sibling marks a video-blob dataset; defer to
            # the `lance_video` format so it claims the directory instead.
            if any(p.glob('*_videos.lance')):
                return False
            return any(p.glob('*.lance'))
        return False

    @classmethod
    def open_reader(cls, path, **kwargs) -> LanceDataset:
        if '://' in str(path) and 'connect_kwargs' not in kwargs:
            opts = {
                'region': os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
                'virtual_hosted_style_request': 'true',
            }
            # `token` collides with AWS session token on s3:// — only inject for hf://.
            if str(path).startswith('hf://') and os.environ.get('HF_TOKEN'):
                opts['token'] = os.environ['HF_TOKEN']
            kwargs['connect_kwargs'] = {'storage_options': opts}
        return LanceDataset(path=path, **kwargs)

    @classmethod
    def open_writer(cls, path, **kwargs) -> LanceWriter:
        return LanceWriter(path, **kwargs)


__all__ = ['Lance', 'LanceDataset', 'LanceWriter']
