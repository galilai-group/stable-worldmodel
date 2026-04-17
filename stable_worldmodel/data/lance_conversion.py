"""Utilities for converting stable-worldmodel HDF5 datasets to LanceDB tables."""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pyarrow as pa
from PIL import Image
from tqdm import tqdm

import lancedb


def _table_exists(db: Any, name: str) -> bool:
    for entry in db.list_tables():
        if isinstance(entry, str) and entry == name:
            return True
        if isinstance(entry, dict) and entry.get('name') == name:
            return True
    return False

DEFAULT_JPEG_QUALITY = 95
DEFAULT_BATCH_ROWS = 4096


def _to_lance_name(name: str) -> str:
    """Sanitise an HDF5 column name for use as a Lance field name.

    Lance uses ``.`` as a struct-field path separator and rejects top-level
    field names that contain it.  This function replaces every dot with an
    underscore so callers never need to think about the constraint.

    Examples::

        >>> _to_lance_name('pixels.top')
        'pixels_top'
        >>> _to_lance_name('observation.state')
        'observation_state'
        >>> _to_lance_name('action')
        'action'
    """
    return name.replace('.', '_')


def _is_image_column(name: str) -> bool:
    """Return True for columns that follow the image naming convention.

    Recognised patterns after Lance-name sanitisation: ``pixels`` (single
    camera) or ``pixels_<view>`` (multi-camera, e.g. ``pixels_top``).
    """
    return name == 'pixels' or name.startswith('pixels_')


def _encode_frame(frame: np.ndarray, jpeg_quality: int) -> bytes:
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    buffer = io.BytesIO()
    Image.fromarray(frame.astype(np.uint8)).save(
        buffer, format='JPEG', quality=jpeg_quality
    )
    return buffer.getvalue()


def _build_schema(columns: list[str], dims: dict[str, int], image_keys: list[str]) -> pa.Schema:
    image_key_set = set(image_keys)
    fields = [
        pa.field('episode_idx', pa.int32()),
        pa.field('step_idx', pa.int32()),
    ]
    for col in columns:
        if col in image_key_set:
            fields.append(pa.field(col, pa.binary()))
        else:
            fields.append(pa.field(col, pa.list_(pa.float32(), dims[col])))
    return pa.schema(fields)


def _episode_and_step_arrays(
    lengths: np.ndarray, limit_rows: int
) -> tuple[np.ndarray, np.ndarray, int]:
    episode_idx = np.empty(limit_rows, dtype=np.int32)
    step_idx = np.empty(limit_rows, dtype=np.int32)
    cursor = 0
    ep_count = 0
    for ep_length in lengths.tolist():
        if cursor >= limit_rows:
            break
        span = min(ep_length, limit_rows - cursor)
        episode_idx[cursor : cursor + span] = ep_count
        step_idx[cursor : cursor + span] = np.arange(span, dtype=np.int32)
        cursor += span
        ep_count += 1
    return episode_idx, step_idx, ep_count


def _record_batch_generator(
    f: h5py.File,
    *,
    columns: list[str],
    rename_map: dict[str, str],
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
    dims: dict[str, int],
    image_keys: list[str],
    jpeg_quality: int,
    batch_rows: int,
    schema: pa.Schema,
) -> Iterable[pa.RecordBatch]:
    """Yield Arrow RecordBatches from the HDF5 file.

    ``columns`` are the original HDF5 key names used for reading.
    ``rename_map`` maps each HDF5 name to its sanitised Lance name.
    ``image_keys`` and ``dims`` are keyed by Lance names.
    """
    total = len(episode_idx)
    image_key_set = set(image_keys)
    # non-image columns as (hdf5_name, lance_name) pairs
    non_image = [
        (orig, rename_map[orig])
        for orig in columns
        if rename_map[orig] not in image_key_set
    ]

    def _encode_column(data: np.ndarray) -> list[bytes]:
        # Cap threads to avoid spawning thousands for large batch_rows values.
        n_workers = min(32, len(data))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            return list(pool.map(lambda frame: _encode_frame(frame, jpeg_quality), data))

    # Build a reverse map: lance_name → hdf5_name for image columns
    reverse_map = {v: k for k, v in rename_map.items()}

    for start in tqdm(range(0, total, batch_rows), desc='Converting', unit='batch'):
        end = min(total, start + batch_rows)

        batch_arrays: dict[str, pa.Array] = {
            'episode_idx': pa.array(episode_idx[start:end], type=pa.int32()),
            'step_idx': pa.array(step_idx[start:end], type=pa.int32()),
        }

        for lance_img_key in image_keys:
            h5_key = reverse_map[lance_img_key]
            img_data = np.array(f[h5_key][start:end])
            batch_arrays[lance_img_key] = pa.array(_encode_column(img_data), type=pa.binary())

        for h5_col, lance_col in non_image:
            data = np.array(f[h5_col][start:end], dtype=np.float32)
            flat = data.reshape(end - start, -1).tolist()
            batch_arrays[lance_col] = pa.array(flat, type=pa.list_(pa.float32(), dims[lance_col]))

        yield pa.record_batch(batch_arrays, schema=schema)


def convert_hdf5_to_lance(
    *,
    h5_path: str | Path,
    lance_uri: str,
    table_name: str,
    columns: list[str] | None = None,
    image_keys: list[str] | None = None,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    overwrite: bool = False,
    max_episodes: int | None = None,
    max_steps: int | None = None,
    connect_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert an HDF5 dataset to a LanceDB table.

    Args:
        h5_path: Path to the source ``.h5`` file.
        lance_uri: LanceDB URI (local path or ``s3://`` bucket).
        table_name: Name of the table to create inside the database.
        columns: Columns to include. Defaults to all columns except the HDF5
            metadata keys ``ep_len`` and ``ep_offset``.
        image_keys: Columns that contain image data (stored as JPEG blobs).
            Defaults to auto-detection: any column named ``pixels`` or
            following the ``pixels_<view>`` multi-camera convention (after
            dot-to-underscore sanitisation). HDF5 names with dots are
            accepted — they are renamed transparently for Lance.
        jpeg_quality: JPEG encoding quality (0-100, default 95).
        batch_rows: Number of rows per Arrow record batch during conversion.
        overwrite: Drop the existing table before writing.
        max_episodes: Truncate to this many episodes.
        max_steps: Truncate to this many total steps.
        connect_kwargs: Extra kwargs forwarded to :func:`lancedb.connect`.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    connect_kwargs = connect_kwargs or {}
    db = lancedb.connect(lance_uri, **connect_kwargs)

    if _table_exists(db, table_name):
        if overwrite:
            db.drop_table(table_name)
        else:
            raise FileExistsError(
                f"Table '{table_name}' already exists at {lance_uri}. Use overwrite=True."
            )

    with h5py.File(h5_path, 'r') as f:
        if columns is None:
            columns = [
                key for key in f.keys() if key not in ('ep_len', 'ep_offset')
            ]

        # Sanitise column names for Lance: replace dots with underscores.
        # Lance rejects top-level field names containing '.' (it uses dot
        # notation as a struct-field path separator). We do this transparently
        # so callers never need to rename their HDF5 files.
        rename_map = {col: _to_lance_name(col) for col in columns}
        renamed_cols = [rename_map[c] for c in columns]
        if any(k != v for k, v in rename_map.items()):
            import logging as _logging
            _logging.getLogger(__name__).info(
                'Renamed columns for Lance compatibility: %s',
                {k: v for k, v in rename_map.items() if k != v},
            )

        # Auto-detect image columns from naming convention (after renaming).
        if image_keys is None:
            image_keys = [rename_map[c] for c in columns if _is_image_column(rename_map[c])]
        else:
            # Caller may have passed original HDF5 names or already-sanitised names.
            image_keys = [_to_lance_name(k) for k in image_keys]
        if not image_keys:
            raise ValueError(
                f"No image columns found in {renamed_cols}. Pass image_keys explicitly "
                "or name image columns 'pixels' or 'pixels_<view>'."
            )
        missing = [k for k in image_keys if k not in renamed_cols]
        if missing:
            raise ValueError(f"image_keys {missing} not found in columns {renamed_cols}")

        total_rows = len(f[columns[renamed_cols.index(image_keys[0])]])
        ep_lengths = np.array(f['ep_len'], dtype=np.int64)
        limit_rows = total_rows
        if max_episodes is not None:
            if max_episodes <= 0:
                raise ValueError('max_episodes must be positive if provided')
            limit_rows = min(limit_rows, int(ep_lengths[:max_episodes].sum()))
        if max_steps is not None:
            if max_steps <= 0:
                raise ValueError('max_steps must be positive if provided')
            limit_rows = min(limit_rows, max_steps)
        if limit_rows <= 0:
            raise ValueError('No rows selected for conversion')

        episode_idx, step_idx, used_episodes = _episode_and_step_arrays(
            ep_lengths, limit_rows
        )

        image_key_set = set(image_keys)
        dims: dict[str, int] = {}
        for orig, lance in rename_map.items():
            if lance in image_key_set:
                continue
            sample = np.array(f[orig][0])
            dims[lance] = int(sample.reshape(-1).shape[0])

        schema = _build_schema(renamed_cols, dims, image_keys)
        batches = _record_batch_generator(
            f,
            columns=columns,
            rename_map=rename_map,
            episode_idx=episode_idx,
            step_idx=step_idx,
            dims=dims,
            image_keys=image_keys,
            jpeg_quality=jpeg_quality,
            batch_rows=batch_rows,
            schema=schema,
        )
        reader = pa.RecordBatchReader.from_batches(schema, batches)
        db.create_table(table_name, data=reader, schema=schema)

    final_rows = len(db.open_table(table_name))
    return {
        'rows': final_rows,
        'episodes': used_episodes,
        'table': table_name,
        'uri': lance_uri,
    }
