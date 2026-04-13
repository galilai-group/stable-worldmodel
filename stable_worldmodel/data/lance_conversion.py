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

DEFAULT_IMAGE_KEY = 'pixels'
DEFAULT_JPEG_QUALITY = 95
DEFAULT_BATCH_ROWS = 4096


def _encode_frame(frame: np.ndarray, jpeg_quality: int) -> bytes:
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    buffer = io.BytesIO()
    Image.fromarray(frame.astype(np.uint8)).save(
        buffer, format='JPEG', quality=jpeg_quality
    )
    return buffer.getvalue()


def _build_schema(columns: list[str], dims: dict[str, int]) -> pa.Schema:
    fields = [
        pa.field('episode_idx', pa.int32()),
        pa.field('step_idx', pa.int32()),
        pa.field('pixels', pa.binary()),
        pa.field('pixels_h', pa.int16()),
        pa.field('pixels_w', pa.int16()),
    ]
    for col in columns:
        if col == DEFAULT_IMAGE_KEY:
            continue
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
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
    dims: dict[str, int],
    image_key: str,
    jpeg_quality: int,
    batch_rows: int,
    schema: pa.Schema,
    height: int,
    width: int,
) -> Iterable[pa.RecordBatch]:
    total = len(episode_idx)
    non_pixels = [col for col in columns if col != image_key]

    def _batch_pixels(data: np.ndarray) -> list[bytes]:
        with ThreadPoolExecutor() as pool:
            return list(pool.map(lambda frame: _encode_frame(frame, jpeg_quality), data))

    for start in tqdm(range(0, total, batch_rows), desc='Converting', unit='batch'):
        end = min(total, start + batch_rows)
        pixel_batch = np.array(f[image_key][start:end])
        other = {col: np.array(f[col][start:end], dtype=np.float32) for col in non_pixels}

        px_bytes = _batch_pixels(pixel_batch)

        batch_arrays = {
            'episode_idx': pa.array(episode_idx[start:end], type=pa.int32()),
            'step_idx': pa.array(step_idx[start:end], type=pa.int32()),
            'pixels': pa.array(px_bytes, type=pa.binary()),
            'pixels_h': pa.array([height] * (end - start), type=pa.int16()),
            'pixels_w': pa.array([width] * (end - start), type=pa.int16()),
        }

        for col in non_pixels:
            flat = other[col].reshape(end - start, -1).tolist()
            batch_arrays[col] = pa.array(
                flat, type=pa.list_(pa.float32(), dims[col])
            )

        yield pa.record_batch(batch_arrays, schema=schema)


def convert_hdf5_to_lance(
    *,
    h5_path: str | Path,
    lance_uri: str,
    table_name: str,
    columns: list[str] | None = None,
    image_key: str = DEFAULT_IMAGE_KEY,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    overwrite: bool = False,
    max_episodes: int | None = None,
    max_steps: int | None = None,
    connect_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
                key
                for key in f.keys()
                if key not in ('ep_len', 'ep_offset')
            ]
        if image_key not in columns:
            raise ValueError(f"image_key '{image_key}' not found in columns {columns}")

        total_rows = len(f[image_key])
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

        dims: dict[str, int] = {}
        for col in columns:
            if col == image_key:
                continue
            sample = np.array(f[col][0])
            dims[col] = int(sample.reshape(-1).shape[0])

        sample_frame = np.array(f[image_key][0])
        if sample_frame.ndim == 3 and sample_frame.shape[0] in (1, 3, 4):
            _, height, width = sample_frame.shape
        else:
            height, width = sample_frame.shape[:2]

        schema = _build_schema(columns, dims)
        batches = _record_batch_generator(
            f,
            columns=columns,
            episode_idx=episode_idx,
            step_idx=step_idx,
            dims=dims,
            image_key=image_key,
            jpeg_quality=jpeg_quality,
            batch_rows=batch_rows,
            schema=schema,
            height=height,
            width=width,
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
