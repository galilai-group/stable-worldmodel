"""Lance video format: tabular frames + per-episode MP4 blobs (Lance blob v2).

Where the :mod:`lance` format stores one JPEG per frame in a ``pa.binary``
column, this format keeps image columns as **compressed video**: each episode
is encoded to an MP4 and stored verbatim in a Lance ``large_binary`` column
marked with the ``lance-encoding:blob`` metadata (blob v2). At read time
:py:meth:`lance.LanceDataset.take_blobs` streams the bytes without
materializing them into Arrow buffers, and torchcodec decodes just the frames
a window needs. The idea is borrowed from the ``lerobot-lancedb`` plugin's
``LeRobotLanceVideoDataset``; here it is adapted to swm's episode-contiguous
window model and to data that arrives as raw frames (so we encode the MP4
ourselves instead of copying source files).

Two-table layout::

    <table>.lance/          # frames table: episode_idx, step_idx, tabular cols
    <table>_videos.lance/   # one row per (episode, image col); video_bytes blob

Trade-off vs the frame-level Lance format: far smaller on disk (video codec
beats per-frame JPEG) at the cost of a seek + decode per window. Because swm
windows are episode-contiguous, a single cached decoder serves every window
drawn from the same episode.
"""

from __future__ import annotations

import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import lance
import lancedb
import pyarrow as pa

from stable_worldmodel.data.format import (
    Format,
    register_format,
    validate_write_mode,
)
from stable_worldmodel.data.formats.lance import (
    LanceDataset,
    _force_forkserver,
    _is_image_name,
    _to_lance_name,
)

_BLOB_META = {b'lance-encoding:blob': b'true'}
_DEFAULT_FPS = 10
_DEFAULT_CODEC = 'libx264'


def _is_image_vals(vals) -> bool:
    """Numpy-safe variant of ``is_image_column``: a sequence of HxWxC uint8."""
    if len(vals) == 0:
        return False
    sample = np.asarray(vals[0])
    return (
        sample.dtype == np.uint8
        and sample.ndim == 3
        and sample.shape[-1] in (1, 3)
    )


def _to_hwc_uint8(frame) -> np.ndarray:
    """Normalize a single frame to ``(H, W, 3)`` uint8 for the encoder."""
    arr = np.asarray(frame)
    if (
        arr.ndim == 3
        and arr.shape[0] in (1, 3, 4)
        and arr.shape[-1]
        not in (
            1,
            3,
            4,
        )
    ):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr.astype(np.uint8)


def _encode_video(frames, fps: int, codec: str) -> bytes:
    """Encode a sequence of frames to MP4 and return the raw bytes.

    imageio needs a real file handle for the ffmpeg muxer (the moov atom is
    written on close), so we round-trip through a temp file. ``macro_block_size
    =1`` keeps the original frame dimensions instead of padding to multiples of
    16.
    """
    import imageio

    fd, path = tempfile.mkstemp(suffix='.mp4')
    os.close(fd)
    try:
        writer = imageio.get_writer(
            path, fps=fps, codec=codec, macro_block_size=1
        )
        for frame in frames:
            writer.append_data(_to_hwc_uint8(frame))
        writer.close()
        return Path(path).read_bytes()
    finally:
        os.unlink(path)


class LanceVideoDataset(LanceDataset):
    """Reader for the two-table video-blob layout.

    Reuses :class:`LanceDataset` for all tabular columns (it reads the frames
    table) and overrides only the image path: image columns are decoded from
    per-episode MP4 blobs in the sibling ``_videos`` table.

    Args:
        path / table_name / uri: locate the frames table (the ``_videos``
            sibling is derived by name).
        video_keys: restrict which image columns to load; defaults to all
            present in the videos table.
        decoder_cache_size: per-worker LRU bound on open MP4 decoders.
        Other args match :class:`LanceDataset`.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        table_name: str | None = None,
        *,
        video_keys: list[str] | None = None,
        decoder_cache_size: int = 16,
        **kwargs: Any,
    ) -> None:
        loc = path if path is not None else kwargs.get('uri')
        if loc is None:
            raise TypeError('LanceVideoDataset requires `path` (or `uri`)')

        resolved_uri, frames_name = self._resolve(str(loc), table_name)
        videos_name = f'{frames_name}_videos'
        connect_kwargs = kwargs.get('connect_kwargs') or {}

        _force_forkserver()
        db = lancedb.connect(resolved_uri, **connect_kwargs)
        if videos_name not in db.list_tables().tables:
            raise FileNotFoundError(
                f"LanceVideoDataset: videos table '{videos_name}' not found "
                f"at '{resolved_uri}'. Write it with the 'lance_video' format."
            )
        videos_tbl = db.open_table(videos_name)
        rows = (
            videos_tbl.to_lance()
            .to_table(columns=['episode_idx', 'video_key'])
            .to_pylist()
        )
        avail = {str(r['video_key']) for r in rows}
        avail_set = avail if video_keys is None else (avail & set(video_keys))

        keys_to_load = kwargs.get('keys_to_load')
        if keys_to_load is not None:
            kwargs['keys_to_load'] = [
                k for k in keys_to_load if k not in avail_set
            ]
            wanted_vk = [k for k in keys_to_load if k in avail_set]
        else:
            wanted_vk = sorted(avail_set)

        # `_video_keys` must exist before super().__init__ runs, since the
        # overridden `_update_fetch_columns` reads it.
        self._video_keys: set[str] = set()
        kwargs['image_columns'] = []
        kwargs['connect_kwargs'] = connect_kwargs
        kwargs.pop('uri', None)
        super().__init__(resolved_uri, table_name=frames_name, **kwargs)

        self._video_keys = set(wanted_vk)
        for k in wanted_vk:
            if k not in self._keys:
                self._keys.append(k)
        self._update_fetch_columns()

        # (episode_idx, video_key) -> physical row offset for take_blobs.
        self._blob_row = {
            (int(r['episode_idx']), str(r['video_key'])): i
            for i, r in enumerate(rows)
        }
        self._videos_uri = f'{resolved_uri}/{videos_name}.lance'
        self._storage_options = connect_kwargs.get('storage_options')
        self._decoder_cache_size = decoder_cache_size
        self._videos_ds = None
        self._decoder_cache: OrderedDict | None = None

    @staticmethod
    def _resolve(loc: str, table_name: str | None) -> tuple[str, str]:
        stripped = loc.rstrip('/')
        if stripped.lower().endswith('.lance'):
            sep = stripped.rfind('/')
            parent = stripped[:sep] if sep >= 0 else '.'
            leaf = stripped[sep + 1 :] if sep >= 0 else stripped
            return parent, leaf[: -len('.lance')]
        if table_name is not None:
            return stripped, table_name
        p = Path(stripped)
        if p.is_dir():
            cands = [
                t
                for t in sorted(p.glob('*.lance'))
                if not t.stem.endswith('_videos')
            ]
            if len(cands) == 1:
                return str(p), cands[0].stem
            if len(cands) > 1:
                raise ValueError(
                    f'Ambiguous lance_video dataset in {p}: {[c.name for c in cands]}. '
                    'Pass `table_name=` explicitly.'
                )
        raise ValueError(
            f'LanceVideoDataset: cannot infer frames table from {loc!r}.'
        )

    def _update_fetch_columns(self) -> None:
        cached = set(self._cache.keys()) | self._video_keys
        self._fetch_columns = [k for k in self._keys if k not in cached]
        if not self._fetch_columns:
            self._perm = None

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state['_videos_ds'] = None
        state['_decoder_cache'] = None
        return state

    def _ensure_videos_open(self) -> None:
        if self._videos_ds is None:
            self._videos_ds = lance.dataset(
                self._videos_uri, storage_options=self._storage_options
            )
            self._decoder_cache = OrderedDict()

    def _decoder_for(self, ep_idx: int, vkey: str):
        from torchcodec.decoders import VideoDecoder

        cache = self._decoder_cache
        key = (ep_idx, vkey)
        dec = cache.get(key)
        if dec is not None:
            cache.move_to_end(key)
            return dec
        row = self._blob_row[key]
        blob = self._videos_ds.take_blobs(
            blob_column='video_bytes', indices=[row]
        )[0]
        try:
            data = blob.readall()
        finally:
            blob.close()
        dec = VideoDecoder(data, seek_mode='approximate')
        cache[key] = dec
        while len(cache) > self._decoder_cache_size:
            cache.popitem(last=False)
        return dec

    def _decode_video_window(
        self, ep_idx: int, vkey: str, local_start: int, num_steps: int
    ) -> torch.Tensor:
        self._ensure_videos_open()
        dec = self._decoder_for(ep_idx, vkey)
        indices = [local_start + k * self.frameskip for k in range(num_steps)]
        return dec.get_frames_at(indices=indices).data  # (T, C, H, W) uint8

    def _process_batch(
        self, ep_idx: int, g_start: int, batch, g_end: int | None = None
    ) -> dict:
        if g_end is None:
            g_end = g_start + self.span
        local_start = g_start - int(self.offsets[ep_idx])
        # Number of frame-skipped frames spanning [g_start, g_end). For a
        # fixed window this is `num_steps`, but `load_episode` passes a
        # full-episode slice — decode every frame it covers, matching the
        # frame-skipped tabular columns instead of truncating to one window.
        num_steps = -(-(g_end - g_start) // self.frameskip)
        steps: dict[str, Any] = {}
        for col in self._keys:
            if col in self._video_keys:
                steps[col] = self._decode_video_window(
                    ep_idx, col, local_start, num_steps
                )
            else:
                steps[col] = self._process_col(col, batch, g_start, g_end)
        return steps

    def __getitems__(self, indices: list[int]) -> list[dict]:
        """Batched read with one decode call per (episode, video key).

        The parent decodes a window at a time; here we collect every frame
        index a batch needs from each episode's MP4 and issue a single
        ``get_frames_at`` per decoder. Overlapping strided windows then
        decode each underlying frame once instead of once per window — the
        same flattening the lerobot-lancedb video reader does, and the
        difference between video throughput trailing the JPEG format by ~4×
        (per-window) versus ~1.2× (batched).
        """
        # Lay out window rows; fetch all tabular columns in one shot.
        all_rows: list[int] = []
        row_offsets: list[int] = []
        sample_meta: list[tuple[int, int, int]] = []
        for idx in indices:
            ep_idx, start = self.clip_indices[idx]
            g_start = int(self.offsets[ep_idx] + start)
            row_offsets.append(len(all_rows))
            all_rows.extend(range(g_start, g_start + self.span))
            sample_meta.append((ep_idx, g_start, start))

        big_batch = None
        if self._fetch_columns and all_rows:
            self._ensure_open()
            unique_rows = sorted(set(all_rows))
            unique_batch = self._perm.__getitems__(unique_rows)
            if len(unique_rows) == len(all_rows) and all_rows == unique_rows:
                big_batch = unique_batch
            else:
                row_lookup = {r: i for i, r in enumerate(unique_rows)}
                gather = pa.array(
                    [row_lookup[r] for r in all_rows], type=pa.int64()
                )
                big_batch = unique_batch.take(gather)

        # One decode per (episode, video key): union the per-window indices,
        # decode once, scatter the frames back to each window.
        video_out: dict[tuple[int, str], Any] = {}
        if self._video_keys:
            self._ensure_videos_open()
            plan: dict[tuple[int, str], list[tuple[int, list[int]]]] = {}
            for i, (ep_idx, _g, start) in enumerate(sample_meta):
                idxs = [
                    start + k * self.frameskip for k in range(self.num_steps)
                ]
                for vkey in self._video_keys:
                    plan.setdefault((ep_idx, vkey), []).append((i, idxs))
            for (ep_idx, vkey), items in plan.items():
                dec = self._decoder_for(ep_idx, vkey)
                union = sorted({j for _, idxs in items for j in idxs})
                frames = dec.get_frames_at(indices=union).data
                pos = {ix: j for j, ix in enumerate(union)}
                for sample_i, idxs in items:
                    video_out[(sample_i, vkey)] = frames[
                        [pos[j] for j in idxs]
                    ]

        results: list[dict] = []
        for i, (ep_idx, g_start, _start) in enumerate(sample_meta):
            sub_batch = (
                big_batch.slice(row_offsets[i], self.span)
                if big_batch is not None
                else None
            )
            steps: dict[str, Any] = {}
            for col in self._keys:
                if col in self._video_keys:
                    steps[col] = video_out[(i, col)]
                else:
                    steps[col] = self._process_col(
                        col, sub_batch, g_start, g_start + self.span
                    )
            if self.transform:
                steps = self.transform(steps)
            if 'action' in steps:
                steps['action'] = steps['action'].reshape(self.num_steps, -1)
            results.append(steps)
        return results


class LanceVideoWriter:
    """Append episodes; image columns become one MP4 blob per episode.

    Image columns (names ``pixels`` / ``pixels_<view>`` or uint8 HxWxC arrays)
    are encoded to MP4 and stored in the ``_videos`` table as blob-v2
    ``large_binary``. Everything else lands in the frames table as fixed-size
    float32 lists, exactly like :class:`~stable_worldmodel.data.formats.lance.LanceWriter`.

    Video blobs are compressed and buffered until close (raw frames are encoded
    and discarded per episode, so memory stays bounded to the codec output).
    """

    def __init__(
        self,
        path: str | Path,
        table_name: str | None = None,
        *,
        fps: int = _DEFAULT_FPS,
        codec: str = _DEFAULT_CODEC,
        connect_kwargs: dict[str, Any] | None = None,
        mode: str = 'append',
    ) -> None:
        validate_write_mode(mode)
        loc = str(path).rstrip('/')
        if table_name is not None:
            self.uri = loc
            self.table_name = table_name
        elif loc.lower().endswith('.lance'):
            p = Path(loc)
            self.uri = str(p.parent) if str(p.parent) else '.'
            self.table_name = p.stem
        else:
            # Treat as a dataset directory holding both tables; name the
            # frames table after the directory (mirrors the reader's
            # directory-based resolution).
            self.uri = loc
            self.table_name = Path(loc).name

        Path(self.uri).mkdir(parents=True, exist_ok=True)
        self.videos_name = f'{self.table_name}_videos'
        self.fps = fps
        self.codec = codec
        self.connect_kwargs = connect_kwargs or {}
        self.mode = mode

        self._db = None
        self._appending = False
        self._frames_schema: pa.Schema | None = None
        self._rename_map: dict[str, str] = {}
        self._image_cols: set[str] = set()
        self._dims: dict[str, int] = {}
        self._ep_idx = 0
        self._frames_batches: list[pa.RecordBatch] = []
        self._video_rows: list[tuple[int, str, bytes]] = []

    def __enter__(self):
        self._db = lancedb.connect(self.uri, **self.connect_kwargs)
        existing = self._db.list_tables().tables
        present = self.table_name in existing
        if present:
            if self.mode == 'error':
                raise FileExistsError(
                    f"Lance table '{self.table_name}' already exists at "
                    f"'{self.uri}'. Pass mode='overwrite' or mode='append'."
                )
            if self.mode == 'overwrite':
                self._db.drop_table(self.table_name)
                if self.videos_name in existing:
                    self._db.drop_table(self.videos_name)
            else:
                self._load_existing_state()
        return self

    def __exit__(self, *exc):
        self.close()
        self._db = None

    def _load_existing_state(self) -> None:
        table = self._db.open_table(self.table_name)
        self._frames_schema = table.schema
        for f in table.schema:
            if f.name in ('episode_idx', 'step_idx'):
                continue
            if pa.types.is_fixed_size_list(f.type):
                self._dims[f.name] = f.type.list_size
        ep_col = (
            table.to_lance()
            .to_table(columns=['episode_idx'])
            .column('episode_idx')
        )
        self._ep_idx = int(ep_col.to_numpy().max()) + 1 if len(ep_col) else 0
        videos = self._db.open_table(self.videos_name)
        self._image_cols = {
            str(v)
            for v in videos.to_lance()
            .to_table(columns=['video_key'])
            .column('video_key')
            .to_pylist()
        }
        self._appending = True

    def write_episode(self, ep_data: dict) -> None:
        if self._db is None:
            raise RuntimeError(
                'LanceVideoWriter used outside of a `with` block'
            )
        self._consume_episodes([ep_data])

    def write_episodes(self, episodes) -> None:
        if self._db is None:
            raise RuntimeError(
                'LanceVideoWriter used outside of a `with` block'
            )
        self._consume_episodes(episodes)

    def _consume_episodes(self, episodes) -> None:
        for ep_data in episodes:
            if self._frames_schema is None:
                self._init_schema(ep_data)
            elif not self._rename_map:
                # Appending to an existing dataset: `_load_existing_state`
                # recovered the schema but not the rename map (it depends on
                # the incoming column names). Rebuild it from the first
                # appended episode before any batch is built.
                self._rebuild_rename_map(ep_data)
            self._frames_batches.append(self._frames_batch(ep_data))
            for col, lance_name in self._rename_map.items():
                if lance_name in self._image_cols:
                    self._video_rows.append(
                        (
                            self._ep_idx,
                            lance_name,
                            _encode_video(ep_data[col], self.fps, self.codec),
                        )
                    )
            self._ep_idx += 1

    def _rebuild_rename_map(self, sample_ep: dict) -> None:
        """Recover the original→on-disk column map when reopening to append.

        :meth:`_load_existing_state` reads the frames schema, ``_dims`` and
        ``_image_cols`` back from disk, but ``_rename_map`` keys on the
        *incoming* column names and so cannot be recovered from the table
        alone. Rebuild it from the first appended episode — mirrors
        :meth:`LanceWriter._validate_episode_against_existing`. Without this
        the empty map makes :meth:`_frames_batch` raise ``StopIteration`` and
        the video-row loop emit no blobs.
        """
        reserved = {'episode_idx', 'step_idx'}
        known = set(self._dims) | self._image_cols
        rename_map: dict[str, str] = {}
        for col in sample_ep:
            lance_name = _to_lance_name(col)
            if lance_name in reserved or lance_name not in known:
                continue
            rename_map[col] = lance_name
        missing = known - set(rename_map.values())
        if missing:
            raise ValueError(
                f"LanceVideoWriter: append failed — table '{self.table_name}' "
                f'expects columns {sorted(missing)} that the incoming episode '
                'does not provide.'
            )
        self._rename_map = rename_map

    def _init_schema(self, sample_ep: dict) -> None:
        reserved = {'episode_idx', 'step_idx'}
        fields = [
            pa.field('episode_idx', pa.int32()),
            pa.field('step_idx', pa.int32()),
        ]
        for col, vals in sample_ep.items():
            lance_name = _to_lance_name(col)
            if lance_name in reserved:
                continue
            self._rename_map[col] = lance_name
            if _is_image_name(lance_name) or _is_image_vals(vals):
                self._image_cols.add(lance_name)
                continue
            dim = int(np.asarray(vals[0]).reshape(-1).shape[0])
            self._dims[lance_name] = dim
            fields.append(pa.field(lance_name, pa.list_(pa.float32(), dim)))
        self._frames_schema = pa.schema(fields)

    def _frames_batch(self, ep_data: dict) -> pa.RecordBatch:
        ep_len = len(next(iter(ep_data.values())))
        arrays: list[pa.Array] = [
            pa.array(np.full(ep_len, self._ep_idx, dtype=np.int32)),
            pa.array(np.arange(ep_len, dtype=np.int32)),
        ]
        for f in self._frames_schema:
            if f.name in ('episode_idx', 'step_idx'):
                continue
            col = next(c for c, ln in self._rename_map.items() if ln == f.name)
            dim = self._dims[f.name]
            flat = np.asarray(ep_data[col], dtype=np.float32).reshape(-1)
            arrays.append(
                pa.FixedSizeListArray.from_arrays(
                    pa.array(flat, type=pa.float32()), dim
                )
            )
        return pa.record_batch(arrays, schema=self._frames_schema)

    def _videos_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field('episode_idx', pa.int32()),
                pa.field('video_key', pa.string()),
                pa.field(
                    'video_bytes', pa.large_binary(), metadata=_BLOB_META
                ),
            ]
        )

    def close(self) -> None:
        if self._db is None or not self._frames_batches:
            return
        frames_reader = pa.RecordBatchReader.from_batches(
            self._frames_schema, iter(self._frames_batches)
        )
        videos_schema = self._videos_schema()
        videos_batch = pa.record_batch(
            [
                pa.array([e for e, _, _ in self._video_rows], type=pa.int32()),
                pa.array(
                    [k for _, k, _ in self._video_rows], type=pa.string()
                ),
                pa.array(
                    [b for _, _, b in self._video_rows], type=pa.large_binary()
                ),
            ],
            schema=videos_schema,
        )
        if self._appending:
            self._db.open_table(self.table_name).add(frames_reader)
            self._db.open_table(self.videos_name).add(videos_batch)
        else:
            self._db.create_table(
                self.table_name,
                data=frames_reader,
                schema=self._frames_schema,
            )
            self._db.create_table(
                self.videos_name, data=videos_batch, schema=videos_schema
            )
        self._frames_batches.clear()
        self._video_rows.clear()


@register_format
class LanceVideo(Format):
    name = 'lance_video'

    @classmethod
    def detect(cls, path) -> bool:
        p = Path(str(path).rstrip('/'))
        return p.is_dir() and any(p.glob('*_videos.lance'))

    @classmethod
    def open_reader(cls, path, **kwargs) -> LanceVideoDataset:
        if '://' in str(path) and 'connect_kwargs' not in kwargs:
            opts = {
                'region': os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
                'virtual_hosted_style_request': 'true',
            }
            if str(path).startswith('hf://') and os.environ.get('HF_TOKEN'):
                opts['token'] = os.environ['HF_TOKEN']
            kwargs['connect_kwargs'] = {'storage_options': opts}
        return LanceVideoDataset(path=path, **kwargs)

    @classmethod
    def open_writer(cls, path, **kwargs) -> LanceVideoWriter:
        return LanceVideoWriter(path, **kwargs)


__all__ = ['LanceVideo', 'LanceVideoDataset', 'LanceVideoWriter']
