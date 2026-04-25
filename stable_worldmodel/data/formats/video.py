"""Video format: tabular .npz columns + one .mp4 per episode for image keys."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stable_worldmodel.data.format import Format, register_format
from stable_worldmodel.data.formats.utils import is_image_column
from stable_worldmodel.data.formats.folder import FolderDataset


class VideoDataset(FolderDataset):
    """Loads frames from MP4 files (one per episode) using decord."""

    _decord: Any = None  # Lazy module reference

    def __init__(
        self,
        name: str | None = None,
        video_keys: list[str] | None = None,
        **kw: Any,
    ) -> None:
        if VideoDataset._decord is None:
            try:
                import decord

                decord.bridge.set_bridge('torch')
                VideoDataset._decord = decord
            except ImportError:
                raise ImportError('VideoDataset requires decord')
        super().__init__(name=name, folder_keys=video_keys or ['video'], **kw)

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


class VideoWriter:
    """Append episodes; image columns are encoded as one MP4 per episode.

    Layout::

        <root>/
          ep_len.npz, ep_offset.npz
          <col>.npz                 # tabular columns
          <img_col>/ep_<i>.mp4      # one video per episode per image col
    """

    def __init__(self, path, fps: int = 25, codec: str = 'libx264'):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.codec = codec
        self._tabular: dict[str, list[np.ndarray]] = {}
        self._lengths: list[int] = []
        self._offsets: list[int] = []
        self._global_ptr = 0
        self._ep_idx = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        np.savez(self.path / 'ep_len.npz', np.asarray(self._lengths, np.int32))
        np.savez(
            self.path / 'ep_offset.npz', np.asarray(self._offsets, np.int64)
        )
        for col, parts in self._tabular.items():
            np.savez(self.path / f'{col}.npz', np.concatenate(parts, axis=0))

    def write_episode(self, ep_data: dict) -> None:
        import imageio

        ep_len = len(next(iter(ep_data.values())))
        for col, vals in ep_data.items():
            if is_image_column(vals):
                col_dir = self.path / col
                col_dir.mkdir(exist_ok=True)
                writer = imageio.get_writer(
                    str(col_dir / f'ep_{self._ep_idx}.mp4'),
                    fps=self.fps,
                    codec=self.codec,
                )
                for frame in vals:
                    arr = np.asarray(frame)
                    if arr.shape[-1] == 1:
                        arr = np.repeat(arr, 3, axis=-1)
                    writer.append_data(arr)
                writer.close()
            else:
                self._tabular.setdefault(col, []).append(np.asarray(vals))

        self._lengths.append(ep_len)
        self._offsets.append(self._global_ptr)
        self._global_ptr += ep_len
        self._ep_idx += 1


@register_format
class Video(Format):
    name = 'video'

    @classmethod
    def detect(cls, path) -> bool:
        p = Path(path)
        if not p.is_dir() or not (p / 'ep_len.npz').exists():
            return False
        for sub in p.iterdir():
            if sub.is_dir() and any(sub.glob('*.mp4')):
                return True
        return False

    @classmethod
    def open_reader(cls, path, **kwargs) -> VideoDataset:
        return VideoDataset(path=path, **kwargs)

    @classmethod
    def open_writer(cls, path, **kwargs) -> VideoWriter:
        return VideoWriter(path, **kwargs)


__all__ = ['Video', 'VideoDataset', 'VideoWriter']
