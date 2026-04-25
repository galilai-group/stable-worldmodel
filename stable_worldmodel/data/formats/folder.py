"""Folder format: a directory with .npz tabular columns + per-step image files."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from stable_worldmodel.data.dataset import Dataset
from stable_worldmodel.data.format import Format, register_format
from stable_worldmodel.data.formats.utils import is_image_column
from stable_worldmodel.data.utils import get_cache_dir


class FolderDataset(Dataset):
    """Dataset loading from a folder structure.

    Tabular columns are stored as .npz files; image columns are stored as
    one image file per step under ``<key>/ep_<i>_step_<j>.jpeg``.
    """

    def __init__(
        self,
        name: str | None = None,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
        keys_to_load: list[str] | None = None,
        folder_keys: list[str] | None = None,
        cache_dir: str | Path | None = None,
        path: str | Path | None = None,
    ) -> None:
        if path is not None:
            self.path = Path(path)
        else:
            if name is None:
                raise TypeError(
                    'FolderDataset requires either `name` or `path`'
                )
            self.path = (
                Path(cache_dir or get_cache_dir(sub_folder='datasets')) / name
            )
        # Auto-detect folder keys from on-disk layout: any subdirectory whose
        # name isn't a metadata key is an image/video column.
        if folder_keys is None:
            folder_keys = [p.name for p in self.path.iterdir() if p.is_dir()]
        self.folder_keys = folder_keys
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
    """Convenience alias: ``FolderDataset`` with ``pixels`` as the image folder."""

    def __init__(
        self,
        name: str | None = None,
        image_keys: list[str] | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(name=name, folder_keys=image_keys or ['pixels'], **kw)


class FolderWriter:
    """Append episodes to a folder dataset.

    Layout::

        <root>/
          ep_len.npz, ep_offset.npz
          <col>.npz                            # tabular columns
          <img_col>/ep_<i>_step_<j>.jpeg       # per-step image files

    Image columns are auto-detected: any value that is a ``uint8`` array
    shaped ``(H, W, 3)`` or ``(H, W, 1)`` is written as JPEG.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
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
        ep_len = len(next(iter(ep_data.values())))
        for col, vals in ep_data.items():
            if is_image_column(vals):
                col_dir = self.path / col
                col_dir.mkdir(exist_ok=True)
                for step, frame in enumerate(vals):
                    arr = np.asarray(frame)
                    if arr.shape[-1] == 1:
                        arr = arr.squeeze(-1)
                    Image.fromarray(arr).save(
                        col_dir / f'ep_{self._ep_idx}_step_{step}.jpeg'
                    )
            else:
                self._tabular.setdefault(col, []).append(np.asarray(vals))

        self._lengths.append(ep_len)
        self._offsets.append(self._global_ptr)
        self._global_ptr += ep_len
        self._ep_idx += 1


@register_format
class Folder(Format):
    name = 'folder'

    @classmethod
    def detect(cls, path) -> bool:
        p = Path(path)
        if not p.is_dir() or not (p / 'ep_len.npz').exists():
            return False
        # Folder is the fallback for directories with ep_len.npz; if any
        # subfolder contains .mp4 files, the video format wins instead.
        for sub in p.iterdir():
            if sub.is_dir() and any(sub.glob('*.mp4')):
                return False
        return True

    @classmethod
    def open_reader(cls, path, **kwargs) -> FolderDataset:
        return FolderDataset(path=path, **kwargs)

    @classmethod
    def open_writer(cls, path, **kwargs) -> FolderWriter:
        return FolderWriter(path, **kwargs)


__all__ = ['Folder', 'FolderDataset', 'FolderWriter', 'ImageDataset']
