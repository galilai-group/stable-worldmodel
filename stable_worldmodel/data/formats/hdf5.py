"""HDF5 format: single .h5 file with per-column datasets + ep_len/ep_offset."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from stable_worldmodel.data.dataset import Dataset
from stable_worldmodel.data.format import Format, register_format
from stable_worldmodel.data.utils import get_cache_dir


class HDF5Dataset(Dataset):
    """Dataset loading from a single HDF5 file (SWMR mode for safe reads)."""

    def __init__(
        self,
        name: str | None = None,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        keys_to_merge: dict[str, list[str] | str] | None = None,
        cache_dir: str | Path | None = None,
        path: str | Path | None = None,
    ) -> None:
        if path is not None:
            self.h5_path = Path(path)
        else:
            if name is None:
                raise TypeError('HDF5Dataset requires either `name` or `path`')
            datasets_dir = get_cache_dir(cache_dir, sub_folder='datasets')
            self.h5_path = Path(datasets_dir, f'{name}.h5')

        self.h5_file: h5py.File | None = None
        self._cache: dict[str, np.ndarray] = {}

        with h5py.File(self.h5_path, 'r') as f:
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
            self.h5_file = h5py.File(
                self.h5_path, 'r', swmr=True, rdcc_nbytes=256 * 1024 * 1024
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


class HDF5Writer:
    """Append episodes to a single HDF5 file. Schema is inferred from the
    first episode and locked thereafter."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f: h5py.File | None = None
        self._initialized = False
        self._ep_written = 0
        self._global_ptr = 0

    def __enter__(self):
        self._f = h5py.File(str(self.path), 'w', libver='latest')
        return self

    def __exit__(self, *exc):
        if self._f is not None:
            self._f.close()
            self._f = None

    def write_episode(self, ep_data: dict) -> None:
        if self._f is None:
            raise RuntimeError('HDF5Writer used outside of a `with` block')
        if not self._initialized:
            self._init_schema(ep_data)
            self._initialized = True

        ep_len = len(next(iter(ep_data.values())))
        for col, vals in ep_data.items():
            ds = self._f[col]
            ds.resize(self._global_ptr + ep_len, axis=0)
            ds[self._global_ptr : self._global_ptr + ep_len] = np.array(vals)

        self._f['ep_len'].resize(self._ep_written + 1, axis=0)
        self._f['ep_len'][self._ep_written] = ep_len
        self._f['ep_offset'].resize(self._ep_written + 1, axis=0)
        self._f['ep_offset'][self._ep_written] = self._global_ptr

        self._ep_written += 1
        self._global_ptr += ep_len

    def _init_schema(self, sample_ep: dict) -> None:
        for col, vals in sample_ep.items():
            sample = np.asarray(vals[0])
            self._f.create_dataset(
                col,
                shape=(0, *sample.shape),
                maxshape=(None, *sample.shape),
                dtype=sample.dtype,
                chunks=(1, *sample.shape),
            )
        self._f.create_dataset(
            'ep_len', shape=(0,), maxshape=(None,), dtype=np.int32
        )
        self._f.create_dataset(
            'ep_offset', shape=(0,), maxshape=(None,), dtype=np.int64
        )


@register_format
class HDF5(Format):
    name = 'hdf5'

    @classmethod
    def detect(cls, path) -> bool:
        p = Path(path)
        if p.suffix in ('.h5', '.hdf5'):
            return True
        if p.is_dir():
            return any(p.glob('*.h5')) or any(p.glob('*.hdf5'))
        return False

    @classmethod
    def open_reader(cls, path, **kwargs) -> HDF5Dataset:
        p = Path(path)
        if p.is_dir():
            files = sorted(p.glob('*.h5')) + sorted(p.glob('*.hdf5'))
            if not files:
                raise FileNotFoundError(f'No .h5/.hdf5 file in {p}')
            if len(files) > 1:
                raise ValueError(
                    f'Ambiguous dataset: multiple HDF5 files in {p}. '
                    'Pass the file directly.'
                )
            p = files[0]
        return HDF5Dataset(path=p, **kwargs)

    @classmethod
    def open_writer(cls, path, **kwargs) -> HDF5Writer:
        return HDF5Writer(path, **kwargs)


__all__ = ['HDF5', 'HDF5Dataset', 'HDF5Writer']
