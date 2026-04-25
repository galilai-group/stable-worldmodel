"""LeRobot Hub format (read-only).

Identified by the ``lerobot://`` scheme. Mapping ``World.collect``'s
arbitrary info-dict to LeRobot's prescribed schema is non-trivial and
therefore not supported as a writer here.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stable_worldmodel.data.dataset import Dataset
from stable_worldmodel.data.format import Format, register_format

_SCHEME = 'lerobot://'


def _import_lerobot_hub_dataset() -> type:
    """Import upstream lerobot `LeRobotDataset` lazily (aliased to avoid name clash)."""
    if sys.version_info < (3, 12):
        raise ImportError(
            'stable_worldmodel.data.LeRobotAdapter requires Python 3.12+ because '
            'the official lerobot package is only available on Python 3.12+.'
        )

    try:
        from lerobot.datasets.lerobot_dataset import (
            LeRobotDataset as LerobotHubDataset,
        )
    except ImportError as exc:
        raise ImportError(
            'stable_worldmodel.data.LeRobotAdapter requires the optional '
            'lerobot dependency. Install it with '
            "`pip install 'stable-worldmodel[format]'`."
        ) from exc

    return LerobotHubDataset


def _scalarize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value
    return value


def _column_to_numpy(column: Any) -> np.ndarray:
    if isinstance(column, torch.Tensor):
        return column.detach().cpu().numpy()
    if isinstance(column, np.ndarray):
        return column
    if isinstance(column, list):
        return np.asarray([_scalarize(v) for v in column])
    return np.asarray(column)


class LeRobotAdapter(Dataset):
    """Wraps lerobot's `LeRobotDataset` and exposes the SWM `Dataset` API."""

    _SYNTHETIC_COLUMNS = {'ep_idx', 'step_idx'}

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        primary_camera_key: str | None = None,
        key_aliases: dict[str, str] | None = None,
        **lerobot_kwargs: Any,
    ) -> None:
        LerobotHubDataset = _import_lerobot_hub_dataset()
        self._hub_dataset_cls = LerobotHubDataset
        self._lerobot_kwargs = dict(lerobot_kwargs)
        self.dataset = LerobotHubDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=None,
            delta_timestamps=None,
            **lerobot_kwargs,
        )
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else None
        self.episodes = episodes

        native_keys = self._get_native_keys()
        self._camera_keys = self.dataset.meta.camera_keys
        self._fps = self._get_fps()
        self._primary_camera_key = self._resolve_primary_camera(
            primary_camera_key, native_keys
        )

        self._native_to_alias = self._build_alias_map(native_keys, key_aliases)
        self._alias_to_native = {
            alias: native for native, alias in self._native_to_alias.items()
        }
        self._full_columns: dict[str, np.ndarray] = {}
        self._cache: dict[str, np.ndarray] = {}
        self._window_datasets: dict[
            tuple[tuple[int, ...], tuple[int, ...]], Any
        ] = {}

        episode_index = self._get_native_column('episode_index')
        (
            local_episode_index,
            step_idx,
            lengths,
            offsets,
            absolute_episode_ids,
        ) = self._build_episode_metadata(episode_index)
        self._absolute_episode_ids = absolute_episode_ids
        self._cache['ep_idx'] = local_episode_index
        self._cache['step_idx'] = step_idx

        if keys_to_load is None:
            keys_to_load = list(self._native_to_alias.values()) + [
                'ep_idx',
                'step_idx',
            ]
        self._keys = list(dict.fromkeys(keys_to_load))

        for key in keys_to_cache or []:
            self._cache[key] = self._materialize_column(key)

        super().__init__(lengths, offsets, frameskip, num_steps, transform)

    @property
    def column_names(self) -> list[str]:
        return self._keys

    def _get_native_keys(self) -> list[str]:
        features = self.dataset.features
        if not isinstance(features, Mapping):
            raise TypeError(
                'LeRobot dataset features must be a mapping of column names.'
            )
        return list(features.keys())

    def _resolve_primary_camera(
        self,
        primary_camera_key: str | None,
        native_keys: list[str],
    ) -> str | None:
        if primary_camera_key is None and len(self._camera_keys) > 1:
            raise ValueError(
                'LeRobotAdapter requires `primary_camera_key` when '
                'multiple cameras are available.'
            )
        if primary_camera_key is not None:
            if primary_camera_key not in native_keys:
                raise KeyError(
                    f"Primary camera key '{primary_camera_key}' not found in LeRobot dataset."
                )
            return primary_camera_key

        for key in self._camera_keys:
            if key in native_keys:
                return key
        return None

    def _get_fps(self) -> float:
        meta = self.dataset.meta
        info = meta.info

        if isinstance(info, dict) and 'fps' in info:
            return float(info['fps'])
        if hasattr(info, 'fps'):
            return float(info.fps)

        raise ValueError(
            'LeRobot dataset metadata must expose `meta.info.fps`.'
        )

    def _build_alias_map(
        self,
        native_keys: list[str],
        key_aliases: dict[str, str] | None,
    ) -> dict[str, str]:
        aliases: dict[str, str] = {}
        if self._primary_camera_key is not None:
            aliases[self._primary_camera_key] = 'pixels'
        if 'action' in native_keys:
            aliases['action'] = 'action'
        if 'observation.state' in native_keys:
            aliases['observation.state'] = 'proprio'

        for native, alias in (key_aliases or {}).items():
            if native not in native_keys:
                raise KeyError(
                    f"Key alias source '{native}' not found in LeRobot dataset."
                )
            aliases[native] = alias

        return aliases

    def _build_episode_metadata(
        self,
        absolute_episode_index: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        abs_ids = absolute_episode_index.astype(np.int64)
        unique_abs, first_idx = np.unique(abs_ids, return_index=True)
        order = np.argsort(first_idx)
        absolute_episode_ids = unique_abs[order]
        counts = np.array(
            [(abs_ids == ep_id).sum() for ep_id in absolute_episode_ids],
            dtype=np.int64,
        )

        local_map = {
            int(abs_id): idx for idx, abs_id in enumerate(absolute_episode_ids)
        }
        local_episode_index = np.array(
            [local_map[int(abs_id)] for abs_id in abs_ids],
            dtype=np.int64,
        )

        step_idx = np.empty_like(local_episode_index)
        for local_ep in range(len(absolute_episode_ids)):
            mask = local_episode_index == local_ep
            step_idx[mask] = np.arange(mask.sum(), dtype=np.int64)

        offsets = np.zeros(len(counts), dtype=np.int64)
        if len(counts) > 1:
            offsets[1:] = np.cumsum(counts[:-1])

        return (
            local_episode_index,
            step_idx,
            counts,
            offsets,
            absolute_episode_ids.astype(np.int64),
        )

    def _get_native_column(self, native_key: str) -> np.ndarray:
        if native_key not in self._full_columns:
            column = self.dataset.hf_dataset[native_key]
            self._full_columns[native_key] = _column_to_numpy(column)
        return self._full_columns[native_key]

    def _time_offsets(self, indices: tuple[int, ...]) -> list[float]:
        return [float(idx) / self._fps for idx in indices]

    def _window_dataset(
        self,
        observation_indices: tuple[int, ...],
        action_indices: tuple[int, ...],
    ) -> Any:
        cache_key = (observation_indices, action_indices)
        if cache_key not in self._window_datasets:
            delta_timestamps = {}
            for key in self._keys:
                if key in self._SYNTHETIC_COLUMNS:
                    continue
                native_key = self._alias_to_native.get(key)
                if native_key is None:
                    continue
                if key == 'action':
                    delta_timestamps[native_key] = self._time_offsets(
                        action_indices
                    )
                else:
                    delta_timestamps[native_key] = self._time_offsets(
                        observation_indices
                    )

            self._window_datasets[cache_key] = self._hub_dataset_cls(
                repo_id=self.repo_id,
                root=self.root,
                episodes=self.episodes,
                image_transforms=None,
                delta_timestamps=delta_timestamps or None,
                **self._lerobot_kwargs,
            )
        return self._window_datasets[cache_key]

    def _materialize_column(self, key: str) -> np.ndarray:
        if key in self._cache:
            return self._cache[key]
        if key in self._SYNTHETIC_COLUMNS:
            return self._cache[key]

        native_key = self._alias_to_native.get(key)
        if native_key is None:
            raise KeyError(f"Unknown LeRobot adapter column '{key}'.")
        if native_key in self._camera_keys:
            raise KeyError(
                f"'{key}' cannot be materialized as a full array because it is image/video-backed."
            )
        return self._get_native_column(native_key)

    def _get_item_value(self, item: dict[str, Any], key: str) -> Any:
        if key == 'ep_idx':
            return int(self._cache['ep_idx'][item['_row_idx']])
        if key == 'step_idx':
            return int(self._cache['step_idx'][item['_row_idx']])

        native_key = self._alias_to_native[key]
        return item[native_key]

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        g_start = int(self.offsets[ep_idx] + start)
        length = int(end - start)
        obs_indices = tuple(range(0, length, self.frameskip))
        action_indices = tuple(range(length))
        row = dict(self._window_dataset(obs_indices, action_indices)[g_start])
        row['_row_idx'] = g_start
        steps: dict[str, Any] = {}
        for key in self._keys:
            if key in self._SYNTHETIC_COLUMNS:
                if key == 'ep_idx':
                    data = torch.full(
                        (len(obs_indices),),
                        int(self._cache['ep_idx'][g_start]),
                        dtype=torch.int64,
                    )
                else:
                    data = torch.as_tensor(
                        [start + idx for idx in obs_indices],
                        dtype=torch.int64,
                    )
            else:
                data = self._get_item_value(row, key)

            if isinstance(data, torch.Tensor):
                if data.ndim == 4 and data.shape[-1] in (1, 3):
                    data = data.permute(0, 3, 1, 2)
            steps[key] = data

        return self.transform(steps) if self.transform else steps

    def get_col_data(self, col: str) -> np.ndarray:
        return self._materialize_column(col)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        out = {}
        for col in self._keys:
            try:
                data = self._materialize_column(col)
            except KeyError:
                continue
            out[col] = data[row_idx]
        return out

    def get_dim(self, col: str) -> int:
        data = self.get_col_data(col)
        return np.prod(data.shape[1:]).item() if data.ndim > 1 else 1


@register_format
class LeRobot(Format):
    name = 'lerobot'

    @classmethod
    def detect(cls, path) -> bool:
        return isinstance(path, str) and path.startswith(_SCHEME)

    @classmethod
    def open_reader(cls, path, **kwargs):
        repo_id = path[len(_SCHEME) :] if path.startswith(_SCHEME) else path
        return LeRobotAdapter(repo_id, **kwargs)


__all__ = ['LeRobot', 'LeRobotAdapter']
