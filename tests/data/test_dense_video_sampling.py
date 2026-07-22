"""Network-independent tests for dense video sampling.

These tests stub the decoder interface directly, so they exercise frame-index
planning without requiring TorchCodec or compatible FFmpeg libraries.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from stable_worldmodel.data import LanceVideoDataset, VideoDataset


class _StubDecoder:
    def __init__(self, value_offset: int = 0) -> None:
        self.value_offset = value_offset
        self.calls: list[list[int]] = []

    def get_frames_at(self, *, indices: list[int]) -> SimpleNamespace:
        self.calls.append(list(indices))
        data = torch.tensor(
            [self.value_offset + index for index in indices],
            dtype=torch.int64,
        ).reshape(-1, 1, 1, 1)
        return SimpleNamespace(data=data)


def _video_dataset() -> tuple[VideoDataset, dict[str, _StubDecoder]]:
    dataset = object.__new__(VideoDataset)
    dataset.offsets = np.array([0], dtype=np.int64)
    dataset.frameskip = 2
    dataset.num_steps = 3
    dataset.span = 6
    dataset.transform = None
    dataset.dense_columns = frozenset({'action', 'dense_video'})
    dataset._dense_columns_validated = False
    dataset._keys = ['dense_video', 'sparse_video']
    dataset.folder_keys = ['dense_video', 'sparse_video']
    dataset._cache = {}
    dataset.clip_indices = [(0, 3)]

    decoders = {
        'dense_video': _StubDecoder(value_offset=100),
        'sparse_video': _StubDecoder(value_offset=200),
    }
    dataset._reader = lambda _ep_idx, key: decoders[key]
    return dataset, decoders


def _lance_video_dataset() -> tuple[
    LanceVideoDataset, dict[str, _StubDecoder]
]:
    dataset = object.__new__(LanceVideoDataset)
    dataset.offsets = np.array([0], dtype=np.int64)
    dataset.frameskip = 2
    dataset.num_steps = 2
    dataset.span = 4
    dataset.transform = None
    dataset.dense_columns = frozenset({'action', 'dense_video'})
    dataset._dense_columns_validated = False
    dataset._keys = ['dense_video', 'sparse_video']
    dataset._video_keys = {'dense_video', 'sparse_video'}
    dataset._fetch_columns = []
    dataset.clip_indices = [(0, start) for start in range(4)]

    decoders = {
        'dense_video': _StubDecoder(value_offset=100),
        'sparse_video': _StubDecoder(value_offset=200),
    }
    dataset._ensure_videos_open = lambda: None
    dataset._decoder_for = lambda _ep_idx, key: decoders[key]
    return dataset, decoders


def test_video_dataset_uses_per_key_stride_from_nonzero_start() -> None:
    dataset, decoders = _video_dataset()

    item = dataset[0]

    assert decoders['dense_video'].calls == [[3, 4, 5, 6, 7, 8]]
    assert decoders['sparse_video'].calls == [[3, 5, 7]]
    assert item['dense_video'].shape == (3, 2, 1, 1, 1)
    assert item['sparse_video'].shape == (3, 1, 1, 1)
    torch.testing.assert_close(
        item['dense_video'][:, :, 0, 0, 0],
        torch.tensor([[103, 104], [105, 106], [107, 108]]),
    )
    torch.testing.assert_close(
        item['sparse_video'][:, 0, 0, 0],
        torch.tensor([203, 205, 207]),
    )


def test_lance_video_single_window_uses_per_key_stride() -> None:
    dataset, decoders = _lance_video_dataset()

    dense = dataset._decode_video_window(0, 'dense_video', 3, 6)
    sparse = dataset._decode_video_window(0, 'sparse_video', 3, 6)

    assert decoders['dense_video'].calls == [[3, 4, 5, 6, 7, 8]]
    assert decoders['sparse_video'].calls == [[3, 5, 7]]
    torch.testing.assert_close(
        dense[:, 0, 0, 0], torch.tensor([103, 104, 105, 106, 107, 108])
    )
    torch.testing.assert_close(
        sparse[:, 0, 0, 0], torch.tensor([203, 205, 207])
    )


def test_lance_video_batch_unions_and_scatters_requested_frames() -> None:
    dataset, decoders = _lance_video_dataset()

    items = dataset.__getitems__([2, 0, 2, 1])

    assert decoders['dense_video'].calls == [[0, 1, 2, 3, 4, 5]]
    assert decoders['sparse_video'].calls == [[0, 1, 2, 3, 4]]

    expected_starts = [2, 0, 2, 1]
    for item, start in zip(items, expected_starts):
        assert item['dense_video'].shape == (2, 2, 1, 1, 1)
        assert item['sparse_video'].shape == (2, 1, 1, 1)
        torch.testing.assert_close(
            item['dense_video'][:, :, 0, 0, 0],
            torch.tensor(
                [
                    [100 + start, 101 + start],
                    [102 + start, 103 + start],
                ]
            ),
        )
        torch.testing.assert_close(
            item['sparse_video'][:, 0, 0, 0],
            torch.tensor([200 + start, 202 + start]),
        )

    for key in ('dense_video', 'sparse_video'):
        torch.testing.assert_close(items[0][key], items[2][key])
