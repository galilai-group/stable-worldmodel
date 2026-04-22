"""Integration tests with real environment data collection.

``World.record_dataset`` writes LanceDB tables.  These tests exercise the
full collect → load pipeline against that writer, and also cover the
folder-based ``ImageDataset`` / ``VideoDataset`` formats (decoupled from the
writer — the fixtures build the folder layouts directly from synthetic
data, because the former HDF5-based conversion path no longer exists).
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from stable_worldmodel import World
from stable_worldmodel.policy import RandomPolicy
from stable_worldmodel.data import ImageDataset, LanceDataset, VideoDataset


class TestRealDataCollection:
    """Collect episodes from a real env and load them back via LanceDataset."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        return tmp_path

    @staticmethod
    def _lance_path(cache_dir: Path, name: str) -> Path:
        return cache_dir / 'datasets' / f'{name}.lance'

    def test_collect_and_load_pusht(self, temp_cache_dir):
        """Recording produces a Lance table readable by LanceDataset."""
        world = World(
            env_name='swm/PushT-v1',
            num_envs=2,
            image_shape=(64, 64),
            max_episode_steps=20,
            verbose=0,
        )
        world.set_policy(RandomPolicy())

        dataset_name = 'test_pusht'
        world.record_dataset(
            dataset_name=dataset_name,
            episodes=4,
            seed=42,
            cache_dir=temp_cache_dir,
        )
        world.envs.close()

        lance_path = self._lance_path(temp_cache_dir, dataset_name)
        assert lance_path.exists(), f'Lance table not created at {lance_path}'

        dataset = LanceDataset(uri=str(lance_path))
        assert len(dataset) > 0, 'Dataset should have samples'
        assert len(dataset.lengths) == 4, 'Should have 4 episodes'

        sample = dataset[0]
        assert isinstance(sample, dict)
        assert 'action' in sample
        # Image column should be auto-detected from the schema.
        assert 'pixels' in dataset.image_columns

        # Non-string values come back as tensors (strings stay as str scalars).
        for key, value in sample.items():
            if not isinstance(value, str):
                assert isinstance(value, torch.Tensor), (
                    f'{key} should be a tensor'
                )

        chunk = dataset.load_chunk(
            episodes_idx=np.array([0, 1]),
            start=np.array([0, 0]),
            end=np.array([5, 5]),
        )
        assert len(chunk) == 2

    def test_dataset_frameskip(self, temp_cache_dir):
        """Loading a Lance-recorded dataset with frameskip works."""
        world = World(
            env_name='swm/PushT-v1',
            num_envs=2,
            image_shape=(64, 64),
            max_episode_steps=30,
            verbose=0,
        )
        world.set_policy(RandomPolicy())

        dataset_name = 'test_frameskip'
        world.record_dataset(
            dataset_name=dataset_name,
            episodes=2,
            seed=42,
            cache_dir=temp_cache_dir,
        )
        world.envs.close()

        dataset = LanceDataset(
            uri=str(self._lance_path(temp_cache_dir, dataset_name)),
            frameskip=2,
            num_steps=2,
        )
        if len(dataset) > 0:
            assert isinstance(dataset[0], dict)

    def test_dataset_transform(self, temp_cache_dir):
        """Loading a Lance-recorded dataset with a custom transform works."""
        world = World(
            env_name='swm/PushT-v1',
            num_envs=2,
            image_shape=(64, 64),
            max_episode_steps=20,
            verbose=0,
        )
        world.set_policy(RandomPolicy())

        dataset_name = 'test_transform'
        world.record_dataset(
            dataset_name=dataset_name,
            episodes=2,
            seed=42,
            cache_dir=temp_cache_dir,
        )
        world.envs.close()

        seen = {}

        def mark(sample):
            seen['was_called'] = True
            return sample

        dataset = LanceDataset(
            uri=str(self._lance_path(temp_cache_dir, dataset_name)),
            transform=mark,
        )
        if len(dataset) > 0:
            _ = dataset[0]
            assert seen.get('was_called') is True

    def test_dataset_keys_to_cache(self, temp_cache_dir):
        """``keys_to_cache`` is accepted (with the usual warning) on Lance too."""
        world = World(
            env_name='swm/PushT-v1',
            num_envs=2,
            image_shape=(64, 64),
            max_episode_steps=15,
            verbose=0,
        )
        world.set_policy(RandomPolicy())

        dataset_name = 'test_keys_cache'
        world.record_dataset(
            dataset_name=dataset_name,
            episodes=2,
            seed=42,
            cache_dir=temp_cache_dir,
        )
        world.envs.close()

        dataset = LanceDataset(
            uri=str(self._lance_path(temp_cache_dir, dataset_name)),
            keys_to_load=['action', 'pixels'],
            keys_to_cache=['action'],
        )
        assert 'action' in dataset._cache
        if len(dataset) > 0:
            assert 'action' in dataset[0]


# ---------------------------------------------------------------------------
# ImageDataset / VideoDataset fixtures
#
# These tests target the folder-based dataset formats, not ``record_dataset``.
# We synthesise the expected folder layout directly rather than round-tripping
# through ``record_dataset`` (which no longer writes HDF5), so they stay
# focused on what they actually cover.
# ---------------------------------------------------------------------------


def _write_image_folder(root: Path, ep_lengths: list[int]) -> None:
    """Write an ImageDataset folder layout with deterministic synthetic data."""
    root.mkdir(parents=True, exist_ok=True)
    ep_lengths_arr = np.asarray(ep_lengths, dtype=np.int32)
    ep_offsets = np.concatenate([[0], np.cumsum(ep_lengths_arr)[:-1]]).astype(np.int32)
    total = int(ep_lengths_arr.sum())

    np.savez(root / 'ep_len.npz', ep_lengths_arr)
    np.savez(root / 'ep_offset.npz', ep_offsets)

    rng = np.random.default_rng(0)
    action = rng.standard_normal((total, 2)).astype(np.float32)
    proprio = rng.standard_normal((total, 4)).astype(np.float32)
    np.savez(root / 'action.npz', action)
    np.savez(root / 'proprio.npz', proprio)

    img_dir = root / 'pixels'
    img_dir.mkdir(exist_ok=True)
    for ep_idx, length in enumerate(ep_lengths):
        for step in range(length):
            frame = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
            Image.fromarray(frame).save(img_dir / f'ep_{ep_idx}_step_{step}.jpeg')


class TestImageDatasetReal:
    """ImageDataset end-to-end against a synthetic folder layout."""

    def test_load_folder_dataset(self, tmp_path):
        root = tmp_path / 'image_ds'
        _write_image_folder(root, ep_lengths=[5, 4, 6])

        dataset = ImageDataset(
            name='image_ds', cache_dir=str(tmp_path), image_keys=['pixels']
        )
        assert len(dataset) > 0
        assert len(dataset.lengths) == 3

        sample = dataset[0]
        assert isinstance(sample, dict)
        assert 'action' in sample
        assert 'pixels' in sample
        assert isinstance(sample['pixels'], torch.Tensor)
        assert sample['pixels'].shape[-3] == 3  # channels

    def test_load_chunk(self, tmp_path):
        root = tmp_path / 'image_ds'
        _write_image_folder(root, ep_lengths=[4, 4])

        dataset = ImageDataset(
            name='image_ds', cache_dir=str(tmp_path), image_keys=['pixels']
        )
        chunk = dataset.load_chunk(
            episodes_idx=np.array([0, 1]),
            start=np.array([0, 1]),
            end=np.array([3, 4]),
        )
        assert len(chunk) == 2
        assert 'pixels' in chunk[0]
        assert 'action' in chunk[0]


class TestVideoDatasetReal:
    """VideoDataset end-to-end against a synthetic MP4 folder layout."""

    def test_load_video_dataset(self, tmp_path):
        imageio = pytest.importorskip('imageio.v3')
        pytest.importorskip('decord')

        root = tmp_path / 'video_ds'
        root.mkdir()

        ep_lengths = np.asarray([5, 4, 6], dtype=np.int32)
        ep_offsets = np.concatenate([[0], np.cumsum(ep_lengths)[:-1]]).astype(np.int32)
        total = int(ep_lengths.sum())

        rng = np.random.default_rng(0)
        action = rng.standard_normal((total, 2)).astype(np.float32)
        np.savez(root / 'ep_len.npz', ep_lengths)
        np.savez(root / 'ep_offset.npz', ep_offsets)
        np.savez(root / 'action.npz', action)

        video_dir = root / 'video'
        video_dir.mkdir()
        for ep_idx, length in enumerate(ep_lengths.tolist()):
            frames = rng.integers(0, 255, (length, 16, 16, 3), dtype=np.uint8)
            imageio.imwrite(video_dir / f'ep_{ep_idx}.mp4', frames, fps=30)

        dataset = VideoDataset(
            name='video_ds', cache_dir=str(tmp_path), video_keys=['video']
        )
        assert len(dataset) > 0
        assert len(dataset.lengths) == 3

        sample = dataset[0]
        assert 'video' in sample
        assert isinstance(sample['video'], torch.Tensor)
        assert sample['video'].shape[-3] == 3
