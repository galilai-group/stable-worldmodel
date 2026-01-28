"""Tests for ImageDataset, VideoDataset, MergeDataset, and ConcatDataset."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

import h5py

from stable_worldmodel.data import (
    ImageDataset,
    VideoDataset,
    MergeDataset,
    ConcatDataset,
    HDF5Dataset,
    GoalDataset,
)
from stable_worldmodel.data.dataset import Dataset


class TestDatasetBase:
    """Tests for the base Dataset class abstract methods."""

    def test_column_names_not_implemented(self):
        """Test that column_names raises NotImplementedError."""
        lengths = np.array([10])
        offsets = np.array([0])
        dataset = Dataset(lengths, offsets)
        with pytest.raises(NotImplementedError):
            _ = dataset.column_names

    def test_load_slice_not_implemented(self):
        """Test that _load_slice raises NotImplementedError."""
        lengths = np.array([10])
        offsets = np.array([0])
        dataset = Dataset(lengths, offsets)
        with pytest.raises(NotImplementedError):
            dataset._load_slice(0, 0, 5)

    def test_get_col_data_not_implemented(self):
        """Test that get_col_data raises NotImplementedError."""
        lengths = np.array([10])
        offsets = np.array([0])
        dataset = Dataset(lengths, offsets)
        with pytest.raises(NotImplementedError):
            dataset.get_col_data("col")

    def test_get_row_data_not_implemented(self):
        """Test that get_row_data raises NotImplementedError."""
        lengths = np.array([10])
        offsets = np.array([0])
        dataset = Dataset(lengths, offsets)
        with pytest.raises(NotImplementedError):
            dataset.get_row_data(0)


@pytest.fixture
def sample_image_dataset(tmp_path):
    """Create a sample ImageDataset directory structure for testing."""
    dataset_path = tmp_path / "test_image_dataset"
    dataset_path.mkdir()

    # Create sample data: 2 episodes, 10 steps each
    ep_lengths = np.array([10, 10])
    ep_offsets = np.array([0, 10])
    total_steps = sum(ep_lengths)

    # Save metadata
    np.savez(dataset_path / "ep_len.npz", ep_lengths)
    np.savez(dataset_path / "ep_offset.npz", ep_offsets)

    # Save non-image data as .npz
    np.savez(dataset_path / "observation.npz", np.random.rand(total_steps, 4).astype(np.float32))
    np.savez(dataset_path / "action.npz", np.random.rand(total_steps, 2).astype(np.float32))

    # Create pixels folder with images
    pixels_path = dataset_path / "pixels"
    pixels_path.mkdir()

    for ep_idx in range(2):
        for step_idx in range(10):
            img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(pixels_path / f"ep_{ep_idx}_step_{step_idx}.jpeg")

    return tmp_path, "test_image_dataset"


@pytest.fixture
def sample_image_dataset_short_episode(tmp_path):
    """Create a sample ImageDataset with a short episode."""
    dataset_path = tmp_path / "short_image_dataset"
    dataset_path.mkdir()

    # Create sample data: 2 episodes, different lengths
    ep_lengths = np.array([3, 10])  # First episode too short for default span
    ep_offsets = np.array([0, 3])
    total_steps = sum(ep_lengths)

    np.savez(dataset_path / "ep_len.npz", ep_lengths)
    np.savez(dataset_path / "ep_offset.npz", ep_offsets)
    np.savez(dataset_path / "observation.npz", np.random.rand(total_steps, 4).astype(np.float32))
    np.savez(dataset_path / "action.npz", np.random.rand(total_steps, 2).astype(np.float32))

    pixels_path = dataset_path / "pixels"
    pixels_path.mkdir()

    # Episode 0: 3 steps
    for step_idx in range(3):
        img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(pixels_path / f"ep_0_step_{step_idx}.jpeg")

    # Episode 1: 10 steps
    for step_idx in range(10):
        img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(pixels_path / f"ep_1_step_{step_idx}.jpeg")

    return tmp_path, "short_image_dataset"


@pytest.fixture
def sample_video_dataset(tmp_path):
    """Create a sample VideoDataset directory structure with MP4 files for testing."""
    import imageio.v3 as iio

    dataset_path = tmp_path / "test_video_dataset"
    dataset_path.mkdir()

    ep_lengths = np.array([10, 10])
    ep_offsets = np.array([0, 10])
    total_steps = sum(ep_lengths)

    np.savez(dataset_path / "ep_len.npz", ep_lengths)
    np.savez(dataset_path / "ep_offset.npz", ep_offsets)
    np.savez(dataset_path / "observation.npz", np.random.rand(total_steps, 4).astype(np.float32))
    np.savez(dataset_path / "action.npz", np.random.rand(total_steps, 2).astype(np.float32))

    # Create video folder with MP4 files
    video_path = dataset_path / "video"
    video_path.mkdir()

    for ep_idx in range(2):
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(10)]
        iio.imwrite(video_path / f"ep_{ep_idx}.mp4", frames, fps=30)

    return tmp_path, "test_video_dataset"


@pytest.fixture
def sample_image_dataset_jpg(tmp_path):
    """Create a sample ImageDataset with .jpg extension for testing fallback."""
    dataset_path = tmp_path / "test_image_dataset_jpg"
    dataset_path.mkdir()

    ep_lengths = np.array([5])
    ep_offsets = np.array([0])
    total_steps = 5

    np.savez(dataset_path / "ep_len.npz", ep_lengths)
    np.savez(dataset_path / "ep_offset.npz", ep_offsets)
    np.savez(dataset_path / "action.npz", np.random.rand(total_steps, 2).astype(np.float32))

    pixels_path = dataset_path / "pixels"
    pixels_path.mkdir()

    for step_idx in range(5):
        img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(pixels_path / f"ep_0_step_{step_idx}.jpg")  # .jpg instead of .jpeg

    return tmp_path, "test_image_dataset_jpg"


class MockDataset:
    """A simple mock dataset for testing MergeDataset and ConcatDataset."""

    def __init__(self, data: dict, length: int, num_episodes: int = 1):
        self._data = data
        self._length = length
        # Episode structure for load_chunk support
        self.lengths = np.array([length // num_episodes] * num_episodes)

    @property
    def column_names(self) -> list[str]:
        return list(self._data.keys())

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        return {k: v[idx] for k, v in self._data.items()}

    def load_chunk(self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray) -> list[dict]:
        chunk = []
        for s, e in zip(start, end):
            chunk.append({k: v[s:e] for k, v in self._data.items()})
        return chunk

    def get_col_data(self, col: str) -> np.ndarray:
        return self._data[col]

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        if isinstance(row_idx, int):
            return {k: v[row_idx] for k, v in self._data.items()}
        return {k: v[row_idx] for k, v in self._data.items()}


@pytest.fixture
def mock_dataset_a():
    """Mock dataset A with pixels, action, observation."""
    return MockDataset(
        data={
            "pixels": torch.randn(20, 3, 64, 64),
            "action": torch.randn(20, 2),
            "observation": torch.randn(20, 4),
        },
        length=20,
    )


@pytest.fixture
def mock_dataset_b():
    """Mock dataset B with audio, action, observation."""
    return MockDataset(
        data={
            "audio": torch.randn(20, 16000),
            "action": torch.randn(20, 2),
            "observation": torch.randn(20, 4),
        },
        length=20,
    )


@pytest.fixture
def mock_dataset_c():
    """Mock dataset C with different length for ConcatDataset tests."""
    return MockDataset(
        data={
            "pixels": torch.randn(15, 3, 64, 64),
            "action": torch.randn(15, 2),
        },
        length=15,
    )


class TestImageDataset:
    def test_init(self, sample_image_dataset):
        """Test ImageDataset initialization."""
        cache_dir, name = sample_image_dataset
        dataset = ImageDataset(name, cache_dir=str(cache_dir))

        assert dataset.path == cache_dir / name
        assert len(dataset.lengths) == 2
        assert len(dataset.offsets) == 2

    def test_len(self, sample_image_dataset):
        """Test ImageDataset length calculation."""
        cache_dir, name = sample_image_dataset
        dataset = ImageDataset(name, cache_dir=str(cache_dir))

        assert len(dataset) > 0

    def test_column_names(self, sample_image_dataset):
        """Test column_names property excludes metadata keys."""
        cache_dir, name = sample_image_dataset
        dataset = ImageDataset(name, cache_dir=str(cache_dir))

        column_names = dataset.column_names
        assert "observation" in column_names
        assert "action" in column_names
        assert "pixels" in column_names
        assert "ep_len" not in column_names
        assert "ep_offset" not in column_names

    def test_getitem(self, sample_image_dataset):
        """Test ImageDataset __getitem__ method."""
        cache_dir, name = sample_image_dataset
        dataset = ImageDataset(name, cache_dir=str(cache_dir))

        item = dataset[0]

        assert isinstance(item, dict)
        assert "observation" in item
        assert "action" in item
        assert "pixels" in item
        assert isinstance(item["observation"], torch.Tensor)
        assert isinstance(item["action"], torch.Tensor)
        assert isinstance(item["pixels"], torch.Tensor)

    def test_image_permutation(self, sample_image_dataset):
        """Test that images are permuted to TCHW format."""
        cache_dir, name = sample_image_dataset
        dataset = ImageDataset(name, cache_dir=str(cache_dir))

        item = dataset[0]

        assert "pixels" in item
        # With num_steps=1, shape should be (1, 3, 64, 64)
        assert item["pixels"].shape[-3] == 3  # channels

    def test_frameskip(self, sample_image_dataset):
        """Test ImageDataset with frameskip."""
        cache_dir, name = sample_image_dataset
        dataset = ImageDataset(name, cache_dir=str(cache_dir), frameskip=2, num_steps=2)

        assert len(dataset) > 0
        item = dataset[0]
        assert isinstance(item, dict)

    def test_keys_to_load(self, sample_image_dataset):
        """Test ImageDataset with specific keys_to_load."""
        cache_dir, name = sample_image_dataset
        dataset = ImageDataset(
            name,
            cache_dir=str(cache_dir),
            keys_to_load=["observation", "action"],
        )

        item = dataset[0]
        assert "observation" in item
        assert "action" in item
        assert "pixels" not in item

    def test_load_chunk(self, sample_image_dataset):
        """Test load_chunk returns correct slices."""
        cache_dir, name = sample_image_dataset
        dataset = ImageDataset(name, cache_dir=str(cache_dir))

        episodes_idx = np.array([0, 1])
        start = np.array([0, 0])
        end = np.array([3, 5])

        chunk = dataset.load_chunk(episodes_idx, start, end)

        assert isinstance(chunk, list)
        assert len(chunk) == 2
        assert "observation" in chunk[0]
        assert "action" in chunk[0]

    def test_get_col_data(self, sample_image_dataset):
        """Test get_col_data method."""
        cache_dir, name = sample_image_dataset
        dataset = ImageDataset(name, cache_dir=str(cache_dir))

        col_data = dataset.get_col_data("observation")
        assert isinstance(col_data, np.ndarray)
        assert col_data.shape[0] == 20  # Total steps

    def test_get_col_data_image_key_raises(self, sample_image_dataset):
        """Test get_col_data raises for image keys."""
        cache_dir, name = sample_image_dataset
        dataset = ImageDataset(name, cache_dir=str(cache_dir))

        with pytest.raises(KeyError, match="not in cache"):
            dataset.get_col_data("pixels")

    def test_get_row_data(self, sample_image_dataset):
        """Test get_row_data method."""
        cache_dir, name = sample_image_dataset
        dataset = ImageDataset(name, cache_dir=str(cache_dir))

        row_data = dataset.get_row_data(5)
        assert isinstance(row_data, dict)
        assert "observation" in row_data
        assert "action" in row_data
        # pixels not in row_data because it's an image key

    def test_transform(self, sample_image_dataset):
        """Test ImageDataset with transform function."""
        cache_dir, name = sample_image_dataset

        def double_transform(data):
            for k in data:
                if data[k].dtype == torch.float32:
                    data[k] = data[k] * 2
            return data

        dataset = ImageDataset(
            name,
            cache_dir=str(cache_dir),
            transform=double_transform,
        )

        item = dataset[0]
        assert isinstance(item, dict)

    def test_short_episode_filtered(self, sample_image_dataset_short_episode):
        """Test that episodes shorter than span are filtered out."""
        cache_dir, name = sample_image_dataset_short_episode
        dataset = ImageDataset(name, cache_dir=str(cache_dir), num_steps=5, frameskip=1)

        # Only second episode (length 10) should have valid clips
        for ep_idx, _ in dataset.clip_indices:
            assert ep_idx == 1

    def test_load_file(self, sample_image_dataset):
        """Test _load_file method."""
        cache_dir, name = sample_image_dataset
        dataset = ImageDataset(name, cache_dir=str(cache_dir))

        img = dataset._load_file(0, 0, "pixels")
        assert isinstance(img, np.ndarray)
        assert img.shape == (64, 64, 3)

    def test_load_file_jpg_fallback(self, sample_image_dataset_jpg):
        """Test _load_file falls back to .jpg when .jpeg doesn't exist."""
        cache_dir, name = sample_image_dataset_jpg
        dataset = ImageDataset(name, cache_dir=str(cache_dir))

        img = dataset._load_file(0, 0, "pixels")
        assert isinstance(img, np.ndarray)
        assert img.shape == (64, 64, 3)

    def test_load_episode(self, sample_image_dataset):
        """Test load_episode loads full episode data."""
        cache_dir, name = sample_image_dataset
        dataset = ImageDataset(name, cache_dir=str(cache_dir))

        episode = dataset.load_episode(0)

        assert isinstance(episode, dict)
        assert "observation" in episode
        assert "action" in episode
        assert "pixels" in episode
        # Episode 0 has 10 steps
        assert episode["observation"].shape[0] == 10
        assert episode["pixels"].shape[0] == 10


class TestVideoDataset:
    def test_init(self, sample_video_dataset):
        """Test VideoDataset initialization."""
        cache_dir, name = sample_video_dataset
        dataset = VideoDataset(name, cache_dir=str(cache_dir))

        assert dataset.path == cache_dir / name
        assert len(dataset.lengths) == 2
        assert len(dataset.offsets) == 2

    def test_len(self, sample_video_dataset):
        """Test VideoDataset length calculation."""
        cache_dir, name = sample_video_dataset
        dataset = VideoDataset(name, cache_dir=str(cache_dir))

        assert len(dataset) > 0

    def test_column_names(self, sample_video_dataset):
        """Test column_names property excludes metadata keys."""
        cache_dir, name = sample_video_dataset
        dataset = VideoDataset(name, cache_dir=str(cache_dir))

        column_names = dataset.column_names
        assert "observation" in column_names
        assert "action" in column_names
        assert "video" in column_names
        assert "ep_len" not in column_names
        assert "ep_offset" not in column_names

    def test_getitem(self, sample_video_dataset):
        """Test VideoDataset __getitem__ method."""
        cache_dir, name = sample_video_dataset
        dataset = VideoDataset(name, cache_dir=str(cache_dir))

        item = dataset[0]

        assert isinstance(item, dict)
        assert "observation" in item
        assert "action" in item
        assert "video" in item
        assert isinstance(item["video"], torch.Tensor)

    def test_video_permutation(self, sample_video_dataset):
        """Test that video frames are permuted to TCHW format."""
        cache_dir, name = sample_video_dataset
        dataset = VideoDataset(name, cache_dir=str(cache_dir))

        item = dataset[0]

        assert "video" in item
        assert item["video"].shape[-3] == 3  # channels

    def test_frameskip(self, sample_video_dataset):
        """Test VideoDataset with frameskip."""
        cache_dir, name = sample_video_dataset
        dataset = VideoDataset(name, cache_dir=str(cache_dir), frameskip=2, num_steps=2)

        assert len(dataset) > 0
        item = dataset[0]
        assert isinstance(item, dict)

    def test_keys_to_load(self, sample_video_dataset):
        """Test VideoDataset with specific keys_to_load."""
        cache_dir, name = sample_video_dataset
        dataset = VideoDataset(
            name,
            cache_dir=str(cache_dir),
            keys_to_load=["observation", "action"],
        )

        item = dataset[0]
        assert "observation" in item
        assert "action" in item
        assert "video" not in item

    def test_load_chunk(self, sample_video_dataset):
        """Test load_chunk returns correct slices."""
        cache_dir, name = sample_video_dataset
        dataset = VideoDataset(name, cache_dir=str(cache_dir))

        episodes_idx = np.array([0, 1])
        start = np.array([0, 0])
        end = np.array([3, 5])

        chunk = dataset.load_chunk(episodes_idx, start, end)

        assert isinstance(chunk, list)
        assert len(chunk) == 2
        assert "observation" in chunk[0]
        assert "action" in chunk[0]
        assert "video" in chunk[0]

    def test_get_col_data(self, sample_video_dataset):
        """Test get_col_data method."""
        cache_dir, name = sample_video_dataset
        dataset = VideoDataset(name, cache_dir=str(cache_dir))

        col_data = dataset.get_col_data("observation")
        assert isinstance(col_data, np.ndarray)
        assert col_data.shape[0] == 20

    def test_get_col_data_video_key_raises(self, sample_video_dataset):
        """Test get_col_data raises for video keys."""
        cache_dir, name = sample_video_dataset
        dataset = VideoDataset(name, cache_dir=str(cache_dir))

        with pytest.raises(KeyError, match="not in cache"):
            dataset.get_col_data("video")

    def test_get_row_data(self, sample_video_dataset):
        """Test get_row_data method."""
        cache_dir, name = sample_video_dataset
        dataset = VideoDataset(name, cache_dir=str(cache_dir))

        row_data = dataset.get_row_data(5)
        assert isinstance(row_data, dict)
        assert "observation" in row_data
        assert "action" in row_data

    def test_transform(self, sample_video_dataset):
        """Test VideoDataset with transform function."""
        cache_dir, name = sample_video_dataset

        def double_transform(data):
            for k in data:
                if data[k].dtype == torch.float32:
                    data[k] = data[k] * 2
            return data

        dataset = VideoDataset(
            name,
            cache_dir=str(cache_dir),
            transform=double_transform,
        )

        item = dataset[0]
        assert isinstance(item, dict)

    def test_load_file(self, sample_video_dataset):
        """Test _load_file method."""
        cache_dir, name = sample_video_dataset
        dataset = VideoDataset(name, cache_dir=str(cache_dir))

        frame = dataset._load_file(0, 0, "video")
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (64, 64, 3)

    def test_decord_import_error(self, sample_video_dataset):
        """Test VideoDataset raises ImportError when decord is not available."""
        cache_dir, name = sample_video_dataset

        # Reset the class-level cached decord module
        VideoDataset._decord = None

        # Mock the import to raise ImportError
        with patch.dict(sys.modules, {'decord': None}):
            with pytest.raises(ImportError, match="VideoDataset requires decord"):
                VideoDataset(name, cache_dir=str(cache_dir))


class TestMergeDataset:
    def test_init(self, mock_dataset_a, mock_dataset_b):
        """Test MergeDataset initialization."""
        merged = MergeDataset([mock_dataset_a, mock_dataset_b])

        assert len(merged.datasets) == 2
        assert len(merged) == 20

    def test_init_empty_raises(self):
        """Test MergeDataset raises error for empty list."""
        with pytest.raises(ValueError, match="Need at least one dataset"):
            MergeDataset([])

    def test_column_names_auto_dedupe(self, mock_dataset_a, mock_dataset_b):
        """Test column_names with automatic deduplication."""
        merged = MergeDataset([mock_dataset_a, mock_dataset_b])

        cols = merged.column_names
        # First dataset provides pixels, action, observation
        # Second dataset provides only audio (action, observation deduplicated)
        assert "pixels" in cols
        assert "action" in cols
        assert "observation" in cols
        assert "audio" in cols

    def test_column_names_explicit_keys(self, mock_dataset_a, mock_dataset_b):
        """Test column_names with explicit keys_from_dataset."""
        merged = MergeDataset(
            [mock_dataset_a, mock_dataset_b],
            keys_from_dataset=[
                ["pixels", "action"],
                ["audio"],
            ],
        )

        cols = merged.column_names
        assert "pixels" in cols
        assert "action" in cols
        assert "audio" in cols
        assert "observation" not in cols  # Not requested

    def test_len(self, mock_dataset_a, mock_dataset_b):
        """Test MergeDataset length."""
        merged = MergeDataset([mock_dataset_a, mock_dataset_b])
        assert len(merged) == 20

    def test_getitem_auto_dedupe(self, mock_dataset_a, mock_dataset_b):
        """Test __getitem__ with automatic deduplication."""
        merged = MergeDataset([mock_dataset_a, mock_dataset_b])

        item = merged[0]

        assert "pixels" in item
        assert "action" in item
        assert "observation" in item
        assert "audio" in item

    def test_getitem_explicit_keys(self, mock_dataset_a, mock_dataset_b):
        """Test __getitem__ with explicit keys_from_dataset."""
        merged = MergeDataset(
            [mock_dataset_a, mock_dataset_b],
            keys_from_dataset=[
                ["pixels"],
                ["audio"],
            ],
        )

        item = merged[0]

        assert "pixels" in item
        assert "audio" in item
        assert "action" not in item
        assert "observation" not in item

    def test_getitem_empty_keys(self, mock_dataset_a, mock_dataset_b):
        """Test __getitem__ when one dataset has empty keys list."""
        merged = MergeDataset(
            [mock_dataset_a, mock_dataset_b],
            keys_from_dataset=[
                ["pixels", "action", "observation"],
                [],  # Empty keys for second dataset
            ],
        )

        item = merged[0]

        assert "pixels" in item
        assert "action" in item
        assert "observation" in item
        assert "audio" not in item

    def test_load_chunk(self, mock_dataset_a, mock_dataset_b):
        """Test load_chunk method."""
        merged = MergeDataset(
            [mock_dataset_a, mock_dataset_b],
            keys_from_dataset=[
                ["pixels", "action"],
                ["audio"],
            ],
        )

        episodes_idx = np.array([0, 0])
        start = np.array([0, 5])
        end = np.array([5, 10])

        chunk = merged.load_chunk(episodes_idx, start, end)

        assert isinstance(chunk, list)
        assert len(chunk) == 2
        assert "pixels" in chunk[0]
        assert "action" in chunk[0]
        assert "audio" in chunk[0]

    def test_load_chunk_merges_all_datasets(self, mock_dataset_a, mock_dataset_b):
        """Test load_chunk merges data from all datasets."""
        merged = MergeDataset([mock_dataset_a, mock_dataset_b])

        episodes_idx = np.array([0])
        start = np.array([0])
        end = np.array([5])

        chunk = merged.load_chunk(episodes_idx, start, end)

        assert len(chunk) == 1
        # load_chunk returns data from all datasets
        assert "pixels" in chunk[0]
        assert "audio" in chunk[0]

    def test_get_col_data(self, mock_dataset_a, mock_dataset_b):
        """Test get_col_data method."""
        merged = MergeDataset(
            [mock_dataset_a, mock_dataset_b],
            keys_from_dataset=[
                ["pixels", "action"],
                ["audio"],
            ],
        )

        pixels_data = merged.get_col_data("pixels")
        assert pixels_data.shape[0] == 20

        audio_data = merged.get_col_data("audio")
        assert audio_data.shape[0] == 20

    def test_get_col_data_not_assigned_raises(self, mock_dataset_a, mock_dataset_b):
        """Test get_col_data raises for unassigned column."""
        merged = MergeDataset(
            [mock_dataset_a, mock_dataset_b],
            keys_from_dataset=[
                ["pixels"],
                ["audio"],
            ],
        )

        with pytest.raises(KeyError):
            merged.get_col_data("action")

    def test_get_row_data(self, mock_dataset_a, mock_dataset_b):
        """Test get_row_data method."""
        merged = MergeDataset(
            [mock_dataset_a, mock_dataset_b],
            keys_from_dataset=[
                ["pixels", "action"],
                ["audio"],
            ],
        )

        row_data = merged.get_row_data(5)

        assert "pixels" in row_data
        assert "action" in row_data
        assert "audio" in row_data

    def test_get_row_data_empty_keys(self, mock_dataset_a, mock_dataset_b):
        """Test get_row_data when one dataset has empty keys list."""
        merged = MergeDataset(
            [mock_dataset_a, mock_dataset_b],
            keys_from_dataset=[
                ["pixels"],
                [],
            ],
        )

        row_data = merged.get_row_data(5)

        assert "pixels" in row_data
        assert "audio" not in row_data



class TestConcatDataset:
    def test_init(self, mock_dataset_a, mock_dataset_c):
        """Test ConcatDataset initialization."""
        concat = ConcatDataset([mock_dataset_a, mock_dataset_c])

        assert len(concat.datasets) == 2
        assert concat._cum[-1] == 35  # 20 + 15

    def test_init_empty_raises(self):
        """Test ConcatDataset raises error for empty list."""
        with pytest.raises(ValueError, match="Need at least one dataset"):
            ConcatDataset([])

    def test_len(self, mock_dataset_a, mock_dataset_c):
        """Test ConcatDataset length is sum of individual lengths."""
        concat = ConcatDataset([mock_dataset_a, mock_dataset_c])
        assert len(concat) == 35

    def test_column_names(self, mock_dataset_a, mock_dataset_c):
        """Test column_names returns union of all columns."""
        concat = ConcatDataset([mock_dataset_a, mock_dataset_c])

        cols = concat.column_names
        assert "pixels" in cols
        assert "action" in cols
        assert "observation" in cols  # Only in mock_dataset_a

    def test_getitem_first_dataset(self, mock_dataset_a, mock_dataset_c):
        """Test __getitem__ returns item from first dataset."""
        concat = ConcatDataset([mock_dataset_a, mock_dataset_c])

        item = concat[0]

        assert "pixels" in item
        assert "action" in item
        assert "observation" in item

    def test_getitem_second_dataset(self, mock_dataset_a, mock_dataset_c):
        """Test __getitem__ returns item from second dataset."""
        concat = ConcatDataset([mock_dataset_a, mock_dataset_c])

        # Index 20 should be from second dataset (index 0 in second)
        item = concat[20]

        assert "pixels" in item
        assert "action" in item
        # observation not in mock_dataset_c

    def test_getitem_negative_index(self, mock_dataset_a, mock_dataset_c):
        """Test __getitem__ with negative index."""
        concat = ConcatDataset([mock_dataset_a, mock_dataset_c])

        # -1 should be last item (index 14 in second dataset)
        item = concat[-1]

        assert "pixels" in item
        assert "action" in item

    def test_loc(self, mock_dataset_a, mock_dataset_c):
        """Test _loc mapping."""
        concat = ConcatDataset([mock_dataset_a, mock_dataset_c])

        # First dataset
        ds_idx, local_idx = concat._loc(5)
        assert ds_idx == 0
        assert local_idx == 5

        # Second dataset
        ds_idx, local_idx = concat._loc(25)
        assert ds_idx == 1
        assert local_idx == 5

    def test_load_chunk(self, mock_dataset_a, mock_dataset_c):
        """Test load_chunk delegates to first dataset."""
        concat = ConcatDataset([mock_dataset_a, mock_dataset_c])

        episodes_idx = np.array([0])
        start = np.array([0])
        end = np.array([5])

        chunk = concat.load_chunk(episodes_idx, start, end)

        assert isinstance(chunk, list)
        assert len(chunk) == 1

    def test_get_col_data(self, mock_dataset_a, mock_dataset_c):
        """Test get_col_data concatenates data from all datasets."""
        concat = ConcatDataset([mock_dataset_a, mock_dataset_c])

        col_data = concat.get_col_data("pixels")
        assert col_data.shape[0] == 35  # 20 + 15

        action_data = concat.get_col_data("action")
        assert action_data.shape[0] == 35

    def test_get_col_data_not_found_raises(self, mock_dataset_a, mock_dataset_c):
        """Test get_col_data raises for missing column."""
        concat = ConcatDataset([mock_dataset_a, mock_dataset_c])

        with pytest.raises(KeyError):
            concat.get_col_data("nonexistent")

    def test_get_row_data_single_int(self, mock_dataset_a, mock_dataset_c):
        """Test get_row_data with single int index."""
        concat = ConcatDataset([mock_dataset_a, mock_dataset_c])

        # From first dataset
        row_data = concat.get_row_data(5)
        assert "pixels" in row_data
        assert "action" in row_data

        # From second dataset
        row_data = concat.get_row_data(25)
        assert "pixels" in row_data
        assert "action" in row_data

    def test_get_row_data_list(self, mock_dataset_a, mock_dataset_c):
        """Test get_row_data with list of indices."""
        concat = ConcatDataset([mock_dataset_a, mock_dataset_c])

        # Mix of indices from both datasets
        row_data = concat.get_row_data([5, 25])

        assert "pixels" in row_data
        assert "action" in row_data
        assert row_data["pixels"].shape[0] == 2
        assert row_data["action"].shape[0] == 2



class TestIntegration:
    def test_merge_then_concat(self, mock_dataset_a, mock_dataset_b, mock_dataset_c):
        """Test combining MergeDataset and ConcatDataset."""
        # Create another mock dataset with same structure as merged
        mock_dataset_d = MockDataset(
            data={
                "pixels": torch.randn(15, 3, 64, 64),
                "audio": torch.randn(15, 16000),
                "action": torch.randn(15, 2),
                "observation": torch.randn(15, 4),
            },
            length=15,
        )

        # Merge A and B
        merged = MergeDataset(
            [mock_dataset_a, mock_dataset_b],
            keys_from_dataset=[
                ["pixels", "action", "observation"],
                ["audio"],
            ],
        )

        # Concat merged with D
        concat = ConcatDataset([merged, mock_dataset_d])

        assert len(concat) == 35  # 20 + 15

        # Test accessing items
        item_from_merged = concat[5]
        assert "pixels" in item_from_merged
        assert "audio" in item_from_merged

        item_from_d = concat[25]
        assert "pixels" in item_from_d

    def test_concat_multiple_datasets(self, mock_dataset_a):
        """Test concatenating multiple datasets."""
        ds_list = [mock_dataset_a for _ in range(5)]
        concat = ConcatDataset(ds_list)

        assert len(concat) == 100  # 20 * 5

        # Test boundary access
        assert concat[0] is not None
        assert concat[19] is not None  # Last of first
        assert concat[20] is not None  # First of second
        assert concat[99] is not None  # Last overall


class TestMergeDatasetLengths:
    """Test MergeDataset.lengths property."""

    def test_lengths_property(self, mock_dataset_a, mock_dataset_b):
        """Test lengths property returns first dataset's lengths."""
        merged = MergeDataset([mock_dataset_a, mock_dataset_b])
        np.testing.assert_array_equal(merged.lengths, mock_dataset_a.lengths)


# --- HDF5Dataset fixtures for GoalDataset tests ---


@pytest.fixture
def sample_hdf5_for_goal(tmp_path):
    """Create a sample HDF5 dataset for GoalDataset testing."""
    import h5py

    h5_path = tmp_path / "goal_test.h5"

    ep_lengths = np.array([20, 15])
    ep_offsets = np.array([0, 20])
    total_steps = sum(ep_lengths)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("ep_len", data=ep_lengths)
        f.create_dataset("ep_offset", data=ep_offsets)
        f.create_dataset("observation", data=np.random.rand(total_steps, 4).astype(np.float32))
        f.create_dataset("action", data=np.random.rand(total_steps, 2).astype(np.float32))
        f.create_dataset("pixels", data=np.random.randint(0, 255, (total_steps, 64, 64, 3), dtype=np.uint8))
        f.create_dataset("proprio", data=np.random.rand(total_steps, 6).astype(np.float32))

    return tmp_path, "goal_test"


@pytest.fixture
def sample_hdf5_no_pixels(tmp_path):
    """Create a sample HDF5 dataset without pixels for GoalDataset testing."""
    import h5py

    h5_path = tmp_path / "goal_no_pixels.h5"

    ep_lengths = np.array([10])
    ep_offsets = np.array([0])
    total_steps = 10

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("ep_len", data=ep_lengths)
        f.create_dataset("ep_offset", data=ep_offsets)
        f.create_dataset("observation", data=np.random.rand(total_steps, 4).astype(np.float32))
        f.create_dataset("action", data=np.random.rand(total_steps, 2).astype(np.float32))

    return tmp_path, "goal_no_pixels"


@pytest.fixture
def sample_hdf5_grayscale(tmp_path):
    """Create a sample HDF5 dataset with grayscale images."""
    import h5py

    h5_path = tmp_path / "grayscale_test.h5"

    ep_lengths = np.array([10])
    ep_offsets = np.array([0])
    total_steps = 10

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("ep_len", data=ep_lengths)
        f.create_dataset("ep_offset", data=ep_offsets)
        f.create_dataset("action", data=np.random.rand(total_steps, 2).astype(np.float32))
        # Grayscale images (1 channel)
        f.create_dataset("pixels", data=np.random.randint(0, 255, (total_steps, 64, 64, 1), dtype=np.uint8))

    return tmp_path, "grayscale_test"


class TestGoalDataset:
    """Tests for GoalDataset wrapper."""

    def test_init(self, sample_hdf5_for_goal):
        """Test GoalDataset initialization."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(base_dataset, seed=42)

        assert goal_dataset.dataset is base_dataset
        assert goal_dataset.goal_probabilities == (0.3, 0.5, 0.2)
        assert goal_dataset.gamma == 0.99
        assert len(goal_dataset.episode_lengths) == 2
        assert goal_dataset.episode_offsets is not None

    def test_init_custom_probabilities(self, sample_hdf5_for_goal):
        """Test GoalDataset with custom goal probabilities."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(
            base_dataset,
            goal_probabilities=(0.5, 0.3, 0.2),
            seed=42,
        )

        assert goal_dataset.goal_probabilities == (0.5, 0.3, 0.2)

    def test_init_custom_gamma(self, sample_hdf5_for_goal):
        """Test GoalDataset with custom gamma."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(base_dataset, gamma=0.95, seed=42)

        assert goal_dataset.gamma == 0.95

    def test_init_invalid_probabilities_length(self, sample_hdf5_for_goal):
        """Test GoalDataset raises error for invalid probability tuple length."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))

        with pytest.raises(ValueError, match="3-tuple"):
            GoalDataset(base_dataset, goal_probabilities=(0.5, 0.5))

    def test_init_invalid_probabilities_sum(self, sample_hdf5_for_goal):
        """Test GoalDataset raises error when probabilities don't sum to 1."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))

        with pytest.raises(ValueError, match="sum to 1.0"):
            GoalDataset(base_dataset, goal_probabilities=(0.3, 0.3, 0.3))

    def test_init_custom_goal_keys(self, sample_hdf5_for_goal):
        """Test GoalDataset with custom goal_keys mapping."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(
            base_dataset,
            goal_keys={"observation": "goal_obs"},
            seed=42,
        )

        assert goal_dataset.goal_keys == {"observation": "goal_obs"}

    def test_init_auto_detect_goal_keys(self, sample_hdf5_for_goal):
        """Test GoalDataset auto-detects goal keys from column names."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(base_dataset, seed=42)

        # Should auto-detect pixels and proprio
        assert "pixels" in goal_dataset.goal_keys
        assert "proprio" in goal_dataset.goal_keys
        assert goal_dataset.goal_keys["pixels"] == "goal_pixels"
        assert goal_dataset.goal_keys["proprio"] == "goal_proprio"

    def test_init_no_pixels_or_proprio(self, sample_hdf5_no_pixels):
        """Test GoalDataset with dataset that has no pixels or proprio."""
        cache_dir, name = sample_hdf5_no_pixels
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(base_dataset, seed=42)

        # goal_keys should be empty
        assert goal_dataset.goal_keys == {}

    def test_len(self, sample_hdf5_for_goal):
        """Test GoalDataset length matches base dataset."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(base_dataset, seed=42)

        assert len(goal_dataset) == len(base_dataset)

    def test_column_names(self, sample_hdf5_for_goal):
        """Test GoalDataset column_names property."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(base_dataset, seed=42)

        assert goal_dataset.column_names == base_dataset.column_names

    def test_getitem_adds_goal_keys(self, sample_hdf5_for_goal):
        """Test __getitem__ adds goal observations."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(base_dataset, seed=42)

        item = goal_dataset[0]

        # Original keys should be present
        assert "pixels" in item
        assert "proprio" in item
        assert "action" in item
        assert "observation" in item

        # Goal keys should be added
        assert "goal_pixels" in item
        assert "goal_proprio" in item

    def test_getitem_no_goal_keys(self, sample_hdf5_no_pixels):
        """Test __getitem__ when no goal keys are configured."""
        cache_dir, name = sample_hdf5_no_pixels
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(base_dataset, seed=42)

        item = goal_dataset[0]

        # Should just return base dataset item
        assert "observation" in item
        assert "action" in item
        # No goal keys added
        assert "goal_pixels" not in item
        assert "goal_proprio" not in item

    def test_sample_goal_kind_random(self, sample_hdf5_for_goal):
        """Test _sample_goal_kind returns 'random' with high random probability."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(
            base_dataset,
            goal_probabilities=(1.0, 0.0, 0.0),  # Always random
            seed=42,
        )

        for _ in range(10):
            assert goal_dataset._sample_goal_kind() == "random"

    def test_sample_goal_kind_future(self, sample_hdf5_for_goal):
        """Test _sample_goal_kind returns 'future' with high future probability."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(
            base_dataset,
            goal_probabilities=(0.0, 1.0, 0.0),  # Always future
            seed=42,
        )

        for _ in range(10):
            assert goal_dataset._sample_goal_kind() == "future"

    def test_sample_goal_kind_current(self, sample_hdf5_for_goal):
        """Test _sample_goal_kind returns 'current' with high current probability."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(
            base_dataset,
            goal_probabilities=(0.0, 0.0, 1.0),  # Always current
            seed=42,
        )

        for _ in range(10):
            assert goal_dataset._sample_goal_kind() == "current"

    def test_sample_random_step(self, sample_hdf5_for_goal):
        """Test _sample_random_step returns valid episode and local indices."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(base_dataset, seed=42)

        for _ in range(50):
            ep_idx, local_idx = goal_dataset._sample_random_step()
            assert 0 <= ep_idx < len(goal_dataset.episode_lengths)
            assert 0 <= local_idx < goal_dataset.episode_lengths[ep_idx]

    def test_sample_random_step_empty_dataset(self, tmp_path):
        """Test _sample_random_step with empty dataset."""
        import h5py

        h5_path = tmp_path / "empty.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("ep_len", data=np.array([], dtype=np.int64))
            f.create_dataset("ep_offset", data=np.array([], dtype=np.int64))
            f.create_dataset("action", data=np.zeros((0, 2), dtype=np.float32))

        base_dataset = HDF5Dataset("empty", cache_dir=str(tmp_path))
        goal_dataset = GoalDataset(base_dataset, seed=42)

        # Should return (0, 0) for empty dataset
        ep_idx, local_idx = goal_dataset._sample_random_step()
        assert ep_idx == 0
        assert local_idx == 0

    def test_sample_future_step(self, sample_hdf5_for_goal):
        """Test _sample_future_step returns future step in same episode."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(base_dataset, seed=42)

        # Sample from episode 0, starting at step 5
        ep_idx = 0
        local_start = 5
        future_ep_idx, future_local_idx = goal_dataset._sample_future_step(ep_idx, local_start)

        # Should be same episode
        assert future_ep_idx == ep_idx
        # Should be >= local_start
        assert future_local_idx >= local_start
        # Should be within episode bounds
        assert future_local_idx < goal_dataset.episode_lengths[ep_idx]

    def test_sample_future_step_at_end(self, sample_hdf5_for_goal):
        """Test _sample_future_step when at end of episode returns same position."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(base_dataset, seed=42)

        ep_idx = 0
        # Start at last step
        local_start = goal_dataset.episode_lengths[ep_idx] - 1
        future_ep_idx, future_local_idx = goal_dataset._sample_future_step(ep_idx, local_start)

        # Should return same position
        assert future_ep_idx == ep_idx
        assert future_local_idx == local_start

    def test_get_clip_info(self, sample_hdf5_for_goal):
        """Test _get_clip_info returns correct episode and local start."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(base_dataset, seed=42)

        for idx in range(min(5, len(goal_dataset))):
            ep_idx, local_start = goal_dataset._get_clip_info(idx)
            # Should match base dataset's clip_indices
            expected = base_dataset.clip_indices[idx]
            assert (ep_idx, local_start) == expected

    def test_load_single_step(self, sample_hdf5_for_goal):
        """Test _load_single_step loads correct data."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(base_dataset, seed=42)

        step = goal_dataset._load_single_step(0, 5)

        assert isinstance(step, dict)
        assert "pixels" in step
        assert "observation" in step
        # Should be single step
        assert step["observation"].shape[0] == 1

    def test_getitem_current_goal(self, sample_hdf5_for_goal):
        """Test __getitem__ with 'current' goal sampling."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(
            base_dataset,
            goal_probabilities=(0.0, 0.0, 1.0),  # Always current
            seed=42,
        )

        item = goal_dataset[0]

        # Goal should match current observation (first frame)
        # The shapes should match
        assert item["goal_pixels"].shape == item["pixels"][:1].shape

    def test_getitem_random_goal(self, sample_hdf5_for_goal):
        """Test __getitem__ with 'random' goal sampling."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(
            base_dataset,
            goal_probabilities=(1.0, 0.0, 0.0),  # Always random
            seed=42,
        )

        item = goal_dataset[0]

        assert "goal_pixels" in item
        assert "goal_proprio" in item
        # Goal tensors should have shape (1, ...)
        assert item["goal_pixels"].ndim >= 1

    def test_getitem_future_goal(self, sample_hdf5_for_goal):
        """Test __getitem__ with 'future' goal sampling."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
        goal_dataset = GoalDataset(
            base_dataset,
            goal_probabilities=(0.0, 1.0, 0.0),  # Always future
            seed=42,
        )

        item = goal_dataset[0]

        assert "goal_pixels" in item
        assert "goal_proprio" in item

    def test_seed_reproducibility(self, sample_hdf5_for_goal):
        """Test that same seed produces same results."""
        cache_dir, name = sample_hdf5_for_goal
        base_dataset = HDF5Dataset(name, cache_dir=str(cache_dir))

        goal_dataset1 = GoalDataset(base_dataset, seed=123)
        goal_dataset2 = GoalDataset(base_dataset, seed=123)

        # Sample should be deterministic with same seed
        kind1 = [goal_dataset1._sample_goal_kind() for _ in range(10)]
        kind2 = [goal_dataset2._sample_goal_kind() for _ in range(10)]
        assert kind1 == kind2


class TestHDF5DatasetEdgeCases:
    """Additional edge case tests for HDF5Dataset."""

    def test_grayscale_image_permutation(self, sample_hdf5_grayscale):
        """Test that grayscale images (1 channel) are permuted correctly."""
        cache_dir, name = sample_hdf5_grayscale
        dataset = HDF5Dataset(name, cache_dir=str(cache_dir))

        item = dataset[0]

        assert "pixels" in item
        # Should be permuted to (T, 1, H, W)
        assert item["pixels"].shape[-3] == 1  # 1 channel

    def test_get_row_data_list_indices(self, sample_hdf5_grayscale):
        """Test get_row_data with list of indices."""
        cache_dir, name = sample_hdf5_grayscale
        dataset = HDF5Dataset(name, cache_dir=str(cache_dir))

        row_data = dataset.get_row_data([0, 2, 5])

        assert isinstance(row_data, dict)
        assert "action" in row_data
        # Should have data for 3 rows
        assert row_data["action"].shape[0] == 3


class TestImageDatasetEdgeCases:
    """Additional edge case tests for ImageDataset/FolderDataset."""

    def test_get_row_data_list(self, sample_image_dataset):
        """Test get_row_data with list of indices."""
        cache_dir, name = sample_image_dataset
        dataset = ImageDataset(name, cache_dir=str(cache_dir))

        row_data = dataset.get_row_data([0, 5, 10])

        assert isinstance(row_data, dict)
        assert "observation" in row_data
        assert "action" in row_data
        # pixels not in row_data (folder key)
