"""Tests for data module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch

from stable_worldmodel.data import HDF5Dataset
from stable_worldmodel.data.utils import get_cache_dir


def test_get_cache_dir_default():
    """Test get_cache_dir returns default path when env var not set."""
    with patch.dict(os.environ, {}, clear=True):
        if "STABLEWM_HOME" in os.environ:
            del os.environ["STABLEWM_HOME"]
        result = get_cache_dir()
        assert result == Path(os.path.expanduser("~/.stable_worldmodel"))


def test_get_cache_dir_custom():
    """Test get_cache_dir uses STABLEWM_HOME env var."""
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_path = os.path.join(tmpdir, "custom_cache")
        with patch.dict(os.environ, {"STABLEWM_HOME": custom_path}):
            result = get_cache_dir()
            assert result == Path(custom_path)
            assert result.exists()


def test_get_cache_dir_creates_directory():
    """Test get_cache_dir creates the directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_path = os.path.join(tmpdir, "new_cache_dir")
        assert not os.path.exists(custom_path)
        with patch.dict(os.environ, {"STABLEWM_HOME": custom_path}):
            result = get_cache_dir()
            assert result.exists()

@pytest.fixture
def sample_h5_file(tmp_path):
    """Create a sample HDF5 file for testing."""
    h5_path = tmp_path / "test_dataset.h5"

    # Create sample data: 2 episodes, 10 steps each
    ep_lengths = [10, 10]
    ep_offsets = [0, 10]
    total_steps = sum(ep_lengths)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("ep_len", data=np.array(ep_lengths))
        f.create_dataset("ep_offset", data=np.array(ep_offsets))

        # Sample observation data
        f.create_dataset("observation", data=np.random.rand(total_steps, 4).astype(np.float32))

        # Sample action data
        f.create_dataset("action", data=np.random.rand(total_steps, 2).astype(np.float32))

        # Sample image data (THWC format)
        f.create_dataset("pixels", data=np.random.randint(0, 255, (total_steps, 64, 64, 3), dtype=np.uint8))

    return tmp_path, "test_dataset"


@pytest.fixture
def sample_h5_short_episode(tmp_path):
    """Create a sample HDF5 file with a short episode."""
    h5_path = tmp_path / "short_dataset.h5"

    # Create sample data: 2 episodes, different lengths
    ep_lengths = [3, 10]  # First episode too short for default span
    ep_offsets = [0, 3]
    total_steps = sum(ep_lengths)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("ep_len", data=np.array(ep_lengths))
        f.create_dataset("ep_offset", data=np.array(ep_offsets))
        f.create_dataset("observation", data=np.random.rand(total_steps, 4).astype(np.float32))
        f.create_dataset("action", data=np.random.rand(total_steps, 2).astype(np.float32))

    return tmp_path, "short_dataset"


def test_hdf5_dataset_init(sample_h5_file):
    """Test HDF5Dataset initialization."""
    cache_dir, name = sample_h5_file
    dataset = HDF5Dataset(name, cache_dir=str(cache_dir))

    assert dataset.h5_path == cache_dir / f"{name}.h5"
    assert len(dataset.lengths) == 2
    assert len(dataset.offsets) == 2


def test_hdf5_dataset_len(sample_h5_file):
    """Test HDF5Dataset length calculation."""
    cache_dir, name = sample_h5_file
    dataset = HDF5Dataset(name, cache_dir=str(cache_dir))

    # With default num_steps=1 and frameskip=1, each step is a valid clip
    assert len(dataset) > 0


def test_hdf5_dataset_column_names(sample_h5_file):
    """Test column_names property excludes metadata keys."""
    cache_dir, name = sample_h5_file
    dataset = HDF5Dataset(name, cache_dir=str(cache_dir))

    column_names = dataset.column_names
    assert "observation" in column_names
    assert "action" in column_names
    assert "pixels" in column_names
    assert "ep_len" not in column_names
    assert "ep_offset" not in column_names


def test_hdf5_dataset_getitem(sample_h5_file):
    """Test HDF5Dataset __getitem__ method."""
    cache_dir, name = sample_h5_file
    dataset = HDF5Dataset(name, cache_dir=str(cache_dir))

    item = dataset[0]

    assert isinstance(item, dict)
    assert "observation" in item
    assert "action" in item
    assert isinstance(item["observation"], torch.Tensor)
    assert isinstance(item["action"], torch.Tensor)


def test_hdf5_dataset_image_permutation(sample_h5_file):
    """Test that images are permuted to TCHW format."""
    cache_dir, name = sample_h5_file
    dataset = HDF5Dataset(name, cache_dir=str(cache_dir))

    item = dataset[0]

    # Image should be in TCHW format (channels first)
    assert "pixels" in item
    # With num_steps=1, shape should be (1, 3, 64, 64)
    assert item["pixels"].shape[-3] == 3  # channels


def test_hdf5_dataset_frameskip(sample_h5_file):
    """Test HDF5Dataset with frameskip."""
    cache_dir, name = sample_h5_file
    dataset = HDF5Dataset(name, cache_dir=str(cache_dir), frameskip=2, num_steps=2)

    # Dataset should still work with frameskip
    assert len(dataset) > 0
    item = dataset[0]
    assert isinstance(item, dict)


def test_hdf5_dataset_keys_to_load(sample_h5_file):
    """Test HDF5Dataset with specific keys_to_load."""
    cache_dir, name = sample_h5_file
    dataset = HDF5Dataset(
        name,
        cache_dir=str(cache_dir),
        keys_to_load=["observation", "action", "ep_len", "ep_offset"],
    )

    item = dataset[0]
    assert "observation" in item
    assert "action" in item
    assert "pixels" not in item


def test_hdf5_dataset_keys_to_cache(sample_h5_file):
    """Test HDF5Dataset with keys_to_cache."""
    cache_dir, name = sample_h5_file
    dataset = HDF5Dataset(
        name,
        cache_dir=str(cache_dir),
        keys_to_cache=["observation"],
    )

    assert "observation" in dataset._cache
    assert "action" not in dataset._cache

    # Verify cached data is used during load
    item = dataset[0]
    assert "observation" in item
    assert isinstance(item["observation"], torch.Tensor)


def test_hdf5_dataset_cache_missing_key(sample_h5_file):
    """Test HDF5Dataset raises error for missing cache key."""
    cache_dir, name = sample_h5_file

    with pytest.raises(KeyError):
        HDF5Dataset(
            name,
            cache_dir=str(cache_dir),
            keys_to_cache=["nonexistent_key"],
        )


def test_hdf5_dataset_get_col_data(sample_h5_file):
    """Test get_col_data method."""
    cache_dir, name = sample_h5_file
    dataset = HDF5Dataset(name, cache_dir=str(cache_dir))

    col_data = dataset.get_col_data("observation")
    assert isinstance(col_data, np.ndarray)
    assert col_data.shape[0] == 20  # Total steps


def test_hdf5_dataset_get_row_data(sample_h5_file):
    """Test get_row_data method."""
    cache_dir, name = sample_h5_file
    dataset = HDF5Dataset(name, cache_dir=str(cache_dir))

    row_data = dataset.get_row_data(5)
    assert isinstance(row_data, dict)
    assert "observation" in row_data

def test_hdf5_dataset_load_chunk(sample_h5_file):
    """Test load_chunk returns correct slices for multiple episodes."""
    cache_dir, name = sample_h5_file
    dataset = HDF5Dataset(name, cache_dir=str(cache_dir))

    episodes_idx = np.array([0, 1])
    start = np.array([2, 0])
    end = np.array([5, 5])

    chunk = dataset.load_chunk(episodes_idx, start, end)

    assert isinstance(chunk, list)
    assert len(chunk) == 2

    # First chunk: episode 0, steps 2-5 (3 steps)
    assert "observation" in chunk[0]
    assert "action" in chunk[0]
    assert chunk[0]["observation"].shape == (3, 4)
    assert chunk[0]["action"].shape == (3, 2)

    # Second chunk: episode 1, steps 0-5 (5 steps)
    assert chunk[1]["observation"].shape == (5, 4)
    assert chunk[1]["action"].shape == (5, 2)

    # Verify tensors
    assert isinstance(chunk[0]["observation"], torch.Tensor)
    assert isinstance(chunk[1]["action"], torch.Tensor)


def test_hdf5_dataset_transform(sample_h5_file):
    """Test HDF5Dataset with transform function."""
    cache_dir, name = sample_h5_file

    def double_transform(data):
        for k in data:
            if data[k].dtype == torch.float32:
                data[k] = data[k] * 2
        return data

    dataset = HDF5Dataset(
        name,
        cache_dir=str(cache_dir),
        transform=double_transform,
    )

    item = dataset[0]
    assert isinstance(item, dict)


def test_hdf5_dataset_short_episode_filtered(sample_h5_short_episode):
    """Test that episodes shorter than span are filtered out."""
    cache_dir, name = sample_h5_short_episode
    dataset = HDF5Dataset(name, cache_dir=str(cache_dir), num_steps=5, frameskip=1)

    # Only second episode (length 10) should have valid clips
    # First episode (length 3) is too short for span=5
    for ep_idx, _ in dataset.clip_indices:
        assert ep_idx == 1  # Only second episode


def test_hdf5_dataset_file_not_found(tmp_path):
    """Test HDF5Dataset raises error for missing file."""
    with pytest.raises(FileNotFoundError):
        HDF5Dataset("nonexistent", cache_dir=str(tmp_path))
