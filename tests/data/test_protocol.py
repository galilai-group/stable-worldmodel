"""Tests for Dataset base class and interface."""

import numpy as np
import pytest
import torch
import h5py

from stable_worldmodel.data import Dataset, HDF5Dataset


@pytest.fixture
def sample_h5_file(tmp_path):
    """Create a sample HDF5 file for testing."""
    h5_path = tmp_path / "test_dataset.h5"

    ep_lengths = [10, 10]
    ep_offsets = [0, 10]
    total_steps = sum(ep_lengths)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("ep_len", data=np.array(ep_lengths))
        f.create_dataset("ep_offset", data=np.array(ep_offsets))
        f.create_dataset("observation", data=np.random.rand(total_steps, 4).astype(np.float32))
        f.create_dataset("action", data=np.random.rand(total_steps, 2).astype(np.float32))

    return tmp_path, "test_dataset"


def test_hdf5_dataset_is_dataset_subclass(sample_h5_file):
    """HDF5Dataset should be a subclass of Dataset."""
    cache_dir, name = sample_h5_file
    dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
    assert isinstance(dataset, Dataset)


def test_dataset_len_returns_int(sample_h5_file):
    """__len__ returns an integer."""
    cache_dir, name = sample_h5_file
    dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
    assert isinstance(len(dataset), int)


def test_dataset_getitem_returns_dict(sample_h5_file):
    """__getitem__ returns a dict of tensors."""
    cache_dir, name = sample_h5_file
    dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
    item = dataset[0]
    assert isinstance(item, dict)
    for v in item.values():
        assert isinstance(v, torch.Tensor)


def test_dataset_column_names_returns_list(sample_h5_file):
    """column_names returns a list of strings."""
    cache_dir, name = sample_h5_file
    dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
    cols = dataset.column_names
    assert isinstance(cols, list)
    for c in cols:
        assert isinstance(c, str)


def test_dataset_load_chunk_returns_list(sample_h5_file):
    """load_chunk returns a list of dicts."""
    cache_dir, name = sample_h5_file
    dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
    episodes_idx = np.array([0, 0])
    start = np.array([0, 2])
    end = np.array([2, 5])
    chunk = dataset.load_chunk(episodes_idx, start, end)
    assert isinstance(chunk, list)
    for item in chunk:
        assert isinstance(item, dict)
        for v in item.values():
            assert isinstance(v, torch.Tensor)


def test_dataset_base_class_raises_not_implemented():
    """Base Dataset class raises NotImplementedError for abstract methods."""
    lengths = np.array([10, 10])
    offsets = np.array([0, 10])
    dataset = Dataset(lengths, offsets)

    with pytest.raises(NotImplementedError):
        _ = dataset.column_names

    with pytest.raises(NotImplementedError):
        dataset._load_slice(0, 0, 1)

    with pytest.raises(NotImplementedError):
        dataset.get_col_data("col")

    with pytest.raises(NotImplementedError):
        dataset.get_row_data(0)
