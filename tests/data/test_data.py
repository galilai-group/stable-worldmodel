"""Tests for data module."""

import io
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from stable_worldmodel.data import (
    HDF5Dataset,
    LanceDataset,
    convert_hdf5_to_lance,
)
from stable_worldmodel.data.utils import create_dataset, get_cache_dir
from stable_worldmodel.utils import DEFAULT_CACHE_DIR


def test_get_cache_dir_default():
    """Test get_cache_dir returns default path when env var not set."""
    with patch.dict(os.environ, {}, clear=True):
        if "STABLEWM_HOME" in os.environ:
            del os.environ["STABLEWM_HOME"]
        result = get_cache_dir()
        expected = Path(DEFAULT_CACHE_DIR)
        assert result == expected


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
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    h5_path = datasets_dir / "test_dataset.h5"

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


def _encode_png(frame: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.read()


def _build_shared_arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    ep_lengths = np.array([4, 4], dtype=np.int32)
    ep_offsets = np.array([0, 4], dtype=np.int32)
    total_steps = int(ep_lengths.sum())
    episode_idx = np.repeat(np.arange(len(ep_lengths)), ep_lengths)
    step_idx = np.concatenate(
        [np.arange(length, dtype=np.int32) for length in ep_lengths]
    )
    observation = rng.standard_normal((total_steps, 3)).astype(np.float32)
    action = rng.standard_normal((total_steps, 2)).astype(np.float32)
    pixels = rng.integers(0, 255, (total_steps, 8, 8, 3), dtype=np.uint8)
    return {
        'ep_lengths': ep_lengths,
        'ep_offsets': ep_offsets,
        'episode_idx': episode_idx.astype(np.int32),
        'step_idx': step_idx.astype(np.int32),
        'observation': observation,
        'action': action,
        'pixels': pixels,
    }


@pytest.fixture
def paired_datasets(tmp_path):
    """Create matching HDF5 and Lance datasets from the same arrays."""

    lancedb = pytest.importorskip('lancedb')

    data = _build_shared_arrays()
    dataset_name = 'paired_dataset'

    datasets_dir = tmp_path / 'datasets'
    datasets_dir.mkdir()
    h5_path = datasets_dir / f'{dataset_name}.h5'
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('ep_len', data=data['ep_lengths'])
        f.create_dataset('ep_offset', data=data['ep_offsets'])
        f.create_dataset('observation', data=data['observation'])
        f.create_dataset('action', data=data['action'])
        f.create_dataset('pixels', data=data['pixels'])

    records = []
    for idx in range(len(data['episode_idx'])):
        records.append(
            {
                'episode_idx': int(data['episode_idx'][idx]),
                'step_idx': int(data['step_idx'][idx]),
                'pixels': _encode_png(data['pixels'][idx]),
                'observation': data['observation'][idx].tolist(),
                'action': data['action'][idx].tolist(),
            }
        )

    lance_uri = tmp_path / 'lance_db'
    db = lancedb.connect(str(lance_uri))
    table_name = 'paired_table'
    db.create_table(table_name, records, mode='overwrite')

    return {
        'h5': (tmp_path, dataset_name),
        'lance': {'uri': str(lance_uri), 'table_name': table_name},
        'keys': ['pixels', 'action', 'observation'],
        'total_rows': int(data['ep_lengths'].sum()),
    }


@pytest.fixture
def sample_h5_short_episode(tmp_path):
    """Create a sample HDF5 file with a short episode."""
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    h5_path = datasets_dir / "short_dataset.h5"

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


def _make_hdf5_lance_pair(
    paired_datasets,
    *,
    num_steps: int,
    frameskip: int,
    keys_to_cache: list[str] | None = None,
) -> tuple[HDF5Dataset, LanceDataset]:
    cache_dir, dataset_name = paired_datasets['h5']
    lance_cfg = paired_datasets['lance']
    keys = paired_datasets['keys']
    h5_kwargs = {
        'num_steps': num_steps,
        'frameskip': frameskip,
        'keys_to_load': keys,
        'keys_to_cache': keys_to_cache or [],
    }
    lance_kwargs = {
        'num_steps': num_steps,
        'frameskip': frameskip,
        'keys_to_load': keys,
        'image_columns': ['pixels'],
        'keys_to_cache': keys_to_cache or [],
    }
    h5_ds = HDF5Dataset(dataset_name, cache_dir=str(cache_dir), **h5_kwargs)
    lance_ds = LanceDataset(
        uri=lance_cfg['uri'], table_name=lance_cfg['table_name'], **lance_kwargs
    )
    return h5_ds, lance_ds


def test_hdf5_dataset_init(sample_h5_file):
    """Test HDF5Dataset initialization."""
    cache_dir, name = sample_h5_file
    dataset = HDF5Dataset(name, cache_dir=str(cache_dir))

    assert dataset.h5_path == cache_dir / "datasets" / f"{name}.h5"
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


def test_lance_dataset_matches_hdf5_length(paired_datasets):
    """LanceDataset should expose the same number of samples as HDF5Dataset."""
    h5_ds, lance_ds = _make_hdf5_lance_pair(
        paired_datasets, num_steps=2, frameskip=1
    )
    assert len(h5_ds) == len(lance_ds)


def test_lance_dataset_output_parity(paired_datasets):
    """Individual samples should match between Lance and HDF5 backends."""
    h5_ds, lance_ds = _make_hdf5_lance_pair(
        paired_datasets, num_steps=2, frameskip=1
    )
    for idx in range(len(lance_ds)):
        h_item, l_item = h5_ds[idx], lance_ds[idx]
        assert torch.equal(h_item['pixels'], l_item['pixels'])
        assert torch.allclose(h_item['action'], l_item['action'])
        assert torch.allclose(h_item['observation'], l_item['observation'])


def test_lance_dataset_output_parity_frameskip(paired_datasets):
    """Parity holds with frameskip > 1: downsampling and action reshape must match."""
    # Episodes are length 4; frameskip=2, num_steps=2 → span=4, exactly fills one episode.
    h5_ds, lance_ds = _make_hdf5_lance_pair(
        paired_datasets, num_steps=2, frameskip=2
    )
    assert len(h5_ds) == len(lance_ds)
    for idx in range(len(lance_ds)):
        h_item, l_item = h5_ds[idx], lance_ds[idx]
        # observation is downsampled: shape (num_steps, obs_dim)
        assert h_item['observation'].shape == l_item['observation'].shape
        assert torch.allclose(h_item['observation'], l_item['observation'])
        # action is NOT downsampled, then reshaped to (num_steps, frameskip * action_dim)
        assert h_item['action'].shape == l_item['action'].shape
        assert torch.allclose(h_item['action'], l_item['action'])
        # pixels: one frame per macro-step after frameskip
        assert h_item['pixels'].shape == l_item['pixels'].shape
        assert torch.equal(h_item['pixels'], l_item['pixels'])


def test_lance_dataset_getitems_matches_getitem(paired_datasets):
    """__getitems__ batch path must return the same values as N individual __getitem__ calls."""
    _, lance_ds = _make_hdf5_lance_pair(
        paired_datasets, num_steps=2, frameskip=1
    )
    indices = list(range(len(lance_ds)))
    # Batch path: one Lance take for all samples
    batch_results = lance_ds.__getitems__(indices)
    # Single-item path: one Lance take per sample
    for i, idx in enumerate(indices):
        single = lance_ds[idx]
        assert torch.equal(batch_results[i]['pixels'], single['pixels']), \
            f"pixels mismatch at idx={idx}"
        assert torch.allclose(batch_results[i]['action'], single['action']), \
            f"action mismatch at idx={idx}"
        assert torch.allclose(batch_results[i]['observation'], single['observation']), \
            f"observation mismatch at idx={idx}"


def test_lance_dataset_get_col_data(paired_datasets):
    """get_col_data should return full columns when reading from Lance."""
    _, lance_ds = _make_hdf5_lance_pair(
        paired_datasets, num_steps=2, frameskip=1
    )
    obs = lance_ds.get_col_data('observation')
    assert isinstance(obs, np.ndarray)
    assert obs.shape[0] == paired_datasets['total_rows']


def test_lance_dataset_keys_to_cache(paired_datasets):
    """Keys listed in keys_to_cache are materialized eagerly."""
    _, lance_ds = _make_hdf5_lance_pair(
        paired_datasets,
        num_steps=2,
        frameskip=1,
        keys_to_cache=['observation'],
    )
    assert 'observation' in lance_ds._cache


def test_create_dataset_lance_backend(paired_datasets):
    """create_dataset should instantiate the Lance backend when requested."""
    lance_cfg = paired_datasets['lance']
    dataset = create_dataset(
        {
            'type': 'lance',
            'uri': lance_cfg['uri'],
            'table_name': lance_cfg['table_name'],
            'num_steps': 2,
            'frameskip': 1,
            'keys_to_load': paired_datasets['keys'],
            'image_columns': ['pixels'],
        }
    )
    assert isinstance(dataset, LanceDataset)


def test_create_dataset_infers_lance_backend(paired_datasets):
    """When Lance fields are provided without type, inference should kick in."""
    lance_cfg = paired_datasets['lance']
    dataset = create_dataset(
        {
            'uri': lance_cfg['uri'],
            'table_name': lance_cfg['table_name'],
            'num_steps': 2,
            'frameskip': 1,
            'keys_to_load': paired_datasets['keys'],
            'image_columns': ['pixels'],
        }
    )
    assert isinstance(dataset, LanceDataset)


def test_convert_hdf5_to_lance_multi_camera(tmp_path):
    """Converter should encode multiple image columns and transparently rename
    dot-separated HDF5 names to underscore-separated Lance names."""
    lancedb = pytest.importorskip('lancedb')

    # HDF5 file uses dot-separated names (common in real datasets).
    # The converter must rename them to underscores without user intervention.
    datasets_dir = tmp_path / 'datasets'
    datasets_dir.mkdir()
    h5_path = datasets_dir / 'multi_cam.h5'
    ep_lengths = [4, 4]
    total = sum(ep_lengths)
    rng = np.random.default_rng(0)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('ep_len', data=np.array(ep_lengths))
        f.create_dataset('ep_offset', data=np.array([0, 4]))
        f.create_dataset('pixels.top', data=rng.integers(0, 255, (total, 8, 8, 3), dtype=np.uint8))
        f.create_dataset('pixels.wrist', data=rng.integers(0, 255, (total, 8, 8, 3), dtype=np.uint8))
        f.create_dataset('action', data=rng.standard_normal((total, 2)).astype(np.float32))

    lance_dir = tmp_path / 'lance'
    summary = convert_hdf5_to_lance(
        h5_path=h5_path,
        lance_uri=str(lance_dir),
        table_name='multi_cam',
        columns=['pixels.top', 'pixels.wrist', 'action'],
        overwrite=True,
    )
    assert summary['rows'] == total

    # Lance table has underscore names; LanceDataset auto-detects them as image columns.
    ds = LanceDataset(
        uri=str(lance_dir),
        table_name='multi_cam',
        num_steps=1,
        keys_to_load=['pixels_top', 'pixels_wrist', 'action'],
    )
    assert 'pixels_top' in ds.image_columns
    assert 'pixels_wrist' in ds.image_columns

    item = ds[0]
    assert item['pixels_top'].shape == (1, 3, 8, 8)    # (T, C, H, W)
    assert item['pixels_wrist'].shape == (1, 3, 8, 8)
    assert item['action'].shape == (1, 2)


def test_convert_hdf5_to_lance(tmp_path, sample_h5_file):
    """The conversion helper should produce a Lance table compatible with LanceDataset."""
    cache_dir, name = sample_h5_file
    h5_path = cache_dir / 'datasets' / f'{name}.h5'
    lance_dir = tmp_path / 'lance'

    summary = convert_hdf5_to_lance(
        h5_path=h5_path,
        lance_uri=str(lance_dir),
        table_name='test_table',
        columns=['pixels', 'action', 'observation'],
        batch_rows=4,
        jpeg_quality=90,
        overwrite=True,
        max_steps=10,
    )

    assert summary['rows'] == 10

    h5_dataset = HDF5Dataset(
        name,
        cache_dir=str(cache_dir),
        num_steps=1,
        keys_to_load=['pixels', 'action', 'observation'],
    )
    lance_dataset = LanceDataset(
        uri=str(lance_dir),
        table_name='test_table',
        num_steps=1,
        keys_to_load=['pixels', 'action', 'observation'],
        image_columns=['pixels'],
    )

    h5_item = h5_dataset[0]
    lance_item = lance_dataset[0]

    assert torch.allclose(lance_item['action'], h5_item['action'])
    assert torch.allclose(lance_item['observation'], h5_item['observation'])
    assert lance_item['pixels'].shape[1:] == h5_item['pixels'].shape[1:]
