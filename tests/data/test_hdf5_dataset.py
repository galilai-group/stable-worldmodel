"""Tests for HDF5Dataset picklability (DataLoader spawn/forkserver support)."""

import multiprocessing as mp
import pickle

import h5py
import numpy as np
import pytest
import torch

from stable_worldmodel.data import HDF5Dataset


def _write_synthetic_h5(
    path,
    n_episodes=4,
    ep_len=40,
    image_hw=(64, 64),
    action_dim=2,
    proprio_dim=5,
):
    """Write a minimal pusht-shaped synthetic HDF5 file."""
    lengths = np.full(n_episodes, ep_len, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    total = int(lengths.sum())
    h, w = image_hw
    with h5py.File(path, 'w') as f:
        f.create_dataset('ep_len', data=lengths)
        f.create_dataset('ep_offset', data=offsets)
        f.create_dataset(
            'pixels',
            data=(np.random.rand(total, h, w, 3) * 255).astype(np.uint8),
        )
        f.create_dataset(
            'action',
            data=np.random.randn(total, action_dim).astype(np.float32),
        )
        f.create_dataset(
            'proprio',
            data=np.random.randn(total, proprio_dim).astype(np.float32),
        )
        f.create_dataset(
            'state',
            data=np.random.randn(total, proprio_dim).astype(np.float32),
        )


@pytest.fixture
def synth_dataset(tmp_path):
    """HDF5Dataset over a synthetic pusht-shaped file."""
    datasets_dir = tmp_path / 'datasets'
    datasets_dir.mkdir(parents=True, exist_ok=True)
    _write_synthetic_h5(datasets_dir / 'synth.h5')
    return HDF5Dataset(
        name='synth',
        frameskip=5,
        num_steps=4,
        keys_to_load=['pixels', 'action', 'proprio', 'state'],
        keys_to_cache=['action', 'proprio', 'state'],
        cache_dir=tmp_path,
    )


def test_pickle_roundtrip_closes_handle(synth_dataset):
    """Pickling a dataset with an open h5 handle must succeed and drop the handle."""
    # Force the h5 handle open via a non-cached key ('pixels').
    _ = synth_dataset[0]
    assert synth_dataset.h5_file is not None

    blob = pickle.dumps(synth_dataset)
    clone = pickle.loads(blob)

    # The child must arrive with no live handle; it reopens lazily.
    assert clone.h5_file is None
    sample = clone[0]
    assert clone.h5_file is not None
    assert set(sample.keys()) == {'pixels', 'action', 'proprio', 'state'}
    assert sample['pixels'].shape[0] > 0


def test_dataloader_spawn_with_open_handle(synth_dataset):
    """DataLoader with spawn workers works after main process opens the h5 handle.

    Regression test: before adding ``__getstate__`` to ``HDF5Dataset`` this
    raised ``TypeError: h5py objects cannot be pickled`` on macOS/Windows,
    where the PyTorch DataLoader defaults to the ``spawn`` start method.
    """
    _ = synth_dataset[0]
    assert synth_dataset.h5_file is not None

    spawn_ctx = mp.get_context('spawn')
    loader = torch.utils.data.DataLoader(
        synth_dataset,
        batch_size=2,
        num_workers=2,
        shuffle=False,
        drop_last=True,
        multiprocessing_context=spawn_ctx,
    )
    batches = 0
    for batch in loader:
        assert set(batch.keys()) == {'pixels', 'action', 'proprio', 'state'}
        assert batch['pixels'].shape[0] == 2
        batches += 1
        if batches >= 3:
            break
    assert batches >= 1


def test_dataloader_fork_still_works(synth_dataset):
    """Fork must remain unaffected (handles inherited via copy-on-write)."""
    _ = synth_dataset[0]
    fork_ctx = mp.get_context('fork')
    loader = torch.utils.data.DataLoader(
        synth_dataset,
        batch_size=2,
        num_workers=2,
        shuffle=False,
        drop_last=True,
        multiprocessing_context=fork_ctx,
    )
    for batch in loader:
        assert batch['pixels'].shape[0] == 2
        break
