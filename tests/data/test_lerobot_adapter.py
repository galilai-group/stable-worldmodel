"""Integration tests for the LeRobot dataset adapter."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from stable_worldmodel import World
from stable_worldmodel.data import GoalDataset, HDF5Dataset, LeRobotAdapter
from stable_worldmodel.policy import RandomPolicy

# Lightweight public dataset used as the integration target.
PUSHT_REPO_ID = 'lerobot/pusht'

# pyav is the portable default; avoids a hard dependency on torchcodec/FFmpeg.
PUSHT_VIDEO_BACKEND = 'pyav'


def _skip_if_unavailable() -> None:
    if sys.version_info < (3, 12):
        pytest.skip('LeRobot tests require Python 3.12+.')
    pytest.importorskip('lerobot')


def _make_dataset(**kwargs) -> LeRobotAdapter:
    _skip_if_unavailable()
    kwargs.setdefault('video_backend', PUSHT_VIDEO_BACKEND)
    try:
        dataset = LeRobotAdapter(repo_id=PUSHT_REPO_ID, **kwargs)
    except Exception as exc:  # pragma: no cover - network/disk dependent
        pytest.skip(f'Unable to load {PUSHT_REPO_ID}: {exc!r}')
    if len(dataset) == 0:
        pytest.skip(f'{PUSHT_REPO_ID} is empty for the selected episodes.')
    return dataset


@pytest.fixture(scope='module')
def pusht() -> LeRobotAdapter:
    return _make_dataset()


def test_lerobot_adapter_default_aliases(pusht):
    assert {'pixels', 'action', 'ep_idx', 'step_idx'}.issubset(
        set(pusht.column_names)
    )
    assert len(pusht.lengths) > 0
    assert int(pusht.lengths.sum()) == len(pusht)
    assert pusht.offsets[0] == 0

    ep_idx = pusht.get_col_data('ep_idx')
    step_idx = pusht.get_col_data('step_idx')
    assert ep_idx.ndim == 1
    assert step_idx.ndim == 1
    assert ep_idx.shape == step_idx.shape
    assert ep_idx.shape[0] == len(pusht)


def test_lerobot_adapter_item_and_chunk_behavior():
    dataset = _make_dataset(num_steps=2, frameskip=1, keys_to_cache=['action'])

    item = dataset[0]
    assert item['pixels'].shape[0] == 2
    assert item['pixels'].shape[1] in (1, 3)
    assert item['action'].shape[0] == 2

    chunk = dataset.load_chunk(
        np.array([0]),
        np.array([0]),
        np.array([2]),
    )
    assert len(chunk) == 1
    assert chunk[0]['pixels'].shape[0] == 2
    assert dataset._window_datasets


def test_lerobot_adapter_subset_localizes_episode_indices(pusht):
    if len(pusht.lengths) < 2:
        pytest.skip('Need at least two episodes to validate subset remapping.')

    second_abs_episode = int(pusht._absolute_episode_ids[1])
    subset = _make_dataset(episodes=[second_abs_episode])

    assert subset.lengths.shape == (1,)
    assert subset.offsets.tolist() == [0]
    assert set(subset.get_col_data('ep_idx').tolist()) == {0}
    step_idx = subset.get_col_data('step_idx')
    assert step_idx[0] == 0
    assert int(step_idx[-1]) == int(subset.lengths[0] - 1)


def test_lerobot_adapter_get_row_data_and_image_column_error(pusht):
    row = pusht.get_row_data([0, 1])
    assert row['ep_idx'].shape == (2,)
    assert row['step_idx'].shape == (2,)
    assert row['action'].shape[0] == 2

    with pytest.raises(KeyError):
        pusht.get_col_data('pixels')


def test_lerobot_adapter_goal_dataset_compatibility():
    dataset = _make_dataset(num_steps=2, keys_to_cache=['action'])
    goal_dataset = GoalDataset(
        dataset,
        goal_probabilities=(0.0, 0.0, 0.0, 1.0),
        current_goal_offset=2,
        goal_keys={'pixels': 'goal_pixels', 'action': 'goal_action'},
        seed=123,
    )
    item = goal_dataset[0]
    # goal is a single frame: (C, H, W)
    assert item['goal_pixels'].ndim == 3
    assert item['goal_pixels'].shape[0] in (1, 3)
    # goal action is a single step: (1, action_dim)
    assert item['goal_action'].shape[0] == 1


def test_lerobot_adapter_pusht_matches_native_swm_dataset(tmp_path):
    """Hub `lerobot/pusht` via LeRobotAdapter matches native `swm/PushT-v1` HDF5 layout.

    Records PushT with `World.collect` (the supported path) at the same
    resolution as the Hub dataset, then checks that `__getitem__` batches agree
    on tensor types and shapes for `pixels` (T, C, H, W) and `action` (T, D).
    Trajectories differ (different sources); this test locks the *contract*.
    """
    NUM_STEPS = 2
    FRAMESKIP = 1

    adapter = _make_dataset(
        num_steps=NUM_STEPS,
        frameskip=FRAMESKIP,
        keys_to_cache=['action'],
    )
    hub_item = adapter[0]
    H, W = int(hub_item['pixels'].shape[-2]), int(hub_item['pixels'].shape[-1])

    world = World(
        env_name='swm/PushT-v1',
        num_envs=2,
        image_shape=(H, W),
        max_episode_steps=40,
    )
    world.set_policy(RandomPolicy())
    dataset_name = 'native_pusht_lerobot_compare'
    world.collect(
        tmp_path / 'datasets' / f'{dataset_name}.h5',
        episodes=3,
        seed=123,
        format='hdf5',
    )
    world.envs.close()

    native = HDF5Dataset(
        name=dataset_name,
        cache_dir=str(tmp_path),
        num_steps=NUM_STEPS,
        frameskip=FRAMESKIP,
        keys_to_load=['pixels', 'action'],
        keys_to_cache=['action'],
    )
    assert len(native) > 0
    native_item = native[0]

    assert 'pixels' in hub_item and 'action' in hub_item
    assert 'pixels' in native_item and 'action' in native_item

    for key in ('pixels', 'action'):
        assert isinstance(hub_item[key], torch.Tensor)
        assert isinstance(native_item[key], torch.Tensor)

    assert hub_item['pixels'].shape == native_item['pixels'].shape
    assert hub_item['action'].shape == native_item['action'].shape

    c = hub_item['pixels'].shape[1]
    assert c in (1, 3)
    assert native_item['pixels'].shape[1] == c
