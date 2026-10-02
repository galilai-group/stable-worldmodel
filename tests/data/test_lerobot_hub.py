"""Opt-in check of the LeRobot adapter against the real Hub dataset.

It downloads ``lerobot/pusht`` from the Hugging Face Hub, so it only runs
when ``SWM_LEROBOT_HUB_TESTS=1`` is set. The offline tests with a tiny local
dataset live in ``test_lerobot_adapter.py``.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

if os.environ.get('SWM_LEROBOT_HUB_TESTS') != '1':
    pytest.skip(
        'set SWM_LEROBOT_HUB_TESTS=1 to download lerobot/pusht',
        allow_module_level=True,
    )
if sys.version_info < (3, 12):
    pytest.skip('lerobot requires Python 3.12+', allow_module_level=True)

pytest.importorskip('lerobot')

from stable_worldmodel import World  # noqa: E402
from stable_worldmodel.data import HDF5Dataset, LeRobotAdapter  # noqa: E402
from stable_worldmodel.policy import RandomPolicy  # noqa: E402

PUSHT_REPO_ID = 'lerobot/pusht'


def test_lerobot_adapter_pusht_matches_native_swm_dataset(tmp_path):
    """Hub `lerobot/pusht` via LeRobotAdapter matches native `swm/PushT-v1` HDF5 layout.

    Records PushT with `World.collect` (the supported path) at the same
    resolution as the Hub dataset, then checks that `__getitem__` batches agree
    on tensor types and shapes for `pixels` (T, C, H, W) and `action` (T, D).
    Trajectories differ (different sources); this test locks the *contract*.
    """
    NUM_STEPS = 2
    FRAMESKIP = 1

    adapter = LeRobotAdapter(
        repo_id=PUSHT_REPO_ID,
        num_steps=NUM_STEPS,
        frameskip=FRAMESKIP,
        keys_to_cache=['action'],
        video_backend='pyav',
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
