"""Tests for the LeRobot dataset adapter.

Each test reads a tiny LeRobot v3 dataset that the module writes to a temp
folder with ``LeRobotDataset.create``. Nothing is downloaded: Hub access is
switched off and the ``datasets`` cache lives in the temp folder too.

Frames are flat grey images whose level encodes ``(episode, step)``, and
``action`` / ``observation.state`` hold ``(episode, step)`` as numbers. So the
tests can check that the adapter returns the right frames and rows, not only
the right shapes.

The comparison against the real ``lerobot/pusht`` Hub dataset lives in
``test_lerobot_hub.py`` and is opt-in.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

if sys.version_info < (3, 12):
    pytest.skip('lerobot requires Python 3.12+', allow_module_level=True)

pytest.importorskip('lerobot')

from stable_worldmodel.data import GoalDataset, LeRobotAdapter  # noqa: E402

REPO_ID = 'swm-tests/tiny'
CAMERA_KEY = 'observation.image'
EP_LENGTHS = (5, 7, 4)
FPS = 10
HW = 16  # h264 with yuv420p needs even frame sizes.
ACTION_DIM = 2

# torchcodec needs FFmpeg shared libraries that CI runners may not have.
# PyAV ships its own FFmpeg, so decode with it.
VIDEO_BACKEND = 'pyav'

# Flat grey frames go through h264 and the YUV <-> RGB conversion almost
# unchanged (a couple of levels at most). Neighbouring steps differ by 10
# levels, so this tolerance still tells frames apart.
PIXEL_ATOL = 4 / 255


def _grey_level(ep: int, step: int) -> int:
    return 10 + 80 * ep + 10 * step


def _action(ep: int, step: int) -> list[float]:
    return [float(ep), step + 0.5]


def _state(ep: int, step: int) -> list[float]:
    return [float(ep), float(step)]


@pytest.fixture(scope='module', autouse=True)
def _offline_hf(tmp_path_factory):
    """Keep every Hugging Face read offline and inside a temp folder."""
    import datasets
    import huggingface_hub.constants

    cache = tmp_path_factory.mktemp('hf_datasets_cache')
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('HF_HUB_OFFLINE', '1')
        mp.setenv('HF_DATASETS_CACHE', str(cache))
        # Both libraries read these variables once, at import time.
        mp.setattr(huggingface_hub.constants, 'HF_HUB_OFFLINE', True)
        mp.setattr(datasets.config, 'HF_HUB_OFFLINE', True)
        mp.setattr(datasets.config, 'HF_DATASETS_CACHE', cache)
        yield


@pytest.fixture(scope='module')
def tiny_root(tmp_path_factory):
    """Write the tiny dataset (one video camera, 3 episodes) to disk."""
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path_factory.mktemp('lerobot') / 'ds'
    features = {
        CAMERA_KEY: {
            'dtype': 'video',
            'shape': (HW, HW, 3),
            'names': ['height', 'width', 'channels'],
        },
        'observation.state': {
            'dtype': 'float32',
            'shape': (2,),
            'names': ['ep', 'step'],
        },
        'action': {
            'dtype': 'float32',
            'shape': (ACTION_DIM,),
            'names': ['ep', 'step'],
        },
    }
    writer = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=features,
        root=root,
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec='h264'),
    )
    for ep, length in enumerate(EP_LENGTHS):
        for step in range(length):
            writer.add_frame(
                {
                    CAMERA_KEY: np.full(
                        (HW, HW, 3), _grey_level(ep, step), dtype=np.uint8
                    ),
                    'observation.state': np.array(
                        _state(ep, step), dtype=np.float32
                    ),
                    'action': np.array(_action(ep, step), dtype=np.float32),
                    'task': 'tiny',
                }
            )
        writer.save_episode(parallel_encoding=False)
    writer.finalize()
    return root


def _open(tiny_root, **kwargs) -> LeRobotAdapter:
    kwargs.setdefault('video_backend', VIDEO_BACKEND)
    return LeRobotAdapter(repo_id=REPO_ID, root=tiny_root, **kwargs)


@pytest.fixture(scope='module')
def tiny(tiny_root) -> LeRobotAdapter:
    return _open(tiny_root)


def _assert_frames(pixels: torch.Tensor, ep: int, steps: list[int]) -> None:
    assert pixels.shape == (len(steps), 3, HW, HW)
    assert pixels.dtype == torch.float32
    expected = torch.tensor(
        [_grey_level(ep, s) / 255 for s in steps], dtype=torch.float32
    )
    torch.testing.assert_close(
        pixels.mean(dim=(1, 2, 3)), expected, atol=PIXEL_ATOL, rtol=0
    )


def _expected_actions(ep: int, steps: list[int]) -> torch.Tensor:
    return torch.tensor([_action(ep, s) for s in steps], dtype=torch.float32)


def test_default_aliases_and_episode_structure(tiny):
    assert {'pixels', 'action', 'proprio', 'ep_idx', 'step_idx'}.issubset(
        tiny.column_names
    )
    assert tiny.lengths.tolist() == list(EP_LENGTHS)
    assert tiny.offsets.tolist() == [0, 5, 12]
    assert len(tiny) == sum(EP_LENGTHS)

    ep_idx = tiny.get_col_data('ep_idx')
    step_idx = tiny.get_col_data('step_idx')
    assert ep_idx.tolist() == [
        ep for ep, length in enumerate(EP_LENGTHS) for _ in range(length)
    ]
    assert step_idx.tolist() == [
        step for length in EP_LENGTHS for step in range(length)
    ]


def test_item_and_chunk_behavior(tiny_root):
    dataset = _open(tiny_root, num_steps=2, keys_to_cache=['action'])

    item = dataset[0]
    _assert_frames(item['pixels'], ep=0, steps=[0, 1])
    torch.testing.assert_close(item['action'], _expected_actions(0, [0, 1]))
    assert item['step_idx'].tolist() == [0, 1]
    assert item['ep_idx'].tolist() == [0, 0]

    chunk = dataset.load_chunk(np.array([1]), np.array([2]), np.array([5]))
    assert len(chunk) == 1
    _assert_frames(chunk[0]['pixels'], ep=1, steps=[2, 3, 4])
    torch.testing.assert_close(
        chunk[0]['action'], _expected_actions(1, [2, 3, 4])
    )
    # A slice that starts mid-episode keeps its step and episode numbers.
    assert chunk[0]['step_idx'].tolist() == [2, 3, 4]
    assert chunk[0]['ep_idx'].tolist() == [1, 1, 1]


def test_frameskip_strides_observations_and_keeps_every_action(tiny_root):
    dataset = _open(tiny_root, num_steps=2, frameskip=2)

    item = dataset[dataset.clip_indices.index((1, 0))]
    _assert_frames(item['pixels'], ep=1, steps=[0, 2])
    assert item['step_idx'].tolist() == [0, 2]
    torch.testing.assert_close(
        item['proprio'],
        torch.tensor([_state(1, 0), _state(1, 2)], dtype=torch.float32),
    )
    # Every action in the span is kept and grouped per observation step.
    assert item['action'].shape == (2, 2 * ACTION_DIM)
    torch.testing.assert_close(
        item['action'].reshape(-1, ACTION_DIM),
        _expected_actions(1, [0, 1, 2, 3]),
    )


def test_load_episode_returns_the_whole_episode(tiny):
    episode = tiny.load_episode(2)
    _assert_frames(episode['pixels'], ep=2, steps=[0, 1, 2, 3])
    torch.testing.assert_close(
        episode['action'], _expected_actions(2, [0, 1, 2, 3])
    )
    assert episode['step_idx'].tolist() == [0, 1, 2, 3]


def test_subset_localizes_episode_indices(tiny_root):
    subset = _open(tiny_root, episodes=[1])

    assert subset.lengths.tolist() == [EP_LENGTHS[1]]
    assert subset.offsets.tolist() == [0]
    assert set(subset.get_col_data('ep_idx').tolist()) == {0}
    assert subset.get_col_data('step_idx').tolist() == list(
        range(EP_LENGTHS[1])
    )
    # The rows still come from absolute episode 1.
    assert set(subset.get_col_data('action')[:, 0].tolist()) == {1.0}
    _assert_frames(
        subset.load_episode(0)['pixels'],
        ep=1,
        steps=list(range(EP_LENGTHS[1])),
    )


def test_get_row_data_and_video_column_error(tiny):
    row = tiny.get_row_data([0, 5])
    assert row['ep_idx'].tolist() == [0, 1]
    assert row['step_idx'].tolist() == [0, 0]
    np.testing.assert_array_equal(
        row['action'], np.array([_action(0, 0), _action(1, 0)], np.float32)
    )

    with pytest.raises(KeyError):
        tiny.get_col_data('pixels')


def test_goal_dataset_compatibility(tiny_root):
    dataset = _open(tiny_root, num_steps=2, keys_to_cache=['action'])
    goal_dataset = GoalDataset(
        dataset,
        goal_probabilities=(0.0, 0.0, 0.0, 1.0),
        current_goal_offset=2,
        goal_keys={'pixels': 'goal_pixels', 'action': 'goal_action'},
        seed=123,
    )
    item = goal_dataset[0]
    # The goal is one frame, (C, H, W): here step 1 of episode 0.
    assert item['goal_pixels'].shape == (3, HW, HW)
    _assert_frames(item['goal_pixels'][None], ep=0, steps=[1])
    # The goal action is one step: (1, action_dim).
    torch.testing.assert_close(item['goal_action'], _expected_actions(0, [1]))


def test_missing_lerobot_dependency_names_the_lerobot_extra(monkeypatch):
    # A None entry in sys.modules makes the import fail, as a missing
    # dependency would.
    monkeypatch.setitem(sys.modules, 'lerobot.datasets.lerobot_dataset', None)
    with pytest.raises(ImportError) as exc_info:
        LeRobotAdapter(repo_id=REPO_ID)
    message = str(exc_info.value)
    assert 'stable-worldmodel[lerobot]' in message
    assert 'lerobot.datasets.lerobot_dataset' in message
