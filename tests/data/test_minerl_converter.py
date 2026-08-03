"""Unit tests for MineRL conversion without a Minecraft installation."""

from __future__ import annotations

import numpy as np
import pytest

from stable_worldmodel.data.converters.minerl import (
    ActionVectorizer,
    convert_minerl,
    trajectory_to_episode,
)


def _transition(value: int, *, done: bool = False):
    state = {
        'pov': np.full((4, 5, 3), value, dtype=np.uint8),
    }
    action = {
        'camera': np.array([value, -value], dtype=np.float32),
        'forward': value % 2,
        'jump': bool(value % 2),
    }
    return state, action, float(value), state, done


def test_trajectory_to_episode_preserves_transition_alignment():
    episode = trajectory_to_episode(
        [_transition(1), _transition(2, done=True)],
        episode_id=7,
        action_vectorizer=ActionVectorizer(),
    )

    assert set(episode) == {
        'observation', 'action', 'reward', 'done', 'episode_id', 'timestep'
    }
    assert episode['observation'][0].shape == (4, 5, 3)
    np.testing.assert_allclose(episode['action'][0], [1, -1, 1, 1])
    np.testing.assert_allclose(episode['action'][1], [2, -2, 0, 0])
    assert episode['reward'] == [np.float32(1), np.float32(2)]
    assert episode['done'] == [False, True]
    assert episode['episode_id'] == [7, 7]
    assert episode['timestep'] == [0, 1]


def test_action_vectorizer_rejects_schema_drift():
    vectorizer = ActionVectorizer()
    vectorizer({'forward': 1})
    with pytest.raises(ValueError, match='schema changed'):
        vectorizer({'camera': [0.0, 1.0], 'forward': 1})


class _Pipeline:
    def get_trajectory_names(self):
        return ['first', 'second']

    def load_data(self, name):
        return [_transition(1, done=name == 'second')]


class _Writer:
    instances = []

    def __init__(self, path, *, mode):
        self.path, self.mode, self.episodes = path, mode, []
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write_episode(self, episode):
        self.episodes.append(episode)


def test_convert_minerl_streams_trajectories_to_lance_writer(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        'stable_worldmodel.data.converters.minerl.LanceWriter', _Writer
    )
    summary = convert_minerl(
        tmp_path / 'minerl.lance',
        environment='MineRLBasaltFindCave-v0',
        pipeline=_Pipeline(),
    )

    writer = _Writer.instances[-1]
    assert writer.mode == 'error'
    assert len(writer.episodes) == 2
    assert [ep['episode_id'][0] for ep in writer.episodes] == [0, 1]
    assert summary.trajectories == 2
    assert summary.transitions == 2
    assert summary.action_dim == 4
