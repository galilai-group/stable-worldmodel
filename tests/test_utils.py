import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from stable_worldmodel.utils import (
    flatten_dict,
    get_in,
    pretraining,
    record_video_from_dataset,
)


#######################
## pretraining tests ##
#######################


def test_raises_when_script_not_exists(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda p: False)

    with pytest.raises(ValueError, match=r"does not exist"):
        pretraining("non_existent_script.py", "test_dataset", "test_model")


def test_pretraining_success(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda p: True)
    mock_run = MagicMock(return_value=MagicMock(returncode=0))
    monkeypatch.setattr("subprocess.run", mock_run)

    pretraining("fake_script.py", "test_dataset", "test_model", args="--epochs 10")

    assert mock_run.called


def test_pretraining_exits_on_failure(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda p: True)
    mock_run = MagicMock(side_effect=subprocess.CalledProcessError(1, "cmd"))
    monkeypatch.setattr("subprocess.run", mock_run)

    with pytest.raises(SystemExit, match="1"):
        pretraining("fake_script.py", "test_dataset", "test_model", args="--epochs 10")


def test_pretraining_parses_single_arg(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda p: True)
    mock_run = MagicMock()
    monkeypatch.setattr("subprocess.run", mock_run)

    pretraining("fake_script.py", "test_dataset", "test_model", args="batch-size=32")

    cmd = mock_run.call_args[0][0]
    assert "fake_script.py" in cmd
    assert "batch-size=32" in cmd
    assert "dataset_name=test_dataset" in cmd
    assert "output_model_name=test_model" in cmd
    assert mock_run.call_args[1]["check"] is True


def test_pretraining_parses_multiple_args(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda p: True)
    mock_run = MagicMock()
    monkeypatch.setattr("subprocess.run", mock_run)

    pretraining("fake_script.py", "test_dataset", "test_model", dump_object=True, args="batch-size=32")

    cmd = mock_run.call_args[0][0]
    assert "fake_script.py" in cmd
    assert "batch-size=32" in cmd
    assert "++dump_object=True" in cmd
    assert "dataset_name=test_dataset" in cmd
    assert "output_model_name=test_model" in cmd


def test_pretraining_with_empty_args(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda p: True)
    mock_run = MagicMock()
    monkeypatch.setattr("subprocess.run", mock_run)

    pretraining("fake_script.py", "test_dataset", "test_model")

    cmd = mock_run.call_args[0][0]
    assert "fake_script.py" in cmd
    assert "dataset_name=test_dataset" in cmd
    assert "output_model_name=test_model" in cmd
    assert "++dump_object=True" in cmd


########################
## flatten_dict tests ##
########################


def test_flatten_dict_empty_dict():
    flatten_dict({}) == {}


def test_flatten_dict_single_level():
    assert flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_flatten_dict_nested_dict():
    assert flatten_dict({"a": {"b": 2}}) == {"a.b": 2}


def test_flatten_dict_information_loss():
    assert flatten_dict({"a": {"b": 2}, "a.b": 3}) == {"a.b": 3}


def test_flatten_dict_multiple_nested_levels():
    assert flatten_dict({"a": {"b": {"c": 3}}}) == {"a.b.c": 3}


def test_flatten_dict_other_separators():
    assert flatten_dict({"a": {"b": 2}}, sep="_") == {"a_b": 2}


def test_flatten_dict_parent_key():
    assert flatten_dict({"a": {"b": 2}}, parent_key="root") == {"root.a.b": 2}


def test_flatten_dict_mixed_types():
    assert flatten_dict({"a": {1: "string", (4, "5"): 2}}) == {
        "a.1": "string",
        "a.(4, '5')": 2,
    }


def test_flatten_dict_same_flatten():
    assert flatten_dict({"a": {"b": {"c": 3}}, "d": 4}) == flatten_dict({"a": {"b.c": 3}, "d": 4})


#################
## get_in test ##
#################


def test_get_in_existing_key_depth_one():
    assert get_in({"a": 2}, ["a"]) == 2


def test_get_in_empty_dict():
    with pytest.raises(KeyError):
        get_in({}, ["a"])


def test_get_in_missing_key_depth_one():
    with pytest.raises(KeyError):
        get_in({"a": 1}, ["b"])


def test_get_in_empty_path():
    assert get_in({"a": 1}, []) == {"a": 1}


def test_get_in_existing_key_depth_two():
    assert get_in({"a": {"b": 3}}, ["a", "b"]) == 3


def test_get_in_missing_key_depth_two():
    with pytest.raises(KeyError):
        get_in({"a": {"b": 3}}, ["a", "c"])


def test_get_in_missing_intermediate_key_depth_two():
    with pytest.raises(KeyError):
        get_in({"a": {"b": 3}}, ["x", "b"])


def test_get_in_empty_key_depth_two():
    assert get_in({"a": {"b": 3}}, ["a"]) == {"b": 3}


###################################
## record_video_from_dataset tests ##
###################################


class MockVideoDataset:
    """Mock dataset for testing record_video_from_dataset."""

    def __init__(self, num_steps=10, height=64, width=64):
        self.num_steps = num_steps
        self.height = height
        self.width = width
        self._column_names = ["pixels", "pixels_alt", "action"]

    @property
    def column_names(self):
        return self._column_names

    def load_episode(self, ep_idx):
        return {
            "pixels": torch.randint(0, 255, (self.num_steps, 3, self.height, self.width), dtype=torch.uint8),
            "pixels_alt": torch.randint(0, 255, (self.num_steps, 3, self.height, self.width), dtype=torch.uint8),
            "action": torch.randn(self.num_steps, 2),
        }


def test_record_video_single_episode(tmp_path):
    """Test recording video from a single episode."""
    dataset = MockVideoDataset(num_steps=10)

    record_video_from_dataset(tmp_path, dataset, episode_idx=0, fps=10)

    video_file = tmp_path / "episode_0.mp4"
    assert video_file.exists()


def test_record_video_multiple_episodes(tmp_path):
    """Test recording videos from multiple episodes."""
    dataset = MockVideoDataset(num_steps=10)

    record_video_from_dataset(tmp_path, dataset, episode_idx=[0, 1, 2], fps=10)

    for i in range(3):
        assert (tmp_path / f"episode_{i}.mp4").exists()


def test_record_video_max_steps(tmp_path):
    """Test that max_steps limits the video length."""
    dataset = MockVideoDataset(num_steps=100)

    record_video_from_dataset(tmp_path, dataset, episode_idx=0, max_steps=10, fps=10)

    assert (tmp_path / "episode_0.mp4").exists()


def test_record_video_multiple_views(tmp_path):
    """Test recording video with multiple views stacked."""
    dataset = MockVideoDataset(num_steps=10)

    record_video_from_dataset(tmp_path, dataset, episode_idx=0, viewname=["pixels", "pixels_alt"], fps=10)

    assert (tmp_path / "episode_0.mp4").exists()


def test_record_video_invalid_view_raises(tmp_path):
    """Test that invalid view name raises assertion error."""
    dataset = MockVideoDataset(num_steps=10)

    with pytest.raises(AssertionError, match="not in dataset key names"):
        record_video_from_dataset(tmp_path, dataset, episode_idx=0, viewname="nonexistent")
