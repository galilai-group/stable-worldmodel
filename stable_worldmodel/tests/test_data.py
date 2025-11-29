import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import PIL.Image
import pytest
import torch
from datasets import Dataset

from stable_worldmodel.data import FrameDataset
from stable_worldmodel.data.utils import (
    dataset_info,
    delete_dataset,
    delete_model,
    get_cache_dir,
    is_image,
    list_datasets,
    list_models,
    list_worlds,
    world_info,
)


###########################
## is_image tests        ##
###########################


def test_is_image_valid_rgb():
    """Test is_image with valid RGB image."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    assert is_image(img)


def test_is_image_valid_grayscale():
    """Test is_image with valid grayscale image."""
    img = np.zeros((64, 64, 1), dtype=np.uint8)
    assert is_image(img)


def test_is_image_valid_rgba():
    """Test is_image with valid RGBA image."""
    img = np.zeros((64, 64, 4), dtype=np.uint8)
    assert is_image(img)


def test_is_image_wrong_dtype():
    """Test is_image with wrong dtype."""
    img = np.zeros((64, 64, 3), dtype=np.float32)
    assert not is_image(img)


def test_is_image_wrong_dimensions():
    """Test is_image with wrong number of dimensions."""
    img = np.zeros((64, 64), dtype=np.uint8)
    assert not is_image(img)


def test_is_image_wrong_channels():
    """Test is_image with wrong number of channels."""
    img = np.zeros((64, 64, 5), dtype=np.uint8)
    assert not is_image(img)


def test_is_image_not_array():
    """Test is_image with non-array input."""
    assert not is_image("not an array")
    assert not is_image([1, 2, 3])
    assert not is_image(None)


def test_is_image_four_dimensions():
    """Test is_image with 4D array (batch of images)."""
    img = np.zeros((10, 64, 64, 3), dtype=np.uint8)
    assert not is_image(img)


###########################
## get_cache_dir tests   ##
###########################


def test_get_cache_dir_default():
    """Test get_cache_dir returns default directory."""
    with patch.dict(os.environ, {}, clear=True):
        cache_dir = get_cache_dir()
        assert cache_dir == Path(os.path.expanduser("~/.stable_worldmodel"))


def test_get_cache_dir_custom_env():
    """Test get_cache_dir with custom environment variable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_path = str(Path(tmpdir) / "custom_cache")
        with patch.dict(os.environ, {"STABLEWM_HOME": custom_path}):
            cache_dir = get_cache_dir()
            assert cache_dir == Path(custom_path)


def test_get_cache_dir_creates_directory():
    """Test get_cache_dir creates directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test_cache"
        with patch.dict(os.environ, {"STABLEWM_HOME": str(test_path)}):
            cache_dir = get_cache_dir()
            assert cache_dir.exists()
            assert cache_dir.is_dir()


###########################
## list_datasets tests   ##
###########################


def test_list_datasets_empty():
    """Test list_datasets with empty cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            datasets = list_datasets()
            assert datasets == []


def test_list_datasets_with_datasets():
    """Test list_datasets with multiple datasets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test dataset directories
        (Path(tmpdir) / "dataset1").mkdir()
        (Path(tmpdir) / "dataset2").mkdir()
        (Path(tmpdir) / "dataset3").mkdir()
        # Create a file (should be ignored)
        (Path(tmpdir) / "not_a_dataset.txt").touch()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            datasets = list_datasets()
            assert set(datasets) == {"dataset1", "dataset2", "dataset3"}


def test_list_datasets_only_directories():
    """Test list_datasets only returns directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "dataset1").mkdir()
        (Path(tmpdir) / "file.txt").touch()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            datasets = list_datasets()
            assert datasets == ["dataset1"]


###########################
## list_models tests     ##
###########################


def test_list_models_empty():
    """Test list_models with empty cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            models = list_models()
            assert models == []


def test_list_models_with_weights():
    """Test list_models with weight checkpoint files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test checkpoint files
        (Path(tmpdir) / "model1_weights.ckpt").touch()
        (Path(tmpdir) / "model2_weights-epoch10.ckpt").touch()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            models = list_models()
            assert set(models) == {"model1", "model2"}


def test_list_models_with_objects():
    """Test list_models with object checkpoint files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "model1_object.ckpt").touch()
        (Path(tmpdir) / "model2_object.ckpt").touch()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            models = list_models()
            assert set(models) == {"model1", "model2"}


def test_list_models_mixed():
    """Test list_models with mixed checkpoint types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "model1_weights.ckpt").touch()
        (Path(tmpdir) / "model2_object.ckpt").touch()
        (Path(tmpdir) / "model3_weights-best.ckpt").touch()
        (Path(tmpdir) / "not_a_model.txt").touch()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            models = list_models()
            assert set(models) == {"model1", "model2", "model3"}


def test_list_models_sorted():
    """Test list_models returns sorted list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "zebra_weights.ckpt").touch()
        (Path(tmpdir) / "alpha_object.ckpt").touch()
        (Path(tmpdir) / "beta_weights.ckpt").touch()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            models = list_models()
            assert models == ["alpha", "beta", "zebra"]


def test_list_models_case_insensitive_pattern():
    """Test list_models pattern matching is case insensitive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "Model1_WEIGHTS.CKPT").touch()
        (Path(tmpdir) / "Model2_Object.ckpt").touch()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            models = list_models()
            assert set(models) == {"Model1", "Model2"}


###########################
## list_worlds tests     ##
###########################


def test_list_worlds_returns_sorted_list():
    """Test list_worlds returns sorted list of worlds."""
    worlds = list_worlds()
    assert isinstance(worlds, list)
    assert len(worlds) > 0
    assert worlds == sorted(worlds)


def test_list_worlds_contains_expected_worlds():
    """Test list_worlds contains worlds from swm.envs.WORLDS."""
    from stable_worldmodel import envs

    worlds = list_worlds()
    for world in worlds:
        assert world in envs.WORLDS


###########################
## delete_dataset tests  ##
###########################


def test_delete_dataset_success(monkeypatch):
    """Test delete_dataset successfully deletes a dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test dataset structure
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path.mkdir()
        records_path = dataset_path / "records"
        records_path.mkdir()

        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.cleanup_cache_files = MagicMock()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            with patch("stable_worldmodel.data.utils.load_from_disk", return_value=mock_dataset):
                delete_dataset("test_dataset")

                # Verify dataset was cleaned up
                mock_dataset.cleanup_cache_files.assert_called_once()
                # Verify directory was deleted
                assert not dataset_path.exists()


def test_delete_dataset_not_exists():
    """Test delete_dataset prints error when dataset doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            with patch("stable_worldmodel.data.utils.print") as mock_print:
                delete_dataset("nonexistent_dataset")
                # Should print error message
                assert any("Error" in str(call) for call in mock_print.call_args_list)


def test_delete_dataset_error_handling(monkeypatch):
    """Test delete_dataset handles errors gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path.mkdir()
        records_path = dataset_path / "records"
        records_path.mkdir()

        mock_dataset = MagicMock()
        mock_dataset.cleanup_cache_files = MagicMock(side_effect=Exception("Test error"))

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            with patch("stable_worldmodel.data.utils.load_from_disk", return_value=mock_dataset):
                with patch("stable_worldmodel.data.utils.print") as mock_print:
                    delete_dataset("test_dataset")
                    # Should print error message
                    assert any("Error" in str(call) for call in mock_print.call_args_list)


###########################
## delete_model tests    ##
###########################


def test_delete_model_success():
    """Test delete_model successfully deletes model files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test model files
        (Path(tmpdir) / "model1_weights.ckpt").touch()
        (Path(tmpdir) / "model1_object.ckpt").touch()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            with patch("stable_worldmodel.data.utils.print") as mock_print:
                delete_model("model1")

                # Verify files were deleted
                assert not (Path(tmpdir) / "model1_weights.ckpt").exists()
                assert not (Path(tmpdir) / "model1_object.ckpt").exists()
                # Verify success message was printed
                assert mock_print.call_count == 2


def test_delete_model_partial_match():
    """Test delete_model only deletes matching files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "model1_weights.ckpt").touch()
        (Path(tmpdir) / "model2_weights.ckpt").touch()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            delete_model("model1")

            # Only model1 should be deleted
            assert not (Path(tmpdir) / "model1_weights.ckpt").exists()
            assert (Path(tmpdir) / "model2_weights.ckpt").exists()


def test_delete_model_error_handling():
    """Test delete_model handles errors gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = Path(tmpdir) / "model1_weights.ckpt"
        model_file.touch()

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            with patch("os.remove", side_effect=Exception("Permission denied")):
                with patch("stable_worldmodel.data.utils.print") as mock_print:
                    delete_model("model1")
                    # Should print error message
                    assert any("Error" in str(call) for call in mock_print.call_args_list)


def test_delete_model_no_files():
    """Test delete_model when no matching files exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            with patch("stable_worldmodel.data.utils.print") as mock_print:
                delete_model("nonexistent_model")
                # No files deleted, so print shouldn't be called
                mock_print.assert_not_called()


###########################
## dataset_info tests    ##
###########################


def test_dataset_info_not_found():
    """Test dataset_info raises error when dataset not found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            with pytest.raises(ValueError, match="not found"):
                dataset_info("nonexistent_dataset")


def test_dataset_info_success():
    """Test dataset_info returns correct information."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset" / "records"
        dataset_path.mkdir(parents=True)

        # Create mock dataset
        mock_data = {
            "episode_idx": [0, 0, 1, 1, 1],
            "step_idx": [0, 1, 0, 1, 2],
            "episode_len": [2, 2, 3, 3, 3],
            "pixels": [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)],
            "action": [np.array([0.0, 1.0]) for _ in range(5)],
            "goal": [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)],
        }
        dataset = Dataset.from_dict(mock_data)
        dataset.save_to_disk(str(dataset_path))

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            info = dataset_info("test_dataset")

            assert info["name"] == "test_dataset"
            assert info["num_episodes"] == 2
            assert info["num_steps"] == 5
            assert "pixels" in info["columns"]
            assert "action" in info["columns"]
            assert info["obs_shape"] == (64, 64, 3)
            assert info["action_shape"] == (2,)


def test_dataset_info_with_variation():
    """Test dataset_info detects variation columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset" / "records"
        dataset_path.mkdir(parents=True)

        mock_data = {
            "episode_idx": [0],
            "step_idx": [0],
            "episode_len": [1],
            "pixels": [np.zeros((64, 64, 3), dtype=np.uint8)],
            "action": [np.array([0.0])],
            "goal": [np.zeros((64, 64, 3), dtype=np.uint8)],
            "variation.color": ["red"],
            "variation.size": ["large"],
        }
        dataset = Dataset.from_dict(mock_data)
        dataset.save_to_disk(str(dataset_path))

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            info = dataset_info("test_dataset")

            assert info["variation"]["has_variation"]
            assert set(info["variation"]["names"]) == {"color", "size"}


def test_dataset_info_missing_required_columns():
    """Test dataset_info with missing required columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset" / "records"
        dataset_path.mkdir(parents=True)

        # Dataset missing 'action' column
        mock_data = {
            "episode_idx": [0],
            "step_idx": [0],
            "episode_len": [1],
            "pixels": [np.zeros((64, 64, 3), dtype=np.uint8)],
            "goal": [np.zeros((64, 64, 3), dtype=np.uint8)],
        }
        dataset = Dataset.from_dict(mock_data)
        dataset.save_to_disk(str(dataset_path))

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            with pytest.raises(AssertionError, match="action"):
                dataset_info("test_dataset")


###########################
## world_info tests      ##
###########################


def test_world_info_invalid_world():
    """Test world_info raises error for invalid world."""
    # Clear the cache before running this test
    world_info.cache_clear()

    with pytest.raises(ValueError, match="not found"):
        world_info("nonexistent_world")


@patch("stable_worldmodel.data.utils.swm.World")
@patch("stable_worldmodel.envs.WORLDS", {"test_world"})
def test_world_info_success(mock_world_class):
    """Test world_info returns correct information."""
    # Clear the cache before running this test
    world_info.cache_clear()

    # Mock the World instance
    mock_world = MagicMock()
    mock_world.single_observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    mock_world.single_action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    mock_world.single_variation_space = None
    mock_world.close = MagicMock()

    mock_world_class.return_value = mock_world

    info = world_info("test_world")

    assert info["name"] == "test_world"
    assert "observation_space" in info
    assert "action_space" in info
    assert info["variation"]["has_variation"] is False


@patch("stable_worldmodel.data.utils.swm.World")
@patch("stable_worldmodel.envs.WORLDS", {"test_world"})
def test_world_info_with_variation(mock_world_class):
    """Test world_info with variation space."""
    # Clear the cache before running this test
    world_info.cache_clear()

    mock_world = MagicMock()
    mock_world.single_observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    mock_world.single_action_space = gym.spaces.Discrete(4)

    # Mock variation space
    mock_variation_space = MagicMock()
    mock_variation_space.names = MagicMock(return_value=["color", "size"])
    mock_world.single_variation_space = mock_variation_space
    mock_world.close = MagicMock()

    mock_world_class.return_value = mock_world

    info = world_info("test_world")

    assert info["variation"]["has_variation"]
    assert info["variation"]["names"] == ["color", "size"]


@patch("stable_worldmodel.data.utils.swm.World")
@patch("stable_worldmodel.envs.WORLDS", {"test_world"})
def test_world_info_closes_world(mock_world_class):
    """Test world_info properly closes the world."""
    # Clear the cache before running this test
    world_info.cache_clear()

    mock_world = MagicMock()
    mock_world.single_observation_space = gym.spaces.Discrete(5)
    mock_world.single_action_space = gym.spaces.Discrete(4)
    mock_world.single_variation_space = None
    mock_world.close = MagicMock()

    mock_world_class.return_value = mock_world

    world_info("test_world")
    mock_world.close.assert_called_once()


@patch("stable_worldmodel.data.utils.swm.World")
@patch("stable_worldmodel.envs.WORLDS", {"test_world"})
def test_world_info_closes_world_on_error(mock_world_class):
    """Test world_info closes world even if error occurs."""
    # Clear the cache before running this test
    world_info.cache_clear()

    mock_world = MagicMock()
    mock_world.single_observation_space = None
    mock_world.single_action_space = None
    mock_world.single_variation_space = None
    mock_world.close = MagicMock()

    mock_world_class.return_value = mock_world

    # Should not raise, and should close world
    world_info("test_world")
    mock_world.close.assert_called_once()


@patch("stable_worldmodel.data.utils.swm.World")
@patch("stable_worldmodel.envs.WORLDS", {"test_world"})
def test_world_info_caching(mock_world_class):
    """Test world_info uses caching for repeated calls."""
    # Clear the cache before running this test
    world_info.cache_clear()

    mock_world = MagicMock()
    mock_world.single_observation_space = gym.spaces.Discrete(5)
    mock_world.single_action_space = gym.spaces.Discrete(4)
    mock_world.single_variation_space = None
    mock_world.close = MagicMock()

    mock_world_class.return_value = mock_world

    # First call
    info1 = world_info("test_world")
    # Second call should use cache
    info2 = world_info("test_world")

    # Should only create world once due to caching
    assert mock_world_class.call_count == 1
    assert info1 == info2


@patch("stable_worldmodel.data.utils.swm.World")
@patch("stable_worldmodel.envs.WORLDS", {"test_world"})
def test_world_info_with_dict_space(mock_world_class):
    """Test world_info with Dict observation space."""
    # Clear the cache before running this test
    world_info.cache_clear()

    mock_world = MagicMock()
    # Create a Dict space with multiple sub-spaces
    mock_world.single_observation_space = gym.spaces.Dict(
        {
            "image": gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "vector": gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
        }
    )
    mock_world.single_action_space = gym.spaces.Discrete(4)
    mock_world.single_variation_space = None
    mock_world.close = MagicMock()

    mock_world_class.return_value = mock_world

    info = world_info("test_world")

    assert info["name"] == "test_world"
    assert "observation_space" in info
    # Check that Dict space was properly handled
    assert isinstance(info["observation_space"], dict)
    assert "image" in info["observation_space"]
    assert "vector" in info["observation_space"]


@patch("stable_worldmodel.data.utils.swm.World")
@patch("stable_worldmodel.envs.WORLDS", {"test_world"})
def test_world_info_with_tuple_space(mock_world_class):
    """Test world_info with Tuple action space."""
    # Clear the cache before running this test
    world_info.cache_clear()

    mock_world = MagicMock()
    mock_world.single_observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    # Create a Tuple space with multiple sub-spaces
    mock_world.single_action_space = gym.spaces.Tuple(
        (
            gym.spaces.Discrete(4),
            gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        )
    )
    mock_world.single_variation_space = None
    mock_world.close = MagicMock()

    mock_world_class.return_value = mock_world

    info = world_info("test_world")

    assert info["name"] == "test_world"
    assert "action_space" in info
    # Check that Tuple space was properly handled as a list
    assert isinstance(info["action_space"], list)
    assert len(info["action_space"]) == 2


###########################
## FrameDataset tests    ##
###########################


def create_shard_dataset(tmpdir, dataset_name, mock_data, img_files=None):
    """Helper to create a dataset with the new shard structure.

    The new dataset structure expects:
    - <cache_dir>/<dataset_name>/shard_0/img/ directory to exist
    - Dataset saved to <cache_dir>/<dataset_name>/shard_0/
    - Image files (if any) in the img/ subdirectory with paths like "img/file.jpeg"
    """
    shard_path = Path(tmpdir) / dataset_name / "shard_0"
    img_path = shard_path / "img"
    img_path.mkdir(parents=True)

    # Create image files if provided
    if img_files:
        for filename, img in img_files.items():
            img.save(img_path / filename)

    dataset = Dataset.from_dict(mock_data)
    dataset.save_to_disk(str(shard_path))

    return shard_path


def test_steps_dataset_initialization():
    """Test FrameDataset initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock dataset with new shard structure
        mock_data = {
            "episode_idx": [0, 0, 0, 1, 1, 1],
            "step_idx": [0, 1, 2, 0, 1, 2],
            "action": [np.array([0.0, 1.0]) for _ in range(6)],
            "pixels": [f"img/img_{i}.png" for i in range(6)],
        }
        create_shard_dataset(tmpdir, "test_dataset", mock_data)

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            steps_dataset = FrameDataset("test_dataset", num_steps=2, frameskip=1)

            assert steps_dataset.num_steps == 2
            assert steps_dataset.frameskip == 1
            assert len(steps_dataset.episodes) == 2


def test_steps_dataset_missing_required_columns():
    """Test FrameDataset raises error for missing required columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Missing 'episode_idx' column
        mock_data = {
            "step_idx": [0, 1, 2],
            "action": [np.array([0.0]) for _ in range(3)],
        }
        create_shard_dataset(tmpdir, "test_dataset", mock_data)

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            # The error now comes from build_dataset_from_shards which accesses episode_idx
            with pytest.raises((ValueError, KeyError), match="episode_idx"):
                FrameDataset("test_dataset", num_steps=2)


def test_steps_dataset_episode_too_short():
    """Test FrameDataset raises error when episode is too short."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Episode with only 1 step, but num_steps=2
        mock_data = {
            "episode_idx": [0],
            "step_idx": [0],
            "action": [np.array([0.0])],
        }
        create_shard_dataset(tmpdir, "test_dataset", mock_data)

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            with pytest.raises(ValueError, match="at least"):
                FrameDataset("test_dataset", num_steps=2, frameskip=1)


def test_steps_dataset_length():
    """Test FrameDataset __len__ returns correct length."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Episode with 5 steps
        mock_data = {
            "episode_idx": [0, 0, 0, 0, 0],
            "step_idx": [0, 1, 2, 3, 4],
            "action": [np.array([0.0]) for _ in range(5)],
        }
        create_shard_dataset(tmpdir, "test_dataset", mock_data)

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            # num_steps=2, so we can have 4 possible slices (0-1, 1-2, 2-3, 3-4)
            steps_dataset = FrameDataset("test_dataset", num_steps=2, frameskip=1)
            assert len(steps_dataset) == 4


def test_steps_dataset_getitem():
    """Test FrameDataset __getitem__ returns correct data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create actual image files in the img subdirectory
        img_files = {}
        for i in range(5):
            img = PIL.Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
            img_files[f"img_{i}.jpeg"] = img

        # Episode with 5 steps - paths include img/ prefix
        mock_data = {
            "episode_idx": [0, 0, 0, 0, 0],
            "step_idx": [0, 1, 2, 3, 4],
            "action": [np.array([float(i), float(i)]) for i in range(5)],
            "pixels.jpeg": [f"img/img_{i}.jpeg" for i in range(5)],
        }
        create_shard_dataset(tmpdir, "test_dataset", mock_data, img_files=img_files)

        # Define a transform that converts PIL images to tensors
        def image_to_tensor(batch):
            for key in ["pixels.jpeg"]:
                if key in batch and isinstance(batch[key][0], PIL.Image.Image):
                    batch[key] = [
                        torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 for img in batch[key]
                    ]
            return batch

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            steps_dataset = FrameDataset("test_dataset", num_steps=2, frameskip=1, transform=image_to_tensor)

            # Get first slice (steps 0-1)
            sample = steps_dataset[0]

            assert "action" in sample
            assert "pixels.jpeg" in sample
            assert isinstance(sample["action"], torch.Tensor)
            assert sample["action"].shape == (2, 2)  # num_steps=2, action_dim=2


def test_steps_dataset_frameskip():
    """Test FrameDataset with frameskip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create image files
        img_files = {}
        for i in range(10):
            img = PIL.Image.new("RGB", (32, 32), color=(i * 20, i * 20, i * 20))
            img_files[f"img_{i}.jpeg"] = img

        # Episode with 10 steps
        # Column name must end with .jpeg for determine_img_columns to detect it
        mock_data = {
            "episode_idx": [0] * 10,
            "step_idx": list(range(10)),
            "action": [np.array([float(i)]) for i in range(10)],
            "pixels.jpeg": [f"img/img_{i}.jpeg" for i in range(10)],
        }
        create_shard_dataset(tmpdir, "test_dataset", mock_data, img_files=img_files)

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            # num_steps=2, frameskip=2 means we need 4 steps total (0, 2 for observations)
            steps_dataset = FrameDataset("test_dataset", num_steps=2, frameskip=2)

            sample = steps_dataset[0]

            # Should have 2 frames (with frameskip=2)
            # pixels.jpeg is a torch.Tensor after stacking
            assert sample["pixels.jpeg"].shape[0] == 2


def test_steps_dataset_multiple_episodes():
    """Test FrameDataset with multiple episodes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create image files
        img_files = {}
        for i in range(8):
            img = PIL.Image.new("RGB", (32, 32))
            img_files[f"img_{i}.jpeg"] = img

        # Two episodes with 4 steps each
        mock_data = {
            "episode_idx": [0, 0, 0, 0, 1, 1, 1, 1],
            "step_idx": [0, 1, 2, 3, 0, 1, 2, 3],
            "action": [np.array([0.0]) for _ in range(8)],
            "pixels.jpeg": [f"img/img_{i}.jpeg" for i in range(8)],
        }
        create_shard_dataset(tmpdir, "test_dataset", mock_data, img_files=img_files)

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            steps_dataset = FrameDataset("test_dataset", num_steps=2, frameskip=1)

            # Each episode has 4 steps, so 3 valid slices per episode = 6 total
            assert len(steps_dataset) == 6


def test_steps_dataset_infer_img_path_columns():
    """Test FrameDataset infer_img_path_columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create image files
        img = PIL.Image.new("RGB", (32, 32))
        img_files = {
            "img_0.jpeg": img,
            "goal_0.jpeg": img,
        }

        # The code's determine_img_columns checks if column VALUE ends with .jpeg
        # So we need paths that end with .jpeg for auto-detection
        mock_data = {
            "episode_idx": [0, 0, 0],
            "step_idx": [0, 1, 2],
            "action": [np.array([0.0]) for _ in range(3)],
            "pixels.jpeg": ["img/img_0.jpeg", "img/img_0.jpeg", "img/img_0.jpeg"],
            "goal.jpeg": ["img/goal_0.jpeg", "img/goal_0.jpeg", "img/goal_0.jpeg"],
            "other_data": [1, 2, 3],
        }
        create_shard_dataset(tmpdir, "test_dataset", mock_data, img_files=img_files)

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            steps_dataset = FrameDataset("test_dataset", num_steps=2, frameskip=1)

            # decode_columns is the attribute that holds detected image columns
            assert "pixels.jpeg" in steps_dataset.decode_columns
            assert "goal.jpeg" in steps_dataset.decode_columns
            assert "other_data" not in steps_dataset.decode_columns


def test_steps_dataset_with_transform():
    """Test FrameDataset with transform function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create image files
        img_files = {}
        for i in range(3):
            img = PIL.Image.new("RGB", (32, 32))
            img_files[f"img_{i}.jpeg"] = img

        mock_data = {
            "episode_idx": [0, 0, 0],
            "step_idx": [0, 1, 2],
            "action": [np.array([0.0]) for _ in range(3)],
            "pixels.jpeg": [f"img/img_{i}.jpeg" for i in range(3)],
        }
        create_shard_dataset(tmpdir, "test_dataset", mock_data, img_files=img_files)

        # Define a simple transform
        def mock_transform(batch):
            batch["transformed"] = True
            # Also convert images to tensors to avoid stacking error
            for key in ["pixels.jpeg"]:
                if key in batch and isinstance(batch[key][0], PIL.Image.Image):
                    batch[key] = [
                        torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 for img in batch[key]
                    ]
            return batch

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            steps_dataset = FrameDataset("test_dataset", num_steps=2, frameskip=1, transform=mock_transform)

            sample = steps_dataset[0]
            assert "transformed" in sample
            assert sample["transformed"] is True


def test_steps_dataset_custom_cache_dir():
    """Test FrameDataset with custom cache_dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_cache = Path(tmpdir) / "custom_cache"

        mock_data = {
            "episode_idx": [0, 0, 0],
            "step_idx": [0, 1, 2],
            "action": [np.array([0.0]) for _ in range(3)],
        }
        shard_path = create_shard_dataset(str(custom_cache), "test_dataset", mock_data)

        steps_dataset = FrameDataset("test_dataset", num_steps=2, frameskip=1, cache_dir=str(custom_cache))

        # The shard_dirs should contain our shard
        assert len(steps_dataset.shard_dirs) == 1
        assert str(shard_path) in steps_dataset.shard_dirs[0]


def test_steps_dataset_action_reshape():
    """Test FrameDataset correctly reshapes actions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Actions with shape (2,) that span multiple steps
        # With num_steps=2 and frameskip=1, we need actions for both observation steps
        # The dataset stores actions at full resolution
        mock_data = {
            "episode_idx": [0, 0, 0],
            "step_idx": [0, 1, 2],
            "action": [np.array([float(i), float(i) * 2]) for i in range(3)],
        }
        create_shard_dataset(tmpdir, "test_dataset", mock_data)

        with patch.dict(os.environ, {"STABLEWM_HOME": tmpdir}):
            steps_dataset = FrameDataset("test_dataset", num_steps=2, frameskip=1)

            sample = steps_dataset[0]

            # Action should be reshaped to (num_steps, action_dim)
            assert sample["action"].shape == (2, 2)
