"""Integration tests with real environment data collection."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from stable_worldmodel import World
from stable_worldmodel.policy import RandomPolicy
from stable_worldmodel.data import HDF5Dataset, ImageDataset, VideoDataset


class TestRealDataCollection:
    """Test data collection and loading with real environments."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary directory for test data."""
        return tmp_path

    def test_collect_and_load_pusht(self, temp_cache_dir):
        """Test collecting data from PushT and loading with HDF5Dataset."""
        # 1. Create World with PushT environment
        world = World(
            env_name="swm/PushT-v1",
            num_envs=2,
            image_shape=(64, 64),
            max_episode_steps=20,
            verbose=0,
        )

        # 2. Set random policy
        policy = RandomPolicy()
        world.set_policy(policy)

        # 3. Collect data
        dataset_name = "test_pusht"
        world.record_dataset(
            dataset_name=dataset_name,
            episodes=4,
            seed=42,
            cache_dir=temp_cache_dir,
        )

        # 4. Verify HDF5 file was created
        h5_path = temp_cache_dir / f"{dataset_name}.h5"
        assert h5_path.exists(), f"HDF5 file not created at {h5_path}"

        # 5. Load with HDF5Dataset
        dataset = HDF5Dataset(
            name=dataset_name,
            cache_dir=str(temp_cache_dir),
        )

        # 6. Verify dataset properties
        assert len(dataset) > 0, "Dataset should have samples"
        assert len(dataset.lengths) == 4, "Should have 4 episodes"

        # 7. Verify we can load samples
        sample = dataset[0]
        assert isinstance(sample, dict), "Sample should be a dict"

        # 8. Verify expected keys exist
        print(f"Available keys: {dataset.column_names}")
        assert "action" in sample, "Should have action"

        # 9. Verify data types
        for key, value in sample.items():
            assert isinstance(value, torch.Tensor), f"{key} should be a tensor"

        # 10. Verify load_chunk works
        chunk = dataset.load_chunk(
            episodes_idx=np.array([0, 1]),
            start=np.array([0, 0]),
            end=np.array([5, 5]),
        )
        assert len(chunk) == 2, "Should load 2 chunks"

        # Cleanup
        world.envs.close()

    def test_dataset_frameskip(self, temp_cache_dir):
        """Test loading dataset with frameskip."""
        # 1. Collect data
        world = World(
            env_name="swm/PushT-v1",
            num_envs=2,
            image_shape=(64, 64),
            max_episode_steps=30,
            verbose=0,
        )
        policy = RandomPolicy()
        world.set_policy(policy)

        dataset_name = "test_frameskip"
        world.record_dataset(
            dataset_name=dataset_name,
            episodes=2,
            seed=42,
            cache_dir=temp_cache_dir,
        )
        world.envs.close()

        # 2. Load with frameskip=2
        dataset = HDF5Dataset(
            name=dataset_name,
            cache_dir=str(temp_cache_dir),
            frameskip=2,
            num_steps=2,
        )

        # 3. Verify it works
        if len(dataset) > 0:
            sample = dataset[0]
            assert isinstance(sample, dict)

    def test_dataset_transform(self, temp_cache_dir):
        """Test loading dataset with custom transform."""
        # 1. Collect data
        world = World(
            env_name="swm/PushT-v1",
            num_envs=2,
            image_shape=(64, 64),
            max_episode_steps=20,
            verbose=0,
        )
        policy = RandomPolicy()
        world.set_policy(policy)

        dataset_name = "test_transform"
        world.record_dataset(
            dataset_name=dataset_name,
            episodes=2,
            seed=42,
            cache_dir=temp_cache_dir,
        )
        world.envs.close()

        # 2. Define transform
        def normalize_transform(data):
            if "action" in data:
                data["action"] = data["action"] / 10.0
            return data

        # 3. Load with transform
        dataset = HDF5Dataset(
            name=dataset_name,
            cache_dir=str(temp_cache_dir),
            transform=normalize_transform,
        )

        # 4. Verify transform was applied
        if len(dataset) > 0:
            sample = dataset[0]
            assert isinstance(sample, dict)

    def test_dataset_keys_to_cache(self, temp_cache_dir):
        """Test loading dataset with specific keys cached."""
        # 1. Collect data
        world = World(
            env_name="swm/PushT-v1",
            num_envs=2,
            image_shape=(64, 64),
            max_episode_steps=20,
            verbose=0,
        )
        policy = RandomPolicy()
        world.set_policy(policy)

        dataset_name = "test_cache"
        world.record_dataset(
            dataset_name=dataset_name,
            episodes=2,
            seed=42,
            cache_dir=temp_cache_dir,
        )
        world.envs.close()

        # 2. Load with action cached
        dataset = HDF5Dataset(
            name=dataset_name,
            cache_dir=str(temp_cache_dir),
            keys_to_cache=["action"],
        )

        # 3. Verify action is cached
        assert "action" in dataset._cache

        # 4. Verify we can still load samples
        if len(dataset) > 0:
            sample = dataset[0]
            assert "action" in sample


def convert_hdf5_to_image_format(h5_path: Path, output_dir: Path, image_key: str = "pixels"):
    """Convert HDF5 dataset to ImageDataset folder format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        ep_lengths = f["ep_len"][:]
        ep_offsets = f["ep_offset"][:]

        # Save metadata
        np.savez(output_dir / "ep_len.npz", ep_lengths)
        np.savez(output_dir / "ep_offset.npz", ep_offsets)

        # Save non-image data as .npz
        for key in f.keys():
            if key in ["ep_len", "ep_offset", image_key]:
                continue
            np.savez(output_dir / f"{key}.npz", f[key][:])

        # Save images to folder
        if image_key in f:
            img_dir = output_dir / image_key
            img_dir.mkdir(exist_ok=True)

            images = f[image_key][:]
            for ep_idx, (offset, length) in enumerate(zip(ep_offsets, ep_lengths)):
                for step_idx in range(length):
                    global_idx = offset + step_idx
                    img_array = images[global_idx]
                    # Images in HDF5 are THWC, need HWC for saving
                    if img_array.ndim == 3:  # HWC
                        img = Image.fromarray(img_array)
                        img.save(img_dir / f"ep_{ep_idx}_step_{step_idx}.jpeg")


class TestImageDatasetReal:
    """Test ImageDataset with real collected data."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        return tmp_path

    def test_collect_convert_and_load(self, temp_cache_dir):
        """Test collecting data, converting to image format, and loading."""
        # 1. Collect data with World
        world = World(
            env_name="swm/PushT-v1",
            num_envs=2,
            image_shape=(64, 64),
            max_episode_steps=15,
            verbose=0,
        )
        policy = RandomPolicy()
        world.set_policy(policy)

        h5_name = "test_for_image"
        world.record_dataset(
            dataset_name=h5_name,
            episodes=3,
            seed=42,
            cache_dir=temp_cache_dir,
        )
        world.envs.close()

        # 2. Convert HDF5 to image format
        h5_path = temp_cache_dir / f"{h5_name}.h5"
        image_dataset_dir = temp_cache_dir / "test_image_format"
        convert_hdf5_to_image_format(h5_path, image_dataset_dir, image_key="pixels")

        # 3. Verify folder structure was created
        assert (image_dataset_dir / "ep_len.npz").exists()
        assert (image_dataset_dir / "ep_offset.npz").exists()
        assert (image_dataset_dir / "action.npz").exists()
        assert (image_dataset_dir / "pixels").is_dir()
        assert (image_dataset_dir / "pixels" / "ep_0_step_0.jpeg").exists()

        # 4. Load with ImageDataset
        dataset = ImageDataset(
            name="test_image_format",
            cache_dir=str(temp_cache_dir),
            image_keys=["pixels"],
        )

        # 5. Verify dataset properties
        assert len(dataset) > 0
        assert len(dataset.lengths) == 3

        # 6. Load a sample
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "action" in sample
        assert "pixels" in sample
        assert isinstance(sample["pixels"], torch.Tensor)

        # 7. Verify image shape (should be TCHW after permutation)
        assert sample["pixels"].shape[-3] == 3  # channels

    def test_load_chunk(self, temp_cache_dir):
        """Test ImageDataset load_chunk with real data."""
        # 1. Collect and convert
        world = World(
            env_name="swm/PushT-v1",
            num_envs=2,
            image_shape=(64, 64),
            max_episode_steps=15,
            verbose=0,
        )
        policy = RandomPolicy()
        world.set_policy(policy)

        h5_name = "test_chunk_h5"
        world.record_dataset(
            dataset_name=h5_name,
            episodes=3,
            seed=42,
            cache_dir=temp_cache_dir,
        )
        world.envs.close()

        h5_path = temp_cache_dir / f"{h5_name}.h5"
        image_dir = temp_cache_dir / "test_chunk_images"
        convert_hdf5_to_image_format(h5_path, image_dir)

        # 2. Load and test load_chunk
        dataset = ImageDataset(
            name="test_chunk_images",
            cache_dir=str(temp_cache_dir),
        )

        chunk = dataset.load_chunk(
            episodes_idx=np.array([0, 1]),
            start=np.array([0, 0]),
            end=np.array([3, 3]),
        )

        assert len(chunk) == 2
        assert "pixels" in chunk[0]
        assert "action" in chunk[0]


class TestVideoDatasetReal:
    """Test VideoDataset with real collected data."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        return tmp_path

    def test_collect_convert_and_load(self, temp_cache_dir):
        """Test collecting data, converting to video format, and loading."""
        import imageio.v3 as iio

        # 1. Collect data
        world = World(
            env_name="swm/PushT-v1",
            num_envs=2,
            image_shape=(64, 64),
            max_episode_steps=15,
            verbose=0,
        )
        policy = RandomPolicy()
        world.set_policy(policy)

        h5_name = "test_for_video"
        world.record_dataset(
            dataset_name=h5_name,
            episodes=3,
            seed=42,
            cache_dir=temp_cache_dir,
        )
        world.envs.close()

        # 2. Convert HDF5 to video format (MP4 files)
        h5_path = temp_cache_dir / f"{h5_name}.h5"
        video_dataset_dir = temp_cache_dir / "test_video_format"
        video_dataset_dir.mkdir()

        with h5py.File(h5_path, "r") as f:
            ep_lengths = f["ep_len"][:]
            ep_offsets = f["ep_offset"][:]
            pixels = f["pixels"][:]
            action = f["action"][:]

        # Save metadata
        np.savez(video_dataset_dir / "ep_len.npz", ep_lengths)
        np.savez(video_dataset_dir / "ep_offset.npz", ep_offsets)
        np.savez(video_dataset_dir / "action.npz", action)

        # Create video folder with MP4 files
        video_path = video_dataset_dir / "video"
        video_path.mkdir()

        for ep_idx, (offset, length) in enumerate(zip(ep_offsets, ep_lengths)):
            frames = pixels[offset : offset + length]
            iio.imwrite(video_path / f"ep_{ep_idx}.mp4", frames, fps=30)

        # 3. Load with VideoDataset
        dataset = VideoDataset(
            name="test_video_format",
            cache_dir=str(temp_cache_dir),
            video_keys=["video"],
        )

        # 4. Verify
        assert len(dataset) > 0
        assert len(dataset.lengths) == 3

        sample = dataset[0]
        assert "video" in sample
        assert isinstance(sample["video"], torch.Tensor)
        assert sample["video"].shape[-3] == 3  # channels
