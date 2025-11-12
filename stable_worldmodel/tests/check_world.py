import os
import shutil

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import pytest
from datasets import load_from_disk

import stable_worldmodel as swm


os.environ["MUJOCO_GL"] = "egl"
swm_env = [env_id for env_id in gym.envs.registry.keys() if env_id.startswith("swm/")]


@pytest.fixture(scope="session")
def temp_path(tmp_path_factory, request):
    tmp_dir = tmp_path_factory.mktemp("data")

    def cleanup():
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    request.addfinalizer(cleanup)
    return tmp_dir


@pytest.mark.parametrize("env", swm_env)
def test_each_env(env, temp_path):
    EPISODE_LENGTH = 10

    world = swm.World(
        env,
        num_envs=1,
        image_shape=(224, 224),
        max_episode_steps=EPISODE_LENGTH,
        render_mode="rgb_array",
        verbose=0,
    )

    print(f"Testing env {env} with temp path {temp_path}")

    ds_name = f"tmp-{env.replace('swm/', '').lower()}"

    policy = swm.policy.RandomPolicy(42)

    # dataset 1
    world.set_policy(policy)
    world.record_dataset(ds_name, episodes=1, seed=2347, cache_dir=temp_path)

    # dataset 2
    world.set_policy(policy)  # reset policy for deterministic behavior
    world.record_dataset(f"{ds_name}-2", episodes=1, seed=2347, cache_dir=temp_path)

    # video 1
    world.set_policy(policy)  # reset policy for deterministic behavior
    world.record_video(f"{temp_path}/{ds_name}", seed=2347)

    # video 2
    world.set_policy(policy)  # reset policy for deterministic behavior
    world.record_video(f"{temp_path}/{ds_name}-2", seed=2347)

    world.record_video_from_dataset(f"{temp_path}/{ds_name}", ds_name, episode_idx=0, cache_dir=temp_path)
    world.record_video_from_dataset(f"{temp_path}/{ds_name}-2", ds_name, episode_idx=0, cache_dir=temp_path)

    # assert all files are created
    assert os.path.exists(f"{temp_path}/{ds_name}/dataset_info.json")
    assert os.path.exists(f"{temp_path}/{ds_name}/env_0.mp4")
    assert os.path.exists(f"{temp_path}/{ds_name}/episode_0.mp4")

    assert os.path.exists(f"{temp_path}/{ds_name}-2/dataset_info.json")
    assert os.path.exists(f"{temp_path}/{ds_name}-2/env_0.mp4")
    assert os.path.exists(f"{temp_path}/{ds_name}-2/episode_0.mp4")

    # === load videos
    recorded_video = iio.imread(f"{temp_path}/{ds_name}/env_0.mp4", index=None)
    dataset_video = iio.imread(f"{temp_path}/{ds_name}/episode_0.mp4", index=None)

    recorded_video2 = iio.imread(f"{temp_path}/{ds_name}-2/env_0.mp4", index=None)
    dataset_video2 = iio.imread(f"{temp_path}/{ds_name}-2/episode_0.mp4", index=None)

    assert isinstance(recorded_video, np.ndarray)
    assert isinstance(dataset_video, np.ndarray)
    assert isinstance(recorded_video2, np.ndarray)
    assert isinstance(dataset_video2, np.ndarray)

    assert recorded_video.dtype == np.uint8
    assert dataset_video.dtype == np.uint8
    assert recorded_video2.dtype == np.uint8
    assert dataset_video2.dtype == np.uint8

    assert recorded_video.shape[0] == EPISODE_LENGTH
    assert dataset_video.shape[0] == EPISODE_LENGTH
    assert recorded_video2.shape[0] == EPISODE_LENGTH
    assert dataset_video2.shape[0] == EPISODE_LENGTH

    assert recorded_video.shape[3] == 3  # RGB channels
    assert dataset_video.shape[3] == 3  # RGB channels
    assert recorded_video2.shape[3] == 3  # RGB channels
    assert dataset_video2.shape[3] == 3  # RGB channels

    assert recorded_video.shape == dataset_video.shape  # both videos should have the same shape
    assert recorded_video2.shape == dataset_video2.shape  # both videos should have the same shape
    assert recorded_video2.shape == recorded_video.shape  # both videos should have the same shape

    # assert np.allclose(recorded_video, recorded_video2)  # both videos should be identical
    # assert np.allclose(dataset_video, dataset_video2)  # both videos should be identical

    # ====== load dataset
    dataset = load_from_disk(f"{temp_path}/{ds_name}").with_format("numpy")
    dataset2 = load_from_disk(f"{temp_path}/{ds_name}-2").with_format("numpy")

    assert len(dataset) == EPISODE_LENGTH, "Dataset length should be equal to episode length"
    assert len(dataset2) == EPISODE_LENGTH, "Dataset lengths should be identical"
    assert set(dataset.column_names) == set(dataset2.column_names), "Dataset columns should be identical"

    # check all actions are equivalent
    action = dataset["action"][:]
    action = action[~np.isnan(action)]

    action2 = dataset2["action"][:]
    action2 = action2[~np.isnan(action2)]

    assert np.allclose(action, action2), "Actions should be identical"

    # check the env logic
    world.reset()

    assert "action" in world.infos
    assert any(key.startswith("pixels") for key in world.infos)

    for key in world.infos:
        if key.startswith("pixels"):
            assert world.infos[key].shape[:4] == (1, 1, 224, 224), f"image shape is {world.infos[key].shape}"

    world.step()

    assert "action" in world.infos
    assert any(key.startswith("pixels") for key in world.infos)

    for key in world.infos:
        if key.startswith("pixels"):
            assert world.infos[key].shape[:4] == (1, 1, 224, 224), f"image shape is {world.infos[key].shape}"

    return
