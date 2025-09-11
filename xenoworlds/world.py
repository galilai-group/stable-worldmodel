import gymnasium as gym
import numpy as np
import torch
import datasets
from loguru import logger as logging
from torchvision import transforms
from .wrappers import MegaWrapper
from pathlib import Path

from datasets import Dataset, concatenate_datasets, Features, Value, Image

import os


class World:
    def __init__(
        self,
        env_name,
        num_envs,
        image_shape,
        image_transform=None,
        goal_shape=None,
        goal_transform=None,
        seed: int = 2349867,
        max_episode_steps: int = 100,
        **kwargs,
    ):
        self.envs = gym.make_vec(
            env_name,
            num_envs=num_envs,
            vectorization_mode="sync",
            wrappers=[
                lambda x: MegaWrapper(
                    x, image_shape, image_transform, goal_shape, goal_transform
                )
            ],
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

        self.goal_envs = gym.make_vec(
            env_name,
            num_envs=num_envs,
            vectorization_mode="sync",
            wrappers=[lambda x: MegaWrapper(x, image_shape, image_transform)],
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

        logging.info("WORLD INITIALIZED")
        logging.info(f"ACTION SPACE: {self.envs.action_space}")
        logging.info(f"OBSERVATION SPACE: {self.envs.observation_space}")
        self.seed = seed

        rng = torch.Generator()
        rng.manual_seed(seed)
        self.goal_seed = torch.randint(0, 2**32 - 1, (1,), generator=rng).item()

        # note if sample_goal_every_k_steps is set to -1, will sample goal once per episode
        # TODO implement sample_goal_every_k_steps

    @property
    def num_envs(self):
        return self.envs.num_envs

    @property
    def observation_space(self):
        return self.envs.observation_space

    @property
    def action_space(self):
        return self.envs.action_space

    @property
    def single_action_space(self):
        return self.envs.single_action_space

    @property
    def single_observation_space(self):
        return self.envs.single_observation_space

    def close(self, **kwargs):
        return self.envs.close(**kwargs)

    # TEMOPORARY, need to delete!!!
    def denormalize(self, x):
        # x is (B,C,H,W) in [-1,1]
        return (x * 0.5) + 0.5

    def __iter__(self):
        self.terminations = np.array([False] * self.num_envs)
        self.truncations = np.array([False] * self.num_envs)
        self.rewards = None
        logging.info(f"Resetting the ({self.num_envs}) world(s)!")
        self.states, self.infos = self.envs.reset(seed=self.seed)
        self.cur_goals = self.infos["goal"]
        self.cur_goal_images = self.infos["goal_image"]
        return self

    def __next__(self):
        if not all(self.terminations) and not all(self.truncations):
            return self.states, self.infos
        else:
            raise StopIteration

    def step(self):
        actions = self.policy.get_action(self.infos)
        (self.states, self.rewards, self.terminateds, self.truncateds, self.infos) = (
            self.envs.step(actions)
        )

    def reset(self, seed=None, options=None):
        self.states, self.infos = self.envs.reset(seed=seed, options=options)

    def set_policy(self, policy):
        self.policy = policy
        self.policy.set_env(self.envs)

    def record_video(self, video_path, max_steps=500, fps=30, seed=None, options=None):
        """
        Records a video of the current policy running in the first environment of a Gymnasium VecEnv.
        Args:
            video_path: Output path for the video file.
            max_steps: Maximum number of steps to record.
            fps: Frames per second for the video.
        """
        import imageio

        out = [
            imageio.get_writer(
                Path(video_path) / f"env_{i}.mp4",
                "output.mp4",
                fps=fps,
                codec="libx264",
            )
            for i in range(self.num_envs)
        ]

        self.reset(seed, options)
        for i, o in enumerate(out):
            if "goal" in self.infos:
                frame = np.vstack([self.infos["pixels"][i], self.infos["goal"][i]])
            else:
                frame = self.infos["pixels"][i]
            o.append_data(frame)
        for _ in range(max_steps):
            self.step()
            for i, o in enumerate(out):
                if "goal" in self.infos:
                    frame = np.vstack([self.infos["pixels"][i], self.infos["goal"][i]])
                else:
                    frame = self.infos["pixels"][i]
                o.append_data(frame)
            if np.any(self.terminateds) or np.any(self.truncateds):
                break
        [o.close() for o in out]
        print(f"Video saved to {video_path}")

    def record_dataset(
        self,
        dataset_path,
        episodes=10,
        max_steps=500,
        seed=None,
        options=None,
        episode_per_shard=float("inf"),
    ):
        """
        Records a dataset with the current policy running in the first environment of a Gymnasium VecEnv.
        Args:
            dataset_path: Output path for the dataset files.
            episodes: Number of episodes to record.
            max_steps: Maximum number of steps per episode.
        """

        hf_dataset = None
        dataset_path = Path(dataset_path)
        dataset_path.mkdir(parents=True, exist_ok=True)

        if not hasattr(self, "episode_saved"):
            self.episode_saved = 0

        # REM: max_episodes steps is overriden the max num steps provided in env instantiation
        # because of the self.truncated condition!

        # epsiodes // num_envs?

        for episode in range(episodes):
            self.reset(seed, options)

            episodes_idx = (
                self.episode_saved
                + (episode * self.num_envs)
                + np.arange(self.num_envs)
            )

            # create dict buffer to store episode data
            data = {
                "episode_idx": episodes_idx,
                "step_idx": np.array([0] * self.num_envs),
                "state": self.states["state"],
                "pixels": self.infos["pixels"],
                "policy": np.array([self.policy.type] * self.num_envs),
            }

            for step in range(max_steps):
                self.step()

                # append data
                data["episode_idx"] = np.concatenate(
                    [data["episode_idx"], episodes_idx], axis=0
                )
                data["step_idx"] = np.concatenate(
                    [data["step_idx"], np.array([step + 1] * self.num_envs)], axis=0
                )
                data["state"] = np.concatenate(
                    [data["state"], self.states["state"]], axis=0
                )
                data["pixels"] = np.concatenate(
                    [data["pixels"], self.infos["pixels"]], axis=0
                )
                data["policy"] = np.concatenate(
                    [data["policy"], np.array([self.policy.type] * self.num_envs)],
                    axis=0,
                )

                # add more data here if needed

                if np.any(self.terminateds) or np.any(self.truncateds):
                    break

            # determine feature
            state_shape = data["state"].shape[1:]
            state_dtype = data["state"].dtype.name
            state_ndim = len(state_shape)
            if 1 < state_ndim <= 6:
                feature_cls = getattr(datasets, f"Array{state_ndim}D")
                state_feature = [feature_cls(shape=state_shape, dtype=state_dtype)]
            else:
                state_feature = [Value(state_dtype)]

            features = Features(
                {
                    "episode_idx": Value("int32"),
                    "step_idx": Value("int32"),
                    "state": state_feature,
                    "pixels": Image(),
                    "policy": Value("string"),
                }
            )

            # concat data
            data = Dataset.from_dict(data, features=features)
            hf_dataset = (
                data if hf_dataset is None else concatenate_datasets([hf_dataset, data])
            )

            # save shard if needed
            if (episode + 1) % episode_per_shard == 0 or (episode + 1) == episodes:
                assert hf_dataset is not None, "hf_dataset should not be None here"

                if episode_per_shard < float("inf"):
                    shard_idx = (
                        episode // episode_per_shard
                        if episode_per_shard < float("inf")
                        else 0
                    )
                    shard_path = dataset_path / f"data_shard_{shard_idx:05d}.parquet"
                else:
                    shard_path = dataset_path / "data.parquet"

                hf_dataset.to_parquet(shard_path)

                old_episode_saved = self.episode_saved
                num_episode_saved = len(hf_dataset.unique("episode_idx"))
                self.episode_saved += num_episode_saved
                hf_dataset = None
                print(
                    f"Saved {num_episode_saved-old_episode_saved} episodes to {shard_path} (total episodes saved: {self.episode_saved})"
                )

            print(f"📹 Episode {episode+1}/{episodes} done.")

        # dataset = load_dataset("parquet", data_files=str(dataset_path/"*.parquet"))
