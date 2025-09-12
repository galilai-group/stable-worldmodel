import os
from pathlib import Path

import json
import datasets
import gymnasium as gym
import numpy as np
import torch
from datasets import Dataset, Features, Image, Value, concatenate_datasets, load_dataset
from loguru import logger as logging
from torchvision import transforms
from .wrappers import MegaWrapper


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

    def record_dataset(self, dataset_path, episodes=10, seed=None, options=None):
        """
        Records a dataset with the current policy running in the first environment of a Gymnasium VecEnv.
        Args:
            dataset_path: Output path for the dataset files.
            episodes: Number of episodes to record.
            max_steps: Maximum number of steps per episode.
        """

        # 1. don't wait until last episode before dump everything but do it in shards
        # 2. save metadata (env name, policy type, seed, date, etc..) and episode lengths
        # 3. shift actions by one timestep so you start with obs and action even for reset (you dont need action for last step)
        # 4. add frameskip/actions chunk save option to reduce dataset size, i.e save every k frames and but block of actions


        dataset_path = Path(dataset_path)
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        if not hasattr(self, "episode_saved"):
            self.episode_saved = 0

        self.reset(seed, options)

        # init record with post-reset data
        episode_idx = np.arange(self.num_envs)

        records = {
            key: [v for v in value]
            for key, value in self.infos.items()
            if key[0] != "_"
        }
        records["episode_idx"] = list(episode_idx)
        records["policy"] = [self.policy.type] * self.num_envs

        recorded_episodes = 0
        self.terminateds = np.zeros(self.num_envs)
        self.truncateds = np.zeros(self.num_envs)

        hf_dataset = None
        episodes_len = {i: 0 for i in episode_idx}

        while True:
            # start new episode for done envs
            for i in range(self.num_envs):
                if self.terminateds[i] or self.truncateds[i]:
                    states, infos = self.envs.envs[i].reset()
                    for k, v in infos.items():
                        self.infos[k][i] = v

                    # determine new episode idx
                    next_ep_idx = episode_idx.max() + 1
                    episode_idx[i] = next_ep_idx
                    episodes_len[next_ep_idx] = 0
                    recorded_episodes += 1

            if recorded_episodes >= episodes:
                break

            # increase step counter by one for all envs
            for i in episode_idx:
                episodes_len[int(i)] += 1

            self.step()
            for key in self.infos:
                if key[0] == "_":
                    continue
                records[key].extend(list(self.infos[key]))
            records["episode_idx"].extend(list(episode_idx))
            records["policy"].extend([self.policy.type] * self.num_envs)
        
        # add the episode length
        records["episode_len"] = [episodes_len[int(ep)] for ep in records["episode_idx"]]

        # determine feature
        features = {
            "pixels": Image(),
            "episode_idx": Value("int32"),
            "step_idx": Value("int32"),
            "episode_len": Value("int32"),
        }
        if "goal" in records:
            features["goal"] = Image()
        for k in records:
            if k in features:
                continue
            if type(records[k][0]) is str:
                state_feature = Value("string")
            elif records[k][0].ndim == 1:
                state_feature = datasets.Sequence(
                    feature=Value(dtype=records[k][0].dtype.name)
                )
            elif 2 <= records[k][0].ndim <= 6:
                feature_cls = getattr(datasets, f"Array{records[k][0].ndim}D")
                state_feature = feature_cls(
                    shape=records[k][0].shape, dtype=records[k][0].dtype
                )
            else:
                state_feature = Value(records[k][0].dtype.name)
            features[k] = state_feature
        # concat data
        features = Features(features)
        print(features)
        hf_dataset = Dataset.from_dict(records, features=features)

        # flush extra data
        # incomplete episode
        ep_col = np.array(hf_dataset["episode_idx"])
        non_complete_episodes = np.array(episode_idx[~(self.terminateds | self.truncateds)])
        keep_episode = np.nonzero(~np.isin(ep_col, non_complete_episodes))[0].tolist()
        hf_dataset = hf_dataset.select(keep_episode)

        # re-index episode idx
        unique_eps = np.unique(hf_dataset["episode_idx"])
        id_map = {old: new for new, old in enumerate(unique_eps)}
        hf_dataset = hf_dataset.map(
            lambda row: {"episode_idx": id_map[row["episode_idx"]]}
        )

        # extra episode
        hf_dataset = hf_dataset.filter(
            lambda row: row["episode_idx"] <= episodes
        )

        shard_idx = 0
        while True:
            if not (dataset_path / f"data_shard_{shard_idx:05d}.parquet").is_file():
                break
            shard_idx += 1

        hf_dataset.to_parquet(dataset_path / f"data_shard_{shard_idx:05d}.parquet")

    def record_video_from_dataset(
        self, video_path, dataset_path, episode_idx, max_steps=500, fps=30, num_proc=4
    ):
        """
        Records a video of the current policy running in the first environment of a Gymnasium VecEnv.
        Args:
            video_path: Output path for the video file.
            dataset_path: Output path for the video file.
            episode_idx: Episode index to record or list of episode indices.
            max_steps: Maximum number of steps to record.
            fps: Frames per second for the video.
        """
        import imageio

        # TODO add goal support?

        if isinstance(episode_idx, int):
            episode_idx = [episode_idx]

        out = [
            imageio.get_writer(
                Path(video_path) / f"episode_{i}.mp4",
                "output.mp4",
                fps=fps,
                codec="libx264",
            )
            for i in episode_idx
        ]

        dataset = load_dataset(
            "parquet", data_files=str(Path(dataset_path, "*.parquet")), split="train"
        )

        for i, o in zip(episode_idx, out):
            episode = dataset.filter(
                lambda ex: ex["episode_idx"] == i, num_proc=num_proc
            )
            episode = episode.sort("step_idx")
            episode_len = len(episode)

            for step_idx in range(min(episode_len, max_steps)):

                frame = episode[step_idx]["pixels"]
                frame = np.array(frame.convert("RGB"), dtype=np.uint8)

                if "goal" in episode.column_names:
                    goal = episode[step_idx]["goal"]
                    goal = np.array(goal.convert("RGB"), dtype=np.uint8)
                    frame = np.vstack([frame, goal])
                o.append_data(frame)
        [o.close() for o in out]
        print(f"Video saved to {video_path}")
