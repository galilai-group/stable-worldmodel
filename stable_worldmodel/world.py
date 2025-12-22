"""World environment manager.

This module provides the World class, a high-level manager for vectorized Gymnasium
environments with integrated support for data collection, policy evaluation, video
recording, and dataset management. It serves as the central orchestration layer for
training and evaluating world models with domain randomization support.

The World class handles:
    - Vectorized environment creation and management
    - Policy attachment and execution
    - Episode data collection with automatic sharding
    - Video recording from live rollouts or stored datasets
    - Policy evaluation with comprehensive results
    - Visual domain randomization through variation spaces

Example:
    Basic usage for policy evaluation::

        from stable_worldmodel import World
        from stable_worldmodel.policy import RandomPolicy

        # Create a world with 4 parallel environments
        world = World(
            env_name="PushT-v1",
            num_envs=4,
            image_shape=(96, 96),
            max_episode_steps=200
        )

        # Attach a policy
        policy = RandomPolicy()
        world.set_policy(policy)

        # Evaluate the policy
        results = world.evaluate(episodes=100, seed=42)
        print(f"Success rate: {results['success_rate']:.2f}%")

    Data collection example::

        # Collect demonstration data
        world.record_dataset(
            dataset_name="pusht_demos",
            episodes=1000,
            seed=42,
            options={'variation': ['all']}
        )

Todo:
    * Add real-time metric visualization during evaluation

.. _Gymnasium:
   https://gymnasium.farama.org/
"""

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import datasets
import gymnasium as gym
import imageio
import imageio.v3 as iio
import numpy as np
import torch
from datasets import Dataset, Features, Value
from loguru import logger as logging
from PIL import Image
from rich import print
from tqdm import tqdm

from stable_worldmodel.data.dataset import Dataset as SWMDataset
from stable_worldmodel.data.dataset import VideoDataset
from stable_worldmodel.data.utils import get_cache_dir, is_image

from .wrappers import MegaWrapper, VariationWrapper


class World:
    """High-level manager for vectorized Gymnasium environments with integrated data collection.

    World orchestrates multiple parallel environments, providing a unified interface for
    policy execution, data collection, evaluation, and visualization. It automatically
    handles environment resets, batched action execution, and comprehensive episode
    tracking with support for visual domain randomization.

    The World class is the central component of the stable-worldmodel library, designed
    to streamline the workflow from data collection to policy training and evaluation.
    It wraps Gymnasium's vectorized environments with additional functionality for
    world model research.

    Attributes:
        envs (VectorEnv): Vectorized Gymnasium environment wrapped with MegaWrapper
            and VariationWrapper for image processing and domain randomization.
        seed (int): Base random seed for reproducibility.
        policy (Policy): Currently attached policy that generates actions.
        states (ndarray): Current observation states for all environments.
        rewards (ndarray): Current rewards for all environments.
        terminateds (ndarray): Terminal flags indicating episode completion.
        truncateds (ndarray): Truncation flags indicating episode timeout.
        infos (dict): Dictionary containing major information from environments.

    Properties:
        num_envs (int): Number of parallel environments.
        observation_space (Space): Batched observation space.
        action_space (Space): Batched action space.
        variation_space (Dict): Variation space for domain randomization (batched).
        single_variation_space (Dict): Variation space for a single environment.
        single_action_space (Space): Action space for a single environment.
        single_observation_space (Space): Observation space for a single environment.

    Note:
        Environments use DISABLED autoreset mode to enable custom reset behavior
        with seed and options support, which is not provided by Gymnasium's default
        autoreset mechanism.
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        image_shape: tuple,
        goal_transform: Callable | None = None,
        image_transform: Callable | None = None,
        seed: int = 2349867,
        history_size: int = 1,
        frame_skip: int = 1,
        max_episode_steps: int = 100,
        verbose: int = 1,
        extra_wrappers: list | None = None,
        goal_conditioned: bool = True,
        **kwargs,
    ):
        """Initialize the World with vectorized environments.

        Creates and configures a vectorized environment with the specified number of
        parallel instances. Applies image and goal transformations, sets up variation
        support, and configures autoreset behavior.

        Args:
            env_name (str): Name of the Gymnasium environment to create. Must be a
                registered environment ID (e.g., 'PushT-v0', 'CubeEnv-v0').
            num_envs (int): Number of parallel environment instances to create.
                Higher values increase data collection throughput but require more memory.
            image_shape (tuple): Target shape for image observations as (height, width)
                or (height, width, channels). Images are resized to this shape.
            goal_transform (Callable, optional): Function to transform goal observations.
                Should accept and return numpy arrays. Applied after resizing.
                Defaults to None.
            image_transform (Callable, optional): Function to transform image observations.
                Should accept and return numpy arrays. Applied after resizing.
                Defaults to None.
            seed (int, optional): Base random seed for environment initialization.
                Each environment gets an offset seed. Defaults to 2349867.
            max_episode_steps (int, optional): Maximum number of steps per episode
                before truncation. Episodes terminate early on task success.
                Defaults to 100.
            extra_wrappers (list, optional): List of extra wrappers to apply to each
                environment. Useful for adding custom behavior or modifications.
                Defaults to None.
            verbose (int, optional): Verbosity level. 0 for silent, 1 for basic info,
                2+ for detailed debugging information. Defaults to 1.
            **kwargs: Additional keyword arguments passed to gym.make_vec() and
                subsequently to the underlying environment constructor.

        Note:
            The MegaWrapper applies image transformations and resizing.
            The VariationWrapper enables domain randomization support.
            Autoreset is disabled to allow custom reset with seeds and options.

        Example:
            Create a world with goal-conditioned observations::

                world = World(
                    env_name="PushT-v1",
                    num_envs=8,
                    image_shape=(96, 96),
                    max_episode_steps=150,
                    seed=42
                )
        """
        self.envs = gym.make_vec(
            env_name,
            num_envs=num_envs,
            vectorization_mode="sync",
            wrappers=[
                lambda x: MegaWrapper(
                    x,
                    image_shape,
                    image_transform,
                    goal_transform,
                    history_size=history_size,
                    frame_skip=frame_skip,
                    separate_goal=goal_conditioned,
                )
            ]
            + (extra_wrappers or []),
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

        self.envs = VariationWrapper(self.envs)
        self.envs.unwrapped.autoreset_mode = gym.vector.AutoresetMode.DISABLED

        self._history_size = history_size

        if verbose > 0:
            logging.info(f"🌍🌍🌍 World {env_name} initialized 🌍🌍🌍")

            logging.info("🕹️ 🕹️ 🕹️ Action space 🕹️ 🕹️ 🕹️")
            logging.info(f"{self.envs.action_space}")

            logging.info("👁️ 👁️ 👁️ Observation space 👁️ 👁️ 👁️")
            logging.info(f"{str(self.envs.observation_space)}")

            if self.envs.variation_space is not None:
                logging.info("⚗️ ⚗️ ⚗️ Variation space ⚗️ ⚗️ ⚗️")
                print(self.single_variation_space.to_str())
            else:
                logging.warning("No variation space provided!")

        self.seed = seed

    @property
    def num_envs(self):
        """int: Number of parallel environment instances."""
        return self.envs.num_envs

    @property
    def observation_space(self):
        """Space: Batched observation space for all environments."""
        return self.envs.observation_space

    @property
    def action_space(self):
        """Space: Batched action space for all environments."""
        return self.envs.action_space

    @property
    def variation_space(self):
        """Dict: Batched variation space for domain randomization across all environments."""
        return self.envs.variation_space

    @property
    def single_variation_space(self):
        """Dict: Variation space for a single environment instance."""
        return self.envs.single_variation_space

    @property
    def single_action_space(self):
        """Space: Action space for a single environment instance."""
        return self.envs.single_action_space

    @property
    def single_observation_space(self):
        """Space: Observation space for a single environment instance."""
        return self.envs.single_observation_space

    def close(self, **kwargs):
        """Close all environments and clean up resources.

        Args:
            **kwargs: Additional keyword arguments passed to the underlying
                vectorized environment's close method.

        Returns:
            Any: Return value from the underlying close method.
        """
        return self.envs.close(**kwargs)

    def step(self):
        """Advance all environments by one step using the current policy.

        Queries the attached policy for actions based on current info, executes
        those actions in all environments, and updates internal state with the
        results.

        Updates:
            - self.states: New observations from all environments
            - self.rewards: Rewards received from the step
            - self.terminateds: Episode termination flags (task success)
            - self.truncateds: Episode truncation flags (timeout)
            - self.infos: Auxiliary information dictionaries

        Note:
            Requires a policy to be attached via set_policy() before calling.
            The policy's get_action() method receives the current infos dict.

        Raises:
            AttributeError: If no policy has been set via set_policy().
        """
        # note: reset happens before because of auto-reset, should fix that
        actions = self.policy.get_action(self.infos)
        self.states, self.rewards, self.terminateds, self.truncateds, self.infos = self.envs.step(actions)

    def reset(self, seed=None, options=None):
        """Reset all environments to initial states.

        Args:
            seed (int, optional): Random seed for reproducible resets. If None,
                uses non-deterministic seeding. Defaults to None.
            options (dict, optional): Dictionary of reset options passed to
                environments. Common keys include 'variation' for domain
                randomization. Defaults to None.

        Updates:
            - self.states: Initial observations from all environments
            - self.infos: Initial auxiliary information

        Example:
            Reset with domain randomization::

                world.reset(seed=42, options={'variation': ['all']})

            Reset specific variations::

                world.reset(options={'variation': ['cube.color', 'light.intensity']})
        """
        self.states, self.infos = self.envs.reset(seed=seed, options=options)

    def set_policy(self, policy):
        """Attach a policy to the world and configure it with environment context.

        The policy will be used for action generation during step(), record_video(),
        record_dataset(), and evaluate() calls.

        Args:
            policy (Policy): Policy instance that implements get_action(infos) method.
                The policy receives environment context through set_env() and optional
                seeding through set_seed().

        Note:
            If the policy has a 'seed' attribute, it will be applied via set_seed().
            The policy's set_env() method receives the wrapped vectorized environment.
        """
        self.policy = policy
        self.policy.set_env(self.envs)

        if hasattr(self.policy, "seed") and self.policy.seed is not None:
            self.policy.set_seed(self.policy.seed)

    def record_video(
        self,
        video_path,
        max_steps=500,
        fps=30,
        viewname="pixels",
        seed=None,
        options=None,
    ):
        """Record rollout videos for each environment under the current policy.

        Executes policy rollouts in all environments and saves them as MP4 videos,
        one per environment. Videos show stacked observation and goal images when
        goals are available.

        Args:
            video_path (str or Path): Directory path where videos will be saved.
                Created if it doesn't exist. Videos are named 'env_0.mp4', 'env_1.mp4', etc.
            max_steps (int, optional): Maximum number of steps to record per episode.
                Recording stops earlier if any environment terminates. Defaults to 500.
            fps (int, optional): Frames per second for output videos. Higher values
                produce smoother playback but larger files. Defaults to 30.
            seed (int, optional): Random seed for reproducible rollouts. Defaults to None.
            options (dict, optional): Reset options passed to environments (e.g.,
                variation settings). Defaults to None.

        Note:
            - Videos use libx264 codec for compatibility
            - If 'goal' is present in infos, frames show observation stacked above goal
            - All environments are recorded simultaneously
            - Recording stops when ANY environment terminates or truncates

        Example:
            Record evaluation videos::

                world.set_policy(trained_policy)
                world.record_video(
                    video_path="./eval_videos",
                    max_steps=200,
                    fps=30,
                    seed=42
                )
        """

        viewname = [viewname] if isinstance(viewname, str) else viewname
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
            frames_to_stack = []
            for v_name in viewname:
                frame_data = self.infos[v_name][i]
                # if frame_data has a history dimension, take the last frame
                if frame_data.ndim > 3:
                    frame_data = frame_data[-1]
                frames_to_stack.append(frame_data)
            frame = np.vstack(frames_to_stack)

            if "goal" in self.infos:
                goal_data = self.infos["goal"][i]
                if goal_data.ndim > 3:
                    goal_data = goal_data[-1]
                frame = np.vstack([frame, goal_data])
            o.append_data(frame)

        for _ in range(max_steps):
            self.step()

            if np.any(self.terminateds) or np.any(self.truncateds):
                break

            for i, o in enumerate(out):
                frames_to_stack = []
                for v_name in viewname:
                    frame_data = self.infos[v_name][i]
                    # if frame_data has a history dimension, take the last frame
                    if frame_data.ndim > 3:
                        frame_data = frame_data[-1]
                    frames_to_stack.append(frame_data)
                frame = np.vstack(frames_to_stack)

                if "goal" in self.infos:
                    goal_data = self.infos["goal"][i]
                    if goal_data.ndim > 3:
                        goal_data = goal_data[-1]
                    frame = np.vstack([frame, goal_data])
                o.append_data(frame)
        [o.close() for o in out]
        print(f"Video saved to {video_path}")

    def record_dataset(self, dataset_name, episodes=10, seed=None, cache_dir=None, mode="frame", options=None):
        if self._history_size > 1:
            raise NotImplementedError("Dataset recording with frame history > 1 is not supported.")

        cache_dir = cache_dir or get_cache_dir()
        dataset_path = Path(cache_dir, dataset_name)
        dataset_path.mkdir(parents=True, exist_ok=True)

        if (dataset_path / "state.json").exists():
            logging.warning(f"Dataset {dataset_name} already exists at {dataset_path}. Aborting recording.")
            return

        recorded_episodes = 0
        episode_idx = np.arange(self.num_envs)

        self.terminateds = np.zeros(self.num_envs)
        self.truncateds = np.zeros(self.num_envs)

        self.reset(seed, options)  # <- incr global seed by num_envs
        root_seed = seed + self.num_envs if seed is not None else None

        records = {k: [] for k in ["episode_len", "episode_idx", "policy"]}
        episode_buffers = [defaultdict(list) for _ in range(self.num_envs)]

        def dump_to_buffer(env_idx=None):
            if not isinstance(env_idx, int) and env_idx is not None:
                raise ValueError("env_idx must be an integer or None.")

            env_idx = [env_idx] if env_idx is not None else list(range(self.num_envs))

            for k, v in self.infos.items():
                if k[0] == "_":
                    continue

                if isinstance(v, np.ndarray):
                    v = v.squeeze(1) if len(v.shape) > 1 and v.shape[1] == 1 else v
                    v = v if v.dtype.type != np.object_ else np.concatenate(v).tolist()

                for i in env_idx:
                    buffer = episode_buffers[i]
                    buffer[k].append(deepcopy(v[i]))

        def flush_episode(env_idx):
            buffer = episode_buffers[env_idx]
            episode_len = len(buffer["step_idx"])
            for k, v in buffer.items():
                if k not in records:
                    records[k] = []

                # rotate actions to align with observations
                if k == "action":
                    nan = v.pop(0).astype(v[0].dtype)
                    v.append(nan)

                records[k].extend(v)

            records["episode_len"].extend([episode_len for _ in range(episode_len)])
            records["episode_idx"].extend([recorded_episodes for _ in range(episode_len)])
            records["policy"].extend([self.policy.type for _ in range(episode_len)])

            episode_buffers[env_idx] = defaultdict(list)

            self.terminateds[env_idx] = False
            self.truncateds[env_idx] = False

        # dump the original reset data
        dump_to_buffer()

        with tqdm(total=episodes, desc="Collecting Episodes", unit="ep") as pbar:
            while True:
                # start new episode for done envs
                for i in range(self.num_envs):
                    if self.terminateds[i] or self.truncateds[i]:
                        flush_episode(i)

                        # TODO open issue double nested insert with numpy arrays

                        # re-reset env with seed and options (no supported by auto-reset)
                        new_seed = root_seed + recorded_episodes if seed is not None else None

                        # determine new episode idx
                        next_ep_idx = episode_idx.max() + 1
                        episode_idx[i] = next_ep_idx
                        recorded_episodes += 1

                        # Update the progress bar by 1
                        pbar.update(1)

                        self.envs.unwrapped._autoreset_envs = np.zeros((self.num_envs,))
                        _, infos = self.envs.envs[i].reset(seed=new_seed, options=options)

                        for k, v in infos.items():
                            self.infos[k][i] = v

                        dump_to_buffer(env_idx=i)

                if recorded_episodes >= episodes:
                    break

                self.step()
                dump_to_buffer()

        # flatten time dimension
        for k, v in records.items():
            if isinstance(v[0], np.ndarray):
                records[k] = [item.squeeze() for item in v]

        # add the episode length
        # counts = np.bincount(np.array(records["episode_idx"]), minlength=max(records["episode_idx"]) + 1)
        # records["episode_len"] = [int(counts[ep]) for ep in records["episode_idx"]]

        ########################
        # Save dataset to disk #
        ########################

        assert any(key.startswith("pixels") for key in records), "pixels key is required in records"
        assert "episode_idx" in records, "episode_idx key is required in records"
        assert "step_idx" in records, "step_idx key is required in records"
        assert "episode_len" in records, "episode_len key is required in records"

        # Create the dataset directory structure
        dataset_path.mkdir(parents=True, exist_ok=True)

        # save all jpeg images
        image_cols = {col for col in records if is_image(records[col][0])}

        # TODO: should check if already some data has been saved
        episode_idx = np.unique(records["episode_idx"])

        for ep in tqdm(episode_idx, desc="Saving Media to Disk", unit="ep"):
            ep_mask = records["episode_idx"] == ep
            step_mask = np.flatnonzero(ep_mask)

            for img_col in image_cols:
                col_name = img_col.replace(".", "_")
                imgs = np.stack([records[img_col][i] for i in step_mask], axis=0)

                # Prepare output path
                if mode == "video":
                    output_dir = dataset_path / "videos"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    file_path = output_dir / f"{ep}_{col_name}.mp4"
                    iio.imwrite(file_path, imgs)
                    paths = [str(file_path.relative_to(dataset_path))] * len(step_mask)
                elif mode == "frame":
                    output_dir = dataset_path / "img" / f"{ep}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    paths = []
                    for idx, img in enumerate(imgs):
                        file_path = output_dir / f"{idx}_{col_name}.jpeg"
                        iio.imwrite(file_path, img)
                        paths.append(str(file_path.relative_to(dataset_path)))
                else:
                    raise ValueError(f"Unknown mode {mode} for saving dataset images/videos.")

                # Update records
                for i, path in zip(step_mask, paths):
                    records[img_col][i] = path

        def determine_features(records):
            features = {
                "episode_idx": Value("int32"),
                "step_idx": Value("int32"),
                "episode_len": Value("int32"),
            }

            for col_name in records:
                if col_name in features:
                    continue

                first_elem = records[col_name][0]

                if type(first_elem) is str:
                    features[col_name] = Value("string")

                elif isinstance(first_elem, np.ndarray):
                    if first_elem.ndim == 1:
                        state_feature = datasets.Sequence(
                            feature=Value(dtype=first_elem.dtype.name),
                            length=len(first_elem),
                        )
                    elif 2 <= first_elem.ndim <= 6:
                        feature_cls = getattr(datasets, f"Array{first_elem.ndim}D")
                        state_feature = feature_cls(shape=first_elem.shape, dtype=first_elem.dtype.name)
                    else:
                        state_feature = Value(first_elem.dtype.name)
                    features[col_name] = state_feature

                elif isinstance(first_elem, (np.generic)):
                    features[col_name] = Value(first_elem.dtype.name)
                else:
                    features[col_name] = Value(type(first_elem).__name__)

            return Features(features)

        records_feat = determine_features(records)
        records_ds = Dataset.from_dict(records, features=records_feat)

        # flush all extra episodes saved (keep only first N episodes)
        episodes_to_keep = np.unique(records_ds["episode_idx"])[:episodes]
        keep_mask = np.isin(records_ds["episode_idx"], episodes_to_keep)
        records_ds = records_ds.select(np.nonzero(keep_mask)[0])

        # save dataset
        records_path = dataset_path
        num_chunks = episodes // 50
        records_path.mkdir(parents=True, exist_ok=True)
        records_ds.save_to_disk(records_path, num_shards=num_chunks or 1)

        print(f"Dataset saved to {dataset_path} with {episodes} episodes!")

    def record_video_from_dataset(
        self,
        video_path,
        dataset,
        episode_idx,
        max_steps=500,
        fps=30,
        num_proc=4,
        viewname: str | list[str] = "pixels",
    ):
        """Replay stored dataset episodes and export them as MP4 videos.

        Loads episodes from a previously recorded dataset and renders them as videos,
        useful for visualization, debugging, and qualitative evaluation of collected data.

        Args:
            video_path (str or Path): Directory where videos will be saved. Videos are
                named 'episode_{idx}.mp4'. Directory is created if it doesn't exist.
            dataset (Dataset): the dataset object to load episodes from.
            episode_idx (int or list of int): Episode index or list of episode indices
                to render. Each episode is saved as a separate video file.
            max_steps (int, optional): Maximum number of steps to render per episode.
                Useful for limiting video length. Defaults to 500.
            fps (int, optional): Frames per second for output videos. Defaults to 30.
            num_proc (int, optional): Number of processes for parallel dataset filtering.
                Higher values speed up loading for large datasets. Defaults to 4.
            cache_dir (str or Path, optional): Root directory where dataset is stored.
                If None, uses get_cache_dir(). Defaults to None.

        Raises:
            AssertionError: If dataset doesn't exist in cache_dir, or if episode
                length inconsistencies are detected in the data.

        Note:
            - Images are loaded from JPEG files stored in the dataset
            - If 'goal' column exists, observation and goal are stacked vertically
            - Videos use libx264 codec for broad compatibility
            - Episodes are validated for consistency (length matches metadata)

        Example:
            Render specific episodes from a dataset::

                world.record_video_from_dataset(
                    video_path="./visualizations",
                    dataset_name="cube_expert_1k",
                    episode_idx=[0, 5, 10, 99],
                    fps=30
                )

            Render a single episode::

                world.record_video_from_dataset(
                    video_path="./debug",
                    dataset_name="failed_episodes",
                    episode_idx=42,
                    max_steps=100
                )
        """
        episode_idx = [episode_idx] if isinstance(episode_idx, int) else episode_idx
        viewname = [viewname] if isinstance(viewname, str) else viewname

        if isinstance(dataset, VideoDataset):
            for ep in episode_idx:
                ep_indices = np.nonzero(np.array(dataset.dataset["episode_idx"]) == ep)[0]
                data = dataset.dataset[ep_indices[0]]

                frames = []
                for view in viewname:
                    video_file = Path(data["data_dir"], data[view])
                    frames.append(iio.imread(video_file))

                frames = np.vstack(frames)
                output_dir = Path(video_path)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"episode_{ep}.mp4"
                iio.imwrite(output_path, frames, fps=fps, codec="libx264")
            print(f"Video saved to {video_path}")
            return

        out = [
            imageio.get_writer(
                Path(video_path) / f"episode_{i}.mp4",
                "output.mp4",
                fps=fps,
                codec="libx264",
            )
            for i in episode_idx
        ]

        for i, o in zip(episode_idx, out):
            episode = dataset.dataset.filter(lambda ex: ex["episode_idx"] == i, num_proc=num_proc)
            episode = episode.sort("step_idx")
            episode_len = len(episode)

            all_lengths = episode["episode_len"][:].tolist()

            assert len(set(all_lengths)) == 1, "'episode_len' contains different values for the same episode"
            assert len(episode) == episode["episode_len"][0], (
                f"Episode {i} has {len(episode)} steps, but 'episode_len' is {episode['episode_len'][0]}"
            )

            for step_idx in range(min(episode_len, max_steps)):
                frame = []
                for view in viewname:
                    img_path = Path(episode[step_idx]["data_dir"], episode[step_idx][view])
                    frame.append(np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8))
                frame = np.vstack(frame)  # should try hstack?

                if "goal" in episode.column_names:
                    goal_path = Path(episode[step_idx]["data_dir"], episode[step_idx]["goal"])
                    goal = Image.open(goal_path)
                    goal = np.array(goal.convert("RGB"), dtype=np.uint8)
                    frame = np.vstack([frame, goal])
                o.append_data(frame)

        [o.close() for o in out]
        print(f"Video saved to {video_path}")

    def evaluate(self, episodes=10, eval_keys=None, seed=None, options=None, dump_every=-1):
        """Evaluate the current policy over multiple episodes and return comprehensive results.

        Runs the attached policy for a specified number of episodes, tracking success
        rates and optionally other results from environment info. Handles episode
        boundaries and ensures reproducibility through seeding.

        Args:
            episodes (int, optional): Total number of episodes to evaluate. More
                episodes provide more statistically reliable results. Defaults to 10.
            eval_keys (list of str, optional): Additional info keys to track across
                episodes. Must be keys present in self.infos (e.g., 'reward_total',
                'steps_to_success'). Defaults to None (track only success rate).
            seed (int, optional): Base random seed for reproducible evaluation. Each
                episode gets an incremental offset. Defaults to None (non-deterministic).
            options (dict, optional): Reset options passed to environments (e.g.,
                task selection, variation settings). Defaults to None.
            dump_every (int, optional): Frequency of logging intermediate results into tmp file. Defaults to -1 (disabled).

        Returns:
            dict: Dictionary containing evaluation results:
                - 'success_rate' (float): Percentage of successful episodes (0-100)
                - 'episode_successes' (ndarray): Boolean array of episode outcomes
                - 'seeds' (ndarray): Random seeds used for each episode
                - Additional keys from eval_keys if specified (ndarray)

        Raises:
            AssertionError: If eval_key is not found in infos, if episode count
                mismatch occurs, or if duplicate seeds are detected.

        Note:
            - Success is determined by the 'terminateds' flag (True = success)
            - 'truncateds' flag (timeout) is treated as failure
            - Seeds are validated for uniqueness to ensure independent episodes
            - Environments are manually reset with seeds (autoreset is bypassed)
            - All environments are evaluated in parallel for efficiency

        Example:
            Basic evaluation::

                results = world.evaluate(episodes=100, seed=42)
                print(f"Success: {results['success_rate']:.1f}%")

            Evaluate with additional results::

                results = world.evaluate(
                    episodes=100,
                    eval_keys=['reward_total', 'episode_length'],
                    seed=42,
                    options={'task_id': 3}
                )
                print(f"Success: {results['success_rate']:.1f}%")
                print(f"Avg reward: {results['reward_total'].mean():.2f}")

            Evaluate across different variations::

                for var_type in ['none', 'color', 'light', 'all']:
                    options = {'variation': [var_type]} if var_type != 'none' else None
                    results = world.evaluate(episodes=50, seed=42, options=options)
                    print(f"{var_type}: {results['success_rate']:.1f}%")
        """

        options = options or {}

        results = {
            "episode_count": 0,
            "success_rate": 0,
            "episode_successes": np.zeros(episodes),
            "seeds": np.zeros(episodes, dtype=np.int32),
        }

        if eval_keys:
            for key in eval_keys:
                results[key] = np.zeros(episodes)

        self.terminateds = np.zeros(self.num_envs)
        self.truncateds = np.zeros(self.num_envs)

        episode_idx = np.arange(self.num_envs)
        self.reset(seed=seed, options=options)
        root_seed = seed + self.num_envs if seed is not None else None

        eval_done = False

        # determine "unique" hash for this eval run
        config = {
            "episodes": episodes,
            "eval_keys": tuple(sorted(eval_keys)) if eval_keys else None,
            "seed": seed,
            "options": tuple(sorted(options.items())) if options else None,
            "dump_every": dump_every,
        }

        config_str = json.dumps(config, sort_keys=True)
        run_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
        run_tmp_path = Path(f"eval_tmp_{run_hash}.npy")

        # load back intermediate results if file exists
        if run_tmp_path.exists():
            tmp_results = np.load(run_tmp_path, allow_pickle=True).item()
            results.update(tmp_results)

            ep_count = results["episode_count"]
            episode_idx = np.arange(ep_count, ep_count + self.num_envs)

            # reset seed where we left off
            last_seed = seed + ep_count if seed is not None else None
            self.reset(seed=last_seed, options=options)

            logging.success(
                f"Found existing eval tmp file {run_tmp_path}, resuming from episode {ep_count}/{episodes}"
            )

        while True:
            self.step()

            # start new episode for done envs
            for i in range(self.num_envs):
                if self.terminateds[i] or self.truncateds[i]:
                    # record eval info
                    ep_idx = episode_idx[i]
                    results["episode_successes"][ep_idx] = self.terminateds[i]
                    results["seeds"][ep_idx] = self.envs.envs[i].unwrapped.np_random_seed

                    if eval_keys:
                        for key in eval_keys:
                            assert key in self.infos, f"key {key} not found in infos"
                            results[key][ep_idx] = self.infos[key][i]

                    # determine new episode idx
                    # re-reset env with seed and options (no supported by auto-reset)
                    new_seed = root_seed + results["episode_count"] if seed is not None else None
                    next_ep_idx = episode_idx.max() + 1
                    episode_idx[i] = next_ep_idx
                    results["episode_count"] += 1

                    # break if enough episodes evaluated
                    if results["episode_count"] >= episodes:
                        eval_done = True
                        if run_tmp_path.exists():
                            logging.info(f"Eval done, deleting tmp file {run_tmp_path}")
                            os.remove(run_tmp_path)
                        break

                    # dump temporary results in a file
                    if dump_every > 0 and (results["episode_count"] % dump_every == 0):
                        np.save(run_tmp_path, results)
                        logging.success(
                            f"Dumped intermediate eval results to {run_tmp_path} ({results['episode_count']}/{episodes})"
                        )
                    self.envs.unwrapped._autoreset_envs = np.zeros((self.num_envs,))
                    _, infos = self.envs.envs[i].reset(seed=new_seed, options=options)

                    for k, v in infos.items():
                        if k not in self.infos:
                            continue
                        # Convert to array and extract scalar to preserve dtype
                        self.infos[k][i] = np.asarray(v)

            if eval_done:
                break

        # compute success rate
        results["success_rate"] = float(np.sum(results["episode_successes"])) / episodes * 100.0

        assert results["episode_count"] == episodes, f"episode_count {results['episode_count']} != episodes {episodes}"

        assert np.unique(results["seeds"]).shape[0] == episodes, "Some episode seeds are identical!"

        return results

    def evaluate_from_dataset(
        self,
        dataset: SWMDataset,
        episodes_idx: int | list[int],
        start_steps: int | list[int],
        goal_offset_steps: int,
        eval_budget: int,
        callables: dict | None = None,
    ):
        assert (
            self.envs.envs[0].spec.max_episode_steps is None
            or self.envs.envs[0].spec.max_episode_steps >= goal_offset_steps
        ), "env max_episode_steps must be greater than eval_budget"

        episodes_idx = np.array(episodes_idx)
        start_steps = np.array(start_steps)
        end_steps = start_steps + goal_offset_steps

        if not (len(episodes_idx) == len(start_steps)):
            raise ValueError("episodes_idx and start_steps must have the same length")

        if len(episodes_idx) != self.num_envs:
            raise ValueError("Number of episodes to evaluate must match number of envs")

        data = dataset.load_chunk(episodes_idx, start_steps, end_steps)
        columns = dataset.dataset.column_names

        # keep relevant part of the chunk
        init_step_per_env = {c: [] for c in columns}
        goal_step_per_env = {c: [] for c in columns}

        for i, ep in enumerate(data):
            for col in columns:
                if col.startswith("goal"):
                    continue
                if col.startswith("pixels"):
                    # permute channel to be last
                    ep[col] = ep[col].permute(0, 2, 3, 1)

                init_data = ep[col][0]
                goal_data = ep[col][-1]

                init_data = init_data.numpy() if isinstance(init_data, torch.Tensor) else init_data
                goal_data = goal_data.numpy() if isinstance(goal_data, torch.Tensor) else goal_data

                init_step_per_env[col].append(init_data)
                goal_step_per_env[col].append(goal_data)

        init_step = {k: np.stack(v) for k, v in deepcopy(init_step_per_env).items()}

        goal_step = {}
        for key, value in goal_step_per_env.items():
            key = "goal" if key == "pixels" else f"goal_{key}"
            goal_step[key] = np.stack(value)

        # get dataset info
        seeds = init_step.get("seed")
        # get dataset variation
        vkey = "variation."
        variations = [col.removeprefix(vkey) for col in columns if col.startswith(vkey)]
        options = {"variations": variations or None}

        init_step.update(deepcopy(goal_step))
        self.reset(seed=seeds, options=options)  # set seeds for all envs

        init_step = {k: v for k, v in init_step.items() if k in self.infos}
        goal_step = {k: v for k, v in goal_step.items() if k in self.infos}

        # apply callable list (e.g used for set initial position if not access to seed)
        callables = callables or {}
        for i, env in enumerate(self.envs.unwrapped.envs):
            env = env.unwrapped
            for method_name, col_name in callables.items():
                if not hasattr(env, method_name):
                    logging.warning(f"Env {env} has no method {method_name}, skipping callable")
                    continue

                if col_name not in init_step:
                    logging.warning(f"Column {col_name} not found in dataset, skipping callable for env {env}")
                    continue

                method = getattr(env, method_name)
                data = deepcopy(init_step[col_name][i])
                method(data)

        for i, env in enumerate(self.envs.unwrapped.envs):
            env = env.unwrapped

            # assert np.allclose(init_step["state"][i], env._get_obs()), "State info does not match at reset"
            # assert np.array_equal(init_step["goal_state"][i], goal_step["goal_state"][i]), (
            #     "Goal state info does not match at reset"
            # )

            if "goal_state" in init_step and "goal_state" in goal_step:
                assert np.array_equal(init_step["goal_state"][i], goal_step["goal_state"][i]), (
                    "Goal state info does not match at reset"
                )
        results = {
            "success_rate": 0,
            "episode_successes": np.zeros(len(episodes_idx)),
            "seeds": seeds,
        }

        # broadcast info dict from dataset to match envs infos shape
        goal_step = {
            k: (np.broadcast_to(v[:, None, ...], self.infos[k].shape) if k in self.infos else v)
            for k, v in goal_step.items()
        }

        # TODO get the data from the previous step in the dataset for history
        init_step = {
            k: (np.broadcast_to(v[:, None, ...], self.infos[k].shape) if k in self.infos else v)
            for k, v in init_step.items()
        }

        # update the reset with our new init and goal infos
        self.infos.update(deepcopy(init_step))
        self.infos.update(deepcopy(goal_step))

        # assert np.allclose(self.infos["goal"], goal_step["goal"]), "Goal info does not match"
        if "goal" in goal_step and "goal" in self.infos:
            assert np.allclose(self.infos["goal"], goal_step["goal"]), "Goal info does not match"

        # TODO assert goal and start state are identical as in the rollout
        # run normal evaluation for eval_budget and TODO: record video
        for _ in range(eval_budget):
            self.infos.update(deepcopy(goal_step))
            self.step()
            results["episode_successes"] = np.logical_or(results["episode_successes"], self.terminateds)
            # for auto-reset
            self.envs.unwrapped._autoreset_envs = np.zeros((self.num_envs,))

        n_episodes = len(episodes_idx)

        # compute success rate
        results["success_rate"] = float(np.sum(results["episode_successes"])) / n_episodes * 100.0

        if results["seeds"] is not None:
            assert np.unique(results["seeds"]).shape[0] == n_episodes, "Some episode seeds are identical!"

        return results
