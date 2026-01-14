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

import gymnasium as gym
import h5py
import hdf5plugin
import imageio
import imageio.v3 as iio
import numpy as np
import torch
from loguru import logger as logging
from PIL import Image
from rich import print
from tqdm import tqdm

from stable_worldmodel.data.dataset import Dataset as SWMDataset
from stable_worldmodel.data.dataset import VideoDataset
from stable_worldmodel.data.utils import get_cache_dir

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

    def record_dataset(
        self,
        dataset_name: str,
        episodes: int = 10,
        seed: int | None = None,
        cache_dir: os.PathLike | None = None,
        options: dict | None = None,
    ):
        """Records episodes from the environment into an HDF5 dataset."""
        if self._history_size > 1:
            raise NotImplementedError("Frame history > 1 not supported for dataset recording.")

        path = Path(cache_dir or get_cache_dir()) / f"{dataset_name}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)

        self.terminateds = np.zeros(self.num_envs, dtype=bool)
        self.truncateds = np.zeros(self.num_envs, dtype=bool)

        episode_buffers = [defaultdict(list) for _ in range(self.num_envs)]

        h5_kwargs = {
            "name": str(path),
            "mode": "a" if path.exists() else "w",
            "libver": "latest",
        }

        if not path.exists():  # creation only args
            h5_kwargs.update({"fs_strategy": "page", "fs_page_size": 4 * 1024 * 1024})

        with h5py.File(**h5_kwargs) as f:
            f.swmr_mode = True  # avoid issue when killed

            if "ep_len" in f:
                n_ep_recorded = f["ep_len"].shape[0]
                global_step_ptr = f["ep_offset"][-1] + f["ep_len"][-1] if n_ep_recorded > 0 else 0
                initialized = True
                seed = None if seed is None else (seed + n_ep_recorded)
                logging.info(f"Resuming: {n_ep_recorded} episodes already on disk.")
            else:
                n_ep_recorded = 0
                global_step_ptr = 0
                initialized = False

            self.reset(seed)
            seed = None if seed is None else (seed + self.num_envs)
            self._dump_step_data(episode_buffers)  # record initial state

            with tqdm(total=episodes, initial=n_ep_recorded, desc="Recording") as pbar:
                while n_ep_recorded < episodes:
                    self.step()
                    self._dump_step_data(episode_buffers)

                    for i in range(self.num_envs):
                        if self.terminateds[i] or self.truncateds[i]:
                            finished_ep = self._handle_done_ep(episode_buffers, i, n_ep_recorded)

                            # lazy dataset initialization
                            if not initialized:
                                self._init_h5_datasets(f, finished_ep)
                                initialized = True

                            # contiguous writing
                            steps_written = self._write_episode(f, finished_ep, global_step_ptr)
                            global_step_ptr += steps_written
                            n_ep_recorded += 1
                            pbar.update(1)

                            f.flush()  # flush metadata to avoid corruption

                            if n_ep_recorded >= episodes:
                                break

                            # reset terminated env and record initial state
                            self._reset_single_env(i, seed + n_ep_recorded, options)
                            self._dump_step_data(episode_buffers, env_idx=i)

        logging.info(f"Recording complete. Total frames: {global_step_ptr}")

    def _init_h5_datasets(self, f, sample_episode):
        """Initializes resizable datasets based on the first recorded episode."""
        for key, data_list in sample_episode.items():
            if key in ["ep_len", "ep_idx", "policy"]:
                continue

            # determine array shape and dtype from sample data
            sample_data = np.array(data_list[0])
            shape = (0,) + sample_data.shape
            maxshape = (None,) + sample_data.shape

            # determine chunk size and compression
            if sample_data.ndim >= 2:
                chunks = (100,) + sample_data.shape
                compression = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)

            else:
                chunks = (1000,) + sample_data.shape
                compression = None

            f.create_dataset(
                key,
                shape=shape,
                maxshape=maxshape,
                dtype=sample_data.dtype,
                chunks=chunks,
                compression=compression,
            )

        # index metadata
        f.create_dataset("ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64)
        f.create_dataset("ep_len", shape=(0,), maxshape=(None,), dtype=np.int32)

    def _reset_single_env(self, env_idx, seed=None, options=None):
        """Resets a single environment and updates infos dict."""
        self.envs.unwrapped._autoreset_envs = np.zeros(self.num_envs)
        _, infos = self.envs.envs[env_idx].reset(seed=seed, options=options)

        for k, v in infos.items():
            self.infos[k][env_idx] = v

    def _handle_done_ep(self, tmp_buffer, env_idx, n_ep_recorded):
        """Prepares the episode buffer for writing (alignment and casting)."""
        ep_buffer = tmp_buffer[env_idx]

        # left-shift actions to align with observations i.e. (o_t, a_t)
        if "action" in ep_buffer:
            actions = ep_buffer["action"]
            nan = actions.pop(0)
            actions.append(nan)

        # Extract a copy and clear the temporary buffer
        out = {k: list(v) for k, v in ep_buffer.items()}
        ep_buffer.clear()
        self.terminateds[env_idx] = False
        self.truncateds[env_idx] = False
        return out

    def _write_episode(self, f, ep_data, global_ptr):
        """Writes a single contiguous episode to the HDF5 file."""
        ep_len = len(ep_data["step_idx"])

        # append data to each dataset
        for key in f.keys():
            if key in ["ep_offset", "ep_len"]:
                continue

            ds = f[key]
            curr_size = ds.shape[0]
            ds.resize(curr_size + ep_len, axis=0)
            ds[curr_size:] = np.array(ep_data[key])

        # update metadata
        meta_idx = f["ep_offset"].shape[0]
        f["ep_offset"].resize(meta_idx + 1, axis=0)
        f["ep_len"].resize(meta_idx + 1, axis=0)

        f["ep_offset"][meta_idx] = global_ptr
        f["ep_len"][meta_idx] = ep_len

        return ep_len

    def _dump_step_data(self, tmp_buffer, env_idx=None):
        """Append currefnt step data to tmp episode buffers."""
        env_indices = range(self.num_envs) if env_idx is None else [env_idx]

        for col, data in self.infos.items():
            if col.startswith("_"):
                continue

            # normalize data shape and type
            if isinstance(data, np.ndarray):
                data = np.squeeze(data, axis=1) if data.ndim > 1 and data.shape[1] == 1 else data
                if data.dtype == object:
                    data = np.concatenate(data).tolist()

            # append to buffers
            for i in env_indices:
                env_data = data[i].copy() if isinstance(data[i], np.ndarray) else data[i]
                tmp_buffer[i][col].append(env_data)

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
        save_video: bool = True,
        video_path="./",
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
        init_step_per_env = defaultdict(list)
        goal_step_per_env = defaultdict(list)

        for i, ep in enumerate(data):
            for col in columns:
                if col.startswith("goal"):
                    continue
                if col.startswith("pixels"):
                    # permute channel to be last
                    ep[col] = ep[col].permute(0, 2, 3, 1)

                if not isinstance(ep[col], (torch.Tensor | np.ndarray)):
                    continue

                init_data = ep[col][0]
                goal_data = ep[col][-1]

                # TODO handle that better
                if not isinstance(init_data, (np.ndarray | torch.Tensor)):
                    logging.warning(
                        f"Data type {type(init_data)} for column {col} not supported, yet skipping conversion"
                    )
                    continue

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

        # init_step = {k: v for k, v in init_step.items() if k in self.infos}
        # goal_step = {k: v for k, v in goal_step.items() if k in self.infos}

        # apply callable list (e.g used for set initial position if not access to seed)
        callables = callables or {}
        for i, env in enumerate(self.envs.unwrapped.envs):
            env = env.unwrapped

            for spec in callables:
                method_name = spec["method"]
                if not hasattr(env, method_name):
                    logging.warning(f"Env {env} has no method {method_name}, skipping callable")
                    continue

                method = getattr(env, method_name)
                args = spec.get("args", spec)

                # prepare args
                prepared_args = {}
                for args_name, args_data in args.items():
                    value = args_data.get("value", None)
                    is_in_datset = args_data.get("in_dataset", True)

                    if is_in_datset:
                        if value not in init_step:
                            logging.warning(f"Col {value} not found in dataset, skipping callable for env {env}")
                            continue
                        prepared_args[args_name] = deepcopy(init_step[value][i])
                    else:
                        prepared_args[args_name] = args_data.get("value")

                # call method with prepared args
                method(**prepared_args)

        for i, env in enumerate(self.envs.unwrapped.envs):
            env = env.unwrapped

            # TODO remove this
            if "goal_state" in init_step and "goal_state" in goal_step:
                assert np.array_equal(init_step["goal_state"][i], goal_step["goal_state"][i]), (
                    "Goal state info does not match at reset"
                )

        results = {
            "success_rate": 0,
            "episode_successes": np.zeros(len(episodes_idx)),
            "seeds": seeds,
        }

        # expend all data to the right shape (x, y, (original_shape))
        shape_prefix = next(iter(self.infos.values())).shape[:2]

        # TODO get the data from the previous step in the dataset for history
        init_step = {k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:]) for k, v in init_step.items()}
        goal_step = {k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:]) for k, v in goal_step.items()}

        # update the reset with our new init and goal infos
        self.infos.update(deepcopy(init_step))
        self.infos.update(deepcopy(goal_step))

        # assert np.allclose(self.infos["goal"], goal_step["goal"]), "Goal info does not match"
        if "goal" in goal_step and "goal" in self.infos:
            assert np.allclose(self.infos["goal"], goal_step["goal"]), "Goal info does not match"

        target_frames = torch.stack([ep["pixels"] for ep in data]).numpy()
        video_frames = np.empty((self.num_envs, eval_budget, *self.infos["pixels"].shape[-3:]), dtype=np.uint8)

        # TODO assert goal and start state are identical as in the rollout
        # run normal evaluation for eval_budget and TODO: record video
        for i in range(eval_budget):
            video_frames[:, i] = self.infos["pixels"].squeeze(1)
            self.infos.update(deepcopy(goal_step))
            self.step()
            results["episode_successes"] = np.logical_or(results["episode_successes"], self.terminateds)
            # for auto-reset
            self.envs.unwrapped._autoreset_envs = np.zeros((self.num_envs,))

        video_frames[:, -1] = self.infos["pixels"].squeeze(1)

        n_episodes = len(episodes_idx)

        # compute success rate
        results["success_rate"] = float(np.sum(results["episode_successes"])) / n_episodes * 100.0

        # save video if required
        if save_video:
            target_len = target_frames.shape[1]
            video_path = Path(video_path)
            video_path.mkdir(parents=True, exist_ok=True)
            for i in range(self.num_envs):
                out = imageio.get_writer(video_path / f"rollout_{i}.mp4", "output.mp4", fps=15, codec="libx264")
                goals = np.vstack([target_frames[i, -1], target_frames[i, -1]])
                for t in range(eval_budget):
                    stacked_frame = np.vstack([video_frames[i, t], target_frames[i, t % target_len]])
                    frame = np.hstack([stacked_frame, goals])
                    out.append_data(frame)
                out.close()
            print(f"Video saved to {video_path}")

        if results["seeds"] is not None:
            assert np.unique(results["seeds"]).shape[0] == n_episodes, "Some episode seeds are identical!"

        return results
