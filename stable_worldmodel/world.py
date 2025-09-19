from typing import Callable

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

import stable_worldmodel as swm

class World:

    """
    A high-level environment manager for vectorized gym environments with support for
    goal-conditioned tasks, video recording, dataset generation, and evaluation.

    The `World` class wraps multiple parallel environments using Gymnasium’s vectorized API and
    integrates with custom wrappers (e.g., `MegaWrapper`) to handle image-based observations and goals.
    It provides utility methods for recording trajectories, saving datasets in HuggingFace Datasets .parquet format,
    evaluating policies, and exporting video rollouts.

    Parameters
    ----------
    env_name : str
        The name of the Gymnasium environment to instantiate.
    num_envs : int
        Number of parallel environments to run.
    image_shape : tuple[int]
        Shape of image observations (C, H, W).
    image_transform : callable, optional
        Transformation function applied to image observations.
    goal_shape : tuple[int], optional
        Shape of goal image observations.
    goal_transform : callable, optional
        Transformation function applied to goal observations.
    seed : int, default=2349867
        Random seed for environment initialization and goal sampling.
    max_episode_steps : int, default=100
        Maximum number of steps per episode.
    **kwargs : dict
        Additional keyword arguments passed to `gym.make_vec`.

    Attributes
    ----------
    envs : gym.vector.VectorEnv
        Vectorized environments wrapped with `MegaWrapper`.
    goal_envs : gym.vector.VectorEnv
        Separate vectorized environments for goal sampling.
    seed : int
        Random seed for reproducibility.
    goal_seed : int
        Secondary random seed for sampling goals.
    policy : object
        Policy instance with `get_action` and `set_env` methods.

    Properties
    ----------
    num_envs : int
        Number of environments in the vectorized setup.
    observation_space : gym.Space
        Observation space of the environments.
    action_space : gym.Space
        Action space of the environments.
    single_action_space : gym.Space
        Action space for a single environment.
    single_observation_space : gym.Space
        Observation space for a single environment.

    Methods
    -------
    close(**kwargs)
        Closes all managed environments.
    denormalize(x)
        Converts normalized images in [-1, 1] back to [0, 1].
    set_policy(policy)
        Assigns a policy object for interaction.
    reset(seed=None, options=None)
        Resets the environments and returns initial states and infos.
    step()
        Steps all environments using the current policy’s actions.
    __iter__()
        Prepares the iterator by resetting environments.
    __next__()
        Iterates over environment states until all episodes are done.
    record_video(video_path, max_steps=500, fps=30, seed=None, options=None)
        Records a rollout video for each environment.
    record_dataset(dataset_name, episodes=10, seed=None, options=None)
        Collects episodes into a HuggingFace Dataset and saves as Parquet shards.
    record_video_from_dataset(video_path, dataset_name, episode_idx, max_steps=500, fps=30, num_proc=4)
        Replays stored dataset episodes and exports them as videos.
    evaluate(episodes=10, eval_keys=None, seed=None, options=None)
        Evaluates the policy across multiple episodes and computes metrics.

    Notes
    -----
    - Supports parallelized rollout collection with episode tracking.
    - Integrates tightly with HuggingFace Datasets for storage and replay.
    - Provides success-rate evaluation with optional custom metrics.
    - Video export uses `imageio` with `libx264` encoding.
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        image_shape: tuple,
        goal_shape: tuple = None,
        goal_transform: Callable = None,
        image_transform: Callable = None,
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

    def __iter__(self):
        """Prepare the world to iterate over steps (Python iterator protocol).

        Initializes episode bookkeeping, resets all environments, and caches
        initial infos (including current goals if provided by the wrapper).
    
        Returns
        -------
        World
            The iterator (``self``).
        """
        self.terminations = np.array([False] * self.num_envs)
        self.truncations = np.array([False] * self.num_envs)
        self.rewards = None
        logging.info(f"Resetting the ({self.num_envs}) world(s)!")
        self.states, self.infos = self.envs.reset(seed=self.seed)
        self.cur_goals = self.infos["goal"]
        self.cur_goal_images = self.infos["goal_image"]
        return self

    def __next__(self):
        """Return the current states and infos until all episodes terminate or truncate."""
        if not all(self.terminations) and not all(self.truncations):
            return self.states, self.infos
        else:
            raise StopIteration

    def step(self):
        """Advance all environments by one step using the current policy."""
        actions = self.policy.get_action(self.infos)
        (self.states, self.rewards, self.terminateds, self.truncateds, self.infos) = (
            self.envs.step(actions)
        )

    def reset(self, seed=None, options=None):
        """Reset all environments."""
        self.states, self.infos = self.envs.reset(seed=seed, options=options)

    def set_policy(self, policy):
        """Attach a policy to the world and provide it the env context."""
        self.policy = policy
        self.policy.set_env(self.envs)

    def record_video(self, video_path, max_steps=500, fps=30, seed=None, options=None):
        """
        Record rollout videos for each environment under the current policy.

        Parameters
        ----------
        video_path : str or Path
            Directory to which MP4 files will be written; one file per env named ``env_{i}.mp4``.
        max_steps : int, default=500
            Maximum number of steps to record per environment.
        fps : int, default=30
            Frames per second for the output video.
        seed : int, optional
            Seed to use for the initial reset prior to recording.
        options : dict, optional
            Env reset options.

        Notes
        -----
        - If ``infos`` contains ``"goal"``, the goal image is vertically stacked
          beneath the observation pixels in the output frames.
        - Uses ``imageio.get_writer(..., codec="libx264")`` for MP4 encoding.
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

    def record_dataset(self, dataset_name, episodes=10, seed=None, options=None):
        """
        Collect episodes with the current policy and save them as a HuggingFace Dataset (Parquet shards).

        Parameters
        ----------
        dataset_name : str
            Name of the dataset directory under the stable_worldmodel cache dir.
        episodes : int, default=10
            Number of *complete* episodes to record (across all envs, total).
        seed : int, optional
            Base seed used for the initial reset; subsequent resets for per-env continuation
            derive unique seeds from this value.
        options : dict, optional
            Env reset options.
        
        Side Effects
        ------------
        - Creates ``{cache_dir}/{dataset_name}/data_shard_{NNNNN}.parquet`` files.
        - Appends only *complete* episodes to the shard (partial episodes are dropped).
        - Re-indexes ``episode_idx`` to be contiguous across shards.

        Dataset Schema
        --------------
        The resulting dataset includes (at minimum):
            - ``pixels`` : :class:`datasets.Image`
            - ``episode_idx`` : :class:`datasets.Value("int32")`
            - ``step_idx`` : :class:`datasets.Value("int32")`
            - ``episode_len`` : :class:`datasets.Value("int32")`

        If available, it may also include:
            - ``goal`` : :class:`datasets.Image`
            - ``action`` : array-like feature shaped like the env action space
            - Per-step scalar/vector fields exposed by the wrapper in ``infos``

        Notes
        -----
        - The function handles action shifting to ensure alignment with continuing episodes.
        - Sharding logic ensures a new shard index is chosen if prior shards exist.
        - Only the first ``episodes`` fully completed episodes are persisted for the new shard.
        """

        dataset_path = Path(swm.utils.get_cache_dir(), dataset_name)
        dataset_path.mkdir(parents=True, exist_ok=True) 

        hf_dataset = None
        recorded_episodes = 0
        dataset_path = Path(dataset_path)
        dataset_path.mkdir(parents=True, exist_ok=True)

        self.terminateds = np.zeros(self.num_envs)
        self.truncateds = np.zeros(self.num_envs)
        episode_idx = np.arange(self.num_envs)

        self.reset(seed, options)

        records = {
            key: [v for v in value]
            for key, value in self.infos.items()
            if key[0] != "_"
        }
        
        records["episode_idx"] = list(episode_idx)
        records["policy"] = [self.policy.type] * self.num_envs
        
        while True:

            # take before step to handle the auto-reset
            truncations_before = self.truncateds.copy()
            terminations_before = self.terminateds.copy()

            # take step
            self.step()

            # start new episode for done envs
            for i in range(self.num_envs):
                if terminations_before[i] or truncations_before[i]:

                    # determine new episode idx
                    next_ep_idx = episode_idx.max() + 1
                    episode_idx[i] = next_ep_idx
                    recorded_episodes += 1

                    # re-reset env with seed and options (no supported by auto-reset)
                    new_seed = seed + i + recorded_episodes if seed is not None else None
                    states, infos = self.envs.envs[i].reset(seed=new_seed, options=options)

                    for k, v in infos.items():
                        self.infos[k][i] = v

            if recorded_episodes >= episodes:
                break

            for key in self.infos:
                if key[0] == "_":
                    continue
                
                # shift actions
                if key == "action":
                    n_action = len(self.infos[key])
                    last_episode = records["episode_idx"][-n_action:]
                    action_mask = (last_episode == episode_idx)[:, None]
               
                    # overwride last actions of continuing episodes
                    records[key][-n_action:] = np.where(
                        action_mask,
                        self.infos[key],
                        np.nan,
                    )

                    # add new dummy action
                    action_shape = np.shape(self.infos[key][0])
                    dummy_block = [np.full(action_shape, np.nan) for _ in range(self.num_envs)]
                    records[key].extend(dummy_block)
                else:
                    records[key].extend(list(self.infos[key]))

            records["episode_idx"].extend(list(episode_idx))
            records["policy"].extend([self.policy.type] * self.num_envs)
        
        # add the episode length
        counts = np.bincount(np.array(records["episode_idx"]), minlength=max(records["episode_idx"])+1)
        records["episode_len"] = [int(counts[ep]) for ep in records["episode_idx"]]

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
                    feature=Value(dtype=records[k][0].dtype.name), length=len(records[k][0])
                )
            elif 2 <= records[k][0].ndim <= 6:
                feature_cls = getattr(datasets, f"Array{records[k][0].ndim}D")
                state_feature = feature_cls(
                    shape=records[k][0].shape, dtype=records[k][0].dtype
                )
            else:
                state_feature = Value(records[k][0].dtype.name)
            features[k] = state_feature

        features = Features(features)
        print(features)

        # make dataset
        hf_dataset = Dataset.from_dict(records, features=features)

        # determine shard index and num episode recorded so far
        shard_idx = 0
        while True:
            if not (dataset_path / f"data_shard_{shard_idx:05d}.parquet").is_file():
                break
            shard_idx += 1

        episode_counter = 0
        if shard_idx > 0:
            shard_dataset = load_dataset(
            "parquet",
            split="train",
            data_files=str(dataset_path / f"data_shard_{shard_idx-1:05d}.parquet")
            )
            episode_counter = np.max(shard_dataset["episode_idx"]) + 1

        # flush incomplete episode
        ep_col = np.array(hf_dataset["episode_idx"])
        non_complete_episodes = np.array(episode_idx[~(self.terminateds | self.truncateds)])
        keep_episode = np.nonzero(~np.isin(ep_col, non_complete_episodes))[0].tolist()
        hf_dataset = hf_dataset.select(keep_episode)

        # re-index remaining episode starting from the last shard episode number
        unique_eps = np.unique(hf_dataset["episode_idx"])
        id_map = {old: new for new, old in enumerate(unique_eps, start=episode_counter)}
        hf_dataset = hf_dataset.map(
            lambda row: {"episode_idx": id_map[row["episode_idx"]]}
        )

        # flush all extra episode saved for the current shard
        hf_dataset = hf_dataset.filter(
            lambda row: (row["episode_idx"]-episode_counter) < episodes
        )

        hf_dataset.to_parquet(dataset_path / f"data_shard_{shard_idx:05d}.parquet")
        
        # display info
        final_num_episodes = np.unique(hf_dataset["episode_idx"])
        episode_range = (final_num_episodes.min().item(), final_num_episodes.max().item())
        print(f"Dataset saved to {shard_idx}th shard file ({dataset_path}) with {len(final_num_episodes)} episodes, range: {episode_range}")

    def record_video_from_dataset(
        self, video_path, dataset_name, episode_idx, max_steps=500, fps=30, num_proc=4
    ):
        """
        Replay stored dataset episodes and export them as MP4 videos.

        Parameters
        ----------
        video_path : str or Path
            Directory to which MP4 files will be written; one file per requested episode
            named ``episode_{idx}.mp4``.
        dataset_name : str
            Name of the dataset directory under the stable_worldmodel cache dir.
        episode_idx : int or list[int]
            Episode index or indices to render from the dataset.
        max_steps : int, default=500
            Maximum number of steps to render per episode (truncates longer episodes).
        fps : int, default=30
            Frames per second for the output video(s).
        num_proc : int, default=4
            Degree of parallelism for dataset filtering.
        """
        import imageio

        dataset_path = Path(swm.utils.get_cache_dir(), dataset_name)
        assert dataset_path.is_dir(), f"Dataset {dataset_name} not found in cache dir {swm.utils.get_cache_dir()}"

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

            assert len(set(episode['episode_len'])) == 1, "'episode_len' contains different values for the same episode"
            assert len(episode) == episode['episode_len'][0], f"Episode {i} has {len(episode)} steps, but 'episode_len' is {episode['episode_len'][0]}"

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

    def evaluate(self, episodes=10, eval_keys=None, seed=None, options=None):
        """
        Evaluate the current policy across multiple episodes and compute metrics.

        Parameters
        ----------
        episodes : int, default=10
            Number of complete episodes to evaluate.
        eval_keys : list[str], optional
            Additional keys in ``infos`` to log as episode-wise metrics. Each key will
            produce a vector of length ``episodes`` in the returned metrics dict.
        seed : int, optional
            Base seed for the initial reset; per-env continuation seeds derive from this value.
        options : dict, optional
            Env reset options.

        Returns
        -------
        dict
            Metrics dictionary with the following entries:
            - ``"success_rate"`` : float
                Percentage of successful episodes (terminations) in [0, 100].
            - ``"episode_successes"`` : np.ndarray, shape (episodes,)
                Boolean array indicating success (True) or failure (False) per episode.
            - ``"seeds"`` : np.ndarray, shape (episodes,)
                The per-episode seeds used for evaluation.
            - Additional arrays for each key in ``eval_keys``, each of shape (episodes,).

        Notes
        -----
        Success is counted using the ``terminated`` signal observed prior to an env's auto-reset.
        """

        metrics = {
            'success_rate': 0,
            'episode_successes': np.zeros(episodes),
            'seeds': np.empty(episodes, dtype=int),
        }

        if eval_keys:
            for key in eval_keys:
                metrics[key] = np.zeros(episodes)

        self.terminateds = np.zeros(self.num_envs)
        self.truncateds = np.zeros(self.num_envs)
        
        episode_idx = np.arange(self.num_envs)
        self.reset(seed, options)
        
        eval_ep_count = 0
        eval_done = False

        while True:

            # take before step to handle the auto-reset
            truncations_before = self.truncateds.copy()
            terminations_before = self.terminateds.copy()

            # take step
            self.step()

            # start new episode for done envs
            for i in range(self.num_envs):
                if terminations_before[i] or truncations_before[i]:

                    # record eval info
                    ep_idx = episode_idx[i]-1
                    metrics['episode_successes'][ep_idx] = terminations_before[i]
                    metrics['seeds'][ep_idx] = self.envs.envs[i].unwrapped.np_random_seed

                    if eval_keys:
                        for key in eval_keys:
                            assert key in self.infos, f"key {key} not found in infos"
                            metrics[key][ep_idx] = self.infos[key][i]

                    # break if enough episodes evaluated
                    if eval_ep_count >= episodes:
                        eval_done = True
                        break

                    # determine new episode idx
                    next_ep_idx = episode_idx.max() + 1
                    episode_idx[i] = next_ep_idx
                    eval_ep_count += 1

                    # re-reset env with seed and options (no supported by auto-reset)
                    new_seed = seed + i + eval_ep_count if seed is not None else None
                    _, infos = self.envs.envs[i].reset(seed=new_seed, options=options)

                    for k, v in infos.items():
                        self.infos[k][i] = v

            if eval_done:
                break
        
        # compute success rate
        metrics['success_rate'] = float(np.sum(metrics['episode_successes'])) / episodes * 100.0

        return metrics

