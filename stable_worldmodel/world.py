from collections.abc import Callable
from pathlib import Path

import datasets
import gymnasium as gym
import imageio.v3 as iio
import numpy as np
from datasets import Dataset, Features, Value, load_dataset
from loguru import logger as logging
from rich import print

import stable_worldmodel as swm
from stable_worldmodel.data import is_image

from .wrappers import MegaWrapper, VariationWrapper


class World:
    """A high-level environment manager for vectorized gym environments with support for"""

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        image_shape: tuple,
        goal_shape: tuple | None = None,
        goal_transform: Callable | None = None,
        image_transform: Callable | None = None,
        seed: int = 2349867,
        max_episode_steps: int = 100,
        verbose: int = 1,
        **kwargs,
    ):
        self.envs = gym.make_vec(
            env_name,
            num_envs=num_envs,
            vectorization_mode="sync",
            wrappers=[lambda x: MegaWrapper(x, image_shape, image_transform, goal_shape, goal_transform)],
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

        self.envs = VariationWrapper(self.envs)

        if verbose > 0:
            logging.info(f"🌍🌍🌍 World {env_name} initialized 🌍🌍🌍")

            logging.info("🕹️ 🕹️ 🕹️ Action space 🕹️ 🕹️ 🕹️")
            logging.info(f"{self.envs.action_space}")

            logging.info("👁️ 👁️ 👁️ Observation space 👁️ 👁️ 👁️")
            logging.info(f"{self.envs.observation_space}")

            if self.envs.variation_space is not None:
                logging.info("⚗️ ⚗️ ⚗️ Variation space ⚗️ ⚗️ ⚗️")
                print(self.envs.variation_space)
            else:
                logging.warning("No variation space provided!")

        self.seed = seed

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
    def variation_space(self):
        return self.envs.variation_space

    @property
    def single_variation_space(self):
        return self.envs.single_variation_space

    @property
    def single_action_space(self):
        return self.envs.single_action_space

    @property
    def single_observation_space(self):
        return self.envs.single_observation_space

    def close(self, **kwargs):
        return self.envs.close(**kwargs)

    def step(self):
        """Advance all environments by one step using the current policy."""
        # note: reset happens before because of auto-reset, should fix that
        actions = self.policy.get_action(self.infos)
        (self.states, self.rewards, self.terminateds, self.truncateds, self.infos) = self.envs.step(actions)

    def reset(self, seed=None, options=None):
        """Reset all environments."""
        self.states, self.infos = self.envs.reset(seed=seed, options=options)

    def set_policy(self, policy):
        """Attach a policy to the world and provide it the env context."""
        self.policy = policy
        self.policy.set_env(self.envs)

        if hasattr(self.policy, "seed") and self.policy.seed is not None:
            self.policy.set_seed(self.policy.seed)

    def record_video(self, video_path, max_steps=500, fps=30, seed=None, options=None):
        """Record rollout videos for each environment under the current policy."""
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

    def record_dataset(self, dataset_name, episodes=10, seed=None, cache_dir=None, options=None):
        """Collect episodes with the current policy and save them as a HuggingFace Dataset (Parquet shards)."""
        cache_dir = cache_dir or swm.data.get_cache_dir()
        dataset_path = Path(cache_dir, dataset_name)
        dataset_path.mkdir(parents=True, exist_ok=True)

        recorded_episodes = 0

        self.terminateds = np.zeros(self.num_envs)
        self.truncateds = np.zeros(self.num_envs)
        episode_idx = np.arange(self.num_envs)

        self.reset(seed, options)  # <- incr global seed by num_envs
        root_seed = seed + self.num_envs if seed is not None else None

        records = {key: list(value) for key, value in self.infos.items() if key[0] != "_"}

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
                    # re-reset env with seed and options (no supported by auto-reset)
                    new_seed = root_seed + recorded_episodes if seed is not None else None

                    # determine new episode idx
                    next_ep_idx = episode_idx.max() + 1
                    episode_idx[i] = next_ep_idx
                    recorded_episodes += 1

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

                    # override last actions of continuing episodes
                    records[key][-n_action:] = np.where(
                        action_mask,
                        self.infos[key],
                        np.nan,
                    )

                    # add new dummy action
                    action_shape = np.shape(self.infos[key][0])
                    action_dtype = self.single_action_space.dtype
                    dummy_block = [np.full(action_shape, np.nan, dtype=action_dtype) for _ in range(self.num_envs)]
                    records[key].extend(dummy_block)
                else:
                    records[key].extend(list(self.infos[key]))

            records["episode_idx"].extend(list(episode_idx))
            records["policy"].extend([self.policy.type] * self.num_envs)

        # add the episode length
        counts = np.bincount(np.array(records["episode_idx"]), minlength=max(records["episode_idx"]) + 1)
        records["episode_len"] = [int(counts[ep]) for ep in records["episode_idx"]]

        ########################
        # Save dataset to disk #
        ########################

        assert "pixels" in records, "pixels key is required in records"
        assert "episode_idx" in records, "episode_idx key is required in records"
        assert "step_idx" in records, "step_idx key is required in records"
        assert "episode_len" in records, "episode_len key is required in records"

        # Create the dataset directory structure
        dataset_path.mkdir(parents=True, exist_ok=True)

        # save all jpeg images
        image_cols = {col for col in records if is_image(records[col][0])}

        # pre-create all directories
        for ep_idx in set(records["episode_idx"]):
            img_folder = dataset_path / "img" / f"{ep_idx}"
            img_folder.mkdir(parents=True, exist_ok=True)

        # dump all data
        for i in range(len(records["episode_idx"])):
            ep_idx = records["episode_idx"][i]
            step_idx = records["step_idx"][i]
            for img_col in image_cols:
                img = records[img_col][i]
                img_folder = dataset_path / "img" / f"{ep_idx}"
                img_path = img_folder / f"{step_idx}_{img_col}.jpeg"
                iio.imwrite(img_path, img)

                # replace image in records with relative path
                records[img_col][i] = str(img_path.relative_to(dataset_path))

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

        # flush incomplete episodes
        # get episodes that are currently running (not done)
        incomplete_episodes = episode_idx[~(self.terminateds | self.truncateds)]
        # keep only episodes that are NOT in the incomplete list
        keep_mask = ~np.isin(records_ds["episode_idx"], incomplete_episodes)
        records_ds = records_ds.select(np.nonzero(keep_mask)[0])

        # flush all extra episodes saved (keep only first N episodes)
        episodes_to_keep = np.unique(records_ds["episode_idx"])[:episodes]
        keep_mask = np.isin(records_ds["episode_idx"], episodes_to_keep)
        records_ds = records_ds.select(np.nonzero(keep_mask)[0])

        # save dataset
        records_path = dataset_path / "records"
        num_chunks = episodes // 50
        records_path.mkdir(parents=True, exist_ok=True)
        records_ds.save_to_disk(records_path, num_shards=num_chunks or 1)

        print(f"Dataset saved to {dataset_path} with {episodes} episodes!")

    def record_video_from_dataset(
        self,
        video_path,
        dataset_name,
        episode_idx,
        max_steps=500,
        fps=30,
        num_proc=4,
        cache_dir=None,
    ):
        """Replay stored dataset episodes and export them as MP4 videos."""
        import imageio

        cache_dir = cache_dir or swm.data.get_cache_dir()
        dataset_path = Path(cache_dir, dataset_name)
        assert dataset_path.is_dir(), f"Dataset {dataset_name} not found in cache dir {swm.data.get_cache_dir()}"

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

        # TODO: should load from disk directly
        dataset = load_dataset("parquet", data_files=str(Path(dataset_path, "*.parquet")), split="train")

        for i, o in zip(episode_idx, out):
            episode = dataset.filter(lambda ex: ex["episode_idx"] == i, num_proc=num_proc)
            episode = episode.sort("step_idx")
            episode_len = len(episode)

            assert len(set(episode["episode_len"])) == 1, (
                "'episode_len' contains different values for the same episode"
            )
            assert len(episode) == episode["episode_len"][0], (
                f"Episode {i} has {len(episode)} steps, but 'episode_len' is {episode['episode_len'][0]}"
            )

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
        """Evaluate the current policy over a number of episodes and return metrics."""
        metrics = {
            "success_rate": 0,
            "episode_successes": np.zeros(episodes),
            "seeds": np.empty(episodes, dtype=int),
        }

        if eval_keys:
            for key in eval_keys:
                metrics[key] = np.zeros(episodes)

        self.terminateds = np.zeros(self.num_envs)
        self.truncateds = np.zeros(self.num_envs)

        episode_idx = np.arange(self.num_envs)
        self.reset(seed, options)
        root_seed = seed + self.num_envs if seed is not None else None

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
                    ep_idx = episode_idx[i] - 1
                    metrics["episode_successes"][ep_idx] = terminations_before[i]
                    metrics["seeds"][ep_idx] = self.envs.envs[i].unwrapped.np_random_seed

                    logging.error(
                        "EXTRACTED SEED : ", metrics["seeds"][ep_idx], self.envs.envs[i].unwrapped.np_random_seed
                    )

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
                    new_seed = root_seed + eval_ep_count if seed is not None else None

                    _, infos = self.envs.envs[i].reset(seed=new_seed, options=options)

                    for k, v in infos.items():
                        self.infos[k][i] = v

            if eval_done:
                break

        # compute success rate
        metrics["success_rate"] = float(np.sum(metrics["episode_successes"])) / episodes * 100.0

        assert eval_ep_count == episodes, f"eval_ep_count {eval_ep_count} != episodes {episodes}"

        assert np.unique(metrics["seeds"]).shape[0] == episodes, "Some episode seeds are identical!"

        return metrics
