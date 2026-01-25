import logging
import numbers
import os
from collections.abc import Callable
from pathlib import Path

import decord
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from datasets import concatenate_datasets, load_from_disk
from decord import VideoReader, cpu
from torchvision.io import decode_image

from stable_worldmodel.data.utils import get_cache_dir


def find_shard_dirs(root: str, target_dir: str = "img") -> list[str]:
    """Find all subdirectories named `target_dir` within `root`."""
    result = []
    for current_path, dirs, files in os.walk(root):
        if target_dir in dirs:
            result.append(current_path)
            dirs.remove(target_dir)  # stop recursion
    return result


def build_dataset_from_shards(shard_dirs):
    """Merge multiple datasets and re-index episode indices."""
    merged_datasets = []
    current_episode_idx = 0
    for shard in shard_dirs:
        dataset = load_from_disk(shard)

        # re-index episode idx
        episode_col = np.array(dataset["episode_idx"][:])
        new_episode_col = episode_col + current_episode_idx
        dataset = dataset.remove_columns("episode_idx")
        dataset = dataset.add_column("episode_idx", new_episode_col)

        # add shard path in dir
        dataset = dataset.add_column("data_dir", [shard] * len(dataset))
        current_episode_idx += episode_col.max() + 1
        merged_datasets.append(dataset)

    return concatenate_datasets(merged_datasets)


class Dataset:
    def __init__(
        self,
        name,
        frameskip=1,
        num_steps=1,
        decode_columns=None,
        transform=None,
        obs_type="img",
        cache_dir=None,
        subset_prop=1.0,
    ):
        # load dataset from disk (handle shards if any)
        data_dir = Path(cache_dir or get_cache_dir(), name)
        self.shard_dirs = find_shard_dirs(data_dir, target_dir=obs_type)
        self.dataset = build_dataset_from_shards(self.shard_dirs)

        logging.info(f"🛢️🛢️🛢️ Loaded raw dataset '{name}' from {len(self.shard_dirs)} shards 🛢️🛢️🛢️")

        self.frameskip = frameskip
        self.num_steps = num_steps
        self.dataset.set_format("torch")
        self.complete_traj = num_steps < 0
        self.transform = transform

        if type(decode_columns) is str:
            decode_columns = [decode_columns]
        self.decode_columns = decode_columns

        assert "episode_idx" in self.dataset.column_names, "Dataset must have 'episode_idx' column"
        assert "step_idx" in self.dataset.column_names, "Dataset must have 'step_idx' column"
        assert "data_dir" in self.dataset.column_names, (
            "Dataset must have 'data_dir' column (relative path to img/video)"
        )

        episode_col = self.dataset["episode_idx"][:]

        unique_episodes = np.unique(episode_col)
        max_episodes = int(len(unique_episodes) * subset_prop)

        self.episodes = unique_episodes[:max_episodes]
        self.episode_indices = {ep: np.flatnonzero(episode_col == ep) for ep in self.episodes}

        self.clip_len = max(frameskip * num_steps, 1) if not self.complete_traj else 0

        # # Uncomment to print episode length distribution stats
        # lengths = [len(self.episode_indices[ep]) for ep in self.episodes]
        # print("Episode length distribution:")
        # dist = Counter(lengths)
        # for length in sorted(dist):
        #     print(f"{length:4d}: {dist[length]}")
        # print(f"Min episode length: {min(lengths)}")
        # print(f"Max episode length: {max(lengths)}")
        # print(f"Average episode length: {sum(lengths) / len(lengths):.2f}")
        # ######################################################

        if any(len(self.episode_indices[ep]) < self.clip_len for ep in self.episodes):
            if all(len(self.episode_indices[ep]) < self.clip_len for ep in self.episodes):
                raise ValueError(f"At least one episode must have at least {self.clip_len} steps")
            logging.warning(
                f"Some episodes have fewer steps than the clip length {self.clip_len}. These episodes will be skipped."
            )
            # remove these episodes
            self.episodes = [ep for ep in self.episodes if len(self.episode_indices[ep]) >= self.clip_len]
            self.episode_indices = {ep: self.episode_indices[ep] for ep in self.episodes}

        episode_max_end = [max(0, len(ep) - self.clip_len + 1) for ep in self.episode_indices.values()]
        self.episode_starts = np.cumsum([0] + episode_max_end)
        self.idx_to_episode = np.searchsorted(self.episode_starts, np.arange(len(self)), side="right") - 1

        return

    @property
    def column_names(self):
        return self.dataset.column_names

    def __len__(self):
        return int(self.episode_starts[-1]) if not self.complete_traj else len(self.episodes)

    def decode(self, data_dir, col_data, indices):
        raise NotImplementedError("Dataset.decode must be implemented in subclass")

    def load_chunk(self, episode, start, end):
        if isinstance(episode, numbers.Integral):
            episode = [episode]

        if isinstance(start, numbers.Integral):
            start = [start] * len(episode)

        if isinstance(end, numbers.Integral):
            end = [end] * len(episode)

        # check that the episode was not filtered out when loading the dataset in __init__
        for ep in episode:
            if ep not in self.episodes:
                raise ValueError(f"Episode {ep} was filtered out due to insufficient length")

        chunks = []

        for ep, s, en in zip(episode, start, end):
            episode_indices = self.episode_indices[ep]

            if ep > len(self.episodes) or ep < 0:
                raise ValueError(f"Episode {ep} index out of range [0, {len(self.episodes)})")

            if en > len(episode_indices) or s < 0 or en <= s:
                raise ValueError(f"Invalid start/end indices for episode {ep}: [{s}, {en})")

            if (en - s) % self.frameskip != 0:
                raise ValueError(
                    f"Invalid start/end indices for episode {ep} with frameskip {self.frameskip}: [{s}, {en}). Must be divisible by frameskip."
                )

            episode_mask = self.dataset["episode_idx"][:] == ep
            steps_mask = self.dataset["step_idx"][:]
            steps_mask = (steps_mask >= s) & (steps_mask < en)

            indices = np.flatnonzero(episode_mask & steps_mask)
            steps = self.dataset[indices]

            for col, data in steps.items():
                if col == "action":
                    continue

                data = data[:: self.frameskip]
                steps[col] = data

                if col in self.decode_columns:
                    steps[col] = self.decode(steps["data_dir"], steps[col], start=s, end=en)

            if self.transform:
                steps = self.transform(steps)

            # stack frames
            for col in self.decode_columns:
                if col not in steps:
                    continue
                steps[col] = torch.stack(steps[col])

            # reshape action
            if "action" in steps:
                act_shape = (en - s) // self.frameskip
                steps["action"] = steps["action"].reshape(act_shape, -1)

            chunks.append(steps)

        return chunks


class FrameDataset(Dataset):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, obs_type="img", **kwargs)
        self.decode_columns = self.decode_columns or self.determine_img_columns(self.dataset[0])

    def decode(self, data_dirs, col_data, start=0, end=-1):
        pairs = zip(data_dirs, col_data)
        return [decode_image(os.path.join(dir, img_path)) for dir, img_path in pairs]

    def __getitem__(self, index):
        episode = self.idx_to_episode[index]
        episode_id = self.episodes[episode]
        episode_indices = self.episode_indices[episode_id]
        offset = index - self.episode_starts[episode]

        # determine clip bounds
        start = offset if not self.complete_traj else 0
        stop = start + self.clip_len if not self.complete_traj else len(self.episode_indices[episode])
        step_slice = episode_indices[start:stop]
        steps = self.dataset[step_slice]

        for col, data in steps.items():
            if col == "action":
                continue

            data = data[:: self.frameskip]
            steps[col] = data

            if col in self.decode_columns:
                steps[col] = self.decode(steps["data_dir"], steps[col])

        if self.transform:
            steps = self.transform(steps)

        # stack frames
        for col in self.decode_columns:
            if col not in steps:
                continue
            steps[col] = torch.stack(list(steps[col]))

        # reshape action
        if "action" in steps:
            act_shape = self.num_steps if not self.complete_traj else len(self.episode_indices[episode])
            steps["action"] = steps["action"].reshape(act_shape, -1)

        return steps

    def determine_img_columns(self, sample):
        # TODO: support other image formats
        img_columns = {k for k in sample.keys() if isinstance(sample[k], str) and sample[k].endswith(".jpeg")}
        return img_columns


class VideoDataset(Dataset):
    def __init__(self, name, *args, device="cpu", **kwargs):
        super().__init__(name, *args, obs_type="videos", **kwargs)
        self.device = device
        self.decode_columns = self.decode_columns or self.determine_video_columns(self.dataset[0])
        decord.bridge.set_bridge("torch")

    def decode(self, data_dirs, col_data, start=0, end=-1):
        path = os.path.join(data_dirs[0], col_data[0])
        vr = VideoReader(path, ctx=cpu(0))
        idxs = list(range(start, end, self.frameskip))
        frames = vr.get_batch(idxs).permute(0, 3, 1, 2)  # TCHW
        return list(frames)

    def __getitem__(self, index):
        episode = self.idx_to_episode[index]
        episode_id = self.episodes[episode]
        episode_indices = self.episode_indices[episode_id]
        offset = index - self.episode_starts[episode]

        # determine clip bounds
        start = offset if not self.complete_traj else 0
        stop = start + self.clip_len if not self.complete_traj else len(self.episode_indices[episode])
        step_slice = episode_indices[start:stop]
        steps = self.dataset[step_slice]

        for col, data in steps.items():
            if col == "action":
                continue

            data = data[:: self.frameskip]
            steps[col] = data

            if col in self.decode_columns:
                steps[col] = self.decode(steps["data_dir"], steps[col], start=start, end=stop)

        if self.transform:
            steps = self.transform(steps)

        # stack frames
        for col in self.decode_columns:
            if col not in steps:
                continue
            steps[col] = torch.stack(list(steps[col]))

        # reshape action
        if "action" in steps:
            act_shape = self.num_steps if not self.complete_traj else len(self.episode_indices[episode])
            steps["action"] = steps["action"].reshape(act_shape, -1)

        return steps

    def determine_video_columns(self, sample):
        # TODO: support other video formats
        video_columns = {k for k in sample.keys() if isinstance(sample[k], str) and sample[k].endswith(".mp4")}
        return video_columns


class InjectedDataset:
    def __init__(
        self,
        original_dataset,
        external_datasets: list[Dataset],
        proportions: list[float] | None = None,
        seed: int | None = None,
    ):
        self.datasets = [original_dataset] + external_datasets
        self.proportions = proportions or [1.0 / len(self.datasets)] * len(self.datasets)

        # infer original dataset proportion
        if len(self.proportions) == len(self.datasets) - 1:
            assert sum(self.proportions) < 1.0, "Sum of external dataset proportions must be < 1.0"
            og_prop = 1.0 - sum(self.proportions)
            self.proportions = [og_prop] + self.proportions

        if len(self.proportions) != (len(self.datasets)):
            raise ValueError("Proportions length must match number of datasets (original + external)")

        self.check_consistency()

        # determine length of injected dataset
        target_ds_len = len(original_dataset)
        per_dataset_samples = [int(target_ds_len * p) for p in self.proportions]
        self.length = sum(per_dataset_samples)

        # draw random indices for each dataset
        gen = np.random.default_rng(seed)
        self.mapping = []
        for ds, n in zip(self.datasets, per_dataset_samples):
            self.mapping.extend(gen.choice(len(ds), size=n, replace=True).tolist())

        self.cumulative_sizes = np.cumsum([0] + per_dataset_samples)
        self.idx_to_ds = np.searchsorted(self.cumulative_sizes, np.arange(len(self)), side="right") - 1
        return

    @property
    def column_names(self):
        return self.datasets[0].column_names

    def check_consistency(self):
        fs = self.datasets[0].frameskip
        ns = self.datasets[0].num_steps
        for ds in self.datasets[1:]:
            if ds.frameskip != fs or ds.num_steps != ns:
                raise ValueError("All datasets must have the same frameskip and num_steps")

        # keep only the intersection of column names
        column_sets = [set(ds.column_names) for ds in self.datasets]
        common_columns = set.intersection(*column_sets)

        for i, ds in enumerate(self.datasets):
            cols_to_remove = set(ds.column_names) - common_columns
            if len(cols_to_remove) > 0:
                print(f"InjectedDataset: Removing columns {cols_to_remove} from dataset {i} for consistency")
                self.datasets[i].dataset = ds.dataset.remove_columns(list(cols_to_remove))

        return

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        dataset_idx = self.idx_to_ds[index]
        sample_idx = self.mapping[index]
        return self.datasets[dataset_idx][sample_idx]


class HDF5Dataset:
    def __init__(
        self,
        name: str,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable | None = None,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        cache_dir: str | None = None,
    ):
        self.h5_path = Path(cache_dir or get_cache_dir(), f"{name}.h5")
        self.h5_file = None
        self._cache = {}

        self.keys_to_load = keys_to_load
        self.metadata_keys = ["ep_len", "ep_offset"]
        self.keys_to_cache = keys_to_cache or []

        with h5py.File(self.h5_path, "r") as f:
            self.offsets = f["ep_offset"][:]
            self.lengths = f["ep_len"][:]

            if self.keys_to_load is None:
                self.keys_to_load = list(f.keys())

            for key in self.keys_to_cache:
                if key in f:
                    self._cache[key] = f[key][:]
                    logging.info(f"Cached key '{key}' from HDF5 file '{self.h5_path}'")
                else:
                    raise KeyError(f"Key '{key}' not found in HDF5 file '{self.h5_path}'")

        self.transform = transform
        self.frameskip = frameskip
        self.num_steps = num_steps
        self.span = num_steps * frameskip

        # valid episode indices
        self.clip_indices = []
        for ep_idx, length in enumerate(self.lengths):
            if length >= self.span:
                for start_f in np.linspace(0, length - self.span, dtype=int):
                    self.clip_indices.append((ep_idx, start_f))
        return

    @property
    def column_names(self):
        return [key for key in self.keys_to_load if key not in self.metadata_keys]

    def __len__(self):
        return len(self.clip_indices)

    def _init_h5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)
        return

    def load_slice(self, start_idx, end_idx):
        steps = {}
        for col in self.keys_to_load:
            if col in self.metadata_keys:
                continue

            if col in self._cache:
                data = self._cache[col][start_idx:end_idx]
            else:
                data = self.h5_file[col][start_idx:end_idx]

            # apply frameskip if not action
            if col != "action":
                data = data[:: self.frameskip]

            steps[col] = torch.from_numpy(data)

            # channel first for images
            is_img = len(data.shape) == 4 and data.shape[-1] in [1, 3]
            if is_img:
                steps[col] = steps[col].permute(0, 3, 1, 2)  # TCHW

        if self.transform:
            steps = self.transform(steps)

        return steps

    def __getitem__(self, idx: int):
        ep_idx, local_start = self.clip_indices[idx]
        start = self.offsets[ep_idx] + local_start
        end = start + self.span

        self._init_h5()
        steps = self.load_slice(start, end)

        if "action" in steps:
            act_shape = self.num_steps
            steps["action"] = steps["action"].reshape(act_shape, -1)

        return steps

    def get_chunk_data(self, episodes_idx, start, end):
        self._init_h5()
        global_start = self.offsets[episodes_idx] + start
        end = global_start + (end - start)

        chunk = []
        for s_idx, e_idx in zip(global_start, end):
            steps = self.load_slice(s_idx, e_idx)

            # reshape action
            if "action" in steps:
                act_shape = (e_idx - s_idx) // self.frameskip
                steps["action"] = steps["action"].reshape(act_shape, -1)
            chunk.append(steps)

        return chunk

    def get_col_data(self, col: str):
        self._init_h5()
        return self.h5_file[col][:]

    def get_row_data(self, row_idx: int | list[int]):
        self._init_h5()
        sample = {}
        for col in self.keys_to_load:
            if col in self.metadata_keys:
                continue
            sample[col] = self.h5_file[col][row_idx]

        return sample


class GoalDataset:
    """
    Dataset wrapper that samples an additional goal observation per item.

    Works with any dataset type (HDF5Dataset, FrameDataset, VideoDataset, etc.)

    Goals are sampled from:
      - random state (uniform over dataset steps)
      - future state in same episode (Geom(1-gamma))
      - current state
    with probabilities (0.3, 0.5, 0.2) by default.
    """

    def __init__(
        self,
        dataset: Dataset | HDF5Dataset,
        goal_probabilities: tuple[float, float, float] = (0.3, 0.5, 0.2),
        gamma: float = 0.99,
        goal_keys: dict[str, str] | None = None,
        seed: int | None = None,
    ):
        """
        Args:
            dataset: Base dataset to wrap.
            goal_probabilities: Tuple of (p_random, p_future, p_current) for goal sampling.
            gamma: Discount factor for future goal sampling.
            goal_keys: Mapping of source observation keys to goal observation keys. If None, defaults to {"pixels": "goal", "proprio": "goal_proprio"}.
            seed: Random seed for goal sampling.
        """
        self.dataset = dataset
        self.is_hdf5 = isinstance(dataset, HDF5Dataset)

        if len(goal_probabilities) != 3:
            raise ValueError("goal_probabilities must be a 3-tuple (random, future, current)")
        if not np.isclose(sum(goal_probabilities), 1.0):
            raise ValueError("goal_probabilities must sum to 1.0")

        self.goal_probabilities = goal_probabilities
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        # Setup episode info based on dataset type
        if self.is_hdf5:
            self.episode_lengths = dataset.lengths
            self.episode_offsets = dataset.offsets
        else:
            self.episode_lengths = np.array([len(dataset.episode_indices[ep]) for ep in dataset.episodes])
            self.episode_offsets = None

        self._episode_cumlen = np.cumsum(self.episode_lengths)
        self._total_steps = int(self._episode_cumlen[-1]) if len(self._episode_cumlen) else 0

        # Auto-detect goal keys if not provided
        if goal_keys is None:
            goal_keys = {}
            column_names = dataset.column_names
            if "pixels" in column_names:
                goal_keys["pixels"] = "goal_pixels"
            if "proprio" in column_names:
                goal_keys["proprio"] = "goal_proprio"
        self.goal_keys = goal_keys

    def __len__(self):
        return len(self.dataset)

    @property
    def column_names(self):
        return self.dataset.column_names

    def _sample_goal_kind(self) -> str:
        r = self.rng.random()
        p_random, p_future, _ = self.goal_probabilities
        if r < p_random:
            return "random"
        if r < p_random + p_future:
            return "future"
        return "current"

    def _sample_random_step(self) -> tuple[int, int]:
        """Sample random (ep_idx, local_idx) from entire dataset."""
        if self._total_steps == 0:
            return 0, 0
        flat_idx = int(self.rng.integers(0, self._total_steps))
        ep_idx = int(np.searchsorted(self._episode_cumlen, flat_idx, side="right"))
        prev = self._episode_cumlen[ep_idx - 1] if ep_idx > 0 else 0
        local_idx = flat_idx - prev
        return ep_idx, local_idx

    def _sample_future_step(self, ep_idx: int, local_start: int) -> tuple[int, int]:
        """Sample future (ep_idx, local_idx) from same episode using geometric distribution."""
        frameskip = self.dataset.frameskip
        max_steps = (self.episode_lengths[ep_idx] - 1 - local_start) // frameskip
        if max_steps <= 0:
            return ep_idx, local_start

        p = max(1.0 - self.gamma, 1e-6)
        k = int(self.rng.geometric(p))
        k = min(k, max_steps)
        local_idx = local_start + k * frameskip
        return ep_idx, local_idx

    def _get_clip_info(self, idx: int) -> tuple[int, int]:
        """Returns (episode_idx, local_start) for a given dataset index."""
        if self.is_hdf5:
            return self.dataset.clip_indices[idx]
        else:
            episode = self.dataset.idx_to_episode[idx]
            offset = idx - self.dataset.episode_starts[episode]
            return episode, offset

    def _load_single_step(self, ep_idx: int, local_idx: int) -> dict[str, torch.Tensor]:
        """Load a single step from episode ep_idx at local index local_idx."""
        if self.is_hdf5:
            abs_idx = int(self.episode_offsets[ep_idx] + local_idx)
            return self.dataset.load_slice(abs_idx, abs_idx + 1)
        else:
            episode_id = self.dataset.episodes[ep_idx]
            chunks = self.dataset.load_chunk(episode_id, local_idx, local_idx + 1)
            return chunks[0]

    def __getitem__(self, idx: int):
        # Get base sample from wrapped dataset
        steps = self.dataset[idx]

        if not self.goal_keys:
            return steps

        # Get episode and local start for this index
        ep_idx, local_start = self._get_clip_info(idx)

        # Sample goal (transform will be applied via underlying dataset's load_chunk/load_slice)
        goal_kind = self._sample_goal_kind()
        if goal_kind == "random":
            goal_ep_idx, goal_local_idx = self._sample_random_step()
        elif goal_kind == "future":
            goal_ep_idx, goal_local_idx = self._sample_future_step(ep_idx, local_start)
        else:  # current
            goal_ep_idx, goal_local_idx = ep_idx, local_start

        # Load goal step
        goal_step = self._load_single_step(goal_ep_idx, goal_local_idx)

        # Add goal observations to steps
        for src_key, goal_key in self.goal_keys.items():
            if src_key not in goal_step or src_key not in steps:
                continue
            goal_val = goal_step[src_key]
            if goal_val.ndim == 0:
                goal_val = goal_val.unsqueeze(0)
            steps[goal_key] = goal_val

        return steps
