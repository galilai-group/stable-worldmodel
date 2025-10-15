import os
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import gymnasium as gym
import numpy as np
import PIL
import stable_pretraining as spt
import torch
from datasets import load_from_disk
from rich import print

import stable_worldmodel as swm


class StepsDataset(spt.data.HFDataset):
    def __init__(
        self,
        path,
        *args,
        num_steps=2,
        frameskip=1,
        **kwargs,
    ):
        data_dir = Path(kwargs.get("cache_dir", swm.data.get_cache_dir()), path)
        super().__init__(str(data_dir), *args, **kwargs)

        self.data_dir = data_dir
        self.num_steps = num_steps
        self.frameskip = frameskip

        assert "episode_idx" in self.dataset.column_names, "Dataset must have 'episode_idx' column"
        assert "step_idx" in self.dataset.column_names, "Dataset must have 'step_idx' column"
        assert "action" in self.dataset.column_names, "Dataset must have 'action' column"

        self.dataset.set_format("torch")

        # get number of episodes
        ep_indices = self.dataset["episode_idx"][:]
        self.episodes = np.unique(ep_indices)

        # get dataset indices of each episode
        self.episode_slices = {e: self.get_episode_slice(e, ep_indices) for e in self.episodes}

        # start index for each episode
        valid_samples_per_ep = [
            max(0, len(ep_slice) - self.num_steps * self.frameskip + 1) for ep_slice in self.episode_slices.values()
        ]
        self.cum_slices = np.cumsum([0] + valid_samples_per_ep)

        # map from sample to their episode
        self.idx_to_ep = np.searchsorted(self.cum_slices, torch.arange(len(self)), side="right") - 1

        self.img_cols = self.infer_img_path_columns()

    def get_episode_slice(self, episode_idx, episode_indices):
        """Return number of possible slices for a given episode index"""
        indices = np.flatnonzero(episode_indices == episode_idx)
        if len(indices) <= (self.num_steps * self.frameskip):
            raise ValueError(
                f"Episode {episode_idx} is too short ({len(indices)} steps) for {self.num_steps} steps with {self.frameskip} frameskip"
            )
        return indices

    def __len__(self):
        return int(self.cum_slices[-1])

    def __getitem__(self, idx):
        ep = self.idx_to_ep[idx]
        episode_indices = self.episode_slices[ep]
        offset = idx - self.cum_slices[ep]
        start = offset
        stop = start + self.num_steps * self.frameskip
        idx_slice = episode_indices[start:stop]
        steps = self.dataset[idx_slice]

        for k, v in steps.items():
            if k == "action":
                continue

            v = v[:: self.frameskip]
            steps[k] = v

            if k in self.img_cols:
                steps[k] = [PIL.Image.open(self.data_dir / img_path) for img_path in v]

        if self.transform:
            steps = self.transform(steps)

        # stack images into a single tensor
        for k in self.img_cols:
            steps[k] = torch.stack(steps[k])

        # reshape action
        steps["action"] = steps["action"].reshape(self.num_steps, -1)

        return steps

    def infer_img_path_columns(self):
        """Return list of columns that contain image file paths."""

        IMG_EXTENSIONS = (".jpeg", ".png", ".jpg")

        img_cols = set()
        first_elem = self.dataset[0]
        for col in self.dataset.column_names:
            if isinstance(first_elem[col], str) and first_elem[col].endswith(IMG_EXTENSIONS):
                img_cols.add(col)
        return img_cols


#####################
###     utils     ###
#####################


def is_image(x):
    return type(x) is np.ndarray and x.ndim == 3 and x.shape[2] in [1, 3, 4] and x.dtype == np.uint8


#####################
###   CLI Info   ####
#####################


class SpaceInfo(TypedDict, total=False):
    shape: tuple[int, ...]
    type: str
    dtype: str
    low: Any
    high: Any
    n: int  # for discrete spaces


class VariationInfo(TypedDict):
    has_variation: bool
    type: str | None
    names: list[str] | None


class WorldInfo(TypedDict):
    name: str
    observation_space: SpaceInfo
    action_space: SpaceInfo
    variation: VariationInfo
    config: dict[str, Any]


def get_cache_dir() -> Path:
    """Return the cache directory for stable_worldmodel."""
    cache_dir = os.getenv("XENOWORLDS_HOME", os.path.expanduser("~/.stable_worldmodel"))
    os.makedirs(cache_dir, exist_ok=True)
    return Path(cache_dir)


def list_datasets():
    with os.scandir(get_cache_dir()) as entries:
        return [e.name for e in entries if e.is_dir()]


def list_models():
    pattern = re.compile(r"^(.*?)(?=_(?:weights(?:-[^.]*)?|object)\.ckpt$)", re.IGNORECASE)

    cache_dir = get_cache_dir()
    models = set()

    for fname in os.listdir(cache_dir):
        m = pattern.match(fname)
        if m:
            models.add(m.group(1))

    return sorted(models)


def dataset_info(name):
    # check name exists
    if name not in list_datasets():
        raise ValueError(f"Dataset '{name}' not found. Available: {list_datasets()}")

    dataset = load_from_disk(str(Path(get_cache_dir(), name, "records")))

    dataset.set_format("numpy")

    def assert_msg(col):
        return f"Dataset must have '{col}' column"  # type: ignore

    assert "episode_idx" in dataset.column_names, assert_msg("episode_idx")
    assert "step_idx" in dataset.column_names, assert_msg("step_idx")
    assert "episode_len" in dataset.column_names, assert_msg("episode_len")
    assert "pixels" in dataset.column_names, assert_msg("pixels")
    assert "action" in dataset.column_names, assert_msg("action")
    assert "goal" in dataset.column_names, assert_msg("goal")

    info = {
        "name": name,
        "num_episodes": len(np.unique(dataset["episode_idx"])),
        "num_steps": len(dataset),
        "columns": dataset.column_names,
        "obs_shape": dataset["pixels"][0].shape,
        "action_shape": dataset["action"][0].shape,
        "goal_shape": dataset["goal"][0].shape,
        "variation": {
            "has_variation": any(col.startswith("variation.") for col in dataset.column_names),
            "names": [col.removeprefix("variation.") for col in dataset.column_names if col.startswith("variation.")],
        },
    }

    return info


def list_worlds():
    return sorted(swm.WORLDS)


def _space_meta(space) -> SpaceInfo | dict[str, SpaceInfo] | list[SpaceInfo]:
    if isinstance(space, gym.spaces.Dict):
        return {k: _space_meta(v) for k, v in space.spaces.items()}

    if isinstance(space, gym.spaces.Sequence) or isinstance(space, gym.spaces.Tuple):
        return [_space_meta(s) for s in space.spaces]

    info: SpaceInfo = {
        "shape": getattr(space, "shape", None),
        "type": type(space).__name__,
    }

    if hasattr(space, "dtype") and getattr(space, "dtype") is not None:
        info["dtype"] = str(space.dtype)
    if hasattr(space, "low"):
        info["low"] = getattr(space, "low", None)
    if hasattr(space, "high"):
        info["high"] = getattr(space, "high", None)
    if hasattr(space, "n"):
        info["n"] = getattr(space, "n")
    return info


@lru_cache(maxsize=128)
def world_info(
    name: str,
    *,
    image_shape: tuple[int, int] = (224, 224),
    render_mode: str = "rgb_array",
) -> WorldInfo:
    if name not in swm.WORLDS:
        raise ValueError(f"World '{name}' not found. Available: {', '.join(list_worlds())}")
    world = None

    try:
        world = swm.World(
            name,
            num_envs=1,
            image_shape=image_shape,
            render_mode=render_mode,
            verbose=0,
        )

        obs_space = getattr(world, "single_observation_space", None)
        act_space = getattr(world, "single_action_space", None)
        var_space = getattr(world, "single_variation_space", None)

        variation: VariationInfo = {
            "has_variation": var_space is not None,
            "type": type(var_space).__name__ if var_space is not None else None,
            "names": var_space.names() if hasattr(var_space, "names") else None,
        }

        return {
            "name": name,
            "observation_space": _space_meta(obs_space) if obs_space else {},
            "action_space": _space_meta(act_space) if act_space else {},
            "variation": variation,
        }

    finally:
        if world is not None and hasattr(world, "close"):
            try:
                world.close()
            except Exception:
                pass


def delete_dataset(name):
    from datasets import logging as ds_logging

    ds_logging.set_verbosity_error()

    try:
        dataset_path = Path(get_cache_dir(), name)

        if not dataset_path.exists():
            raise ValueError(f"Dataset {name} does not exist at {dataset_path}")

        dataset = load_from_disk(str(Path(dataset_path, "records")))

        # remove cache files
        dataset.cleanup_cache_files()

        # delete dataset directory
        shutil.rmtree(dataset_path, ignore_errors=False)

        print(f"🗑️ Dataset {dataset_path} deleted!")

    except Exception as e:
        print(f"[red]Error cleaning up dataset [cyan]{name}[/cyan]: {e}[/red]")


def delete_model(name):
    pattern = re.compile(rf"^{re.escape(name)}(?:_[^-].*)?\.ckpt$")
    cache_dir = get_cache_dir()

    for fname in os.listdir(cache_dir):
        if pattern.match(fname):
            filepath = os.path.join(cache_dir, fname)
            try:
                os.remove(filepath)
                print(f"🔮 Model {fname} deleted")
            except Exception as e:
                print(f"[red]Error occurred while deleting model [cyan]{name}[/cyan]: {e}[/red]")
