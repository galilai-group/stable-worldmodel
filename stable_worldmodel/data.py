from pathlib import Path
import numpy as np

import stable_pretraining as spt
import torch
import os
import gymnasium as gym

from dataclasses import dataclass, asdict
from functools import lru_cache

from typing import Any, Dict, List, Optional, Tuple, TypedDict

from torch.utils.data import default_collate
from datasets import load_dataset

import stable_worldmodel as swm
import shutil
from rich import print

import re


class StepsDataset(spt.data.HFDataset):
    def __init__(
        self,
        *args,
        num_steps=2,
        frameskip=1,
        torch_exclude_column={
            "pixels",
            "goal",
        },
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_steps = num_steps
        self.frameskip = frameskip

        assert "episode_idx" in self.dataset.column_names, (
            "Dataset must have 'episode_idx' column"
        )
        assert "step_idx" in self.dataset.column_names, (
            "Dataset must have 'step_idx' column"
        )

        self.episodes = np.unique(self.dataset["episode_idx"])
        self.slices = {e: self.num_slice(e) for e in self.episodes}
        self.cum_slices = np.cumsum([0] + [self.slices[e] for e in self.episodes])

        self.torch_exclude_column = torch_exclude_column

        # TODO: add assert for basic column name

        cols = [
            c for c in self.dataset.column_names if c not in self.torch_exclude_column
        ]
        self.dataset = self.dataset.with_format(
            "torch", columns=cols, output_all_columns=True
        )

    def num_slice(self, episode_idx):
        """Return number of possible slices for a given episode index"""
        idx = np.nonzero(self.dataset["episode_idx"] == episode_idx)[0][0].item()
        episode_len = self.dataset["episode_len"][idx]
        num_slices = 1 + (episode_len - self.num_steps * self.frameskip)

        assert num_slices > 0, (
            f"Episode {episode_idx} is too short for {self.num_steps} steps with {self.frameskip} frameskip (len={episode_len})"
        )

        return num_slices

    def process_sample(self, sample):
        if self._trainer is not None:
            if "global_step" in sample:
                raise ValueError("Can't use that keywords")
            if "current_epoch" in sample:
                raise ValueError("Can't use that keywords")
            sample["global_step"] = self._trainer.global_step
            sample["current_epoch"] = self._trainer.current_epoch
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return int(self.cum_slices[-1])

    def __getitem__(self, idx):
        # find which episode this idx belongs to
        ep_idx = np.searchsorted(self.cum_slices, idx, side="right") - 1
        episode = self.episodes[ep_idx]

        # find local slice within episode
        local_idx = idx - self.cum_slices[ep_idx]

        # get dataset indices for this slice
        ep_mask = (
            torch.nonzero(self.dataset["episode_idx"] == episode).squeeze().tolist()
        )

        # starting step index inside this episode
        start_step = local_idx

        # slice steps with frameskip
        slice_idx = [
            ep_mask[start_step + i] for i in range(self.num_steps * self.frameskip)
        ]

        # transform the data
        raw = [self.transform(self.dataset[i]) for i in slice_idx]
        raw_steps = default_collate(raw)

        # add the frameskip
        steps = {}
        for k, v in raw_steps.items():
            steps[k] = v[:: self.frameskip]

        # process actions
        actions = raw_steps["action"]
        if self.frameskip > 1:
            actions = actions.reshape(self.num_steps, -1)

        steps["action"] = actions

        return steps


#####################
### CLI Info ####
#####################


class SpaceInfo(TypedDict, total=False):
    shape: Tuple[int, ...]
    type: str
    dtype: str
    low: Any
    high: Any
    n: int  # for discrete spaces


class VariationInfo(TypedDict):
    has_variation: bool
    type: Optional[str]
    names: Optional[List[str]]


class WorldInfo(TypedDict):
    name: str
    observation_space: SpaceInfo
    action_space: SpaceInfo
    variation: VariationInfo
    config: Dict[str, Any]


def get_cache_dir() -> str:
    """Return the cache directory for stable_worldmodel."""
    cache_dir = os.getenv("XENOWORLDS_HOME", os.path.expanduser("~/.stable_worldmodel"))
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def list_datasets():
    with os.scandir(get_cache_dir()) as entries:
        return [e.name for e in entries if e.is_dir()]


def list_models():
    pattern = re.compile(
        r"^(.*?)(?=_(?:weights(?:-[^.]*)?|object)\.ckpt$)", re.IGNORECASE
    )

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

    dataset = load_dataset(
        "parquet",
        data_files=str(Path(get_cache_dir(), name, "*.parquet")),
        split="train",
    )

    dataset.set_format("numpy")

    assert_msg = lambda col: (f"Dataset must have '{col}' column")
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
            "has_variation": any(
                col.startswith("variation.") for col in dataset.column_names
            ),
            "names": [
                col.removeprefix("variation.")
                for col in dataset.column_names
                if col.startswith("variation.")
            ],
        },
    }

    return info


def list_worlds():
    return sorted(list(swm.WORLDS))


def _space_meta(space) -> SpaceInfo | Dict[str, SpaceInfo] | List[SpaceInfo]:
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
    image_shape: Tuple[int, int] = (224, 224),
    render_mode: str = "rgb_array",
) -> WorldInfo:
    if name not in swm.WORLDS:
        raise ValueError(
            f"World '{name}' not found. Available: {', '.join(list_worlds())}"
        )
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

        # remove cache files
        dataset = load_dataset(
            "parquet", data_files=str(Path(dataset_path, "*.parquet"))
        )
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
                print(
                    f"[red]Error occured while deleting model [cyan]{name}[/cyan]: {e}[/red]"
                )
