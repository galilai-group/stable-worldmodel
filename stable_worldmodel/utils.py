"""Utility functions for stable_worldmodel."""

import os
import shlex
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger as logging


def pretraining(
    script_path: str,
    dataset_name: str,
    output_model_name: str,
    dump_object: bool = True,
    args: str = '',
) -> None:
    """Run a pretraining script as a subprocess.

    Args:
        script_path: Path to the python script to execute.
        dataset_name: Name of the dataset to use for training.
        output_model_name: Name to save the resulting model under.
        dump_object: Whether to export the model object.
        args: Additional CLI arguments for the script.

    Raises:
        ValueError: If script_path does not exist.
        SystemExit: If the subprocess returns a non-zero exit code.
    """
    if not os.path.isfile(script_path):
        raise ValueError(f'Script {script_path} does not exist.')

    logging.info(
        f'🏃🏃🏃 Running pretraining script: {script_path} with args: {args} 🏃🏃🏃'
    )
    env = os.environ.copy()
    env.setdefault('PYTHONUNBUFFERED', '1')

    args = f'{args} ++dump_object={dump_object} dataset_name={dataset_name} output_model_name={output_model_name}'
    cmd = [sys.executable, script_path] + shlex.split(args)
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

    logging.info('🏁🏁🏁 Pretraining script finished 🏁🏁🏁')


def flatten_dict(
    d: dict, parent_key: str = '', sep: str = '.'
) -> dict:
    """Flatten a nested dictionary into a single-level dictionary.

    Args:
        d: The dictionary to flatten.
        parent_key: Prefix for keys at the current recursion level.
        sep: Separator between nested keys.

    Returns:
        A flattened dictionary with dotted (or otherwise separated) keys.
    """
    items: dict = {}
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def get_in(mapping: Any, path: Iterable[str]) -> Any:
    """Retrieve a value from a nested mapping using a sequence of keys.

    Args:
        mapping: The nested dictionary or object to search.
        path: An iterable of keys/indices to follow.

    Returns:
        The value found at the end of the path.
    """
    cur = mapping
    for key in list(path):
        cur = cur[key]
    return cur


def record_video_from_dataset(
    video_path: str | Path,
    dataset: Any,
    episode_idx: int | list[int],
    max_steps: int = 500,
    fps: int = 30,
    viewname: str | list[str] = 'pixels',
) -> None:
    """Replay stored dataset episodes and export them as MP4 videos.

    Args:
        video_path: Directory or file path to save the video(s).
        dataset: The dataset object to load episodes from.
        episode_idx: Index or list of indices of episodes to record.
        max_steps: Maximum frames per video.
        fps: Frames per second for the output video.
        viewname: Key(s) in the dataset to use as video frames.
    """
    import imageio

    episode_idx = (
        [episode_idx] if isinstance(episode_idx, int) else episode_idx
    )
    viewname = [viewname] if isinstance(viewname, str) else viewname

    assert all(view in dataset.column_names for view in viewname), (
        f'Some views in {viewname} are not in dataset key names {dataset.column_names}'
    )

    for ep_idx in episode_idx:
        file_path = Path(video_path, f'episode_{ep_idx}.mp4')
        steps = dataset.load_episode(ep_idx)
        frames = np.concatenate([steps[v].numpy() for v in viewname], axis=2)
        frames = frames[:max_steps]
        imageio.mimsave(file_path, frames.transpose(0, 2, 3, 1), fps=fps)

    print(f'Video saved to {video_path}')
