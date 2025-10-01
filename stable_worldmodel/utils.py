"""Utility functions for stable_worldmodel."""

import inspect
import os
import shlex
import subprocess
import sys
import time
import types
from typing import Any, Iterable, MutableMapping

from loguru import logger as logging


def pretraining(script_path: str, args: str) -> int:
    assert os.path.isfile(script_path), f"Script {script_path} does not exist."
    logging.info(
        f"🏃🏃🏃 Running pretraining script: {script_path} with args: {args} 🏃🏃🏃"
    )
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd = [sys.executable, script_path] + shlex.split(args)
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

    logging.info("🏁🏁🏁 Pretraining script finished 🏁🏁🏁")
    return


def flatten_dict(d, parent_key="", sep="."):
    """Flatten a nested dictionary into a single-level dictionary with concatenated keys.

    The naming convention for the new keys is similar to Hydra's, using a `.` separator to denote levels of nesting.
    Attention is needed when flattening dictionaries with overlapping keys, as this may lead to information loss.

    Args:
        d (dict): The nested dictionary to flatten.
        parent_key (str, optional): The base key to use for the flattened keys.
        sep (str, optional): The separator to use between levels of nesting. Defaults to '.'.

    Returns:
        dict: A flattened version of the input dictionary.

    Examples:
        >>> info = {"a": {"b": {"c": 42, "d": 43}}, "e": 44}
        >>> flatten_dict(info)
        {'a.b.c': 42, 'a.b.d': 43, 'e': 44}

        >>> flatten_dict({"a": {"b": 2}, "a.b": 3})
        {'a.b': 3}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def get_in(mapping: dict, path: Iterable[str]) -> Any:
    """Retrieve a value from a nested dictionary using a sequence of keys.

    Args:
        mapping (dict): A nested dictionary.
        path (Iterable[str]): An iterable of keys representing the path to the desired value in mapping.

    Returns:
        Any: The value located at the specified path in the nested dictionary.

    Raises:
        KeyError: If any key in the path does not exist in the mapping dict.

    Examples:
        >>> variations = {"a": {"b": {"c": 42}}}
        >>> get_in(variations, ["a", "b", "c"])
        42
    """
    cur = mapping
    for key in list(path):
        cur = cur[key]
    return cur
