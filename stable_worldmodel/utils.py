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
