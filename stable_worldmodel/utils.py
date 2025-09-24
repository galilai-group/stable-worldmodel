import inspect
import os
import shlex
import subprocess
import sys
import time
import types
from typing import Any, Iterable, MutableMapping

from loguru import logger as logging


def get_cache_dir() -> str:
    """Return the cache directory for stable_worldmodel."""
    cache_dir = os.getenv("XENOWORLDS_HOME", os.path.expanduser("~/.stable_worldmodel"))
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


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
    """Return mapping[path[0]][path[1]]... ; raises KeyError if missing."""
    cur = mapping
    for key in list(path):
        cur = cur[key]
    return cur


def set_in(mapping: MutableMapping, path: Iterable[str], value: Any) -> None:
    """Set mapping[path[:-1]][last] = value, creating nested dicts as needed."""
    *parents, last = list(path)
    cur = mapping
    for key in parents:
        cur = cur.setdefault(key, {})
    cur[last] = value
