import inspect
import os
import shlex
import subprocess
import sys
import types
import time

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

def flatten_dict(d, parent_key='', sep='.'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def patch_sampling(space, condition_fn, *, max_tries=100000, warn_after_s=5.0):
    """
    Mutates `space` so that space.sample(...) repeatedly samples from the original
    sampler until `condition_fn(...)` returns True.

    condition_fn: callable taking (sample) or (sample, space) -> bool
    max_tries:    hard guard to prevent infinite loops
    warn_after_s: log a warning if rejection sampling takes long
    """
    if not callable(condition_fn):
        raise TypeError("condition_fn must be callable")

    original_sample = space.sample  # bound method

    # Detect whether predicate wants (sample) or (sample, space)
    try:
        wants_space = len(inspect.signature(condition_fn).parameters) >= 2
    except (ValueError, TypeError):
        wants_space = False

    def patched(self, *args, **kwargs):
        start = time.time()
        for i in range(1, max_tries + 1):
            value = original_sample(*args, **kwargs)
            ok = condition_fn(value, self) if wants_space else condition_fn(value)
            if ok:
                return value
            if warn_after_s is not None and (time.time() - start) > warn_after_s and i == 1:
                logging.warning("patch_sampling: rejection sampling is taking a while...")
        raise RuntimeError(
            f"patch_sampling: predicate not satisfied after {max_tries} draws"
        )

    space.sample = types.MethodType(patched, space)
    return space
