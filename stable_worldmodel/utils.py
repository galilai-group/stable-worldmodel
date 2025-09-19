import os
import sys
import shlex
import subprocess

from loguru import logger as logging


def get_cache_dir() -> str:
    """Return the cache directory for stable_worldmodel"""
    cache_dir = os.getenv("XENOWORLDS_HOME", os.path.expanduser("~/.stable_worldmodel"))
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def pretraining(script_path: str, args: str) -> int:
    assert os.path.isfile(script_path), f"Script {script_path} does not exist."
    logging.info(f"🏃🏃🏃 Running pretraining script: {script_path} with args: {args} 🏃🏃🏃")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd = [sys.executable, script_path] + shlex.split(args)
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

    logging.info("🏁🏁🏁 Pretraining script finished 🏁🏁🏁")
    return