import os
from pathlib import Path

from stable_worldmodel.utils import DEFAULT_CACHE_DIR


def get_cache_dir(
    override_root: Path | None = None,
    sub_folder: str | None = None,
) -> Path:
    base = override_root
    if override_root is None:
        base = os.getenv('STABLEWM_HOME', str(DEFAULT_CACHE_DIR))

    cache_path = (
        Path(base, sub_folder) if sub_folder is not None else Path(base)
    )

    return cache_path


def ensure_dir_exists(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
