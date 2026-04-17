"""OpenApps gymnasium envs for stable-worldmodel.

Importing this module registers:
  * One generic env per OpenApps application (no task bound, reward always 0).
  * One task-bound env per entry in ``openapps/config/tasks/all_tasks.yaml``.
"""

from pathlib import Path

from gymnasium.envs import registration
from loguru import logger

from stable_worldmodel.envs import WORLDS
from .env import OpenAppsEnv  # noqa: F401 — also triggers server path setup

_APPS = ["todo", "calendar", "messages", "codeeditor", "map"]

for _app in _APPS:
    _env_id = f"swm/OpenApps-{_app.capitalize()}-v0"
    registration.register(
        id=_env_id,
        entry_point="stable_worldmodel.envs.openapps.env:OpenAppsEnv",
        kwargs={"app_name": _app},
    )
    WORLDS.add(_env_id)


# ── Task-bound env registration ──────────────────────────────────────

# Maps Task class name → OpenApps app key (must match _APPS).
_TASK_CLASS_TO_APP = {
    "AddEventTask": "calendar",
    "RemoveEventTask": "calendar",
    "AddToDoTask": "todo",
    "MarkToDoDoneTask": "todo",
    "SendMessageTask": "messages",
    "SavePlaceTask": "map",
}


def _snake_to_pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def _register_tasks_from_yaml() -> None:
    """Load canonical OpenApps tasks and register one env id per task.

    Silently skips if the yaml or openapps repo is unavailable — the
    generic shells already cover the basic case.
    """
    try:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
    except ImportError as e:
        logger.warning(f"Skipping OpenApps task registration (hydra missing): {e}")
        return

    here = Path(__file__).resolve()
    workspace_root = here.parents[4]
    tasks_yaml = workspace_root / "openapps" / "config" / "tasks" / "all_tasks.yaml"

    if not tasks_yaml.is_file():
        logger.warning(f"OpenApps tasks yaml not found at {tasks_yaml}; skipping")
        return

    try:
        cfg = OmegaConf.load(tasks_yaml)
    except Exception as e:
        logger.warning(f"Failed to load {tasks_yaml}: {e}")
        return

    for task_key, task_cfg in cfg.items():
        target = task_cfg.get("_target_", "")
        task_class = target.rsplit(".", 1)[-1]
        app_name = _TASK_CLASS_TO_APP.get(task_class)
        if app_name is None:
            logger.warning(
                f"No app mapping for task class {task_class!r} "
                f"(task {task_key!r}); skipping"
            )
            continue

        try:
            task_obj = instantiate(task_cfg)
        except Exception as e:
            logger.warning(f"Failed to instantiate task {task_key!r}: {e}")
            continue

        env_id = (
            f"swm/OpenApps-{app_name.capitalize()}-"
            f"{_snake_to_pascal(task_key)}-v0"
        )
        registration.register(
            id=env_id,
            entry_point="stable_worldmodel.envs.openapps.env:OpenAppsEnv",
            kwargs={
                "app_name": app_name,
                "task": task_obj,
                "task_description": task_obj.goal,
            },
        )
        WORLDS.add(env_id)


_register_tasks_from_yaml()

__all__ = ["OpenAppsEnv"]
