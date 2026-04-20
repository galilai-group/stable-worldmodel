"""OpenApps gymnasium envs for stable-worldmodel.

Importing this module registers one gym env id per OpenApps application
(``swm/OpenApps-Todo-v0``, ``-Calendar-v0``, ...). The task is selected
at ``gym.make`` time via the ``task=`` kwarg, mirroring the convention
used by the DMControl wrappers (e.g. ``CheetahDMControlWrapper``):

    gym.make("swm/OpenApps-Calendar-v0", task="add_meeting_with_dennis")

``task`` may be:
  * ``None`` — no task bound, reward is always 0.0 (useful for data
    collection / world-modeling).
  * a task key (str) — resolved against
    ``openapps/config/tasks/all_tasks.yaml`` and Hydra-instantiated.
  * a ``Task`` instance — used directly, no yaml lookup.
"""

from gymnasium.envs import registration

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


def list_tasks(app_name: str | None = None) -> list[str]:
    """List task keys from ``all_tasks.yaml``, optionally filtered by app.

    Returns the keys you can pass as ``task=`` to ``gym.make``. Intended
    for discovery from notebooks / scripts.
    """
    from pathlib import Path

    from omegaconf import OmegaConf

    from .env import _TASK_CLASS_TO_APP

    here = Path(__file__).resolve()
    workspace_root = here.parents[4]
    tasks_yaml = workspace_root / "openapps" / "config" / "tasks" / "all_tasks.yaml"
    if not tasks_yaml.is_file():
        return []

    cfg = OmegaConf.load(tasks_yaml)
    keys: list[str] = []
    for k, v in cfg.items():
        if app_name is None:
            keys.append(k)
            continue
        cls = v.get("_target_", "").rsplit(".", 1)[-1]
        if _TASK_CLASS_TO_APP.get(cls) == app_name:
            keys.append(k)
    return keys


__all__ = ["OpenAppsEnv", "list_tasks"]
