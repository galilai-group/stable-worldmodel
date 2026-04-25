"""OpenApps gymnasium envs for stable-worldmodel.

Importing this module registers one gym env id per OpenApps application
(``swm/OpenApps-Todo-v0``, ``-Calendar-v0``, ...). The task is selected
at ``gym.make`` time via the ``task=`` kwarg:

    gym.make("swm/OpenApps-Calendar-v0", task="add_meeting_with_dennis")

``task`` may be:
  * ``None`` — no task bound, reward is always 0.0 (useful for data
    collection / world-modeling).
  * a task key (str) — resolved against
    ``openapps/config/tasks/all_tasks.yaml`` and Hydra-instantiated.
  * a ``Task`` instance — used directly, no yaml lookup.

Registration is kept lazy, users opt in with a one-liner:
    import stable_worldmodel.envs.openapps  # noqa: F401
"""

from gymnasium.envs import registration

from stable_worldmodel.envs import WORLDS

from .agent_policy import DummyPolicy, VLMPolicy
from .env import OpenAppsEnv, list_tasks


_APPS = ('todo', 'calendar', 'messages', 'codeeditor', 'map')

for _app in _APPS:
    _env_id = f'swm/OpenApps-{_app.capitalize()}-v0'
    registration.register(
        id=_env_id,
        entry_point='stable_worldmodel.envs.openapps.env:OpenAppsEnv',
        kwargs={'app_name': _app},
    )
    WORLDS.add(_env_id)


__all__ = ['DummyPolicy', 'OpenAppsEnv', 'VLMPolicy', 'list_tasks']
