"""OpenApps gymnasium envs for stable-worldmodel.

Tasks are selected at ``gym.make`` time via the ``task=`` kwarg:
    gym.make("swm/OpenApps-Calendar-v0", task="add_meeting_with_dennis")

``task`` may be:
  * ``None`` — no task bound, reward always 0.0 (data collection).
  * a task key (str) — resolved via :func:`open_apps.tasks.load_task`.
  * an :class:`open_apps.tasks.Task` instance — used directly.
"""

from .agent_policy import DummyPolicy, VLMPolicy
from .env import OpenAppsEnv


__all__ = ['DummyPolicy', 'OpenAppsEnv', 'VLMPolicy']
