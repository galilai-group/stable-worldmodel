"""OpenApps gymnasium envs for stable-worldmodel.

A task is selected at ``gym.make`` time via the ``task=`` kwarg: a task key
string (scored server-side) or ``None`` for reward-free data collection.
"""

import mcp  # noqa: F401
import open_apps  # noqa: F401
import playwright  # noqa: F401

from .agent_policy import DummyPolicy, VLMPolicy
from .env import OpenAppsEnv


__all__ = ['DummyPolicy', 'OpenAppsEnv', 'VLMPolicy']
