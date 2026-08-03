"""Optional MineRL environment entry point.

The data converter does not need a running Minecraft client.  This small,
lazy factory is provided for the later planning/evaluation stage of MineJEPA-
SWM, where an installed MineRL task is required.
"""

from __future__ import annotations

from typing import Any


def make_minerl_env(environment: str, **kwargs: Any):
    """Construct a MineRL Gym environment only when the extra is installed."""
    try:
        import gym
        import minerl  # noqa: F401 - importing registers MineRL environments.
    except ImportError as exc:
        raise ImportError(
            'MineRL environments require the optional `minerl` extra. '
            "Install it with `pip install 'stable-worldmodel[minerl]'`."
        ) from exc
    return gym.make(environment, **kwargs)


__all__ = ['make_minerl_env']
