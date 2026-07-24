"""stable_worldmodel public API.

Submodules and top-level attributes are loaded lazily (PEP 562) so that
``import stable_worldmodel`` stays fast: torch, torchvision, and lancedb are
only imported on first access to the attribute that needs them.

Only :mod:`stable_worldmodel.envs` is imported eagerly — it registers the
``swm/*`` gymnasium environment IDs, which must exist before any
``gym.make("swm/...")`` call.
"""

import importlib
from typing import TYPE_CHECKING

from stable_worldmodel import envs


_LAZY_SUBMODULES = {
    'data',
    'planning',
    'policy',
    'spaces',
    'utils',
    'wm',
    'wrapper',
}

_LAZY_ATTRS = {
    'World': ('stable_worldmodel.world', 'World'),
    'PlanConfig': ('stable_worldmodel.policy', 'PlanConfig'),
    'pretraining': ('stable_worldmodel.utils', 'pretraining'),
}

if TYPE_CHECKING:
    from stable_worldmodel import (
        buffer,
        data,
        planning,
        policy,
        spaces,
        utils,
        wm,
        wrapper,
    )
    from stable_worldmodel.policy import PlanConfig
    from stable_worldmodel.utils import pretraining
    from stable_worldmodel.world import World

__all__ = [
    'World',
    'PlanConfig',
    'pretraining',
    'buffer',
    'data',
    'envs',
    'planning',
    'policy',
    'spaces',
    'utils',
    'wm',
    'wrapper',
]


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(f'.{name}', __name__)
        globals()[name] = mod
        return mod
    if name in _LAZY_ATTRS:
        modpath, attrname = _LAZY_ATTRS[name]
        attr = getattr(importlib.import_module(modpath), attrname)
        globals()[name] = attr
        return attr
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))
