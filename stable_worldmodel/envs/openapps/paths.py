"""Filesystem paths into the sibling ``openapps/`` monorepo.

Single source of truth for the ``parents[4]`` walk so env.py, server.py,
and __init__.py don't each recompute it. The swm package lives at
``openapps-swm/stable-worldmodel/stable_worldmodel/envs/openapps/`` and
resolves configs under ``openapps-swm/openapps/config/`` — four parents up.

Also runs :func:`ensure_openapps_importable` at import time so that any
module doing ``from open_apps...`` can safely be imported top-level
after ``from .paths import ...``.
"""

import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
OPENAPPS_ROOT = WORKSPACE_ROOT / 'openapps'
OPENAPPS_CONFIG = OPENAPPS_ROOT / 'config'
TASKS_YAML = OPENAPPS_CONFIG / 'tasks' / 'all_tasks.yaml'


def ensure_openapps_importable() -> None:
    """Make sure ``open_apps`` and its internal ``src.*`` imports work.

    First try a plain ``import open_apps`` — if it succeeds (e.g. openapps
    was pip-installed or is already on ``sys.path``), we're done. Only
    fall back to patching the monorepo layout if the import fails.

    Individual app modules inside openapps do ``from src.open_apps...``
    internally, which requires ``openapps/`` itself on the path too.
    """
    try:
        import open_apps  # noqa: F401
    except ImportError:
        src = OPENAPPS_ROOT / 'src'
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
        if OPENAPPS_ROOT.is_dir() and str(OPENAPPS_ROOT) not in sys.path:
            sys.path.insert(0, str(OPENAPPS_ROOT))
        return

    # open_apps imported fine, but `from src.open_apps...` inside apps
    # still requires the openapps/ repo root on sys.path if it exists.
    if OPENAPPS_ROOT.is_dir() and str(OPENAPPS_ROOT) not in sys.path:
        sys.path.insert(0, str(OPENAPPS_ROOT))


# Run at import time so any module that imports from .paths (including
# env.py and server.py) can safely use top-level ``from open_apps...``.
ensure_openapps_importable()


# OpenApps app key (as used by OpenAppsEnv / the _APPS list) → the
# corresponding directory name under ``openapps/config/apps/`` and the
# URL path the FastHTML server mounts the app at. A few app keys don't
# match either (``messages`` → ``messenger`` directory; ``map`` → ``maps``
# URL + directory; ``codeeditor`` wants a trailing slash), so both sides
# of the rename live here together.
APP_TABLE: dict[str, dict[str, str]] = {
    'todo': {'config_dir': 'todo', 'url_path': '/todo'},
    'calendar': {'config_dir': 'calendar', 'url_path': '/calendar'},
    'messages': {'config_dir': 'messenger', 'url_path': '/messages'},
    'codeeditor': {'config_dir': 'code_editor', 'url_path': '/codeeditor/'},
    'map': {'config_dir': 'maps', 'url_path': '/maps'},
}


def config_dir_for(app_name: str) -> str:
    """Return the ``openapps/config/apps/<dir>`` name for an app key."""
    entry = APP_TABLE.get(app_name)
    return entry['config_dir'] if entry else app_name


def url_path_for(app_name: str) -> str:
    """Return the URL path the FastHTML server exposes for an app key."""
    entry = APP_TABLE.get(app_name)
    return entry['url_path'] if entry else f'/{app_name}'


def app_group_dir(app_name: str, group: str) -> Path:
    """Return ``openapps/config/apps/<app>/<group>/`` (variant discovery)."""
    return OPENAPPS_CONFIG / 'apps' / config_dir_for(app_name) / group


def app_content_yaml(app_name: str, content: str) -> Path:
    """Path to a specific content variant yaml (used for map pin overrides)."""
    return (
        OPENAPPS_CONFIG
        / 'apps'
        / config_dir_for(app_name)
        / 'content'
        / f'{content}.yaml'
    )
