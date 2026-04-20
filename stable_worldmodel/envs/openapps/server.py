"""In-process FastHTML server for OpenApps.

Starts the OpenApps FastHTML server in a background thread within the
same Python process. This avoids the need for an external serve script
and allows direct Python-level reset (no HTTP round-trip).

Used by OpenAppsEnv.__init__() and the tools/ scripts.
"""

import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path

import requests
import uvicorn
from fastlite import database as fl_database
from hydra import compose, initialize_config_dir
from loguru import logger

# Importing .paths also runs ``ensure_openapps_importable()`` as a side
# effect, which must happen before we can import ``open_apps.*``.
from .paths import OPENAPPS_CONFIG, app_group_dir  # noqa: I001
from open_apps.apps.start_page.main import (  # noqa: E402
    AVAILABLE_APPS,
    app as _fasthtml_app,
    initialize_routes_and_configure_task,
)


def _load_hydra_config(extra_overrides: list[str] | None = None):
    """Load the OpenApps Hydra config using the compose API.

    Returns ``(cfg, tmp_logs_dir)`` — the tmp dir is returned so the
    caller can clean it up on ``close()``.

    Args:
        extra_overrides: Additional Hydra override strings appended after
            the defaults. Used by the variation space to swap appearance/
            content groups per-reset, e.g.
            ``["apps/todo/appearance=dark_theme", "seed=17"]``.
    """
    tmp_logs = tempfile.mkdtemp(prefix='openapps_logs_')

    overrides = [
        f'logs_dir={tmp_logs}',
        'use_wandb=False',
    ]
    if extra_overrides:
        overrides.extend(extra_overrides)

    with initialize_config_dir(
        config_dir=str(OPENAPPS_CONFIG), version_base=None
    ):
        cfg = compose(config_name='config', overrides=overrides)

    return cfg, tmp_logs


# ── Variation discovery ─────────────────────────────────────────────


def discover_variants(app_name: str, group: str) -> list[str]:
    """List Hydra variant yamls available for an app's group.

    Returns a sorted list of variant stems (without ``.yaml``) found in
    ``openapps/config/apps/{app}/{group}/``. ``"default"`` is always
    placed first so it maps to index 0. Returns ``["default"]`` as a
    fallback if the directory is missing.

    Args:
        app_name: OpenApps app key (e.g. ``"todo"``, ``"messages"``).
        group: Variant group name (``"appearance"`` or ``"content"``).
    """
    group_dir = app_group_dir(app_name, group)

    if not group_dir.is_dir():
        return ['default']

    stems = sorted(p.stem for p in group_dir.glob('*.yaml'))
    if 'default' in stems:
        stems.remove('default')
        return ['default'] + stems
    return stems or ['default']


def _init_app(cfg):
    """Wire up all OpenApps routes on the shared FastHTML ``app`` object.

    Returns the ASGI ``app`` ready to be served, plus the full config
    (needed for resets).
    """
    initialize_routes_and_configure_task(cfg.apps)
    return _fasthtml_app, cfg


# ── Reset helpers ────────────────────────────────────────────────────

# Map app module names → config attribute keys for database paths
_APP_CFG_KEYS = {
    'open_apps.apps.todo_app': 'todo',
    'open_apps.apps.calendar_app': 'calendar',
    'open_apps.apps.messenger_app': 'messenger',
    'open_apps.apps.codeeditor_app': 'code_editor',
    'open_apps.apps.map_app': 'maps',
}


def _drop_app_tables(module, apps_cfg):
    """Drop all SQLite tables for an app so ``set_environment`` can re-seed."""
    cfg_key = _APP_CFG_KEYS.get(module.__name__)
    if cfg_key is None:
        return

    sub_cfg = getattr(apps_cfg, cfg_key, None)
    if sub_cfg is None or not hasattr(sub_cfg, 'database_path'):
        return

    db_path = sub_cfg.database_path
    db = fl_database(db_path)
    for table_name in db.table_names():
        db[table_name].drop()
    logger.debug(f'Dropped tables in {db_path}')


def reset_app(app_name: str, apps_cfg) -> None:
    """Reset a single app's state by dropping its storage and re-seeding.

    This is a direct Python call — no HTTP needed.  It handles both
    database-backed apps (todo, calendar, messenger, map) and
    filesystem-backed apps (codeeditor).

    Args:
        app_name: The app key, e.g. ``"todo"``, ``"calendar"``.
        apps_cfg: The ``cfg.apps`` Hydra config object.
    """
    if app_name not in AVAILABLE_APPS:
        raise ValueError(f'Unknown app: {app_name}')

    module_path, getter_func = AVAILABLE_APPS[app_name]
    module = __import__(module_path, fromlist=[getter_func])

    # Code editor is filesystem-backed, not SQLite-backed: wipe the
    # directory and skip _drop_app_tables (treating the folder as a DB
    # file would create a SQLite file at that path, which then confuses
    # the re-seed in set_environment).
    if app_name == 'codeeditor':
        if hasattr(apps_cfg, 'code_editor') and hasattr(
            apps_cfg.code_editor, 'database_path'
        ):
            folder = Path(apps_cfg.code_editor.database_path)
            if folder.exists():
                shutil.rmtree(folder)
    else:
        _drop_app_tables(module, apps_cfg)

    # Re-seed from config
    if hasattr(module, 'set_environment'):
        module.set_environment(apps_cfg)

    logger.debug(f'Reset complete for {app_name}')


def reset_all_apps(apps_cfg) -> None:
    """Reset every registered app."""
    for app_name in AVAILABLE_APPS:
        try:
            reset_app(app_name, apps_cfg)
        except Exception as e:
            logger.warning(f'Failed to reset {app_name}: {e}')


# ── Server lifecycle ─────────────────────────────────────────────────


def pick_free_port(host: str = '127.0.0.1') -> int:
    """Ask the OS for an unused TCP port on ``host``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def start_server_thread(
    asgi_app,
    port: int = 5001,
    host: str = '127.0.0.1',
) -> tuple[threading.Thread, 'uvicorn.Server']:
    """Start uvicorn serving ``asgi_app`` in a daemon thread.

    Constructs a ``uvicorn.Server`` explicitly (instead of ``uvicorn.run``)
    so the caller retains a handle with a ``should_exit`` flag — letting
    ``OpenAppsEnv.close()`` shut the server down cleanly and free the
    port on the next run.

    Args:
        asgi_app: The FastHTML/Starlette ASGI application.
        port: Port to listen on.
        host: Host to bind to.

    Returns:
        ``(thread, server)``. Set ``server.should_exit = True`` to stop.
    """
    config = uvicorn.Config(
        asgi_app, host=host, port=port, log_level='warning'
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    logger.info(f'FastHTML server thread started on {host}:{port}')
    return thread, server


def wait_until_healthy(
    base_url: str = 'http://127.0.0.1:5001',
    timeout: float = 60.0,
    poll_interval: float = 1.0,
) -> bool:
    """Poll the server until it responds to HTTP requests.

    Args:
        base_url: The server URL to check.
        timeout: Maximum seconds to wait.
        poll_interval: Seconds between poll attempts.

    Returns:
        True if server became healthy within timeout.

    Raises:
        TimeoutError: If server didn't respond within timeout.
    """
    start = time.monotonic()
    last_error = None
    while time.monotonic() - start < timeout:
        try:
            resp = requests.get(base_url, timeout=3)
            if resp.status_code < 500:
                elapsed = time.monotonic() - start
                logger.info(
                    f'FastHTML server healthy at {base_url} ({elapsed:.1f}s)'
                )
                return True
        except requests.ConnectionError as e:
            last_error = e
        except requests.Timeout:
            last_error = 'timeout'
        time.sleep(poll_interval)

    raise TimeoutError(
        f'FastHTML server at {base_url} did not become healthy '
        f'within {timeout}s (last error: {last_error})'
    )
