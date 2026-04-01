"""FastHTML server lifecycle management for OpenApps.

Starts the FastHTML server as a subprocess and provides health-check
utilities. Used by OpenAppsEnv.__init__() and the tools/ scripts.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from loguru import logger


def _find_serve_script() -> Path:
    """Locate tools/serve_apps.py relative to the workspace root."""
    here = Path(__file__).resolve()
    # server.py -> openapps/ -> envs/ -> stable_worldmodel/ -> stable-worldmodel/ -> openapps-swm/
    workspace_root = here.parents[4]
    serve_script = workspace_root / "tools" / "serve_apps.py"
    if serve_script.exists():
        return serve_script

    # Fallback: check CWD-based paths
    for candidate in [
        Path.cwd() / "tools" / "serve_apps.py",
        Path.cwd().parent / "tools" / "serve_apps.py",
    ]:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Cannot find tools/serve_apps.py. "
        "Make sure you're running from the openapps-swm workspace root."
    )


def start_server_process(
    config_path: str | None = None,
    port: int = 5001,
    host: str = "127.0.0.1",
) -> subprocess.Popen:
    """Start the FastHTML server as a subprocess.

    Launches tools/serve_apps.py which uses Hydra compose to load configs
    and start just the web apps (no agent/BrowserGym overhead).

    Args:
        config_path: Path to config override. If None, uses default.
        port: Port to serve on.
        host: Host to bind to.

    Returns:
        The subprocess.Popen object for the running server.
    """
    serve_script = _find_serve_script()

    cmd = [
        sys.executable,
        str(serve_script),
        "--port", str(port),
        "--host", host,
    ]
    if config_path:
        cmd.extend(["--config", config_path])

    logger.info(f"Starting FastHTML server on {host}:{port}")
    logger.debug(f"  cmd: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def wait_until_healthy(
    base_url: str = "http://127.0.0.1:5001",
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
                logger.info(f"FastHTML server healthy at {base_url} ({elapsed:.1f}s)")
                return True
        except requests.ConnectionError as e:
            last_error = e
        except requests.Timeout:
            last_error = "timeout"
        time.sleep(poll_interval)

    raise TimeoutError(
        f"FastHTML server at {base_url} did not become healthy "
        f"within {timeout}s (last error: {last_error})"
    )


def stop_server(proc: subprocess.Popen) -> None:
    """Gracefully stop the server subprocess.

    Args:
        proc: The server subprocess to stop.
    """
    if proc and proc.poll() is None:
        logger.info("Stopping FastHTML server")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Server did not stop gracefully, killing")
            proc.kill()
            proc.wait()
