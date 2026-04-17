"""OpenAppsEnv — gymnasium environment wrapping Playwright + FastHTML.

This is the core gym.Env implementation for OpenApps in swm. It manages
the full lifecycle of the FastHTML server (in-process background thread)
and Playwright browser, presenting a standard gym interface with
MultiDiscrete(NUM_ACTIONS, GRID_X, GRID_Y) actions and screenshot observations.

Action space: MultiDiscrete([3, 32, 20])
  action[0] in {0, 1, 2}   — 0=click, 1=scroll_down, 2=scroll_up
  action[1] in {0..31}     — x grid cell (32px per cell)
  action[2] in {0..19}     — y grid cell (32px per cell)

The FastHTML server runs in a daemon thread within the same process,
which means resets can be done via direct Python calls (no HTTP needed
for state management).
"""

import io
import shutil

import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces
from loguru import logger
from PIL import Image

from open_apps.tasks.add_tasks_to_browsergym import get_current_state
from open_apps.tasks.tasks import Task

from .executor import (
    GRID_X,
    GRID_Y,
    NUM_ACTIONS,
    VIEWPORT_HEIGHT,
    VIEWPORT_WIDTH,
    action_multidiscrete_to_playwright,
)
from .server import (
    _init_app,
    _load_hydra_config,
    pick_free_port,
    reset_app,
    start_server_thread,
    wait_until_healthy,
)

# Lazy imports for Playwright (only needed at runtime)
_playwright_ctx = None
_browser = None


def _get_playwright():
    """Lazy-init Playwright to avoid import cost at registration time."""
    global _playwright_ctx, _browser
    if _browser is None:
        from playwright.sync_api import sync_playwright

        _playwright_ctx = sync_playwright().start()
        _browser = _playwright_ctx.chromium.launch(headless=True)
    return _browser


class OpenAppsEnv(gym.Env):
    """Gymnasium environment for OpenApps browser-based UI tasks.

    Wraps a FastHTML web server and a headless Chromium browser managed
    via Playwright. Actions are MultiDiscrete([NUM_ACTIONS, GRID_X, GRID_Y])
    integer vectors encoding click or scroll operations on a pixel grid.
    Observations are screenshots.

    Action space: MultiDiscrete([3, 32, 20])
      action[0]: action type — 0=click, 1=scroll_down, 2=scroll_up
      action[1]: x grid cell [0, GRID_X-1], maps to pixel x = cell * 32 + 16
      action[2]: y grid cell [0, GRID_Y-1], maps to pixel y = cell * 32 + 16

    The server runs in a background daemon thread within this process.
    Resets are direct Python calls — no HTTP round-trip needed.

    Args:
        app_name: Which OpenApps app to target (e.g. "todo", "calendar").
        task: An OpenApps Task object whose ``check_if_task_is_complete``
            method is used for reward computation. When provided,
            ``task_description`` defaults to the task's goal string.
        task_description: Natural-language task goal (falls back to task.goal).
        port: Port for the FastHTML server.
        max_steps: Maximum steps per episode before truncation.
        render_mode: "rgb_array" (default) for numpy screenshots.
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 5,
    }
    reward_range = (0.0, 1.0)

    VIEWPORT_WIDTH = VIEWPORT_WIDTH
    VIEWPORT_HEIGHT = VIEWPORT_HEIGHT
    DEFAULT_IMAGE_SHAPE = (VIEWPORT_HEIGHT, VIEWPORT_WIDTH)

    def __init__(
        self,
        app_name: str = "todo",
        task: Task | None = None,
        task_description: str = "",
        port: int | None = None,
        max_steps: int = 50,
        render_mode: str = "rgb_array",
    ):
        super().__init__()

        self.app_name = app_name
        self.env_name = f"OpenApps-{app_name}"
        self.task = task
        self.task_description = task_description or (task.goal if task else "")
        self.port = port if port is not None else pick_free_port()
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.base_url = f"http://127.0.0.1:{self.port}"
        self._step_count = 0
        self._last_screenshot = None
        self._initial_state: dict | None = None

        # Spaces
        self.observation_space = spaces.Box(
            0, 255, (VIEWPORT_HEIGHT, VIEWPORT_WIDTH, 3), np.uint8
        )
        self.action_space = spaces.MultiDiscrete([NUM_ACTIONS, GRID_X, GRID_Y])

        # ── Start server in-process ──────────────────────────────────
        self._cfg, self._tmp_logs_dir = _load_hydra_config()
        self._asgi_app, self._cfg = _init_app(self._cfg)
        self._server_thread, self._server = start_server_thread(
            self._asgi_app, port=self.port
        )
        wait_until_healthy(self.base_url)

        # ── Start browser ────────────────────────────────────────────
        browser = _get_playwright()
        self._context = browser.new_context(
            viewport={
                "width": VIEWPORT_WIDTH,
                "height": VIEWPORT_HEIGHT,
            }
        )
        self._page = self._context.new_page()

    # ── Reset ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed, options=options)
        self._step_count = 0

        reset_app(self.app_name, self._cfg.apps)
        self._initial_state = get_current_state(self.base_url)

        self._page.goto(f"{self.base_url}/{self.app_name}")
        self._page.wait_for_load_state("networkidle")

        obs = self._capture_screenshot()
        info = {
            "pixels": obs,
            "env_name": self.env_name,
            "goal": self.task_description,
        }
        return obs, info

    # ── Step ──────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        """Execute one action and return the new observation.

        Args:
            action: MultiDiscrete int64 array [action_type, grid_x, grid_y].

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        self._step_count += 1

        action_desc = action_multidiscrete_to_playwright(action, self._page)
        logger.debug(f"Step {self._step_count}: {action_desc}")

        self._page.wait_for_timeout(300)

        obs = self._capture_screenshot()
        reward = self._compute_reward()
        terminated = reward == 1.0
        truncated = self._step_count >= self.max_steps

        info = {
            "pixels": obs,
            "env_name": self.env_name,
            "goal": self.task_description,
            "_action_desc": action_desc,
        }

        return obs, reward, terminated, truncated, info

    # ── Render ────────────────────────────────────────────────────────

    def render(self) -> np.ndarray:
        """Return the current screenshot as numpy (H, W, 3) uint8."""
        if self._last_screenshot is not None:
            return self._last_screenshot
        return self._capture_screenshot()

    def _capture_screenshot(self) -> np.ndarray:
        """Take a Playwright screenshot and convert to numpy."""
        png_bytes = self._page.screenshot()
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        self._last_screenshot = arr
        return arr

    # ── Reward ────────────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        """Compute reward by delegating to the OpenApps Task.

        Returns 1.0 on task completion, 0.0 otherwise.
        """
        if self.task is None or self._initial_state is None:
            return 0.0

        try:
            current_state = get_current_state(self.base_url)
            return (
                1.0
                if self.task.check_if_task_is_complete(
                    self._initial_state, current_state
                )
                else 0.0
            )
        except Exception as e:
            logger.warning(f"Reward computation failed: {e}")
            return 0.0

    # ── Cleanup ───────────────────────────────────────────────────────

    def close(self):
        """Tear down browser, uvicorn server, and tmp config dir."""
        try:
            if getattr(self, "_page", None):
                self._page.close()
            if getattr(self, "_context", None):
                self._context.close()
        except Exception as e:
            logger.debug(f"Browser close failed: {e}")

        server = getattr(self, "_server", None)
        thread = getattr(self, "_server_thread", None)
        if server is not None:
            server.should_exit = True
            if thread is not None:
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logger.warning("uvicorn thread did not exit within 5s")

        tmp_dir = getattr(self, "_tmp_logs_dir", None)
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        super().close()
