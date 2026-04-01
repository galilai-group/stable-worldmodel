"""OpenAppsEnv — gymnasium environment wrapping Playwright + FastHTML.

This is the core gym.Env implementation for OpenApps in swm. It manages
the full lifecycle of the FastHTML server and Playwright browser,
presenting a standard gym interface with Box(5,) actions and screenshot
observations.

From swm's perspective this looks identical to PushTEnv — the server and
browser complexity is fully encapsulated.
"""

import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger

from .executor import (
    VIEWPORT_HEIGHT,
    VIEWPORT_WIDTH,
    action_vec_to_playwright,
)
from .server import start_server_process, stop_server, wait_until_healthy

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
    via Playwright. Actions are Box(5,) float vectors encoding click,
    double-click, or scroll operations. Observations are screenshots.

    Args:
        app_name: Which OpenApps app to target (e.g. "todo", "calendar").
        task_description: Natural-language task goal for reward computation.
        config_path: Path to OpenApps Hydra config file. None for default.
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
        task_description: str = "",
        config_path: str | None = None,
        port: int = 5001,
        max_steps: int = 50,
        render_mode: str = "rgb_array",
        target_state: dict | None = None,
    ):
        super().__init__()

        self.app_name = app_name
        self.task_description = task_description
        self.config_path = config_path
        self.port = port
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.target_state = target_state or {}

        self.base_url = f"http://127.0.0.1:{port}"
        self._step_count = 0
        self._last_screenshot = None

        # Spaces — must match design notes exactly
        self.observation_space = spaces.Box(
            0, 255, (VIEWPORT_HEIGHT, VIEWPORT_WIDTH, 3), np.uint8
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )

        # Start server
        self._server_proc = start_server_process(
            config_path=config_path, port=port
        )
        wait_until_healthy(self.base_url)

        # Start browser
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
        """Reset the environment to initial state.

        Calls reset_all_apps via Python import (per design notes) to
        re-seed all app databases, then navigates the browser to the
        target app page.
        """
        super().reset(seed=seed, options=options)
        self._step_count = 0

        # Reset all apps via the generic loop (design notes §5.2)
        self._reset_apps()

        # Navigate to the target app
        self._page.goto(f"{self.base_url}/{self.app_name}")
        self._page.wait_for_load_state("networkidle")

        obs = self._capture_screenshot()
        info = {"pixels": obs}
        return obs, info

    def _reset_apps(self):
        """Reset all OpenApps databases using the generic loop.

        Per design notes: calls reset_all_apps() which re-runs the
        set_environment loop from start_page/main.py. No per-app
        HTTP endpoints needed.
        """
        try:
            from open_apps.apps.start_page.main import (
                AVAILABLE_APPS,
            )
            from open_apps.configs import load_config

            config = load_config(self.config_path)

            # Clean code editor filesystem before reset (design notes §5.3)
            if hasattr(config, "codeeditor") and hasattr(
                config.codeeditor, "folder_path"
            ):
                folder = Path(config.codeeditor.folder_path)
                if folder.exists():
                    shutil.rmtree(folder)

            # Generic reset loop — mirrors start_page/main.py init
            for app_name, (module_path, getter_func) in AVAILABLE_APPS.items():
                module = __import__(module_path, fromlist=[getter_func])
                if hasattr(module, "set_environment"):
                    module.set_environment(config)
                    logger.debug(f"Reset app: {app_name}")

        except Exception as e:
            logger.warning(f"reset_all_apps failed: {e}. Falling back to page reload.")
            self._page.reload()

    # ── Step ──────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        """Execute one action and return the new observation.

        Args:
            action: Box(5,) float vector.

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        self._step_count += 1

        # Execute the action via Playwright
        action_desc = action_vec_to_playwright(action, self._page)
        logger.debug(f"Step {self._step_count}: {action_desc}")

        # Brief wait for UI to settle after action
        self._page.wait_for_timeout(300)

        obs = self._capture_screenshot()
        reward = self._compute_reward()
        terminated = reward == 1.0
        truncated = self._step_count >= self.max_steps

        info = {
            "pixels": obs,
            "action_desc": action_desc,
            "step": self._step_count,
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
        # Decode PNG bytes to numpy array
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        self._last_screenshot = arr
        return arr

    # ── Reward ────────────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        """Compute reward by comparing current app state to target.

        GETs the app's _all endpoint (e.g. /todo_all) and uses DeepDiff
        to compare against the target state. Returns 1.0 on match, 0.0
        otherwise.
        """
        if not self.target_state:
            return 0.0

        try:
            import requests
            from deepdiff import DeepDiff

            # Map app names to their state endpoints
            state_endpoints = {
                "todo": "/todo_all",
                "calendar": "/calendar_all",
                "messages": "/messages_all",
                "codeeditor": "/codeeditor_all",
                "map": "/maps/landmarks",
            }

            endpoint = state_endpoints.get(self.app_name)
            if not endpoint:
                return 0.0

            resp = requests.get(f"{self.base_url}{endpoint}", timeout=5)
            current_state = resp.json()

            diff = DeepDiff(
                current_state,
                self.target_state,
                ignore_order=True,
            )
            return 0.0 if diff else 1.0

        except Exception as e:
            logger.warning(f"Reward computation failed: {e}")
            return 0.0

    # ── Cleanup ───────────────────────────────────────────────────────

    def close(self):
        """Clean up browser and server resources."""
        try:
            if hasattr(self, "_page") and self._page:
                self._page.close()
            if hasattr(self, "_context") and self._context:
                self._context.close()
        except Exception:
            pass

        if hasattr(self, "_server_proc"):
            stop_server(self._server_proc)

        super().close()
