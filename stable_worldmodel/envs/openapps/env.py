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
from stable_worldmodel import spaces as swm_space

from .executor import (
    GRID_X,
    GRID_Y,
    NUM_ACTIONS,
    VIEWPORT_HEIGHT,
    VIEWPORT_WIDTH,
    action_multidiscrete_to_playwright,
)
from .server import (
    _APP_TO_CONFIG_DIR,
    _init_app,
    _load_hydra_config,
    discover_variants,
    pick_free_port,
    reset_app,
    start_server_thread,
    wait_until_healthy,
)


# Viewport choice tables — indexed by variation_space["browser"]["viewport"].
VIEWPORT_CHOICES_W = [800, 1024, 1280, 1920]
VIEWPORT_CHOICES_H = [600, 640, 720, 1080]

# Initial scroll offset applied to every app after page.goto. All apps
# are shorter than 600px at the default viewport, so larger values are
# mostly clamped by the browser — which is fine: we want "scroll bottom"
# to end up at the actual bottom, not sail off into empty space.
SCROLL_Y_CHOICES = [0, 100, 300, 600]

# ``map.city`` variation: indexes this list to pick the lat/lng that
# gets written into ``cfg.apps.maps.init_location`` on reset. Names are
# for debugging/logging; coordinates are the city center as used by the
# OpenStreetMap tile server.
MAP_CITY_CHOICES: list[tuple[str, list[float]]] = [
    ("NYC",        [40.7831, -73.9712]),
    ("Paris",      [48.8566, 2.3522]),
    ("Tokyo",      [35.6762, 139.6503]),
    ("Berlin",     [52.5200, 13.4050]),
    ("Sydney",     [-33.8688, 151.2093]),
    ("Cairo",      [30.0444, 31.2357]),
    ("Sao Paulo",  [-23.5505, -46.6333]),
    ("Moscow",     [55.7558, 37.6173]),
]


# app_name → URL path. Most apps mount at /<app_name>, but the map app
# registers its routes under /maps (plural) and codeeditor wants a
# trailing slash. Override here so ``self._page.goto(…)`` always lands
# on a real route rather than a 404.
_APP_URL_PATH = {
    "map": "/maps",
    "codeeditor": "/codeeditor/",
}

def _read_variant_saved_places(content: str) -> list | None:
    """Return the ``saved_places`` list declared in the map content yaml.

    Upstream yamls use Hydra's ``+saved_places:`` (add-key) which silently
    no-ops when the key already exists via defaults. We read the yaml
    file directly so the sidebar pins swap per-variant even though Hydra
    never merges them in.
    """
    import yaml as _yaml
    from pathlib import Path as _Path

    here = _Path(__file__).resolve()
    workspace_root = here.parents[4]
    yaml_path = (
        workspace_root / "openapps" / "config" / "apps" / "maps" / "content"
        / f"{content}.yaml"
    )
    if not yaml_path.is_file():
        return None
    try:
        raw = _yaml.safe_load(yaml_path.read_text())
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    # ``+saved_places`` (Hydra add-key form) or ``saved_places`` (plain).
    for key in ("+saved_places", "saved_places"):
        if key in raw and isinstance(raw[key], list):
            return raw[key]
    return None


# Task class → OpenApps app key. Used to validate that a task key resolved
# from the yaml is compatible with the env's app_name. Kept here (rather
# than in __init__.py) so it stays colocated with the env that uses it.
_TASK_CLASS_TO_APP = {
    "AddEventTask": "calendar",
    "RemoveEventTask": "calendar",
    "AddToDoTask": "todo",
    "MarkToDoDoneTask": "todo",
    "SendMessageTask": "messages",
    "SavePlaceTask": "map",
}


def _load_task_from_yaml(task_key: str, app_name: str) -> Task:
    """Resolve a task key against ``openapps/config/tasks/all_tasks.yaml``.

    Raises ValueError if the key is unknown or resolves to a Task whose
    class doesn't match ``app_name``.
    """
    from pathlib import Path

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    here = Path(__file__).resolve()
    workspace_root = here.parents[4]
    tasks_yaml = workspace_root / "openapps" / "config" / "tasks" / "all_tasks.yaml"
    if not tasks_yaml.is_file():
        raise FileNotFoundError(f"Tasks yaml not found at {tasks_yaml}")

    cfg = OmegaConf.load(tasks_yaml)
    if task_key not in cfg:
        raise ValueError(
            f"Unknown task key {task_key!r}. Available: {list(cfg.keys())}"
        )

    task_cfg = cfg[task_key]
    target = task_cfg.get("_target_", "")
    task_class = target.rsplit(".", 1)[-1]
    expected_app = _TASK_CLASS_TO_APP.get(task_class)
    if expected_app is not None and expected_app != app_name:
        raise ValueError(
            f"Task {task_key!r} ({task_class}) targets app "
            f"{expected_app!r}, but env was constructed with "
            f"app_name={app_name!r}"
        )
    return instantiate(task_cfg)


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
        task: Either an OpenApps ``Task`` instance, or a task key (str)
            referencing an entry in ``openapps/config/tasks/all_tasks.yaml``
            which will be Hydra-instantiated. ``None`` disables reward
            (always 0.0) — useful for data collection.
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
        task: "Task | str | None" = None,
        task_description: str = "",
        port: int | None = None,
        max_steps: int = 50,
        render_mode: str = "rgb_array",
    ):
        super().__init__()

        self.app_name = app_name
        self.env_name = f"OpenApps-{app_name}"
        if isinstance(task, str):
            task = _load_task_from_yaml(task, app_name=app_name)
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

        # ── Variation space ──────────────────────────────────────────
        # Discrete dimensions index into _appearance_variants / _content_variants
        # so the variation values stay portable; reset() translates them into
        # Hydra group overrides (e.g. apps/todo/appearance=dark_theme).
        self._appearance_variants = discover_variants(app_name, "appearance")
        self._content_variants = discover_variants(app_name, "content")

        spaces_dict: dict = {
            "appearance": swm_space.Dict(
                {
                    "theme": swm_space.Discrete(
                        len(self._appearance_variants), init_value=0
                    ),
                    "font_scale": swm_space.Box(
                        low=0.8,
                        high=1.4,
                        shape=(1,),
                        dtype=np.float32,
                        init_value=np.array([1.0], dtype=np.float32),
                    ),
                }
            ),
            "content": swm_space.Dict(
                {
                    "variant": swm_space.Discrete(
                        len(self._content_variants), init_value=0
                    ),
                    "seed": swm_space.Discrete(2**31 - 1, init_value=233),
                }
            ),
            "browser": swm_space.Dict(
                {
                    "viewport": swm_space.MultiDiscrete(
                        [len(VIEWPORT_CHOICES_W), len(VIEWPORT_CHOICES_H)],
                        init_value=np.array([1, 1], dtype=np.int64),
                    ),
                    # Initial scroll offset (px) applied after page.goto.
                    # Indexes SCROLL_Y_CHOICES.
                    "scroll_y": swm_space.Discrete(
                        len(SCROLL_Y_CHOICES), init_value=0
                    ),
                }
            ),
        }

        # Per-app axes: only included in the variation space for envs
        # where they actually do something.
        if self.app_name == "map":
            spaces_dict["map"] = swm_space.Dict(
                {
                    # Index into MAP_CITY_CHOICES. Default 0 == NYC.
                    "city": swm_space.Discrete(
                        len(MAP_CITY_CHOICES), init_value=0
                    ),
                }
            )

        self.variation_space = swm_space.Dict(spaces_dict)

        # Keys resampled by default when reset() is called without an
        # explicit ``options['variation']`` list. Per-app axes are only
        # added if the env actually has them.
        default_vars: list[str] = [
            "appearance.theme",
            "content.variant",
            "content.seed",
            "browser.scroll_y",
        ]
        if self.app_name == "map":
            default_vars.append("map.city")
        self._default_variations = tuple(default_vars)

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
        """Reset the environment to initial state.

        Samples the variation space (per the standard swm protocol) and
        translates the result into:
          * Hydra group overrides for appearance/content (recompose + merge
            into the live ``app.config`` — routes read it dynamically per
            request, so no re-registration needed).
          * SQLite re-seed via ``reset_app`` (picks up new content).
          * Playwright viewport + CSS injection for browser-side knobs.
        """
        super().reset(seed=seed, options=options)
        self._step_count = 0

        swm_space.reset_variation_space(
            self.variation_space,
            seed=seed,
            options=options,
            default_variations=self._default_variations,
        )
        v = self.variation_space.value
        self._apply_variations(v)

        reset_app(self.app_name, self._cfg.apps)
        self._initial_state = get_current_state(self.base_url)

        url_path = _APP_URL_PATH.get(self.app_name, f"/{self.app_name}")
        self._page.goto(f"{self.base_url}{url_path}")
        self._page.wait_for_load_state("networkidle")

        font_scale = float(np.asarray(v["appearance"]["font_scale"]).reshape(-1)[0])
        if abs(font_scale - 1.0) > 1e-3:
            self._page.add_style_tag(
                content=f"html {{ font-size: {font_scale * 16:.2f}px !important; }}"
            )

        # Scroll the page to the sampled initial offset. Playwright
        # silently clamps to the actual scrollable area, so requesting
        # 600px on a short page just pins us at the bottom — that's the
        # behaviour we want for "scrolled down" variation.
        scroll_idx = int(np.asarray(v["browser"]["scroll_y"]).reshape(-1)[0])
        scroll_y = SCROLL_Y_CHOICES[scroll_idx]
        if scroll_y > 0:
            self._page.evaluate(f"window.scrollTo(0, {scroll_y})")
            self._page.wait_for_timeout(50)

        obs = self._capture_screenshot()
        info = {
            "pixels": obs,
            "env_name": self.env_name,
            "task_description": self.task_description,
        }
        return obs, info

    def _apply_variations(self, v: dict) -> None:
        """Push sampled variation values into Hydra cfg + Playwright state.

        Recomposes the OpenApps Hydra config with appearance/content/seed
        overrides for ``self.app_name`` and merges the result into the live
        ``app.config`` so already-registered FastHTML routes see the new
        values on their next request. Also resizes the browser viewport.
        """
        from omegaconf import OmegaConf, open_dict

        cfg_dir_name = _APP_TO_CONFIG_DIR.get(self.app_name, self.app_name)
        appearance = self._appearance_variants[int(v["appearance"]["theme"])]
        content = self._content_variants[int(v["content"]["variant"])]
        overrides = [
            f"apps/{cfg_dir_name}/appearance={appearance}",
            f"apps/{cfg_dir_name}/content={content}",
            f"seed={int(v['content']['seed'])}",
        ]

        new_cfg, new_tmp_logs = _load_hydra_config(extra_overrides=overrides)

        # In-place merge into self._cfg. ``app.config`` (set by
        # initialize_routes_and_configure_task) holds the same object, so
        # already-registered routes see the new values on their next request.
        OmegaConf.set_struct(self._cfg, False)
        with open_dict(self._cfg):
            self._cfg.apps = new_cfg.apps
            self._cfg.seed = new_cfg.seed

            # Map: pin ``init_location`` to the sampled city (defaults
            # to NYC at variation index 0) and override ``saved_places``
            # from the variant yaml. The variant's ``saved_places`` is
            # declared with ``+saved_places:`` — Hydra's "add new key"
            # form — which silently no-ops when the key already exists
            # via defaults. We re-parse the yaml directly so the
            # sidebar overlay actually swaps per-variant.
            if self.app_name == "map" and hasattr(self._cfg.apps, "maps"):
                city_idx = int(np.asarray(v["map"]["city"]).reshape(-1)[0])
                _, coords = MAP_CITY_CHOICES[city_idx]
                self._cfg.apps.maps.init_location = list(coords)
                variant_places = _read_variant_saved_places(content)
                if variant_places is not None:
                    self._cfg.apps.maps.saved_places = variant_places

        # Discard the new tmp logs dir — we keep the original from __init__.
        shutil.rmtree(new_tmp_logs, ignore_errors=True)

        # Browser viewport: map MultiDiscrete index → pixel size.
        vp_idx = np.asarray(v["browser"]["viewport"]).reshape(-1)
        vw = VIEWPORT_CHOICES_W[int(vp_idx[0])]
        vh = VIEWPORT_CHOICES_H[int(vp_idx[1])]
        if (vw, vh) != (self._page.viewport_size["width"], self._page.viewport_size["height"]):
            self._page.set_viewport_size({"width": vw, "height": vh})

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
            "task_description": self.task_description,
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
        """Take a Playwright screenshot and convert to numpy.

        Resizes to the fixed (VIEWPORT_HEIGHT, VIEWPORT_WIDTH) declared by
        ``observation_space`` so that browser-viewport variation does not
        break the gym contract — the agent always sees a canonical shape.
        """
        png_bytes = self._page.screenshot()
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        if img.size != (VIEWPORT_WIDTH, VIEWPORT_HEIGHT):
            img = img.resize((VIEWPORT_WIDTH, VIEWPORT_HEIGHT), Image.BILINEAR)
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
