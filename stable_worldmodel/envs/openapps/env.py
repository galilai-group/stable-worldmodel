"""OpenAppsEnv — gymnasium environment wrapping Playwright + OpenApps Runtime."""

import io

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger
from PIL import Image

from open_apps.runtime import Runtime, list_variants, make_runtime
from open_apps.tasks import Task, load_task

from stable_worldmodel import spaces as swm_space

from .executor import (
    GRID_X,
    GRID_Y,
    NUM_ACTIONS,
    VIEWPORT_HEIGHT,
    VIEWPORT_WIDTH,
    action_multidiscrete_to_playwright,
)


SCROLL_Y_CHOICES = [0, 100, 300, 600]

MAP_CITY_CHOICES: list[tuple[str, list[float]]] = [
    ('NYC', [40.7831, -73.9712]),
    ('Paris', [48.8566, 2.3522]),
    ('Tokyo', [35.6762, 139.6503]),
    ('Berlin', [52.5200, 13.4050]),
    ('Sydney', [-33.8688, 151.2093]),
    ('Cairo', [30.0444, 31.2357]),
    ('Sao Paulo', [-23.5505, -46.6333]),
    ('Moscow', [55.7558, 37.6173]),
]


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
    """Gymnasium env for OpenApps browser-based UI tasks.
    Args:
        app_name: Which OpenApps app to target (e.g. ``"todo"``).
        task: Either an :class:`open_apps.tasks.Task` instance, a task
            key (str) into ``config/tasks/all_tasks.yaml``, or ``None``
            (reward always 0.0 — useful for data collection).
        task_description: Natural-language task goal (falls back to
            ``task.goal``).
        port: Port for the FastHTML server. Auto-picked if ``None``.
        max_steps: Max steps per episode before truncation.
        render_mode: ``"rgb_array"`` (default).
    """

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 5}
    reward_range = (0.0, 1.0)

    VIEWPORT_WIDTH = VIEWPORT_WIDTH
    VIEWPORT_HEIGHT = VIEWPORT_HEIGHT
    DEFAULT_IMAGE_SHAPE = (VIEWPORT_HEIGHT, VIEWPORT_WIDTH)

    _active_instances: int = 0

    def __init__(
        self,
        app_name: str = 'todo',
        task: 'Task | str | None' = None,
        task_description: str = '',
        port: int | None = None,
        max_steps: int = 50,
        render_mode: str = 'rgb_array',
        **_unused,
    ):
        super().__init__()

        if OpenAppsEnv._active_instances > 0:
            raise RuntimeError(
                'OpenAppsEnv only supports a single live instance per '
                'process (Playwright sync API and OpenApps Runtime are '
                'global singletons). Close the existing env first, and '
                'use num_envs=1 with swm.World for OpenApps.'
            )

        self.app_name = app_name
        self.env_name = f'OpenApps-{app_name}'
        if isinstance(task, str):
            task = load_task(task, app=app_name)
        self.task = task
        self.task_description = task_description or (
            task.goal if task else ''
        )
        self.max_steps = max_steps
        self.render_mode = render_mode

        self._step_count = 0
        self._last_screenshot = None
        self._initial_state: dict | None = None

        self.observation_space = spaces.Box(
            0, 255, (VIEWPORT_HEIGHT, VIEWPORT_WIDTH, 3), np.uint8
        )
        self.action_space = spaces.MultiDiscrete(
            [NUM_ACTIONS, GRID_X, GRID_Y]
        )

        self._appearance_variants = list_variants(app_name, 'appearance')
        self._content_variants = list_variants(app_name, 'content')

        spaces_dict: dict = {
            'appearance': swm_space.Dict(
                {
                    'theme': swm_space.Discrete(
                        len(self._appearance_variants), init_value=0
                    ),
                }
            ),
            'content': swm_space.Dict(
                {
                    'variant': swm_space.Discrete(
                        len(self._content_variants), init_value=0
                    ),
                    'seed': swm_space.Discrete(2**31 - 1, init_value=233),
                }
            ),
            'browser': swm_space.Dict(
                {
                    'scroll_y': swm_space.Discrete(
                        len(SCROLL_Y_CHOICES), init_value=0
                    ),
                }
            ),
        }
        if self.app_name == 'map':
            spaces_dict['map'] = swm_space.Dict(
                {
                    'city': swm_space.Discrete(
                        len(MAP_CITY_CHOICES), init_value=0
                    ),
                }
            )
        self.variation_space = swm_space.Dict(spaces_dict)

        default_vars: list[str] = [
            'appearance.theme',
            'content.variant',
            'content.seed',
            'browser.scroll_y',
        ]
        if self.app_name == 'map':
            default_vars.append('map.city')
        self._default_variations = tuple(default_vars)

        self.runtime: Runtime = make_runtime(app_name, port=port)
        self.base_url = self.runtime.base_url

        browser = _get_playwright()
        self._context = browser.new_context(
            viewport={
                'width': VIEWPORT_WIDTH,
                'height': VIEWPORT_HEIGHT,
            }
        )
        self._page = self._context.new_page()

        OpenAppsEnv._active_instances += 1

    def reset(self, *, seed=None, options=None):
        """Sample a variation, push it to the runtime, and load the page."""
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

        self.runtime.reset()
        self._initial_state = self.runtime.get_state()

        self._page.goto(self.runtime.url_for())
        self._page.wait_for_load_state('networkidle')

        scroll_idx = int(
            np.asarray(v['browser']['scroll_y']).reshape(-1)[0]
        )
        scroll_y = SCROLL_Y_CHOICES[scroll_idx]
        if scroll_y > 0:
            self._page.evaluate(f'window.scrollTo(0, {scroll_y})')
            self._page.wait_for_timeout(50)

        obs = self._capture_screenshot()
        info = {
            'pixels': obs,
            'env_name': self.env_name,
            'task_description': self.task_description,
        }
        return obs, info

    def _apply_variations(self, v: dict) -> None:
        """Translate sampled variation values into ``runtime.reconfigure(...)``."""
        appearance = self._appearance_variants[
            int(v['appearance']['theme'])
        ]
        content = self._content_variants[int(v['content']['variant'])]
        seed = int(v['content']['seed'])

        extras: dict = {}
        if self.app_name == 'map':
            city_idx = int(np.asarray(v['map']['city']).reshape(-1)[0])
            _, coords = MAP_CITY_CHOICES[city_idx]
            extras['apps.maps.init_location'] = list(coords)

        self.runtime.reconfigure(
            appearance=appearance,
            content=content,
            seed=seed,
            extras=extras or None,
        )

    def step(self, action: np.ndarray):
        """Execute one action and return the new observation."""
        self._step_count += 1

        action_desc = action_multidiscrete_to_playwright(action, self._page)
        logger.debug(f'Step {self._step_count}: {action_desc}')

        self._page.wait_for_timeout(300)

        obs = self._capture_screenshot()
        reward = self._compute_reward()
        terminated = reward == 1.0
        truncated = self._step_count >= self.max_steps

        info = {
            'pixels': obs,
            'env_name': self.env_name,
            'task_description': self.task_description,
            '_action_desc': action_desc,
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """Return the current screenshot as numpy (H, W, 3) uint8."""
        if self._last_screenshot is not None:
            return self._last_screenshot
        return self._capture_screenshot()

    def _capture_screenshot(self) -> np.ndarray:
        png_bytes = self._page.screenshot()
        img = Image.open(io.BytesIO(png_bytes)).convert('RGB')
        if img.size != (VIEWPORT_WIDTH, VIEWPORT_HEIGHT):
            img = img.resize(
                (VIEWPORT_WIDTH, VIEWPORT_HEIGHT), Image.BILINEAR
            )
        arr = np.array(img, dtype=np.uint8)
        self._last_screenshot = arr
        return arr

    def _compute_reward(self) -> float:
        """Compute reward by delegating to the OpenApps Task."""
        if self.task is None or self._initial_state is None:
            return 0.0

        try:
            current_state = self.runtime.get_state()
            try:
                current_state['_url'] = self._page.url
            except Exception:
                current_state['_url'] = ''
            return (
                1.0
                if self.task.check_if_task_is_complete(
                    self._initial_state, current_state
                )
                else 0.0
            )
        except Exception as e:
            logger.warning(f'Reward computation failed: {e}')
            return 0.0

    def close(self):
        """Tear down browser context and the OpenApps runtime."""
        try:
            if getattr(self, '_page', None):
                self._page.close()
            if getattr(self, '_context', None):
                self._context.close()
        except Exception as e:
            logger.debug(f'Browser close failed: {e}')

        runtime = getattr(self, 'runtime', None)
        if runtime is not None:
            runtime.close()

        OpenAppsEnv._active_instances = max(
            0, OpenAppsEnv._active_instances - 1
        )
        super().close()
