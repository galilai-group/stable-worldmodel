"""OpenAppsEnv — gymnasium env backed by an OpenApps MCP server.

Each env spawns its own ``python -m open_apps.mcp`` subprocess (process =
session), so N envs run independently in one process — no Playwright /
FastHTML singletons, no single-instance cap. Actions are full-resolution
pixel actions (typed dicts or Dict-space samples) or raw UI-TARS /
BrowserGym action strings; reward is computed server-side.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger
from PIL import Image

from stable_worldmodel import spaces as swm_space

from .executor import (
    VIEWPORT_HEIGHT,
    VIEWPORT_WIDTH,
    make_action_space,
    to_browsergym_action,
)
from .mcp_client import OpenAppsMCPClient


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


class OpenAppsEnv(gym.Env):
    """Gymnasium env for OpenApps browser UI tasks (MCP-backed).

    Args:
        app_name: Which OpenApps app to target (e.g. ``"todo"``).
        task: A task key (str) into ``config/tasks/all_tasks.yaml``, or
            ``None`` (reward always 0.0 — data collection). Task instances
            are not accepted: tasks are resolved/scored server-side.
        task_description: Natural-language goal (falls back to the task goal).
        port: Unused (kept for API compatibility; the server auto-picks a port).
        max_steps: Max steps per episode before truncation.
        render_mode: ``"rgb_array"`` (default).
    """

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 5}
    reward_range = (0.0, 1.0)

    VIEWPORT_WIDTH = VIEWPORT_WIDTH
    VIEWPORT_HEIGHT = VIEWPORT_HEIGHT
    DEFAULT_IMAGE_SHAPE = (VIEWPORT_HEIGHT, VIEWPORT_WIDTH)

    def __init__(
        self,
        app_name: str = 'todo',
        task: 'str | None' = None,
        task_description: str = '',
        port: int | None = None,
        max_steps: int = 50,
        render_mode: str = 'rgb_array',
        **_unused,
    ):
        super().__init__()

        if task is not None and not isinstance(task, str):
            raise TypeError(
                'OpenAppsEnv task must be a task-key string or None; '
                'Task instances are resolved/scored server-side via MCP.'
            )

        self.app_name = app_name
        self.env_name = f'OpenApps-{app_name}'
        self._task_key = task
        self.max_steps = max_steps
        self.render_mode = render_mode

        self._step_count = 0
        self._last_screenshot: np.ndarray | None = None

        self.observation_space = spaces.Box(
            0, 255, (VIEWPORT_HEIGHT, VIEWPORT_WIDTH, 3), np.uint8
        )
        self.action_space = make_action_space()

        # Spawn the OpenApps MCP server subprocess (process = session).
        self.client = OpenAppsMCPClient(app_name)

        # Discover variant sets from the server for the variation space.
        self._appearance_variants = self.client.list_variants(app_name, 'appearance')
        self._content_variants = self.client.list_variants(app_name, 'content')

        # Bind the task for server-side reward scoring; returns the goal.
        goal = ''
        if self._task_key is not None:
            goal = self.client.load_task(self._task_key)
        self.task_description = task_description or goal

        spaces_dict: dict = {
            'appearance': swm_space.Dict(
                {'theme': swm_space.Discrete(len(self._appearance_variants), init_value=0)}
            ),
            'content': swm_space.Dict(
                {
                    'variant': swm_space.Discrete(len(self._content_variants), init_value=0),
                    'seed': swm_space.Discrete(2**31 - 1, init_value=233),
                }
            ),
            'browser': swm_space.Dict(
                {'scroll_y': swm_space.Discrete(len(SCROLL_Y_CHOICES), init_value=0)}
            ),
        }
        if self.app_name == 'map':
            spaces_dict['map'] = swm_space.Dict(
                {'city': swm_space.Discrete(len(MAP_CITY_CHOICES), init_value=0)}
            )
        self.variation_space = swm_space.Dict(spaces_dict)

        default_vars = ['appearance.theme', 'content.variant', 'content.seed', 'browser.scroll_y']
        if self.app_name == 'map':
            default_vars.append('map.city')
        self._default_variations = tuple(default_vars)

    def reset(self, *, seed=None, options=None):
        """Sample a variation, push it to the server, reset, and return obs."""
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

        img, _meta = self.client.reset(seed=seed)
        obs = self._to_obs(img)

        scroll_idx = int(np.asarray(v['browser']['scroll_y']).reshape(-1)[0])
        scroll_y = SCROLL_Y_CHOICES[scroll_idx]
        if scroll_y > 0:
            img, _ = self.client.act(f"scroll(0, {scroll_y})", with_reward=False)
            obs = self._to_obs(img)

        info = {
            'pixels': obs,
            'env_name': self.env_name,
            'task_description': self.task_description,
        }
        return obs, info

    def _apply_variations(self, v: dict) -> None:
        """Translate sampled variation values into a server ``reconfigure``."""
        appearance = self._appearance_variants[int(v['appearance']['theme'])]
        content = self._content_variants[int(v['content']['variant'])]
        seed = int(v['content']['seed'])

        extras: dict = {}
        if self.app_name == 'map':
            city_idx = int(np.asarray(v['map']['city']).reshape(-1)[0])
            _, coords = MAP_CITY_CHOICES[city_idx]
            extras['apps.maps.init_location'] = list(coords)

        self.client.reconfigure(
            appearance=appearance, content=content, seed=seed, extras=extras or None
        )

    def step(self, action):
        """Execute one action (typed dict, Dict-space sample, or action string)."""
        self._step_count += 1
        img, meta = self._act(action)
        obs = self._to_obs(img)

        reward = float(meta.get('reward', 0.0))
        terminated = bool(meta.get('done', reward >= 1.0))
        truncated = self._step_count >= self.max_steps

        info = {
            'pixels': obs,
            'env_name': self.env_name,
            'task_description': self.task_description,
            '_action_desc': meta.get('action_desc'),
        }
        return obs, reward, terminated, truncated, info

    def _act(self, action):
        # ``action`` is a BrowserGym action string (VLM/scripted) or a
        # Dict-space sample (random/learned); both map to a BrowserGym string.
        img, meta = self.client.act(to_browsergym_action(action))
        if img is None:
            # Failed/errored action (e.g. unparseable output): fall back to
            # the current screenshot so observations are never blank.
            obs_img, obs_meta = self.client.observe()
            if obs_img is not None:
                img, meta = obs_img, (meta or obs_meta)
        return img, meta

    def render(self) -> np.ndarray:
        if self._last_screenshot is not None:
            return self._last_screenshot
        img, _ = self.client.observe()
        return self._to_obs(img)

    def _to_obs(self, img: np.ndarray | None) -> np.ndarray:
        if img is None:
            return np.zeros((VIEWPORT_HEIGHT, VIEWPORT_WIDTH, 3), np.uint8)
        if img.shape[:2] != (VIEWPORT_HEIGHT, VIEWPORT_WIDTH):
            img = np.asarray(
                Image.fromarray(img).resize(
                    (VIEWPORT_WIDTH, VIEWPORT_HEIGHT), Image.BILINEAR
                ),
                dtype=np.uint8,
            )
        self._last_screenshot = img
        return img

    def close(self):
        """Tear down the MCP server subprocess."""
        client = getattr(self, 'client', None)
        if client is not None:
            try:
                client.close()
            except Exception as e:
                logger.debug(f'MCP client close failed: {e}')
        super().close()
