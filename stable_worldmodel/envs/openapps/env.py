"""OpenApps gymnasium env, backed by an OpenApps MCP server.

Each env spawns its own ``python -m open_apps.mcp`` subprocess (process =
session), so N envs run independently in one process. Observations are
1024x640 RGB screenshots; actions are BrowserGym action strings or samples
of a pixel-native Dict action space; reward is scored server-side.
"""

import asyncio
import base64
import io
import json
import os
import sys
import threading

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image

from stable_worldmodel import spaces as swm_spaces


VIEWPORT_WIDTH = 1024
VIEWPORT_HEIGHT = 640

ACTION_TYPES = [
    'click',
    'double_click',
    'move',
    'scroll',
    'type',
    'press',
    'drag',
    'wait',
]
SCROLL_CHOICES = [-600, -300, -100, 0, 100, 300, 600]
SAFE_KEYS = ['Enter', 'Backspace', 'Tab', 'Escape', 'ArrowDown', 'ArrowUp']
BUTTONS = ['left', 'right', 'middle']

SCROLL_Y_CHOICES = [0, 100, 300, 600]
MAP_CITY_CHOICES = [
    ('NYC', [40.7831, -73.9712]),
    ('Paris', [48.8566, 2.3522]),
    ('Tokyo', [35.6762, 139.6503]),
    ('Berlin', [52.5200, 13.4050]),
    ('Sydney', [-33.8688, 151.2093]),
    ('Cairo', [30.0444, 31.2357]),
    ('Sao Paulo', [-23.5505, -46.6333]),
    ('Moscow', [55.7558, 37.6173]),
]


def _i(value):
    return int(np.asarray(value).reshape(-1)[0])


def make_action_space():
    """Pixel-native Dict action space; samples map to BrowserGym strings."""
    return spaces.Dict(
        {
            'type': spaces.Discrete(len(ACTION_TYPES)),
            'x': spaces.Box(0, VIEWPORT_WIDTH - 1, (), dtype=np.int32),
            'y': spaces.Box(0, VIEWPORT_HEIGHT - 1, (), dtype=np.int32),
            'to_x': spaces.Box(0, VIEWPORT_WIDTH - 1, (), dtype=np.int32),
            'to_y': spaces.Box(0, VIEWPORT_HEIGHT - 1, (), dtype=np.int32),
            'button': spaces.Discrete(len(BUTTONS)),
            'scroll': spaces.Discrete(len(SCROLL_CHOICES)),
            'text': spaces.Text(max_length=64),
            'key': spaces.Discrete(len(SAFE_KEYS)),
        }
    )


def to_browsergym_action(action):
    """Map a Dict-space sample (or passthrough string) to a BrowserGym action."""
    if isinstance(action, str):
        return action

    a = action
    kind = ACTION_TYPES[_i(a['type'])]
    x, y = _i(a['x']), _i(a['y'])

    if kind == 'click':
        button = BUTTONS[_i(a['button'])]
        if button == 'left':
            return f'mouse_click({x}, {y})'
        return f'mouse_click({x}, {y}, button={button!r})'
    if kind == 'double_click':
        return f'mouse_dblclick({x}, {y})'
    if kind == 'move':
        return f'mouse_move({x}, {y})'
    if kind == 'scroll':
        return f'scroll(0, {SCROLL_CHOICES[_i(a["scroll"])]})'
    if kind == 'type':
        return f'keyboard_type({str(a.get("text", ""))!r})'
    if kind == 'press':
        return f'keyboard_press({SAFE_KEYS[_i(a["key"])]!r})'
    if kind == 'drag':
        to_x, to_y = _i(a['to_x']), _i(a['to_y'])
        return f'mouse_drag_and_drop({x}, {y}, {to_x}, {to_y})'
    return 'noop(300)'


def _unwrap_result(res):
    """Unwrap a scalar/list tool result (FastMCP wraps it as ``result``)."""
    sc = getattr(res, 'structuredContent', None)
    if isinstance(sc, dict) and 'result' in sc:
        return sc['result']
    return [c.text for c in res.content if getattr(c, 'type', None) == 'text']


def _decode_obs(res):
    """Decode an obs tool result to ``(img: uint8 (H,W,3) | None, meta: dict)``."""
    png = None
    meta = {}
    for c in res.content:
        ctype = getattr(c, 'type', None)
        if ctype == 'image':
            png = base64.b64decode(c.data)
        elif ctype == 'text':
            try:
                meta = json.loads(c.text)
            except Exception:
                pass
    img = None
    if png is not None:
        img = np.asarray(
            Image.open(io.BytesIO(png)).convert('RGB'), dtype=np.uint8
        )
    return img, meta


class OpenAppsMCPClient:
    """Blocking handle to one OpenApps MCP server subprocess.

    The MCP SDK is async and gymnasium ``Env`` is sync, so the stdio session
    runs on a dedicated background event-loop thread and each call is
    marshalled onto it via ``run_coroutine_threadsafe``.
    """

    def __init__(self, app_name, *, server_args=None, ready_timeout=120.0):
        self.app_name = app_name
        self._extra_args = server_args or []
        self._loop = asyncio.new_event_loop()
        self._session = None
        self._shutdown = None
        self._ready = threading.Event()
        self._exc = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if (
            not self._ready.wait(timeout=ready_timeout)
            or self._session is None
        ):
            raise RuntimeError(
                f'OpenApps MCP server ({app_name}) failed to start: {self._exc}'
            )

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            self._exc = e
            self._ready.set()

    async def _serve(self):
        self._shutdown = asyncio.Event()
        params = StdioServerParameters(
            command=sys.executable,
            args=[
                '-m',
                'open_apps.mcp',
                '--app',
                self.app_name,
                *self._extra_args,
            ],
            env=os.environ.copy(),
        )
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._shutdown.wait()
        except Exception as e:
            self._exc = e
            self._ready.set()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def call_tool(self, name, args=None):
        if self._session is None:
            raise RuntimeError('MCP session not available')
        return self._run(self._session.call_tool(name, args or {}))

    def close(self):
        try:
            if self._shutdown is not None:
                self._loop.call_soon_threadsafe(self._shutdown.set)
            self._thread.join(timeout=10)
        except Exception:
            pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass

    def reset(self, seed=None):
        args = {} if seed is None else {'seed': int(seed)}
        return _decode_obs(self.call_tool('reset', args))

    def reconfigure(self, **kwargs):
        args = {k: v for k, v in kwargs.items() if v is not None}
        self.call_tool('reconfigure', args)

    def act(self, action, with_reward=True):
        args = {'action': action, 'with_reward': with_reward}
        return _decode_obs(self.call_tool('act', args))

    def observe(self):
        return _decode_obs(self.call_tool('observe', {}))

    def load_task(self, key):
        r = _unwrap_result(self.call_tool('load_task', {'key': key}))
        return r if isinstance(r, str) else (r[0] if r else '')

    def list_variants(self, app, group):
        r = _unwrap_result(
            self.call_tool('list_variants', {'app': app, 'group': group})
        )
        return list(r) if isinstance(r, list) else [r]

    def get_reward(self):
        r = _unwrap_result(self.call_tool('get_reward', {}))
        return float(r if not isinstance(r, list) else (r[0] if r else 0.0))


class OpenAppsEnv(gym.Env):
    """Gymnasium env for OpenApps browser-UI tasks (MCP-backed).

    Args:
        app_name: Which OpenApps app to target (e.g. ``'todo'``).
        task: A task key (str) into ``config/tasks/all_tasks.yaml``, or
            ``None`` (reward always 0.0 — data collection). Task instances
            are not accepted; tasks are resolved/scored server-side.
        task_description: Natural-language goal (falls back to the task goal).
        max_steps: Max steps per episode before truncation.
        render_mode: ``'rgb_array'`` (default).
    """

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 5}
    reward_range = (0.0, 1.0)

    def __init__(
        self,
        app_name='todo',
        task=None,
        task_description='',
        max_steps=50,
        render_mode='rgb_array',
        **_unused,
    ):
        super().__init__()

        if task is not None and not isinstance(task, str):
            raise TypeError(
                'OpenAppsEnv task must be a task-key string or None; task '
                'instances are resolved/scored server-side via MCP.'
            )

        self.app_name = app_name
        self.env_name = f'OpenApps-{app_name}'
        self._task_key = task
        self.max_steps = max_steps
        self.render_mode = render_mode

        self._step_count = 0
        self._last_screenshot = None

        self.observation_space = spaces.Box(
            0, 255, (VIEWPORT_HEIGHT, VIEWPORT_WIDTH, 3), np.uint8
        )
        self.action_space = make_action_space()

        # Spawn the OpenApps MCP server subprocess (process = session).
        self.client = OpenAppsMCPClient(app_name)

        # Discover variant sets from the server to build the variation space.
        self._appearance_variants = self.client.list_variants(
            app_name, 'appearance'
        )
        self._content_variants = self.client.list_variants(app_name, 'content')

        # Bind the task for server-side reward scoring; returns the goal.
        goal = self.client.load_task(task) if task is not None else ''
        self.task_description = task_description or goal

        spaces_dict = {
            'appearance': swm_spaces.Dict(
                {
                    'theme': swm_spaces.Discrete(
                        len(self._appearance_variants), init_value=0
                    )
                }
            ),
            'content': swm_spaces.Dict(
                {
                    'variant': swm_spaces.Discrete(
                        len(self._content_variants), init_value=0
                    ),
                    'seed': swm_spaces.Discrete(2**31 - 1, init_value=233),
                }
            ),
            'browser': swm_spaces.Dict(
                {
                    'scroll_y': swm_spaces.Discrete(
                        len(SCROLL_Y_CHOICES), init_value=0
                    )
                }
            ),
        }
        default_vars = [
            'appearance.theme',
            'content.variant',
            'content.seed',
            'browser.scroll_y',
        ]
        if app_name == 'map':
            spaces_dict['map'] = swm_spaces.Dict(
                {
                    'city': swm_spaces.Discrete(
                        len(MAP_CITY_CHOICES), init_value=0
                    )
                }
            )
            default_vars.append('map.city')
        self.variation_space = swm_spaces.Dict(spaces_dict)
        self._default_variations = tuple(default_vars)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._step_count = 0

        swm_spaces.reset_variation_space(
            self.variation_space,
            seed=seed,
            options=options,
            default_variations=self._default_variations,
        )
        v = self.variation_space.value
        self._apply_variations(v)

        img, _meta = self.client.reset(seed=seed)
        obs = self._to_obs(img)

        scroll_y = SCROLL_Y_CHOICES[_i(v['browser']['scroll_y'])]
        if scroll_y > 0:
            img, _ = self.client.act(
                f'scroll(0, {scroll_y})', with_reward=False
            )
            obs = self._to_obs(img)

        info = {
            'pixels': obs,
            'env_name': self.env_name,
            'task_description': self.task_description,
        }
        return obs, info

    def _apply_variations(self, v):
        """Translate sampled variation values into a server ``reconfigure``."""
        appearance = self._appearance_variants[_i(v['appearance']['theme'])]
        content = self._content_variants[_i(v['content']['variant'])]
        seed = int(v['content']['seed'])

        extras = {}
        if self.app_name == 'map':
            _, coords = MAP_CITY_CHOICES[_i(v['map']['city'])]
            extras['apps.maps.init_location'] = list(coords)

        self.client.reconfigure(
            appearance=appearance,
            content=content,
            seed=seed,
            extras=extras or None,
        )

    def step(self, action):
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
            'action_desc': meta.get('action_desc'),
        }
        return obs, reward, terminated, truncated, info

    def _act(self, action):
        # ``action`` is a BrowserGym action string or a Dict-space sample;
        # both map to a BrowserGym string for the server's ``act``.
        img, meta = self.client.act(to_browsergym_action(action))
        if img is None:
            # Errored/unparseable action: fall back to the current screenshot
            # so observations are never blank.
            obs_img, obs_meta = self.client.observe()
            if obs_img is not None:
                img, meta = obs_img, (meta or obs_meta)
        return img, meta

    def render(self):
        if self._last_screenshot is not None:
            return self._last_screenshot
        img, _ = self.client.observe()
        return self._to_obs(img)

    def _to_obs(self, img):
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
        client = getattr(self, 'client', None)
        if client is not None:
            try:
                client.close()
            except Exception as e:
                logger.debug(f'MCP client close failed: {e}')
        super().close()
