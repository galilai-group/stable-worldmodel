"""Action helpers for the OpenApps env.

The MCP server's action space *is* BrowserGym's (``mouse_click(x, y)``,
``keyboard_type('...')``, ``scroll(delta_x, delta_y)``, ...). This module
provides:

  - ``make_action_space()`` — a gymnasium ``Dict`` space so random/learned
    policies have something to sample.
  - ``to_browsergym_action(sample) -> str`` — Dict sample (or passthrough
    string) -> a BrowserGym action string for the server's ``act``.

UI-TARS output is translated to BrowserGym in :mod:`agent_policy` using
OpenApps' own ``uitars_parser`` (so the mapping matches direct-OpenApps).
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces


VIEWPORT_WIDTH = 1024
VIEWPORT_HEIGHT = 640

ACTION_TYPES = [
    "click",
    "double_click",
    "move",
    "scroll",
    "type",
    "press",
    "drag",
    "wait",
]

_SCROLL_CHOICES = [-600, -300, -100, 0, 100, 300, 600]
_SAFE_KEYS = ["Enter", "Backspace", "Tab", "Escape", "ArrowDown", "ArrowUp"]
_BUTTONS = ["left", "right", "middle"]


def make_action_space() -> spaces.Dict:
    """Pixel-native Dict action space (samples map to BrowserGym strings)."""
    return spaces.Dict(
        {
            "type": spaces.Discrete(len(ACTION_TYPES)),
            "x": spaces.Box(0, VIEWPORT_WIDTH - 1, (), dtype=np.int32),
            "y": spaces.Box(0, VIEWPORT_HEIGHT - 1, (), dtype=np.int32),
            "to_x": spaces.Box(0, VIEWPORT_WIDTH - 1, (), dtype=np.int32),
            "to_y": spaces.Box(0, VIEWPORT_HEIGHT - 1, (), dtype=np.int32),
            "button": spaces.Discrete(3),
            "scroll": spaces.Discrete(len(_SCROLL_CHOICES)),
            "text": spaces.Text(max_length=64),
            "key": spaces.Discrete(len(_SAFE_KEYS)),
        }
    )


def _i(v) -> int:
    return int(np.asarray(v).reshape(-1)[0])


def to_browsergym_action(action) -> str:
    """Convert a Dict-space sample (or passthrough string) to a BrowserGym action."""
    if isinstance(action, str):
        return action

    a = action
    t = ACTION_TYPES[_i(a["type"])]
    x, y = _i(a["x"]), _i(a["y"])

    if t == "click":
        button = _BUTTONS[_i(a["button"])]
        if button == "left":
            return f"mouse_click({x}, {y})"
        return f"mouse_click({x}, {y}, button={button!r})"
    if t == "double_click":
        return f"mouse_dblclick({x}, {y})"
    if t == "move":
        return f"mouse_move({x}, {y})"
    if t == "scroll":
        return f"scroll(0, {_SCROLL_CHOICES[_i(a['scroll'])]})"
    if t == "type":
        return f"keyboard_type({str(a.get('text', ''))!r})"
    if t == "press":
        return f"keyboard_press({_SAFE_KEYS[_i(a['key'])]!r})"
    if t == "drag":
        return f"mouse_drag_and_drop({x}, {y}, {_i(a['to_x'])}, {_i(a['to_y'])})"
    return "noop(300)"
