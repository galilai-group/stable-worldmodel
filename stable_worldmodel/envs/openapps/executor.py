"""Action codec: Box(5,) float vectors <-> Playwright calls.

This module is the only place pixel coordinates are computed from
normalised values, and the only place that talks directly to Playwright
for action execution.
"""

import re

import numpy as np

VIEWPORT_WIDTH = 1024
VIEWPORT_HEIGHT = 640


# ── Box(5,) → Playwright ─────────────────────────────────────────────

def action_vec_to_playwright(action: np.ndarray, page) -> str:
    """Decode a Box(5,) float vector into a Playwright call.

    Args:
        action: 5-dim float32 vector [action_type, x_norm, y_norm,
                scroll_dx, scroll_dy].
        page: Playwright Page object.

    Returns:
        Human-readable string describing the executed action.
    """
    action_type = float(action[0])
    x = int(action[1] * VIEWPORT_WIDTH)
    y = int(action[2] * VIEWPORT_HEIGHT)

    if action_type < 0.33:
        page.mouse.click(x, y)
        return f"mouse_click(x={x}, y={y})"
    elif action_type < 0.66:
        page.mouse.dblclick(x, y)
        return f"mouse_dblclick(x={x}, y={y})"
    else:
        dx = float(action[3]) * VIEWPORT_WIDTH
        dy = float(action[4]) * VIEWPORT_HEIGHT
        page.mouse.wheel(dx, dy)
        return f"scroll(dx={dx:.1f}, dy={dy:.1f})"


# ── Action string → Box(5,) ──────────────────────────────────────────

def _extract_coords(action_str: str) -> tuple[int, int]:
    """Extract (x, y) pixel coordinates from an action string."""
    match = re.search(r"x=(\d+).*?y=(\d+)", action_str)
    if not match:
        raise ValueError(f"Cannot extract coords from: {action_str}")
    return int(match.group(1)), int(match.group(2))


def _extract_scroll(action_str: str) -> tuple[float, float]:
    """Extract (dx, dy) scroll values from an action string."""
    match = re.search(r"dx=([-\d.]+).*?dy=([-\d.]+)", action_str)
    if not match:
        raise ValueError(f"Cannot extract scroll from: {action_str}")
    return float(match.group(1)), float(match.group(2))


def action_str_to_box5(action_str: str) -> np.ndarray:
    """Encode a BrowserGym-style action string into a Box(5,) vector.

    This is the inverse of action_vec_to_playwright. Used by VLMPolicy
    to convert VLM output into the float vector swm expects.

    Args:
        action_str: e.g. "mouse_click(x=375, y=292)" or
                    "scroll(dx=0.0, dy=-200.0)"

    Returns:
        5-dim float32 vector.
    """
    vec = np.zeros(5, dtype=np.float32)

    if action_str.startswith("mouse_click"):
        x, y = _extract_coords(action_str)
        vec[0] = 0.16  # center of click bucket [0, 0.33)
        vec[1] = x / VIEWPORT_WIDTH
        vec[2] = y / VIEWPORT_HEIGHT

    elif action_str.startswith("mouse_dblclick"):
        x, y = _extract_coords(action_str)
        vec[0] = 0.50  # center of dblclick bucket [0.33, 0.66)
        vec[1] = x / VIEWPORT_WIDTH
        vec[2] = y / VIEWPORT_HEIGHT

    elif action_str.startswith("scroll"):
        dx, dy = _extract_scroll(action_str)
        vec[0] = 0.83  # center of scroll bucket [0.66, 1.0]
        vec[3] = dx / VIEWPORT_WIDTH
        vec[4] = dy / VIEWPORT_HEIGHT

    return vec
