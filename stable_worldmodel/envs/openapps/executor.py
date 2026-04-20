"""Action codec: MultiDiscrete(3, GRID_X, GRID_Y) <-> Playwright calls.

Action space: MultiDiscrete([NUM_ACTIONS, GRID_X, GRID_Y])
  action[0] in {0, 1, 2}        — action type: click / scroll_down / scroll_up
  action[1] in {0..GRID_X-1}    — x grid cell
  action[2] in {0..GRID_Y-1}    — y grid cell

Pixel coordinates are cell centres, e.g. cell (i, j) maps to
  x = i * CELL_W + CELL_W // 2
  y = j * CELL_H + CELL_H // 2
"""

import re

import numpy as np

VIEWPORT_WIDTH = 1024
VIEWPORT_HEIGHT = 640

GRID_X = 32  # 1024 / 32 = 32 px per cell
GRID_Y = 20  # 640  / 20 = 32 px per cell
NUM_ACTIONS = 3  # 0=click, 1=scroll_down, 2=scroll_up

CELL_W = VIEWPORT_WIDTH // GRID_X  # 32 px
CELL_H = VIEWPORT_HEIGHT // GRID_Y  # 32 px

SCROLL_DELTA = 300  # px per scroll step (~3–4 list items)


def _cell_to_px(gx: int, gy: int) -> tuple[int, int]:
    """Convert grid cell indices to pixel coordinates (cell centre)."""
    x = int(gx) * CELL_W + CELL_W // 2
    y = int(gy) * CELL_H + CELL_H // 2
    return x, y


def action_multidiscrete_to_playwright(action: np.ndarray, page) -> str:
    """Decode a MultiDiscrete(3, GRID_X, GRID_Y) action into a Playwright call.

    Args:
        action: int64 array [action_type, grid_x, grid_y].
        page: Playwright Page object.

    Returns:
        Human-readable string describing the executed action.
    """
    action_type = int(action[0])
    x, y = _cell_to_px(int(action[1]), int(action[2]))

    if action_type == 0:
        page.mouse.click(x, y)
        return f'mouse_click(x={x}, y={y})'
    elif action_type == 1:
        page.mouse.move(x, y)
        page.mouse.wheel(0, SCROLL_DELTA)
        return f'scroll_down(x={x}, y={y}, delta={SCROLL_DELTA})'
    else:  # action_type == 2
        page.mouse.move(x, y)
        page.mouse.wheel(0, -SCROLL_DELTA)
        return f'scroll_up(x={x}, y={y}, delta={SCROLL_DELTA})'


# ── Action string → MultiDiscrete ────────────────────────────────────


def _px_to_cell(x_px: int, y_px: int) -> tuple[int, int]:
    """Convert pixel coordinates to the nearest grid cell."""
    gx = min(int(x_px) // CELL_W, GRID_X - 1)
    gy = min(int(y_px) // CELL_H, GRID_Y - 1)
    return gx, gy


def _extract_coords(action_str: str) -> tuple[int, int]:
    """Extract (x, y) pixel coordinates from an action string.

    Accepts mouse_click(x=N, y=N), TARS click(start_box='(x,y)'), and
    click(point='<point>x y</point>') forms.
    """
    match = re.search(r'x=(\d+).*?y=(\d+)', action_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r'<point>\s*(\d+)[\s,]+(\d+)\s*</point>', action_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    # TARS sometimes uses comma, sometimes whitespace: (97,185) or (97 185).
    match = re.search(r'\(\s*(\d+)\s*[,\s]\s*(\d+)\s*\)', action_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f'Cannot extract coords from: {action_str}')


def _extract_scroll_dy(action_str: str) -> float:
    """Extract dy scroll value from a BrowserGym scroll string."""
    match = re.search(r'dy=([-\d.]+)', action_str)
    if not match:
        raise ValueError(f'Cannot extract dy from: {action_str}')
    return float(match.group(1))


def action_str_to_multidiscrete(action_str: str) -> np.ndarray:
    """Encode a BrowserGym-style action string into a MultiDiscrete vector.

    Inverse of action_multidiscrete_to_playwright. Used by VLMPolicy to
    convert VLM output into the int64 vector the env expects.

    Args:
        action_str: e.g. "mouse_click(x=375, y=292)" or
                    "scroll(dx=0.0, dy=200.0)" or "scroll_down(x=512, y=320)"

    Returns:
        int64 array [action_type, grid_x, grid_y].
    """
    vec = np.zeros(3, dtype=np.int64)

    if action_str.startswith('mouse_click') or action_str.startswith('click'):
        x_px, y_px = _extract_coords(action_str)
        vec[0] = 0
        vec[1], vec[2] = _px_to_cell(x_px, y_px)

    elif 'scroll' in action_str:
        # Determine direction from dy sign or action name
        if action_str.startswith('scroll_up'):
            dy = -1.0
        elif action_str.startswith('scroll_down'):
            dy = 1.0
        else:
            try:
                dy = _extract_scroll_dy(action_str)
            except ValueError:
                dy = 1.0  # default scroll down

        vec[0] = 1 if dy >= 0 else 2
        # Use center of viewport as scroll anchor
        vec[1] = GRID_X // 2
        vec[2] = GRID_Y // 2

    else:
        # Unknown action — default to clicking center
        vec[0] = 0
        vec[1] = GRID_X // 2
        vec[2] = GRID_Y // 2

    return vec
