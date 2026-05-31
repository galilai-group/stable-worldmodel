"""
TwoRoom Navigation Environment - Clean Torch-based Implementation (Refactored).

- Torch-only rendering + collision
- Fixes mask shape bug (W,H vs H,W) by using meshgrid(y,x,indexing="ij")
- Consistent door semantics: door_size is half-extent in pixels
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_worldmodel import spaces as swm_spaces

DEFAULT_VARIATIONS = ('agent.position', 'target.position')


# Collision helpers
def _sign_nonzero(x):
    return 1.0 if float(x) >= 0 else -1.0


def _segment_axis_intersect(pos1, pos2, plane_coord, plane_axis):
    """Intersect segment pos1->pos2 with the axis-aligned plane
    `coord[plane_axis] = plane_coord`. Returns (2,) tensor or None."""
    a1 = float(pos1[plane_axis])
    a2 = float(pos2[plane_axis])
    if a1 == a2:
        return None
    if (a1 > plane_coord and a2 > plane_coord) or (
        a1 < plane_coord and a2 < plane_coord
    ):
        return None
    t = (plane_coord - a1) / (a2 - a1)
    other_axis = 1 - plane_axis
    b1 = float(pos1[other_axis])
    b2 = float(pos2[other_axis])
    out = torch.zeros(2, dtype=torch.float32, device=pos1.device)
    out[plane_axis] = float(plane_coord)
    out[other_axis] = b1 + t * (b2 - b1)
    return out


def _intersect_with_hole(
    pos1,
    pos2,
    plane_coord,
    plane_axis,
    hole_centers,
    hole_sizes,
    hole_passable,
):
    """Plane intercept, but returns None if the intercept along the orthogonal
    axis falls inside a passable door (`coord in [c-s, c+s]` for some i with
    hole_passable[i] = True)."""
    intersect = _segment_axis_intersect(pos1, pos2, plane_coord, plane_axis)
    if intersect is None:
        return None
    if hole_centers is None or hole_passable is None:
        return intersect
    par_axis = 1 - plane_axis
    par_coord = float(intersect[par_axis])
    for i in range(len(hole_centers)):
        if not bool(hole_passable[i]):
            continue
        c = float(hole_centers[i])
        s = float(hole_sizes[i])
        if (c - s) <= par_coord <= (c + s):
            return None
    return intersect


def _clamp_to_playable(pos, low_x, high_x, low_y, high_y, nudge):
    """DotWall's post-pushback clamp with ±nudge buffer at every boundary."""
    out = pos.clone()
    out[0] = torch.clamp(out[0], min=float(low_x), max=float(high_x))
    out[1] = torch.clamp(out[1], min=float(low_y), max=float(high_y))
    if float(out[0]) <= low_x:
        out[0] = float(low_x) + nudge
    if float(out[0]) >= high_x:
        out[0] = float(high_x) - nudge
    if float(out[1]) <= low_y:
        out[1] = float(low_y) + nudge
    if float(out[1]) >= high_y:
        out[1] = float(high_y) - nudge
    return out


def _check_wall_intersect(
    pos1,
    pos2,
    *,
    wall_pos,
    wall_axis,
    wall_width,
    door_centers,
    door_sizes,
    door_is_passable,
    border_loc,
    img_size,
    pushback,
    nudge,
):
    """Analytic line-segment vs wall collision, generalized over wall axis
    and multiple doors. Returns (intersect, intersect_w_pushback) or (None, None).

    Convention: `wall_axis == 1` means a vertical wall (at constant x); the dot
    crosses the wall along the x-axis. `wall_axis == 0` means horizontal.
    `door_is_passable` is a BoolTensor[num_doors] — locked doors (subclass
    override) leave it False, making the wall solid at those lips."""
    if torch.equal(pos1, pos2):
        return None, None

    perp_idx = 1 - wall_axis  # axis the dot crosses (x for vertical wall)
    par_idx = wall_axis  # axis the wall runs along (y for vertical wall)
    half_w = wall_width // 2
    near_corner = float(wall_pos) - half_w
    far_corner = float(wall_pos) + half_w

    # Candidate planes
    low_perp_plane = float(border_loc - 1)
    high_perp_plane = float(img_size - border_loc)
    low_par_plane = float(border_loc - 1)
    high_par_plane = float(img_size - border_loc)
    low_perp_holes = None
    high_perp_holes = None
    if float(pos1[perp_idx]) < float(wall_pos):
        high_perp_plane = near_corner
        high_perp_holes = (door_centers, door_sizes, door_is_passable)
    else:
        low_perp_plane = far_corner
        low_perp_holes = (door_centers, door_sizes, door_is_passable)

    if perp_idx == 0:
        clamp_bounds = (
            low_perp_plane,
            high_perp_plane,
            low_par_plane,
            high_par_plane,
        )
    else:
        clamp_bounds = (
            low_par_plane,
            high_par_plane,
            low_perp_plane,
            high_perp_plane,
        )

    # Step A: door-lip check
    d_par = float(pos2[par_idx]) - float(pos1[par_idx])
    for i in range(len(door_centers)):
        if not bool(door_is_passable[i]):
            continue
        c_i = float(door_centers[i])
        s_i = float(door_sizes[i])
        lip_lo = c_i - s_i
        lip_hi = c_i + s_i

        for lip_plane, going_in_sign in ((lip_hi, +1), (lip_lo, -1)):
            if going_in_sign == +1:
                if not (
                    d_par > 0
                    and float(pos1[par_idx]) < lip_plane
                    and float(pos2[par_idx]) > lip_plane
                ):
                    continue
            else:
                if not (
                    d_par < 0
                    and float(pos1[par_idx]) > lip_plane
                    and float(pos2[par_idx]) < lip_plane
                ):
                    continue
            intersect = _segment_axis_intersect(pos1, pos2, lip_plane, par_idx)
            if intersect is None:
                continue
            if near_corner <= float(intersect[perp_idx]) <= far_corner:
                push = torch.zeros(2, dtype=torch.float32, device=pos1.device)
                push[perp_idx] = pushback
                push[par_idx] = pushback * (-going_in_sign)
                return intersect, intersect + push

    # Step B: perp/par plane checks
    def _perp(plane_coord, holes):
        if holes is None:
            return _intersect_with_hole(
                pos1, pos2, plane_coord, perp_idx, None, None, None
            )
        return _intersect_with_hole(pos1, pos2, plane_coord, perp_idx, *holes)

    perp_hit = _perp(low_perp_plane, low_perp_holes)
    if perp_hit is None:
        perp_hit = _perp(high_perp_plane, high_perp_holes)

    par_hit = _intersect_with_hole(
        pos1, pos2, low_par_plane, par_idx, None, None, None
    )
    if par_hit is None:
        par_hit = _intersect_with_hole(
            pos1, pos2, high_par_plane, par_idx, None, None, None
        )

    if perp_hit is not None and par_hit is not None:
        if float(torch.norm(pos1 - perp_hit)) <= float(
            torch.norm(pos1 - par_hit)
        ):
            chosen, kind = perp_hit, 'perp'
        else:
            chosen, kind = par_hit, 'par'
    elif perp_hit is not None:
        chosen, kind = perp_hit, 'perp'
    elif par_hit is not None:
        chosen, kind = par_hit, 'par'
    else:
        return None, None

    push = torch.full((2,), pushback, dtype=torch.float32, device=pos1.device)
    if kind == 'perp':
        push[perp_idx] = pushback * _sign_nonzero(
            float(pos1[perp_idx]) - float(chosen[perp_idx])
        )
    else:
        push[par_idx] = pushback * _sign_nonzero(
            float(pos1[par_idx]) - float(chosen[par_idx])
        )

    return chosen, _clamp_to_playable(chosen + push, *clamp_bounds, nudge)


class TwoRoomEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    _REF_CANVAS = 65
    _REF_BORDER = 5
    _REF_WALL_WIDTH = 6
    _REF_DOOR_HALF_EXTENT = 4
    _REF_DOT_STD = 1.7
    _REF_WALL_POS = 32  # x of vertical wall
    _REF_DOOR_POS = 30  # y of door center

    # Collision / success constants, in native 65px units. Every
    # one is multiplied by self._scale below, so the env is a pure rescale of
    # the 65px reference at any img_size (nothing is a fixed pixel quantity).
    _REF_SUCCESS_DIST = 4.5
    _REF_BORDER_DRAW = 1
    _REF_PUSHBACK = 0.5  # collision pushback off a wall
    _REF_NUDGE = 0.3  # post-pushback clamp buffer

    # In the original implementation, next_pos = pos + action * 2, with action
    # magnitude topping out at 1.8. We clamp action to [-1, 1], so to match the
    # max per-step displacement we set speed = 2 * 1.8 = 3.6.
    _REF_ACTION_SPEED = 2.0 * 1.8

    DEFAULT_IMG_SIZE = 224
    MAX_DOOR = 3

    def __init__(
        self,
        render_mode: str = 'rgb_array',
        render_target: bool = False,
        init_value: dict | None = None,
        img_size: int = DEFAULT_IMG_SIZE,
    ):
        assert render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        self.render_target_flag = bool(render_target)

        self.IMG_SIZE = int(img_size)
        self._scale = self.IMG_SIZE / self._REF_CANVAS
        self.BORDER_SIZE = int(round(self._REF_BORDER * self._scale))
        self.WALL_CENTER = self.IMG_SIZE // 2

        self._default_wall_thickness = int(
            round(self._REF_WALL_WIDTH * self._scale)
        )
        self._default_door_half_extent = int(
            round(self._REF_DOOR_HALF_EXTENT * self._scale)
        )
        self._default_dot_std = float(self._REF_DOT_STD * self._scale)
        self._default_action_speed = float(
            self._REF_ACTION_SPEED * self._scale
        )
        self._default_door_pos = int(round(self._REF_DOOR_POS * self._scale))
        self._success_dist = float(self._REF_SUCCESS_DIST * self._scale)
        self._pushback = float(self._REF_PUSHBACK * self._scale)
        self._nudge = float(self._REF_NUDGE * self._scale)
        self._border_draw = max(
            1, int(round(self._REF_BORDER_DRAW * self._scale))
        )

        y = torch.arange(self.IMG_SIZE, dtype=torch.float32)
        x = torch.arange(self.IMG_SIZE, dtype=torch.float32)
        self.grid_y, self.grid_x = torch.meshgrid(y, x, indexing='ij')  # (H,W)

        # Observation space: state = agent(2) + target(2) + door_centers(max_door*2)
        state_dim = 2 + 2 + self.MAX_DOOR * 2
        self.observation_space = spaces.Box(
            low=0,
            high=self.IMG_SIZE,
            shape=(state_dim,),
            dtype=np.float32,
        )

        # Action: 2D velocity direction (scaled by speed)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.env_name = 'TwoRoom'

        # Variation space
        self.variation_space = self._build_variation_space()
        if init_value is not None:
            self.variation_space.set_init_value(init_value)

        # Runtime state
        self.agent_position = torch.zeros(2, dtype=torch.float32)
        self.target_position = torch.zeros(2, dtype=torch.float32)
        self._target_img = None

        # Cached params set in reset()
        self.wall_axis = 1
        self.wall_thickness = self._default_wall_thickness
        self.num_doors = 1
        self.door_positions = torch.zeros(self.MAX_DOOR, dtype=torch.float32)
        self.door_sizes = torch.zeros(self.MAX_DOOR, dtype=torch.float32)
        self.wall_pos = float(self.WALL_CENTER)
        self.door_passable_cache = torch.ones(self.MAX_DOOR, dtype=torch.bool)

    # Variation Space

    def _build_variation_space(self):
        # Valid position bounds: inside border with some padding for agent radius
        pos_min = float(self.BORDER_SIZE)
        pos_max = float(self.IMG_SIZE - self.BORDER_SIZE - 1)

        # All pixel-quantity bounds scale with self._scale so geometry stays
        # identical (in canvas-relative terms) across `img_size` instances.
        r_def = self._default_dot_std
        r_lo = max(1.0 * self._scale, 0.5 * r_def)
        r_hi = 3.0 * r_def

        s_def = self._default_action_speed
        s_lo = max(0.5 * self._scale, 0.25 * s_def)
        s_hi = 2.0 * s_def

        wall_def = self._default_wall_thickness
        wall_lo = max(1, int(round(0.5 * wall_def)))
        wall_hi = int(round(2.0 * wall_def))
        wall_n = wall_hi - wall_lo + 1

        door_def = self._default_door_half_extent
        door_lo = 1
        door_hi = max(door_lo, int(round(2.0 * door_def)))
        door_size_n = door_hi - door_lo + 1

        door_pos_start = 0
        door_pos_n = self.IMG_SIZE
        door_pos_def = self._default_door_pos

        agent_def_v = max(pos_min + 1, self.WALL_CENTER * 0.6)
        target_def_v = min(pos_max - 1, self.WALL_CENTER * 1.4)

        return swm_spaces.Dict(
            {
                'agent': swm_spaces.Dict(
                    {
                        'color': swm_spaces.RGBBox(
                            init_value=np.array([255, 0, 0], dtype=np.uint8)
                        ),
                        'radius': swm_spaces.Box(
                            low=np.array([r_lo], dtype=np.float32),
                            high=np.array([r_hi], dtype=np.float32),
                            init_value=np.array([r_def], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        'position': swm_spaces.Box(
                            low=np.array([pos_min, pos_min], dtype=np.float32),
                            high=np.array(
                                [pos_max, pos_max], dtype=np.float32
                            ),
                            shape=(2,),
                            dtype=np.float32,
                            init_value=np.array(
                                [agent_def_v, agent_def_v], dtype=np.float32
                            ),
                            constrain_fn=self._constrain_agent_not_in_wall,
                        ),
                        'speed': swm_spaces.Box(
                            low=np.array([s_lo], dtype=np.float32),
                            high=np.array([s_hi], dtype=np.float32),
                            init_value=np.array([s_def], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                    },
                    sampling_order=['color', 'radius', 'position', 'speed'],
                ),
                'target': swm_spaces.Dict(
                    {
                        'color': swm_spaces.RGBBox(
                            init_value=np.array([0, 255, 0], dtype=np.uint8)
                        ),
                        'radius': swm_spaces.Box(
                            low=np.array([r_lo], dtype=np.float32),
                            high=np.array([r_hi], dtype=np.float32),
                            init_value=np.array([r_def], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        'position': swm_spaces.Box(
                            low=np.array([pos_min, pos_min], dtype=np.float32),
                            high=np.array(
                                [pos_max, pos_max], dtype=np.float32
                            ),
                            shape=(2,),
                            dtype=np.float32,
                            init_value=np.array(
                                [target_def_v, target_def_v], dtype=np.float32
                            ),
                            # constrain_fn=self._constrain_target_by_min_steps,
                        ),
                    },
                    sampling_order=['color', 'radius', 'position'],
                ),
                'wall': swm_spaces.Dict(
                    {
                        'color': swm_spaces.RGBBox(
                            init_value=np.array([0, 0, 0], dtype=np.uint8)
                        ),
                        'thickness': swm_spaces.Discrete(
                            wall_n, start=wall_lo, init_value=wall_def
                        ),
                        'axis': swm_spaces.Discrete(
                            2, init_value=1
                        ),  # 0: horizontal, 1: vertical
                        'border_color': swm_spaces.RGBBox(
                            init_value=np.array([0, 0, 0], dtype=np.uint8)
                        ),
                    },
                    sampling_order=[
                        'color',
                        'border_color',
                        'thickness',
                        'axis',
                    ],
                ),
                'door': swm_spaces.Dict(
                    {
                        'color': swm_spaces.RGBBox(
                            init_value=np.array(
                                [255, 255, 255], dtype=np.uint8
                            )
                        ),
                        'number': swm_spaces.Discrete(
                            3, start=1, init_value=1
                        ),
                        # door size: half-extent of opening, in pixels
                        'size': swm_spaces.MultiDiscrete(
                            nvec=[door_size_n] * self.MAX_DOOR,
                            start=[door_lo] * self.MAX_DOOR,
                            init_value=[door_def] * self.MAX_DOOR,
                            constrain_fn=self._check_door_fit,
                        ),
                        # door position: center coord along the wall direction
                        'position': swm_spaces.MultiDiscrete(
                            nvec=[door_pos_n] * self.MAX_DOOR,
                            start=[door_pos_start] * self.MAX_DOOR,
                            init_value=[door_pos_def] * self.MAX_DOOR,
                        ),
                    },
                    sampling_order=['color', 'number', 'size', 'position'],
                ),
                'background': swm_spaces.Dict(
                    {
                        'color': swm_spaces.RGBBox(
                            init_value=np.array(
                                [255, 255, 255], dtype=np.uint8
                            )
                        )
                    }
                ),
                'rendering': swm_spaces.Dict(
                    {'render_target': swm_spaces.Discrete(2, init_value=0)}
                ),
                'task': swm_spaces.Dict(
                    {
                        'min_steps': swm_spaces.Discrete(
                            100, start=15, init_value=25
                        ),
                    }
                ),
            },
            sampling_order=[
                'background',
                'wall',
                'agent',
                'door',
                'task',
                'target',
                'rendering',
            ],
        )

    # Gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        swm_spaces.reset_variation_space(
            self.variation_space, seed, options, DEFAULT_VARIATIONS
        )

        agent_pos = options.get(
            'state', self.variation_space['agent']['position'].value
        )
        target_pos = options.get(
            'target_state', self.variation_space['target']['position'].value
        )

        self.agent_position = torch.as_tensor(agent_pos, dtype=torch.float32)
        self.target_position = torch.as_tensor(target_pos, dtype=torch.float32)

        self._cache_params()
        self.door_passable_cache = self._door_is_passable()

        # render “target image” = agent drawn at target position
        self._target_img = self._render_frame(agent_pos=self.target_position)

        obs = self._get_obs()
        info = self._get_info()
        info['distance_to_target'] = float(
            torch.norm(self.agent_position - self.target_position)
        )
        return obs, info

    def step(self, action):
        action_t = torch.as_tensor(action, dtype=torch.float32)
        action_t = torch.clamp(action_t, -1.0, 1.0)

        speed = float(self.variation_space['agent']['speed'].value.item())
        pos1 = self.agent_position
        pos2 = pos1 + action_t * speed

        pos_new = self._apply_collisions(pos1, pos2)
        self.agent_position = pos_new

        # DoorKey overrides for key pickup
        self._after_step_state_update(pos1, pos_new)

        dist = float(torch.norm(self.agent_position - self.target_position))
        terminated = dist < self._success_dist
        truncated = False
        reward = 0.0

        obs = self._get_obs()
        info = self._get_info()
        info['distance_to_target'] = dist
        return obs, reward, terminated, truncated, info

    def render(self):
        img_chw = (
            self._render_frame(agent_pos=self.agent_position).cpu().numpy()
        )
        return img_chw.transpose(1, 2, 0)  # CHW -> HWC

    # Internal helpers

    def _cache_params(self):
        self.wall_thickness = int(
            self.variation_space['wall']['thickness'].value
        )
        self.wall_axis = int(self.variation_space['wall']['axis'].value)

        self.num_doors = int(self.variation_space['door']['number'].value)
        door_pos = self.variation_space['door']['position'].value[
            : self.num_doors
        ]
        door_size = self.variation_space['door']['size'].value[
            : self.num_doors
        ]

        self.door_positions = torch.as_tensor(
            door_pos, dtype=torch.float32
        )  # center positions
        self.door_sizes = torch.as_tensor(
            door_size, dtype=torch.float32
        )  # half-extent

        # For policy / observation: wall position on relevant axis
        self.wall_pos = float(self.WALL_CENTER)

    def _get_obs(self):
        # state = agent(2) + target(2) + door_centers(MAX_DOOR*2)
        door_coords = []
        for i in range(self.MAX_DOOR):
            if i < self.num_doors:
                center_1d = float(self.door_positions[i].item())
                if self.wall_axis == 1:  # vertical wall => door varies along y
                    door_coords.extend([self.wall_pos, center_1d])
                else:  # horizontal wall => door varies along x
                    door_coords.extend([center_1d, self.wall_pos])
            else:
                door_coords.extend([0.0, 0.0])

        state = torch.tensor(
            [
                float(self.agent_position[0]),
                float(self.agent_position[1]),
                float(self.target_position[0]),
                float(self.target_position[1]),
                *door_coords,
            ],
            dtype=torch.float32,
        )

        return state

    def _get_info(self):
        return {
            'env_name': self.env_name,
            'proprio': self.agent_position.detach().cpu().numpy(),
            'state': self.agent_position.detach().cpu().numpy(),
            'goal_state': self.target_position.detach().cpu().numpy(),
        }

    # Rendering

    def _render_frame(self, agent_pos: torch.Tensor):
        H = W = self.IMG_SIZE

        bg = self.variation_space['background']['color'].value
        img = torch.empty((3, H, W), dtype=torch.uint8)
        img[0].fill_(int(bg[0]))
        img[1].fill_(int(bg[1]))
        img[2].fill_(int(bg[2]))

        wall_mask, door_masks = self._wall_and_door_masks()

        for i, dm in enumerate(door_masks):
            if not dm.any():
                continue
            c = self._door_color_for_index(i)
            img[0, dm] = int(c[0])
            img[1, dm] = int(c[1])
            img[2, dm] = int(c[2])

        # walls
        wall_color = self.variation_space['wall']['color'].value
        if wall_mask.any():
            img[0, wall_mask] = int(wall_color[0])
            img[1, wall_mask] = int(wall_color[1])
            img[2, wall_mask] = int(wall_color[2])

        # Gaussian's tail does not bleed through walls.
        not_wall = (~wall_mask).to(torch.float32)

        # optional target
        render_target = (
            bool(self.variation_space['rendering']['render_target'].value)
            or self.render_target_flag
        )
        if render_target:
            tgt_color = self.variation_space['target']['color'].value
            tgt_r = float(
                self.variation_space['target']['radius'].value.item()
            )
            tgt_dot = (
                self._gaussian_dot(self.target_position, tgt_r) * not_wall
            )
            img = self._alpha_blend(img, tgt_dot, tgt_color)

        img = self._render_extras(img, not_wall)

        # agent
        agent_color = self.variation_space['agent']['color'].value
        agent_r = float(self.variation_space['agent']['radius'].value.item())
        agent_dot = self._gaussian_dot(agent_pos, agent_r) * not_wall
        img = self._alpha_blend(img, agent_dot, agent_color)

        return img

    @staticmethod
    def _alpha_blend(
        img_u8: torch.Tensor, alpha_01: torch.Tensor, rgb_u8: np.ndarray
    ):
        """
        img_u8: (3,H,W) uint8
        alpha_01: (H,W) float32 in [0,1]
        rgb_u8: (3,) uint8-like
        """
        a = alpha_01.clamp(0, 1).to(torch.float32)
        out = img_u8.to(torch.float32)
        for c in range(3):
            out[c] = out[c] * (1.0 - a) + float(rgb_u8[c]) * a
        return out.to(torch.uint8)

    def _gaussian_dot(self, pos_xy: torch.Tensor, radius: float):
        # pos_xy is (2,) in x,y coordinates
        # grid_x/grid_y are (H,W)
        dx = self.grid_x - float(pos_xy[0])
        dy = self.grid_y - float(pos_xy[1])
        dist2 = dx * dx + dy * dy
        std = max(1e-6, float(radius))
        dot = torch.exp(-dist2 / (2.0 * std * std))
        m = dot.max()
        if m > 0:
            dot = dot / m
        return dot

    def _wall_and_door_masks(self):
        """
        Returns:
          wall_mask: (H,W) bool — wall including borders, with all door cutouts removed.
          door_masks: list[(H,W) bool] of length `num_doors` — one mask per door on
            the central wall. Per-door masks let subclasses (DoorKey) paint each
            door with a state-dependent color.
        """
        H = W = self.IMG_SIZE
        half = self.wall_thickness // 2

        if self.wall_axis == 1:  # vertical wall at x = center
            wall_stripe = (self.grid_x >= (self.WALL_CENTER - half)) & (
                self.grid_x <= (self.WALL_CENTER + half)
            )
            per_door_spans = []
            for i in range(self.num_doors):
                c = self.door_positions[i]
                s = self.door_sizes[i]
                per_door_spans.append(
                    (self.grid_y >= (c - s)) & (self.grid_y <= (c + s))
                )
        else:  # horizontal wall at y = center
            wall_stripe = (self.grid_y >= (self.WALL_CENTER - half)) & (
                self.grid_y <= (self.WALL_CENTER + half)
            )
            per_door_spans = []
            for i in range(self.num_doors):
                c = self.door_positions[i]
                s = self.door_sizes[i]
                per_door_spans.append(
                    (self.grid_x >= (c - s)) & (self.grid_x <= (c + s))
                )

        door_masks = [wall_stripe & span for span in per_door_spans]
        door_span_any = torch.zeros((H, W), dtype=torch.bool)
        for m in door_masks:
            door_span_any |= m
        wall_mask = wall_stripe & (~door_span_any)

        # Borders
        bs = self.BORDER_SIZE
        t = (
            self._border_draw
        )  # 1px at the 65px reference, scaled with img_size
        wall_mask[:, bs - t : bs] = True
        wall_mask[:, W - bs : W - bs + t] = True
        wall_mask[bs - t : bs, :] = True
        wall_mask[H - bs : H - bs + t, :] = True

        return wall_mask, door_masks

    # Hooks for subclasses

    def _door_is_passable(self) -> torch.Tensor:
        return torch.ones(self.MAX_DOOR, dtype=torch.bool)

    def _door_color_for_index(self, i: int) -> np.ndarray:
        return self.variation_space['door']['color'].value

    def _render_extras(
        self, img: torch.Tensor, not_wall_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return img

    def _after_step_state_update(
        self, pre_pos: torch.Tensor, post_pos: torch.Tensor
    ):
        pass

    # Collision

    def _apply_collisions(self, pos1: torch.Tensor, pos2: torch.Tensor):
        """Route through the DotWall-style analytic line-segment intersection.
        On no-hit, return `pos2` unchanged (point collision — no agent-radius
        padding); on hit, return the intersect + scaled pushback."""
        intersect, pushed = _check_wall_intersect(
            pos1,
            pos2,
            wall_pos=self.wall_pos,
            wall_axis=self.wall_axis,
            wall_width=self.wall_thickness,
            door_centers=self.door_positions[: self.num_doors],
            door_sizes=self.door_sizes[: self.num_doors],
            door_is_passable=self.door_passable_cache[: self.num_doors],
            border_loc=self.BORDER_SIZE,
            img_size=self.IMG_SIZE,
            pushback=self._pushback,
            nudge=self._nudge,
        )
        if intersect is None:
            return pos2
        return pushed

    # Constraint

    def _constrain_agent_not_in_wall(self, agent_pos):
        """
        Ensure agent position is not inside the wall (unless in a door).
        Agent can start in either room.
        """
        wall_axis = int(self.variation_space['wall']['axis'].value)
        wall_thickness = int(self.variation_space['wall']['thickness'].value)
        half_thickness = wall_thickness // 2
        agent_r = float(self.variation_space['agent']['radius'].value.item())

        # Effective wall zone including agent radius
        wall_min = self.WALL_CENTER - half_thickness - agent_r
        wall_max = self.WALL_CENTER + half_thickness + agent_r

        if wall_axis == 1:  # vertical wall
            # Check if agent x is in wall zone
            if wall_min <= agent_pos[0] <= wall_max:
                return False  # Agent would be in wall
        else:  # horizontal wall
            # Check if agent y is in wall zone
            if wall_min <= agent_pos[1] <= wall_max:
                return False  # Agent would be in wall

        return True

    def _check_door_fit(self, sizes):
        """
        Ensure at least one door half-extent can fit agent radius.
        """
        num = int(self.variation_space['door']['number'].value)
        agent_r = float(self.variation_space['agent']['radius'].value.item())
        return any(float(s) >= 1.1 * agent_r for s in sizes[:num])

    def _constrain_target_by_min_steps(self, target_pos):
        """
        Check if target position satisfies:
        1. Target must be in the opposite room from agent
        2. min_steps constraint (if set)

        min_steps specifies the minimum number of steps to reach target.
        Path length = dist(agent -> door) + dist(door -> target)
        min_steps <= path_length / speed
        """
        agent_pos = self.variation_space['agent']['position'].value
        wall_axis = int(self.variation_space['wall']['axis'].value)
        wall_thickness = int(self.variation_space['wall']['thickness'].value)
        half_thickness = wall_thickness // 2
        agent_r = float(self.variation_space['agent']['radius'].value.item())

        # First check: target must be in opposite room from agent
        if wall_axis == 1:  # vertical wall
            agent_side = agent_pos[0] < self.WALL_CENTER  # True = left room
            target_side = target_pos[0] < self.WALL_CENTER  # True = left room
            if agent_side == target_side:
                return False  # Same room, reject
            # Also ensure target is not in wall zone
            wall_min = self.WALL_CENTER - half_thickness - agent_r
            wall_max = self.WALL_CENTER + half_thickness + agent_r
            if wall_min <= target_pos[0] <= wall_max:
                return False
        else:  # horizontal wall
            agent_side = agent_pos[1] < self.WALL_CENTER  # True = top room
            target_side = target_pos[1] < self.WALL_CENTER  # True = top room
            if agent_side == target_side:
                return False  # Same room, reject
            # Also ensure target is not in wall zone
            wall_min = self.WALL_CENTER - half_thickness - agent_r
            wall_max = self.WALL_CENTER + half_thickness + agent_r
            if wall_min <= target_pos[1] <= wall_max:
                return False

        # Second check: min_steps constraint
        min_steps = int(self.variation_space['task']['min_steps'].value)
        if min_steps <= 0:
            return True  # No min_steps constraint

        speed = float(self.variation_space['agent']['speed'].value.item())

        # Get door info
        num_doors = int(self.variation_space['door']['number'].value)
        door_positions = self.variation_space['door']['position'].value[
            :num_doors
        ]
        door_sizes = self.variation_space['door']['size'].value[:num_doors]

        # Find the best (shortest path) door that fits the agent
        min_path_length = float('inf')
        for i in range(num_doors):
            door_size = float(door_sizes[i])
            if door_size < 1.1 * agent_r:
                continue  # Door too small

            door_center_1d = float(door_positions[i])

            if wall_axis == 1:  # vertical wall
                # Door is at x=wall_center, y=door_center_1d
                door_x = float(self.WALL_CENTER)
                door_y = door_center_1d
                # Distance from agent to door
                dist_to_door = np.sqrt(
                    (agent_pos[0] - door_x) ** 2 + (agent_pos[1] - door_y) ** 2
                )
                # Distance from door to target
                dist_door_to_target = np.sqrt(
                    (target_pos[0] - door_x) ** 2
                    + (target_pos[1] - door_y) ** 2
                )
            else:  # horizontal wall
                # Door is at x=door_center_1d, y=wall_center
                door_x = door_center_1d
                door_y = float(self.WALL_CENTER)
                dist_to_door = np.sqrt(
                    (agent_pos[0] - door_x) ** 2 + (agent_pos[1] - door_y) ** 2
                )
                dist_door_to_target = np.sqrt(
                    (target_pos[0] - door_x) ** 2
                    + (target_pos[1] - door_y) ** 2
                )

            path_length = dist_to_door + dist_door_to_target
            min_path_length = min(min_path_length, path_length)

        if min_path_length == float('inf'):
            return True  # No valid door, accept any target

        # Check if min_steps constraint is satisfied
        steps_required = min_path_length / speed
        return steps_required >= min_steps

    # Convenience setters

    def _set_state(self, state):
        self.agent_position = torch.tensor(state, dtype=torch.float32)

    def _set_goal_state(self, goal_state):
        self.target_position = torch.tensor(goal_state, dtype=torch.float32)
        self.variation_space['target']['position'].set_value(
            np.array(goal_state, dtype=np.float32)
        )
        self._target_img = self._render_frame(agent_pos=self.target_position)
