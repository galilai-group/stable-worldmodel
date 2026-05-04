"""TwoRoomDoorKey Navigation Environment.

Continuous analog of MiniGrid's DoorKey:
  - Agent and target start in opposite rooms separated by a wall with one or
    more door openings. Doors start locked (impassable, rendered with
    ``door.locked_color``).
  - A key is randomly placed in the agent's room.
  - When the agent gets within ``key.pickup_radius`` of the key, the key
    disappears, doors unlock (rendered with ``door.unlocked_color``), and
    the agent can pass through to reach the target.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_worldmodel import spaces as swm_spaces

DEFAULT_VARIATIONS = (
    'wall.axis',
    'door.number',
    'door.position',
    'agent.position',
    'key.position',
    'target.position',
    'agent.color',
    'key.color',
    'target.color',
    'door.locked_color',
    'door.unlocked_color',
)


class TwoRoomDoorKeyEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    # Fixed geometry for 224x224 (scale = 224/64 = 3.5)
    IMG_SIZE = 224
    BORDER_SIZE = 14
    DOT_STD = 7.0
    PADDING = 14
    MAX_SPEED = 10.5
    WALL_CENTER = 112
    WALL_WIDTH_DEFAULT = 10
    MAX_DOOR = 3

    def __init__(
        self,
        render_mode: str = 'rgb_array',
        render_target: bool = False,
        init_value: dict | None = None,
    ):
        assert render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        self.render_target_flag = bool(render_target)

        # Precompute coordinate grids once (H,W)
        y = torch.arange(self.IMG_SIZE, dtype=torch.float32)
        x = torch.arange(self.IMG_SIZE, dtype=torch.float32)
        self.grid_y, self.grid_x = torch.meshgrid(y, x, indexing='ij')

        # state = agent(2) + target(2) + door_centers(MAX_DOOR*2) + key(2) + has_key + door_open
        state_dim = 2 + 2 + self.MAX_DOOR * 2 + 2 + 1 + 1
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

        self.env_name = 'TwoRoomDoorKey'

        self.variation_space = self._build_variation_space()
        if init_value is not None:
            self.variation_space.set_init_value(init_value)

        # Runtime state
        self.agent_position = torch.zeros(2, dtype=torch.float32)
        self.target_position = torch.zeros(2, dtype=torch.float32)
        self.key_position = torch.zeros(2, dtype=torch.float32)
        self.has_key = False
        self.door_open = False
        self._target_img = None

        # Cached params set in reset()
        self.wall_axis = 1
        self.wall_thickness = self.WALL_WIDTH_DEFAULT
        self.num_doors = 1
        self.door_positions = torch.zeros(self.MAX_DOOR, dtype=torch.float32)
        self.door_sizes = torch.zeros(self.MAX_DOOR, dtype=torch.float32)
        self.wall_pos = float(self.WALL_CENTER)

    # ---------------- Variation Space ----------------

    def _build_variation_space(self):
        # Valid position bounds: inside border with some padding for radius
        pos_min = float(self.BORDER_SIZE)
        pos_max = float(self.IMG_SIZE - self.BORDER_SIZE - 1)

        return swm_spaces.Dict(
            {
                'agent': swm_spaces.Dict(
                    {
                        'color': swm_spaces.RGBBox(
                            init_value=np.array([255, 0, 0], dtype=np.uint8)
                        ),
                        'radius': swm_spaces.Box(
                            low=np.array([7.0], dtype=np.float32),
                            high=np.array([14.0], dtype=np.float32),
                            init_value=np.array([7.0], dtype=np.float32),
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
                                [60.0, 112.0], dtype=np.float32
                            ),
                            constrain_fn=self._constrain_agent_not_in_wall,
                        ),
                        'speed': swm_spaces.Box(
                            low=np.array([1.75], dtype=np.float32),
                            high=np.array([10.5], dtype=np.float32),
                            init_value=np.array([5.0], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                    },
                    sampling_order=['color', 'radius', 'position', 'speed'],
                ),
                'key': swm_spaces.Dict(
                    {
                        'color': swm_spaces.RGBBox(
                            init_value=np.array([255, 215, 0], dtype=np.uint8)
                        ),
                        'radius': swm_spaces.Box(
                            low=np.array([7.0], dtype=np.float32),
                            high=np.array([14.0], dtype=np.float32),
                            init_value=np.array([7.0], dtype=np.float32),
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
                                [60.0, 60.0], dtype=np.float32
                            ),
                            constrain_fn=self._constrain_key_in_agent_room,
                        ),
                        'pickup_radius': swm_spaces.Box(
                            low=np.array([15.0], dtype=np.float32),
                            high=np.array([30.0], dtype=np.float32),
                            init_value=np.array([20.0], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                    },
                    sampling_order=[
                        'color',
                        'radius',
                        'position',
                        'pickup_radius',
                    ],
                ),
                'target': swm_spaces.Dict(
                    {
                        'color': swm_spaces.RGBBox(
                            init_value=np.array([0, 255, 0], dtype=np.uint8)
                        ),
                        'radius': swm_spaces.Box(
                            low=np.array([7.0], dtype=np.float32),
                            high=np.array([14.0], dtype=np.float32),
                            init_value=np.array([7.0], dtype=np.float32),
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
                                [164.0, 112.0], dtype=np.float32
                            ),
                            constrain_fn=self._constrain_target_in_other_room,
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
                            35, start=7, init_value=10
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
                        # Color of the door while locked.
                        'locked_color': swm_spaces.RGBBox(
                            init_value=np.array([200, 0, 0], dtype=np.uint8)
                        ),
                        # Color of the door once the key has been picked up.
                        'unlocked_color': swm_spaces.RGBBox(
                            init_value=np.array(
                                [255, 255, 255], dtype=np.uint8
                            )
                        ),
                        'number': swm_spaces.Discrete(
                            3, start=1, init_value=1
                        ),
                        # door size is half-extent: range [1, 21] pixels
                        'size': swm_spaces.MultiDiscrete(
                            nvec=[21, 21, 21],
                            start=[1, 1, 1],
                            init_value=[14, 14, 14],
                            constrain_fn=self._check_door_fit,
                        ),
                        # door position is center coord along wall: [0, 223]
                        'position': swm_spaces.MultiDiscrete(
                            nvec=[224, 224, 224],
                            init_value=[49, 49, 49],
                        ),
                    },
                    sampling_order=[
                        'locked_color',
                        'unlocked_color',
                        'number',
                        'size',
                        'position',
                    ],
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
            },
            sampling_order=[
                'background',
                'wall',
                'agent',
                'key',
                'door',
                'target',
                'rendering',
            ],
        )

    # ---------------- Gym API ----------------

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
        key_pos = options.get(
            'key_state', self.variation_space['key']['position'].value
        )

        self.agent_position = torch.as_tensor(agent_pos, dtype=torch.float32)
        self.target_position = torch.as_tensor(target_pos, dtype=torch.float32)
        self.key_position = torch.as_tensor(key_pos, dtype=torch.float32)
        self.has_key = False
        self.door_open = False

        self._cache_params()

        # render "target image" = agent drawn at target position
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
        pos_next = self.agent_position + action_t * speed

        pos_new = self._apply_collisions(self.agent_position, pos_next)
        self.agent_position = pos_new

        # Key pickup: once the agent is within pickup_radius, the key disappears
        # and every door unlocks. Persists for the rest of the episode.
        if not self.has_key:
            pickup_r = float(
                self.variation_space['key']['pickup_radius'].value.item()
            )
            d_key = float(torch.norm(self.agent_position - self.key_position))
            if d_key < pickup_r:
                self.has_key = True
                self.door_open = True

        dist = float(torch.norm(self.agent_position - self.target_position))
        terminated = dist < 16.0  # ~4.5 * 3.5 scale
        truncated = False
        reward = 0.0

        obs = self._get_obs()
        info = self._get_info()
        info['distance_to_target'] = dist
        return obs, reward, terminated, truncated, info

    def render(self):
        # returns HWC uint8 numpy for compatibility with PIL/wrappers
        img_chw = (
            self._render_frame(agent_pos=self.agent_position).cpu().numpy()
        )
        return img_chw.transpose(1, 2, 0)  # CHW -> HWC

    # ---------------- Internal helpers ----------------

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
        # state = agent(2) + target(2) + door_centers(MAX_DOOR*2) + key(2) + has_key + door_open
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
                float(self.key_position[0]),
                float(self.key_position[1]),
                float(self.has_key),
                float(self.door_open),
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
            'key_position': self.key_position.detach().cpu().numpy(),
            'has_key': np.float32(self.has_key),
            'door_open': np.float32(self.door_open),
        }

    # ---------------- Rendering ----------------

    def _render_frame(self, agent_pos: torch.Tensor):
        H = W = self.IMG_SIZE

        bg = self.variation_space['background']['color'].value
        img = torch.empty((3, H, W), dtype=torch.uint8)
        img[0].fill_(int(bg[0]))
        img[1].fill_(int(bg[1]))
        img[2].fill_(int(bg[2]))

        wall_mask, door_mask = self._wall_and_door_masks()

        # doors — locked or unlocked color depending on state
        if self.door_open:
            door_color = self.variation_space['door']['unlocked_color'].value
        else:
            door_color = self.variation_space['door']['locked_color'].value
        if door_mask.any():
            img[0, door_mask] = int(door_color[0])
            img[1, door_mask] = int(door_color[1])
            img[2, door_mask] = int(door_color[2])

        # walls
        wall_color = self.variation_space['wall']['color'].value
        if wall_mask.any():
            img[0, wall_mask] = int(wall_color[0])
            img[1, wall_mask] = int(wall_color[1])
            img[2, wall_mask] = int(wall_color[2])

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
            tgt_dot = self._gaussian_dot(self.target_position, tgt_r)
            img = self._alpha_blend(img, tgt_dot, tgt_color)

        # key (only while not picked up)
        if not self.has_key:
            key_color = self.variation_space['key']['color'].value
            key_r = float(self.variation_space['key']['radius'].value.item())
            key_dot = self._gaussian_dot(self.key_position, key_r)
            img = self._alpha_blend(img, key_dot, key_color)

        # agent
        agent_color = self.variation_space['agent']['color'].value
        agent_r = float(self.variation_space['agent']['radius'].value.item())
        agent_dot = self._gaussian_dot(agent_pos, agent_r)
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
          wall_mask: (H,W) bool (wall including borders, with door cutouts removed)
          door_mask: (H,W) bool (door pixels only, on the central wall)
        """
        H = W = self.IMG_SIZE
        half = self.wall_thickness // 2

        # Central wall stripe
        if self.wall_axis == 1:  # vertical wall at x = center
            wall_stripe = (self.grid_x >= (self.WALL_CENTER - half)) & (
                self.grid_x <= (self.WALL_CENTER + half)
            )
            door_span = torch.zeros((H, W), dtype=torch.bool)
            for i in range(self.num_doors):
                c = self.door_positions[i]
                s = self.door_sizes[i]
                door_span |= (self.grid_y >= (c - s)) & (
                    self.grid_y <= (c + s)
                )
        else:  # horizontal wall at y = center
            wall_stripe = (self.grid_y >= (self.WALL_CENTER - half)) & (
                self.grid_y <= (self.WALL_CENTER + half)
            )
            door_span = torch.zeros((H, W), dtype=torch.bool)
            for i in range(self.num_doors):
                c = self.door_positions[i]
                s = self.door_sizes[i]
                door_span |= (self.grid_x >= (c - s)) & (
                    self.grid_x <= (c + s)
                )

        door_mask = wall_stripe & door_span
        wall_mask = wall_stripe & (~door_span)

        # Borders
        bs = self.BORDER_SIZE
        t = 4  # thickness of border line (was int(round(3.5)))
        # left / right
        wall_mask[:, bs - t : bs] = True
        wall_mask[:, W - bs : W - bs + t] = True
        # top / bottom
        wall_mask[bs - t : bs, :] = True
        wall_mask[H - bs : H - bs + t, :] = True

        return wall_mask, door_mask

    # ---------------- Collision ----------------

    def _apply_collisions(self, pos1: torch.Tensor, pos2: torch.Tensor):
        """
        If attempting to cross central wall outside doors => clamp at wall edge with small pushback.
        Collision is triggered when agent radius touches the wall, not just the center.
        Also handles border clamping.
        """
        bs = float(self.BORDER_SIZE)
        door_margin = 1.75  # was 0.5 * 3.5 scale
        agent_r = float(self.variation_space['agent']['radius'].value.item())

        # border clamp first - account for agent radius
        x2, y2 = float(pos2[0]), float(pos2[1])
        x2 = min(max(x2, bs + agent_r), self.IMG_SIZE - bs - agent_r)
        y2 = min(max(y2, bs + agent_r), self.IMG_SIZE - bs - agent_r)
        pos2c = torch.tensor([x2, y2], dtype=torch.float32)

        # central wall collision - account for agent radius
        half = self.wall_thickness // 2
        c = float(self.WALL_CENTER)

        if self.wall_axis == 1:
            # For vertical wall
            wall_left = c - half
            wall_right = c + half

            effective_left = wall_left - agent_r
            effective_right = wall_right + agent_r

            x1, x2_val = float(pos1[0]), float(pos2c[0])
            y2_val = float(pos2c[1])

            started_left = x1 < c

            if started_left:
                if x2_val > effective_left:
                    if not self._in_any_door_1d(y2_val, door_margin):
                        pos2c[0] = effective_left - 0.5
            else:
                if x2_val < effective_right:
                    if not self._in_any_door_1d(y2_val, door_margin):
                        pos2c[0] = effective_right + 0.5
        else:
            # For horizontal wall
            wall_top = c - half
            wall_bottom = c + half

            effective_top = wall_top - agent_r
            effective_bottom = wall_bottom + agent_r

            y1, y2_val = float(pos1[1]), float(pos2c[1])
            x2_val = float(pos2c[0])

            started_top = y1 < c

            if started_top:
                if y2_val > effective_top:
                    if not self._in_any_door_1d(x2_val, door_margin):
                        pos2c[1] = effective_top - 0.5
            else:
                if y2_val < effective_bottom:
                    if not self._in_any_door_1d(x2_val, door_margin):
                        pos2c[1] = effective_bottom + 0.5

        return pos2c

    def _in_any_door_1d(self, coord_1d: float, margin: float):
        # Locked door blocks every opening; treat the door span as solid wall.
        if not self.door_open:
            return False
        for i in range(self.num_doors):
            c = float(self.door_positions[i])
            s = float(self.door_sizes[i])
            if (c - s - margin) <= coord_1d <= (c + s + margin):
                return True
        return False

    # ---------------- Constraints ----------------

    def _constrain_agent_not_in_wall(self, agent_pos):
        """
        Ensure agent position is not inside the wall (unless in a door).
        Agent can start in either room.
        """
        wall_axis = int(self.variation_space['wall']['axis'].value)
        wall_thickness = int(self.variation_space['wall']['thickness'].value)
        half_thickness = wall_thickness // 2
        agent_r = float(self.variation_space['agent']['radius'].value.item())

        wall_min = self.WALL_CENTER - half_thickness - agent_r
        wall_max = self.WALL_CENTER + half_thickness + agent_r

        if wall_axis == 1:  # vertical wall
            if wall_min <= agent_pos[0] <= wall_max:
                return False
        else:  # horizontal wall
            if wall_min <= agent_pos[1] <= wall_max:
                return False

        return True

    def _constrain_key_in_agent_room(self, key_pos):
        """Key must be in the agent's room and outside the wall zone."""
        agent_pos = self.variation_space['agent']['position'].value
        wall_axis = int(self.variation_space['wall']['axis'].value)
        wall_thickness = int(self.variation_space['wall']['thickness'].value)
        half_thickness = wall_thickness // 2
        key_r = float(self.variation_space['key']['radius'].value.item())

        wall_min = self.WALL_CENTER - half_thickness - key_r
        wall_max = self.WALL_CENTER + half_thickness + key_r

        if wall_axis == 1:  # vertical wall — rooms split on x
            agent_side = agent_pos[0] < self.WALL_CENTER
            key_side = key_pos[0] < self.WALL_CENTER
            if agent_side != key_side:
                return False
            if wall_min <= key_pos[0] <= wall_max:
                return False
        else:  # horizontal wall — rooms split on y
            agent_side = agent_pos[1] < self.WALL_CENTER
            key_side = key_pos[1] < self.WALL_CENTER
            if agent_side != key_side:
                return False
            if wall_min <= key_pos[1] <= wall_max:
                return False

        return True

    def _constrain_target_in_other_room(self, target_pos):
        """Target must be in the room opposite the agent and outside the wall zone.

        DoorKey only makes sense as a cross-room task — same-room targets
        would skip the key/door entirely.
        """
        agent_pos = self.variation_space['agent']['position'].value
        wall_axis = int(self.variation_space['wall']['axis'].value)
        wall_thickness = int(self.variation_space['wall']['thickness'].value)
        half_thickness = wall_thickness // 2
        target_r = float(self.variation_space['target']['radius'].value.item())

        wall_min = self.WALL_CENTER - half_thickness - target_r
        wall_max = self.WALL_CENTER + half_thickness + target_r

        if wall_axis == 1:  # vertical wall
            agent_side = agent_pos[0] < self.WALL_CENTER
            target_side = target_pos[0] < self.WALL_CENTER
            if agent_side == target_side:
                return False
            if wall_min <= target_pos[0] <= wall_max:
                return False
        else:  # horizontal wall
            agent_side = agent_pos[1] < self.WALL_CENTER
            target_side = target_pos[1] < self.WALL_CENTER
            if agent_side == target_side:
                return False
            if wall_min <= target_pos[1] <= wall_max:
                return False

        return True

    def _check_door_fit(self, sizes):
        """
        Ensure at least one door half-extent can fit agent radius.
        """
        num = int(self.variation_space['door']['number'].value)
        agent_r = float(self.variation_space['agent']['radius'].value.item())
        return any(float(s) >= 1.1 * agent_r for s in sizes[:num])

    # ---------------- Convenience setters ----------------

    def _set_state(self, state):
        self.agent_position = torch.tensor(state, dtype=torch.float32)

    def _set_goal_state(self, goal_state):
        self.target_position = torch.tensor(goal_state, dtype=torch.float32)
        self.variation_space['target']['position'].set_value(
            np.array(goal_state, dtype=np.float32)
        )
        self._target_img = self._render_frame(agent_pos=self.target_position)

    def _set_key_state(self, key_state):
        self.key_position = torch.tensor(key_state, dtype=torch.float32)
        self.variation_space['key']['position'].set_value(
            np.array(key_state, dtype=np.float32)
        )

    def _set_has_key(self, has_key):
        self.has_key = bool(has_key)
        # Picking up the key always unlocks the door; this keeps the two
        # flags consistent when the env state is replayed externally.
        if self.has_key:
            self.door_open = True

    def _set_door_open(self, door_open):
        self.door_open = bool(door_open)
