"""
Piecewise Navigation Environment.

Like TwoRoomEnv but without any central wall — agent navigates a single
open room bounded only by the 4 border walls.

The room is virtually divided into a grid_n × grid_n grid of zones.
Each zone applies an additive bias to the agent's motion.
An optional rendering mode colors each zone with a distinct background.
"""

from __future__ import annotations

import colorsys
import math

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_worldmodel import spaces as swm_spaces

DEFAULT_VARIATIONS = ('agent.position', 'target.position')


class PiecewiseEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    IMG_SIZE = 224
    BORDER_SIZE = 14

    def __init__(
        self,
        render_mode: str = 'rgb_array',
        render_target: bool = False,
        grid_n: int = 2,
        init_value: dict | None = None,
    ):
        assert render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        self.render_target_flag = bool(render_target)
        self.grid_n = int(grid_n)

        y = torch.arange(self.IMG_SIZE, dtype=torch.float32)
        x = torch.arange(self.IMG_SIZE, dtype=torch.float32)
        self.grid_y, self.grid_x = torch.meshgrid(y, x, indexing='ij')  # (H,W)

        # Observation: agent(2) + target(2)
        self.observation_space = spaces.Box(
            low=0, high=self.IMG_SIZE, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.env_name = 'Piecewise'

        self.variation_space = self._build_variation_space()
        if init_value is not None:
            self.variation_space.set_init_value(init_value)

        self.agent_position = torch.zeros(2, dtype=torch.float32)
        self.target_position = torch.zeros(2, dtype=torch.float32)
        self._target_img = None

    # ---------------- Variation Space ----------------

    def _build_variation_space(self):
        pos_min = float(self.BORDER_SIZE)
        pos_max = float(self.IMG_SIZE - self.BORDER_SIZE - 1)
        n = self.grid_n
        n_zones = n * n

        zones_dict = {}
        for i in range(n_zones):
            angle = 2.0 * math.pi * i / n_zones
            bx = 2.0 * math.cos(angle)
            by = 2.0 * math.sin(angle)
            zones_dict[f'bias_{i}'] = swm_spaces.Box(
                low=np.array([-4.0, -4.0], dtype=np.float32),
                high=np.array([4.0, 4.0], dtype=np.float32),
                init_value=np.array([bx, by], dtype=np.float32),
                shape=(2,),
                dtype=np.float32,
            )
            # Evenly spaced pastel hues across zones
            h = i / n_zones
            r, g, b = colorsys.hsv_to_rgb(h, 0.25, 1.0)
            zones_dict[f'color_{i}'] = swm_spaces.RGBBox(
                init_value=np.array(
                    [int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8
                )
            )

        zones_sampling_order = [f'bias_{i}' for i in range(n_zones)] + [
            f'color_{i}' for i in range(n_zones)
        ]

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
                        ),
                    },
                    sampling_order=['color', 'radius', 'position'],
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
                'border': swm_spaces.Dict(
                    {
                        'color': swm_spaces.RGBBox(
                            init_value=np.array([0, 0, 0], dtype=np.uint8)
                        ),
                    }
                ),
                'rendering': swm_spaces.Dict(
                    {
                        'render_target': swm_spaces.Discrete(2, init_value=0),
                        'render_zones': swm_spaces.Discrete(2, init_value=0),
                        'render_bias_field': swm_spaces.Discrete(
                            2, init_value=0
                        ),
                    }
                ),
                'task': swm_spaces.Dict(
                    {
                        'min_steps': swm_spaces.Discrete(
                            100, start=15, init_value=25
                        ),
                    }
                ),
                'zones': swm_spaces.Dict(
                    zones_dict,
                    sampling_order=zones_sampling_order,
                ),
            },
            sampling_order=[
                'background',
                'border',
                'agent',
                'task',
                'target',
                'rendering',
                'zones',
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

        self.agent_position = torch.as_tensor(agent_pos, dtype=torch.float32)
        self.target_position = torch.as_tensor(target_pos, dtype=torch.float32)

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
        zone_idx = self._get_zone(self.agent_position)
        bias = torch.as_tensor(
            self.variation_space['zones'][f'bias_{zone_idx}'].value,
            dtype=torch.float32,
        )
        pos_next = self.agent_position + action_t * speed + bias

        self.agent_position = self._apply_border_clamp(pos_next)

        dist = float(torch.norm(self.agent_position - self.target_position))
        terminated = dist < 16.0
        reward = 0.0

        obs = self._get_obs()
        info = self._get_info()
        info['distance_to_target'] = dist
        return obs, reward, terminated, False, info

    def render(self):
        img_chw = (
            self._render_frame(agent_pos=self.agent_position).cpu().numpy()
        )
        return img_chw.transpose(1, 2, 0)  # CHW -> HWC

    # ---------------- Zone helpers ----------------

    def _get_zone(self, pos: torch.Tensor) -> int:
        """Return zone index (row * grid_n + col) for the given position."""
        n = self.grid_n
        bs = self.BORDER_SIZE
        play_size = float(self.IMG_SIZE - 2 * bs)
        col = int((float(pos[0]) - bs) / play_size * n)
        row = int((float(pos[1]) - bs) / play_size * n)
        col = max(0, min(n - 1, col))
        row = max(0, min(n - 1, row))
        return row * n + col

    def _zone_pixel_map(self) -> torch.Tensor:
        """Return (H, W) long tensor with zone index per pixel (-1 outside playfield)."""
        n = self.grid_n
        bs = self.BORDER_SIZE
        play_w = self.IMG_SIZE - 2 * bs
        play_h = self.IMG_SIZE - 2 * bs

        zone_map = torch.full(
            (self.IMG_SIZE, self.IMG_SIZE), -1, dtype=torch.long
        )
        for row in range(n):
            for col in range(n):
                y_lo = bs + int(row * play_h / n)
                y_hi = bs + int((row + 1) * play_h / n)
                x_lo = bs + int(col * play_w / n)
                x_hi = bs + int((col + 1) * play_w / n)
                zone_map[y_lo:y_hi, x_lo:x_hi] = row * n + col
        return zone_map

    # ---------------- Internal helpers ----------------

    def _get_obs(self):
        return torch.tensor(
            [
                float(self.agent_position[0]),
                float(self.agent_position[1]),
                float(self.target_position[0]),
                float(self.target_position[1]),
            ],
            dtype=torch.float32,
        )

    def _get_info(self):
        return {
            'env_name': self.env_name,
            'proprio': self.agent_position.detach().cpu().numpy(),
            'state': self.agent_position.detach().cpu().numpy(),
            'goal_state': self.target_position.detach().cpu().numpy(),
        }

    def _apply_border_clamp(self, pos: torch.Tensor) -> torch.Tensor:
        bs = float(self.BORDER_SIZE)
        agent_r = float(self.variation_space['agent']['radius'].value.item())
        lo = bs + agent_r
        hi = self.IMG_SIZE - bs - agent_r
        x = float(pos[0].clamp(lo, hi))
        y = float(pos[1].clamp(lo, hi))
        return torch.tensor([x, y], dtype=torch.float32)

    # ---------------- Rendering ----------------

    def _render_frame(self, agent_pos: torch.Tensor):
        H = W = self.IMG_SIZE

        bg = self.variation_space['background']['color'].value
        img = torch.empty((3, H, W), dtype=torch.uint8)
        img[0].fill_(int(bg[0]))
        img[1].fill_(int(bg[1]))
        img[2].fill_(int(bg[2]))

        render_zones = bool(
            self.variation_space['rendering']['render_zones'].value
        )
        if render_zones:
            zone_map = self._zone_pixel_map()
            for i in range(self.grid_n * self.grid_n):
                mask = zone_map == i
                if mask.any():
                    zc = self.variation_space['zones'][f'color_{i}'].value
                    img[0, mask] = int(zc[0])
                    img[1, mask] = int(zc[1])
                    img[2, mask] = int(zc[2])

        border_mask = self._border_mask()
        border_color = self.variation_space['border']['color'].value
        if border_mask.any():
            img[0, border_mask] = int(border_color[0])
            img[1, border_mask] = int(border_color[1])
            img[2, border_mask] = int(border_color[2])

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

        render_bias_field = bool(
            self.variation_space['rendering']['render_bias_field'].value
        )
        if render_bias_field:
            img = self._render_bias_field(img)

        agent_color = self.variation_space['agent']['color'].value
        agent_r = float(self.variation_space['agent']['radius'].value.item())
        agent_dot = self._gaussian_dot(agent_pos, agent_r)
        img = self._alpha_blend(img, agent_dot, agent_color)

        return img

    def _render_bias_field(
        self, img: torch.Tensor, samples_per_zone: int = 3
    ) -> torch.Tensor:
        """Overlay a vector field showing per-zone biases as arrows (via quiver)."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        n = self.grid_n
        bs = self.BORDER_SIZE
        H = W = self.IMG_SIZE
        play_w = W - 2 * bs
        play_h = H - 2 * bs

        X, Y, U, V = [], [], [], []
        for row in range(n):
            for col in range(n):
                zone_idx = row * n + col
                bias = self.variation_space['zones'][f'bias_{zone_idx}'].value
                bx, by = float(bias[0]), float(bias[1])

                y_lo = bs + row * play_h / n
                y_hi = bs + (row + 1) * play_h / n
                x_lo = bs + col * play_w / n
                x_hi = bs + (col + 1) * play_w / n

                for si in range(samples_per_zone):
                    for sj in range(samples_per_zone):
                        cx = (
                            x_lo
                            + (sj + 0.5) * (x_hi - x_lo) / samples_per_zone
                        )
                        cy = (
                            y_lo
                            + (si + 0.5) * (y_hi - y_lo) / samples_per_zone
                        )
                        X.append(cx)
                        Y.append(cy)
                        U.append(bx)
                        V.append(
                            by
                        )  # y-down matches image coords with ylim(H, 0)

        dpi = 100
        fig = Figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # y increases downward, matching image coords
        ax.axis('off')
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        # Scale so the longest arrow reaches ~15 % of a zone cell
        cell_size = min(play_w, play_h) / n
        scale = 5.0 * math.sqrt(2) / (cell_size * 0.15)

        ax.quiver(
            X,
            Y,
            U,
            V,
            color='#111111',
            alpha=0.85,
            scale=scale,
            scale_units='xy',
            angles='xy',
            width=0.006,
            headwidth=4,
            headlength=4,
            headaxislength=3.5,
            minlength=0.5,
        )

        canvas.draw()
        buf, (w, h) = canvas.print_to_buffer()
        rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)

        base = img.permute(1, 2, 0).cpu().float()
        alpha = (
            torch.tensor(rgba[..., 3], dtype=torch.float32).unsqueeze(-1)
            / 255.0
        )
        overlay = torch.tensor(rgba[..., :3], dtype=torch.float32)
        result = base * (1.0 - alpha) + overlay * alpha
        return result.to(torch.uint8).permute(2, 0, 1)

    def _border_mask(self):
        bs = self.BORDER_SIZE
        t = 4
        H = W = self.IMG_SIZE
        mask = torch.zeros((H, W), dtype=torch.bool)
        mask[:, bs - t : bs] = True
        mask[:, W - bs : W - bs + t] = True
        mask[bs - t : bs, :] = True
        mask[H - bs : H - bs + t, :] = True
        return mask

    @staticmethod
    def _alpha_blend(img_u8, alpha_01, rgb_u8):
        a = alpha_01.clamp(0, 1).to(torch.float32)
        out = img_u8.to(torch.float32)
        for c in range(3):
            out[c] = out[c] * (1.0 - a) + float(rgb_u8[c]) * a
        return out.to(torch.uint8)

    def _gaussian_dot(self, pos_xy: torch.Tensor, radius: float):
        dx = self.grid_x - float(pos_xy[0])
        dy = self.grid_y - float(pos_xy[1])
        dist2 = dx * dx + dy * dy
        std = max(1e-6, float(radius))
        dot = torch.exp(-dist2 / (2.0 * std * std))
        m = dot.max()
        if m > 0:
            dot = dot / m
        return dot

    # ---------------- Convenience setters ----------------

    def _set_state(self, state):
        self.agent_position = torch.tensor(state, dtype=torch.float32)

    def _set_goal_state(self, goal_state):
        self.target_position = torch.tensor(goal_state, dtype=torch.float32)
        self.variation_space['target']['position'].set_value(
            np.array(goal_state, dtype=np.float32)
        )
        self._target_img = self._render_frame(agent_pos=self.target_position)
