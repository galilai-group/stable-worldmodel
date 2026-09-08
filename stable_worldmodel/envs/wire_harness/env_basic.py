"""
V0 Wire Harness MuJoCo Environment — single mover, sequential targets, no cables.
Follows the Gymnasium API (gymnasium.Env).

Episode flow:
    init_pos → t_pos_1 → t_pos_2 → t_pos_3 → t_pos_4 → t_pos_5 → terminated

stable_worldmodel port of the V0 playground env. Additions over the original
(each marked "swm:" below) follow the same checklist as env.py / WireHarnessEnv:
  - packaged XML + default targets so ``gym.make('swm/WireHarnessBasic-v0')``
    needs no arguments,
  - lazy shared offscreen renderer; ``render()`` always returns an RGB frame
    (required by ``World(add_pixels=True)``),
  - ``variation_space`` (start / target positions) integrated with reset,
  - ``state`` / ``goal_state`` / ``goal`` image in every info dict,
  - ``_set_state`` / ``_set_goal_state`` hooks for dataset-driven evaluation
    (``World.evaluate``), same semantics as WireHarnessEnv.
"""

import os
import time

import numpy as np
import mujoco as mj
import imageio.v2 as imageio
import gymnasium as gym
from gymnasium import spaces

from stable_worldmodel import spaces as swm_spaces  # swm: variation space

from .model.mover import Mover  # swm: packaged Mover (was utils/ path import)


# swm: packaged single-mover XML and default target sequence (platform1's
# targets across the five WireHarnessEnv configurations, in stage order) so the
# env is constructible without arguments.
XML_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data',
    'WireHarness_1Mover.xml',
)

DEFAULT_TARGETS = [
    [5.2, 2.8],  # Configuration 0
    [5.2, 0.6],  # Configuration 1
    [3.0, 2.8],  # Configuration 2
    [3.2, 0.4],  # Configuration 3
    [1.2, 0.8],  # Configuration 4
]

MOVER_START = [4.0, 3.0]

# swm: MuJoCo's EGL backend supports only ONE OpenGL context per process (see
# env.py) — all instances of this env share a single lazily created renderer.
_SHARED_RENDERER = None

# Sample every configurable layout component on reset unless the caller
# supplies ``options['variation']`` or ``options['variation_values']``.
DEFAULT_VARIATIONS = (
    'mover.start_position',
    'mover.target_positions',
)


class WireHarnessBasicEnv(gym.Env):
    """
    Gymnasium environment: one mover visits N targets in sequence.

    Observation (4,):
        [dx_to_target, dy_to_target, dist_to_target_norm, angle_to_target_norm]
        — always relative to the current active target.

    Action (2,):
        [vx, vy] in [-1, 1] — scaled by vel and applied as joint velocities.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    X_MIN, X_MAX = 0.0, 6.72
    Y_MIN, Y_MAX = 0.0, 3.84

    def __init__(
        self,
        stage: int = None,  # swm: ignored, use variation_space targets instead
        xml_path: str = None,  # swm: defaults to the packaged 1-mover XML
        targets: list = None,  # swm: defaults to DEFAULT_TARGETS
        simend: int = 60,
        vel: float = 2.0,
        goal_radius: float = 0.1,
        render_mode: str = None,
        camera_view: str = 'oblique',  # swm: "top" for world-model frames
    ):
        super().__init__()

        xml_path = os.path.abspath(XML_PATH) if xml_path is None else xml_path
        targets = DEFAULT_TARGETS if targets is None else targets

        self.xml_path = xml_path
        self.targets = [list(t) for t in targets]  # [[x1,y1], ..., [x5,y5]]
        self.simend = simend
        self.vel = vel
        self.goal_radius = goal_radius
        self.render_mode = render_mode

        # ── MuJoCo setup ──────────────────────────────────────────────────
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)

        self._platform1_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_BODY, 'platform1'
        )
        if self._platform1_id < 0:
            raise RuntimeError("Body 'platform1' not found in XML.")

        # ── Mover ─────────────────────────────────────────────────────────
        self.mover = Mover(
            env=self,
            mu_index=self._platform1_id,
            mu_start=list(MOVER_START),
            mu_joint='slide_joint1',
            mu_start_move=[0.0, 0.0],
            follow=False,
            max_dist=0.0,
            vel=vel,
            cable_connect=[],
            cable_start_mu=[],
        )
        self.current_target_idx = 0
        self.mover.set_target(*self.targets[0])

        # ── Gymnasium spaces ──────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # ── Camera & renderer ─────────────────────────────────────────────
        # swm: renderer is created lazily and shared process-wide (EGL allows a
        # single context per process); plain training never touches the GL.
        self._video_w, self._video_h = 640, 352
        self.renderer = None
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        self.camera_view = camera_view
        self.cam.azimuth = 90.0
        if camera_view == 'top':
            # swm: near-nadir view (see env.py) — planar task maps ~affinely
            # to pixels; -89.9° avoids the gimbal singularity at exactly -90°.
            self.cam.elevation = -89.9
            self.cam.distance = 6.5
            self.cam.lookat = np.array([3.36, 1.92, 0.0])
        else:
            self.cam.elevation = -60.0
            self.cam.distance = 4.5
            self.cam.lookat = np.array([3.36, 1.6, 0.0])

        # ── Episode state ─────────────────────────────────────────────────
        self.sim_step = 0
        self._max_steps = int(simend * 60)
        self._goal_image = (
            None  # swm: visual goal, rendered at reset / target switch
        )
        self._eval_goal = (
            None  # swm: dataset-eval override target (_set_goal_state)
        )

        # ── Video writer ──────────────────────────────────────────────────
        self._video_writer = None
        self._video_path = None

        # ── Variation space (stable_worldmodel World integration) ─────────
        # swm: declarative start / target positions; reset() reads these back
        # so World-driven variation options control the episode layout.
        lo = np.array([self.X_MIN, self.Y_MIN], dtype=np.float32)
        hi = np.array([self.X_MAX, self.Y_MAX], dtype=np.float32)
        n = len(self.targets)
        self.variation_space = swm_spaces.Dict(
            {
                'mover': swm_spaces.Dict(
                    {
                        'start_position': swm_spaces.Box(
                            low=lo,
                            high=hi,
                            shape=(2,),
                            dtype=np.float32,
                            init_value=np.array(MOVER_START, dtype=np.float32),
                        ),
                        'target_positions': swm_spaces.Box(
                            low=np.tile(lo, (n, 1)),
                            high=np.tile(hi, (n, 1)),
                            shape=(n, 2),
                            dtype=np.float32,
                            init_value=np.array(
                                self.targets, dtype=np.float32
                            ),
                        ),
                    }
                ),
            }
        )

    # ──────────────────────────────────────────────────────────────────────
    # Gymnasium core API
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # swm: let World / options drive the episode layout, then read back.
        swm_spaces.reset_variation_space(
            self.variation_space,
            seed=seed,
            options=options,
            default_variations=DEFAULT_VARIATIONS,
        )
        mover_var = self.variation_space.spaces['mover']
        start = [float(v) for v in mover_var.spaces['start_position'].value]
        self.targets = [
            [float(x), float(y)]
            for x, y in mover_var.spaces['target_positions'].value
        ]

        mj.mj_resetData(self.model, self.data)

        self.sim_step = 0
        self.current_target_idx = 0
        self._eval_goal = None

        self.mover.reward_sum = 0
        self.mover.done = False
        self.mover.coords_x = []
        self.mover.coords_y = []
        self.mover.path = []
        self.mover.path_original = []

        self._teleport(start)  # swm: variation-space start (was XML rest pose)

        self.mover.set_target(*self.targets[0])
        self.mover.update_pos()

        # swm: visual goal for the active target; None when rendering is off.
        self._goal_image = self._render_goal_image()

        return self._get_obs(), self._make_info()

    def _mask_action(self, action: np.ndarray) -> np.ndarray:
        vx, vy = float(action[0]), float(action[1])
        if self.mover.x <= self.X_MIN and vx < 0:
            vx = 0.0
        if self.mover.x >= self.X_MAX and vx > 0:
            vx = 0.0
        if self.mover.y <= self.Y_MIN and vy < 0:
            vy = 0.0
        if self.mover.y >= self.Y_MAX and vy > 0:
            vy = 0.0
        return np.array([vx, vy], dtype=np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        action = self._mask_action(action)

        self.mover.update_pos()
        self.mover.make_move(action.tolist())

        # Distance before stepping (used for dense shaping)
        prev_dist = self.mover.get_distance_target(norm=False)

        # Physics: advance one control frame (1/60 s)
        simstart = self.data.time
        while self.data.time - simstart < 1.0 / 60.0:
            mj.mj_step(self.model, self.data)

        self.sim_step += 1
        self.mover.update_pos()

        current_dist = self.mover.get_distance_target(norm=False)

        # Dense reward: positive when moving toward the current target
        reward = (prev_dist - current_dist) * 10.0 - 0.01

        terminated = False
        if current_dist < self.goal_radius:
            reward += 10.0
            if self._eval_goal is not None:
                # swm: dataset-eval goal reached — success, episode over.
                terminated = True
            else:
                self.current_target_idx += 1
                if self.current_target_idx >= len(self.targets):
                    terminated = True  # all targets reached
                else:
                    # Advance to next target — episode continues
                    self.mover.set_target(
                        *self.targets[self.current_target_idx]
                    )
                    # swm: goal signal follows the active target.
                    self._goal_image = self._render_goal_image()

        truncated = self.sim_step >= self._max_steps

        self.mover.coords_x.append(self.mover.x)
        self.mover.coords_y.append(self.mover.y)

        if self._video_writer is not None:
            self._capture_frame()

        info = self._make_info(
            sim_step=self.sim_step,
            dist_to_target=float(current_dist),
            current_target_idx=self.current_target_idx,
            targets_reached=self.current_target_idx,
        )
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        """RGB array (H, W, 3) uint8 of the current scene.

        swm: World(add_pixels=True) requires render() to return an RGB frame,
        so the offscreen renderer is created lazily and we render regardless
        of render_mode.
        """
        if not self._ensure_renderer():
            return None
        self.renderer.update_scene(
            self.data, camera=self.cam, scene_option=self.opt
        )
        return self.renderer.render()

    def close(self):
        self.finish_video()

    # ──────────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        dx = self.mover.get_distance_x(self.mover.x_t)
        dy = self.mover.get_distance_y(self.mover.y_t)
        dist_norm = self.mover.get_distance_target(norm=True)
        angle_norm = self.mover.get_angle_target(norm=True)
        return np.array([dx, dy, dist_norm, angle_norm], dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # swm: shared renderer, teleport helper
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_renderer(self) -> bool:
        global _SHARED_RENDERER
        if _SHARED_RENDERER is None:
            try:
                _SHARED_RENDERER = mj.Renderer(
                    self.model, width=self._video_w, height=self._video_h
                )
            except Exception as e:
                print(f'[WireHarnessBasicEnv] Renderer unavailable: {e}')
                return False
        self.renderer = _SHARED_RENDERER
        return True

    def _teleport(self, position):
        px, py = float(position[0]), float(position[1])
        self.data.joint(self.mover.joint_x).qpos[0] = (
            px - self.mover.mu_start[0]
        )
        self.data.joint(self.mover.joint_y).qpos[0] = (
            py - self.mover.mu_start[1]
        )
        self.data.joint(self.mover.joint_x).qvel[0] = 0.0
        self.data.joint(self.mover.joint_y).qvel[0] = 0.0
        mj.mj_forward(self.model, self.data)

    # ──────────────────────────────────────────────────────────────────────
    # swm: world-model info signals (stable_worldmodel checklist)
    # ──────────────────────────────────────────────────────────────────────

    def _state_vec(self) -> np.ndarray:
        """Compact ground-truth state: the mover's (x, y), shape (2,)."""
        return np.array([self.mover.x, self.mover.y], dtype=np.float32)

    def _goal_state_vec(self) -> np.ndarray:
        """Compact goal signal: the active target's (x, y), shape (2,)."""
        if self._eval_goal is not None:
            return self._eval_goal.reshape(-1).astype(np.float32)
        return np.array([self.mover.x_t, self.mover.y_t], dtype=np.float32)

    def _make_info(self, **extra) -> dict:
        """Info dict carrying the world-model signals on every reset/step."""
        info = {
            'state': self._state_vec(),
            'goal_state': self._goal_state_vec(),
            **extra,
        }
        if self._goal_image is not None:
            info['goal'] = self._goal_image
        return info

    # ──────────────────────────────────────────────────────────────────────
    # swm: dataset-driven evaluation hooks (World.evaluate callables)
    # ──────────────────────────────────────────────────────────────────────

    def _set_state(self, state):
        """Place the mover at a dataset-provided start state (x, y), shape (2,)."""
        pos = np.asarray(state, dtype=np.float32).reshape(2)
        self._teleport(pos)
        self.mover.update_pos()
        self.sim_step = 0

    def _set_goal_state(self, goal_state):
        """Point the success target at a dataset goal (x, y), shape (2,).

        step() terminates (= success in dataset eval) when the mover is within
        goal_radius of it. Transient: reset() restores the sequential targets.
        """
        pos = np.asarray(goal_state, dtype=np.float32).reshape(2)
        self._eval_goal = pos
        self.mover.set_target(float(pos[0]), float(pos[1]))
        self._goal_image = self._render_goal_image()

    def _render_goal_image(self):
        """RGB image of the goal (mover at the active target). No cables, so a
        bare teleport + mj_forward is a faithful goal state; the episode state
        is restored afterwards. None when rendering is off (training)."""
        if self.render_mode is None or not self._ensure_renderer():
            return None
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        t = self.data.time
        self._teleport(self._goal_state_vec())
        self.renderer.update_scene(
            self.data, camera=self.cam, scene_option=self.opt
        )
        img = self.renderer.render().copy()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.time = t
        mj.mj_forward(self.model, self.data)
        return img

    # ──────────────────────────────────────────────────────────────────────
    # Video
    # ──────────────────────────────────────────────────────────────────────

    def start_video(self, path: str, fps: int = 30):
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._video_path = path
        self._ensure_renderer()
        try:
            self._video_writer = imageio.get_writer(
                path, fps=fps, codec='libx264', macro_block_size=1
            )
        except Exception as e:
            self._video_writer = None
            print(f'[Video] Could not open writer: {e}')

    def _capture_frame(self):
        if self._video_writer is None or self.renderer is None:
            return
        self.renderer.update_scene(
            self.data, camera=self.cam, scene_option=self.opt
        )
        self._video_writer.append_data(self.renderer.render())

    def finish_video(self):
        if self._video_writer is not None:
            try:
                self._video_writer.close()
            finally:
                self._video_writer = None
            time.sleep(0.1)
            if self._video_path and os.path.exists(self._video_path):
                print(f'[Video] saved to {self._video_path}')
            self._video_path = None
