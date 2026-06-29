import itertools
import math
import os
import sys
import time

import numpy as np
import mujoco as mj
import imageio.v2 as imageio
import gymnasium as gym
from gymnasium import spaces
from stable_worldmodel import spaces as swm_spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from model.mover import Mover


# MuJoCo's EGL backend supports only ONE OpenGL context per process, so a
# pool of envs (e.g. swm.World / EnvPool, all in one process) cannot each own a
# renderer — the second eglMakeCurrent fails with EGL_BAD_ACCESS. Rendering is
# sequential within the process, so all envs share a single renderer. Every env
# loads the same XML, so the model is structurally identical and rendering any
# env's MjData through the shared renderer is correct (poses come from data).
_SHARED_RENDERER = None


class WireHarnessEnv(gym.Env):
    """
    Gymnasium env: N movers, one target configuration (stage), physical cables.
    Fully learned control; collision awareness via local grid maps in the
    observation and grid penalties in the reward (ported from environment_ludwig_sb3).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    X_MIN, X_MAX = 0.0, 6.72
    Y_MIN, Y_MAX = 0.0, 3.84

    GRID_W = 72
    GRID_H = 50

    APF_INFLUENCE = 0.5
    APF_ETA       = 0.5

    SETTLE_TIME = 0.5   # seconds of physics after teleport-reset so cables relax

    def __init__(
        self,
        stage: int = 0,
        xml_path: str = None,
        mover_starts: list = None,
        mover_body_names: list = None,
        mover_joint_names: list = None,
        targets: list = None,
        simend: int = None,
        vel: float = None,
        goal_radius: float = None,
        w_obstacle_map: float = None,
        w_cable_map: float = None,
        cable_connect: list = None,
        cable_start_mu: list = None,
        render_mode: str = None,
        camera_view: str = "oblique",
    ):
        super().__init__()

        # config.py is the single source of truth; explicit args override it.
        xml_path          = os.path.abspath(config.XML_PATH) if xml_path is None else xml_path
        mover_starts      = config.MOVER_STARTS      if mover_starts      is None else mover_starts
        mover_body_names  = config.MOVER_BODY_NAMES  if mover_body_names  is None else mover_body_names
        mover_joint_names = config.MOVER_JOINT_NAMES if mover_joint_names is None else mover_joint_names
        targets           = config.MOVER_TARGETS     if targets           is None else targets
        simend            = config.SIMEND            if simend            is None else simend
        vel               = config.VEL               if vel               is None else vel
        goal_radius       = config.GOAL_RADIUS       if goal_radius       is None else goal_radius
        w_obstacle_map    = config.W_OBSTACLE_MAP    if w_obstacle_map    is None else w_obstacle_map
        w_cable_map       = config.W_CABLE_MAP       if w_cable_map       is None else w_cable_map
        cable_connect     = config.CABLE_CONNECT     if cable_connect     is None else cable_connect
        cable_start_mu    = config.CABLE_START_MU    if cable_start_mu    is None else cable_start_mu

        self.xml_path    = xml_path
        self.simend      = simend
        self.vel         = vel
        self.goal_radius = goal_radius
        self.render_mode = render_mode
        self.num_agents  = len(mover_body_names)
        self._frame_time = 1.0 / 60.0

        # Configuration-major: targets[k] = configuration k, one [x, y] per
        # mover in mover order (MOVER_TARGETS layout from config).
        self.targets  = [[list(t) for t in konf] for konf in targets]
        self.n_stages = len(self.targets)
        for k, konf in enumerate(self.targets):
            if len(konf) != self.num_agents:
                raise ValueError(
                    f"Configuration {k} has {len(konf)} targets, "
                    f"expected one per mover ({self.num_agents})."
                )
        if not 0 <= stage < self.n_stages:
            raise ValueError(f"stage must be in [0, {self.n_stages - 1}], got {stage}")
        self.stage = stage

        self.mover_starts   = [list(s) for s in mover_starts]
        self.w_obstacle_map = w_obstacle_map
        self.w_cable_map    = w_cable_map

        # ── MuJoCo ────────────────────────────────────────────────────────
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data  = mj.MjData(self.model)
        self.model.opt.timestep = 0.00025
        # Substep counts instead of absolute-time loops: MuJoCo 3.x auto-resets
        # (zeroing data.time) on NaN/huge state, which would turn a
        # `while data.time - simstart < dt` loop into a near-infinite
        # re-simulation and silently teleport everything to the rest pose.
        self._frame_substeps  = math.ceil(self._frame_time / self.model.opt.timestep)
        self._settle_substeps = math.ceil(self.SETTLE_TIME / self.model.opt.timestep)

        # ── Resolve body IDs ──────────────────────────────────────────────
        body_ids = []
        for name in mover_body_names:
            bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise RuntimeError(f"Body '{name}' not found in XML.")
            body_ids.append(bid)

        # Cable bodies for the global collision map (selected by name, not by
        # index range). The bodies adjacent to each platform attachment
        # (CABLE_START_MU from config_base / environment_ludwig_sb3) are
        # excluded so attachment stubs don't register as permanent collisions
        # right under the platforms; without that list, fall back to excluding
        # the welded B_first/B_last bodies only.
        excluded = set()
        for ids in (cable_start_mu or []):
            excluded.update(ids)
        self._cable_bodies = []
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if not name.startswith("Wire"):
                continue
            if excluded:
                if i in excluded:
                    continue
            elif name.endswith("B_first") or name.endswith("B_last"):
                continue
            self._cable_bodies.append((i, int(name[4])))

        # ── Instantiate movers ────────────────────────────────────────────
        cable_connect  = cable_connect  or [[] for _ in range(self.num_agents)]
        cable_start_mu = cable_start_mu or [[] for _ in range(self.num_agents)]
        self.movers = []
        for i in range(self.num_agents):
            m = Mover(
                env=self,
                mu_index=body_ids[i],
                mu_start=list(mover_starts[i]),
                mu_joint=mover_joint_names[i],
                mu_start_move=[0.0, 0.0],
                follow=False,
                max_dist=float("inf"),
                vel=vel,
                cable_connect=list(cable_connect[i]),
                cable_start_mu=list(cable_start_mu[i]),
            )
            m.set_target(*self.targets[self.stage][i])
            self.movers.append(m)

        # Convenience references (used in tests/callbacks) — all 5 movers
        self.mover1 = self.movers[0]
        self.mover2 = self.movers[1]
        self.mover3 = self.movers[2]
        self.mover4 = self.movers[3]
        self.mover5 = self.movers[4]

        # ── Global collision map (cables + movers) ────────────────────────
        # collision_map carries both (cable ids 1..4, then mover ids overwrite);
        # the split cable/mover maps feed the cable-on-mover constraint check.
        self.collision_map       = np.zeros((self.GRID_H, self.GRID_W), dtype=np.float32)
        self.cable_collision_map = np.zeros((self.GRID_H, self.GRID_W), dtype=np.float32)
        self.mover_collision_map = np.zeros((self.GRID_H, self.GRID_W), dtype=np.float32)

        # ── Episode state ─────────────────────────────────────────────────
        self._reached    = [False] * self.num_agents
        self.sim_step    = 0
        self._max_steps  = int(simend * 60)
        self._goal_image = None   # visual goal, rendered once per episode at reset
        self._eval_goal  = None   # dataset-eval override target (set via _set_goal_state)

        # ── Spaces ────────────────────────────────────────────────────────
        # Matches Mover/get_states: per pair (i<j) two raw offsets, plus
        # dist_target_norm + angle_target_norm per mover. No grid maps in obs.
        self._n_obs = (2 * self.num_agents
                       + self.num_agents * (self.num_agents - 1))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2 * self.num_agents,), dtype=np.float32
        )




        # ── Camera / renderer ─────────────────────────────────────────────
        # Created lazily and shared process-wide (see _ensure_renderer); not
        # built eagerly here so a pool of envs doesn't open conflicting EGL
        # contexts, and so non-rendering runs (training) never touch the GL.
        self._video_w, self._video_h = 640, 352
        self.renderer = None
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        # "oblique" (default): the angled view used for human-readable training
        # videos. "top": near-nadir view so the planar (x, y) task maps ~affinely
        # to pixels and cables are seen flat — better for world-model frames.
        # (-89.9° avoids the gimbal singularity at exactly -90°.)
        self.camera_view = camera_view
        self.cam.azimuth = 90.0
        if camera_view == "top":
            self.cam.elevation = -89.9
            self.cam.distance  = 6.5
            self.cam.lookat    = np.array([3.36, 1.92, 0.0])  # table center
        else:
            self.cam.elevation = -60.0
            self.cam.distance  = 4.5
            self.cam.lookat    = np.array([3.36, 1.6, 0.0])

        self._video_writer = None
        self._video_path   = None

        # ── Variation space (stable_worldmodel World integration) ─────────
        # Declarative: per-mover start/target positions, initialised from the
        # constructor args. reset() resets these to their init values; the
        # existing _sample_starts / stage logic still drives the actual physics.
        lo = np.array([self.X_MIN, self.Y_MIN], dtype=np.float32)
        hi = np.array([self.X_MAX, self.Y_MAX], dtype=np.float32)
        self.variation_space = swm_spaces.Dict({
            f"mover_{i+1}": swm_spaces.Dict({
                "start_position": swm_spaces.Box(
                    low=lo, high=hi, shape=(2,), dtype=np.float32,
                    init_value=np.array(self.mover_starts[i], dtype=np.float32),
                ),
                "target_position": swm_spaces.Box(
                    low=lo, high=hi, shape=(2,), dtype=np.float32,
                    init_value=np.array(self.targets[self.stage][i], dtype=np.float32),
                ),
            })
            for i in range(self.num_agents)
        })

        

    # ──────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        swm_spaces.reset_variation_space(
            self.variation_space, seed=seed, options=options,
        )

        # variation_space is the single source of truth for this episode's start
        # and target poses. Goal is FIXED to this env's stage configuration (one
        # SAC is trained per target configuration, so the target must not vary);
        # only the start is sampled — jointly, over the discrete configuration
        # set {MOVER_STARTS} ∪ {targets[j != stage]} (see _sample_starts). When a
        # caller injects explicit variation_values (dataset eval), trust those.
        options = options or {}
        if not options.get('variation_values'):
            start_cfg = self._sample_starts(options)
            self._write_variation_positions(start_cfg, self.targets[self.stage])
        starts, targets = self._read_variation_positions()

        mj.mj_resetData(self.model, self.data)
        self.sim_step = 0
        self._reached = [False] * self.num_agents
        # Drop any dataset-eval target override; a normal episode uses the
        # variation targets (set below). _set_goal_state re-arms it after reset.
        self._eval_goal = None

        self._teleport_and_settle(starts)

        for i, m in enumerate(self.movers):
            m.reward_sum    = 0
            m.done          = False
            m.coords_x      = []
            m.coords_y      = []
            m.path          = []
            m.path_original = []
            m.set_target(*targets[i])
            m.update_pos()

        self._update_collision_map()
        for m in self.movers:
            self._update_local_maps(m)

        # Visual goal for this episode (stage target is fixed for the whole
        # episode). Rendered once here; None when rendering is off (training).
        self._goal_image = self._render_goal_image()

        return self._get_obs(), self._make_info(stage=self.stage)

    def set_stage(self, stage: int):
        """Switch the active target configuration mid-episode (chained execution)."""
        if not 0 <= stage < self.n_stages:
            raise ValueError(f"stage must be in [0, {self.n_stages - 1}], got {stage}")
        self.stage    = stage
        self._reached = [False] * self.num_agents
        for i, m in enumerate(self.movers):
            m.set_target(*self.targets[stage][i])

    def _sample_starts(self, options):
        """
        Start positions for this episode. Default: uniform over the initial
        XML layout and every configuration except this stage's own — covers
        every possible predecessor when stages are chained in arbitrary order.
        options={"start": "initial"} forces the XML layout (chained test).
        """
        if options and options.get("start") == "initial":
            return self.mover_starts
        candidates = [self.mover_starts] + [
            self.targets[j] for j in range(self.n_stages) if j != self.stage
        ]
        return candidates[self.np_random.integers(len(candidates))]

    def _write_variation_positions(self, starts, targets):
        """Record this episode's start/target poses into the variation space.

        `starts` / `targets` are one (x, y) per mover (mover order). Writing them
        here makes variation_space the single source of truth that reset() reads
        back to drive the physics; the Box bounds also validate the values.
        """
        for i in range(self.num_agents):
            mv = self.variation_space.spaces[f"mover_{i+1}"]
            mv.spaces["start_position"].set_value(
                np.asarray(starts[i], dtype=np.float32))
            mv.spaces["target_position"].set_value(
                np.asarray(targets[i], dtype=np.float32))

    def _read_variation_positions(self):
        """Read back the per-mover (start, target) poses from the variation space.

        Returns (starts, targets), each a list of [x, y] floats in mover order.
        """
        starts, targets = [], []
        for i in range(self.num_agents):
            mv = self.variation_space.spaces[f"mover_{i+1}"]
            starts.append([float(v) for v in mv.spaces["start_position"].value])
            targets.append([float(v) for v in mv.spaces["target_position"].value])
        return starts, targets

    def _teleport_and_settle(self, positions):
        """
        Place movers at `positions` (one (x, y) per mover) via their slide-joint
        qpos (XML rest pose equals mover_starts, so offset = desired − rest),
        then run SETTLE_TIME of physics with the platforms pinned so the cables
        relax to the new endpoints. Used both to seed the episode start and to
        build the goal configuration (movers at targets + equilibrium cables).

        Returns False if MuJoCo's auto-reset fired (NaN/huge state) and wiped
        the teleport, leaving the XML rest layout; True otherwise.
        """
        for m, (px, py) in zip(self.movers, positions):
            self.data.joint(m.joint_x).qpos[0] = px - m.mu_start[0]
            self.data.joint(m.joint_y).qpos[0] = py - m.mu_start[1]
            self.data.joint(m.joint_x).qvel[0] = 0.0
            self.data.joint(m.joint_y).qvel[0] = 0.0
        mj.mj_forward(self.model, self.data)

        settle_start = self.data.time
        for _ in range(self._settle_substeps):
            mj.mj_step(self.model, self.data)
            # data.time going backwards = MuJoCo auto-reset fired (NaN/huge
            # state) and wiped the teleport — fall back to the XML rest layout.
            if (self.data.time < settle_start
                    or np.isnan(self.data.qpos).any()
                    or np.abs(self.data.qpos).max() > 1e6):
                mj.mj_resetData(self.model, self.data)
                mj.mj_forward(self.model, self.data)
                return False
            for m in self.movers:   # pin platforms while the cables relax
                self.data.joint(m.joint_x).qvel[0] = 0.0
                self.data.joint(m.joint_y).qvel[0] = 0.0
        self.data.qvel[:] = 0.0
        return True

    def _mask_action(self, action: np.ndarray) -> np.ndarray:
        action = action.copy()
        for i, m in enumerate(self.movers):
            if m.x <= self.X_MIN and action[2*i]   < 0: action[2*i]   = 0.0
            if m.x >= self.X_MAX and action[2*i]   > 0: action[2*i]   = 0.0
            if m.y <= self.Y_MIN and action[2*i+1] < 0: action[2*i+1] = 0.0
            if m.y >= self.Y_MAX and action[2*i+1] > 0: action[2*i+1] = 0.0
        return action

    def step(self, action):
        action = np.atleast_1d(np.asarray(action, dtype=np.float32))
        if np.isnan(action).any():
            return self._get_obs(), -0.02, True, False, \
                self._make_info(sim_step=self.sim_step, n_at_goal=0, nan_action=True)

        action = np.clip(action, -1.0, 1.0)
        action = self._mask_action(action)

        for m in self.movers:
            m.update_pos()
        prev_dists = [m.get_distance_target(norm=False) for m in self.movers]

        # Collision maps must be current BEFORE the action decision so the
        # constraint logic (collision avoidance + cable-on-mover) reads this
        # step's state, not the previous one.
        self._update_collision_map()
        for m in self.movers:
            self._update_local_maps(m)

        # A mover whose own cable currently runs across ANOTHER mover is held
        # (its action is overridden below) until the crossing clears.
        stopped = [self._check_cable_on_mover(m.cable_connect) for m in self.movers]

        # ── Per-mover action (v0_4 pipeline) ───────────────────────────────
        #   far from target (> 0.5 m) → learned action
        #   within 0.5 m of target    → deterministic straight-to-target
        #   collision / spacing / cable-on-mover → constraint override
        self.actions = []
        for i, m in enumerate(self.movers):
            if abs(m.get_distance_target(norm=False)) > 0.5:
                a = np.asarray([action[2*i], action[2*i + 1]], dtype=float)
            else:
                a = np.asarray(m.deterministic_move_t(), dtype=float)

            constraint_action = m.choose_constraint_action(
                self.sim_step,
                m.get_distance(self.movers[0].x, self.movers[0].y),
            )

            if not np.array_equal(constraint_action, [0, 0]) or stopped[i]:
                def_action = np.asarray(constraint_action, dtype=float)
            else:
                def_action = a

            length = math.sqrt(def_action[0]**2 + def_action[1]**2)
            if length > 1.0:
                def_action = def_action / length

            m.make_move(def_action)
            self.actions.append(def_action)

        # ── Physics ───────────────────────────────────────────────────────
        # Fixed substep count + auto-reset detection: if MuJoCo's internal
        # checks fire (NaN/huge qpos/qvel/qacc) it calls mj_resetData itself,
        # which zeroes data.time and teleports everything to the rest pose —
        # detect that (time went backwards) and terminate as unstable instead
        # of silently continuing from a corrupted state.
        simstart = self.data.time
        unstable = False
        for _ in range(self._frame_substeps):
            mj.mj_step(self.model, self.data)
            if self.data.time < simstart:
                unstable = True
                break

        # Airbag for divergence below the engine's auto-reset thresholds
        _qpos = self.data.qpos
        _qvel = self.data.qvel
        _qacc = self.data.qacc
        if (unstable or
                np.isnan(_qpos).any() or np.isinf(_qpos).any() or np.abs(_qpos).max() > 1e6 or
                np.isnan(_qvel).any() or np.isinf(_qvel).any() or np.abs(_qvel).max() > 1e6):
            mj.mj_resetData(self.model, self.data)
            return (np.zeros(self._n_obs, dtype=np.float32), -20.0, True, False,
                    self._make_info(sim_step=self.sim_step, n_at_goal=0, physics_unstable=True))
        if np.isnan(_qacc).any() or np.isinf(_qacc).any() or np.abs(_qacc).max() > 1e9:
            self.data.qvel[:] = 0.0

        self.sim_step += 1

        for m in self.movers:
            m.update_pos()
        self._update_collision_map()
        for m in self.movers:
            self._update_local_maps(m)

        curr_dists = [m.get_distance_target(norm=False) for m in self.movers]

        # ── Reward ────────────────────────────────────────────────────────
        reward = -0.01

        for prev, curr in zip(prev_dists, curr_dists):
            reward += (prev - curr) * 10.0

        for ma, mb in itertools.combinations(self.movers, 2):
            d_pair = math.dist([ma.x, ma.y], [mb.x, mb.y])
            if d_pair < self.APF_INFLUENCE:
                d_safe  = max(d_pair, 1e-3)
                penalty = 0.5 * self.APF_ETA * (1.0 / d_safe - 1.0 / self.APF_INFLUENCE) ** 2
                reward -= min(penalty, 20.0)

        # Grid penalties (ludwig_sb3): obstacles seen in the 5×5 map,
        # cables/movers seen in the 7×7 map. Movers already inside their goal
        # radius are exempt — the target formation itself dictates cable
        # proximity there (measured up to -0.86/step held at Konf 3), and
        # penalizing it would teach hovering OFF target instead of holding.
        for m, d in zip(self.movers, curr_dists):
            if d < self.goal_radius:
                continue
            reward -= self.w_obstacle_map * float(np.sum(m.mu_collision_map))
            reward -= self.w_cable_map    * float(np.sum(m.mu_cable_collision_map))

        # One-time per-mover bonus on first touch of the stage target
        for i, d in enumerate(curr_dists):
            if d < self.goal_radius and not self._reached[i]:
                reward += 10.0
                self._reached[i] = True

        # The configuration counts as reached only when ALL movers are inside
        # the goal radius at the same time — matches the chained-execution
        # hand-over criterion, so the policy must learn to hold the formation.
        terminated = all(d < self.goal_radius for d in curr_dists)
        truncated  = self.sim_step >= self._max_steps

        if terminated:
            reward += 25.0

        for m in self.movers:
            m.coords_x.append(m.x)
            m.coords_y.append(m.y)

        # if self._video_writer is not None:
        #     self._capture_frame()

        info = self._make_info(
            sim_step=self.sim_step,
            stage=self.stage,
            n_at_goal=sum(d < self.goal_radius for d in curr_dists),
            **{f"dist_to_target_{i+1}": float(curr_dists[i])
               for i in range(self.num_agents)},
        )
        return self._get_obs(), float(reward), terminated, truncated, info

    def _ensure_renderer(self) -> bool:
        """Bind the process-shared offscreen renderer. Returns True if usable."""
        global _SHARED_RENDERER
        if self.renderer is not None:
            return True
        if _SHARED_RENDERER is None:
            try:
                _SHARED_RENDERER = mj.Renderer(self.model,
                                               width=self._video_w,
                                               height=self._video_h)
            except Exception as e:
                print(f"[WireHarnessEnv] Renderer unavailable: {e}")
                return False
        self.renderer = _SHARED_RENDERER
        return True

    def render(self):
        """RGB array (H, W, 3) uint8 of the current scene.

        stable_worldmodel's World(add_pixels=True) requires render() to return
        an RGB frame, so the offscreen renderer is created lazily and we render
        regardless of render_mode. MuJoCo's Renderer.render() returns a
        (height, width, 3) uint8 array.
        """
        if not self._ensure_renderer():
            return None
        self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.opt)
        return self.renderer.render()

    # ──────────────────────────────────────────────────────────────────────
    # World-model info signals (stable_worldmodel checklist)
    # ──────────────────────────────────────────────────────────────────────

    def _state_vec(self) -> np.ndarray:
        """Compact ground-truth state: each mover's (x, y), shape (2N,)."""
        return np.array([c for m in self.movers for c in (m.x, m.y)],
                        dtype=np.float32)

    def _goal_state_vec(self) -> np.ndarray:
        """Compact goal signal: target (x, y) per mover, shape (2N,).

        Normally the stage's target configuration. During dataset eval a goal is
        injected via _set_goal_state, and we report that instead so the env's
        own goal_state matches what step() scores success against.
        """
        if self._eval_goal is not None:
            return self._eval_goal.reshape(-1).astype(np.float32)
        _, targets = self._read_variation_positions()
        return np.array([c for t in targets for c in t], dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Dataset-driven evaluation hooks (stable_worldmodel World.evaluate)
    # ──────────────────────────────────────────────────────────────────────
    # World.evaluate(dataset=...) resets each env, then calls these on the
    # unwrapped env with raw values sliced from the dataset (see the callables
    # in scripts/plan/config/wireharness.yaml). Methods that don't exist are
    # silently skipped by swm, so these MUST be present for dataset starts/goals
    # to take effect.

    def _set_state(self, state):
        """Place movers at a dataset-provided start state: (x, y) per mover, (2N,).

        Overrides the random start chosen in reset(). Reuses the reset
        teleport+settle so the cables relax to equilibrium at the new positions,
        then refreshes the maps and zeroes the per-episode step/goal bookkeeping.
        """
        pos = np.asarray(state, dtype=np.float32).reshape(self.num_agents, 2)
        self._teleport_and_settle([(float(x), float(y)) for x, y in pos])
        for m in self.movers:
            m.update_pos()
        self._update_collision_map()
        for m in self.movers:
            self._update_local_maps(m)
        self.sim_step = 0
        self._reached = [False] * self.num_agents

    def _set_goal_state(self, goal_state):
        """Point each mover's success target at a dataset goal: (x, y) per mover, (2N,).

        step() terminates (= success in dataset eval) when every mover is within
        goal_radius of its target. This redirects that check at the dataset goal
        (the state goal_offset steps ahead of the start) instead of the fixed
        stage target. Transient: a later reset() clears _eval_goal and restores
        the stage targets.
        """
        pos = np.asarray(goal_state, dtype=np.float32).reshape(self.num_agents, 2)
        self._eval_goal = pos
        for i, m in enumerate(self.movers):
            m.set_target(float(pos[i][0]), float(pos[i][1]))
        self._reached = [False] * self.num_agents

    def _render_goal_image(self):
        """RGB image of the goal configuration (movers at their stage targets).

        The cable shape is path-dependent, so it is NOT determined by the mover
        positions alone — a bare mj_forward after teleporting leaves the cables
        in their pre-teleport shape and renders them broken/stretched. Instead we
        define the goal canonically as "movers at targets, cables at physical
        equilibrium": teleport to the targets and run the same settle as reset
        (_teleport_and_settle) so the cables relax, render, then restore the
        episode-start state. Guarded by render_mode so it never runs during plain
        training (render_mode=None / MUJOCO_GL=disabled).
        """
        if self.render_mode is None or not self._ensure_renderer():
            return None
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        time = self.data.time
        _, targets = self._read_variation_positions()
        self._teleport_and_settle(targets)
        self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.opt)
        img = self.renderer.render().copy()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.time = time
        mj.mj_forward(self.model, self.data)
        return img

    def _make_info(self, **extra) -> dict:
        """Info dict carrying the world-model signals on every reset/step.

        Always includes `state` and `goal_state`; includes the visual `goal`
        image when it was rendered this episode (i.e. when rendering is on).
        """
        info = {
            "state":      self._state_vec(),
            "goal_state": self._goal_state_vec(),
            **extra,
        }
        if self._goal_image is not None:
            info["goal"] = self._goal_image
        return info

    def close(self):
        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None

    # ──────────────────────────────────────────────────────────────────────
    # Collision maps (cables + movers, ported from environment_ludwig_sb3)
    # ──────────────────────────────────────────────────────────────────────

    # Table-edge bounds for the local maps: grid cells outside the table read
    # as wall. Table: x ∈ [0, 6.72] → grid x 0..67; y ∈ [0, 3.84] → grid y 0..38.
    _GRID_X_TABLE = 67
    _GRID_Y_TABLE = 38

    def _update_local_maps(self, m):
        """
        Env-owned port of Mover.lokal_collision_map():
        - both maps use the same table-edge wall bounds (mover.py uses 38 for
          the 5×5 but 48 for the 7×7).
        5×5 map: solid obstacles near the mover (other movers, other cables, walls).
        7×7 map: wider cable-awareness window (other cables, movers, walls).
        Own body (mu_index) and own cables (cable_connect ids) are free space.
        """
        x_idx = int(round(m.x, 1) * 10)
        y_idx = int(round(m.y, 1) * 10)
        own = set(m.cable_connect)

        m.mu_collision_map = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                gy, gx = y_idx - 2 + i, x_idx - 2 + j
                if 0 <= gy <= self._GRID_Y_TABLE and 0 <= gx <= self._GRID_X_TABLE:
                    entry = self.collision_map[gy, gx]
                    if entry != 0 and entry != m.mu_index and entry not in own:
                        m.mu_collision_map[i, j] = 1
                else:
                    m.mu_collision_map[i, j] = 1

        m.mu_cable_collision_map = np.zeros((7, 7))
        for i in range(7):
            for j in range(7):
                gy, gx = y_idx - 3 + i, x_idx - 3 + j
                if 0 <= gy <= self._GRID_Y_TABLE and 0 <= gx <= self._GRID_X_TABLE:
                    entry = self.collision_map[gy, gx]
                    if entry != 0 and entry != m.mu_index and entry not in own:
                        m.mu_cable_collision_map[i, j] = 1
                else:
                    m.mu_cable_collision_map[i, j] = 1

    def _update_collision_map(self):
        self.collision_map[:]       = 0.0
        self.cable_collision_map[:] = 0.0
        self.mover_collision_map[:] = 0.0
        # Cables first: one cell per cable body, value = cable id (1..4). The
        # cable-only map keeps these even after movers overwrite collision_map.
        for bid, cable_id in self._cable_bodies:
            x, y, _ = self.data.xpos[bid]
            gx = int(round(x, 1) * 10)
            gy = int(round(y, 1) * 10)
            if 0 <= gy < self.GRID_H and 0 <= gx < self.GRID_W:
                self.collision_map[gy, gx]       = cable_id
                self.cable_collision_map[gy, gx] = cable_id
        # Movers second (3×3 safety margin, value = body id): a mover
        # overwrites its own cable's cells beneath it, so it does not see
        # itself as a permanent collision.
        for m in self.movers:
            x_idx = int(round(m.x, 1) * 10)
            y_idx = int(round(m.y, 1) * 10)
            for j in range(-1, 2):
                for k in range(-1, 2):
                    gy, gx = y_idx + j, x_idx + k
                    if 0 <= gy < self.GRID_H and 0 <= gx < self.GRID_W:
                        self.collision_map[gy, gx]       = m.mu_index
                        self.mover_collision_map[gy, gx] = m.mu_index

    def _check_cable_on_mover(self, cable_connect) -> bool:
        """Env-level port

        True if any of this mover's cables (`cable_connect`) currently runs
        across a DIFFERENT mover — i.e. cable-only cells overlap mover-only
        cells other than the two movers the cable physically connects, in more
        than one grid cell. Reads the cable/mover maps from _update_collision_map.
        """
        for cable in cable_connect:
            carriers = [m.mu_index for m in self.movers if cable in m.cable_connect]
            if not carriers:
                continue
            start, end = carriers[0], carriers[-1]
            cable_cells = (self.cable_collision_map == cable)
            mover_cells = ((self.mover_collision_map != 0)
                           & (self.mover_collision_map != start)
                           & (self.mover_collision_map != end))
            if np.count_nonzero(cable_cells & mover_cells) > 1:
                return True
        return False

    # ──────────────────────────────────────────────────────────────────────
    # Observation — fully learnable, no privileged/deterministic features
    # ──────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        # Exactly the v0_4 Environment.get_states layout: per mover i, the raw
        # signed offsets to each later mover j (i MINUS j), then i's normalized
        # target distance and angle. No grid maps (collision is constraint-handled).
        for m in self.movers:
            m.update_pos()
        obs = []
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                obs.append(self.movers[i].get_distance_x(self.movers[j].x))
                obs.append(self.movers[i].get_distance_y(self.movers[j].y))
            obs.append(self.movers[i].get_distance_target())
            obs.append(self.movers[i].get_angle_target())
        return np.array(obs, dtype=np.float32)

   