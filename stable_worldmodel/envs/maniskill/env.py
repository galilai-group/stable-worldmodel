"""ManiSkill3 environment wrapper for stable_worldmodel.

Wraps any ManiSkill3 (``mani_skill``) task — Franka Panda table-top
manipulation, the SIMPLER/real2sim Bridge digital twins (WidowX), and others —
as a single-env, numpy Gymnasium env compatible with ``World``/``EnvPool``.

ManiSkill envs are GPU-batched and return torch tensors with a leading
``num_envs`` dimension even at ``num_envs=1``. This wrapper instantiates one
sub-env per slot, squeezes that batch dimension, and converts everything to
numpy so the rest of stable_worldmodel sees a plain single environment.

The wrapper is intentionally robot-agnostic: nothing here branches on the
embodiment. New robots/tasks are added as data in ``tasks.py`` (``TASK_SPECS``),
not as code here.
"""

import logging
import os

import gymnasium as gym
import numpy as np
import torch
from stable_worldmodel import spaces as swm_spaces


logger = logging.getLogger(__name__)

# Visual factors resampled at every reset unless reset() is given an explicit
# 'variation' / 'variation_values' option. Aligned with SIMPLER's
# distribution-shift axes (lighting, camera pose, object appearance).
DEFAULT_VARIATIONS = (
    'light.intensity',
    'camera.angle_delta',
    'object.color',
)


def build_variation_space():
    """Build the ManiSkill variation_space (Factors of Variation).

    Visual distribution-shift knobs aligned with SIMPLER's axes, restricted to
    the ones verified to actually change the rendered frame through the wrapper:
    scene lighting, camera pose, manipulated-object color, and arm transparency
    (≈ SIMPLER's "arm texture" shift). Pure (no ``mani_skill`` dependency) so it
    can be unit-tested on CPU; the sampled values are applied to the live SAPIEN
    scene in ``ManiSkillWrapper._apply_variations``.

    Follow-ups: table/background texture (those surfaces are textured, so
    ``set_base_color`` is a no-op — needs a texture swap or ``BaseDigitalTwinEnv``
    greenscreen), and distractor objects.
    """
    return swm_spaces.Dict(
        {
            'light': swm_spaces.Dict(
                {
                    'intensity': swm_spaces.Box(
                        low=0.3,
                        high=1.0,
                        shape=(1,),
                        dtype=np.float64,
                        init_value=np.array([0.7]),
                    )
                }
            ),
            'camera': swm_spaces.Dict(
                {
                    'angle_delta': swm_spaces.Box(
                        low=-10.0,
                        high=10.0,
                        shape=(1, 2),
                        dtype=np.float64,
                        init_value=np.array([[0.0, 0.0]]),
                    )
                }
            ),
            'object': swm_spaces.Dict(
                {
                    'color': swm_spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(3,),
                        dtype=np.float64,
                        init_value=np.array([0.8, 0.1, 0.1]),
                    )
                }
            ),
            'rendering': swm_spaces.Dict(
                {'transparent_arm': swm_spaces.Discrete(2, init_value=0)}
            ),
        }
    )


class ManiSkillWrapper(gym.Wrapper):
    """Wraps a ManiSkill3 task as a single-env, numpy Gymnasium env.

    Args:
        task_id: ManiSkill task id, e.g. ``'PickCube-v1'`` or
            ``'PutCarrotOnPlateInScene-v1'``.
        robot_uids: Optional embodiment override (e.g. ``'panda'``,
            ``'widowx'``). ``None`` uses the task default.
        control_mode: Optional controller override (e.g.
            ``'pd_ee_delta_pose'`` for a uniform 7-D end-effector action).
            ``None`` uses the task default.
        obs_mode: ManiSkill observation mode. ``'rgb'`` is enough for the
            pixel pipeline + proprioception.
        camera_name: Sensor used for ``render()``. ``None`` auto-detects the
            first available sensor.
        resolution: Square size the rendered RGB frame is resized to.
        render_mode: Kept for Gymnasium compatibility; rendering is sourced
            from the sensor observation, not ManiSkill's render camera.
        sim_backend: ManiSkill sim backend (``'auto'`` by default).
        **kwargs: Forwarded verbatim to ``gym.make`` (any ManiSkill option).
    """

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}

    def __init__(
        self,
        task_id,
        robot_uids=None,
        control_mode=None,
        obs_mode='rgb',
        camera_name=None,
        resolution=224,
        render_mode='rgb_array',
        sim_backend='auto',
        goal_threshold=2.0,
        **kwargs,
    ):
        # Auto-download missing scene/robot assets on first use instead of
        # prompting (the interactive prompt raises EOFError under a headless /
        # non-interactive stdin). setdefault keeps it overridable: set
        # MS_SKIP_ASSET_DOWNLOAD_PROMPT=0 to be prompted instead.
        os.environ.setdefault('MS_SKIP_ASSET_DOWNLOAD_PROMPT', '1')

        # Lazy import: mani_skill needs a CUDA+Vulkan GPU, so it must not be
        # imported on `import stable_worldmodel`. Importing it here registers
        # the ManiSkill task ids as a side effect.
        import mani_skill.envs  # noqa: F401

        make_kwargs = {
            'num_envs': 1,
            'obs_mode': obs_mode,
            'sim_backend': sim_backend,
            'render_mode': render_mode,
        }
        if robot_uids is not None:
            make_kwargs['robot_uids'] = robot_uids
        if control_mode is not None:
            make_kwargs['control_mode'] = control_mode
        make_kwargs.update(kwargs)

        env = gym.make(task_id, **make_kwargs)
        super().__init__(env)

        self.env_name = task_id
        # render_mode is a read-only property on gym.Wrapper (proxied from the
        # wrapped env); it's passed to gym.make above rather than assigned here.
        self.render_size = resolution
        self.camera_name = camera_name
        self._camera = None
        self._render_frame = None
        self._cam_default_pose = None  # (pos, quat) cached on first FoV apply
        # Goal-conditioned eval (PushT-style): set via _set_goal_state; when set,
        # step() terminates on flat-state distance < goal_threshold.
        self._goal_state = None
        self._goal_threshold = goal_threshold

        # Expose single-env, numpy spaces (ManiSkill's own spaces are batched).
        single_action = getattr(
            env.unwrapped, 'single_action_space', env.action_space
        )
        self.action_space = self._clean_box(single_action)

        # Probe one reset to determine the proprio dimensionality and cache the
        # render camera. This runs on the GPU the env already requires.
        obs, _ = env.reset()
        proprio = self._extract_proprio(obs)
        self._render_frame = self._extract_rgb(obs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=proprio.shape, dtype=np.float32
        )

        # Factors of Variation (lighting / camera / colors / arm). Declared and
        # sampled here; applied to the live SAPIEN scene in _apply_variations.
        self.variation_space = build_variation_space()

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def _debatch(self, x):
        """Convert to numpy and drop a leading ``num_envs == 1`` dimension."""
        x = self._to_numpy(x)
        if isinstance(x, np.ndarray) and x.ndim >= 1 and x.shape[0] == 1:
            x = x[0]
        return x

    def _clean_box(self, space):
        low = np.asarray(space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(space.high, dtype=np.float32).reshape(-1)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _collect_leaves(self, node, out):
        if isinstance(node, dict):
            for v in node.values():
                self._collect_leaves(v, out)
        else:
            arr = self._debatch(node)
            if isinstance(arr, np.ndarray):
                out.append(np.asarray(arr, dtype=np.float32).ravel())

    def _flatten(self, node):
        out = []
        self._collect_leaves(node, out)
        if not out:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out).astype(np.float32)

    def _extract_proprio(self, obs):
        """Flatten ``obs['agent']`` (qpos/qvel/...) — works for any robot."""
        return self._flatten(obs.get('agent', {}))

    def _flat_state(self):
        """Full flat simulator state (ManiSkill ``get_state``) — recorded in
        info['state'] so the eval harness can restore it via ``_set_state`` and
        use it as the goal-reaching target. Falls back to a flattened
        ``get_state_dict`` if a flat ``get_state`` isn't available."""
        u = self.env.unwrapped
        get_state = getattr(u, 'get_state', None)
        if callable(get_state):
            return np.asarray(
                self._debatch(get_state()), dtype=np.float32
            ).reshape(-1)
        return self._flatten(u.get_state_dict())

    def _pick_camera(self, obs):
        if self._camera is None:
            sensors = obs.get('sensor_data', {})
            if self.camera_name is not None and self.camera_name in sensors:
                self._camera = self.camera_name
            elif sensors:
                self._camera = next(iter(sensors))
        return self._camera

    def _extract_rgb(self, obs):
        cam = self._pick_camera(obs)
        if cam is None:
            return None
        rgb = self._to_numpy(obs['sensor_data'][cam]['rgb'])
        if rgb.ndim == 4:  # (num_envs, H, W, 3)
            rgb = rgb[0]
        return rgb.astype(np.uint8)

    def _build_info(self, obs, info):
        info = {k: self._debatch(v) for k, v in info.items()}
        success = info.get('success', info.get('terminated', False))
        info['success'] = bool(np.asarray(success).reshape(-1)[0])
        info['env_name'] = self.env_name
        info['proprio'] = self._extract_proprio(obs)
        info['state'] = self._flat_state()
        get_instr = getattr(
            self.env.unwrapped, 'get_language_instruction', None
        )
        if callable(get_instr):
            instr = get_instr()
            info['instruction'] = (
                instr[0] if isinstance(instr, list) else instr
            )
        return info, info['success']

    def reset(self, seed=None, options=None):
        options = options or {}
        # Sample the factors of variation for this episode (or set explicit
        # values via options['variation'] / options['variation_values']).
        swm_spaces.reset_variation_space(
            self.variation_space,
            seed=seed,
            options=options,
            default_variations=DEFAULT_VARIATIONS,
        )
        # swm variation keys aren't ManiSkill reset options; don't forward them.
        inner = {
            k: v
            for k, v in options.items()
            if k not in ('variation', 'variation_values')
        }
        obs, info = self.env.reset(seed=seed, options=inner or None)
        obs = self._apply_variations(obs)
        self._render_frame = self._extract_rgb(obs)
        info, _ = self._build_info(obs, info)
        return info['proprio'], info

    @staticmethod
    def _iter_materials(struct, render_body_cls):
        """Yield render materials of a ManiSkill Actor/Link (grasp_cube idiom)."""
        for o in getattr(struct, '_objs', []):
            entity = getattr(o, 'entity', o)
            rbc = entity.find_component_by_type(render_body_cls)
            if rbc is None:
                continue
            for shape in rbc.render_shapes:
                for part in shape.parts:
                    yield part.material

    def _apply_camera_angle(self, u, az, el):
        """Perturb the render camera's view by (azimuth, elevation) radians,
        relative to its default pose (cached once)."""
        import sapien
        from transforms3d.euler import euler2quat
        from transforms3d.quaternions import qmult

        sensors = getattr(u, '_sensors', {}) or {}
        cam = sensors.get(self._camera) or (
            next(iter(sensors.values())) if sensors else None
        )
        rc = getattr(cam, 'camera', None) if cam is not None else None
        if rc is None:
            return
        if self._cam_default_pose is None:
            p0 = rc.get_local_pose()
            pos = self._to_numpy(p0.p).reshape(-1)[:3].copy()
            quat = self._to_numpy(p0.q).reshape(-1)[:4].copy()
            self._cam_default_pose = (pos, quat)
        pos, quat = self._cam_default_pose
        rc.set_local_pose(
            sapien.Pose(p=pos, q=qmult(euler2quat(0.0, el, az), quat))
        )

    def _apply_variations(self, obs):
        """Apply the sampled variation_space values to the live SAPIEN scene.

        Visual-only (lighting / material colors / arm alpha) so physics is
        untouched. Uses the same SAPIEN APIs as ManiSkill's reference DR env
        ``digital_twins/so100_arm/grasp_cube.py``. After mutating the scene we
        ``update_render()`` and re-fetch the observation so the returned frame
        reflects the change (our render frame is sourced from the obs).
        """
        u = self.env.unwrapped
        scene = getattr(u, 'scene', None)
        if scene is None:
            return obs
        try:
            from sapien.render import RenderBodyComponent
        except Exception:  # pragma: no cover
            return obs

        vs = self.variation_space
        changed = False

        def _val(*keys):
            node = vs
            for k in keys:
                node = node[k]
            return np.asarray(node.value).reshape(-1)

        try:
            inten = float(_val('light', 'intensity')[0])
            scene.set_ambient_light([inten, inten, inten])
            changed = True
        except Exception as e:
            logger.warning('FoV lighting failed: %s', e)

        # Manipulated-object color. The 'cube' actor is the PickCube target;
        # textured surfaces (table/ground) ignore base_color, so they're not
        # exposed as factors here (see build_variation_space docstring).
        actor = (getattr(scene, 'actors', {}) or {}).get('cube')
        if actor is not None:
            try:
                rgb = [float(c) for c in _val('object', 'color')[:3]]
                for mat in self._iter_materials(actor, RenderBodyComponent):
                    mat.set_base_color(rgb + [1.0])
                changed = True
            except Exception as e:
                logger.warning('FoV object color failed: %s', e)

        try:
            if int(_val('rendering', 'transparent_arm')[0]) == 1:
                for link in u.agent.robot.links:
                    for mat in self._iter_materials(link, RenderBodyComponent):
                        c = list(mat.base_color)
                        mat.set_base_color([c[0], c[1], c[2], 0.3])
                changed = True
        except Exception as e:
            logger.warning('FoV arm transparency failed: %s', e)

        try:
            # Always set the camera from default+delta so a prior episode's
            # perturbation is reset (delta 0 -> default pose).
            az, el = (
                np.radians(float(x)) for x in _val('camera', 'angle_delta')[:2]
            )
            self._apply_camera_angle(u, az, el)
            changed = True
        except Exception as e:
            logger.warning('FoV camera angle failed: %s', e)

        if changed:
            try:
                scene.update_render()
                obs = u.get_obs()
            except Exception as e:
                logger.warning('FoV render refresh failed: %s', e)
        return obs

    def _set_state(self, state):
        """Restore the simulator to a recorded flat state (eval start state).
        Mirrors PushT's ``_set_state`` for ManiSkill's flat get/set_state."""
        u = self.env.unwrapped
        t = torch.as_tensor(np.asarray(state, dtype=np.float32)).reshape(1, -1)
        t = t.to(getattr(u, 'device', 'cpu'))
        set_state = getattr(u, 'set_state', None)
        if callable(set_state):
            set_state(t)
        else:
            u.set_state_dict(t)
        if getattr(u, 'scene', None) is not None:
            u.scene.update_render()

    def _set_goal_state(self, goal_state):
        """Store the goal state; enables goal-reaching termination in step()."""
        self._goal_state = np.asarray(goal_state, dtype=np.float32).reshape(-1)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        obs, reward, terminated, truncated, info = self.env.step(action)

        reward = float(np.asarray(self._debatch(reward)).reshape(-1)[0])
        truncated = bool(np.asarray(self._debatch(truncated)).reshape(-1)[0])
        self._render_frame = self._extract_rgb(obs)
        info, success = self._build_info(obs, info)
        native = (
            bool(np.asarray(self._debatch(terminated)).reshape(-1)[0])
            or success
        )
        if self._goal_state is not None:
            # Goal-conditioned eval: terminate on flat-state distance (PushT-style).
            cur = info['state']
            n = min(cur.shape[0], self._goal_state.shape[0])
            dist = float(np.linalg.norm(cur[:n] - self._goal_state[:n]))
            info['goal_distance'] = dist
            terminated = dist < self._goal_threshold
        else:
            # World.evaluate scores success off `terminated`; map task success.
            terminated = native

        return info['proprio'], reward, terminated, truncated, info

    def render(self):
        if self._render_frame is None:
            return None
        img = self._render_frame
        if self.render_size is not None and img.shape[:2] != (
            self.render_size,
            self.render_size,
        ):
            import cv2

            img = cv2.resize(img, (self.render_size, self.render_size))
        return img
