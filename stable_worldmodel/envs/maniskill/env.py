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

import gymnasium as gym
import numpy as np
import torch
from stable_worldmodel import spaces as swm_spaces


logger = logging.getLogger(__name__)

# Sampled-at-reset variations. Empty for now — the distribution-shift knobs
# (lighting / camera / background / distractors) are a follow-up; an empty
# tuple keeps reset() a no-op over the (currently empty) variation_space.
DEFAULT_VARIATIONS = ()


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
        **kwargs,
    ):
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

        # No distribution-shift knobs yet; an empty Dict keeps EnvPool's
        # variation_space wiring happy.
        self.variation_space = swm_spaces.Dict({})

    # ----- conversion helpers (embodiment-independent) ---------------------

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

    def _extract_state(self, obs):
        """Proprio plus task ``extra`` (tcp/object poses) where present."""
        return np.concatenate(
            [self._extract_proprio(obs), self._flatten(obs.get('extra', {}))]
        ).astype(np.float32)

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
        info['state'] = self._extract_state(obs)
        get_instr = getattr(
            self.env.unwrapped, 'get_language_instruction', None
        )
        if callable(get_instr):
            instr = get_instr()
            info['instruction'] = (
                instr[0] if isinstance(instr, list) else instr
            )
        return info, info['success']

    # ----- gym interface ---------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._render_frame = self._extract_rgb(obs)
        info, _ = self._build_info(obs, info)
        return info['proprio'], info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        obs, reward, terminated, truncated, info = self.env.step(action)

        reward = float(np.asarray(self._debatch(reward)).reshape(-1)[0])
        truncated = bool(np.asarray(self._debatch(truncated)).reshape(-1)[0])
        self._render_frame = self._extract_rgb(obs)
        info, success = self._build_info(obs, info)
        # World.evaluate scores success off `terminated`, so map task success
        # onto it (taking either the native terminated flag or the detector).
        terminated = (
            bool(np.asarray(self._debatch(terminated)).reshape(-1)[0])
            or success
        )

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
