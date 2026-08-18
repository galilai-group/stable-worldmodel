"""Meta-World Sawyer tasks with visual and physical domain randomization.

Wraps a single Meta-World task and exposes a
:class:`stable_worldmodel.spaces.Dict` ``variation_space`` so the scene can be
randomized for out-of-distribution and zero-shot evaluation. The space is
assembled per task from what the loaded MuJoCo model actually contains, so a
reaching task (free object, target site) and an articulated task (door, drawer,
window) each advertise only the factors they can honor.
"""

import logging

import gymnasium as gym
import numpy as np

from stable_worldmodel import spaces as swm_spaces


try:
    import mujoco
except ImportError:
    mujoco = None

logger = logging.getLogger(__name__)

# Visual factors are randomized on every reset by default; physics and position
# overrides are opt-in so the task sampled by Meta-World stays solvable.
DEFAULT_VARIATIONS = (
    'table.color',
    'object.color',
    'background.color',
    'light.intensity',
)

# Name fragments of the Sawyer arm and its fixture. Geoms attached to these
# bodies are treated as the robot, never as the task object or the scene.
_ARM_BODY_HINTS = (
    'right_l',
    'right_hand',
    'right_wrist',
    'right_arm',
    'hand',
    'claw',
    'pad',
    'pedestal',
    'torso',
    'head',
    'screen',
    'controller',
    'itb',
    'mocap',
)

# Named geoms that are part of the static scene rather than the task object.
_SCENE_GEOMS = ('floor', 'rail')


def _id2name(model, objtype, idx):
    return mujoco.mj_id2name(model, objtype, idx)


class MetaWorldWrapper(gym.Wrapper):
    """A single Meta-World task with a stable-worldmodel variation space.

    Args:
        env_name: Meta-World task id, e.g. ``'reach-v3'`` or ``'push-v3'``.
        init_value: Optional mapping of variation keys to fixed initial values,
            applied once at construction (fixed-per-run domain randomization).
        resolution: Square pixel size that :meth:`render` resizes frames to.
        render_mode: Forwarded to Meta-World (``'rgb_array'`` for pixels).
        camera_name: Meta-World camera used for rendering.
        terminate_on_success: End the episode as terminated once Meta-World
            reports ``info['success']``, so World's success-rate eval reflects
            task completion. Meta-World otherwise only truncates at its step
            limit.
        **kwargs: Forwarded to ``gym.make('Meta-World/MT1', ...)`` (for example
            ``reward_function_version`` or ``max_episode_steps``).
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 25}

    def __init__(
        self,
        env_name,
        init_value=None,
        resolution=224,
        render_mode=None,
        camera_name='corner2',
        terminate_on_success=True,
        **kwargs,
    ):
        # Importing metaworld registers the 'Meta-World/*' gym ids.
        import metaworld  # noqa: F401

        # Skip the passive env checker on this trusted inner env so it is not
        # re-validated on every reset and step.
        env = gym.make(
            'Meta-World/MT1',
            env_name=env_name,
            render_mode=render_mode,
            camera_name=camera_name,
            disable_env_checker=True,
            **kwargs,
        )
        super().__init__(env)

        self.env_name = env_name
        self.render_size = resolution
        self.terminate_on_success = terminate_on_success
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # gym.make wraps the Sawyer env in OrderEnforcing / TimeLimit; reach
        # past them to the real env that owns the MuJoCo model and target pose.
        self._base = env.unwrapped
        self._discover_model()
        self.variation_space = swm_spaces.Dict(self._build_variation_space())
        if init_value is not None:
            self.variation_space.set_init_value(init_value)

        # MegaWrapper and the CLI read ``env.unwrapped.variation_space``, and
        # gym wrappers do not forward attribute writes, so publish the space on
        # the underlying env where ``unwrapped`` will find it.
        self._base.variation_space = self.variation_space

        self._visual_cache_ready = False

    def _discover_model(self):
        """Cache the geom/body/site handles each variation factor needs."""
        self._floor_geoms = []
        self._table_geoms = []
        self._object_geoms = []
        self._arm_geoms = []
        self._obj_body_id = -1
        self._objgeom_id = -1
        self._obj_qadr = -1
        self._goal_site_id = -1
        self._obj_size0 = None
        self._obj_mass0 = 1.0
        self._obj_friction0 = 1.0

        if mujoco is None:
            return

        model = self._base.model

        for g in range(model.ngeom):
            name = _id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            body = _id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]
            )
            material = None
            if model.geom_matid[g] >= 0:
                material = _id2name(
                    model, mujoco.mjtObj.mjOBJ_MATERIAL, model.geom_matid[g]
                )

            is_arm = body is not None and any(
                hint in body.lower() for hint in _ARM_BODY_HINTS
            )
            is_floor = name == 'floor' or material == 'basic_floor'
            is_table = material is not None and material.startswith('table')

            if is_arm:
                self._arm_geoms.append(g)
            elif is_floor:
                self._floor_geoms.append(g)
            elif is_table:
                self._table_geoms.append(g)
            elif name and name not in _SCENE_GEOMS:
                self._object_geoms.append(g)

        self._obj_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, 'obj'
        )
        self._goal_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, 'goal'
        )

        self._objgeom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, 'objGeom'
        )
        if self._objgeom_id < 0 and self._obj_body_id >= 0:
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == self._obj_body_id:
                    self._objgeom_id = g
                    break

        if self._obj_body_id >= 0:
            self._obj_mass0 = float(model.body_mass[self._obj_body_id])
            jnt_adr = model.body_jntadr[self._obj_body_id]
            jnt_num = model.body_jntnum[self._obj_body_id]
            if (
                jnt_num > 0
                and model.jnt_type[jnt_adr] == mujoco.mjtJoint.mjJNT_FREE
            ):
                self._obj_qadr = model.jnt_qposadr[jnt_adr]
        if self._objgeom_id >= 0:
            self._obj_size0 = model.geom_size[self._objgeom_id].copy()
            self._obj_friction0 = float(
                model.geom_friction[self._objgeom_id][0]
            )

    @staticmethod
    def _color_box(init_value):
        init_value = np.clip(
            np.asarray(init_value, dtype=np.float64), 0.0, 1.0
        )
        return swm_spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float64,
            init_value=init_value,
        )

    def _workspace_bounds(self):
        """Return finite (low, high) xyz bounds from the observation space."""
        low = np.asarray(self.observation_space.low[:3], dtype=np.float64)
        high = np.asarray(self.observation_space.high[:3], dtype=np.float64)
        low = np.where(np.isfinite(low), low, -1.0)
        high = np.where(np.isfinite(high), high, 1.0)
        low[2] = max(low[2], 0.0)
        return low, high

    def _build_variation_space(self):
        model = self._base.model
        low, high = self._workspace_bounds()

        space = {
            'table': swm_spaces.Dict(
                {'color': self._color_box([0.55, 0.43, 0.32])}
            ),
            'background': swm_spaces.Dict(
                {'color': self._color_box([0.3, 0.3, 0.3])}
            ),
            'light': swm_spaces.Dict(
                {
                    'intensity': swm_spaces.Box(
                        low=0.2,
                        high=1.0,
                        shape=(1,),
                        dtype=np.float64,
                        init_value=np.array([0.7]),
                    )
                }
            ),
            'rendering': swm_spaces.Dict(
                {'transparent_arm': swm_spaces.Discrete(2, init_value=0)}
            ),
        }

        obj_keys = {}
        if self._object_geoms and mujoco is not None:
            base_rgba = model.geom_rgba[self._object_geoms[0]][:3]
            obj_keys['color'] = self._color_box(base_rgba)
        if self._obj_body_id >= 0:
            obj_keys['mass'] = swm_spaces.Box(
                low=0.05,
                high=5.0,
                shape=(1,),
                dtype=np.float64,
                init_value=np.array([np.clip(self._obj_mass0, 0.05, 5.0)]),
            )
            obj_keys['friction'] = swm_spaces.Box(
                low=0.2,
                high=2.0,
                shape=(1,),
                dtype=np.float64,
                init_value=np.array([np.clip(self._obj_friction0, 0.2, 2.0)]),
            )
            obj_keys['scale'] = swm_spaces.Box(
                low=0.6,
                high=1.4,
                shape=(1,),
                dtype=np.float64,
                init_value=np.array([1.0]),
            )
            obj_keys['position'] = swm_spaces.Box(
                low=low[:2].copy(),
                high=high[:2].copy(),
                dtype=np.float64,
                init_value=self._initial_object_xy(low, high),
            )
        if obj_keys:
            space['object'] = swm_spaces.Dict(obj_keys)

        if self._goal_site_id >= 0:
            space['goal'] = swm_spaces.Dict(
                {
                    'position': swm_spaces.Box(
                        low=low.copy(),
                        high=high.copy(),
                        dtype=np.float64,
                        init_value=self._initial_goal(low, high),
                    )
                }
            )

        return space

    def _initial_object_xy(self, low, high):
        if self._obj_qadr >= 0:
            xy = np.asarray(
                self._base.data.qpos[self._obj_qadr : self._obj_qadr + 2],
                dtype=np.float64,
            )
            return np.clip(xy, low[:2], high[:2])
        return (low[:2] + high[:2]) / 2.0

    def _initial_goal(self, low, high):
        target = getattr(self._base, '_target_pos', None)
        if target is not None and np.all(np.isfinite(target)):
            return np.clip(np.asarray(target, dtype=np.float64), low, high)
        return (low + high) / 2.0

    def _active_variations(self, options):
        sampled = options.get('variation', DEFAULT_VARIATIONS)
        if 'all' in set(sampled):
            return set(self.variation_space.sampling_order)
        explicit = options.get('variation_values', {}).keys()
        return set(sampled) | set(explicit)

    def reset(self, seed=None, options=None):
        options = options or {}

        swm_spaces.reset_variation_space(
            self.variation_space,
            seed=seed,
            options=options,
            default_variations=DEFAULT_VARIATIONS,
        )
        active = self._active_variations(options)

        # Meta-World handles its own task/goal sampling; variations are layered
        # on top, so its reset options are intentionally not forwarded.
        obs, info = self.env.reset(seed=seed)

        self._apply_visual_variations(active)
        changed = self._apply_object_physics()
        changed = self._apply_positions(active) or changed
        if changed and mujoco is not None:
            mujoco.mj_forward(self._base.model, self._base.data)
            obs = self._refresh_obs(obs)

        return obs, self._make_info(obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = self._make_info(obs, info)
        # Meta-World runs to truncation and reports success in info; surface it
        # as termination so World's success-rate eval (which keys off
        # terminated) reflects task completion.
        succeeded = float(info.get('success', 0.0)) >= 1.0
        if self.terminate_on_success and succeeded:
            terminated = True
        return obs, reward, terminated, truncated, info

    def render(self):
        img = self.env.render()
        if img is None:
            return img
        # Meta-World's MuJoCo renderer returns a vertically flipped frame
        # (OpenGL's bottom-left origin); flip it back to image convention.
        img = np.ascontiguousarray(img[::-1])
        if self.render_size:
            import cv2

            img = cv2.resize(img, (self.render_size, self.render_size))
        return img

    def _refresh_obs(self, obs):
        getter = getattr(self._base, '_get_obs', None)
        if callable(getter):
            try:
                return np.asarray(getter(), dtype=np.asarray(obs).dtype)
            except Exception as exc:
                logger.warning('Meta-World obs refresh failed: %s', exc)
        return obs

    def _make_info(self, obs, info):
        obs = np.asarray(obs, dtype=np.float32)
        out = dict(info)
        out['env_name'] = self.env_name
        out['state'] = obs
        # Meta-World packs the hand xyz and gripper opening in the first 4 dims.
        out['proprio'] = obs[:4]
        target = getattr(self._base, '_target_pos', None)
        out['goal_state'] = (
            np.asarray(target, dtype=np.float32)
            if target is not None
            else obs[-3:]
        )
        out.setdefault('success', float('nan'))
        return out

    def _apply_visual_variations(self, active):
        if mujoco is None:
            return
        model = self._base.model

        needs_table = 'table.color' in active and self._table_geoms
        needs_bg = 'background.color' in active and self._floor_geoms
        needs_object = 'object.color' in active and self._object_geoms
        needs_light = 'light.intensity' in active
        needs_arm = 'rendering.transparent_arm' in active and self._arm_geoms

        if not any(
            [needs_table, needs_bg, needs_object, needs_light, needs_arm]
        ):
            return

        if not self._visual_cache_ready:
            # Unbind materials so geom_rgba edits show up immediately. Done
            # lazily so untouched resets keep the original MJCF appearance.
            for g in (
                self._table_geoms + self._floor_geoms + self._object_geoms
            ):
                model.geom_matid[g] = -1
            self._visual_cache_ready = True

        if needs_table:
            color = self.variation_space['table']['color'].value
            for g in self._table_geoms:
                model.geom_rgba[g][:3] = color
        if needs_bg:
            color = self.variation_space['background']['color'].value
            for g in self._floor_geoms:
                model.geom_rgba[g][:3] = color
        if needs_object:
            color = self.variation_space['object']['color'].value
            for g in self._object_geoms:
                model.geom_rgba[g][:3] = color
        if needs_light:
            intensity = float(
                self.variation_space['light']['intensity'].value[0]
            )
            for i in range(model.nlight):
                model.light_diffuse[i][:3] = intensity
        if needs_arm:
            transparent = (
                int(self.variation_space['rendering']['transparent_arm'].value)
                == 1
            )
            alpha = 0.25 if transparent else 1.0
            for g in self._arm_geoms:
                model.geom_rgba[g][3] = alpha

    def _apply_object_physics(self):
        """Push mass, friction, and scale to the free object every reset.

        Applied unconditionally (not only when sampled) so a fixed
        ``init_value`` sticks across resets while a freshly sampled value still
        takes effect. Defaults to a no-op when the values match the model.
        """
        if mujoco is None or self._obj_body_id < 0:
            return False
        model = self._base.model
        obj = self.variation_space['object']

        model.body_mass[self._obj_body_id] = float(obj['mass'].value[0])
        if self._objgeom_id >= 0:
            model.geom_friction[self._objgeom_id][0] = float(
                obj['friction'].value[0]
            )
            if self._obj_size0 is not None:
                model.geom_size[self._objgeom_id] = self._obj_size0 * float(
                    obj['scale'].value[0]
                )
        return True

    def _apply_positions(self, active):
        if mujoco is None:
            return False
        model, data = self._base.model, self._base.data
        changed = False

        if 'object.position' in active and self._obj_qadr >= 0:
            xy = self.variation_space['object']['position'].value
            data.qpos[self._obj_qadr : self._obj_qadr + 2] = xy
            changed = True

        if 'goal.position' in active and self._goal_site_id >= 0:
            pos = np.asarray(
                self.variation_space['goal']['position'].value,
                dtype=np.float64,
            )
            model.site_pos[self._goal_site_id] = pos
            if hasattr(self._base, '_target_pos'):
                self._base._target_pos = pos.copy()
            changed = True

        return changed
