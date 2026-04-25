import logging

import gymnasium as gym
import numpy as np
from stable_worldmodel import spaces as swm_spaces

try:
    import mujoco
except ImportError:
    mujoco = None

logger = logging.getLogger(__name__)

DEFAULT_VARIATIONS = (
    'table.color',
    'object.color',
    'light.intensity',
    'background.color',
    'camera.angle_delta',
)


class FetchWrapper(gym.Wrapper):
    """Wrapper for Gymnasium Robotics Fetch environments, adding visual and
    physical domain randomization via variation_space.

    By default, observations are flattened (observation + desired_goal) into a
    single Box space. Set ``flatten=False`` to preserve the original Dict
    observation space (``observation``/``achieved_goal``/``desired_goal``),
    which is required by goal-conditioned algorithms such as SB3's
    HerReplayBuffer.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 25}

    def __init__(
        self,
        env_id,
        init_value=None,
        resolution=224,
        render_mode=None,
        flatten=True,
        **kwargs,
    ):
        env = gym.make(env_id, render_mode=render_mode, **kwargs)
        super().__init__(env)

        self.env_name = env_id
        self.render_size = resolution
        self._flatten = flatten

        # Original observation space is a Dict
        orig_obs_space = env.observation_space

        obs_dim = orig_obs_space['observation'].shape[0]
        goal_dim = orig_obs_space['desired_goal'].shape[0]

        if self._flatten:
            flat_dim = obs_dim + goal_dim
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
            )
        # else: keep env.observation_space as-is (Dict), inherited from super().__init__

        has_object = obs_dim >= 25

        # Variation space for visual domain randomization
        space_dict = {
            'table': swm_spaces.Dict(
                {
                    'color': swm_spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(3,),
                        dtype=np.float64,
                        init_value=np.array([0.3, 0.3, 0.3]),
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
            'background': swm_spaces.Dict(
                {
                    'color': swm_spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(3,),
                        dtype=np.float64,
                        init_value=np.array([0.1, 0.1, 0.1]),
                    )
                }
            ),
            'light': swm_spaces.Dict(
                {
                    'intensity': swm_spaces.Box(
                        low=0.0,
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
            'agent': swm_spaces.Dict(
                {
                    'start_position': swm_spaces.Box(
                        low=np.array([1.25, 0.6]),
                        high=np.array([1.45, 0.9]),
                        dtype=np.float64,
                        init_value=np.array([1.3418, 0.7491]),
                    )
                }
            ),
            'goal': swm_spaces.Dict(
                {
                    'start_position': swm_spaces.Box(
                        low=np.array([1.15, 0.6, 0.4247]),
                        high=np.array([1.45, 0.9, 0.4247]),
                        dtype=np.float64,
                        init_value=np.array([1.3, 0.74, 0.4247]),
                    )
                }
            ),
            'rendering': swm_spaces.Dict(
                {'transparent_arm': swm_spaces.Discrete(2, init_value=0)}
            ),
        }

        # Inject explicit physical object placements only if the target object exists
        self._object_body_id = -1
        if has_object:
            default_mass = 2.0
            if mujoco is not None:
                body_id = mujoco.mj_name2id(
                    env.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, 'object0'
                )
                if body_id >= 0:
                    self._object_body_id = body_id
                    default_mass = float(
                        env.unwrapped.model.body_mass[body_id]
                    )

            space_dict['block'] = swm_spaces.Dict(
                {
                    'start_position': swm_spaces.Box(
                        low=np.array([1.15, 0.6]),
                        high=np.array([1.45, 0.9]),
                        dtype=np.float64,
                        init_value=np.array([1.3, 0.74]),
                    ),
                    'angle': swm_spaces.Box(
                        low=-np.pi,
                        high=np.pi,
                        shape=(1,),
                        dtype=np.float64,
                        init_value=np.array([0.0]),
                    ),
                    'mass': swm_spaces.Box(
                        low=0.01,
                        high=50.0,
                        shape=(1,),
                        dtype=np.float64,
                        init_value=np.array([default_mass]),
                    ),
                }
            )

        self.variation_space = swm_spaces.Dict(space_dict)
        if init_value is not None:
            self.variation_space.set_init_value(init_value)

    def _flatten_obs(self, obs):
        return np.concatenate(
            [obs['observation'], obs['desired_goal']], axis=0
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        options = options or {}

        swm_spaces.reset_variation_space(
            self.variation_space,
            seed=seed,
            options=options,
            default_variations=DEFAULT_VARIATIONS,
        )

        sampled_keys = options.get('variation', DEFAULT_VARIATIONS)
        explicit_keys = list(options.get('variation_values', {}).keys())
        active_variations = list(sampled_keys) + explicit_keys

        if 'agent.start_position' in active_variations and hasattr(
            self.env.unwrapped, 'initial_gripper_xpos'
        ):
            agent_xy = self.variation_space['agent']['start_position'].value
            self.env.unwrapped.initial_gripper_xpos[:2] = agent_xy

        obs, info = super().reset(seed=seed, options=options)

        self._apply_visual_variations(active_variations)

        changed_physics = False
        if any(
            k in active_variations
            for k in [
                'block.start_position',
                'block.angle',
                'goal.start_position',
            ]
        ):
            self._apply_physical_variations(active_variations)
            changed_physics = True

        # Always push current mass to the model — fixed-per-run via init_value sticks,
        # and randomized-per-reset picks up the freshly sampled value.
        if self._object_body_id >= 0 and mujoco is not None:
            mass = float(self.variation_space['block']['mass'].value[0])
            self.env.unwrapped.model.body_mass[self._object_body_id] = mass

        if changed_physics and mujoco is not None:
            mujoco.mj_forward(
                self.env.unwrapped.model, self.env.unwrapped.data
            )
            obs = self.env.unwrapped._get_obs()

        flat_obs = self._flatten_obs(obs)
        info['env_name'] = self.env_name
        info['proprio'] = obs['observation']
        info['state'] = flat_obs
        info['goal_state'] = obs['desired_goal']

        return (flat_obs if self._flatten else obs), info

    def _apply_physical_variations(self, variations):
        """Manually overrides the generated MuJoCo physics states with strict variation parameters."""
        if mujoco is None:
            return
        try:
            model = self.env.unwrapped.model
            data = self.env.unwrapped.data

            if (
                'block.start_position' in variations
                or 'block.angle' in variations
            ):
                jnt_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, 'object0:joint'
                )
                if jnt_id >= 0:
                    qpos_adr = model.jnt_qposadr[jnt_id]

                    if 'block.start_position' in variations:
                        pos = self.variation_space['block'][
                            'start_position'
                        ].value
                        data.qpos[qpos_adr : qpos_adr + 2] = pos

                    if 'block.angle' in variations:
                        theta = self.variation_space['block']['angle'].value[0]
                        data.qpos[qpos_adr + 3 : qpos_adr + 7] = [
                            np.cos(theta / 2),
                            0,
                            0,
                            np.sin(theta / 2),
                        ]

            if 'goal.start_position' in variations:
                pos = self.variation_space['goal'][
                    'start_position'
                ].value.copy()
                self.env.unwrapped.goal = pos
                if hasattr(self.env.unwrapped, 'target_site_id'):
                    model.site_pos[self.env.unwrapped.target_site_id] = pos
        except Exception as e:
            logger.warning('Failed to apply physical variations: %s', e)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        flat_obs = self._flatten_obs(obs)
        info['env_name'] = self.env_name
        info['proprio'] = obs['observation']
        info['state'] = flat_obs
        info['goal_state'] = obs['desired_goal']
        return (
            (flat_obs if self._flatten else obs),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        """Returns standard render output scaled to the explicit target environment resolution."""
        img = self.env.render()
        if self.env.render_mode == 'rgb_array' and img is not None:
            import cv2

            img = cv2.resize(img, (self.render_size, self.render_size))
        return img

    def _get_geoms_for_material(self, model, mat_name):
        if mujoco is None:
            return []
        mat_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_name
        )
        if mat_id < 0:
            return []
        return [i for i in range(model.ngeom) if model.geom_matid[i] == mat_id]

    def _apply_visual_variations(self, active_variations):
        """Modifies the underlying MuJoCo model to apply visual variations.

        Only variations whose key appears in ``active_variations`` are applied —
        anything else is left untouched so the original MJCF defaults (skybox,
        floor, table, etc.) are preserved on unmodified resets.
        """
        if mujoco is None:
            return

        model = self.env.unwrapped.model
        if model is None:
            return

        active = set(active_variations)
        needs_table = 'table.color' in active
        needs_bg = 'background.color' in active
        needs_object = 'object.color' in active
        needs_light = 'light.intensity' in active
        needs_camera = 'camera.angle_delta' in active
        needs_arm = 'rendering.transparent_arm' in active

        if not any(
            [
                needs_table,
                needs_bg,
                needs_object,
                needs_light,
                needs_camera,
                needs_arm,
            ]
        ):
            return

        if not hasattr(self, '_table_geoms'):
            self._table_geoms = self._get_geoms_for_material(
                model, 'table_mat'
            )
            self._floor_geoms = self._get_geoms_for_material(
                model, 'floor_mat'
            )

            obj_geoms = self._get_geoms_for_material(model, 'block_mat')
            if not obj_geoms:
                obj_geoms = self._get_geoms_for_material(model, 'puck_mat')
            self._object_geoms = obj_geoms

            # Find and cache the skybox texture ID
            self._skybox_tex_id = -1
            for t in range(model.ntex):
                if model.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                    self._skybox_tex_id = t
                    break

            # Cache default camera pose for angle perturbation
            self._default_cam_pos = model.cam_pos.copy()
            self._default_cam_quat = model.cam_quat.copy()

            # Unbind materials so geom_rgba updates instantly without OpenGL caching bugs
            for i in (
                self._table_geoms + self._floor_geoms + self._object_geoms
            ):
                model.geom_matid[i] = -1

        if needs_table:
            table_color = self.variation_space['table']['color'].value
            for i in self._table_geoms:
                model.geom_rgba[i][:3] = table_color

        if needs_bg:
            bg_color = self.variation_space['background']['color'].value
            for i in self._floor_geoms:
                model.geom_rgba[i][:3] = bg_color

            if getattr(self, '_skybox_tex_id', -1) >= 0:
                skybox_tex_id = self._skybox_tex_id
                bg_color_uint8 = (bg_color * 255).astype(np.uint8)
                start_idx = model.tex_adr[skybox_tex_id]
                channels = model.tex_nchannel[skybox_tex_id]
                num_pixels = (
                    model.tex_width[skybox_tex_id]
                    * model.tex_height[skybox_tex_id]
                )

                if channels >= 3:
                    view = model.tex_data[
                        start_idx : start_idx + num_pixels * channels
                    ].reshape(-1, channels)
                    view[:, :3] = bg_color_uint8[:3]

                if hasattr(self.env, 'unwrapped') and hasattr(
                    self.env.unwrapped, 'mujoco_renderer'
                ):
                    renderer = self.env.unwrapped.mujoco_renderer
                    if renderer is not None and hasattr(renderer, 'viewer'):
                        viewer = renderer.viewer
                        if getattr(viewer, 'con', None) is not None:
                            mujoco.mjr_uploadTexture(
                                model, viewer.con, skybox_tex_id
                            )

        if needs_object:
            object_color = self.variation_space['object']['color'].value
            for i in self._object_geoms:
                model.geom_rgba[i][:3] = object_color

        if needs_light:
            light_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_LIGHT, 'light0'
            )
            if light_id >= 0:
                intensity = self.variation_space['light']['intensity'].value[0]
                model.light_diffuse[light_id][:3] = np.array(
                    [intensity, intensity, intensity]
                )

        if needs_camera:
            angle_delta = self.variation_space['camera']['angle_delta'].value[
                0
            ]
            for cam_id in range(model.ncam):
                model.cam_pos[cam_id] = self._default_cam_pos[cam_id]
                model.cam_quat[cam_id] = self._default_cam_quat[cam_id]

                azimuth_rad = np.radians(angle_delta[0])
                elevation_rad = np.radians(angle_delta[1])

                pos = model.cam_pos[cam_id].copy()
                cos_az, sin_az = np.cos(azimuth_rad), np.sin(azimuth_rad)
                x, y = pos[0], pos[1]
                pos[0] = x * cos_az - y * sin_az
                pos[1] = x * sin_az + y * cos_az

                cos_el, sin_el = np.cos(elevation_rad), np.sin(elevation_rad)
                z, r = pos[2], np.sqrt(pos[0] ** 2 + pos[1] ** 2)
                pos[2] = z * cos_el + r * sin_el

                model.cam_pos[cam_id] = pos

        if needs_arm:
            is_transparent = (
                self.variation_space['rendering']['transparent_arm'].value == 1
            )
            alpha_val = 0.3 if is_transparent else 1.0
            for i in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                if name and 'robot0:' in name:
                    model.geom_rgba[i][3] = alpha_val
