import gymnasium as gym
import gymnasium_robotics
import numpy as np
import stable_worldmodel as swm
from stable_worldmodel import spaces as swm_spaces
from collections.abc import Sequence

DEFAULT_VARIATIONS = (
    "table.color",
    "object.color",
    "light.intensity",
    "background.color",
    "camera.angle_delta",
)

class FetchWrapper(gym.Wrapper):
    """Wrapper for Gymnasium Robotics Fetch environments, flattening observations
    and adding visual and physical domain randomization support via variation_space.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, env_id, init_value=None, resolution=224, render_mode=None, **kwargs):
        env = gym.make(env_id, render_mode=render_mode, **kwargs)
        super().__init__(env)
        
        self.env_name = env_id
        self.render_size = resolution
        
        # Original observation space is a Dict
        orig_obs_space = env.observation_space
        
        # flatten observation + desired_goal
        obs_dim = orig_obs_space["observation"].shape[0]
        goal_dim = orig_obs_space["desired_goal"].shape[0]
        flat_dim = obs_dim + goal_dim
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )
        
        # Determine if this environment manipulates a physical object block/puck
        has_object = obs_dim >= 25
        
        # Variation space for visual domain randomization
        space_dict = {
            "table": swm_spaces.Dict({
                "color": swm_spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float64, init_value=np.array([0.3, 0.3, 0.3]))
            }),
            "object": swm_spaces.Dict({
                "color": swm_spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float64, init_value=np.array([0.8, 0.1, 0.1]))
            }),
            "background": swm_spaces.Dict({
                "color": swm_spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float64, init_value=np.array([0.1, 0.1, 0.1]))
            }),
            "light": swm_spaces.Dict({
                "intensity": swm_spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64, init_value=np.array([0.7]))
            }),
            "camera": swm_spaces.Dict({
                "angle_delta": swm_spaces.Box(low=-10.0, high=10.0, shape=(1, 2), dtype=np.float64, init_value=np.array([[0.0, 0.0]]))
            }),
            "agent": swm_spaces.Dict({
                "start_position": swm_spaces.Box(low=np.array([1.25, 0.6]), high=np.array([1.45, 0.9]), dtype=np.float64, init_value=np.array([1.3418, 0.7491]))
            }),
            "goal": swm_spaces.Dict({
                "start_position": swm_spaces.Box(low=np.array([1.15, 0.6, 0.4247]), high=np.array([1.45, 0.9, 0.4247]), dtype=np.float64, init_value=np.array([1.3, 0.74, 0.4247]))
            })
        }
        
        # Inject explicit physical object placements only if the target object exists
        if has_object:
            space_dict["block"] = swm_spaces.Dict({
                "start_position": swm_spaces.Box(low=np.array([1.15, 0.6]), high=np.array([1.45, 0.9]), dtype=np.float64, init_value=np.array([1.3, 0.74])),
                "angle": swm_spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float64, init_value=np.array([0.0]))
            })
            
        self.variation_space = swm_spaces.Dict(space_dict)
        if init_value is not None:
            self.variation_space.set_init_value(init_value)
        
    def _flatten_obs(self, obs):
        return np.concatenate([obs["observation"], obs["desired_goal"]], axis=0).astype(np.float32)

    def reset(self, seed=None, options=None):
        # 1. Variation space PRE-RESET sampling
        self.variation_space.seed(seed)
        self.variation_space.reset()
        options = options or {}
        variations = options.get("variation", DEFAULT_VARIATIONS)
        if variations:
            self.variation_space.update(variations)
            
            # Conditionally override the environment's hardcoded initial gripper spawn cartesian position
            if "agent.start_position" in variations and hasattr(self.env.unwrapped, "initial_gripper_xpos"):
                agent_xy = self.variation_space["agent"]["start_position"].value
                self.env.unwrapped.initial_gripper_xpos[:2] = agent_xy
                
        # 2. Base Environment Reset (executes PyMuJoCo internal initialization and IK solvers for the arm)
        obs, info = super().reset(seed=seed, options=options)
        
        # 3. Apply Visual Domain Randomization (Colors, Textures, Lighting)
        self._apply_visual_variations()
        
        # 4. Apply Physical Domain Randomization (Free Joint & Goal overwrites)
        changed_physics = False
        if variations:
            if "block.start_position" in variations or "block.angle" in variations or "goal.start_position" in variations:
                self._apply_physical_variations(variations)
                changed_physics = True
                
        # 5. If we hacked the physical joints or goal, we MUST forcefully recompute the observation array
        if changed_physics:
            import mujoco
            mujoco.mj_forward(self.env.unwrapped.model, self.env.unwrapped.data)
            obs = self.env.unwrapped._get_obs()
            
        flat_obs = self._flatten_obs(obs)
        info["env_name"] = self.env_name
        info["proprio"] = obs["observation"]
        info["state"] = flat_obs
        info["goal_state"] = obs["desired_goal"]
        
        return flat_obs, info
        
    def _apply_physical_variations(self, variations):
        """Manually overrides the generated MuJoCo physics states with strict variation parameters."""
        try:
            import mujoco
            model = self.env.unwrapped.model
            data = self.env.unwrapped.data
            
            if "block.start_position" in variations or "block.angle" in variations:
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object0:joint")
                if jnt_id >= 0:
                    qpos_adr = model.jnt_qposadr[jnt_id]
                    
                    if "block.start_position" in variations:
                        # Override XY position
                        pos = self.variation_space["block"]["start_position"].value
                        data.qpos[qpos_adr : qpos_adr+2] = pos
                    
                    if "block.angle" in variations:
                        # Override Z-angle (converted to Quaternion)
                        theta = self.variation_space["block"]["angle"].value[0]
                        data.qpos[qpos_adr+3 : qpos_adr+7] = [np.cos(theta/2), 0, 0, np.sin(theta/2)]
                        
            if "goal.start_position" in variations:
                pos = self.variation_space["goal"]["start_position"].value.copy()
                self.env.unwrapped.goal = pos
                if hasattr(self.env.unwrapped, "target_site_id"):
                    model.site_pos[self.env.unwrapped.target_site_id] = pos
        except Exception:
            pass
        
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        flat_obs = self._flatten_obs(obs)
        info["env_name"] = self.env_name
        info["proprio"] = obs["observation"]
        info["state"] = flat_obs
        info["goal_state"] = obs["desired_goal"]
        return flat_obs, reward, terminated, truncated, info
        
    def render(self):
        """Returns standard render output scaled to the explicit target environment resolution."""
        img = self.env.render()
        if self.env.render_mode == "rgb_array" and img is not None:
            import cv2
            img = cv2.resize(img, (self.render_size, self.render_size))
        return img
        
    def _get_geoms_for_material(self, model, mat_name):
        try:
            import mujoco
        except ImportError:
            return []
        mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_name)
        if mat_id < 0:
            return []
        return [i for i in range(model.ngeom) if model.geom_matid[i] == mat_id]

    def _apply_visual_variations(self):
        """Modifies the underlying MuJoCo model to apply visual variations."""
        try:
            import mujoco
        except ImportError:
            return
            
        model = self.env.unwrapped.model
        if not model:
            return
            
        if not hasattr(self, "_table_geoms"):
            self._table_geoms = self._get_geoms_for_material(model, "table_mat")
            self._floor_geoms = self._get_geoms_for_material(model, "floor_mat")
            
            obj_geoms = self._get_geoms_for_material(model, "block_mat")
            if not obj_geoms:
                obj_geoms = self._get_geoms_for_material(model, "puck_mat")
            self._object_geoms = obj_geoms
            
            # Find and cache the skybox texture ID
            import mujoco
            self._skybox_tex_id = -1
            for t in range(model.ntex):
                if model.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                    self._skybox_tex_id = t
                    break
            
            # Unbind materials so geom_rgba updates instantly without OpenGL caching bugs
            for i in self._table_geoms + self._floor_geoms + self._object_geoms:
                model.geom_matid[i] = -1
                
        # Now we can safely modify geom_rgba
        table_color = self.variation_space["table"]["color"].value
        for i in self._table_geoms:
            model.geom_rgba[i][:3] = table_color
            
        bg_color = self.variation_space["background"]["color"].value
        for i in self._floor_geoms:
            model.geom_rgba[i][:3] = bg_color
            
        # Modify skybox or background gradient
        if getattr(self, "_skybox_tex_id", -1) >= 0:
            import mujoco
            skybox_tex_id = self._skybox_tex_id
            # Overwrite the skybox pixel data with the new background color
            bg_color_uint8 = (bg_color * 255).astype(np.uint8)
            start_idx = model.tex_adr[skybox_tex_id]
            channels = model.tex_nchannel[skybox_tex_id]
            num_pixels = model.tex_width[skybox_tex_id] * model.tex_height[skybox_tex_id]
            
            # Fill the texture array slice with the solid new background color
            if channels >= 3:
                # PyMuJoCo tex_data is a flat array. We reshape it, fill it, and flatten it back.
                view = model.tex_data[start_idx : start_idx + num_pixels * channels].reshape(-1, channels)
                view[:, :3] = bg_color_uint8[:3]
            
            # If OpenGL context is already initialized, manually upload the texture dynamically
            # We strictly check the renderer state since Gymnasium initializes the Viewer lazily
            if hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "mujoco_renderer"):
                renderer = self.env.unwrapped.mujoco_renderer
                if renderer is not None and hasattr(renderer, "viewer"):
                    viewer = renderer.viewer
                    # Only upload if the viewer has a valid PyMuJoCo rendering context (con)
                    if getattr(viewer, "con", None) is not None:
                        mujoco.mjr_uploadTexture(model, viewer.con, skybox_tex_id)
            
        object_color = self.variation_space["object"]["color"].value
        for i in self._object_geoms:
            model.geom_rgba[i][:3] = object_color
            
        # Apply light intensity
        light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "light0")
        if light_id >= 0:
            intensity = self.variation_space["light"]["intensity"].value[0]
            model.light_diffuse[light_id][:3] = np.array([intensity, intensity, intensity])
