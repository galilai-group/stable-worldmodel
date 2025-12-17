import mujoco
import numpy as np
from dm_control import mjcf
from ogbench.manipspace import lie
from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv

import stable_worldmodel as swm
from stable_worldmodel.envs.utils import perturb_camera_angle


class SceneEnv(ManipSpaceEnv):
    """Scene environment.

    This environment consists of a cube, two buttons, a drawer, and a window. The goal is to manipulate the objects
    to a target configuration. The buttons toggle the lock state of the drawer and window.

    In addition to `qpos` and `qvel`, it maintains the following state variables.
    - `button_states`: A binary array of size `num_buttons` representing the state of each button. Stored in
        `_cur_button_states`.
    """

    def __init__(self, env_type="scene", ob_type="pixels", permute_blocks=True, multiview=False, *args, **kwargs):
        """Initialize the Scene environment.

        Args:
            env_type: Unused; defined for compatibility with the other environments.
            permute_blocks: Whether to randomly permute the order of the blocks at task initialization.
            *args: Additional arguments to pass to the parent class.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        self._env_type = env_type
        self._permute_blocks = permute_blocks
        self._multiview = multiview

        super().__init__(*args, **kwargs)

        self._ob_type = ob_type

        # Adjust workspace bounds to a smaller region.
        self._arm_sampling_bounds = np.asarray([[0.25, -0.2, 0.20], [0.6, 0.2, 0.35]])
        self._object_sampling_bounds = np.asarray([[0.3, -0.07], [0.45, 0.18]])
        self._target_sampling_bounds = self._object_sampling_bounds

        # Define constants.
        self._drawer_center = np.array([0.33, -0.24, 0.066])
        self._cube_colors = np.array([self._colors["red"], self._colors["blue"]])
        self._cube_success_colors = np.array([self._colors["lightred"], self._colors["lightblue"]])
        self._num_cubes = 1
        self._num_buttons = 2
        self._num_button_states = 2
        self._cur_button_states = np.array([0] * self._num_buttons)

        # Target info.
        self._target_task = "cube"
        # The target cube position is stored in the mocap object.
        self._target_block = 0
        self._target_button = 0
        self._target_button_states = np.array([0] * self._num_buttons)
        self._target_drawer_pos = 0.0
        self._target_window_pos = 0.0

        self.variation_space = swm.spaces.Dict(
            {
                "cube": swm.spaces.Dict(
                    {
                        # "num": swm.spaces.Discrete(),
                        "color": swm.spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(self._num_cubes, 3),
                            dtype=np.float64,
                            init_value=self._cube_colors[: self._num_cubes, :3].copy(),
                        ),
                        "size": swm.spaces.Box(
                            low=0.01,
                            high=0.03,
                            shape=(self._num_cubes,),
                            dtype=np.float64,
                            init_value=0.02 * np.ones((self._num_cubes,), dtype=np.float32),
                        ),
                    }
                    # sampling_order=["num", "color", "size"]
                ),
                "lock_color": swm.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(2, 3),
                    dtype=np.float64,
                    init_value=np.array([self._colors["red"][:3], self._colors["red"][:3]]),
                ),
                "agent": swm.spaces.Dict(
                    {
                        "color": swm.spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(3,),
                            dtype=np.float64,
                            init_value=self._colors["purple"][:3],
                        ),
                    }
                ),
                "floor": swm.spaces.Dict(
                    {
                        "color": swm.spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(2, 3),
                            dtype=np.float64,
                            init_value=np.array([[0.08, 0.11, 0.16], [0.15, 0.18, 0.25]]),
                        ),
                    }
                ),
                "camera": swm.spaces.Dict(
                    {
                        "angle_delta": swm.spaces.Box(
                            low=-5.0,
                            high=5.0,
                            shape=(2, 2) if self._multiview else (1, 2),
                            dtype=np.float64,
                            init_value=np.zeros([2, 2]) if self._multiview else np.zeros([1, 2]),
                        ),
                    }
                ),
                "light": swm.spaces.Dict(
                    {
                        "intensity": swm.spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(1,),
                            dtype=np.float64,
                            init_value=[0.6],
                        ),
                    }
                ),
            }
        )

    def set_state(self, qpos, qvel, button_states):
        self._cur_button_states = button_states.copy()
        self._apply_button_states()
        super().set_state(qpos, qvel)

    def set_tasks(self):
        self.task_infos = [
            {
                "task_name": "task1_open",
                "init": {
                    "block_xyzs": np.array([[0.35, 0.05, 0.02]]),
                    "button_states": np.array([1, 1]),
                    "drawer_pos": 0.0,
                    "window_pos": 0.0,
                },
                "goal": {
                    "block_xyzs": np.array([[0.35, 0.05, 0.02]]),
                    "button_states": np.array([1, 1]),
                    "drawer_pos": -0.16,
                    "window_pos": 0.2,
                },
            },
            {
                "task_name": "task2_unlock_and_lock",
                "init": {
                    "block_xyzs": np.array([[0.35, -0.05, 0.02]]),
                    "button_states": np.array([0, 0]),
                    "drawer_pos": -0.16,
                    "window_pos": 0.2,
                },
                "goal": {
                    "block_xyzs": np.array([[0.35, -0.05, 0.02]]),
                    "button_states": np.array([0, 0]),
                    "drawer_pos": 0.0,
                    "window_pos": 0.0,
                },
            },
            {
                "task_name": "task3_rearrange_medium",
                "init": {
                    "block_xyzs": np.array([[0.4, -0.05, 0.02]]),
                    "button_states": np.array([1, 0]),
                    "drawer_pos": 0.0,
                    "window_pos": 0.2,
                },
                "goal": {
                    "block_xyzs": np.array([[0.4, 0.15, 0.02]]),
                    "button_states": np.array([1, 1]),
                    "drawer_pos": -0.16,
                    "window_pos": 0.0,
                },
            },
            {
                "task_name": "task4_put_in_drawer",
                "init": {
                    "block_xyzs": np.array([[0.35, 0.05, 0.02]]),
                    "button_states": np.array([0, 0]),
                    "drawer_pos": 0.0,
                    "window_pos": 0.0,
                },
                "goal": {
                    "block_xyzs": np.array([[0.33, -0.356, 0.065986]]),
                    "button_states": np.array([1, 0]),
                    "drawer_pos": 0.0,
                    "window_pos": 0.0,
                },
            },
            {
                "task_name": "task5_rearrange_hard",
                "init": {
                    "block_xyzs": np.array([[0.35, 0.15, 0.02]]),
                    "button_states": np.array([0, 0]),
                    "drawer_pos": 0.0,
                    "window_pos": 0.0,
                },
                "goal": {
                    "block_xyzs": np.array([[0.33, -0.356, 0.065986]]),
                    "button_states": np.array([0, 0]),
                    "drawer_pos": 0.0,
                    "window_pos": 0.2,
                },
            },
        ]

        if self._reward_task_id == 0:
            self._reward_task_id = 2  # Default task.

    def reset(self, options=None, *args, **kwargs):
        options = options or {}

        self.variation_options = options.get("variation", {})

        self.variation_space.reset()

        if "variation" in options:
            assert isinstance(options["variation"], list | tuple), (
                "variation option must be a list or tuple containing variation names to sample"
            )

            if len(options["variation"]) == 1 and options["variation"][0] == "all":
                self.variation_space.sample()

            else:
                self.variation_space.update(set(options["variation"]))

        assert self.variation_space.check(debug=True), "Variation values must be within variation space!"

        return super().reset(options, *args, **kwargs)

    def add_objects(self, arena_mjcf):
        # Add objects to scene.
        cube_mjcf = mjcf.from_path((self._desc_dir / "cube.xml").as_posix())
        arena_mjcf.include_copy(cube_mjcf)
        button_mjcf = mjcf.from_path((self._desc_dir / "buttons.xml").as_posix())
        arena_mjcf.include_copy(button_mjcf)
        drawer_mjcf = mjcf.from_path((self._desc_dir / "drawer.xml").as_posix())
        arena_mjcf.include_copy(drawer_mjcf)
        window_mjcf = mjcf.from_path((self._desc_dir / "window.xml").as_posix())
        arena_mjcf.include_copy(window_mjcf)

        # Save geoms.
        self._cube_geoms_list = []
        for i in range(self._num_cubes):
            self._cube_geoms_list.append(cube_mjcf.find("body", f"object_{i}").find_all("geom"))
        self._cube_target_geoms_list = []
        for i in range(self._num_cubes):
            self._cube_target_geoms_list.append(cube_mjcf.find("body", f"object_target_{i}").find_all("geom"))
        self._button_geoms_list = []
        for i in range(self._num_buttons):
            self._button_geoms_list.append([button_mjcf.find("geom", f"btngeom_{i}")])

        # Add cameras.
        self.cameras = {
            "front": {
                "pos": (1.139, 0.000, 0.821),
                "xyaxes": (0.000, 1.000, 0.000, -0.627, 0.000, 0.779),
            },
            "front_pixels": {
                "pos": (0.905, 0.000, 0.762),
                "xyaxes": (0.000, 1.000, 0.000, -0.771, 0.000, 0.637),
            },
            "side_pixels": {
                "pos": (0.400, -0.505, 0.762),
                "xyaxes": (1.000, 0.000, 0.000, 0.00, 0.771, 0.637),
            },
        }
        for camera_name, camera_kwargs in self.cameras.items():
            arena_mjcf.worldbody.add("camera", name=camera_name, **camera_kwargs)

    def post_compilation_objects(self):
        # Cube geom IDs.
        self._cube_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in cube_geoms] for cube_geoms in self._cube_geoms_list
        ]
        self._cube_target_mocap_ids = [
            self._model.body(f"object_target_{i}").mocapid[0] for i in range(self._num_cubes)
        ]
        self._cube_target_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in cube_target_geoms]
            for cube_target_geoms in self._cube_target_geoms_list
        ]

        # Button geom IDs.
        self._button_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in button_geoms]
            for button_geoms in self._button_geoms_list
        ]
        self._button_site_ids = [self._model.site(f"btntop_{i}").id for i in range(self._num_buttons)]

        # Drawer and window site IDs.
        self._drawer_site_id = self._model.site("drawer_handle_center").id
        self._drawer_target_site_id = self._model.site("drawer_handle_center_target").id

        self._window_site_id = self._model.site("window_handle_center").id
        self._window_target_site_id = self._model.site("window_handle_center_target").id

    def _apply_button_states(self):
        # Adjust button colors based on the current state.
        for i in range(self._num_buttons):
            for gid in self._button_geom_ids_list[i]:
                self._model.geom(gid).rgba[:3] = (
                    self.variation_space["lock_color"].value[i]
                    if self._cur_button_states[i] == 0
                    else self._colors["white"][:3]
                )
                self._model.geom(gid).rgba[3] = 1.0

        # Lock or unlock the drawer and window based on the button states.
        # We adjust the damping of the joints to lock the drawer and window. This needs to be set carefully because
        # setting it to a very high value can cause numerical instability. We use 1e6. This is a reasonably safe value,
        # but it still allows the drawer and window to move very slightly with a strong enough force. We also tested
        # 1e7, but it caused numerical instability when interacting with the cube.
        if self._cur_button_states[0] == 0:
            # Set the damping to a high value to lock the drawer.
            self._model.joint("drawer_slide").damping[0] = 1e6
            self._model.material("drawer_handle").rgba[:3] = self.variation_space["lock_color"].value[0]
            self._model.material("drawer_handle").rgba[3] = 1.0
        else:
            # Unset the damping to unlock the drawer.
            self._model.joint("drawer_slide").damping[0] = 2.0
            self._model.material("drawer_handle").rgba = self._colors["white"]
        if self._cur_button_states[1] == 0:
            # Set the damping to a high value to lock the window.
            self._model.joint("window_slide").damping[0] = 1e6
            self._model.material("window_handle").rgba[:3] = self.variation_space["lock_color"].value[1]
            self._model.material("window_handle").rgba[3] = 1.0
        else:
            # Unset the damping to unlock the window.
            self._model.joint("window_slide").damping[0] = 2.0
            self._model.material("window_handle").rgba = self._colors["white"]

        mujoco.mj_forward(self._model, self._data)

    def modify_mjcf_model(self, mjcf_model):
        if "all" in self.variation_options or "floor.color" in self.variation_options:
            # Modify floor color
            grid_texture = mjcf_model.find("texture", "grid")
            grid_texture.rgb1 = self.variation_space["floor"]["color"].value[0]
            grid_texture.rgb2 = self.variation_space["floor"]["color"].value[1]

        if "all" in self.variation_options or "agent.color" in self.variation_options:
            # Modify arm color
            mjcf_model.find("material", "ur5e/robotiq/black").rgba[:3] = self.variation_space["agent"]["color"].value
            mjcf_model.find("material", "ur5e/robotiq/pad_gray").rgba[:3] = self.variation_space["agent"][
                "color"
            ].value

        if "all" in self.variation_options or "cube.size" in self.variation_options:
            # Modify cube size based on variation space
            for i in range(self._num_cubes):
                # Regular cubes
                body = mjcf_model.find("body", f"object_{i}")
                if body:
                    for geom in body.find_all("geom"):
                        geom.size = self.variation_space["cube"]["size"].value[i] * np.ones(
                            (3), dtype=np.float32
                        )  # half-extents (x, y, z)

                # Target cubes (if any)
                target_body = mjcf_model.find("body", f"object_target_{i}")
                if target_body:
                    for geom in target_body.find_all("geom"):
                        geom.size = self.variation_space["cube"]["size"].value[i] * np.ones((3), dtype=np.float32)

            self.mark_dirty()

        if "all" in self.variation_options or "camera.angle_delta" in self.variation_options:
            # Perturb camera angle
            cameras_to_vary = ["front_pixels", "side_pixels"] if self._multiview else ["front_pixels"]
            for i, cam_name in enumerate(cameras_to_vary):
                cam = mjcf_model.find("camera", cam_name)
                cam.xyaxes = perturb_camera_angle(
                    self.cameras[cam_name]["xyaxes"], self.variation_space["camera"]["angle_delta"].value[i]
                )

        if "all" in self.variation_options or "light.intensity" in self.variation_options:
            # Modify light intensity
            light = mjcf_model.find("light", "global")
            light.diffuse = self.variation_space["light"]["intensity"].value[0] * np.ones((3), dtype=np.float32)
            self.mark_dirty()

        return mjcf_model

    def initialize_episode(self):
        # Set cube colors.
        for i in range(self._num_cubes):
            for gid in self._cube_geom_ids_list[i]:
                self._model.geom(gid).rgba[:3] = self.variation_space["cube"]["color"].value[i]
                self._model.geom(gid).rgba[3] = 1.0
            for gid in self._cube_target_geom_ids_list[i]:
                self._model.geom(gid).rgba[:3] = self.variation_space["cube"]["color"].value[i]

        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)

        if self._mode == "data_collection":
            # Randomize the scene.

            self.initialize_arm()

            # Randomize block positions and orientations.
            for i in range(self._num_cubes):
                xy = self.np_random.uniform(*self._object_sampling_bounds)
                obj_pos = (*xy, 0.02)
                yaw = self.np_random.uniform(0, 2 * np.pi)
                obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                self._data.joint(f"object_joint_{i}").qpos[:3] = obj_pos
                self._data.joint(f"object_joint_{i}").qpos[3:] = obj_ori

            # Randomize button states.
            for i in range(self._num_buttons):
                self._cur_button_states[i] = self.np_random.choice(self._num_button_states)
            self._apply_button_states()

            # Randomize drawer and window positions.
            self._data.joint("drawer_slide").qpos[0] = self.np_random.uniform(-0.16, 0)
            self._data.joint("window_slide").qpos[0] = self.np_random.uniform(0, 0.2)

            # Set a new target.
            self.set_new_target(return_info=False)

            # NOTE: Goal observation is not used in data collection mode.
            self._cur_goal_ob = np.zeros_like(self.compute_observation())
        else:
            # Set object positions and orientations based on the current task.

            if self._permute_blocks:
                # Randomize the order of the cubes when there are multiple cubes.
                block_permutation = self.np_random.permutation(self._num_cubes)
            else:
                block_permutation = np.arange(self._num_cubes)
            init_block_xyzs = self.cur_task_info["init"]["block_xyzs"].copy()[block_permutation]
            goal_block_xyzs = self.cur_task_info["goal"]["block_xyzs"].copy()[block_permutation]
            # Get the current task info for the other objects.
            init_button_states = self.cur_task_info["init"]["button_states"].copy()
            goal_button_states = self.cur_task_info["goal"]["button_states"].copy()
            init_drawer_pos = self.cur_task_info["init"]["drawer_pos"]
            goal_drawer_pos = self.cur_task_info["goal"]["drawer_pos"]
            init_window_pos = self.cur_task_info["init"]["window_pos"]
            goal_window_pos = self.cur_task_info["goal"]["window_pos"]

            # First, force set the current scene to the goal state to obtain the goal observation.
            saved_qpos = self._data.qpos.copy()
            saved_qvel = self._data.qvel.copy()
            self.initialize_arm()
            for i in range(self._num_cubes):
                self._data.joint(f"object_joint_{i}").qpos[:3] = goal_block_xyzs[i]
                self._data.joint(f"object_joint_{i}").qpos[3:] = lie.SO3.identity().wxyz.tolist()
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = goal_block_xyzs[i]
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()
            self._cur_button_states = goal_button_states.copy()
            self._apply_button_states()
            self._data.joint("drawer_slide").qpos[0] = goal_drawer_pos
            self._data.joint("window_slide").qpos[0] = goal_window_pos
            mujoco.mj_forward(self._model, self._data)

            # Do a few random steps to make the scene stable.
            for _ in range(2):
                self.step(self.action_space.sample())

            # Save the goal observation.
            self._cur_goal_ob = (
                self.compute_oracle_observation() if self._use_oracle_rep else self.compute_observation()
            )
            if self._render_goal:
                self._cur_goal_rendered = self.render()
            else:
                self._cur_goal_rendered = None

            # Now, do the actual reset.
            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            self.initialize_arm()
            for i in range(self._num_cubes):
                # Randomize the position and orientation of the cube slightly.
                obj_pos = init_block_xyzs[i].copy()
                obj_pos[:2] += self.np_random.uniform(-0.01, 0.01, size=2)
                self._data.joint(f"object_joint_{i}").qpos[:3] = obj_pos
                yaw = self.np_random.uniform(0, 2 * np.pi)
                obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                self._data.joint(f"object_joint_{i}").qpos[3:] = obj_ori
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = goal_block_xyzs[i]
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()
            # Set the button states.
            self._cur_button_states = init_button_states.copy()
            self._target_button_states = goal_button_states.copy()
            self._apply_button_states()
            # Randomize the drawer and window positions slightly.
            self._data.joint("drawer_slide").qpos[0] = np.clip(
                init_drawer_pos + self.np_random.uniform(-0.01, 0.01), -0.16, 0
            )
            self._model.site("drawer_handle_center_target").pos[1] = goal_drawer_pos
            self._target_drawer_pos = goal_drawer_pos
            self._data.joint("window_slide").qpos[0] = np.clip(
                init_window_pos + self.np_random.uniform(-0.01, 0.01), 0, 0.2
            )
            self._model.site("window_handle_center_target").pos[0] = goal_window_pos
            self._target_window_pos = goal_window_pos

        # Forward kinematics to update site positions.
        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()

        self._success = False

    def _is_in_drawer(self, obj_pos):
        """Check if the object is in the drawer."""
        drawer_pos_y = self._data.site_xpos[self._drawer_site_id][1]
        drawer_low = np.array([0.21, drawer_pos_y - 0.27, 0.0])
        drawer_high = np.array([0.45, drawer_pos_y - 0.07, 0.15])
        return np.all(drawer_low <= obj_pos) and np.all(obj_pos <= drawer_high)

    def set_new_target(self, return_info=True, p_stack=0.5):
        """Set a new random target for data collection.

        Args:
            return_info: Whether to return the observation and reset info.
            p_stack: Probability of stacking the target block on top of another block when there are multiple blocks
                and the target task is 'cube'.
        """
        assert self._mode == "data_collection"

        # Only consider blocks not in the drawer.
        available_blocks = []
        for i in range(self._num_cubes):
            if not self._is_in_drawer(self._data.joint(f"object_joint_{i}").qpos[:3]):
                available_blocks.append(i)

        # Probability of each task.
        p_cube = 1.0 if len(available_blocks) > 0 else 0.0
        p_button = 1.0
        p_drawer = 0.25 if self._cur_button_states[0] == 0 else 1.0
        p_window = 0.25 if self._cur_button_states[1] == 0 else 1.0
        probs = np.array([p_cube, p_button, p_drawer, p_window])
        probs /= probs.sum()

        # Probability of putting the target block in the drawer when the target task is 'cube'.
        p_put_in_drawer = 0.3

        self._target_task = self.np_random.choice(["cube", "button", "drawer", "window"], p=probs)

        if self._target_task == "cube":
            # Set cube target.
            block_xyzs = np.array([self._data.joint(f"object_joint_{i}").qpos[:3] for i in range(self._num_cubes)])

            # Compute the top blocks.
            top_blocks = []
            for i in range(self._num_cubes):
                if i not in available_blocks:
                    continue
                for j in range(self._num_cubes):
                    if i == j:
                        continue
                    if (
                        block_xyzs[j][2] > block_xyzs[i][2]
                        and np.linalg.norm(block_xyzs[i][:2] - block_xyzs[j][:2]) < 0.02
                    ):
                        break
                else:
                    top_blocks.append(i)

            # Pick one of the top cubes as the target.
            self._target_block = self.np_random.choice(top_blocks)

            put_in_drawer = (
                self._data.joint("drawer_slide").qpos[0] < -0.12 and self.np_random.uniform() < p_put_in_drawer
            )
            stack = len(top_blocks) >= 2 and self.np_random.uniform() < p_stack
            if put_in_drawer:
                # Put the target block in the drawer.
                tar_pos = self._drawer_center.copy()
                tar_pos[:2] = tar_pos[:2] + self.np_random.uniform(-0.005, 0.005, size=2)
            elif stack:
                # Stack the target block on top of another block.
                block_idx = self.np_random.choice(list(set(top_blocks) - {self._target_block}))
                block_pos = self._data.joint(f"object_joint_{block_idx}").qpos[:3]
                tar_pos = np.array([block_pos[0], block_pos[1], block_pos[2] + 0.04])
            else:
                # Randomize target position.
                xy = self.np_random.uniform(*self._target_sampling_bounds)
                tar_pos = (*xy, 0.02)
            # Randomize target orientation.
            yaw = self.np_random.uniform(0, 2 * np.pi)
            tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()

            # Only show the target block.
            for i in range(self._num_cubes):
                if i == self._target_block:
                    # Set the target position and orientation.
                    self._data.mocap_pos[self._cube_target_mocap_ids[i]] = tar_pos
                    self._data.mocap_quat[self._cube_target_mocap_ids[i]] = tar_ori
                else:
                    # Move the non-target blocks out of the way.
                    self._data.mocap_pos[self._cube_target_mocap_ids[i]] = (0, 0, -0.3)
                    self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()

            # Set the target colors.
            for i in range(self._num_cubes):
                if self._visualize_info and i == self._target_block:
                    for gid in self._cube_target_geom_ids_list[i]:
                        self._model.geom(gid).rgba[3] = 0.2
                else:
                    for gid in self._cube_target_geom_ids_list[i]:
                        self._model.geom(gid).rgba[3] = 0.0
        elif self._target_task == "button":
            # Set target button.
            self._target_button = self.np_random.choice(self._num_buttons)
            self._target_button_states[self._target_button] = (
                self._cur_button_states[self._target_button] + 1
            ) % self._num_button_states
        elif self._target_task == "drawer":
            # Set target drawer position.
            if self._data.joint("drawer_slide").qpos[0] >= -0.08:  # Drawer closed.
                self._target_drawer_pos = -0.16
            else:  # Drawer open.
                self._target_drawer_pos = 0.0
            self._model.site("drawer_handle_center_target").pos[1] = self._target_drawer_pos
        elif self._target_task == "window":
            # Set target window position.
            if self._data.joint("window_slide").qpos[0] <= 0.1:  # Window closed.
                self._target_window_pos = 0.2
            else:  # Window open.
                self._target_window_pos = 0.0
            self._model.site("window_handle_center_target").pos[0] = self._target_window_pos

        mujoco.mj_kinematics(self._model, self._data)

        if return_info:
            return self.compute_observation(), self.get_reset_info()

    def pre_step(self):
        self._prev_button_states = self._cur_button_states.copy()
        super().pre_step()

    def _compute_successes(self):
        """Compute object successes."""
        cube_successes = []
        for i in range(self._num_cubes):
            obj_pos = self._data.joint(f"object_joint_{i}").qpos[:3]
            tar_pos = self._data.mocap_pos[self._cube_target_mocap_ids[i]]
            if np.linalg.norm(obj_pos - tar_pos) <= 0.04:
                cube_successes.append(True)
            else:
                cube_successes.append(False)
        button_successes = [
            (self._cur_button_states[i] == self._target_button_states[i]) for i in range(self._num_buttons)
        ]
        drawer_success = np.abs(self._data.joint("drawer_slide").qpos[0] - self._target_drawer_pos) <= 0.04
        window_success = np.abs(self._data.joint("window_slide").qpos[0] - self._target_window_pos) <= 0.04

        return cube_successes, button_successes, drawer_success, window_success

    def post_step(self):
        # Check numerical stability.
        if self._mode == "task":
            # Very rarely, the blocks can go out of bounds due to numerical instability. This can (rarely) happen
            # when the robot presses the drawer lock button while the drawer is moving and the block is in the drawer.
            # We only check this in the task mode, because we will manually filter out these cases outside the class in
            # the data collection mode with a more stringent check.
            is_healthy = True
            for i in range(self._num_cubes):
                obj_pos = self._data.joint(f"object_joint_{i}").qpos[:3]
                # Check if the object is out of bounds.
                if np.any(obj_pos <= self._workspace_bounds[0] - 0.2) or np.any(
                    obj_pos >= self._workspace_bounds[1] + 0.2
                ):
                    is_healthy = False
                    break

            if not is_healthy:
                # Manually reset the cube position to a random initial position.
                print("Numerical instability detected. Resetting cube positions.", flush=True)
                for i in range(self._num_cubes):
                    xy = self.np_random.uniform(*self._object_sampling_bounds)
                    obj_pos = (*xy, 0.02)
                    yaw = self.np_random.uniform(0, 2 * np.pi)
                    obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                    self._data.joint(f"object_joint_{i}").qpos[:3] = obj_pos
                    self._data.joint(f"object_joint_{i}").qpos[3:] = obj_ori
                    self._data.joint("object_joint_0").qvel[:] = 0.0
                mujoco.mj_forward(self._model, self._data)

        # Update button states.
        for i in range(self._num_buttons):
            prev_joint_pos = self._prev_ob_info[f"privileged/button_{i}_pos"][0]
            cur_joint_pos = self._data.joint(f"buttonbox_joint_{i}").qpos.copy()[0]
            if prev_joint_pos > -0.02 and cur_joint_pos <= -0.02:
                # Button pressed: change the state of the button.
                self._cur_button_states[i] = (self._cur_button_states[i] + 1) % self._num_button_states
        self._apply_button_states()

        # Evaluate successes.
        cube_successes, button_successes, drawer_success, window_success = self._compute_successes()
        if self._mode == "data_collection":
            self._success = {
                "cube": cube_successes[self._target_block],
                "button": button_successes[self._target_button],
                "drawer": drawer_success,
                "window": window_success,
            }[self._target_task]
        else:
            self._success = all(cube_successes) and all(button_successes) and drawer_success and window_success

        # Adjust the colors of the cubes based on success.
        for i in range(self._num_cubes):
            if self._visualize_info and (self._mode == "task" or i == self._target_block):
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.2
            else:
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.0

            # if self._visualize_info and cube_successes[i]:
            #     for gid in self._cube_geom_ids_list[i]:
            #         self._model.geom(gid).rgba[:3] = self._cube_success_colors[i, :3]
            # else:
            #     for gid in self._cube_geom_ids_list[i]:
            #         self._model.geom(gid).rgba[:3] = self._cube_colors[i, :3]

    def get_reset_info(self):
        reset_info = self.compute_ob_info()
        reset_info["goal"] = self._cur_goal_ob
        reset_info["success"] = self._success
        return reset_info

    def get_step_info(self):
        ob_info = self.compute_ob_info()
        ob_info["goal"] = self._cur_goal_ob
        ob_info["success"] = self._success
        return ob_info

    def add_object_info(self, ob_info):
        # Cube positions and orientations.
        for i in range(self._num_cubes):
            ob_info[f"privileged/block_{i}_pos"] = self._data.joint(f"object_joint_{i}").qpos[:3].copy()
            ob_info[f"privileged/block_{i}_quat"] = self._data.joint(f"object_joint_{i}").qpos[3:].copy()
            ob_info[f"privileged/block_{i}_yaw"] = np.array(
                [lie.SO3(wxyz=self._data.joint(f"object_joint_{i}").qpos[3:]).compute_yaw_radians()]
            )

        # Button states.
        for i in range(self._num_buttons):
            ob_info[f"privileged/button_{i}_state"] = self._cur_button_states[i]
            ob_info[f"privileged/button_{i}_pos"] = self._data.joint(f"buttonbox_joint_{i}").qpos.copy()
            ob_info[f"privileged/button_{i}_vel"] = self._data.joint(f"buttonbox_joint_{i}").qvel.copy()

        # Drawer states.
        ob_info["privileged/drawer_pos"] = self._data.joint("drawer_slide").qpos.copy()
        ob_info["privileged/drawer_vel"] = self._data.joint("drawer_slide").qvel.copy()
        ob_info["privileged/drawer_handle_pos"] = self._data.site_xpos[self._drawer_site_id].copy()
        ob_info["privileged/drawer_handle_yaw"] = np.array(
            [lie.SO3.from_matrix(self._data.site_xmat[self._drawer_site_id].reshape(3, 3)).compute_yaw_radians()]
        )

        # Window states.
        ob_info["privileged/window_pos"] = self._data.joint("window_slide").qpos.copy()
        ob_info["privileged/window_vel"] = self._data.joint("window_slide").qvel.copy()
        ob_info["privileged/window_handle_pos"] = self._data.site_xpos[self._window_site_id].copy()
        ob_info["privileged/window_handle_yaw"] = np.array(
            [lie.SO3.from_matrix(self._data.site_xmat[self._window_site_id].reshape(3, 3)).compute_yaw_radians()]
        )

        if self._mode == "data_collection":
            ob_info["privileged/target_task"] = self._target_task

            # Target cube info.
            target_mocap_id = self._cube_target_mocap_ids[self._target_block]
            ob_info["privileged/target_block"] = self._target_block
            ob_info["privileged/target_block_pos"] = self._data.mocap_pos[target_mocap_id].copy()
            ob_info["privileged/target_block_yaw"] = np.array(
                [lie.SO3(wxyz=self._data.mocap_quat[target_mocap_id]).compute_yaw_radians()]
            )

            # Target button info.
            ob_info["privileged/target_button"] = self._target_button
            ob_info["privileged/target_button_state"] = self._target_button_states[self._target_button]
            ob_info["privileged/target_button_top_pos"] = self._data.site_xpos[
                self._button_site_ids[self._target_button]
            ].copy()

            # Target drawer info.
            ob_info["privileged/target_drawer_pos"] = np.array([self._target_drawer_pos])
            ob_info["privileged/target_drawer_handle_pos"] = self._data.site_xpos[self._drawer_target_site_id].copy()

            # Target window info.
            ob_info["privileged/target_window_pos"] = np.array([self._target_window_pos])
            ob_info["privileged/target_window_handle_pos"] = self._data.site_xpos[self._window_target_site_id].copy()

        ob_info["prev_button_states"] = self._prev_button_states.copy()
        ob_info["button_states"] = self._cur_button_states.copy()

    def compute_observation(self):
        if self._ob_type == "pixels":
            return self.get_pixel_observation()
        else:
            xyz_center = np.array([0.425, 0.0, 0.0])
            xyz_scaler = 10.0
            gripper_scaler = 3.0
            button_scaler = 120.0
            drawer_scaler = 18.0
            window_scaler = 15.0

            ob_info = self.compute_ob_info()
            ob = [
                ob_info["proprio/joint_pos"],
                ob_info["proprio/joint_vel"],
                (ob_info["proprio/effector_pos"] - xyz_center) * xyz_scaler,
                np.cos(ob_info["proprio/effector_yaw"]),
                np.sin(ob_info["proprio/effector_yaw"]),
                ob_info["proprio/gripper_opening"] * gripper_scaler,
                ob_info["proprio/gripper_contact"],
            ]
            for i in range(self._num_cubes):
                ob.extend(
                    [
                        (ob_info[f"privileged/block_{i}_pos"] - xyz_center) * xyz_scaler,
                        ob_info[f"privileged/block_{i}_quat"],
                        np.cos(ob_info[f"privileged/block_{i}_yaw"]),
                        np.sin(ob_info[f"privileged/block_{i}_yaw"]),
                    ]
                )
            for i in range(self._num_buttons):
                button_state = np.eye(self._num_button_states)[self._cur_button_states[i]]
                ob.extend(
                    [
                        button_state,
                        ob_info[f"privileged/button_{i}_pos"] * button_scaler,
                        ob_info[f"privileged/button_{i}_vel"],
                    ]
                )
            ob.extend(
                [
                    ob_info["privileged/drawer_pos"] * drawer_scaler,
                    ob_info["privileged/drawer_vel"],
                    ob_info["privileged/window_pos"] * window_scaler,
                    ob_info["privileged/window_vel"],
                ]
            )

            return np.concatenate(ob)

    def compute_oracle_observation(self):
        """Return the oracle goal representation of the current state."""
        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0
        drawer_scaler = 18.0
        window_scaler = 15.0

        ob_info = self.compute_ob_info()
        ob = []
        for i in range(self._num_cubes):
            ob.append((ob_info[f"privileged/block_{i}_pos"] - xyz_center) * xyz_scaler)
        ob.append(self._cur_button_states.astype(np.float64))
        ob.extend(
            [
                ob_info["privileged/drawer_pos"] * drawer_scaler,
                ob_info["privileged/window_pos"] * window_scaler,
            ]
        )

        return np.concatenate(ob)

    def compute_reward(self, ob, action):
        if self._reward_task_id is None:
            return super().compute_reward(ob, action)

        # Compute the reward based on the task.
        cube_successes, button_successes, drawer_success, window_success = self._compute_successes()
        successes = cube_successes + button_successes + [drawer_success, window_success]
        reward = float(sum(successes) - len(successes))
        return reward

    def render(
        self,
        camera="front_pixels",
        *args,
        **kwargs,
    ):
        """Render the current scene from a specified camera view.

        Generates an RGB image of the current environment state from a single
        camera viewpoint. This method renders from one camera at a time.

        Args:
            camera (str, optional): Camera name to render from. Defaults to
                'front_pixels'. Supports any camera defined in self.cameras
                (e.g., 'front_pixels', 'side_pixels').
            *args: Additional positional arguments passed to parent render method.
            **kwargs: Additional keyword arguments passed to parent render method.

        Returns:
            ndarray: Rendered image with shape (H, W, C) where H is height,
                W is width, and C is the number of color channels (typically 3 for RGB).

        Note:
            For rendering from multiple cameras simultaneously, use the
            `render_multiview()` method instead.
        """
        return super().render(camera=camera, *args, **kwargs)

    def render_multiview(
        self,
        camera="front_pixels",
        *args,
        **kwargs,
    ):
        """Render the current scene from multiple camera views or a fallback single view.

        When multiview mode is enabled (`_multiview=True`), renders the scene from
        both 'front_pixels' and 'side_pixels' cameras and returns them as a
        dictionary. When multiview is disabled, falls back to rendering from a
        single camera.

        Args:
            camera (str, optional): Camera name to use for fallback rendering when
                multiview is disabled. Defaults to 'front_pixels'. Ignored when
                multiview is enabled.
            *args: Additional positional arguments passed to the render method.
            **kwargs: Additional keyword arguments passed to the render method.

        Returns:
            dict or ndarray: If multiview is enabled, returns a dictionary with camera
                names as keys ('front_pixels', 'side_pixels') and rendered images as
                values, where each image has shape (H, W, C). If multiview is disabled,
                returns a single rendered image array with shape (H, W, C).

        Note:
            The multiview dictionary format is useful for policies that process
            multiple viewpoints separately. The 'front_pixels' camera provides an
            oblique view while 'side_pixels' shows a perpendicular side view.
        """

        if not self._multiview:
            return self.render(camera=camera, *args, **kwargs)

        cam_names = ["front_pixels", "side_pixels"]
        multi_view = {cam: self.render(camera=cam, *args, **kwargs) for cam in cam_names}
        return multi_view
