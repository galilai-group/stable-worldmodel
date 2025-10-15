import mujoco
import numpy as np
from dm_control import mjcf
from ogbench.manipspace import lie
from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv

import stable_worldmodel as swm


def perturb_camera_angle(xyaxis, deg_dif=[3, 3]):
    xaxis = np.array(xyaxis[:3])
    yaxis = np.array(xyaxis[3:])

    # Compute z-axis
    zaxis = np.cross(xaxis, yaxis)
    zaxis /= np.linalg.norm(zaxis)

    # Small random rotation (e.g. ±3 degrees)
    yaw = np.deg2rad(deg_dif[0])
    pitch = np.deg2rad(deg_dif[1])

    # Build rotation matrices
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    R_pitch = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])

    # Combine and rotate the basis
    R = R_pitch @ R_yaw
    xaxis_new = R @ xaxis
    yaxis_new = R @ yaxis

    # Flatten back to tuple for MuJoCo
    xyaxes_new = tuple(np.concatenate([xaxis_new, yaxis_new]))

    return xyaxes_new


class CubeEnv(ManipSpaceEnv):
    """Cube environment.

    This environment consists of a single or multiple cubes. The goal is to move the cubes to target positions. It
    supports the following variants:
    - `env_type`: 'single', 'double', 'triple', 'quadruple'.
    """

    def __init__(self, env_type, permute_blocks=True, multiview=False, *args, **kwargs):
        """Initialize the Cube environment.

        Args:
            env_type: Environment type corresponding to the number of cubes. One of 'single', 'double', 'triple', 'quadruple' or 'octuple'.
            permute_blocks: Whether to randomly permute the order of the blocks at task initialization.
            multiview: Whether to render the scene from both a front and side view.
            *args: Additional arguments to pass to the parent class.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        self._env_type = env_type
        self._permute_blocks = permute_blocks
        self._multiview = multiview

        if self._env_type == "single":
            self._num_cubes = 1
        elif self._env_type == "double":
            self._num_cubes = 2
        elif self._env_type == "triple":
            self._num_cubes = 3
        elif self._env_type == "quadruple":
            self._num_cubes = 4
        elif self._env_type == "octuple":
            self._num_cubes = 8
        else:
            raise ValueError(f"Invalid env_type: {env_type}")

        super().__init__(*args, **kwargs)

        # Define constants.
        self._cube_colors = np.array(
            [
                self._colors["red"],
                self._colors["blue"],
                self._colors["orange"],
                self._colors["green"],
                self._colors["purple"],
                self._colors["yellow"],
                self._colors["magenta"],
                self._colors["gray"],
            ]
        )
        self._cube_success_colors = np.array(
            [
                self._colors["lightred"],
                self._colors["lightblue"],
                self._colors["lightorange"],
                self._colors["lightgreen"],
                self._colors["lightpurple"],
                self._colors["lightyellow"],
                self._colors["lightmagenta"],
                self._colors["white"],
            ]
        )

        # Target info.
        self._target_task = "cube"
        # The target cube position is stored in the mocap object.
        self._target_block = 0

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
                            high=0.04,
                            shape=(self._num_cubes,),
                            dtype=np.float64,
                            init_value=0.02 * np.ones((self._num_cubes,), dtype=np.float32),
                        ),
                    }
                    # sampling_order=["num", "color", "size"]
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
                            low=-10.0,
                            high=10.0,
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

    def set_tasks(self):
        if self._env_type == "single":
            self.task_infos = [
                {
                    "task_name": "task1_horizontal",
                    "init_xyzs": np.array([[0.425, 0.1, 0.02]]),
                    "goal_xyzs": np.array([[0.425, -0.1, 0.02]]),
                },
                {
                    "task_name": "task2_vertical1",
                    "init_xyzs": np.array([[0.35, 0.0, 0.02]]),
                    "goal_xyzs": np.array([[0.50, 0.0, 0.02]]),
                },
                {
                    "task_name": "task3_vertical2",
                    "init_xyzs": np.array([[0.50, 0.0, 0.02]]),
                    "goal_xyzs": np.array([[0.35, 0.0, 0.02]]),
                },
                {
                    "task_name": "task4_diagonal1",
                    "init_xyzs": np.array([[0.35, -0.2, 0.02]]),
                    "goal_xyzs": np.array([[0.50, 0.2, 0.02]]),
                },
                {
                    "task_name": "task5_diagonal2",
                    "init_xyzs": np.array([[0.35, 0.2, 0.02]]),
                    "goal_xyzs": np.array([[0.50, -0.2, 0.02]]),
                },
            ]
        elif self._env_type == "double":
            self.task_infos = [
                {
                    "task_name": "task1_single_pnp",
                    "init_xyzs": np.array(
                        [
                            [0.425, 0.0, 0.02],
                            [0.425, -0.1, 0.02],
                        ]
                    ),
                    "goal_xyzs": np.array(
                        [
                            [0.425, 0.0, 0.02],
                            [0.425, 0.1, 0.02],
                        ]
                    ),
                },
                {
                    "task_name": "task2_double_pnp1",
                    "init_xyzs": np.array(
                        [
                            [0.35, -0.1, 0.02],
                            [0.50, -0.1, 0.02],
                        ]
                    ),
                    "goal_xyzs": np.array(
                        [
                            [0.35, 0.1, 0.02],
                            [0.50, 0.1, 0.02],
                        ]
                    ),
                },
                {
                    "task_name": "task3_double_pnp2",
                    "init_xyzs": np.array(
                        [
                            [0.35, 0.0, 0.02],
                            [0.50, 0.0, 0.02],
                        ]
                    ),
                    "goal_xyzs": np.array(
                        [
                            [0.425, -0.2, 0.02],
                            [0.425, 0.2, 0.02],
                        ]
                    ),
                },
                {
                    "task_name": "task4_swap",
                    "init_xyzs": np.array(
                        [
                            [0.425, -0.1, 0.02],
                            [0.425, 0.1, 0.02],
                        ]
                    ),
                    "goal_xyzs": np.array(
                        [
                            [0.425, 0.1, 0.02],
                            [0.425, -0.1, 0.02],
                        ]
                    ),
                },
                {
                    "task_name": "task5_stack",
                    "init_xyzs": np.array(
                        [
                            [0.425, -0.2, 0.02],
                            [0.425, 0.2, 0.02],
                        ]
                    ),
                    "goal_xyzs": np.array(
                        [
                            [0.425, 0.0, 0.02],
                            [0.425, 0.0, 0.06],
                        ]
                    ),
                },
            ]
        elif self._env_type == "triple":
            self.task_infos = [
                {
                    "task_name": "task1_single_pnp",
                    "init_xyzs": np.array(
                        [
                            [0.35, -0.1, 0.02],
                            [0.35, 0.1, 0.02],
                            [0.50, -0.1, 0.02],
                        ]
                    ),
                    "goal_xyzs": np.array(
                        [
                            [0.35, -0.1, 0.02],
                            [0.35, 0.1, 0.02],
                            [0.50, 0.1, 0.02],
                        ]
                    ),
                },
                {
                    "task_name": "task2_triple_pnp",
                    "init_xyzs": np.array(
                        [
                            [0.35, -0.2, 0.02],
                            [0.35, 0.0, 0.02],
                            [0.35, 0.2, 0.02],
                        ]
                    ),
                    "goal_xyzs": np.array(
                        [
                            [0.50, 0.0, 0.02],
                            [0.50, 0.2, 0.02],
                            [0.50, -0.2, 0.02],
                        ]
                    ),
                },
                {
                    "task_name": "task3_pnp_from_stack",
                    "init_xyzs": np.array(
                        [
                            [0.425, 0.2, 0.02],
                            [0.425, 0.2, 0.06],
                            [0.425, 0.2, 0.10],
                        ]
                    ),
                    "goal_xyzs": np.array(
                        [
                            [0.35, -0.1, 0.02],
                            [0.50, -0.2, 0.02],
                            [0.50, 0.0, 0.02],
                        ]
                    ),
                },
                {
                    "task_name": "task4_cycle",
                    "init_xyzs": np.array(
                        [
                            [0.35, 0.0, 0.02],
                            [0.50, -0.1, 0.02],
                            [0.50, 0.1, 0.02],
                        ]
                    ),
                    "goal_xyzs": np.array(
                        [
                            [0.50, -0.1, 0.02],
                            [0.50, 0.1, 0.02],
                            [0.35, 0.0, 0.02],
                        ]
                    ),
                },
                {
                    "task_name": "task5_stack",
                    "init_xyzs": np.array(
                        [
                            [0.35, -0.1, 0.02],
                            [0.50, -0.2, 0.02],
                            [0.50, 0.0, 0.02],
                        ]
                    ),
                    "goal_xyzs": np.array(
                        [
                            [0.425, 0.2, 0.02],
                            [0.425, 0.2, 0.06],
                            [0.425, 0.2, 0.10],
                        ]
                    ),
                },
            ]
        elif self._env_type == "quadruple":
            self.task_infos = [
                {
                    "task_name": "task1_double_pnp",
                    "init_xyzs": np.array(
                        [
                            [0.35, -0.1, 0.02],
                            [0.35, 0.1, 0.02],
                            [0.50, -0.1, 0.02],
                            [0.50, 0.1, 0.02],
                        ]
                    ),
                    "goal_xyzs": np.array(
                        [
                            [0.35, -0.25, 0.02],
                            [0.35, 0.1, 0.02],
                            [0.50, -0.1, 0.02],
                            [0.50, 0.25, 0.02],
                        ]
                    ),
                },
                {
                    "task_name": "task2_quadruple_pnp",
                    "init_xyzs": np.array(
                        [
                            [0.325, -0.2, 0.02],
                            [0.325, 0.2, 0.02],
                            [0.525, -0.2, 0.02],
                            [0.525, 0.2, 0.02],
                        ]
                    ),
                    "goal_xyzs": np.array(
                        [
                            [0.375, 0.1, 0.02],
                            [0.475, 0.1, 0.02],
                            [0.375, -0.1, 0.02],
                            [0.475, -0.1, 0.02],
                        ]
                    ),
                },
                {
                    "task_name": "task3_pnp_from_square",
                    "init_xyzs": np.array(
                        [
                            [0.425, -0.02, 0.02],
                            [0.425, 0.02, 0.02],
                            [0.425, -0.02, 0.06],
                            [0.425, 0.02, 0.06],
                        ]
                    ),
                    "goal_xyzs": np.array(
                        [
                            [0.525, -0.2, 0.02],
                            [0.325, 0.2, 0.02],
                            [0.325, -0.2, 0.02],
                            [0.525, 0.2, 0.02],
                        ]
                    ),
                },
                {
                    "task_name": "task4_cycle",
                    "init_xyzs": np.array(
                        [
                            [0.525, -0.1, 0.02],
                            [0.525, 0.1, 0.02],
                            [0.325, 0.1, 0.02],
                            [0.325, -0.1, 0.02],
                        ]
                    ),
                    "goal_xyzs": np.array(
                        [
                            [0.525, 0.1, 0.02],
                            [0.325, 0.1, 0.02],
                            [0.325, -0.1, 0.02],
                            [0.525, -0.1, 0.02],
                        ]
                    ),
                },
                {
                    "task_name": "task5_stack",
                    "init_xyzs": np.array(
                        [
                            [0.50, -0.05, 0.02],
                            [0.50, -0.2, 0.02],
                            [0.35, -0.2, 0.02],
                            [0.35, -0.05, 0.02],
                        ]
                    ),
                    "goal_xyzs": np.array(
                        [
                            [0.425, 0.2, 0.02],
                            [0.425, 0.2, 0.06],
                            [0.425, 0.2, 0.10],
                            [0.425, 0.2, 0.14],
                        ]
                    ),
                },
            ]
        elif self._env_type == "octuple":
            self.task_infos = [
                {
                    "task_name": "task1_quadruple_pnp",
                    "init_xyzs": np.array(
                        [
                            [0.325, -0.15, 0.02],
                            [0.425, -0.15, 0.02],
                            [0.425, -0.05, 0.02],
                            [0.525, -0.05, 0.02],
                            [0.325, 0.05, 0.02],
                            [0.425, 0.05, 0.02],
                            [0.425, 0.15, 0.02],
                            [0.525, 0.15, 0.02],
                        ]
                    ),
                    "goal_xyzs": np.array(
                        [
                            [0.525, 0.05, 0.02],
                            [0.425, -0.15, 0.02],
                            [0.425, -0.05, 0.02],
                            [0.325, 0.15, 0.02],
                            [0.525, -0.15, 0.02],
                            [0.425, 0.05, 0.02],
                            [0.425, 0.15, 0.02],
                            [0.325, -0.05, 0.02],
                        ]
                    ),
                },
                {
                    "task_name": "task2_octuple_pnp1",
                    "init_xyzs": np.array(
                        [
                            [0.40, -0.15, 0.02],
                            [0.40, -0.05, 0.02],
                            [0.40, 0.05, 0.02],
                            [0.40, 0.15, 0.02],
                            [0.45, -0.15, 0.02],
                            [0.45, -0.05, 0.02],
                            [0.45, 0.05, 0.02],
                            [0.45, 0.15, 0.02],
                        ]
                    ),
                    "goal_xyzs": np.array(
                        [
                            [0.525, 0.2, 0.02],
                            [0.525, 0.0, 0.02],
                            [0.525, -0.2, 0.02],
                            [0.425, -0.2, 0.02],
                            [0.325, -0.2, 0.02],
                            [0.325, 0.0, 0.02],
                            [0.325, 0.2, 0.02],
                            [0.425, 0.2, 0.02],
                        ]
                    ),
                },
                {
                    "task_name": "task3_octuple_pnp2",
                    "init_xyzs": np.array(
                        [
                            [0.32, -0.15, 0.02],
                            [0.32, 0.05, 0.02],
                            [0.39, -0.05, 0.02],
                            [0.39, 0.15, 0.02],
                            [0.46, -0.15, 0.02],
                            [0.46, 0.05, 0.02],
                            [0.53, -0.05, 0.02],
                            [0.53, 0.15, 0.02],
                        ]
                    ),
                    "goal_xyzs": np.array(
                        [
                            [0.32, 0.15, 0.02],
                            [0.46, 0.15, 0.02],
                            [0.39, 0.05, 0.02],
                            [0.53, 0.05, 0.02],
                            [0.32, -0.05, 0.02],
                            [0.46, -0.05, 0.02],
                            [0.39, -0.15, 0.02],
                            [0.53, -0.15, 0.02],
                        ]
                    ),
                },
                {
                    "task_name": "task4_stack1",
                    "init_xyzs": np.array(
                        [
                            [0.40, -0.025, 0.02],
                            [0.40, -0.025, 0.06],
                            [0.40, 0.025, 0.02],
                            [0.40, 0.025, 0.06],
                            [0.45, -0.025, 0.02],
                            [0.45, -0.025, 0.06],
                            [0.45, 0.025, 0.02],
                            [0.45, 0.025, 0.06],
                        ]
                    ),
                    "goal_xyzs": np.array(
                        [
                            [0.525, 0.05, 0.02],
                            [0.525, -0.05, 0.02],
                            [0.425, -0.15, 0.06],
                            [0.425, -0.15, 0.02],
                            [0.425, 0.15, 0.06],
                            [0.425, 0.15, 0.02],
                            [0.325, -0.05, 0.02],
                            [0.325, 0.05, 0.02],
                        ]
                    ),
                },
                {
                    "task_name": "task5_stack2",
                    "init_xyzs": np.array(
                        [
                            [0.50, 0.2, 0.06],
                            [0.50, 0.2, 0.02],
                            [0.50, -0.2, 0.06],
                            [0.50, -0.2, 0.02],
                            [0.35, -0.2, 0.06],
                            [0.35, -0.2, 0.02],
                            [0.35, 0.2, 0.06],
                            [0.35, 0.2, 0.02],
                        ]
                    ),
                    "goal_xyzs": np.array(
                        [
                            [0.325, 0.0, 0.02],
                            [0.325, 0.0, 0.06],
                            [0.425, 0.2, 0.02],
                            [0.425, 0.2, 0.06],
                            [0.525, 0.0, 0.02],
                            [0.525, 0.0, 0.06],
                            [0.425, -0.2, 0.02],
                            [0.425, -0.2, 0.06],
                        ]
                    ),
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
        # Add cube scene.
        cube_outer_mjcf = mjcf.from_path((self._desc_dir / "cube_outer.xml").as_posix())
        arena_mjcf.include_copy(cube_outer_mjcf)

        # Add `num_cubes` cubes to the scene.
        distance = 0.05
        for i in range(self._num_cubes):
            cube_mjcf = mjcf.from_path((self._desc_dir / "cube_inner.xml").as_posix())
            pos = -distance * (self._num_cubes - 1) + 2 * distance * i
            cube_mjcf.find("body", "object_0").pos[1] = pos
            cube_mjcf.find("body", "object_target_0").pos[1] = pos
            for tag in ["body", "joint", "geom", "site"]:
                for item in cube_mjcf.find_all(tag):
                    if hasattr(item, "name") and item.name is not None and item.name.endswith("_0"):
                        item.name = item.name[:-2] + f"_{i}"
            arena_mjcf.include_copy(cube_mjcf)

        # Save cube geoms.
        self._cube_geoms_list = []
        for i in range(self._num_cubes):
            self._cube_geoms_list.append(arena_mjcf.find("body", f"object_{i}").find_all("geom"))
        self._cube_target_geoms_list = []
        for i in range(self._num_cubes):
            self._cube_target_geoms_list.append(arena_mjcf.find("body", f"object_target_{i}").find_all("geom"))

        # Add cameras.
        self.cameras = {
            "front": {
                "pos": (1.287, 0.000, 0.509),
                "xyaxes": (0.000, 1.000, 0.000, -0.342, 0.000, 0.940),
            },
            "front_pixels": {
                "pos": (1.053, -0.014, 0.639),
                "xyaxes": (0.000, 1.000, 0.000, -0.628, 0.001, 0.778),
            },
            "side_pixels": {
                "pos": (0.414, -0.753, 0.639),
                "xyaxes": (1.000, 0.000, 0.000, -0.001, 0.628, 0.778),
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

            # Randomize object positions and orientations.
            for i in range(self._num_cubes):
                xy = self.np_random.uniform(*self._object_sampling_bounds)
                obj_pos = (*xy, 0.02)
                yaw = self.np_random.uniform(0, 2 * np.pi)
                obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                self._data.joint(f"object_joint_{i}").qpos[:3] = obj_pos
                self._data.joint(f"object_joint_{i}").qpos[3:] = obj_ori

            # Set a new target.
            self.set_new_target(return_info=False)
        else:
            # Set object positions and orientations based on the current task.

            if self._permute_blocks:
                # Randomize the order of the cubes when there are multiple cubes.
                permutation = self.np_random.permutation(self._num_cubes)
            else:
                permutation = np.arange(self._num_cubes)
            init_xyzs = self.cur_task_info["init_xyzs"].copy()[permutation]
            goal_xyzs = self.cur_task_info["goal_xyzs"].copy()[permutation]

            # First, force set the current scene to the goal state to obtain the goal observation.
            saved_qpos = self._data.qpos.copy()
            saved_qvel = self._data.qvel.copy()
            self.initialize_arm()
            for i in range(self._num_cubes):
                self._data.joint(f"object_joint_{i}").qpos[:3] = goal_xyzs[i]
                self._data.joint(f"object_joint_{i}").qpos[3:] = lie.SO3.identity().wxyz.tolist()
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = goal_xyzs[i]
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()
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
                obj_pos = init_xyzs[i].copy()
                obj_pos[:2] += self.np_random.uniform(-0.01, 0.01, size=2)
                self._data.joint(f"object_joint_{i}").qpos[:3] = obj_pos
                yaw = self.np_random.uniform(0, 2 * np.pi)
                obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                self._data.joint(f"object_joint_{i}").qpos[3:] = obj_ori
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = goal_xyzs[i]
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()

        # Forward kinematics to update site positions.
        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()

        self._success = False

    def set_new_target(self, return_info=True, p_stack=0.5):
        """Set a new random target for data collection.

        Args:
            return_info: Whether to return the observation and reset info.
            p_stack: Probability of stacking the target block on top of another block when there are multiple blocks.
        """
        assert self._mode == "data_collection"

        block_xyzs = np.array([self._data.joint(f"object_joint_{i}").qpos[:3] for i in range(self._num_cubes)])

        # Compute the top blocks.
        top_blocks = []
        for i in range(self._num_cubes):
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

        stack = len(top_blocks) >= 2 and self.np_random.uniform() < p_stack
        if stack:
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

        if return_info:
            return self.compute_observation(), self.get_reset_info()

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

        return cube_successes

    def post_step(self):
        # Check if the cubes are in the target positions.
        cube_successes = self._compute_successes()
        if self._mode == "data_collection":
            self._success = cube_successes[self._target_block]
        else:
            self._success = all(cube_successes)

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
        if self._mode == "task":
            reset_info["goal"] = self._cur_goal_ob
        reset_info["success"] = self._success
        return reset_info

    def get_step_info(self):
        ob_info = self.compute_ob_info()
        if self._mode == "task":
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

        if self._mode == "data_collection":
            # Target cube info.
            ob_info["privileged/target_task"] = self._target_task

            target_mocap_id = self._cube_target_mocap_ids[self._target_block]
            ob_info["privileged/target_block"] = self._target_block
            ob_info["privileged/target_block_pos"] = self._data.mocap_pos[target_mocap_id].copy()
            ob_info["privileged/target_block_yaw"] = np.array(
                [lie.SO3(wxyz=self._data.mocap_quat[target_mocap_id]).compute_yaw_radians()]
            )

    def compute_observation(self):
        if self._ob_type == "pixels":
            return self.get_pixel_observation()
        else:
            xyz_center = np.array([0.425, 0.0, 0.0])
            xyz_scaler = 10.0
            gripper_scaler = 3.0

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

            return np.concatenate(ob)

    def compute_oracle_observation(self):
        """Return the oracle goal representation of the current state."""
        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0

        ob_info = self.compute_ob_info()
        ob = []
        for i in range(self._num_cubes):
            ob.append((ob_info[f"privileged/block_{i}_pos"] - xyz_center) * xyz_scaler)

        return np.concatenate(ob)

    def compute_reward(self, ob, action):
        if self._reward_task_id is None:
            return super().compute_reward(ob, action)

        # Compute the reward based on the task.
        successes = self._compute_successes()
        reward = float(sum(successes) - len(successes))
        return reward

    def render(
        self,
        camera="front_pixels",
        *args,
        **kwargs,
    ):
        camera = "front_pixels" if not self._multiview else ["front_pixels", "side_pixels"]
        if isinstance(camera, list | tuple):
            imgs = []
            for cam in camera:
                img = super().render(camera=cam, *args, **kwargs)
                imgs.append(img)
            stacked_views = np.stack(imgs, axis=0)
            return stacked_views
        else:
            return super().render(camera=camera, *args, **kwargs)
