"""OGBench Cube manipulation environment with multiple task variants.

This module implements a robotic manipulation environment using cubes with various
task configurations ranging from single cube pick-and-place to complex multi-cube
stacking and rearrangement tasks. The environment supports visual variations including
object colors, sizes, lighting, and camera angles for robust policy learning.

The environment is built on top of the ManipSpaceEnv from OGBench and uses MuJoCo
for physics simulation. It provides both pixel-based and state-based observations,
with support for goal-conditioned learning and data collection modes.

Example:
    Basic usage of the cube environment::

        from stable_worldmodel.envs.ogbench_cube import CubeEnv

        # Create a double cube environment with pixel observations
        env = CubeEnv(env_type='double', ob_type='pixels', multiview=True)

        # Reset with variation sampling
        obs, info = env.reset(options={'variation': ['all']})

        # Run an episode
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if info['success']:
                break
.. _OGBench:
   https://github.com/seohongpark/ogbench/
"""

import mujoco
import numpy as np
from dm_control import mjcf
from ogbench.manipspace import lie
from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv

import stable_worldmodel as swm


def perturb_camera_angle(xyaxis, deg_dif=[3, 3]):
    """Perturb camera orientation by applying yaw and pitch rotations.

    Applies random rotations to the camera's coordinate frame defined by its
    x and y axes. The perturbation helps create visual variations during training
    to improve policy robustness.

    Args:
        xyaxis (array-like): Six-element array representing the camera's coordinate
            frame in MuJoCo format. First three elements are the x-axis direction,
            last three elements are the y-axis direction.
        deg_dif (list, optional): Two-element list specifying [yaw, pitch] rotation
            angles in degrees. Defaults to [3, 3].

    Returns:
        tuple: Six-element tuple containing the perturbed camera axes in MuJoCo
            format (xaxis_new, yaxis_new).

    Note:
        The z-axis is computed from the cross product of x and y axes and used
        to construct proper rotation matrices.
    """
    xaxis = np.array(xyaxis[:3])
    yaxis = np.array(xyaxis[3:])

    # Compute z-axis
    zaxis = np.cross(xaxis, yaxis)
    zaxis /= np.linalg.norm(zaxis)

    # random rotation
    yaw = np.deg2rad(deg_dif[0])
    pitch = np.deg2rad(deg_dif[1])

    # rotation matrices
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R_pitch = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])

    # Combine and rotate the basis
    R = R_pitch @ R_yaw
    xaxis_new = R @ xaxis
    yaxis_new = R @ yaxis

    xyaxes_new = tuple(np.concatenate([xaxis_new, yaxis_new]))  # mujoco format

    return xyaxes_new


class CubeEnv(ManipSpaceEnv):
    """Robotic manipulation environment with cube objects and multiple task variants.

    This environment provides a suite of manipulation tasks involving 1-8 colored cubes
    that must be moved to target positions. It supports various task types including
    pick-and-place, stacking, swapping, and cyclic rearrangement. The environment
    includes comprehensive variation spaces for visual domain randomization.

    The environment operates in two modes:
        - 'task': Goal-conditioned mode where the robot must achieve specific configurations
        - 'data_collection': Mode for collecting demonstrations with random targets

    Attributes:
        _env_type (str): Type of environment determining number of cubes.
            One of: 'single', 'double', 'triple', 'quadruple', 'octuple'.
        _num_cubes (int): Number of cubes in the environment (1, 2, 3, 4, or 8).
        _permute_blocks (bool): Whether to randomly permute cube order at task init.
        _multiview (bool): Whether to render from both front and side cameras.
        _ob_type (str): Observation type, either 'pixels' or state-based.
        _cube_colors (ndarray): Array of RGB colors for cubes.
        _target_task (str): Task type identifier, set to 'cube'.
        _target_block (int): Index of the target cube in data collection mode.
        variation_space (Dict): Hierarchical space defining variation ranges for:
            - cube: color (RGB), size (half-extents)
            - agent: color (RGB)
            - floor: color (two RGB values for checkerboard)
            - camera: angle_delta (yaw/pitch offsets)
            - light: intensity (diffuse lighting strength)
        task_infos (list): List of dictionaries defining task configurations,
            each containing 'task_name', 'init_xyzs', and 'goal_xyzs'.
        cameras (dict): Dictionary of camera configurations with position and orientation.
        _cube_geoms_list (list): MuJoCo geom objects for each cube.
        _cube_target_geoms_list (list): MuJoCo geom objects for target visualizations.
        _cube_geom_ids_list (list): MuJoCo geom IDs for each cube.
        _cube_target_mocap_ids (list): MuJoCo mocap body IDs for target positions.
        _cube_target_geom_ids_list (list): MuJoCo geom IDs for target visualizations.
        _success (bool): Whether the current task has been completed successfully.
        _cur_goal_ob (ndarray): Goal observation for goal-conditioned learning.
        _cur_goal_rendered (ndarray): Rendered image of goal state, if enabled.

    Note:
        Inherits from ManipSpaceEnv which provides the underlying robotic arm
        control, physics simulation, and base functionality.
    """

    def __init__(
        self,
        env_type="single",
        ob_type="pixels",
        permute_blocks=True,
        multiview=False,
        height=224,
        width=224,
        *args,
        **kwargs,
    ):
        """Initialize the CubeEnv with specified configuration.

        Sets up the manipulation environment with the specified number of cubes
        and configures observation type, block permutation, and camera views.
        Initializes the variation space for visual domain randomization.

        Args:
            env_type (str): Environment type corresponding to number of cubes.
                Must be one of: 'single' (1 cube), 'double' (2 cubes),
                'triple' (3 cubes), 'quadruple' (4 cubes), 'octuple' (8 cubes).
            ob_type (str, optional): Type of observation to return. Either 'pixels'
                for image observations or 'state' for proprioceptive/object states.
                Defaults to 'pixels'.
            permute_blocks (bool, optional): Whether to randomly shuffle the order
                of cubes at the start of each episode. Helps with generalization.
                Defaults to True.
            multiview (bool, optional): Whether to render the scene from both front
                and side camera views simultaneously. Returns stacked images when True.
                Defaults to False.
            *args: Variable length argument list passed to parent ManipSpaceEnv.
            **kwargs: Arbitrary keyword arguments passed to parent ManipSpaceEnv.

        Raises:
            ValueError: If env_type is not one of the supported values.

        Note:
            The variation_space is automatically configured with appropriate ranges
            for all visual variations including colors, sizes, lighting, and cameras.
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

        super().__init__(*args, height=height, width=width, **kwargs)

        self._ob_type = ob_type
        self._cube_colors = np.stack(list(self._colors.values()))[:, :3]
        self._target_task = "cube"
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
                            init_value=self._cube_colors[: self._num_cubes].copy(),
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
        """Define all task configurations for the environment.

        Initializes the task_infos list with predefined manipulation tasks appropriate
        for the current env_type. Each task specifies initial and goal positions for
        all cubes. Tasks increase in complexity from simple pick-and-place to
        multi-object stacking and cyclic rearrangements.

        Task types by environment:
            - single: 5 tasks (horizontal, vertical movements, diagonals)
            - double: 5 tasks (single/double pick-place, swap, stack)
            - triple: 5 tasks (single/triple pick-place, unstack, cycle, stack)
            - quadruple: 5 tasks (double/quad pick-place, unstack, cycle, stack)
            - octuple: 5 tasks (quad/octuple pick-place, unstacking, stacking)

        Note:
            Also sets the default reward_task_id to 2 if not already configured.
            All positions are in MuJoCo world coordinates (x, y, z) in meters.
        """
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

    def reset(self, seed=None, options=None, *args, **kwargs):
        """Reset the environment to an initial state.

        Resets the environment and optionally samples from the variation space to
        create visual diversity. Handles both task mode (with predefined goals) and
        data collection mode (with random targets).

        Args:
            options (dict, optional): Dictionary of reset options. Supported keys:
                - 'variation': List/tuple of variation names to sample. Use ['all']
                  to sample all variations, or specify individual ones like
                  ['cube.color', 'light.intensity']. Defaults to None (no variation).
            *args: Variable length argument list passed to parent reset.
            **kwargs: Arbitrary keyword arguments passed to parent reset.

        Returns:
            tuple: (observation, info) where:
                - observation: Current observation based on ob_type configuration
                - info: Dictionary containing reset information and task state

        Raises:
            AssertionError: If variation option is not a list/tuple, or if variation
                values are outside their defined spaces.

        Example:
            Reset with all variations enabled::

                obs, info = env.reset(options={'variation': ['all']})

            Reset with specific variations::

                obs, info = env.reset(options={'variation': ['cube.color', 'camera.angle_delta']})
        """

        if hasattr(self, "variation_space"):
            self.variation_space.seed(seed)

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

        return super().reset(seed=seed, options=options, *args, **kwargs)

    def add_objects(self, arena_mjcf):
        """Add cube objects and cameras to the MuJoCo scene.

        Constructs the manipulation scene by loading cube XML descriptions and
        positioning them appropriately. Sets up multiple camera viewpoints for
        rendering observations.

        Args:
            arena_mjcf (mjcf.RootElement): The MuJoCo XML root element representing
                the arena where objects and cameras will be added.

        Note:
            - Cubes are positioned with 0.05m spacing along the y-axis
            - Each cube has both a physical object and a semi-transparent target marker
            - Three cameras are added: 'front', 'front_pixels', and 'side_pixels'
            - All cube geoms are stored for later color and property modifications
        """
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
        """Extract MuJoCo object IDs after model compilation.

        Retrieves and stores the integer IDs for all cube geoms, target mocap bodies,
        and target geoms. These IDs are used for efficient access during simulation
        and rendering.

        Note:
            Must be called after the MuJoCo model has been compiled. IDs are needed
            for direct manipulation of model properties like colors and positions.
        """
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
        """Apply visual variations to the MuJoCo model based on variation space.

        Modifies the MJCF model XML to apply sampled variations including floor colors,
        robot arm colors, cube sizes, camera angles, and lighting. Only variations
        that are enabled in the variation_options are applied.

        Args:
            mjcf_model (mjcf.RootElement): The MuJoCo XML model to modify.

        Returns:
            mjcf.RootElement: The modified model with variations applied.

        Note:
            - Variations are only applied if specified in variation_options during reset
            - Some variations (size, light) call self.mark_dirty() to trigger recompilation
            - Camera angle perturbations use the perturb_camera_angle helper function
        """
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
        """Initialize the environment state at the start of an episode.

        Sets up cube colors, arm position, and object placements based on the current
        mode (task or data_collection). In task mode, creates goal observations and
        places cubes according to the current task definition. In data collection mode,
        randomizes cube placements and sets a random target.

        The initialization process:
            1. Apply cube colors from variation space
            2. Reset arm to home position
            3. In task mode: Set cubes to task-specific positions, generate goal observation
            4. In data_collection mode: Randomize cube positions and orientations
            5. Run forward kinematics to stabilize the scene

        Note:
            - In task mode, goal observation is computed by first placing cubes at
              goal positions, rendering/observing, then resetting to initial positions
            - Small random perturbations (±0.01m) are added to initial positions
            - Random yaw rotations (0-2π) are applied to all cubes
        """
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
        """Set a new random target for data collection mode.

        Randomly selects one of the "top" cubes (not stacked under another) as the
        target and assigns it a random goal position. The goal can be either a flat
        surface position or stacked on top of another cube.

        Args:
            return_info (bool, optional): Whether to return the observation and reset
                info after setting the new target. Defaults to True.
            p_stack (float, optional): Probability of setting the target to stack on
                top of another block (when multiple blocks are available). Must be
                in range [0, 1]. Defaults to 0.5.

        Returns:
            tuple or None: If return_info is True, returns (observation, reset_info).
                Otherwise returns None.

        Raises:
            AssertionError: If called when mode is not 'data_collection'.

        Note:
            - Only cubes that are not underneath other cubes can be selected as targets
            - Target markers are made visible for the selected cube, invisible for others
            - Stacking targets are positioned 0.04m above the base cube's z-position
            - Non-stacking targets are randomly sampled from the target sampling bounds
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
        """Compute success status for each cube.

        Checks whether each cube is within the success threshold distance of its
        corresponding target position. A cube is considered successful if the
        Euclidean distance to its target is ≤ 0.04m.

        Returns:
            list of bool: Boolean list where each element indicates whether the
                corresponding cube has reached its target position. Length equals
                the number of cubes in the environment.

        Note:
            Success threshold of 0.04m (40mm) allows for small positioning errors
            while ensuring cubes are substantially at their goals.
        """
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
        """Update environment state after each simulation step.

        Computes success status and adjusts target marker visibility based on the
        current mode. In task mode, all cube targets are evaluated. In data collection
        mode, only the designated target cube is evaluated.

        Updates:
            - self._success: Set to True if success conditions are met
            - Target geom alpha: Made visible (0.2) for relevant targets when
              visualization is enabled, invisible (0.0) otherwise

        Note:
            Success in task mode requires ALL cubes to reach their targets.
            Success in data collection mode requires only the target cube to succeed.
        """
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

    def get_reset_info(self):
        """Get information dictionary at environment reset.

        Compiles observation info along with goal observations (in task mode) and
        success status to provide comprehensive reset information.

        Returns:
            dict: Dictionary containing:
                - All keys from compute_ob_info() (proprioception and object states)
                - 'goal': Goal observation (task mode only)
                - 'success': Boolean indicating current success status

        Note:
            Called after initialize_episode() to provide initial state information.
        """
        reset_info = self.compute_ob_info()
        if self._mode == "task":
            reset_info["goal"] = self._cur_goal_ob
        reset_info["success"] = self._success
        return reset_info

    def get_step_info(self):
        """Get information dictionary after each environment step.

        Compiles current observation info along with goal observations (in task mode)
        and success status to provide comprehensive step information.

        Returns:
            dict: Dictionary containing:
                - All keys from compute_ob_info() (proprioception and object states)
                - 'goal': Goal observation (task mode only)
                - 'success': Boolean indicating whether task is completed

        Note:
            Called after each step to provide feedback about current state and progress.
        """
        ob_info = self.compute_ob_info()
        if self._mode == "task":
            ob_info["goal"] = self._cur_goal_ob
        ob_info["success"] = self._success
        return ob_info

    def add_object_info(self, ob_info):
        """Add cube-specific information to the observation info dictionary.

        Augments the info dictionary with privileged state information about all cubes
        including positions, orientations, and target information (in data collection mode).

        Args:
            ob_info (dict): Observation info dictionary to augment. Modified in-place.

        Adds to ob_info:
            - 'privileged/block_{i}_pos': 3D position (x, y, z) of cube i
            - 'privileged/block_{i}_quat': Quaternion (w, x, y, z) of cube i
            - 'privileged/block_{i}_yaw': Yaw angle in radians of cube i
            - 'privileged/target_task': Task type string (data collection mode only)
            - 'privileged/target_block': Index of target cube (data collection mode only)
            - 'privileged/target_block_pos': Target position (data collection mode only)
            - 'privileged/target_block_yaw': Target yaw angle (data collection mode only)

        Note:
            All positions are in world coordinates. Quaternions use (w, x, y, z) format.
            Privileged information is typically not available to policies during deployment.
        """
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
        """Compute the current observation based on observation type.

        Generates either pixel-based or state-based observations depending on the
        ob_type configuration. State observations include scaled proprioceptive
        and object state information.

        Returns:
            ndarray: Observation array. If ob_type is 'pixels', returns image array
                with shape (H, W, C) or (N, H, W, C) for multiview. If ob_type is
                not 'pixels', returns flattened state vector containing:
                - Arm joint positions (6D) and velocities (6D)
                - End-effector position (3D, scaled), yaw angle (2D: cos/sin)
                - Gripper opening (1D, scaled) and contact (binary)
                - For each cube: position (3D, scaled), quaternion (4D), yaw (2D: cos/sin)

        Note:
            State observations use a centering offset (0.425, 0.0, 0.0) and scaling
            factors (10.0 for positions, 3.0 for gripper) to normalize values.
            Yaw angles are encoded as (cos, sin) pairs for continuity.
        """
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
        """Compute oracle goal representation containing only cube positions.

        Returns a compact state representation containing only the positions of all
        cubes, useful for goal-conditioned learning where the goal is defined by
        object configurations rather than the full state.

        Returns:
            ndarray: Concatenated cube positions with shape (num_cubes * 3,).
                Each cube contributes its (x, y, z) position, centered and scaled
                by the same factors used in compute_observation().

        Note:
            This representation excludes robot state and object orientations,
            focusing only on cube positions. Used primarily in task mode for
            goal specification.
        """
        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0

        ob_info = self.compute_ob_info()
        ob = []
        for i in range(self._num_cubes):
            ob.append((ob_info[f"privileged/block_{i}_pos"] - xyz_center) * xyz_scaler)

        return np.concatenate(ob)

    def compute_reward(self, ob, action):
        """Compute the reward for the current step.

        Calculates reward based on task success. If a specific reward_task_id is set,
        uses a custom reward function that counts successful cube placements minus
        the total number of cubes (range: -num_cubes to 0). Otherwise defers to
        parent class reward computation.

        Args:
            ob (ndarray): Current observation (not used in custom reward computation).
            action (ndarray): Action taken in this step (not used in custom reward).

        Returns:
            float: Scalar reward value. Custom reward ranges from -num_cubes (all
                cubes far from targets) to 0 (all cubes at targets). Parent class
                reward depends on its implementation.

        Note:
            The custom reward provides dense feedback about task progress by counting
            how many cubes are successfully positioned. Each successful cube adds 1
            to the base value of -num_cubes.
        """
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
