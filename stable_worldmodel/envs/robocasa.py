import logging
import os
import re
import sys
import time

import gymnasium as gym
import numpy as np
import robosuite
from gymnasium import spaces
from robocasa.utils.dataset_registry import (
    MULTI_STAGE_TASK_DATASETS,
    SINGLE_STAGE_TASK_DATASETS,
)
from robocasa.utils.env_utils import create_env
from scipy.spatial.transform import Rotation as R

import stable_worldmodel as swm


BASE_ASSET_ROOT_PATH = '~/robocasa/robocasa/models/assets/objects'

os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

RCASA_CONTROLLER_INPUT_LIMS = np.array([1.0, -1])
RCASA_CONTROLLER_OUTPUT_LIMS = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0])

# Default goal sampling parameters
DEFAULT_GOAL_SAMPLING_CUBE_SIZE = (
    0.15  # meters - half-size of cube around initial EEF position
)
DEFAULT_GOAL_SAMPLING_MAX_ATTEMPTS = (
    10  # max IK attempts before using fallback
)
DEFAULT_GOAL_IK_TOLERANCE = (
    0.02  # meters - max position error for IK to be considered feasible
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


class RoboCasa(gym.Env):
    """
    RoboCasa environment wrapper compatible with stable_worldmodel.
    """

    metadata = {
        'render_modes': ['rgb_array'],
        'render_fps': 10,
    }

    def __init__(
        self,
        env=None,
        cfg={},
        env_name='PnPCounterToSink',
        camera_name='robot0_agentview_left',
        render_mode='rgb_array',
    ):
        super().__init__()

        if env is None:
            logger.info(f'Creating RoboCasa environment: {env_name}')
            env = create_env(
                env_name=env_name,
                robots='PandaOmron',
                camera_names=['robot0_agentview_left'],
                camera_widths=224,
                camera_heights=224,
                seed=42,
                render_onscreen=False,
            )

        self.env = env
        self.cfg = cfg

        self.rescale_act_droid_to_rcasa = (
            cfg.get('task_specification', {})
            .get('env', {})
            .get('rescale_act_droid_to_rcasa', False)
        )
        self.custom_task = (
            cfg.get('task_specification', {})
            .get('env', {})
            .get('custom_task', False)
        )
        self.subtask = (
            cfg.get('task_specification', {})
            .get('env', {})
            .get('subtask', None)
        )
        self.manip_only = (
            cfg.get('task_specification', {})
            .get('env', {})
            .get('manip_only', True)
        )
        self.reach_threshold = (
            cfg.get('task_specification', {})
            .get('env', {})
            .get('reach_threshold', 0.2)
        )
        self.place_threshold = (
            cfg.get('task_specification', {})
            .get('env', {})
            .get('place_threshold', 0.15)
        )

        logger.info(f'RoboCasaWrapper: {self.rescale_act_droid_to_rcasa=}')
        logger.info(f'Set {self.reach_threshold=} and {self.place_threshold=}')

        self.goal_obj_pos = None
        self.goal_state = (
            None  # Stores goal EEF state (7D: pos + euler + gripper)
        )
        self.env_name = env_name
        self.camera_name = camera_name
        self.custom_camera_name = self.camera_name
        self.camera_width = self.env.camera_widths[0]
        self.camera_height = self.env.camera_heights[0]
        self.full_action_dim = self.env.action_dim
        self.action_dim = 7 if self.manip_only else self.full_action_dim
        self.action_space = gym.spaces.Box(
            low=np.full(self.action_dim, -1.0),
            high=np.full(self.action_dim, 1.0),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                'proprio': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(7,),
                    dtype=np.float32,
                ),
            }
        )

        if self.custom_task:
            self.custom_camera_name = 'robot0_droid_agentview_left'
            self.custom_camera_pos = ([0.4, 0.4, 0.6],)
            # [0.1, 0.4, 0.8]  # Example position [x, y, z]
            self.custom_camera_quat = [0.0, -0.0, 0.6, 1.0]
            self.custom_camera_fovy = 85

        self.variation_space = swm.spaces.Dict(
            {
                # Visual/Lighting Variations
                'lighting': swm.spaces.Dict(
                    {
                        'ambient': swm.spaces.Box(
                            low=np.array([0.3], dtype=np.float32),
                            high=np.array([0.9], dtype=np.float32),
                            init_value=np.array([0.6], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        'directional': swm.spaces.Box(
                            low=np.array([0.2], dtype=np.float32),
                            high=np.array([0.8], dtype=np.float32),
                            init_value=np.array([0.5], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        'specular': swm.spaces.Box(
                            low=np.array([0.1], dtype=np.float32),
                            high=np.array([0.7], dtype=np.float32),
                            init_value=np.array([0.3], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                    }
                ),
                # Camera Variations
                'camera': swm.spaces.Dict(
                    {
                        'position': swm.spaces.Box(
                            low=np.array([0.3, 0.3, 0.5], dtype=np.float32),
                            high=np.array([0.5, 0.5, 0.8], dtype=np.float32),
                            init_value=np.array(
                                [0.4, 0.4, 0.6], dtype=np.float32
                            ),
                            shape=(3,),
                            dtype=np.float32,
                        ),
                        'orientation': swm.spaces.Box(
                            low=np.array([-0.3, -0.3, 0.3], dtype=np.float32),
                            high=np.array([0.3, 0.3, 0.9], dtype=np.float32),
                            init_value=np.array(
                                [0.0, -0.0, 0.6], dtype=np.float32
                            ),
                            shape=(3,),
                            dtype=np.float32,
                        ),
                        'fovy': swm.spaces.Box(
                            low=np.array([70], dtype=np.float32),
                            high=np.array([100], dtype=np.float32),
                            init_value=np.array([85], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                    }
                ),
                # Material Properties
                'materials': swm.spaces.Dict(
                    {
                        # Kitchen surface materials
                        'countertop_color': swm.spaces.RGBBox(
                            init_value=np.array(
                                [180, 180, 180], dtype=np.uint8
                            )
                        ),
                        'cabinet_color': swm.spaces.RGBBox(
                            init_value=np.array(
                                [139, 69, 19], dtype=np.uint8
                            )  # Brown
                        ),
                        'floor_color': swm.spaces.RGBBox(
                            init_value=np.array(
                                [210, 180, 140], dtype=np.uint8
                            )  # Tan
                        ),
                        'wall_color': swm.spaces.RGBBox(
                            init_value=np.array(
                                [245, 245, 245], dtype=np.uint8
                            )  # White
                        ),
                        # Material reflectance
                        'surface_reflectance': swm.spaces.Box(
                            low=np.array([0.1], dtype=np.float32),
                            high=np.array([0.9], dtype=np.float32),
                            init_value=np.array([0.5], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        'roughness': swm.spaces.Box(
                            low=np.array([0.0], dtype=np.float32),
                            high=np.array([1.0], dtype=np.float32),
                            init_value=np.array([0.3], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                    }
                ),
                # Object Variations
                'objects': swm.spaces.Dict(
                    {
                        # Object appearance
                        'object_colors': swm.spaces.RGBBox(
                            init_value=np.array(
                                [255, 0, 0], dtype=np.uint8
                            )  # Red default
                        ),
                        # Object scale variations
                        'object_scale': swm.spaces.Box(
                            low=np.array([0.8], dtype=np.float32),
                            high=np.array([1.2], dtype=np.float32),
                            init_value=np.array([1.0], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        # Object position noise
                        'position_noise': swm.spaces.Box(
                            low=np.array(
                                [-0.05, -0.05, 0.0], dtype=np.float32
                            ),
                            high=np.array(
                                [0.05, 0.05, 0.02], dtype=np.float32
                            ),
                            init_value=np.array(
                                [0.0, 0.0, 0.0], dtype=np.float32
                            ),
                            shape=(3,),
                            dtype=np.float32,
                        ),
                    }
                ),
                # Physics Variations
                'physics': swm.spaces.Dict(
                    {
                        'gravity': swm.spaces.Box(
                            low=np.array([-9.81], dtype=np.float32),
                            high=np.array(
                                [-9.81], dtype=np.float32
                            ),  # Keep gravity constant for realism
                            init_value=np.array([-9.81], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        # Joint friction
                        'joint_friction': swm.spaces.Box(
                            low=np.array([0.1], dtype=np.float32),
                            high=np.array([0.3], dtype=np.float32),
                            init_value=np.array([0.2], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        # Surface friction
                        'surface_friction': swm.spaces.Box(
                            low=np.array([0.5], dtype=np.float32),
                            high=np.array([1.5], dtype=np.float32),
                            init_value=np.array([1.0], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                    }
                ),
                # Noise/Disturbances
                'noise': swm.spaces.Dict(
                    {
                        # Observation noise
                        'observation_noise': swm.spaces.Box(
                            low=np.array([0.0], dtype=np.float32),
                            high=np.array([0.05], dtype=np.float32),
                            init_value=np.array([0.01], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        # Action noise
                        'action_noise': swm.spaces.Box(
                            low=np.array([0.0], dtype=np.float32),
                            high=np.array([0.1], dtype=np.float32),
                            init_value=np.array([0.02], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                    }
                ),
            }
        )

    def eef_quat_to_xyz(self, eef_quat):
        # shape (4,)
        # If your quaternion is [w, x, y, z], convert to [x, y, z, w] for scipy
        eef_quat_xyzw = np.array(
            [eef_quat[1], eef_quat[2], eef_quat[3], eef_quat[0]]
        )
        # Convert to Euler angles (xyz order, radians)
        eef_euler = R.from_quat(eef_quat_xyzw).as_euler('xyz', degrees=False)
        return eef_euler  # shape (3,)

    def gripper_2d_to_1d(self, gripper_qpos):
        """
        Convert 2D gripper position to 1D representation.
        Args:
            gripper_qpos: tensor of shape (2,) for gripper position
        Returns:
            tensor of shape (1,) for gripper state
        """
        return gripper_qpos[0:1] - gripper_qpos[1:2]

    def get_obs_proprio_succ_from_info(self, info):
        """
        Eitherway the obs part is not being used here, the only way for visual data to reach the PixelWrapper is
        via the self.render() function.
        """
        obs = {}
        # info[f'{self.camera_name}_image'] # H W 3
        eef_angle = self.eef_quat_to_xyz(
            info['robot0_eef_quat']
        )  # Convert quaternion to Euler angles
        gripper_closure = self.gripper_2d_to_1d(
            info['robot0_gripper_qpos']
        )  # Gripper position (2,) to closure (1,)
        obs['proprio'] = np.concatenate(
            [
                info[
                    'robot0_eef_pos'
                ],  # Cartesian position of the end effector (3,)
                eef_angle,  # Euler angles of the end effector (3,)
                gripper_closure,  # Gripper state (1,)
            ]
        )
        # Need to call this function to define env.obj_up_once
        # and other variables used in subtask_success()
        info['success'] = self.env._check_success()
        if self.subtask is not None:
            info = self.subtask_success(info)
        return obs, info

    def subtask_success(self, info):
        obj = self.env.objects['obj']
        obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj.name]])
        hand_pos = np.array(
            self.sim.data.body_xpos[
                self.sim.model.body_name2id(
                    self.robots[0].gripper['right'].root_body
                )
            ]
        )
        hand_obj_dist = np.linalg.norm(hand_pos - obj_pos)
        reach = hand_obj_dist < self.reach_threshold
        # We set goal_obj_pos after having reset the environment
        obj_goal_dist = (
            np.linalg.norm(self.goal_obj_pos - obj_pos)
            if self.goal_obj_pos is not None
            else -1.0
        )
        place = obj_goal_dist < self.place_threshold
        if self.subtask == 'reach-pick-place':
            success = place
        elif self.subtask == 'reach-pick':
            success = reach and self.env.obj_up_once
        elif self.subtask == 'pick-place':
            success = self.env.obj_up_once and place
        elif self.subtask == 'reach':
            success = reach
        elif self.subtask == 'pick':
            success = self.env.obj_up_once
        elif self.subtask == 'place':
            success = place
        else:
            raise ValueError(f'Unknown subtask: {self.subtask}')

        info['success'] = success
        info['obj_pos'] = obj_pos
        info['hand_pos'] = hand_pos
        info['obj_goal_dist'] = obj_goal_dist
        info['hand_obj_dist'] = hand_obj_dist
        info['obj_initial_height'] = (
            self.env.obj_initial_height
            if hasattr(self.env, 'obj_initial_height')
            else -1
        )
        info['obj_lift'] = obj_pos[2] - info['obj_initial_height']
        info['near_object'] = hand_obj_dist
        info['obj_up_once'] = (
            self.env.obj_up_once if hasattr(self.env, 'obj_up_once') else -1
        )
        return info

    def _get_goal_state(self):
        """Get the current goal state (7D: EEF pos + euler + gripper).

        Returns:
            np.ndarray: Goal state of shape (7,) or None if not set
        """
        return self.goal_state

    def _set_goal_state(self, goal_state):
        """Set the goal state for evaluation.

        Args:
            goal_state: np.ndarray of shape (7,) containing:
                - EEF position (3,)
                - EEF euler angles (3,)
                - Gripper state (1,)
        """
        if isinstance(goal_state, (list, tuple)):
            goal_state = np.array(goal_state, dtype=np.float32)
        self.goal_state = goal_state

    def _get_current_eef_pos(self):
        """Get the current end-effector position from the simulator."""
        # Get EEF position from the robot's site
        robot = self.env.robots[0]
        eef_site_id = robot.eef_site_id['right']

        # eef_site_id might be an integer ID or a string name
        if isinstance(eef_site_id, str):
            eef_site_id = self.env.sim.model.site_name2id(eef_site_id)

        return self.env.sim.data.site_xpos[eef_site_id].copy()

    def _get_current_eef_quat(self):
        """Get the current end-effector quaternion (wxyz) from the simulator."""
        robot = self.env.robots[0]
        eef_site_id = robot.eef_site_id['right']

        # eef_site_id might be an integer ID or a string name
        if isinstance(eef_site_id, str):
            eef_site_id = self.env.sim.model.site_name2id(eef_site_id)

        # MuJoCo stores rotation matrix, convert to quaternion
        xmat = self.env.sim.data.site_xmat[eef_site_id].reshape(3, 3)
        r = R.from_matrix(xmat)
        quat_xyzw = r.as_quat()  # scipy returns xyzw
        # Convert to wxyz format
        return np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        )

    def _save_sim_state(self):
        """Save the current simulator state for later restoration."""
        return self.env.sim.get_state().flatten()

    def _restore_sim_state(self, state):
        """Restore the simulator to a previously saved state."""
        self.env.sim.set_state_from_flattened(state)
        self.env.sim.forward()

    def _sample_goal_eef_position(
        self,
        initial_eef_pos,
        cube_half_size=DEFAULT_GOAL_SAMPLING_CUBE_SIZE,
        z_bias=0.0,
    ):
        """
        Sample a random EEF position in a cube around the initial position.

        Args:
            initial_eef_pos: Initial end-effector position (3,)
            cube_half_size: Half-size of the sampling cube in meters
            z_bias: Bias to add to z-coordinate (positive = upward)

        Returns:
            target_pos: Sampled target position (3,)
        """
        # Sample uniform random offset in a cube
        offset = self.np_random.uniform(
            -cube_half_size, cube_half_size, size=3
        )
        # Apply z-bias (e.g., to prefer positions above initial)
        offset[2] += z_bias
        target_pos = initial_eef_pos + offset
        return target_pos

    def _try_reach_position(
        self,
        target_pos,
        max_steps=50,
        tolerance=DEFAULT_GOAL_IK_TOLERANCE,
    ):
        """
        Attempt to move the arm to a target position using the environment's controller.

        This uses the environment's built-in IK controller by stepping with
        actions that command the EEF toward the target position.

        Args:
            target_pos: Target end-effector position (3,)
            max_steps: Maximum number of simulation steps to attempt
            tolerance: Position error tolerance in meters

        Returns:
            success: Whether the target was reached within tolerance
            final_pos: Final EEF position achieved
            error: Position error at the end
        """
        for _ in range(max_steps):
            current_pos = self._get_current_eef_pos()
            error = np.linalg.norm(target_pos - current_pos)

            if error < tolerance:
                return True, current_pos, error

            # Compute direction to target
            direction = target_pos - current_pos
            direction_norm = np.linalg.norm(direction)

            if direction_norm > 0:
                # Scale the action - use smaller steps for fine control
                step_size = min(0.05, direction_norm)  # Max 5cm per step
                action_pos = (direction / direction_norm) * step_size

                # Create full action with proper dimension
                # The environment expects full_action_dim (e.g., 12 for mobile manipulator)
                action = np.zeros(self.full_action_dim)
                action[:3] = (
                    action_pos * 10
                )  # Scale for controller input limits
                action[3:6] = 0.0  # No rotation change
                action[6] = 0.0  # Gripper unchanged
                # Remaining dimensions (e.g., base movement) stay at 0

                # Step the environment directly (bypass our step() which handles manip_only)
                self.env.step(action)

        # Final check
        final_pos = self._get_current_eef_pos()
        error = np.linalg.norm(target_pos - final_pos)
        return error < tolerance, final_pos, error

    def _sample_feasible_goal_state(
        self,
        cube_half_size=DEFAULT_GOAL_SAMPLING_CUBE_SIZE,
        max_attempts=DEFAULT_GOAL_SAMPLING_MAX_ATTEMPTS,
        tolerance=DEFAULT_GOAL_IK_TOLERANCE,
    ):
        """
        Sample a feasible goal state by finding a reachable EEF position.

        This method:
        1. Saves the current simulator state
        2. Samples random EEF positions in a cube around the current position
        3. Attempts to move the arm to each sampled position
        4. If successful, renders the goal image
        5. Restores the original state

        Args:
            cube_half_size: Half-size of the sampling cube in meters
            max_attempts: Maximum number of IK attempts before using fallback
            tolerance: Position error tolerance for IK success

        Returns:
            goal_image: Rendered image at the goal state
            goal_eef_pos: The goal EEF position (or None if using fallback)
            success: Whether a feasible goal was found
        """
        # Save current state
        saved_state = self._save_sim_state()
        initial_eef_pos = self._get_current_eef_pos()

        goal_image = None
        goal_eef_pos = None
        success = False

        for attempt in range(max_attempts):
            # Sample a target position
            target_pos = self._sample_goal_eef_position(
                initial_eef_pos,
                cube_half_size=cube_half_size,
                z_bias=0.02,  # Slight upward bias
            )

            # Restore to initial state before each attempt
            self._restore_sim_state(saved_state)

            # Try to reach the target
            reached, final_pos, error = self._try_reach_position(
                target_pos,
                tolerance=tolerance,
            )

            if reached:
                # Success! Render the goal image
                goal_image = self.render()
                goal_eef_pos = final_pos.copy()
                success = True
                logger.debug(
                    f'Goal sampling succeeded on attempt {attempt + 1}, error={error:.4f}m'
                )
                break
            else:
                logger.debug(
                    f'Goal sampling attempt {attempt + 1} failed, error={error:.4f}m'
                )

        # Restore the original state
        self._restore_sim_state(saved_state)

        if not success:
            # Fallback: use current render as goal
            logger.warning(
                f'Goal sampling failed after {max_attempts} attempts. Using current render as fallback goal.'
            )
            goal_image = self.render()

        return goal_image, goal_eef_pos, success

    def reset(self, seed=None, options=None, **kwargs):
        """
        Reset the environment and return the initial observation.

        Args:
            seed: Random seed for reproducibility
            options: Optional dict that can contain:
                - sample_goal: If True, sample a feasible goal EEF position (default: False)
                - goal_cube_size: Half-size of goal sampling cube in meters (default: 0.15)
                - goal_max_attempts: Max IK attempts for goal sampling (default: 10)
                - goal_tolerance: IK position tolerance in meters (default: 0.02)
        """
        # Handle seed
        if seed is not None:
            self.seed(seed)
        self._np_random_seed = getattr(self, '_seed', None)

        info = self.env.reset()
        obs, info = self.get_obs_proprio_succ_from_info(info)

        # Parse options for goal sampling
        options = options or {}
        sample_goal = options.get('sample_goal', False)

        if sample_goal:
            # Sample a feasible goal state via IK
            goal_cube_size = options.get(
                'goal_cube_size', DEFAULT_GOAL_SAMPLING_CUBE_SIZE
            )
            goal_max_attempts = options.get(
                'goal_max_attempts', DEFAULT_GOAL_SAMPLING_MAX_ATTEMPTS
            )
            goal_tolerance = options.get(
                'goal_tolerance', DEFAULT_GOAL_IK_TOLERANCE
            )

            goal_image, goal_eef_pos, success = (
                self._sample_feasible_goal_state(
                    cube_half_size=goal_cube_size,
                    max_attempts=goal_max_attempts,
                    tolerance=goal_tolerance,
                )
            )
            self._goal = goal_image
            self._goal_eef_pos = goal_eef_pos
            self._goal_sampling_success = success
        else:
            # Fallback: use current render as goal (placeholder)
            self._goal = self.render()
            self._goal_eef_pos = None
            self._goal_sampling_success = False
            goal_eef_pos = None
            success = False

        # Always include these keys in info for consistency with StackedWrapper
        info['goal'] = self._goal
        info['goal_eef_pos'] = self._goal_eef_pos
        info['goal_sampling_success'] = self._goal_sampling_success
        # Include goal_state in info (will be None initially, set via _set_goal_state or from dataset)
        info['goal_state'] = self.goal_state
        return obs, info

    def step(self, action):
        """
        Perform a step in the environment.
        action: np array of shape (action_dim,)
        """
        if self.manip_only:
            full_action = np.zeros(self.full_action_dim)
            full_action[:7] = action
        else:
            full_action = action

        scaled_action = full_action.copy()
        if self.rescale_act_droid_to_rcasa:
            scaled_action[:7] = (
                full_action[:7]
                * RCASA_CONTROLLER_INPUT_LIMS[0]
                / RCASA_CONTROLLER_OUTPUT_LIMS
            )

        info, reward, done, _ = self.env.step(scaled_action)
        obs, info = self.get_obs_proprio_succ_from_info(info)
        # Add goal-related keys to info (same as what was set in reset)
        info['goal'] = self._goal
        info['goal_eef_pos'] = self._goal_eef_pos
        info['goal_sampling_success'] = getattr(
            self, '_goal_sampling_success', False
        )
        # Include goal_state in step info as well
        info['goal_state'] = self.goal_state
        if info['success']:
            logger.info('RoboCasaWrapper: Task success detected in step()')
        return obs, reward, None, done, info

    def render(self, *args, **kwargs):
        """
        Render the environment using the specified camera.
        Returns: H W 3
        Making a deepcopy is essential to avoid race conditions or corrupted images
        when the underlying simulator updates the visual buffer asynchronously
        """
        if (
            self.custom_camera_name
            in self.env.sim.model._camera_name2id.keys()
        ):
            camera_to_use = self.custom_camera_name
        else:
            camera_to_use = self.camera_name

        result = self.env.sim.render(
            height=self.camera_height,
            width=self.camera_width,
            camera_name=camera_to_use,
        ).copy()
        if camera_to_use != 'robot0_rightview':
            result = result[::-1]  # flip vertically
        else:
            # flip horizontally
            result = result[:, ::-1]
        return result

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def all_tasks(self):
        """
        Return all tasks available in the RoboCasa environment.
        """
        return (
            list(SINGLE_STAGE_TASK_DATASETS.keys())
            + list(MULTI_STAGE_TASK_DATASETS.keys())
            + ['PnPCounterTop']
        )

    def update_env(self, env_info):
        pass

    def prepare(self, seed, init_state, env_info=None):
        """
        Inspired from robocasa/robocasa/utils/robomimic/robomimic_env_wrapper.py
        And updated with run_on_jimmy_mac branch of robocasa-murp/robocasa/scripts/playback_utils.py::reset_to()
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        prep_start_time = time.time()
        self.seed(seed)
        model_xml = env_info.get('model_xml', None)
        ep_meta = env_info.get('ep_meta', None)

        if self.custom_task:
            # Modify the XML to add the custom camera
            import xml.etree.ElementTree as ET

            tree = ET.ElementTree(ET.fromstring(model_xml))
            camera_container = tree.find(".//body[@name='base0_support']")
            # for child in camera_container:
            #     logger.info(f"- {child.tag} with attributes: {child.attrib}")
            # worldbody = tree.find(".//worldbody")

            # Add the custom camera
            # camera_elem = ET.SubElement(worldbody, "camera")
            camera_elem = ET.SubElement(camera_container, 'camera')
            camera_elem.set('name', self.custom_camera_name)
            camera_elem.set('pos', ' '.join(map(str, self.custom_camera_pos)))
            camera_elem.set(
                'quat', ' '.join(map(str, self.custom_camera_quat))
            )
            camera_elem.set('fovy', str(self.custom_camera_fovy))
            camera_elem.set('mode', 'fixed')

            # Convert the modified XML back to a string
            model_xml = ET.tostring(tree.getroot(), encoding='unicode')
        # First handle model reset if model_xml is provided
        if model_xml is not None:
            # Set episode metadata if provided
            if ep_meta is not None:
                # filter xml file to make sure asset paths do not point to jimmyyang path
                # like '/Users/jimmytyyang/research/robot-skills-sim/robocasa-murp/robocasa/models/assets/objects/aigen_objs/boxed_food/boxed_food_4/model.xml'
                ep_meta['object_cfgs'] = update_mjcf_paths(
                    ep_meta['object_cfgs']
                )
                # Once filtere, prepare env for reset with this xml
                if hasattr(self.env, 'set_attrs_from_ep_meta'):
                    self.env.set_attrs_from_ep_meta(ep_meta)
                elif hasattr(self.env, 'set_ep_meta'):
                    self.env.set_ep_meta(ep_meta)

            # Reset the environment
            obs, info = self.reset()

            # Process the model XML based on robosuite version
            # try:
            logger.info('Resetting from provided model XML')
            robosuite_version_id = int(robosuite.__version__.split('.')[1])
            if robosuite_version_id <= 3:
                from robosuite.utils.mjcf_utils import postprocess_model_xml

                xml = postprocess_model_xml(model_xml)
            else:
                # v1.4 and above use the class-based edit_model_xml function
                xml = self.env.edit_model_xml(model_xml)
            xml = path_change(xml)
            # Reset from XML string
            self.env.reset_from_xml_string(xml)
            self.env.sim.reset()
            logger.info('Finished resetting from provided model XML')
            # except Exception as e:
            #     logger.info(f"Warning: Failed to reset from model XML: {e}")
        else:
            # Otherwise use standard reset
            obs, info = self.reset()

        try:
            self.env.sim.set_state_from_flattened(init_state)
            self.env.sim.forward()

            # Update state as needed
            if hasattr(self.env, 'update_sites'):
                # older versions of environment had update_sites function
                self.env.update_sites()
            if hasattr(self.env, 'update_state'):
                # later versions renamed this to update_state
                self.env.update_state()

            # Get updated observation
            if hasattr(self.env, '_get_observation'):
                obs = self.env._get_observation()
            elif hasattr(self.env, '_get_observations'):
                obs = self.env._get_observations(force_update=True)
        except Exception as e:
            logger.info(f'Warning: Failed to set simulator state: {e}')
        logger.info(
            f'robocasa env.prepare() took {time.time() - prep_start_time:.2f} seconds'
        )
        return obs, info


def update_mjcf_paths(object_cfgs):
    """
    Update mjcf_path in object_cfgs by replacing src path with target path.

    Args:
        object_cfgs (list): list of object configuration dicts containing 'info' with 'mjcf_path'.
        src (str): source path substring to replace.
        target (str): target path substring to replace with.

    Returns:
        list: Updated object_cfgs with modified mjcf_path.
    """
    for i, object_cfg in enumerate(object_cfgs):
        path = object_cfg['info']['mjcf_path']
        models_index = path.find('objects')
        relative_path = path[
            models_index:
        ]  # e.g. 'models/assets/objects/aigen_objs/apple/apple_5/model.xml'
        full_local_path = os.path.join(
            BASE_ASSET_ROOT_PATH, relative_path[len('objects/') :]
        )
        object_cfgs[i]['info']['mjcf_path'] = full_local_path
    return object_cfgs


def path_change(xml_string):
    """
    Fix absolute file paths in the MJCF XML by replacing them with local paths
    rooted at BASE_ASSET_ROOT_PATH.
    """

    def replace_path(match):
        original_path = match.group(1)
        model_index = original_path.find('objects/')
        if model_index == -1:
            return f'file="{original_path}"'

        relative_path = original_path[model_index + len('objects/') :]
        new_path = os.path.join(BASE_ASSET_ROOT_PATH, relative_path)
        new_path = os.path.normpath(new_path)

        return f'file="{new_path}"'

    updated_xml = re.sub(r'file="([^"]+)"', replace_path, xml_string)
    return updated_xml
