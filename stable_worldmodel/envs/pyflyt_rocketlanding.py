from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np
import pybullet as p
import PyFlyt
from gymnasium.spaces import Box
from PyFlyt.gym_envs.rocket_envs.rocket_base_env import RocketBaseEnv


class RocketLandingEnv(RocketBaseEnv):
    """Rocket Landing Environment.

    Actions are finlet_x, finlet_y, finlet_roll, booster ignition, throttle, booster gimbal x, booster gimbal y
    The goal is to land the rocket on the landing pad.

    Args:
        sparse_reward (bool): whether to use sparse rewards or not.
        ceiling (float): the absolute ceiling of the flying area.
        max_displacement (float): the maximum horizontal distance the rocket can go.
        max_duration_seconds (float): maximum simulation time of the environment.
        angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
        agent_hz (int): looprate of the agent to environment interaction.
        render_mode (None | Literal["human", "rgb_array"]): render_mode
        render_resolution (tuple[int, int]): render_resolution.

    """

    def __init__(
        self,
        sparse_reward: bool = False,
        ceiling: float = 500.0,
        max_displacement: float = 200.0,
        max_duration_seconds: float = 30.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 40,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
    ):
        """__init__.

        Args:
            sparse_reward (bool): whether to use sparse rewards or not.
            ceiling (float): the absolute ceiling of the flying area.
            max_displacement (float): the maximum horizontal distance the rocket can go.
            max_duration_seconds (float): maximum simulation time of the environment.
            angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
            agent_hz (int): looprate of the agent to environment interaction.
            render_mode (None | Literal["human", "rgb_array"]): render_mode
            render_resolution (tuple[int, int]): render_resolution.

        """
        super().__init__(
            start_pos=np.array([[0.0, 0.0, ceiling * 0.9]]),
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            ceiling=ceiling,
            max_displacement=max_displacement,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        """GYMNASIUM STUFF"""
        # the space is the standard space + pad touch indicator
        self.observation_space = Box(
            low=np.array([*self.combined_space.low, 0.0]),
            high=np.array([*self.combined_space.high, 1.0]),
            dtype=np.float64,
        )

        # the landing pad
        # file_dir = os.path.dirname(os.path.realpath(__file__))
        # self.targ_obj_dir = os.path.join(file_dir, "../../models/landing_pad.urdf")

        pyflyt_dir = os.path.dirname(os.path.realpath(PyFlyt.__file__))
        self.targ_obj_dir = os.path.join(pyflyt_dir, "models/landing_pad.urdf")

        """CONSTANTS"""
        self.sparse_reward = sparse_reward

    def reset(self, *, seed: None | int = None, options: None | dict[str, Any] = None) -> tuple[np.ndarray, dict]:
        """Resets the environment.

        Args:
            seed: int
            options: None

        """
        if options is None:
            options = {
                "randomize_drop": True,
                "accelerate_drop": True,
            }

        super().begin_reset(
            seed=seed,
            options=options,
            drone_options={"starting_fuel_ratio": 0.05},
        )

        # reset the tracked parameters
        self.landing_pad_contact = 0.0
        self.ang_vel = np.zeros((3,))
        self.lin_vel = np.zeros((3,))
        self.lin_pos = np.zeros((3,))
        self.ground_lin_vel = np.zeros((3,))

        self.previous_ang_vel = np.zeros((3,))
        self.previous_lin_vel = np.zeros((3,))
        self.previous_lin_pos = np.zeros((3,))
        self.previous_ground_lin_vel = np.zeros((3,))

        # randomly generate the target landing location
        self.landing_pad_id = self.env.loadURDF(
            self.targ_obj_dir,
            basePosition=np.array([0.0, 0.0, 0.1]),
            useFixedBase=True,
        )

        super().end_reset(seed, options)

        return self.state, self.info

    def compute_state(self) -> None:
        """Computes the state of the current timestep.

        This returns the observation.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3/4 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - previous_action (vector of 4 values)
        - auxiliary information (vector of 4 values)
        """
        # update the previous values to current values
        self.previous_ang_vel = self.ang_vel.copy()
        self.previous_lin_vel = self.lin_vel.copy()
        self.previous_lin_pos = self.lin_pos.copy()
        self.previous_ground_lin_vel = self.ground_lin_vel.copy()

        # update current values
        (
            self.ang_vel,
            self.ang_pos,
            self.lin_vel,
            self.lin_pos,
            quaternion,
        ) = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        # compute rotation matrices for converting things
        rotation = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)

        # compute ground velocity for reward computation later
        self.ground_lin_vel = np.matmul(self.lin_vel, rotation.T)

        # combine everything
        if self.angle_representation == 0:
            self.state = np.concatenate(
                [
                    self.ang_vel,
                    self.ang_pos,
                    self.lin_vel,
                    self.lin_pos,
                    self.action,
                    aux_state,
                    np.array([self.landing_pad_contact]),
                ],
                axis=-1,
            )
        elif self.angle_representation == 1:
            self.state = np.concatenate(
                [
                    self.ang_vel,
                    quaternion,
                    self.lin_vel,
                    self.lin_pos,
                    self.action,
                    aux_state,
                    np.array([self.landing_pad_contact]),
                ],
                axis=-1,
            )

    def compute_term_trunc_reward(self) -> None:
        """Computes the termination, truncation, and reward of the current timestep."""
        super().compute_base_term_trunc_reward(collision_ignore_mask=[self.env.drones[0].Id, self.landing_pad_id])

        # compute reward
        if not self.sparse_reward:
            # progress to the pad
            lateral_progress = float(  # noqa
                np.linalg.norm(self.previous_lin_pos[:2]) - np.linalg.norm(self.lin_pos[:2])
            )
            vertical_progress = float(self.previous_lin_pos[-1] - self.lin_pos[-1])

            # absolute distances to the pad
            lateral_distance = np.linalg.norm(self.lin_pos[:2]) + 0.1  # noqa

            # deceleration as long as we're still falling
            # (x+1)/(e^x)
            deceleration_progress = (
                (self.ground_lin_vel[-1] - self.previous_ground_lin_vel[-1] + 1.0)
                # scale reward to height, lower height more deceleration is better
                / np.exp(self.lin_pos[-1])
                # bonus if still descending, penalty if started to ascend
                * (1.0 if (self.ground_lin_vel[-1] < 0.0) else -1.0)
            )

            # dictionarize reward components for debugging

            # composite reward together
            self.reward += (
                -0.3  # negative offset to discourage staying in the air
                + (0.3 / lateral_distance)  # reward for staying over landing pad
                + (10.0 * lateral_progress)  # reward for making progress to landing pad
                + (0.2 * vertical_progress)  # reward for descending
                + (4.0 * deceleration_progress)  # reward for decelerating
                - (1.0 * abs(self.ang_vel[-1]))  # minimize spinning
                - (1.0 * np.linalg.norm(self.ang_pos[:2]))  # minimize aggressive angles
            )

        # check if we touched the landing pad
        if self.env.contact_array[self.env.drones[0].Id, self.landing_pad_id]:
            self.landing_pad_contact = 1.0

            # the reward minus collision speed
            self.reward += 5.0 - (0.3 * abs(self.ground_lin_vel[-1]))
        else:
            self.landing_pad_contact = 0.0
            return

        # if collision has more than 0.35 rad/s angular velocity, we dead
        # truthfully, if collision has more than 0.55 m/s linear acceleration, we dead
        # number taken from here:
        # https://cosmosmagazine.com/space/launch-land-repeat-reusable-rockets-explained/
        # but doing so is kinda impossible for RL, so I've lessened the requirement to 1.0
        if np.linalg.norm(self.previous_ang_vel) > 0.35 or np.linalg.norm(self.previous_lin_vel) > 1.0:
            self.termination |= True
            self.info["fatal_collision"] = True
            return

        # if our both velocities are less than 0.02 m/s and we upright, we LANDED!
        if (
            np.linalg.norm(self.previous_ang_vel) < 0.02
            and np.linalg.norm(self.previous_lin_vel) < 0.02
            and np.linalg.norm(self.ang_pos[:2]) < 0.1
        ):
            self.truncation |= True
            self.info["env_complete"] = True
            self.reward += 3.0
            return
