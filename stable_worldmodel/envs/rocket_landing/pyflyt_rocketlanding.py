from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, Literal

import numpy as np
import pybullet as p
import PyFlyt
from gymnasium.spaces import Box
from PyFlyt.gym_envs.rocket_envs.rocket_base_env import RocketBaseEnv

import stable_worldmodel as swm


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

        """VARIATION SPACE"""
        self.variation_space = swm.spaces.Dict(
            {
                "rocket": swm.spaces.Dict(
                    {
                        "body_color": swm.spaces.RGBBox(
                            init_value=np.array([255, 204, 0], dtype=np.uint8)  # yellow
                        ),
                        "fin_color": swm.spaces.RGBBox(
                            init_value=np.array([51, 51, 51], dtype=np.uint8)  # grey
                        ),
                        "leg_color": swm.spaces.RGBBox(
                            init_value=np.array([0, 0, 0], dtype=np.uint8)  # black
                        ),
                        "booster_color": swm.spaces.RGBBox(
                            init_value=np.array([51, 51, 51], dtype=np.uint8)  # grey
                        ),
                    }
                ),
                "pad": swm.spaces.Dict(
                    {
                        "color": swm.spaces.RGBBox(
                            init_value=np.array([200, 200, 200], dtype=np.uint8)  # light grey
                        ),
                    }
                ),
                "environment": swm.spaces.Dict(
                    {
                        "sky_color": swm.spaces.RGBBox(
                            init_value=np.array([135, 206, 235], dtype=np.uint8)  # sky blue
                        ),
                        "start_height_ratio": swm.spaces.Box(
                            low=0.7,
                            high=0.95,
                            init_value=0.9,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "start_horizontal_offset": swm.spaces.Box(
                            low=-2.0,
                            high=2.0,
                            init_value=np.array([0.0, 0.0], dtype=np.float32),
                            shape=(2,),
                            dtype=np.float32,
                        ),
                        "start_tilt": swm.spaces.Box(
                            low=-0.2,
                            high=0.2,
                            init_value=np.array([0.0, 0.0], dtype=np.float32),
                            shape=(2,),
                            dtype=np.float32,
                        ),
                    }
                ),
            },
            sampling_order=["environment", "rocket", "pad"],
        )

        # Store original parameters
        self.ceiling = ceiling
        self.original_start_pos = np.array([[0.0, 0.0, ceiling * 0.9]])

        # Track modified URDF paths
        self.modified_rocket_urdf = None
        self.modified_pad_urdf = None
        self._spawn_offset_limit = 2.0  # meters
        self._entry_speed_limit = -30.0  # m/s (downwards)

    def _modify_rocket_urdf(self) -> str:
        """Modify rocket URDF with current variation colors.

        Returns:
            Path to the modified URDF file.
        """
        # Find the original rocket URDF
        pyflyt_dir = os.path.dirname(os.path.realpath(PyFlyt.__file__))
        original_urdf = os.path.join(pyflyt_dir, "models/vehicles/rocket/rocket.urdf")

        # Parse the URDF
        tree = ET.parse(original_urdf)
        root = tree.getroot()

        # Get colors from variation space (convert from 0-255 to 0-1)
        body_color = self.variation_space["rocket"]["body_color"].value / 255.0
        fin_color = self.variation_space["rocket"]["fin_color"].value / 255.0
        leg_color = self.variation_space["rocket"]["leg_color"].value / 255.0
        # booster_color = self.variation_space["rocket"]["booster_color"].value / 255.0

        # Update material colors
        for material in root.findall(".//material[@name='yellow']"):
            color = material.find("color")
            color.set("rgba", f"{body_color[0]:.3f} {body_color[1]:.3f} {body_color[2]:.3f} 1.0")

        for material in root.findall(".//material[@name='grey']"):
            color = material.find("color")
            # Use fin color for grey materials
            color.set("rgba", f"{fin_color[0]:.3f} {fin_color[1]:.3f} {fin_color[2]:.3f} 1.0")

        for material in root.findall(".//material[@name='black']"):
            color = material.find("color")
            color.set("rgba", f"{leg_color[0]:.3f} {leg_color[1]:.3f} {leg_color[2]:.3f} 1.0")

        # Write to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
        tree.write(temp_file.name)
        temp_file.close()

        return temp_file.name

    def _modify_pad_urdf(self) -> str:
        """Modify landing pad URDF with current variation colors.

        Returns:
            Path to the modified URDF file.
        """
        # Parse the original landing pad URDF
        tree = ET.parse(self.targ_obj_dir)
        root = tree.getroot()

        # Get color from variation space (convert from 0-255 to 0-1)
        pad_color = self.variation_space["pad"]["color"].value / 255.0

        # Add material if it doesn't exist
        material = root.find(".//material[@name='pad_material']")
        if material is None:
            material = ET.Element("material", name="pad_material")
            color_elem = ET.SubElement(material, "color")
            color_elem.set("rgba", f"{pad_color[0]:.3f} {pad_color[1]:.3f} {pad_color[2]:.3f} 1.0")
            root.insert(0, material)
        else:
            color = material.find("color")
            color.set("rgba", f"{pad_color[0]:.3f} {pad_color[1]:.3f} {pad_color[2]:.3f} 1.0")

        # Apply material to visual
        visual = root.find(".//visual")
        if visual is not None:
            mat_ref = visual.find("material")
            if mat_ref is None:
                mat_ref = ET.SubElement(visual, "material")
            mat_ref.set("name", "pad_material")

        # Write to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
        tree.write(temp_file.name)
        temp_file.close()

        return temp_file.name

    def reset(self, *, seed: None | int = None, options: None | dict[str, Any] = None) -> tuple[np.ndarray, dict]:
        """Resets the environment.

        Args:
            seed: int
            options: dict with optional keys:
                - 'variation': list of variation names to sample (e.g., ['rocket.body_color', 'pad.color'])
                  Use ['all'] to sample all variations
                - 'randomize_drop': whether to add random initial velocities
                - 'accelerate_drop': whether to add downward velocity

        """
        if options is None:
            options = {}
        options = dict(options)
        options.setdefault("randomize_drop", False)
        options.setdefault("accelerate_drop", True)

        # Handle variations
        self.variation_space.seed(seed)
        self.variation_space.reset()

        variation_options = list(options.get("variation", []))
        if variation_options:
            from collections.abc import Sequence

            if not isinstance(variation_options, Sequence):
                raise ValueError("variation option must be a Sequence containing variation names to sample")

            self.variation_space.update(variation_options)

        variation_controls_offset = any(
            opt == "all" or opt == "environment" or opt.startswith("environment.start_horizontal_offset")
            for opt in variation_options
        )

        # Apply start position variations
        start_height_ratio = self.variation_space["environment"]["start_height_ratio"].value
        start_offset = self.variation_space["environment"]["start_horizontal_offset"].value
        start_tilt = self.variation_space["environment"]["start_tilt"].value

        # Update start position and orientation based on variations
        self.start_pos = np.array(
            [[start_offset[0], start_offset[1], self.ceiling * start_height_ratio]], dtype=np.float64
        )
        self.start_orn = np.array([[start_tilt[0], start_tilt[1], 0.0]], dtype=np.float64)

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

        # Limit initial displacement and entry speed
        rocket_id = self.env.drones[0].Id
        limit = self._spawn_offset_limit
        if options["randomize_drop"] and not variation_controls_offset:
            spawn_offset = self.np_random.uniform(-limit, limit, size=2)
        else:
            spawn_offset = np.clip(start_offset, -limit, limit)

        base_pos, base_orn = p.getBasePositionAndOrientation(rocket_id, physicsClientId=self.env._client)
        base_pos = np.array(base_pos, dtype=np.float64)
        base_pos[:2] = spawn_offset
        p.resetBasePositionAndOrientation(
            rocket_id,
            base_pos,
            base_orn,
            physicsClientId=self.env._client,
        )

        if options["accelerate_drop"]:
            lin_vel, ang_vel = p.getBaseVelocity(rocket_id, physicsClientId=self.env._client)
            lin_vel = np.array(lin_vel, dtype=np.float64)
            ang_vel = np.array(ang_vel, dtype=np.float64)
            lin_vel[2] = self._entry_speed_limit
            self.env.resetBaseVelocity(rocket_id, lin_vel.tolist(), ang_vel.tolist())

        self.compute_state()

        # Apply color variations to rocket and pad
        if variation_options and (
            "all" in variation_options
            or any("rocket" in v for v in variation_options)
            or any("pad" in v for v in variation_options)
        ):
            # Modify and load landing pad with variations
            if self.modified_pad_urdf:
                try:
                    os.unlink(self.modified_pad_urdf)
                except Exception:
                    pass
            self.modified_pad_urdf = self._modify_pad_urdf()
            pad_urdf_path = self.modified_pad_urdf
        else:
            pad_urdf_path = self.targ_obj_dir

        # Load the landing pad
        self.landing_pad_id = self.env.loadURDF(
            pad_urdf_path,
            basePosition=np.array([0.0, 0.0, 0.1]),
            useFixedBase=True,
        )

        super().end_reset(seed, options)

        # Apply rocket color variations using changeVisualShape
        if variation_options and ("all" in variation_options or any("rocket" in v for v in variation_options)):
            rocket_id = self.env.drones[0].Id
            body_color = self.variation_space["rocket"]["body_color"].value / 255.0
            fin_color = self.variation_space["rocket"]["fin_color"].value / 255.0
            leg_color = self.variation_space["rocket"]["leg_color"].value / 255.0
            booster_color = self.variation_space["rocket"]["booster_color"].value / 255.0

            # Get number of links in the rocket
            num_joints = p.getNumJoints(rocket_id, physicsClientId=self.env._client)

            # Change colors for each link based on joint names
            for i in range(-1, num_joints):
                if i == -1:
                    # Base link (main rocket body)
                    p.changeVisualShape(
                        rocket_id, i, rgbaColor=list(body_color) + [1.0], physicsClientId=self.env._client
                    )
                else:
                    joint_info = p.getJointInfo(rocket_id, i, physicsClientId=self.env._client)
                    link_name = joint_info[12].decode("utf-8")  # link name

                    if "fin" in link_name.lower():
                        p.changeVisualShape(
                            rocket_id, i, rgbaColor=list(fin_color) + [1.0], physicsClientId=self.env._client
                        )
                    elif "leg" in link_name.lower():
                        p.changeVisualShape(
                            rocket_id, i, rgbaColor=list(leg_color) + [1.0], physicsClientId=self.env._client
                        )
                    elif "booster" in link_name.lower():
                        p.changeVisualShape(
                            rocket_id, i, rgbaColor=list(booster_color) + [1.0], physicsClientId=self.env._client
                        )


        if self.render_mode is not None:
            init_state_id = p.saveState(physicsClientId=self.env._client)
            landed_position = [0.0, 0.0, 1.5]  # slightly above pad
            landed_orientation = p.getQuaternionFromEuler([0.0, 0.0, 0.0])  # upright
            p.resetBasePositionAndOrientation(
                self.env.drones[0].Id,
                landed_position,
                landed_orientation,
                physicsClientId=self.env._client,
            )
            self.current_goal = self.render()
            p.restoreState(stateId=init_state_id, physicsClientId=self.env._client)
            p.removeState(stateUniqueId=init_state_id, physicsClientId=self.env._client)
        else:
            height, width = self.render_resolution
            self.current_goal = np.zeros((height, width, 3), dtype=np.uint8)

        # Add goal to info dict
        self.info["goal"] = self.current_goal

        state = self.state.copy()
        info = dict(self.info)
        return state, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step the environment.

        Args:
            action: the action to take

        Returns:
            state, reward, terminated, truncated, info
        """
        state, reward, terminated, truncated, info = super().step(action)
        info["goal"] = self.current_goal
        return state, reward, terminated, truncated, dict(info)

    def render(self, mode: str = "rgb_array"):
        return super().render()[..., :3]  # discard alpha channel if present

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


        if np.linalg.norm(self.previous_ang_vel) > 10.0 or np.linalg.norm(self.previous_lin_vel) > 5.0:
            self.termination |= True
            self.info["fatal_collision"] = True
            return


        if (
            np.linalg.norm(self.previous_ang_vel) < 5.0
            and np.linalg.norm(self.previous_lin_vel) < 4.0
            and np.linalg.norm(self.ang_pos[:2]) < 0.6
            and np.linalg.norm(self.lin_pos[:2]) < 12.0
            and self.lin_pos[2] < 3.0
        ):
            self.truncation |= True
            self.info["env_complete"] = True
            self.reward += 3.0
            return
