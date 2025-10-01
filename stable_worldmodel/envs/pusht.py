from typing import Sequence

import cv2
import gymnasium as gym
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from gymnasium import spaces
from pymunk.vec2d import Vec2d

import stable_worldmodel as swm
from stable_worldmodel.envs.utils import light_color, to_pygame

from .utils import DrawOptions, pymunk_to_shapely


class PushT(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "video.frames_per_second": 10,
        "render_fps": 10,
    }
    reward_range = (0.0, 1.0)

    def __init__(
        self,
        block_cog=None,
        damping=None,
        render_action=False,
        resolution=224,
        with_target=True,
        render_mode="rgb_array",
    ):
        self._seed = None
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = resolution

        # physics
        self.control_hz = self.metadata["render_fps"]
        self.k_p, self.k_v = 100, 20
        self.dt = 0.01

        self.shapes = ["o", "L", "T", "Z", "square", "I", "small_tee", "+"]

        self.observation_space = spaces.Dict(
            {
                "proprio": spaces.Box(
                    low=np.array([0, 0, 0, 0]),
                    high=np.array([ws, ws, ws, ws]),
                    dtype=np.float64,
                ),
                "state": spaces.Box(
                    low=np.array([0, 0, 0, 0, 0, 0, 0]),
                    high=np.array([ws, ws, ws, ws, np.pi * 2, ws, ws]),
                    dtype=np.float64,
                ),
            }
        )

        # positional goal for agent
        self.action_space = spaces.Box(low=0, high=ws, shape=(2,), dtype=np.float32)

        self.variation_space = swm.spaces.Dict(
            {
                "agent": swm.spaces.Dict(
                    {
                        # "shape": swm.spaces.Categorical(
                        #     categories=["circle", "square", "triangle"],   SHOULD IMPLEMENT THIS
                        #     init_value="circle",
                        # ),
                        "color": swm.spaces.RGBBox(
                            init_value=np.array(
                                pygame.Color("RoyalBlue")[:3], dtype=np.uint8
                            )
                        ),
                        "scale": swm.spaces.Box(
                            low=20,
                            high=60,
                            init_value=40,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "shape": swm.spaces.Discrete(
                            len(self.shapes), start=0, init_value=3
                        ),
                        "angle": swm.spaces.Box(
                            low=-2 * np.pi,
                            high=2 * np.pi,
                            init_value=0.0,
                            shape=(),
                            dtype=np.float64,
                        ),
                        "start_position": swm.spaces.Box(
                            low=50,
                            high=450,
                            init_value=np.array((256, 400), dtype=np.float64),
                            shape=(2,),
                            dtype=np.float64,
                        ),
                        "velocity": swm.spaces.Box(
                            low=0,
                            high=ws,
                            init_value=np.array((0.0, 0.0), dtype=np.float64),
                            shape=(2,),
                            dtype=np.float64,
                        ),
                    }
                ),
                "block": swm.spaces.Dict(
                    {
                        "color": swm.spaces.RGBBox(
                            init_value=np.array(
                                pygame.Color("LightSlateGray")[:3], dtype=np.uint8
                            )
                        ),
                        "scale": swm.spaces.Box(
                            low=20,
                            high=60,
                            init_value=40,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "shape": swm.spaces.Discrete(
                            len(self.shapes) - 1, start=1, init_value=2
                        ),
                        "angle": swm.spaces.Box(
                            low=-2 * np.pi,
                            high=2 * np.pi,
                            init_value=0.0,
                            shape=(),
                            dtype=np.float64,
                        ),
                        "start_position": swm.spaces.Box(
                            low=100,
                            high=400,
                            init_value=np.array((400, 100), dtype=np.float64),
                            shape=(2,),
                            dtype=np.float64,
                        ),
                    }
                ),
                "goal": swm.spaces.Dict(
                    {
                        "color": swm.spaces.RGBBox(
                            init_value=np.array(
                                pygame.Color("LightGreen")[:3], dtype=np.uint8
                            )
                        ),
                        "scale": swm.spaces.Box(
                            low=20,
                            high=60,
                            init_value=40,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "angle": swm.spaces.Box(
                            low=-2 * np.pi,
                            high=2 * np.pi,
                            init_value=np.pi / 4,
                            shape=(),
                            dtype=np.float64,
                        ),
                        "position": swm.spaces.Box(
                            low=50,
                            high=450,
                            init_value=np.array([256, 256], dtype=np.float64),
                            shape=(2,),
                            dtype=np.float64,
                        ),
                    }
                ),
                "background": swm.spaces.Dict(
                    {
                        "color": swm.spaces.RGBBox(
                            init_value=np.array(
                                np.array([255, 255, 255], dtype=np.uint8)
                            )
                        ),
                    }
                ),
            },
            sampling_order=["background", "goal", "block", "agent"],
        )

        # TODO ADD CONSTRAINT TO NOT SAMPLE OVERLAPPING START POSITIONS (block and agent)

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.render_buffer = None
        self.latest_action = None

        self.with_target = with_target
        self.coverage_arr = []

    def reset(self, seed=None, options=None):
        self.seed(seed)

        ### update variation space
        options = options or {}

        if hasattr(self, "variation_space"):
            self.variation_space.seed(seed)

        options = options or {}

        self.variation_space.reset()

        if "variation" in options:
            assert isinstance(options["variation"], Sequence), (
                "variation option must be a Sequence containing variations names to sample"
            )

            if len(options["variation"]) == 1 and options["variation"][0] == "all":
                self.variation_space.sample()

            else:
                self.variation_space.update(set(options["variation"]))

        assert self.variation_space.check(debug=True), (
            "Variation values must be within variation space!"
        )

        ### setup pymunk space
        self._setup()

        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        ### get the state
        goal_state = np.concatenate(
            [
                self.variation_space["agent"]["start_position"]
                .sample(set_value=False)
                .tolist(),
                self.variation_space["block"]["start_position"]
                .sample(set_value=False)
                .tolist(),
                [self.variation_space["block"]["angle"].sample(set_value=False)],
                self.variation_space["agent"]["velocity"].value.tolist(),
            ]
        )

        ### generate goal
        self.goal_state = goal_state
        self._set_state(goal_state)
        self._goal = self.render()

        # restore original pos
        state = np.concatenate(
            [
                self.variation_space["agent"]["start_position"].value.tolist(),
                self.variation_space["block"]["start_position"].value.tolist(),
                [self.variation_space["block"]["angle"].value],
                self.variation_space["agent"]["velocity"].value.tolist(),
            ]
        )

        self._set_state(state)

        #### OBS
        state = self._get_obs()
        proprio = np.concatenate((state[:2], state[-2:]))

        observation = {"proprio": proprio, "state": state}
        info = self._get_info()

        return observation, info

    def step(self, action):
        self.n_contact_points = 0
        n_steps = int(1 / (self.dt * self.control_hz))

        self.latest_action = action
        for _ in range(n_steps):
            # Step PD control.
            acceleration = self.k_p * (action - self.agent.position) + self.k_v * (
                Vec2d(0, 0) - self.agent.velocity
            )
            self.agent.velocity += acceleration * self.dt

            # Step physics.
            self.space.step(self.dt)

        # make the observation
        state = self._get_obs()
        proprio = np.concatenate((state[:2], state[-2:]))
        observation = {"proprio": proprio, "state": state}

        # collect info
        info = self._get_info()

        # compute reward and termination
        terminated, distance = self.eval_state(self.goal_state, state)
        reward = -distance  # the closer the better

        truncated = False
        return observation, reward, terminated, truncated, info

    def eval_state(self, goal_state, cur_state):
        # success if position difference is < 20, and angle difference < np.pi/9
        pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])
        angle_diff = np.abs(goal_state[4] - cur_state[4])
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        success = pos_diff < 20 and angle_diff < np.pi / 9
        state_dist = np.linalg.norm(goal_state - cur_state)
        return success, state_dist

    def render(self):
        return self._render_frame(self.render_mode)

    def _get_obs(self):
        obs = (
            tuple(self.agent.position)
            + tuple(self.block.position)
            + (self.block.angle % (2 * np.pi),)
            + tuple(self.agent.velocity)
        )

        return np.array(obs, dtype=np.float64)

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = int(1 / self.dt * self.control_hz)
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            "pos_agent": np.array(self.agent.position),
            "vel_agent": np.array(self.agent.velocity),
            "block_pose": np.array(list(self.block.position) + [self.block.angle]),
            "goal_pose": self.goal_pose,
            "n_contacts": n_contact_points_per_step,
            "goal": self._goal,
        }
        return info

    def _render_frame(self, mode):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.variation_space["background"]["color"].value)

        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            if isinstance(shape, pymunk.Circle):
                center_pg = pymunk.pygame_util.to_pygame(
                    goal_body.local_to_world(shape.offset), draw_options.surface
                )
                pygame.draw.circle(
                    canvas,
                    self.variation_space["goal"]["color"].value,
                    (int(center_pg[0]), int(center_pg[1])),
                    int(shape.radius),
                )

            else:
                goal_points = [
                    pymunk.pygame_util.to_pygame(
                        goal_body.local_to_world(v), draw_options.surface
                    )
                    for v in shape.get_vertices()
                ]
                goal_points += [goal_points[0]]
                pygame.draw.polygon(
                    canvas,
                    self.variation_space["goal"]["color"].value,
                    goal_points,
                )

        # change agent color
        self._set_body_color(
            self.agent, self.variation_space["agent"]["color"].value.tolist()
        )

        # change block color
        self._set_body_color(
            self.block, self.variation_space["block"]["color"].value.tolist()
        )

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is already ticked during in step for "human"

        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8 / 96 * self.render_size)
                thickness = int(1 / 96 * self.render_size)
                cv2.drawMarker(
                    img,
                    coord,
                    color=(255, 0, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size,
                    thickness=thickness,
                )
        return img

    def _set_body_color(self, body, color):
        color = pygame.Color(*color) if not isinstance(color, pygame.Color) else color
        for s in body.shapes:
            s.color = color

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
        self.random_state = np.random.RandomState(seed)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        self.variation_space.seed(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        vel_block = tuple(state[-2:])
        self.agent.velocity = vel_block
        self.agent.position = pos_agent
        self.block.angle = rot_block
        self.block.position = pos_block

        # Run physics to take effect
        self.space.step(self.dt)

    def _setup(self):
        ## create the space with physics
        self.space = pymunk.Space()
        self.space.gravity = 0, 0  # TODO add physics support
        self.space.damping = 0
        self.render_buffer = list()

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]

        self.space.add(*walls)

        #### agent ####

        agent_params = dict(
            position=self.variation_space["agent"]["start_position"].value.tolist(),
            angle=self.variation_space["agent"]["angle"].value,
            scale=self.variation_space["agent"]["scale"].value,
            color=self.variation_space["agent"]["color"].value.tolist(),
            shape=self.shapes[self.variation_space["agent"]["shape"].value],
        )

        self.agent = self.add_shape(**agent_params)

        #### block ####

        block_params = dict(
            position=self.variation_space["block"]["start_position"].value.tolist(),
            angle=self.variation_space["block"]["angle"].value,
            scale=self.variation_space["block"]["scale"].value,
            color=self.variation_space["block"]["color"].value.tolist(),
            shape=self.shapes[self.variation_space["block"]["shape"].value],
        )

        self.block = self.add_shape(**block_params)

        self.goal_pose = np.concatenate(
            [
                self.variation_space["goal"]["position"].value,
                [self.variation_space["goal"]["angle"].value],
            ]
        )

        # Add collision handling
        self.space.on_collision(0, 0, post_solve=self._handle_collision)
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95  # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color(
            "LightGray"
        )  # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(
        self,
        position,
        angle=0,
        scale=1,
        color="RoyalBlue",
    ):
        base_radius = 0.375
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, base_radius * scale)
        shape.color = pygame.Color(color)
        self.space.add(body, shape)
        return body

    def add_box(
        self, position, height, width, color="LightSlateGray", scale=1, angle=0
    ):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height * scale, width * scale))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height * scale, width * scale))
        shape.color = pygame.Color(color)
        self.space.add(body, shape)
        return body

    def add_tee(
        self,
        position,
        angle,
        scale=30,
        color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        scale = 30
        mass = 1
        length = 4
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity
        ) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body

    def add_small_tee(
        self,
        position,
        angle,
        scale=30,
        color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        vertices1 = [
            (-3 * scale / 2, scale),
            (3 * scale / 2, scale),
            (3 * scale / 2, 0),
            (-3 * scale / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, 2 * scale),
            (scale / 2, 2 * scale),
            (scale / 2, scale),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity
        ) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body

    def add_plus(
        self,
        position,
        angle,
        scale=30,
        color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        vertices1 = [
            (-3 * scale / 2, scale / 2),
            (3 * scale / 2, scale / 2),
            (3 * scale / 2, -scale / 2),
            (-3 * scale / 2, -scale / 2),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale / 2),
            (-scale / 2, 3 * scale / 2),
            (scale / 2, scale / 2),
            (scale / 2, 3 * scale / 2),
        ]
        vertices3 = [
            (-scale / 2, -scale / 2),
            (-scale / 2, -3 * scale / 2),
            (scale / 2, -scale / 2),
            (scale / 2, -3 * scale / 2),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        inertia3 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2 + inertia3)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape3 = pymunk.Poly(body, vertices3)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape3.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        shape3.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (
            shape1.center_of_gravity
            + shape2.center_of_gravity
            + shape3.center_of_gravity
        ) / 3
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2, shape3)
        return body

    def add_L(
        self,
        position,
        angle,
        scale=30,
        color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        length = 2
        vertices1 = [
            (0, 0),
            (0, scale * length),
            (scale * length / 2, scale * length),
            (scale * length / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (0, 0),
            (scale * length, 0),
            (scale * length, -scale * length / 2),
            (0, -scale * length / 2),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity
        ) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body

    def add_Z(
        self,
        position,
        angle,
        scale=30,
        color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        length = 2
        vertices1 = [
            (0, 0),
            (0, length * scale / 2),
            (length * scale, length * scale / 2),
            (length * scale, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-length * scale / 2, 0),
            (length * scale / 2, 0),
            (length * scale / 2, -length * scale / 2),
            (-length * scale / 2, -length * scale / 2),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity
        ) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body

    def add_square(
        self,
        position,
        angle,
        scale=30,
        color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        vertices1 = [
            (-scale, -scale),
            (-scale, scale),
            (scale, scale),
            (scale, -scale),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1)
        shape1 = pymunk.Poly(body, vertices1)
        shape1.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = shape1.center_of_gravity
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1)
        return body

    def add_I(
        self,
        position,
        angle,
        scale=30,
        color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        vertices1 = [
            (-scale / 2, -scale * 2),
            (-scale / 2, scale * 2),
            (scale / 2, scale * 2),
            (scale / 2, -scale * 2),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1)
        shape1 = pymunk.Poly(body, vertices1)
        shape1.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = shape1.center_of_gravity
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1)
        return body

    def add_shape(self, shape, *args, **kwargs):
        # Dispatch method based on the 'shape' parameter

        if shape == "L":
            return self.add_L(*args, **kwargs)
        elif shape == "T":
            return self.add_tee(*args, **kwargs)
        elif shape == "Z":
            return self.add_Z(*args, **kwargs)
        elif shape == "o":
            return self.add_circle(*args, **kwargs)
        elif shape == "square":
            return self.add_square(*args, **kwargs)
        elif shape == "I":
            return self.add_I(*args, **kwargs)
        elif shape == "small_tee":
            return self.add_small_tee(*args, **kwargs)
        if shape == "+":
            return self.add_plus(*args, **kwargs)
        else:
            raise ValueError(f"Unknown shape type: {shape}")
