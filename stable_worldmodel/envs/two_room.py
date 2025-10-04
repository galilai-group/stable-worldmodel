import math
from collections.abc import Sequence

import cv2
import gymnasium as gym
import numpy as np
import pygame
import pymunk
from gymnasium import spaces
from pymunk.vec2d import Vec2d

import stable_worldmodel as swm

from .utils import DrawOptions, light_color, pymunk_to_shapely, to_pygame


class TwoRoomEnv(gym.Env):
    """A simple navigation two-room environment."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        render_size=224,
        render_mode="rgb_array",
    ):
        # gym
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # render
        self.window_size = 512
        self.border_size = bs = 9
        self.energy_bound = 200
        self.size = self.window_size - 2 * self.border_size
        self.render_size = render_size

        # physics
        self.control_hz = self.metadata["render_fps"]
        self.dt = 0.01

        # attributes
        self.max_door = 3
        self.max_speed = 20.0
        self.wall_pos = math.ceil(self.size / 2)
        self.max_step_norm = 2.45

        self.observation_space = spaces.Dict(
            {
                "proprio": spaces.Box(
                    low=np.array([bs, bs, 0, 10]),
                    high=np.array(2 * [self.size] + [self.energy_bound, self.max_speed]),
                    dtype=np.float64,
                ),
                "state": spaces.Box(
                    low=np.array([bs, bs, bs, bs, 50, 0.5]),
                    high=np.array(4 * [self.size] + [self.energy_bound, self.max_speed]),
                    dtype=np.float64,
                ),
            }
        )

        # gym spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # variation space
        self.variation_space = swm.spaces.Dict(
            {
                "agent": swm.spaces.Dict(
                    {
                        "color": swm.spaces.RGBBox(init_value=np.array([255, 0, 0], dtype=np.uint8)),
                        "radius": swm.spaces.Box(
                            low=np.array([15], dtype=np.float32),
                            high=np.array([30], dtype=np.float32),
                            init_value=np.array([15], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        "position": swm.spaces.Box(
                            low=np.array([bs, bs], dtype=np.float32),
                            high=np.array([self.size, self.size], dtype=np.float32),
                            shape=(2,),
                            dtype=np.float32,
                            init_value=np.array([50.0, 50.0], dtype=np.float32),
                            constrain_fn=lambda x: not self.check_collide(x, entity="agent"),
                        ),
                        "max_energy": swm.spaces.Discrete(self.energy_bound - 50, start=50, init_value=100),
                        "speed": swm.spaces.Box(
                            low=np.array([10], dtype=np.float32),
                            high=np.array([self.max_speed], dtype=np.float32),
                            init_value=np.array([10.0], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                    },
                    sampling_order=[
                        "color",
                        "radius",
                        "position",
                        "max_energy",
                        "speed",
                    ],
                ),
                "goal": swm.spaces.Dict(
                    {
                        "color": swm.spaces.RGBBox(init_value=np.array([0, 255, 0], dtype=np.uint8)),
                        "radius": swm.spaces.Box(
                            low=np.array([15], dtype=np.float32),
                            high=np.array([30], dtype=np.float32),
                            init_value=np.array([15], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        # constrain it should be in the other room and at n_steps max
                        "position": swm.spaces.Box(
                            low=np.array([bs, bs], dtype=np.float32),
                            high=np.array([self.size, self.size], dtype=np.float32),
                            shape=(2,),
                            dtype=np.float32,
                            init_value=np.array([450.0, 450.0], dtype=np.float32),
                            constrain_fn=lambda x: not self.check_collide(x, entity="goal")
                            and self.check_other_room(x),
                        ),
                    },
                    sampling_order=["color", "radius", "position"],
                ),
                "wall": swm.spaces.Dict(
                    {
                        "color": swm.spaces.RGBBox(init_value=np.array([115, 127, 145], dtype=np.uint8)),
                        "thickness": swm.spaces.Discrete(25, start=9, init_value=19),
                        # 0: horizontal, 1: vertical
                        "axis": swm.spaces.Discrete(2, init_value=1),
                        # "position": swm.spaces.Discrete(
                        #     self.size,
                        #     init_value=self.size // 2,
                        # ),
                        "border_color": swm.spaces.RGBBox(init_value=np.array([180, 189, 204], dtype=np.uint8)),
                    },
                    sampling_order=["color", "border_color", "thickness", "axis"],
                ),
                "door": swm.spaces.Dict(
                    {
                        "color": swm.spaces.RGBBox(init_value=np.array([255, 255, 255], dtype=np.uint8)),
                        "number": swm.spaces.Discrete(self.max_door, start=1, init_value=1),
                        ## add constraint so that doors do not overlap?
                        "size": swm.spaces.MultiDiscrete(
                            nvec=[50] * self.max_door,
                            start=[35] * self.max_door,
                            init_value=[75] * self.max_door,
                            constrain_fn=self.check_one_door_fit,
                        ),
                        "position": swm.spaces.MultiDiscrete(
                            nvec=[self.size] * self.max_door,
                            init_value=[self.size // 2] * self.max_door,
                        ),
                    },
                    sampling_order=["color", "number", "size", "position"],
                ),
                "background": swm.spaces.Dict(
                    {
                        "color": swm.spaces.RGBBox(init_value=np.array([255, 255, 255], dtype=np.uint8)),
                    }
                ),
            },
            sampling_order=["background", "wall", "agent", "door", "goal"],
        )

        self.window = None
        self.clock = None
        self.screen = None

        return

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

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

        assert self.variation_space.check(debug=True), "Variation values must be within variation space!"

        self._setup()

        # generate goal
        goal_state = self.variation_space["goal"]["position"].value
        self._set_state(np.concatenate([goal_state, goal_state]))
        self._goal = self.render()

        # restore original state
        agent_pos = self.variation_space["agent"]["position"].value
        goal_pos = self.variation_space["goal"]["position"].value
        self._set_state(np.concatenate([agent_pos, goal_pos]))

        # generate observation
        state = self._get_obs()
        proprio = np.concatenate((state[:2], state[-2:]))
        observation = {"proprio": proprio, "state": state}

        info = self._get_info()
        info["fraction_of_goal"] = 0.0
        info["fraction_of_agent"] = 0.0
        return observation, info

    def step(self, action):
        self.n_contact_points = 0
        n_steps = int(1 / (self.dt * self.control_hz))
        control_period = n_steps * self.dt

        action_norm = np.linalg.norm(action)
        if action_norm > self.max_step_norm:
            # action is a numPy array
            action = (action / action_norm) * self.max_step_norm

        velocity = action / control_period

        speed = self.variation_space["agent"]["speed"].value.item()

        self.latest_action = action
        for _ in range(n_steps):
            self.agent.velocity = Vec2d(0, 0) + velocity * speed
            self.space.step(self.dt)

        self.energy -= 1  # TODO energy proportional to action norm?

        state = self._get_obs()
        proprio = np.concatenate((state[:2], state[-2:]))

        observation = {
            "proprio": proprio,
            "state": state,
        }

        info = self._get_info()

        ### check termination condition

        goal_geom = pymunk_to_shapely(self.goal, self.goal.shapes)
        agent_geom = pymunk_to_shapely(self.agent, self.agent.shapes)

        intersection_area = goal_geom.intersection(agent_geom).area
        goal_area = goal_geom.area
        agent_area = agent_geom.area

        fraction_of_goal = intersection_area / goal_area
        fraction_of_agent = intersection_area / agent_area

        info["fraction_of_goal"] = fraction_of_goal
        info["fraction_of_agent"] = fraction_of_agent

        terminated = (
            fraction_of_goal >= 0.5  # at least 50% of goal covered
            or fraction_of_agent >= 0.5  # at least 50% of agent inside
        )

        truncated = self.energy <= 0
        reward = 1.0 if terminated else -0.01

        return observation, reward, terminated, truncated, info

    # def add_circle(
    #     self,
    #     position,
    #     radius,
    #     color,
    # ):
    #     body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
    #     body.position = position
    #     body.friction = 1
    #     shape = pymunk.Circle(body, radius)
    #     shape.color = pygame.Color(color)
    #     self.space.add(body, shape)
    #     return body

    def add_circle(self, position, radius, color, *, is_goal=False):
        if not is_goal:
            mass = 1.0
            moment = pymunk.moment_for_circle(mass, 0, radius)
            body = pymunk.Body(mass, moment, body_type=pymunk.Body.DYNAMIC)
        else:
            body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)

        body.position = position
        body.friction = 1

        shape = pymunk.Circle(body, radius)
        shape.sensor = is_goal
        shape.color = pygame.Color(color)
        shape.friction = 0.8
        shape.elasticity = 0.0

        if not is_goal:
            self.space.add(body, shape)

        return body

    def _add_segment(self, a, b, size, color, collision=True):
        a, b = Vec2d(*a), Vec2d(*b)
        ab = (b - a).normalized()
        perp = ab.perpendicular() * (size / 2)
        points = [a + perp, b + perp, b - perp, a - perp]
        shape = pymunk.Poly(self.space.static_body, points)
        shape.color = pygame.Color(color)
        shape.sensor = not collision
        shape.z_order = 1
        return shape

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.render_buffer = []

        # -- wall and doors
        wall_color = self.variation_space["wall"]["color"].value.tolist()
        wall_thickness = self.variation_space["wall"]["thickness"].value
        wall_axis = self.variation_space["wall"]["axis"].value

        door_number = self.variation_space["door"]["number"].value
        door_positions = self.variation_space["door"]["position"].value[:door_number]
        door_sizes = self.variation_space["door"]["size"].value[:door_number]
        door_color = self.variation_space["door"]["color"].value.tolist()

        door_positions, door_sizes = zip(*sorted(zip(door_positions, door_sizes), key=lambda x: x[0]))

        # if door overlaps, merge them
        merged_positions = []
        merged_sizes = []
        current_pos = door_positions[0]
        current_size = door_sizes[0]
        for pos, size in zip(door_positions[1:], door_sizes[1:]):
            if pos <= current_pos + current_size:
                # overlap
                new_end = max(current_pos + current_size, pos + size)
                current_size = new_end - current_pos
            else:
                merged_positions.append(current_pos)
                merged_sizes.append(current_size)
                current_pos = pos
                current_size = size

        def pt(t):
            return (self.wall_pos, self.border_size + t) if wall_axis == 1 else (self.border_size + t, self.wall_pos)

        wall_segments = []
        door_segments = []
        current = 0

        for pos, size in zip(door_positions, door_sizes):
            wall_span = (current, pos - 1)
            door_span = (
                pos,
                pos + size,
            )

            door = self._add_segment(
                pt(door_span[0]),
                pt(door_span[1]),
                wall_thickness,
                door_color,
                collision=False,
            )
            wall = self._add_segment(pt(wall_span[0]), pt(wall_span[1]), wall_thickness, wall_color)

            door_segments.append(door)
            wall_segments.append(wall)
            current = door_span[1] + 1

        # add last wall segment
        last_wall = self._add_segment(pt(current), pt(self.size), wall_thickness, wall_color)
        wall_segments.append(last_wall)

        self.doors = door_segments
        self.space.add(*wall_segments)

        # -- border
        border_dict = {
            "bottom": ((0, 0), (self.window_size - 1, 0)),
            "left": ((0, 0), (0, self.window_size)),
            "right": ((self.window_size, 0), (self.window_size - 1, self.window_size)),
            "top": ((0, self.window_size), (self.window_size, self.window_size)),
        }

        border_color = self.variation_space["wall"]["border_color"].value.tolist()
        border = [self._add_segment(a, b, self.border_size, border_color) for (a, b) in border_dict.values()]
        self.space.add(*border)

        # TODO add wall and doors

        # consider the whole wall and split it into segments to create the doors?
        # assert the total size is width of the wall
        # to make door traversable, remove friction (shape.sensor = True)

        # -- agent
        agent_pos = self.variation_space["agent"]["position"].value.tolist()
        agent_radius = self.variation_space["agent"]["radius"].value.item()
        agent_color = self.variation_space["agent"]["color"].value.tolist()

        self.agent = self.add_circle(agent_pos, agent_radius, agent_color)

        # -- goal
        goal_pos = self.variation_space["goal"]["position"].value.tolist()
        goal_radius = self.variation_space["goal"]["radius"].value.item()
        goal_color = self.variation_space["goal"]["color"].value.tolist()

        self.goal = self.add_circle(goal_pos, goal_radius, goal_color, is_goal=True)

        # -- energy
        self.energy = self.variation_space["agent"]["max_energy"].value

        # add collision handler
        self.space.on_collision(0, 0, post_solve=self._handle_collision)
        self.n_contact_points = 0

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()

        pos_agent = state[:2]
        pos_goal = state[2:4]
        # energy = state[-1]

        self.agent.position = pos_agent
        self.goal.position = pos_goal

        self.space.step(self.dt)

    def _get_obs(self):
        speed = self.variation_space["agent"]["speed"].value.item()
        obs = tuple(self.agent.position) + tuple(self.goal.position) + (self.energy, speed)
        return np.array(obs, dtype=np.float64)

    def _get_info(self):
        n_steps = int(1 / self.dt * self.control_hz)
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            "pos_agent": np.array(self.agent.position),
            "pos_goal": np.array(self.goal.position),
            "n_contacts": n_contact_points_per_step,
            "goal_pos": self.variation_space["goal"]["position"].value,
            "goal": self._goal,
            "energy": self.energy,
            "max_energy": self.variation_space["agent"]["max_energy"].value,
        }
        return info

    def _set_body_color(self, body, color):
        color = pygame.Color(*color) if not isinstance(color, pygame.Color) else color
        for s in body.shapes:
            s.color = color

    def render(self):
        return self._render_frame(self.render_mode)

    def _get_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

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

        self._set_body_color(self.goal, self.variation_space["goal"]["color"].value.tolist())

        # draw doors
        for door in self.doors:
            door_points = [
                pymunk.pygame_util.to_pygame(door.body.local_to_world(v), draw_options.surface)
                for v in door.get_vertices()
            ]
            door_points.append(door_points[0])  # close shape
            pygame.draw.polygon(canvas, door.color, door_points)

        # draw goal
        for shape in self.goal.shapes:
            p = to_pygame(self.goal.position, draw_options.surface)
            pygame.draw.circle(canvas, shape.color, p, round(shape.radius), 0)
            pygame.draw.circle(
                canvas,
                light_color(shape.color).as_int(),
                p,
                round(shape.radius - 4),
                0,
            )

        self._set_body_color(self.agent, self.variation_space["agent"]["color"].value.tolist())

        self.space.debug_draw(draw_options)

        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        img = cv2.resize(img, (self.render_size, self.render_size))

        return img

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
        self.random_state = np.random.RandomState(seed)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        self.variation_space.seed(seed)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    #### constraint functions for variation space ####
    def check_one_door_fit(self, x):
        number = self.variation_space.value["door"]["number"]
        agent_radius = self.variation_space.value["agent"]["radius"].item()
        for size in x[:number]:
            if size >= 2.5 * agent_radius:
                return True
        return False

    def check_other_room(self, x):
        agent_pos = self.variation_space.value["agent"]["position"]
        wall_axis = self.variation_space.value["wall"]["axis"]
        wall_pos = self.wall_pos

        # pick the relevant axis: 0 = x (vertical wall), 1 = y (horizontal wall)
        i = 1 if wall_axis == 0 else 0
        return (agent_pos[i] < wall_pos and x[i] > wall_pos) or (agent_pos[i] > wall_pos and x[i] < wall_pos)

    def check_collide(self, x, entity="agent"):
        assert entity in ["agent", "goal"]
        cx, cy = x
        r = self.variation_space.value[entity]["radius"]

        # collide with border
        if (cx - r) <= self.border_size or (cx + r) >= self.size:
            return True

        if (cy - r) <= self.border_size or (cy + r) >= self.size:
            return True

        # check collide with wall
        wall_axis = self.variation_space.value["wall"]["axis"]
        wall_pos = self.wall_pos
        wall_thickness = self.variation_space.value["wall"]["thickness"]

        if wall_axis == 0:
            if abs(cy - wall_pos) <= (wall_thickness / 2 + r):
                return True

        else:
            if abs(cx - wall_pos) <= (wall_thickness / 2 + r):
                return True

        return False


# if __name__ == "__main__":
#     env = TwoRoomEnv()
#     obs = env.reset(options={"variation": ["all"]})
#     img = env.render()
#     plt.imshow(img)
#     plt.axis("off")
#     plt.savefig("test.png")
