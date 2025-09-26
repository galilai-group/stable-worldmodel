# env import
import gymnasium as gym
import einops
from gymnasium import spaces
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
from typing import Tuple, Sequence, Dict, Union, Optional
import pygame
import pymunk
import numpy as np
import shapely.geometry as sg
import cv2
import skimage.transform as st
import pymunk.pygame_util
import collections
from matplotlib import cm
import torch

import stable_worldmodel as swm

# @markdown ### **Environment**
# @markdown Defines a PyMunk-based Push-T environment `PushTEnv`.
# @markdown
# @markdown **Goal**: push the gray T-block into the green area.
# @markdown
# @markdown Adapted from [Implicit Behavior Cloning](https://implicitbc.github.io/)


positive_y_is_up: bool = False
"""Make increasing values of y point upwards.

When True::

    y
    ^
    |      . (3, 3)
    |
    |   . (2, 2)
    |
    +------ > x

When False::

    +------ > x
    |
    |   . (2, 2)
    |
    |      . (3, 3)
    v
    y

"""


# def farthest_point_sampling(points: np.ndarray, n_points: int, init_idx: int):
#     """
#     Naive O(N^2)
#     """
#     assert n_points >= 1
#     chosen_points = [points[init_idx]]
#     for _ in range(n_points - 1):
#         cpoints = np.array(chosen_points)
#         all_dists = np.linalg.norm(points[:, None, :] - cpoints[None, :, :], axis=-1)
#         min_dists = all_dists.min(axis=1)
#         next_idx = np.argmax(min_dists)
#         next_pt = points[next_idx]
#         chosen_points.append(next_pt)
#     result = np.array(chosen_points)
#     return result


# class PymunkKeypointManager:
#     def __init__(
#         self,
#         local_keypoint_map: Dict[str, np.ndarray],
#         color_map: Optional[Dict[str, np.ndarray]] = None,
#     ):
#         """
#         local_keypoint_map:
#             "<attribute_name>": (N,2) floats in object local coordinate
#         """
#         if color_map is None:
#             cmap = cm.get_cmap("tab10")
#             color_map = dict()
#             for i, key in enumerate(local_keypoint_map.keys()):
#                 color_map[key] = (np.array(cmap.colors[i]) * 255).astype(np.uint8)

#         self.local_keypoint_map = local_keypoint_map
#         self.color_map = color_map

#     @property
#     def kwargs(self):
#         return {
#             "local_keypoint_map": self.local_keypoint_map,
#             "color_map": self.color_map,
#         }

#     @classmethod
#     def create_from_pusht_env(cls, env, n_block_kps=9, n_agent_kps=3, seed=0, **kwargs):
#         rng = np.random.default_rng(seed=seed)
#         local_keypoint_map = dict()
#         for name in ["block", "agent"]:
#             self = env
#             self.space = pymunk.Space()
#             if name == "agent":
#                 self.agent = obj = self.add_circle((256, 400), 15)
#                 n_kps = n_agent_kps
#             else:
#                 self.block = obj = self.add_tee((256, 300), 0)
#                 n_kps = n_block_kps

#             self.screen = pygame.Surface((512, 512))
#             self.screen.fill(pygame.Color("white"))
#             draw_options = DrawOptions(self.screen)
#             self.space.debug_draw(draw_options)
#             # pygame.display.flip()
#             img = np.uint8(pygame.surfarray.array3d(self.screen).transpose(1, 0, 2))
#             obj_mask = (img != np.array([255, 255, 255], dtype=np.uint8)).any(axis=-1)

#             tf_img_obj = cls.get_tf_img_obj(obj)
#             xy_img = np.moveaxis(np.array(np.indices((512, 512))), 0, -1)[:, :, ::-1]
#             local_coord_img = tf_img_obj.inverse(xy_img.reshape(-1, 2)).reshape(
#                 xy_img.shape
#             )
#             obj_local_coords = local_coord_img[obj_mask]

#             # furthest point sampling
#             init_idx = rng.choice(len(obj_local_coords))
#             obj_local_kps = farthest_point_sampling(obj_local_coords, n_kps, init_idx)
#             small_shift = rng.uniform(0, 1, size=obj_local_kps.shape)
#             obj_local_kps += small_shift

#             local_keypoint_map[name] = obj_local_kps

#         return cls(local_keypoint_map=local_keypoint_map, **kwargs)

#     @staticmethod
#     def get_tf_img(pose: Sequence):
#         pos = pose[:2]
#         rot = pose[2]
#         tf_img_obj = st.AffineTransform(translation=pos, rotation=rot)
#         return tf_img_obj

#     @classmethod
#     def get_tf_img_obj(cls, obj: pymunk.Body):
#         pose = tuple(obj.position) + (obj.angle,)
#         return cls.get_tf_img(pose)

#     def get_keypoints_global(
#         self, pose_map: Dict[set, Union[Sequence, pymunk.Body]], is_obj=False
#     ):
#         kp_map = dict()
#         for key, value in pose_map.items():
#             if is_obj:
#                 tf_img_obj = self.get_tf_img_obj(value)
#             else:
#                 tf_img_obj = self.get_tf_img(value)
#             kp_local = self.local_keypoint_map[key]
#             kp_global = tf_img_obj(kp_local)
#             kp_map[key] = kp_global
#         return kp_map

#     def draw_keypoints(self, img, kps_map, radius=1):
#         scale = np.array(img.shape[:2]) / np.array([512, 512])
#         for key, value in kps_map.items():
#             color = self.color_map[key].tolist()
#             coords = (value * scale).astype(np.int32)
#             for coord in coords:
#                 cv2.circle(img, coord, radius=radius, color=color, thickness=-1)
#         return img

#     def draw_keypoints_pose(self, img, pose_map, is_obj=False, **kwargs):
#         kp_map = self.get_keypoints_global(pose_map, is_obj=is_obj)
#         return self.draw_keypoints(img, kps_map=kp_map, **kwargs)


class DrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface: pygame.Surface) -> None:
        """Draw a pymunk.Space on a pygame.Surface object.

        Typical usage::

        >>> import pymunk
        >>> surface = pygame.Surface((10, 10))
        >>> space = pymunk.Space()
        >>> options = pymunk.pygame_util.DrawOptions(surface)
        >>> space.debug_draw(options)

        You can control the color of a shape by setting shape.color to the color
        you want it drawn in::

        >>> c = pymunk.Circle(None, 10)
        >>> c.color = pygame.Color("pink")

        See pygame_util.demo.py for a full example

        Since pygame uses a coordinate system where y points down (in contrast
        to many other cases), you either have to make the physics simulation
        with Pymunk also behave in that way, or flip everything when you draw.

        The easiest is probably to just make the simulation behave the same
        way as Pygame does. In that way all coordinates used are in the same
        orientation and easy to reason about::

        >>> space = pymunk.Space()
        >>> space.gravity = (0, -1000)
        >>> body = pymunk.Body()
        >>> body.position = (0, 0)  # will be positioned in the top left corner
        >>> space.debug_draw(options)

        To flip the drawing its possible to set the module property
        :py:data:`positive_y_is_up` to True. Then the pygame drawing will flip
        the simulation upside down before drawing::

        >>> positive_y_is_up = True
        >>> body = pymunk.Body()
        >>> body.position = (0, 0)
        >>> # Body will be position in bottom left corner

        :Parameters:
                surface : pygame.Surface
                    Surface that the objects will be drawn on
        """
        self.surface = surface
        super(DrawOptions, self).__init__()

    def draw_circle(
        self,
        pos: Vec2d,
        angle: float,
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p = to_pygame(pos, self.surface)

        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius), 0)
        pygame.draw.circle(
            self.surface, light_color(fill_color).as_int(), p, round(radius - 4), 0
        )

        circle_edge = pos + Vec2d(radius, 0).rotated(angle)
        p2 = to_pygame(circle_edge, self.surface)
        line_r = 2 if radius > 20 else 1
        # pygame.draw.lines(self.surface, outline_color.as_int(), False, [p, p2], line_r)

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        pygame.draw.aalines(self.surface, color.as_int(), False, [p1, p2])

    def draw_fat_segment(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r > 2:
            orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orthog[0] == 0 and orthog[1] == 0:
                return
            scale = radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
            orthog[0] = round(orthog[0] * scale)
            orthog[1] = round(orthog[1] * scale)
            points = [
                (p1[0] - orthog[0], p1[1] - orthog[1]),
                (p1[0] + orthog[0], p1[1] + orthog[1]),
                (p2[0] + orthog[0], p2[1] + orthog[1]),
                (p2[0] - orthog[0], p2[1] - orthog[1]),
            ]
            pygame.draw.polygon(self.surface, fill_color.as_int(), points)
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p1[0]), round(p1[1])),
                round(radius),
            )
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p2[0]), round(p2[1])),
                round(radius),
            )

    def draw_polygon(
        self,
        verts: Sequence[Tuple[float, float]],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        ps = [to_pygame(v, self.surface) for v in verts]
        ps += [ps[0]]

        radius = 2
        pygame.draw.polygon(self.surface, light_color(fill_color).as_int(), ps)

        if radius > 0:
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                self.draw_fat_segment(a, b, radius, fill_color, fill_color)

    def draw_dot(
        self, size: float, pos: Tuple[float, float], color: SpaceDebugColor
    ) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, color.as_int(), p, round(size), 0)


def get_mouse_pos(surface: pygame.Surface) -> Tuple[int, int]:
    """Get position of the mouse pointer in pymunk coordinates."""
    p = pygame.mouse.get_pos()
    return from_pygame(p, surface)


def to_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    """Convenience method to convert pymunk coordinates to pygame surface
    local coordinates.

    Note that in case positive_y_is_up is False, this function won't actually do
    anything except converting the point to integers.
    """
    if positive_y_is_up:
        return round(p[0]), surface.get_height() - round(p[1])
    else:
        return round(p[0]), round(p[1])


def from_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    """Convenience method to convert pygame surface local coordinates to
    pymunk coordinates
    """
    return to_pygame(p, surface)


def light_color(color: SpaceDebugColor):
    color = np.minimum(
        1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255])
    )
    color = SpaceDebugColor(r=color[0], g=color[1], b=color[2], a=color[3])
    return color


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    geom = sg.MultiPolygon(geoms)
    return geom


class PushT(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "video.frames_per_second": 10,
        "render_fps": 10,
    }
    reward_range = (0.0, 1.0)

    def __init__(
        self,
        legacy=False,
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

        self.legacy = legacy
        self.shapes = ["L", "T", "Z", "o", "square", "I", "small_tee", "+"]

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
                            low=0.5,
                            high=2,
                            init_value=1,
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
                            len(self.shapes), start=0, init_value=1
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

        self.variation_space.reset()

        ### update the variation space
        if "variation" in options:
            assert isinstance(options["variation"], Sequence), (
                "variation option must be a Sequence containing variations names to sample"
            )
            # self.update_variation(options["variation"])
            # ... sample variations

            if len(options["variation"]) == 1 and options["variation"][0] == "all":
                self.variation_space.sample()

            else:
                for var in options["variation"]:
                    try:
                        var_path = var.split(".")
                        swm.utils.get_in(self.variation_space, var_path).sample()

                    except (KeyError, TypeError):
                        raise ValueError(
                            f"Variation {var} not found in variation space"
                        )

        assert self.variation_space.check(), (
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

        self.coverage_arr = []
        state = self._get_obs()
        proprio = np.concatenate((state[:2], state[-2:]))

        observation = {"proprio": proprio, "state": state}

        info = self._get_info()
        info["max_coverage"] = 0
        info["final_coverage"] = 0
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

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = False  # coverage > self.success_threshold

        self.coverage_arr.append(coverage)

        state = self._get_obs()

        proprio = np.concatenate((state[:2], state[-2:]))
        observation = {"proprio": proprio, "state": state}

        info = self._get_info()
        info["max_coverage"] = 0
        info["final_coverage"] = self.coverage_arr[-1]
        truncated = False
        return observation, reward, done, truncated, info

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

    # def set_background(self, image):
    #     """image can be a file path, pathlib.Path, or a BytesIO"""
    #     # Load to a Surface; no display needed
    #     self._bg_raw = image
    #     self._bg_cache = None  # invalidate cache when a new image is set

    # def _get_background_for_canvas(self, canvas):
    #     """Convert/scale the raw background to match the canvas only when needed."""
    #     if getattr(self, "_bg_raw", None) is None:
    #         return None
    #     if (
    #         getattr(self, "_bg_cache", None) is not None
    #         and self._bg_cache.get_size() == canvas.get_size()
    #     ):
    #         return self._bg_cache

    #     base = (
    #         self._bg_raw.convert_alpha()
    #         if self._bg_raw.get_alpha()
    #         else self._bg_raw.convert(canvas)
    #     )
    #     scaled = pygame.transform.smoothscale(base, canvas.get_size())
    #     self._bg_cache = scaled.convert(canvas)

    #     return self._bg_cache

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
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatibility with legacy data
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block

        # Run physics to take effect
        self.space.step(self.dt)

    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2], rotation=self.goal_pose[2]
        )
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2], rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(matrix=tf_img_obj.params @ tf_obj_new.params)
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0])
            + list(tf_img_new.translation)
            + [tf_img_new.rotation]
        )
        self._set_state(new_state)
        return new_state

    def set_task_goal(self, goal):
        self.goal_pose = goal

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

        # Add agent, block, and goal zone.
        # self.agent = self.add_circle((256, 400), 15)
        # self.block = self.add_tee((256, 300), 0)
        # self.block = self.add_shape(
        #     self.shape, (256, 300), 0, color=self.color, scale=40
        # )

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
        radius = 15
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius * scale)
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
