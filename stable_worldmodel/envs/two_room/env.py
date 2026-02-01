"""TwoRoom Navigation Environment.

A simple two-room navigation environment where an agent must navigate
through a door opening to reach a target position.
"""

import math
import cv2
import gymnasium as gym
import numpy as np
import pygame
import pymunk
from gymnasium import spaces
from pymunk.vec2d import Vec2d
from stable_worldmodel import spaces as swm_spaces

from ..utils import light_color, pymunk_to_shapely

DEFAULT_VARIATIONS = (
    'agent.position',
    'target.position',
    'door.size',
    'door.position',
)


class TwoRoomEnv(gym.Env):
    """A simple navigation two-room environment.

    The environment consists of two rooms separated by a wall with door openings.
    The agent must navigate from its starting position to the target position.

    Physics:
        - Agent: Dynamic circle body
        - target: Kinematic circle (sensor, no collision)
        - Walls: Static segments (solid, blocks agent)
        - Doors: Empty gaps in walls (agent can pass if it fits)

    Rendering:
        - Layers (back to front): background, target, walls, agent
        - Agent is always rendered on top
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 10,
    }

    def __init__(
        self,
        render_size: int = 224,
        render_mode: str = 'rgb_array',
        init_value: dict | None = None,
    ):
        assert render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        # Render configuration
        self.window_size = 512
        self.border_size = 9
        self.size = self.window_size - 2 * self.border_size
        self.render_size = render_size

        # Physics configuration
        self.control_hz = self.metadata['render_fps']
        self.dt = 0.01
        self.energy_bound = 200
        self.max_door = 3
        self.max_speed = 20.0
        self.wall_pos = math.ceil(self.size / 2)
        self.max_step_norm = 2.45

        bs = self.border_size

        # Observation space
        self.observation_space = spaces.Dict(
            {
                'proprio': spaces.Box(
                    low=np.array([bs, bs]),
                    high=np.array(
                        [
                            self.size + bs,
                            self.size + bs,
                        ]
                    ),
                    dtype=np.float64,
                ),
                'state': spaces.Box(
                    low=np.array([bs, bs, bs, bs, 0, 10]),
                    high=np.array(
                        [
                            self.size + bs,
                            self.size + bs,
                            self.size + bs,
                            self.size + bs,
                            self.energy_bound,
                            self.max_speed,
                        ]
                    ),
                    dtype=np.float64,
                ),
            }
        )

        # Action space: 2D velocity
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Variation space (must be defined before constraint functions)
        self.variation_space = swm_spaces.Dict(
            {
                'agent': swm_spaces.Dict(
                    {
                        'color': swm_spaces.RGBBox(
                            init_value=np.array([255, 0, 0], dtype=np.uint8)
                        ),
                        'radius': swm_spaces.Box(
                            low=np.array([15], dtype=np.float32),
                            high=np.array([30], dtype=np.float32),
                            init_value=np.array([15], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        'position': swm_spaces.Box(
                            low=np.array([bs, bs], dtype=np.float32),
                            high=np.array(
                                [self.size + bs, self.size + bs],
                                dtype=np.float32,
                            ),
                            shape=(2,),
                            dtype=np.float32,
                            init_value=np.array(
                                [50.0, 50.0], dtype=np.float32
                            ),
                            constrain_fn=lambda x: not self.check_collide(
                                x, entity='agent'
                            ),
                        ),
                        'max_energy': swm_spaces.Discrete(
                            self.energy_bound - 50, start=50, init_value=100
                        ),
                        'speed': swm_spaces.Box(
                            low=np.array([10], dtype=np.float32),
                            high=np.array([self.max_speed], dtype=np.float32),
                            init_value=np.array([10.0], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                    },
                    sampling_order=[
                        'color',
                        'radius',
                        'position',
                        'max_energy',
                        'speed',
                    ],
                ),
                'target': swm_spaces.Dict(
                    {
                        'color': swm_spaces.RGBBox(
                            init_value=np.array([0, 255, 0], dtype=np.uint8)
                        ),
                        'radius': swm_spaces.Box(
                            low=np.array([15], dtype=np.float32),
                            high=np.array([30], dtype=np.float32),
                            init_value=np.array([15], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        'position': swm_spaces.Box(
                            low=np.array([bs, bs], dtype=np.float32),
                            high=np.array(
                                [self.size + bs, self.size + bs],
                                dtype=np.float32,
                            ),
                            shape=(2,),
                            dtype=np.float32,
                            init_value=np.array(
                                [450.0, 450.0], dtype=np.float32
                            ),
                            constrain_fn=lambda x: (
                                not self.check_collide(x, entity='target')
                                and self.check_other_room(x)
                            ),
                        ),
                    },
                    sampling_order=['color', 'radius', 'position'],
                ),
                'wall': swm_spaces.Dict(
                    {
                        'color': swm_spaces.RGBBox(
                            init_value=np.array(
                                [115, 127, 145], dtype=np.uint8
                            )
                        ),
                        'thickness': swm_spaces.Discrete(
                            25, start=9, init_value=19
                        ),
                        'axis': swm_spaces.Discrete(
                            2, init_value=1
                        ),  # 0: horizontal, 1: vertical
                        'border_color': swm_spaces.RGBBox(
                            init_value=np.array(
                                [180, 189, 204], dtype=np.uint8
                            )
                        ),
                    },
                    sampling_order=[
                        'color',
                        'border_color',
                        'thickness',
                        'axis',
                    ],
                ),
                'door': swm_spaces.Dict(
                    {
                        'color': swm_spaces.RGBBox(
                            init_value=np.array(
                                [255, 255, 255], dtype=np.uint8
                            )
                        ),
                        'number': swm_spaces.Discrete(
                            self.max_door, start=1, init_value=1
                        ),
                        'size': swm_spaces.MultiDiscrete(
                            nvec=[90] * self.max_door,
                            start=[10] * self.max_door,
                            init_value=[75] * self.max_door,
                            constrain_fn=self.check_one_door_fit,
                        ),
                        'position': swm_spaces.MultiDiscrete(
                            nvec=[self.size] * self.max_door,
                            init_value=[self.size // 2] * self.max_door,
                        ),
                    },
                    sampling_order=['color', 'number', 'size', 'position'],
                ),
                'background': swm_spaces.Dict(
                    {
                        'color': swm_spaces.RGBBox(
                            init_value=np.array(
                                [255, 255, 255], dtype=np.uint8
                            )
                        ),
                    }
                ),
            },
            sampling_order=['background', 'wall', 'agent', 'door', 'target'],
        )

        if init_value is not None:
            self.variation_space.set_init_value(init_value)

        # Pygame state
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        options = options or {}
        swm_spaces.reset_variation_space(
            self.variation_space,
            seed,
            options,
            DEFAULT_VARIATIONS,
        )

        # Initialize physics world
        self._setup()

        # Generate target image
        state = self._get_obs()
        state[:2] = options.get(
            'target_state', self.variation_space['target']['position'].value
        )
        self._set_state(state)
        self._target = self.render()

        # Set initial positions
        state = self._get_obs()
        state[:2] = options.get(
            'state', self.variation_space['agent']['position'].value
        )
        state[2:4] = options.get(
            'target_state', self.variation_space['target']['position'].value
        )
        self._set_state(state)
        proprio = np.array(state[:2])
        observation = {'proprio': proprio, 'state': state}

        info = self._get_info()
        info['fraction_of_target'] = 0.0
        info['fraction_of_agent'] = 0.0

        return observation, info

    def step(self, action):
        self.n_contact_points = 0
        n_steps = int(1 / (self.dt * self.control_hz))
        control_period = n_steps * self.dt

        # Clamp action
        action = np.asarray(action)
        action_norm = np.linalg.norm(action)
        if action_norm > self.max_step_norm:
            action = (action / action_norm) * self.max_step_norm

        velocity = action / control_period
        speed = self.variation_space['agent']['speed'].value.item()

        self.latest_action = action
        self.space.iterations = 30

        # Run physics simulation
        for _ in range(n_steps):
            self.agent.velocity = Vec2d(0, 0) + velocity * speed
            self.space.step(self.dt)

        self.energy -= 1

        # Build observation
        state = self._get_obs()
        proprio = np.array(state[:2])
        observation = {'proprio': proprio, 'state': state}

        # Calculate target overlap
        target_geom = pymunk_to_shapely(self.target, self.target.shapes)
        agent_geom = pymunk_to_shapely(self.agent, self.agent.shapes)

        intersection_area = target_geom.intersection(agent_geom).area
        target_area = target_geom.area
        agent_area = agent_geom.area

        fraction_of_target = intersection_area / target_area
        fraction_of_agent = intersection_area / agent_area

        info = self._get_info()
        info['fraction_of_target'] = fraction_of_target
        info['fraction_of_agent'] = fraction_of_agent

        terminated = (fraction_of_target >= 0.5) or (fraction_of_agent >= 0.5)
        truncated = self.energy <= 0
        reward = 1.0 if terminated else -0.01

        return observation, reward, terminated, truncated, info

    def _setup(self):
        """Initialize the physics world."""
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0

        # Build borders
        self.border_segments = self._build_borders()

        # Build wall with door gaps
        self.wall_segments = self._build_wall_segments()

        # Create agent
        agent_pos = self.variation_space['agent']['position'].value.tolist()
        agent_radius = self.variation_space['agent']['radius'].value.item()
        self.agent = self._add_circle(agent_pos, agent_radius, is_sensor=False)

        # Create target
        target_pos = self.variation_space['target']['position'].value.tolist()
        target_radius = self.variation_space['target']['radius'].value.item()
        self.target = self._add_circle(
            target_pos, target_radius, is_sensor=True
        )

        # Set energy
        self.energy = self.variation_space['agent']['max_energy'].value

        # Collision tracking
        self.space.on_collision(0, 0, post_solve=self._handle_collision)
        self.n_contact_points = 0

    def _build_borders(self):
        """Create border segments around the arena."""
        border_color = self.variation_space['wall'][
            'border_color'
        ].value.tolist()
        radius = self.border_size / 2

        # Border corner points
        corners = [
            (0, 0),
            (self.window_size, 0),
            (self.window_size, self.window_size),
            (0, self.window_size),
        ]

        segments = []
        for i in range(4):
            a = corners[i]
            b = corners[(i + 1) % 4]
            seg = self._add_segment(a, b, radius, border_color)
            segments.append(seg)

        return segments

    def _build_wall_segments(self):
        """Build wall segments with gaps for doors."""
        wall_color = self.variation_space['wall']['color'].value.tolist()
        wall_thickness = float(self.variation_space['wall']['thickness'].value)
        wall_axis = int(self.variation_space['wall']['axis'].value)
        radius = wall_thickness / 2

        door_color = self.variation_space['door']['color'].value.tolist()
        door_number = int(self.variation_space['door']['number'].value)
        door_positions = list(
            self.variation_space['door']['position'].value[:door_number]
        )
        door_sizes = list(
            self.variation_space['door']['size'].value[:door_number]
        )

        # Sort doors by position
        if door_number > 0:
            door_data = sorted(
                zip(door_positions, door_sizes), key=lambda x: x[0]
            )
            door_positions = [d[0] for d in door_data]
            door_sizes = [d[1] for d in door_data]

        # Merge overlapping doors
        merged = []
        for pos, size in zip(door_positions, door_sizes):
            pos, size = int(pos), int(size)
            if merged and pos <= merged[-1][0] + merged[-1][1]:
                # Overlapping - extend previous door
                prev_pos, prev_size = merged[-1]
                new_end = max(prev_pos + prev_size, pos + size)
                merged[-1] = (prev_pos, new_end - prev_pos)
            else:
                merged.append((pos, size))

        def pt(t):
            """Convert t position along wall axis to (x, y) coordinate."""
            # wall_axis == 1 -> vertical wall at x=wall_pos; t moves along y
            # wall_axis == 0 -> horizontal wall at y=wall_pos; t moves along x
            if wall_axis == 1:
                return (self.wall_pos, self.border_size + t)
            else:
                return (self.border_size + t, self.wall_pos)

        def clamp_span(lo, hi):
            lo = int(max(0, min(self.size, int(lo))))
            hi = int(max(0, min(self.size, int(hi))))
            return lo, hi

        wall_segments = []
        door_segments = []
        current = 0

        for door_pos, door_size in merged:
            # Clamp door span into bounds
            d0, d1 = clamp_span(door_pos, door_pos + door_size)

            # Wall span before door
            w0, w1 = clamp_span(current, d0 - 1)

            # Add wall only if it has length >= 1
            if (w1 - w0) >= 1:
                wall = self._add_segment(pt(w0), pt(w1), radius, wall_color)
                if wall is not None:
                    wall_segments.append(wall)

            # Add door only if it has length >= 1 (non-colliding)
            if (d1 - d0) >= 1:
                door = self._add_segment(
                    pt(d0), pt(d1), radius, door_color, collision=False
                )
                if door is not None:
                    door_segments.append(door)

            current = d1 + 1

        # Add last wall segment
        w0, w1 = clamp_span(current, self.size)
        if (w1 - w0) >= 1:
            last_wall = self._add_segment(pt(w0), pt(w1), radius, wall_color)
            if last_wall is not None:
                wall_segments.append(last_wall)

        # Store door segments for rendering
        self.doors = door_segments

        return wall_segments

    def _add_segment(self, a, b, thickness, color, collision=True):
        """Create a thick wall as a convex polygon.

        This creates a rectangle from point a to b with the given thickness.
        """
        a, b = Vec2d(*a), Vec2d(*b)

        # Guard against degenerate inputs
        if (b - a).length < 1e-6:
            return None

        ab = (b - a).normalized()
        perp = ab.perpendicular() * (thickness / 2)
        points = [a + perp, b + perp, b - perp, a - perp]

        shape = pymunk.Poly(self.space.static_body, points)
        shape.color = pygame.Color(*color)
        shape.sensor = not collision
        shape.friction = 0.8
        shape.elasticity = 0.0
        self.space.add(shape)
        return shape

    def _add_circle(self, position, radius, is_sensor=False):
        """Create a circle body."""
        if is_sensor:
            body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        else:
            mass = 1.0
            moment = pymunk.moment_for_circle(mass, 0, radius)
            body = pymunk.Body(mass, moment, body_type=pymunk.Body.DYNAMIC)

        body.position = position
        body.friction = 1

        shape = pymunk.Circle(body, radius)
        shape.sensor = is_sensor
        shape.friction = 0.8
        shape.elasticity = 0.0

        if not is_sensor:
            self.space.add(body, shape)

        return body

    def _set_state(self, state):
        """Set agent and target positions."""

        agent_pos = state[:2]
        target_pos = state[2:4]

        agent_pos = (
            agent_pos.tolist()
            if isinstance(agent_pos, np.ndarray)
            else agent_pos
        )

        target_pos = (
            target_pos.tolist()
            if isinstance(target_pos, np.ndarray)
            else target_pos
        )
        self.energy = state[4]

        self.agent.position = agent_pos
        self.target.position = target_pos
        self.space.step(self.dt)

    def _set_target_state(self, target_state):
        """Set only the target position."""
        target_pos = (
            target_state.tolist()
            if isinstance(target_state, np.ndarray)
            else target_state
        )
        self.target.position = target_pos[2:4]
        self.space.step(self.dt)

    def _get_obs(self):
        """Build observation array."""
        speed = self.variation_space['agent']['speed'].value.item()
        obs = (
            tuple(self.agent.position)
            + tuple(self.target.position)
            + (self.energy, speed)
        )

        return np.array(obs, dtype=np.float64)

    def _get_info(self):
        """Build info dict."""
        n_steps = int(1 / (self.dt * self.control_hz))
        n_contact_points_per_step = int(
            np.ceil(self.n_contact_points / n_steps)
        )
        return {
            'pos_agent': np.array(self.agent.position),
            'pos_target': np.array(self.target.position),
            'n_contacts': n_contact_points_per_step,
            'target_pos': self.variation_space['target']['position'].value,
            'target': self._target,
            'energy': self.energy,
            'max_energy': self.variation_space['agent']['max_energy'].value,
        }

    def _handle_collision(self, arbiter, space, data):
        """Track collision contacts."""
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def render(self):
        """Render the environment."""
        return self._render_frame(self.render_mode)

    def _render_frame(self, mode):
        """Render a single frame with explicit layer ordering."""
        if self.window is None and mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

        if self.clock is None and mode == 'human':
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))

        # Layer 1: Background
        bg_color = self.variation_space['background']['color'].value.tolist()
        canvas.fill(bg_color)

        # Layer 2: target (below walls, can be partially hidden)
        target_color = self.variation_space['target']['color'].value.tolist()
        self._draw_circle(canvas, self.target, target_color)

        # Layer 3: Walls, doors, and borders
        for segment in self.wall_segments:
            self._draw_segment(canvas, segment, rounded=False)
        for segment in self.doors:
            self._draw_segment(canvas, segment, rounded=False)
        for segment in self.border_segments:
            self._draw_segment(canvas, segment, rounded=True)

        # Layer 4: Agent (always on top)
        agent_color = self.variation_space['agent']['color'].value.tolist()
        self._draw_circle(canvas, self.agent, agent_color)

        # Convert to numpy array
        img = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
        img = cv2.resize(img, (self.render_size, self.render_size))

        if mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata['render_fps'])

        return img

    def _draw_circle(self, canvas, body, color):
        """Draw a circle with highlight effect."""
        pos = (round(body.position.x), round(body.position.y))
        radius = round(list(body.shapes)[0].radius)

        # Main circle
        pygame.draw.circle(canvas, color, pos, radius)

        # Highlight (lighter inner circle)
        highlight_color = light_color(pygame.Color(*color)).as_int()
        inner_radius = max(0, radius - 4)
        if inner_radius > 0:
            pygame.draw.circle(canvas, highlight_color, pos, inner_radius)

    def _draw_poly(self, canvas, shape):
        """Draw a polygon shape with highlight effect."""
        if shape is None:
            return

        verts = shape.get_vertices()
        if len(verts) < 3:
            return

        color = shape.color
        points = [(round(v.x), round(v.y)) for v in verts]

        # Draw main polygon
        pygame.draw.polygon(canvas, color, points)

        # Draw lighter inner polygon (highlight)
        highlight_color = light_color(pygame.Color(color)).as_int()

        # Calculate centroid and shrink vertices toward it for highlight
        cx = sum(v.x for v in verts) / len(verts)
        cy = sum(v.y for v in verts) / len(verts)
        shrink = 4
        inner_points = []
        for v in verts:
            dx = cx - v.x
            dy = cy - v.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > shrink:
                scale = shrink / dist
                inner_points.append(
                    (round(v.x + dx * scale), round(v.y + dy * scale))
                )
            else:
                inner_points.append((round(v.x), round(v.y)))

        if len(inner_points) >= 3:
            pygame.draw.polygon(canvas, highlight_color, inner_points)

    def _draw_segment(self, canvas, shape, rounded=True):
        """Draw a polygon segment (for backwards compatibility)."""
        self._draw_poly(canvas, shape)

    def close(self):
        """Clean up pygame resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed=None):
        """Set random seed."""
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
        self.random_state = np.random.RandomState(seed)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        self.variation_space.seed(seed)

    # ---- Constraint functions for variation space ----

    def check_one_door_fit(self, x):
        """Ensure at least one door is large enough for the agent to pass."""
        number = self.variation_space.value['door']['number']
        agent_radius = self.variation_space.value['agent']['radius'].item()
        for size in x[:number]:
            if size >= 2.5 * agent_radius:
                return True
        return False

    def check_other_room(self, x):
        """Check if position x is in a different room than the agent."""
        agent_pos = self.variation_space.value['agent']['position']
        wall_axis = self.variation_space.value['wall']['axis']
        wall_pos = self.wall_pos

        # pick the relevant axis: 0 = x (vertical wall), 1 = y (horizontal wall)
        i = 1 if wall_axis == 0 else 0
        return (agent_pos[i] < wall_pos and x[i] > wall_pos) or (
            agent_pos[i] > wall_pos and x[i] < wall_pos
        )

    def check_collide(self, x, entity='agent'):
        """Check if position collides with walls or borders."""
        assert entity in ['agent', 'target']
        cx, cy = x
        r = self.variation_space.value[entity]['radius']

        # collide with border
        if (cx - r) <= self.border_size or (cx + r) >= self.size:
            return True

        if (cy - r) <= self.border_size or (cy + r) >= self.size:
            return True

        # check collide with wall
        wall_axis = self.variation_space.value['wall']['axis']
        wall_pos = self.wall_pos
        wall_thickness = self.variation_space.value['wall']['thickness']

        if wall_axis == 0:
            if abs(cy - wall_pos) <= (wall_thickness / 2 + r):
                return True
        else:
            if abs(cx - wall_pos) <= (wall_thickness / 2 + r):
                return True

        return False
