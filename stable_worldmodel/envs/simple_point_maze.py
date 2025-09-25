from collections.abc import Sequence
import gymnasium as gym

# from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

import stable_worldmodel as swm
from loguru import logger as logging

from functools import reduce
from operator import getitem


class SimplePointMazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        max_walls=6,
        min_walls=4,
        wall_min_size=0.5,
        wall_max_size=1.5,
        render_mode=None,
        show_goal: bool = True,
    ):
        super().__init__()
        self.show_goal = show_goal
        self.width = 5.0
        self.height = 5.0
        self.render_mode = render_mode

        self.observation_space = swm.spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([self.width, self.height], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        self.action_space = swm.spaces.Box(
            low=np.array([-0.2, -0.2], dtype=np.float32),
            high=np.array([0.2, 0.2], dtype=np.float32),
            dtype=np.float32,
            shape=(2,),
        )

        #### variation space

        wall_pos_high = np.array([[self.width, self.height]], dtype=np.float32).repeat(
            max_walls, axis=0
        )
        wall_pos_low = np.array([[0.0, 0.0]], dtype=np.float32).repeat(
            max_walls, axis=0
        )

        wall_size_low = np.array(
            [[wall_min_size, wall_min_size]], dtype=np.float32
        ).repeat(max_walls, axis=0)
        wall_size_high = np.array(
            [[wall_max_size, wall_max_size]], dtype=np.float32
        ).repeat(max_walls, axis=0)

        # random init walls shape
        rng = np.random.default_rng(42)
        init_wall_shape = rng.uniform(
            low=wall_size_low, high=wall_size_high, size=(max_walls, 2)
        ).astype(np.float32)
        init_wall_positions = rng.uniform(low=1, high=3.5, size=(max_walls, 2)).astype(
            np.float32
        )

        self.variation_space = swm.spaces.Dict(
            {
                "agent": swm.spaces.Dict(
                    {
                        "color": swm.spaces.Box(
                            low=0,
                            high=255,
                            init_value=np.array([255, 0, 0], dtype=np.uint8),
                            shape=(3,),
                            dtype=np.uint8,
                        ),
                        "radius": swm.spaces.Box(
                            low=0.05,
                            high=0.5,
                            init_value=np.array(0.1, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                        "position": swm.spaces.Box(
                            low=np.array([0.0, 0.0], dtype=np.float32),
                            high=np.array([self.width, self.height], dtype=np.float32),
                            init_value=np.array([0.5, 0.5], dtype=np.float32),
                            shape=(2,),
                            dtype=np.float32,
                            constrain_fn=lambda v: not self._collides(
                                v, entity="agent"
                            ),
                        ),
                        "speed": swm.spaces.Box(
                            low=0.05,
                            high=2,
                            init_value=np.array(1.0, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                    }
                ),
                "goal": swm.spaces.Dict(
                    {
                        "color": swm.spaces.Box(
                            low=0,
                            high=255,
                            init_value=np.array([0, 255, 0], dtype=np.uint8),
                            shape=(3,),
                            dtype=np.uint8,
                        ),
                        "radius": swm.spaces.Box(
                            low=0.05,
                            high=0.5,
                            init_value=np.array(0.2, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                        "position": swm.spaces.Box(
                            low=np.array([0.0, 0.0], dtype=np.float32),
                            high=np.array([self.width, self.height], dtype=np.float32),
                            init_value=np.array([4.5, 4.5], dtype=np.float32),
                            shape=(2,),
                            dtype=np.float32,
                            constrain_fn=lambda v: not self._collides(v, entity="goal"),
                        ),
                    }
                ),
                "walls": swm.spaces.Dict(
                    {
                        "number": swm.spaces.Discrete(
                            max_walls - min_walls + 1, start=min_walls, init_value=5
                        ),
                        "color": swm.spaces.Box(
                            low=0,
                            high=255,
                            shape=(3,),
                            init_value=np.array([0, 0, 0], dtype=np.uint8),
                            dtype=np.uint8,
                        ),
                        "shape": swm.spaces.Box(
                            low=wall_size_low,
                            high=wall_size_high,
                            shape=(max_walls, 2),
                            init_value=init_wall_shape,
                            dtype=np.float32,
                        ),
                        "positions": swm.spaces.Box(
                            low=wall_pos_low,
                            high=wall_pos_high,
                            shape=(max_walls, 2),
                            init_value=init_wall_positions,
                            dtype=np.float32,
                            constrain_fn=self._check_walls,
                        ),
                    },
                    sampling_order=["number", "color", "shape", "positions"],
                ),
                "background": swm.spaces.Dict(
                    {
                        "color": swm.spaces.Box(
                            low=0,
                            high=255,
                            init_value=np.array([255, 255, 255], dtype=np.uint8),
                            shape=(3,),
                            dtype=np.uint8,
                        ),
                    }
                ),
            },
            sampling_order=["agent", "goal", "walls", "background"],
        )

        self.state = self.variation_space["agent"]["position"].value.copy()

        # need walls to check validity of default variation values
        assert self.variation_space.check(), (
            "Default variation values must be within variation space"
        )

        self._fig = None
        self._ax = None

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

        # generate goal frame
        original_state = self.variation_space.value["agent"]["position"].copy()
        self.state = self.variation_space["agent"]["position"].sample(set_value=False)
        self._goal = self.render()

        # load back original start and state
        self.state = original_state
        info = {"goal": self._goal}

        return self.state.copy(), info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        next_state = self.state + self.variation_space["agent"]["speed"].value * action
        # Check for wall collisions
        if self._collides(next_state, entity="agent"):
            next_state = self.state  # Stay in place if collision
        # Keep within bounds
        next_state = np.clip(
            next_state,
            self.observation_space.low,
            self.observation_space.high,
        )

        self.state = next_state
        self.variation_space["agent"]["position"]._value = self.state.copy()

        # Check if goal reached
        terminated = (
            np.linalg.norm(self.state - self.variation_space["goal"]["position"].value)
            < self.variation_space["goal"]["radius"].value
        ).item()
        truncated = False  # You can add a max step count if you want
        reward = 1.0 if terminated else -0.01  # Small penalty per step
        info = {"goal": self._goal}
        return self.state.copy(), reward, terminated, truncated, info

    def _collides(self, pos, walls=None, entity="agent"):
        assert entity in ["agent", "goal"], "Entity must be 'agent' or 'goal'"

        x, y = pos
        radius = self.variation_space.value[entity]["radius"]
        num_walls = self.variation_space.value["walls"]["number"]
        wall_shape = self.variation_space.value["walls"]["shape"]
        wall_positions = (
            self.variation_space.value["walls"]["positions"] if walls is None else walls
        )

        wx = wall_positions[:num_walls, 0]
        wy = wall_positions[:num_walls, 1]

        w = wall_shape[:num_walls, 0]
        h = wall_shape[:num_walls, 1]

        for x1, y1, ww, hh in zip(wx, wy, w, h):
            x2 = x1 + ww
            y2 = y1 + hh
            left, right = (x1, x2) if x1 <= x2 else (x2, x1)
            top, bottom = (y1, y2) if y1 <= y2 else (y2, y1)

            if radius <= 0:
                if left <= x <= right and top <= y <= bottom:
                    return True
            else:
                cx = np.clip(x, left, right)
                cy = np.clip(y, top, bottom)
                if (cx - x) ** 2 + (cy - y) ** 2 <= radius**2:
                    return True

        return False

    def _check_walls(self, x):
        n = self.variation_space.value["walls"]["number"]
        pos = x[:n]
        wh = self.variation_space.value["walls"]["shape"][:n]

        x, y = pos[:, 0], pos[:, 1]
        w, h = wh[:, 0], wh[:, 1]

        fits_h = np.all(self.width >= x + w)
        fits_v = np.all(self.height >= y + h)

        agent_pos = self.variation_space.value["agent"]["position"]
        goal_pos = self.variation_space.value["goal"]["position"]
        collide_agent = self._collides(agent_pos, walls=pos, entity="agent")
        collide_goal = self._collides(goal_pos, walls=pos, entity="goal")

        return bool(fits_h and fits_v) and not (collide_agent or collide_goal)

    def render(self, mode=None):
        mode = mode or self.render_mode or "human"
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(5, 5))
        self._ax.clear()
        self._ax.set_xlim(0, self.width)
        self._ax.set_ylim(0, self.height)
        self._ax.set_aspect("equal")
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._ax.set_facecolor(
            self.variation_space["background"]["color"].value / 255.0
        )

        # Draw walls
        num_walls = self.variation_space["walls"]["number"].value
        wall_shape = self.variation_space["walls"]["shape"].value[:num_walls]
        wall_positions = self.variation_space["walls"]["positions"].value[:num_walls]

        w, h = wall_shape[:, 0], wall_shape[:, 1]
        wx, wy = wall_positions[:, 0], wall_positions[:, 1]

        for x1, y1, x2, y2 in zip(wx, wy, wx + w, wy + h):
            rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                facecolor=self.variation_space["walls"]["color"].value / 255.0,
            )
            self._ax.add_patch(rect)

        # Draw goal
        if self.show_goal:
            goal_pos = self.variation_space["goal"]["position"].value
            goal_radius = self.variation_space["goal"]["radius"].value
            goal_color = self.variation_space["goal"]["color"].value
            goal = Circle(
                goal_pos, goal_radius, facecolor=goal_color / 255.0, alpha=0.5
            )
            self._ax.add_patch(goal)

        # Draw agent
        agent = Circle(
            self.state,
            self.variation_space["agent"]["radius"].value,
            facecolor=self.variation_space["agent"]["color"].value / 255.0,
        )
        self._ax.add_patch(agent)

        # # Draw start
        # start = Circle(self.start_pos, 0.1, color="blue", alpha=0.5)
        # self._ax.add_patch(start)

        self._fig.tight_layout(pad=0)
        if mode == "human":
            plt.pause(0.001)
            plt.draw()
        elif mode == "rgb_array":
            self._fig.canvas.draw()
            width, height = self._fig.canvas.get_width_height()
            img = np.frombuffer(self._fig.canvas.tostring_argb(), dtype=np.uint8)
            img = img.reshape(height, width, 4)[:, :, 1:]
            return img
        else:
            raise NotImplementedError(f"Render mode {mode} not supported.")

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
