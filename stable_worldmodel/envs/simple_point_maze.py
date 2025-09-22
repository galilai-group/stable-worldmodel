import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

class SimplePointMazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        n_walls=5,
        wall_min_size=0.5,
        wall_max_size=1.5,
        seed=None,
        render_mode=None,
        show_goal: bool = True,
    ):
        super().__init__()
        self.show_goal = show_goal
        self.width = 5.0
        self.height = 5.0
        self.rng = np.random.default_rng(seed)
        self.render_mode = render_mode

        # Use Dict space for easy extension (e.g., adding "pixels" later)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([self.width, self.height], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-0.2, -0.2], dtype=np.float32),
            high=np.array([0.2, 0.2], dtype=np.float32),
            dtype=np.float32,
            shape=(2,),
        )

        #### variation space
        self.variation_space = spaces.Dict(
            {
                "agent": spaces.Dict(
                    {
                        "color": spaces.Box(
                            low=0, high=255, shape=(3,), dtype=np.uint8
                        ),
                        "radius": spaces.Box(
                            low=0.05, high=0.5, shape=(), dtype=np.float32
                        ),
                        "start_position": spaces.Box(
                            low=np.array([0.0, 0.0], dtype=np.float32),
                            high=np.array([self.width, self.height], dtype=np.float32),
                            shape=(2,),
                            dtype=np.float32,
                        ),
                    }
                ),
                "goal": spaces.Dict(
                    {
                        "color": spaces.Box(
                            low=0, high=255, shape=(3,), dtype=np.uint8
                        ),
                        "radius": spaces.Box(
                            low=0.05, high=0.5, shape=(), dtype=np.float32
                        ),
                        "position": spaces.Box(
                            low=np.array([0.0, 0.0], dtype=np.float32),
                            high=np.array([self.width, self.height], dtype=np.float32),
                            shape=(2,),
                            dtype=np.float32,
                        ),
                    }
                ),
                "physics": spaces.Dict(
                    {
                        "speed": spaces.Box(
                            low=0.05, high=2, shape=(), dtype=np.float32
                        ),
                    }
                ),
                "env": spaces.Dict(
                    {
                        "n_walls": spaces.Discrete(10),  # 0 to 9 walls
                        "wall_min_size": spaces.Box(
                            low=0.1, high=1.0, shape=(), dtype=np.float32
                        ),
                        "wall_max_size": spaces.Box(
                            low=1.0, high=2.0, shape=(), dtype=np.float32
                        ),
                        "wall_color": spaces.Box(
                            low=0, high=255, shape=(3,), dtype=np.uint8
                        ),
                    }
                ),
            }
        )

        self.variation_values = {
            "agent": {
                "color": np.array([255, 0, 0], dtype=np.uint8),
                "radius": 0.1,
                "start_position": np.array([0.5, 0.5], dtype=np.float32),
            },
            "goal": {
                "color": np.array([0, 255, 0], dtype=np.uint8),
                "radius": 0.2,
                "position": np.array([4.5, 4.5], dtype=np.float32),
            },
            "physics": {
                "speed": 1.0,
            },
            "env": {
                "n_walls": n_walls,
                "wall_min_size": wall_min_size,
                "wall_max_size": wall_max_size,
                "wall_color": np.array([0, 0, 0], dtype=np.uint8),
            },
        }

        assert self.variation_space.contains(self.variation_values), "Default variation values must be within variation space"

        self.state = self.variation_values["agent"]["start_position"].copy()
        self.walls = self._generate_walls()
        self._fig = None
        self._ax = None


    def update_variation(self, edit: dict, current_dict=None, path=None) -> dict:
        
        if current_dict is None:
            current_dict = self.variation_values
        
        if path is None:
            path = []

        for key, val in edit.items():
            current_path = path + [str(key)]

            assert key in current_dict, f"Key {key} not found in variation values at path: {' -> '.join(current_path)}"

            if isinstance(current_dict[key], dict) and isinstance(val, dict):
                self.update_variation(val, current_dict[key], current_path)
            else:
                current_dict[key] = val

        assert self.variation_space.contains(self.variation_values), f"Edited variation values must be within variation space after applying edit: {edit}"


    def _generate_walls(self):

        n_walls = self.variation_values["env"]["n_walls"]
        wall_max_size = self.variation_values["env"]["wall_max_size"]
        wall_min_size = self.variation_values["env"]["wall_min_size"]
        start_pos = self.variation_values["agent"]["start_position"]
        goal_pos = self.variation_values["goal"]["position"]

        walls = []
        attempts = 0
        while len(walls) < n_walls and attempts < n_walls * 10:
            w = self.rng.uniform(wall_min_size, wall_max_size)
            h = self.rng.uniform(wall_min_size, wall_max_size)
            x1 = self.rng.uniform(0, self.width - w)
            y1 = self.rng.uniform(0, self.height - h)
            x2 = x1 + w
            y2 = y1 + h
            # Check if wall overlaps start or goal
            margin = 0.3
            if (
                x1 - margin < start_pos[0] < x2 + margin
                and y1 - margin < start_pos[1] < y2 + margin
            ):
                attempts += 1
                continue
            if (
                x1 - margin < goal_pos[0] < x2 + margin
                and y1 - margin < goal_pos[1] < y2 + margin
            ):
                attempts += 1
                continue
            walls.append((x1, y1, x2, y2))
            attempts += 1
        return walls

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        if seed is not None:

            self.observation_space.seed(seed)

            if hasattr(self, "action_space"):
                self.action_space.seed(seed)

            if hasattr(self, "variation_space"):
                self.variation_space.seed(seed)

        options = options or {}

        if "variation" in options:
            self.update_variation(options["variation"])

        if options.get("walls", "random") == "random":
            self.walls = options.get("walls", self._generate_walls()).copy()

        self.variation_values = self.variation_space.sample()

        self.state = self.variation_values["agent"]["start_position"].copy()

        # generate our goal frame
        while True:
            pos = self.observation_space.sample()
            if not self._collides(pos):
                break
        
        # generate goal frame
        original_state = self.state.copy()
        self.state = pos
        self._goal = self.render()

        # load back original start and state
        self.state = original_state

        info = {"goal": self._goal}

        return self.state.copy(), info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        next_state = self.state + self.variation_values["physics"]["speed"] * action
        # Check for wall collisions
        if self._collides(next_state):
            next_state = self.state  # Stay in place if collision
        # Keep within bounds
        next_state = np.clip(
            next_state,
            self.observation_space.low,
            self.observation_space.high,
        )
        self.state = next_state
        # Check if goal reached
        terminated = np.linalg.norm(self.state - self.variation_values["goal"]["position"]) < self.variation_values["goal"]["radius"]
        truncated = False  # You can add a max step count if you want
        reward = 1.0 if terminated else -0.01  # Small penalty per step
        info = {"goal": self._goal}
        return self.state.copy(), reward, terminated, truncated, info

    def _collides(self, pos):
        x, y = pos
        for x1, y1, x2, y2 in self.walls:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

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

        # Draw walls
        for x1, y1, x2, y2 in self.walls:
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor=self.variation_values["env"]["wall_color"]/255.0)
            self._ax.add_patch(rect)
        
        # Draw goal
        if self.show_goal:
            goal_pos = self.variation_values["goal"]["position"]
            goal_radius = self.variation_values["goal"]["radius"]
            goal_color = self.variation_values["goal"]["color"]
            goal = Circle(goal_pos, goal_radius, facecolor=goal_color/255.0, alpha=0.5)
            self._ax.add_patch(goal)
        
        # Draw agent
        agent = Circle(self.state, self.variation_values["agent"]["radius"], facecolor=self.variation_values["agent"]["color"]/255.0)
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
