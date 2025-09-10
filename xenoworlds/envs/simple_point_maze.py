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
    ):
        super().__init__()
        self.width = 5.0
        self.height = 5.0
        self.n_walls = n_walls
        self.wall_min_size = wall_min_size
        self.wall_max_size = wall_max_size
        self.rng = np.random.default_rng(seed)
        self.render_mode = render_mode
        self.start_pos = np.array([0.5, 0.5], dtype=np.float32)
        self.goal_pos = np.array([4.5, 4.5], dtype=np.float32)
        self.goal_radius = 0.2
        # Use Dict space for easy extension (e.g., adding "pixels" later)
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=np.array([0.0, 0.0], dtype=np.float32),
                    high=np.array([self.width, self.height], dtype=np.float32),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = spaces.Box(
            low=np.array([-0.2, -0.2], dtype=np.float32),
            high=np.array([0.2, 0.2], dtype=np.float32),
            dtype=np.float32,
        )
        self.state = self.start_pos.copy()
        self.walls = self._generate_walls()
        self._fig = None
        self._ax = None

    def _generate_walls(self):
        walls = []
        attempts = 0
        while len(walls) < self.n_walls and attempts < self.n_walls * 10:
            w = self.rng.uniform(self.wall_min_size, self.wall_max_size)
            h = self.rng.uniform(self.wall_min_size, self.wall_max_size)
            x1 = self.rng.uniform(0, self.width - w)
            y1 = self.rng.uniform(0, self.height - h)
            x2 = x1 + w
            y2 = y1 + h
            # Check if wall overlaps start or goal
            margin = 0.3
            if (
                x1 - margin < self.start_pos[0] < x2 + margin
                and y1 - margin < self.start_pos[1] < y2 + margin
            ):
                attempts += 1
                continue
            if (
                x1 - margin < self.goal_pos[0] < x2 + margin
                and y1 - margin < self.goal_pos[1] < y2 + margin
            ):
                attempts += 1
                continue
            walls.append((x1, y1, x2, y2))
            attempts += 1
        return walls

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.state = options.get("start_pos", self.start_pos).copy()
        self.walls = options.get("walls", self._generate_walls()).copy()

        while True:
            pos = np.random.randn(2)
            if not self._collides(pos):
                break
        original_start = self.start_pos.copy()
        self.start_pos = pos
        goal = self.render()
        self.start_pos = original_start
        info = {"goal": goal}

        return {"state": self.state.copy()}, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        next_state = self.state + action
        # Check for wall collisions
        if self._collides(next_state):
            next_state = self.state  # Stay in place if collision
        # Keep within bounds
        next_state = np.clip(
            next_state,
            self.observation_space["state"].low,
            self.observation_space["state"].high,
        )
        self.state = next_state
        # Check if goal reached
        terminated = np.linalg.norm(self.state - self.goal_pos) < self.goal_radius
        truncated = False  # You can add a max step count if you want
        reward = 1.0 if terminated else -0.01  # Small penalty per step
        info = {}
        return {"state": self.state.copy()}, reward, terminated, truncated, info

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
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, color="black")
            self._ax.add_patch(rect)
        # Draw goal
        goal = Circle(self.goal_pos, self.goal_radius, color="green", alpha=0.5)
        self._ax.add_patch(goal)
        # Draw agent
        agent = Circle(self.state, 0.1, color="red")
        self._ax.add_patch(agent)
        # Draw start
        start = Circle(self.start_pos, 0.1, color="blue", alpha=0.5)
        self._ax.add_patch(start)
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
