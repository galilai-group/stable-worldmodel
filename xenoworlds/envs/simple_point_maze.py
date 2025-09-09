import gym
from gym import spaces
import numpy as np


class SimpleMazeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        width: float = 5,
        height: float = 5,
        n_walls=5,
        wall_min_size=0.5,
        wall_max_size=1.5,
        seed=None,
    ):
        super(SimpleMazeEnv, self).__init__()
        # Maze size
        self.width = width
        self.height = height
        # Define maze walls as list of rectangles: (x1, y1, x2, y2)
        self.n_walls = n_walls
        self.wall_min_size = wall_min_size
        self.wall_max_size = wall_max_size
        self.rng = np.random.default_rng(seed)

        # Start and goal positions
        self.start_pos = np.array([0.5, 0.5])
        self.goal_pos = np.array([4.5, 4.5])
        self.goal_radius = 0.2
        # Observation: agent's (x, y) position
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([self.width, self.height]),
            dtype=np.float32,
        )
        # Action: delta x, delta y (continuous)
        self.action_space = spaces.Box(
            low=np.array([-0.2, -0.2]), high=np.array([0.2, 0.2]), dtype=np.float32
        )
        self.state = self.start_pos.copy()
        self.walls = []
        self._generate_walls()

    def reset(self):
        self.state = self.start_pos.copy()
        self._generate_walls()
        return self.state.copy()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        next_state = self.state + action
        # Check for wall collisions
        if self._collides(next_state):
            next_state = self.state  # Stay in place if collision
        # Keep within bounds
        next_state = np.clip(
            next_state, self.observation_space.low, self.observation_space.high
        )
        self.state = next_state
        # Check if goal reached
        done = np.linalg.norm(self.state - self.goal_pos) < self.goal_radius
        reward = 1.0 if done else -0.01  # Small penalty per step
        return self.state.copy(), reward, done, {}

    def _collides(self, pos):
        x, y = pos
        for x1, y1, x2, y2 in self.walls:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    def render(self, mode="human"):
        print(f"Agent position: {self.state}")
        print(f"Walls: {self.walls}")

    def close(self):
        pass


# Register the environment (optional, for gym.make)
from gym.envs.registration import register

register(
    id="SimpleMaze-v0",
    entry_point=__name__ + ":SimpleMazeEnv",
)
