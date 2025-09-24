
import gymnasium as gym
#from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from stable_worldmodel import spaces
from loguru import logger as logging

class SimplePointMazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        max_walls=6,
        min_walls=4,
        wall_min_size=0.5,
        wall_max_size=1.5,
        seed=42,
        render_mode=None,
        show_goal: bool = True,
    ):
        super().__init__()
        self.show_goal = show_goal
        self.width = 5.0
        self.height = 5.0
        self.rng = np.random.default_rng(seed)
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([self.width, self.height], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
            seed=seed,
        )

        self.action_space = spaces.Box(
            low=np.array([-0.2, -0.2], dtype=np.float32),
            high=np.array([0.2, 0.2], dtype=np.float32),
            dtype=np.float32,
            shape=(2,),
            seed=seed,
        )

        #### variation space

        wall_pos_high = np.array([[self.width, self.height]], dtype=np.float32).repeat(max_walls, axis=0)
        wall_pos_low = np.array([[0.0, 0.0]], dtype=np.float32).repeat(max_walls, axis=0)

        wall_size_low = np.array([[wall_min_size, wall_min_size]], dtype=np.float32).repeat(max_walls, axis=0)
        wall_size_high = np.array([[wall_max_size, wall_max_size]], dtype=np.float32).repeat(max_walls, axis=0)

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
                        )
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
                        )
                    }
                ),
                "physics": spaces.Dict(
                    {
                        "speed": spaces.Box(
                            low=0.05, high=2, shape=(), dtype=np.float32),
                    }
                ),
                "env": spaces.Dict(
                    {
                        "n_walls": spaces.Discrete(max_walls, start=min_walls),
                        "wall_color": spaces.Box(low=0, high=255, shape=(3,), dtype=np.uint8),
                        "wall_shape": spaces.Box(
                            low=wall_size_low, high=wall_size_high, shape=(max_walls, 2), dtype=np.float32
                        ),
                        "wall_positions": spaces.Box(
                            low=wall_pos_low, high=wall_pos_high, shape=(max_walls, 2), dtype=np.float32,
                        ),

                    },
                    constrain_fn=lambda env: self._env_ok(env),
                ),
            },
            sampling_order = ["agent", "goal", "physics", "env"],
            constrain_fn=lambda v: self._collide(v),
        )


        self.variation_space.seed(seed)
        self.variation_values = self.variation_space.sample() # populate with valid values

        # default colors and shape
        self.variation_values['agent']['color'] = np.array([255, 0, 0], dtype=np.uint8)
        self.variation_values['goal']['color'] = np.array([0, 255, 0], dtype=np.uint8)
        self.variation_values['env']['wall_color'] = np.array([0, 0, 0], dtype=np.uint8)
        self.variation_values['agent']['radius'] = 0.1
        self.variation_values['goal']['radius'] = 0.2
        self.variation_values['physics']['speed'] = 1.0

        self.state = self.variation_values["agent"]["start_position"].copy()

        # need walls to check validity of default variation values
        assert self.variation_space.contains(self.variation_values), "Default variation values must be within variation space"

        self._fig = None
        self._ax = None

        return


    def update_variation(self, edit: dict, current_dict=None, path=None):
        
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

    def reset(self, seed=None, options=None):

        super().reset(seed=seed, options=options)
        self.observation_space.seed(seed)
        if hasattr(self, "action_space"): self.action_space.seed(seed)
        if hasattr(self, "variation_space"): self.variation_space.seed(seed)
        self.rng = np.random.default_rng(seed)

        options = options or {}

        if "variation" in options:
            self.update_variation(options["variation"])

        if options.get("walls", "random") == "random":
            self.variation_values['env']['wall_shape'] =  self.variation_space['env']['wall_shape'].sample()
            self.variation_values['env']['wall_positions'] =  self.variation_space['env']['wall_positions'].sample()

        self.variation_values = self.variation_space.sample()

        # generate goal frame
        original_state = self.variation_values["agent"]["start_position"].copy()
        self.state = self.observation_space.sample()
        self._goal = self.render()

        # load back original start and state
        self.state = original_state
        info = {"goal": self._goal}

        return self.state.copy(), info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        next_state = self.state + self.variation_values["physics"]["speed"] * action
        # Check for wall collisions
        if self._collides(self.variation_values, next_state):
            next_state = self.state  # Stay in place if collision
        # Keep within bounds
        next_state = np.clip(
            next_state,
            self.observation_space.low,
            self.observation_space.high,
        )
        self.state = next_state
        # Check if goal reached
        terminated = (np.linalg.norm(self.state - self.variation_values["goal"]["position"]) < self.variation_values["goal"]["radius"]).item()
        truncated = False  # You can add a max step count if you want
        reward = 1.0 if terminated else -0.01  # Small penalty per step
        info = {"goal": self._goal}
        return self.state.copy(), reward, terminated, truncated, info


    def _collides(self, var, pos, entity='agent'):

        assert entity in ['agent', 'goal'], "Entity must be 'agent' or 'goal'"

        x, y = pos

        radius = var[entity]["radius"]
        num_walls = var["env"]["n_walls"]
        wall_shape = var["env"]["wall_shape"] 
        wall_positions = var["env"]["wall_positions"]  

        w = wall_shape[:num_walls, 0]
        h = wall_shape[:num_walls, 1]

        wx = wall_positions[:num_walls, 0]
        wy = wall_positions[:num_walls, 1]

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
                if (cx - x) ** 2 + (cy - y) ** 2 <= radius ** 2:
                    return True

        return False


    def _env_ok(self, env, margin: float = 0.3) -> bool:
        n = env["n_walls"]
        pos = env["wall_positions"][:n]
        wh  = env["wall_shape"][:n]

        x, y = pos[:, 0], pos[:, 1]
        w, h = wh[:, 0], wh[:, 1]

        # must fit in the 5x5 area
        fits_h = np.all(self.width  >= x + w)
        fits_v = np.all(self.height >= y + h)
        return bool(fits_h and fits_v)

    def _collide(self, v, margin: float = 0.3) -> bool:
        n   = v["env"]["n_walls"]
        pos = v["env"]["wall_positions"][:n]
        wh  = v["env"]["wall_shape"][:n]

        x, y = pos[:, 0], pos[:, 1]
        w, h = wh[:, 0], wh[:, 1]

        fits_h = np.all(self.width  >= x + w)
        fits_v = np.all(self.height >= y + h)


        agent_in = self._collides(v, v["agent"]["start_position"], entity='agent')
        goal_in  = self._collides(v, v["goal"]["position"], entity='goal')

        return bool(fits_h and fits_v and not (agent_in or goal_in))

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
        num_walls = self.variation_values["env"]["n_walls"]
        wall_shape = self.variation_values["env"]["wall_shape"][:num_walls]
        wall_positions = self.variation_values["env"]["wall_positions"][:num_walls]

        w, h = wall_shape[:, 0], wall_shape[:, 1]
        wx, wy = wall_positions[:, 0], wall_positions[:, 1]

        for x1, y1, x2, y2 in zip(wx, wy, wx + w, wy + h):
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


