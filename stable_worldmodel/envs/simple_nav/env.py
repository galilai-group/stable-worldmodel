from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv

import stable_worldmodel as swm
from stable_worldmodel import spaces


DEFAULT_VARIATIONS = (
    "agent.position",
    "agent.direction",
    "goal.position",
    "goal.agent.direction",
    "maze.p_horizontal",
    "maze.p_vertical",
)


class SimpleNavigationEnv(MiniGridEnv):
    """Simple Navigation Environment

    The environment is a MiniGrid with a random maze. The agent is a point agent that can move in the maze. The goal is to reach the goal square.
    The agent can move in the maze by moving forward, turning left, or turning right.
    """

    def __init__(self, size=8, render_mode="rgb_array", *args, **kwargs):
        """Initialize the Simple Navigation Environment

        Args:
            size: size of the maze (width and height)
            render_mode: mode to render the environment
            *args: additional arguments to pass to the parent class
            **kwargs: additional keyword arguments to pass to the parent class
        """

        super().__init__(
            grid_size=size,
            mission_space=MissionSpace(mission_func=self._gen_mission),
            render_mode=render_mode,
            *args,
            **kwargs,
        )

        num_dirs = 4
        inner_width, inner_height = self.width - 2, self.height - 2

        self.observation_space = spaces.Dict(
            {
                # proprio: [ax, ay, a_dir]
                "proprio": spaces.MultiDiscrete(
                    nvec=[inner_width, inner_height, num_dirs],
                    start=[1, 1, 0],
                ),
                # state: [ax, ay, a_dir, gx, gy]
                "state": spaces.MultiDiscrete(
                    nvec=[inner_width, inner_height, num_dirs, inner_width, inner_height],
                    start=[1, 1, 0, 1, 1],
                ),
            }
        )

        # turn left, turn right, move forward
        self.action_space = spaces.Discrete(3)

        self.variation_space = swm.spaces.Dict(
            {
                "agent": swm.spaces.Dict(
                    {
                        "position": swm.spaces.MultiDiscrete(
                            nvec=[inner_width, inner_height],
                            start=[1, 1],
                            init_value=[1, 1],
                            constrain_fn=self._valid_agent_pos,
                        ),
                        "direction": swm.spaces.Discrete(
                            n=4,
                            start=0,
                            init_value=0,
                        ),
                    },
                ),
                "goal": swm.spaces.Dict(
                    {
                        "position": swm.spaces.MultiDiscrete(
                            nvec=[inner_width, inner_height],
                            start=[1, 1],
                            init_value=[inner_width - 2, inner_height - 2],
                            constrain_fn=self._valid_goal_pos,
                        ),
                        "agent": swm.spaces.Dict(
                            {
                                "direction": swm.spaces.Discrete(
                                    n=4,
                                    start=0,
                                    init_value=0,
                                ),
                            },
                        ),
                    },
                ),
                "maze": swm.spaces.Dict(
                    {
                        "p_horizontal": swm.spaces.Box(
                            low=0.0,
                            high=1.0,
                            init_value=0.5,
                            dtype=np.float32,
                        ),
                        "p_vertical": swm.spaces.Box(
                            low=0.0,
                            high=1.0,
                            init_value=0.5,
                            dtype=np.float32,
                        ),
                    }
                ),
            },
            sampling_order=["maze", "agent", "goal"],
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._maze_seed = seed

        super().reset(seed=seed, options=options)

        if hasattr(self, "variation_space"):
            self.variation_space.seed(seed)

        options = options or {}
        self.variation_space.reset()

        variations = options.get("variation", DEFAULT_VARIATIONS)
        if not isinstance(variations, Sequence):
            raise ValueError("variation option must be a Sequence containing variations names to sample")

        self.variation_space.update(variations)
        assert self.variation_space.check(debug=True), "Variation values must be within variation space!"

        # make goal state:
        # [gx, gy, a_dir, gx, gy] agent location is the same as goal location
        goal_state = np.concatenate(
            [
                self.variation_space["goal"]["position"].value.tolist(),
                [self.variation_space["goal"]["agent"]["direction"].value],
                self.variation_space["goal"]["position"].value.tolist(),
            ]
        )
        self._set_goal_state(goal_state)
        self._set_state(goal_state)
        self._goal = self.render()

        # restore original state
        state = np.concatenate(
            [
                self.variation_space["agent"]["position"].value.tolist(),
                [self.variation_space["agent"]["direction"].value],
                self.variation_space["goal"]["position"].value.tolist(),
            ]
        )
        self._set_state(state)

        # generate observation
        state = self._get_obs()
        proprio = state[:3]

        observation = {"proprio": proprio, "state": state}
        info = self._get_info()
        return observation, info

    def step(self, action):
        super().step(action)

        state = self._get_obs()
        proprio = state[:3]
        observation = {"proprio": proprio, "state": state}
        info = self._get_info()

        terminated, distance = self.eval_state(self.goal_state, state)
        reward = -distance  # the closer the better

        truncated = False
        return observation, reward, terminated, truncated, info

    @staticmethod
    def _gen_mission() -> str:
        return "go to the goal square"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        wall_coords = ellers_maze(
            width,
            height,
            self._maze_seed,
            self.variation_space["maze"]["p_horizontal"].value,
            self.variation_space["maze"]["p_vertical"].value,
        )
        for x, y in wall_coords:
            self.grid.wall_rect(x, y, 1, 1)

        # Default goal (will be overridden by variation_space in reset)
        gx, gy = width - 2, height - 2
        self.put_obj(Goal(), gx, gy)
        self.goal_pos = (gx, gy)

        # Default agent placement (will be overridden by variation_space in reset)
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        self.mission = self._gen_mission()

    def render(self):
        return super().render()

    def eval_state(self, goal_state, cur_state):
        # calculate manhattan distance between agent and goal
        goal_pos = goal_state[3:5]
        agent_pos = cur_state[:2]
        distance = np.linalg.norm(goal_pos - agent_pos, ord=1)
        success = distance == 0

        return success, distance

    def _get_info(self):
        goal_proprio = self.goal_state[:3]
        return {
            "pos_agent": np.array(self.agent_pos),
            "dir_agent": np.array(self.agent_dir),
            "pos_goal": np.array(self.goal_pos),
            "goal_state": np.array(self.goal_state),
            "goal_proprio": goal_proprio,
            "goal": self._goal,
        }

    def _get_obs(self):
        obs = tuple(self.agent_pos) + (self.agent_dir,) + tuple(self.goal_pos)
        return np.array(obs, dtype=np.int64)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()

        agent_pos = state[:2]
        agent_dir = state[2]
        goal_pos = state[3:5]

        self.agent_pos = agent_pos
        self.agent_dir = agent_dir

        # move goal to new position
        self.grid.set(self.goal_pos[0], self.goal_pos[1], None)
        self.put_obj(Goal(), goal_pos[0], goal_pos[1])
        self.goal_pos = goal_pos

    def _set_goal_state(self, goal_state):
        self.goal_state = goal_state

    def _is_floor_cell(self, pos: np.ndarray) -> bool:
        x, y = int(pos[0]), int(pos[1])
        tile = self.grid.get(x, y)
        # "Floor" means no wall / blocking object
        return tile is None or tile.can_overlap()

    def _valid_agent_pos(self, pos: np.ndarray) -> bool:
        return self._is_floor_cell(pos)

    def _valid_goal_pos(self, pos: np.ndarray) -> bool:
        x, y = int(pos[0]), int(pos[1])
        if not self._is_floor_cell(pos):
            return False

        agent_pos = self.variation_space["agent"]["position"].value
        return not (int(agent_pos[0]) == x and int(agent_pos[1]) == y)


def ellers_maze(width, height, seed, p_horizontal=0.5, p_vertical=0.5):
    """Generate a maze using the Eller's algorithm.

    Args:
        width: width of the maze
        height: height of the maze
        seed: seed for the random number generator
        p_horizontal: probability of horizontal walls
        p_vertical: probability of vertical walls
    Returns:
        List of (x, y) coordinates of walls in the maze.
    """
    assert width >= 3 and height >= 3
    assert width % 2 == 1 and height % 2 == 1
    assert 0.0 < p_horizontal < 1.0
    assert 0.0 < p_vertical < 1.0

    cells_w = (width - 1) // 2
    cells_h = (height - 1) // 2

    rng = np.random.default_rng(seed)
    maze = np.ones((height, width), dtype=bool)

    row_sets = [0] * cells_w
    next_set_id = 1

    for cy in range(cells_h):
        for cx in range(cells_w):
            if row_sets[cx] == 0:
                row_sets[cx] = next_set_id
                next_set_id += 1
            gy = 2 * cy + 1
            gx = 2 * cx + 1
            maze[gy, gx] = False

        if cy == cells_h - 1:
            for cx in range(cells_w - 1):
                if row_sets[cx] != row_sets[cx + 1]:
                    gy = 2 * cy + 1
                    wx = 2 * cx + 2
                    maze[gy, wx] = False
                    old, new = row_sets[cx + 1], row_sets[cx]
                    for i in range(cells_w):
                        if row_sets[i] == old:
                            row_sets[i] = new
            break

        for cx in range(cells_w - 1):
            if row_sets[cx] == row_sets[cx + 1]:
                continue
            if rng.random() < p_horizontal:
                gy = 2 * cy + 1
                wx = 2 * cx + 2
                maze[gy, wx] = False
                old, new = row_sets[cx + 1], row_sets[cx]
                for i in range(cells_w):
                    if row_sets[i] == old:
                        row_sets[i] = new

        set_members = {}
        for cx in range(cells_w):
            sid = row_sets[cx]
            set_members.setdefault(sid, []).append(cx)

        next_row_sets = [0] * cells_w
        for sid, xs in set_members.items():
            xs = xs[:]
            rng.shuffle(xs)
            selected = []
            for i, cx in enumerate(xs):
                if rng.random() < p_vertical or i == len(xs) - 1:
                    selected.append(cx)
            for cx in selected:
                gy = 2 * cy + 2
                gx = 2 * cx + 1
                maze[gy, gx] = False
                next_row_sets[cx] = sid

        row_sets = next_row_sets

    walls = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if maze[y, x]:
                walls.append((x, y))

    return walls
