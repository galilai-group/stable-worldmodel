from collections import deque

import numpy as np

from stable_worldmodel.policy import BasePolicy


class ExpertPolicy(BasePolicy):
    """Expert Policy for Gridworld Navigation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.type = "expert"
        self.action_buffer = deque()
        self._action_buffer = None
        self._next_init = None

    def set_env(self, env):
        self.env = env
        n_envs = getattr(env, "num_envs", 1)
        self._action_buffer = [deque() for _ in range(n_envs)]

    def get_action(self, info_dict, **kwargs):
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert "pixels" in info_dict, "'pixels' must be provided in info_dict"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"

        for i, env in enumerate(self.env.unwrapped.envs):
            if len(self._action_buffer[i]) == 0:
                agent_pos = info_dict["pos_agent"].squeeze()[i]
                goal_pos = info_dict["pos_goal"].squeeze()[i]
                agent_dir = info_dict["dir_agent"][i]

                path = self._shortest_path(agent_pos, goal_pos, env.unwrapped.grid)
                actions = self._actions_from_path(path, agent_dir)
                self._action_buffer[i].extend(actions)

        action = [self._action_buffer[i].popleft() for i in range(self.env.num_envs)]
        action = np.array(action).reshape(*self.env.action_space.shape)

        return action  # (num_envs, action_dim)

    def _shortest_path(self, start_pos, goal_pos, grid):
        """Find the shortest path on a grid from start to goal using BFS.

        Args:
            start_pos: (x, y) start coordinate.
            goal_pos: (x, y) goal coordinate.
        Returns:
            List of (x, y) coordinates from start to goal (inclusive),
            or None if no path exists.
        """

        width = grid.width
        height = grid.height
        start_pos = tuple(start_pos)
        goal_pos = tuple(goal_pos)

        if start_pos == goal_pos:
            return [start_pos]

        # (up, down, left, right)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        queue = deque([start_pos])
        visited = {start_pos}
        parent = {start_pos: None}

        while queue:
            x, y = queue.popleft()

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                # Check bounds
                if not (0 <= nx < width and 0 <= ny < height):
                    continue

                # Skip walls
                if grid.get(nx, ny) is not None and not grid.get(nx, ny).can_overlap():
                    continue

                neighbor = (nx, ny)
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                parent[neighbor] = (x, y)
                queue.append(neighbor)

                # Stop early if we reached the goal
                if neighbor == goal_pos:
                    # Reconstruct path
                    path = []
                    cur = goal_pos
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    return path

        # No path found
        return None

    def _actions_from_path(self, path, start_dir):
        """Convert a path into a sequence of actions in MiniGrid.

        Args:
            path: List of (x, y) coordinates from start to goal (inclusive).
            start_dir: Starting direction of the agent.
        Returns:
            List of actions in MiniGrid.
        """
        TURN_LEFT, TURN_RIGHT, FORWARD = 0, 1, 2
        RIGHT, DOWN, LEFT, UP = 0, 1, 2, 3

        cur_dir = start_dir
        cur_pos = path[0]
        actions = []

        for next_pos in path[1:]:
            dx = next_pos[0] - cur_pos[0]
            dy = next_pos[1] - cur_pos[1]

            if (dx, dy) == (1, 0):
                next_dir = RIGHT
            elif (dx, dy) == (-1, 0):
                next_dir = LEFT
            elif (dx, dy) == (0, 1):
                next_dir = DOWN
            elif (dx, dy) == (0, -1):
                next_dir = UP

            if next_dir == cur_dir:
                actions.append(FORWARD)
            elif next_dir == (cur_dir + 1) % 4:
                actions.extend([TURN_RIGHT, FORWARD])
            elif next_dir == (cur_dir - 1) % 4:
                actions.extend([TURN_LEFT, FORWARD])
            elif next_dir == (cur_dir + 2) % 4:
                actions.extend([TURN_LEFT, TURN_LEFT, FORWARD])

            cur_pos = next_pos
            cur_dir = next_dir

        return actions
