import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from gymnasium.spaces import Box
from ogbench.locomaze.ant import AntEnv

from stable_worldmodel import spaces


def color_to_str(color):
    return " ".join([str(x / 255.0) for x in color]) + " 1"


class AntMazeEnv(AntEnv):
    """Ant Maze environment.

    It inherits from the locomotion environment and adds a maze to it.
    """

    def __init__(
        self,
        maze_type="medium",
        maze_unit=4.0,
        maze_height=0.5,
        terminate_at_goal=True,
        inside=False,
        add_noise_to_goal=True,
        reward_task_id=None,
        use_oracle_rep=False,
        *args,
        **kwargs,
    ):
        """Initialize the maze environment.

        Args:
            maze_type: Maze type. One of 'arena', 'medium', 'large', 'giant', or 'teleport'.
            maze_unit: Size of a maze unit block.
            maze_height: Height of the maze walls.
            terminate_at_goal: Whether to terminate the episode when the goal is reached.
            inside: Whether to use the inside view (third-person view) camera.
            add_noise_to_goal: Whether to add noise to the goal position.
            reward_task_id: Task ID for single-task RL. If this is not None, the environment operates in a
                single-task mode with the specified task ID. The task ID must be either a valid task ID or 0, where
                0 means using the default task.
            use_oracle_rep: Whether to use oracle goal representations.
            *args: Additional arguments to pass to the parent locomotion environment.
            **kwargs: Additional keyword arguments to pass to the parent locomotion environment.
        """
        self._maze_type = maze_type
        self._maze_unit = maze_unit
        self._maze_height = maze_height
        self._terminate_at_goal = terminate_at_goal
        self._inside = inside
        self._add_noise_to_goal = add_noise_to_goal
        self._reward_task_id = reward_task_id
        self._use_oracle_rep = use_oracle_rep

        # Define constants.
        self._offset_x = 4
        self._offset_y = 4
        self._noise = 1
        self._goal_tol = 0.5

        # Define maze map.
        self._teleport_info = None
        if self._maze_type == "arena":
            maze_map = [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]
        elif self._maze_type == "medium":
            maze_map = [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 1, 1, 0, 0, 1],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]
        elif self._maze_type == "large":
            maze_map = [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        elif self._maze_type == "giant":
            maze_map = [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        elif self._maze_type == "teleport":
            maze_map = [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
                [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
                [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
            self._teleport_info = {
                "teleport_in_ijs": [(4, 6), (5, 1)],
                "teleport_out_ijs": [(1, 7), (6, 1), (6, 10)],
                "teleport_radius": 1,
            }
            self._teleport_info["teleport_in_xys"] = [
                self.ij_to_xy(ij) for ij in self._teleport_info["teleport_in_ijs"]
            ]
            self._teleport_info["teleport_out_xys"] = [
                self.ij_to_xy(ij) for ij in self._teleport_info["teleport_out_ijs"]
            ]
        else:
            raise ValueError(f"Unknown maze type: {self._maze_type}")

        self.maze_map = np.array(maze_map)

        wall_default_color = np.array([153, 153, 153], dtype=np.uint8)  # Gray
        init_camera_post = [2 * (self.maze_map.shape[1] - 3), 2 * (self.maze_map.shape[0] - 3)]
        init_camera_distance = 5 * (self.maze_map.shape[1] - 2)
        init_camera_elevation = 90

        self.variation_space = spaces.Dict(
            {
                # "agent": ...,
                # "goal": ...,
                "walls": spaces.Dict({"color": spaces.RGBBox(init_value=wall_default_color)}),
                "camera": spaces.Dict(
                    {
                        "position": spaces.MultiDiscrete(
                            [12, 12], start=[x - 5 for x in init_camera_post], init_value=init_camera_post
                        ),
                        "distance": spaces.Discrete(
                            20,
                            start=init_camera_distance - 9,
                            init_value=init_camera_distance,
                        ),
                        "elevation": spaces.Discrete(
                            20,
                            start=init_camera_elevation - 9,
                            init_value=init_camera_elevation,
                        ),
                    },
                ),
            }
        )

        # Update XML file.
        xml_file = self.xml_file
        tree = ET.parse(xml_file)
        self.update_tree(tree)
        _, maze_xml_file = tempfile.mkstemp(text=True, suffix=".xml")
        tree.write(maze_xml_file)

        super().__init__(xml_file=maze_xml_file, *args, **kwargs)

        self.custom_camera = self.camera_id or self.camera_name

        if self.custom_camera is None:
            self.custom_camera = mujoco.MjvCamera()
            self.update_camera()

        # Set task goals.
        self.task_infos = []
        self.cur_task_id = None
        self.cur_task_info = None
        self.set_tasks()
        self.num_tasks = len(self.task_infos)
        self.cur_goal_xy = np.zeros(2)

        self.custom_renderer = None
        if self._inside:
            self.observation_space = Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

            # Manually color the floor to enable the agent to infer its position from the observation.
            tex_grid = self.model.tex("grid")
            tex_height = tex_grid.height[0]
            tex_width = tex_grid.width[0]
            # MuJoCo 3.2.1 changed the attribute name from 'tex_rgb' to 'tex_data'.
            attr_name = "tex_rgb" if hasattr(self.model, "tex_rgb") else "tex_data"
            tex_rgb = getattr(self.model, attr_name)[tex_grid.adr[0] : tex_grid.adr[0] + 3 * tex_height * tex_width]
            tex_rgb = tex_rgb.reshape(tex_height, tex_width, 3)
            for x in range(tex_height):
                for y in range(tex_width):
                    min_value = 0
                    max_value = 192
                    r = int(x / tex_height * (max_value - min_value) + min_value)
                    g = int(y / tex_width * (max_value - min_value) + min_value)
                    tex_rgb[x, y, :] = [r, g, 128]
            self.initialize_renderer()
        else:
            ex_ob = self.get_ob()
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=ex_ob.shape, dtype=ex_ob.dtype)

    def update_tree(self, tree):
        """Update the XML tree to include the maze."""
        worldbody = tree.find(".//worldbody")

        # Add walls.
        for i in range(self.maze_map.shape[0]):
            for j in range(self.maze_map.shape[1]):
                struct = self.maze_map[i, j]
                if struct == 1:
                    ET.SubElement(
                        worldbody,
                        "geom",
                        name=f"block_{i}_{j}",
                        pos=f"{j * self._maze_unit - self._offset_x} {i * self._maze_unit - self._offset_y} {self._maze_height / 2 * self._maze_unit}",
                        size=f"{self._maze_unit / 2} {self._maze_unit / 2} {self._maze_height / 2 * self._maze_unit}",
                        type="box",
                        contype="1",
                        conaffinity="1",
                        material="wall",
                    )

        # Adjust floor size.
        center_x, center_y = 2 * (self.maze_map.shape[1] - 3), 2 * (self.maze_map.shape[0] - 3)
        size_x, size_y = 2 * self.maze_map.shape[1], 2 * self.maze_map.shape[0]
        floor = tree.find('.//geom[@name="floor"]')
        floor.set("pos", f"{center_x} {center_y} 0")
        floor.set("size", f"{size_x} {size_y} 0.2")

        if self._teleport_info is not None:
            # Add teleports.
            for i, (x, y) in enumerate(self._teleport_info["teleport_in_xys"]):
                ET.SubElement(
                    worldbody,
                    "geom",
                    name=f"teleport_in_{i}",
                    type="cylinder",
                    size=f"{self._teleport_info['teleport_radius']} .05",
                    pos=f"{x} {y} .05",
                    material="teleport_in",
                    contype="0",
                    conaffinity="0",
                )
            for i, (x, y) in enumerate(self._teleport_info["teleport_out_xys"]):
                ET.SubElement(
                    worldbody,
                    "geom",
                    name=f"teleport_out_{i}",
                    type="cylinder",
                    size=f"{self._teleport_info['teleport_radius']} .05",
                    pos=f"{x} {y} .05",
                    material="teleport_out",
                    contype="0",
                    conaffinity="0",
                )

        if self._inside:
            # Color wall.
            wall = tree.find('.//material[@name="wall"]')
            wall.set("rgba", color_to_str(self.variation_space.value["walls"]["color"]))
            # Remove ambient light.
            light = tree.find('.//light[@name="global"]')
            light.attrib.pop("ambient")
            # Remove torso light.
            torso_light = tree.find('.//light[@name="torso_light"]')
            torso_light_parent = tree.find('.//light[@name="torso_light"]/..')
            torso_light_parent.remove(torso_light)
            # Remove texture repeat.
            grid = tree.find('.//material[@name="grid"]')
            grid.set("texuniform", "false")

            # Color one leg white to break symmetry.
            tree.find('.//geom[@name="aux_1_geom"]').set("material", "self_white")
            tree.find('.//geom[@name="left_leg_geom"]').set("material", "self_white")
            tree.find('.//geom[@name="left_ankle_geom"]').set("material", "self_white")

        else:
            # Only show the target for states-based observation.
            ET.SubElement(
                worldbody,
                "geom",
                name="target",
                type="cylinder",
                size=".5 .05",
                pos="0 0 .05",
                material="target",
                contype="0",
                conaffinity="0",
            )

    def set_tasks(self):
        # `tasks` is a list of tasks, where each task is a list of two tuples: (init_ij, goal_ij).
        if self._maze_type == "arena":
            tasks = [
                [(1, 1), (6, 6)],
            ]
        elif self._maze_type == "medium":
            tasks = [
                [(1, 1), (6, 6)],
                [(6, 1), (1, 6)],
                [(5, 3), (4, 2)],
                [(6, 5), (6, 1)],
                [(2, 6), (1, 1)],
            ]
        elif self._maze_type == "large":
            tasks = [
                [(1, 1), (7, 10)],
                [(5, 4), (7, 1)],
                [(7, 4), (1, 10)],
                [(3, 8), (5, 4)],
                [(1, 1), (5, 4)],
            ]
        elif self._maze_type == "giant":
            tasks = [
                [(1, 1), (10, 14)],
                [(1, 14), (10, 1)],
                [(8, 14), (1, 1)],
                [(8, 3), (5, 12)],
                [(5, 9), (3, 8)],
            ]
        elif self._maze_type == "teleport":
            tasks = [
                [(1, 10), (7, 1)],
                [(1, 1), (7, 10)],
                [(5, 6), (7, 10)],
                [(7, 1), (7, 10)],
                [(5, 6), (7, 1)],
            ]
        else:
            raise ValueError(f"Unknown maze type: {self._maze_type}")

        self.task_infos = []
        for i, task in enumerate(tasks):
            self.task_infos.append(
                {
                    "task_name": f"task{i + 1}",
                    "init_ij": task[0],
                    "init_xy": self.ij_to_xy(task[0]),
                    "goal_ij": task[1],
                    "goal_xy": self.ij_to_xy(task[1]),
                }
            )

        if self._reward_task_id == 0:
            self._reward_task_id = 1  # Default task.

    def initialize_renderer(self):
        # Make custom renderer.
        self.custom_renderer = mujoco.Renderer(
            self.model,
            width=self.width,
            height=self.height,
        )
        self.render()

    def reset(self, seed=None, options=None, *args, **kwargs):
        options = options or {}

        if hasattr(self, "variation_space"):
            self.variation_space.seed(seed)

        self.variation_options = options.get("variation", {})

        self.variation_space.reset()

        if "variation" in options:
            assert isinstance(options["variation"], list | tuple), (
                "variation option must be a list or tuple containing variation names to sample"
            )

            if len(options["variation"]) == 1 and options["variation"][0] == "all":
                self.variation_space.sample()

            else:
                self.variation_space.update(set(options["variation"]))

        assert self.variation_space.check(debug=True), "Variation values must be within variation space!"

        # update camera
        if type(self.custom_camera) is mujoco.MjvCamera:
            self.update_camera()

        # Set the task goal.
        if self._reward_task_id is not None:
            # Use the pre-defined task.
            assert 1 <= self._reward_task_id <= self.num_tasks, f"Task ID must be in [1, {self.num_tasks}]."
            self.cur_task_id = self._reward_task_id
            self.cur_task_info = self.task_infos[self.cur_task_id - 1]
        elif "task_id" in options:
            # Use the pre-defined task.
            assert 1 <= options["task_id"] <= self.num_tasks, f"Task ID must be in [1, {self.num_tasks}]."
            self.cur_task_id = options["task_id"]
            self.cur_task_info = self.task_infos[self.cur_task_id - 1]
        elif "task_info" in options:
            # Use the provided task information.
            self.cur_task_id = None
            self.cur_task_info = options["task_info"]
        else:
            # Randomly sample a task.
            self.cur_task_id = np.random.randint(1, self.num_tasks + 1)
            self.cur_task_info = self.task_infos[self.cur_task_id - 1]

        # Get initial and goal positions with noise.
        init_xy = self.add_noise(self.ij_to_xy(self.cur_task_info["init_ij"]))
        goal_xy = self.ij_to_xy(self.cur_task_info["goal_ij"])
        if self._add_noise_to_goal:
            goal_xy = self.add_noise(goal_xy)

        # First, force set the position to the goal position to obtain the goal observation.
        super().reset(seed=seed, options=options, *args, **kwargs)

        # Do a few random steps to stabilize the environment.
        num_random_actions = 5
        for _ in range(num_random_actions):
            super().step(self.action_space.sample())

        # Save the goal observation.
        self.set_goal(goal_xy=goal_xy)
        self.set_xy(goal_xy)
        self._current_goal_ob = self.get_oracle_rep() if self._use_oracle_rep else self.get_ob()
        self._current_goal = self.render()

        # Now, do the actual reset.
        ob, info = super().reset(*args, **kwargs)
        self.set_goal(goal_xy=goal_xy)
        self.set_xy(init_xy)
        ob = self.get_ob()
        info["goal_ob"] = self._current_goal_ob
        info["goal"] = self._current_goal
        return ob, info

    def step(self, action):
        ob, reward, terminated, truncated, info = super().step(action)

        if self._teleport_info is not None:
            # Check if the agent is close to a inbound teleport.
            for x, y in self._teleport_info["teleport_in_xys"]:
                if np.linalg.norm(self.get_xy() - np.array([x, y])) <= self._teleport_info["teleport_radius"] * 1.5:
                    # Teleport the agent to a random outbound teleport.
                    teleport_out_xy = self._teleport_info["teleport_out_xys"][
                        np.random.randint(len(self._teleport_info["teleport_out_xys"]))
                    ]
                    self.set_xy(np.array(teleport_out_xy))
                    break

        # Check if the agent has reached the goal.
        if np.linalg.norm(self.get_xy() - self.cur_goal_xy) <= self._goal_tol:
            if self._terminate_at_goal:
                terminated = True
            info["success"] = 1.0
            reward = 1.0
        else:
            info["success"] = 0.0
            reward = 0.0

        # If the environment is in the single-task mode, modify the reward.
        if self._reward_task_id is not None:
            reward = reward - 1.0  # -1 (failure) or 0 (success).

        info["goal_ob"] = self._current_goal_ob
        info["goal"] = self._current_goal

        return ob, reward, terminated, truncated, info

    def render(self):
        if self.custom_renderer is None:
            self.initialize_renderer()
        self.custom_renderer.update_scene(self.data, camera=self.custom_camera)
        return self.custom_renderer.render()

    def get_ob(self, pixels=False):
        if pixels or self._inside:
            frame = self.render()
            return frame
        else:
            return super().get_ob()

    def update_camera(self):
        if self.custom_camera is None:
            return

        self.custom_camera.lookat[:2] = self.variation_space.value["camera"]["position"]
        self.custom_camera.distance = self.variation_space.value["camera"]["distance"]
        self.custom_camera.elevation = -self.variation_space.value["camera"]["elevation"]

        return

    def get_oracle_rep(self):
        """Return the oracle goal representation (i.e., the goal position)."""
        return np.array(self.cur_goal_xy)

    def set_goal(self, goal_ij=None, goal_xy=None):
        """Set the goal position and update the target object."""
        if goal_xy is None:
            self.cur_goal_xy = self.ij_to_xy(goal_ij)
            if self._add_noise_to_goal:
                self.cur_goal_xy = self.add_noise(self.cur_goal_xy)
        else:
            self.cur_goal_xy = goal_xy
        if not self._inside:
            self.model.geom("target").pos[:2] = goal_xy

    def get_oracle_subgoal(self, start_xy, goal_xy):
        """Get the oracle subgoal for the agent.

        If the goal is unreachable, it returns the current position as the subgoal.

        Args:
            start_xy: Starting position of the agent.
            goal_xy: Goal position of the agent.
        Returns:
            A tuple of the oracle subgoal and the BFS map.
        """
        start_ij = self.xy_to_ij(start_xy)
        goal_ij = self.xy_to_ij(goal_xy)

        # Run BFS to find the next subgoal.
        bfs_map = self.maze_map.copy()
        for i in range(self.maze_map.shape[0]):
            for j in range(self.maze_map.shape[1]):
                bfs_map[i][j] = -1

        bfs_map[goal_ij[0], goal_ij[1]] = 0
        queue = [goal_ij]
        while len(queue) > 0:
            i, j = queue.pop(0)
            for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                ni, nj = i + di, j + dj
                if (
                    0 <= ni < self.maze_map.shape[0]
                    and 0 <= nj < self.maze_map.shape[1]
                    and self.maze_map[ni, nj] == 0
                    and bfs_map[ni, nj] == -1
                ):
                    bfs_map[ni][nj] = bfs_map[i][j] + 1
                    queue.append((ni, nj))

        # Find the subgoal that attains the minimum BFS value.
        subgoal_ij = start_ij
        for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            ni, nj = start_ij[0] + di, start_ij[1] + dj
            if (
                0 <= ni < self.maze_map.shape[0]
                and 0 <= nj < self.maze_map.shape[1]
                and self.maze_map[ni, nj] == 0
                and bfs_map[ni, nj] < bfs_map[subgoal_ij[0], subgoal_ij[1]]
            ):
                subgoal_ij = (ni, nj)
        subgoal_xy = self.ij_to_xy(subgoal_ij)
        return np.array(subgoal_xy), bfs_map

    def xy_to_ij(self, xy):
        maze_unit = self._maze_unit
        i = int((xy[1] + self._offset_y + 0.5 * maze_unit) / maze_unit)
        j = int((xy[0] + self._offset_x + 0.5 * maze_unit) / maze_unit)
        return i, j

    def ij_to_xy(self, ij):
        i, j = ij
        x = j * self._maze_unit - self._offset_x
        y = i * self._maze_unit - self._offset_y
        return x, y

    def add_noise(self, xy):
        random_x = np.random.uniform(low=-self._noise, high=self._noise) * self._maze_unit / 4
        random_y = np.random.uniform(low=-self._noise, high=self._noise) * self._maze_unit / 4
        return xy[0] + random_x, xy[1] + random_y
