import numpy as np

from stable_worldmodel.policy import BasePolicy


class ExpertPolicy(BasePolicy):
    """Expert Policy for Two Room Environment."""

    def __init__(self, action_noise=1.3, **kwargs):
        super().__init__(**kwargs)
        self.type = "expert"
        self.action_noise = action_noise

    def set_env(self, env):
        self.env = env

    def get_action(self, info_dict, **kwargs):
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert "pixels" in info_dict, "'pixels' must be provided in info_dict"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"
        assert "pos_agent" in info_dict, "'pos_agent' must be provided in info_dict"
        assert "goal_pos" in info_dict, "'goal_pos' must be provided in info_dict"

        # Handle vectorized envs (VecEnv-style) and single envs gracefully
        base_env = self.env.unwrapped
        if hasattr(base_env, "envs"):
            envs = [e.unwrapped for e in base_env.envs]
        else:
            envs = [base_env]

        act_shape = self.env.action_space.shape
        actions = np.zeros(act_shape, dtype=np.float32)

        for i, env in enumerate(envs):
            agent_pos = np.asarray(info_dict["pos_agent"][i]).squeeze().astype(np.float32)
            goal_pos = np.asarray(info_dict["goal_pos"][i]).squeeze().astype(np.float32)
            max_norm = env.max_step_norm

            # --- determine if goal is in the other room w.r.t. CURRENT agent position ---
            wall_axis = env.variation_space.value["wall"][
                "axis"
            ]  # 0: horizontal wall (splits vertically), 1: vertical wall
            wall_pos = env.wall_pos

            # index of coordinate used to distinguish rooms
            # vertical wall at x = wall_pos -> use x (0)
            # horizontal wall at y = wall_pos -> use y (1)
            room_idx = 1 if wall_axis == 0 else 0

            agent_coord = agent_pos[room_idx]
            goal_coord = goal_pos[room_idx]

            goal_other_room = (agent_coord < wall_pos and goal_coord > wall_pos) or (
                agent_coord > wall_pos and goal_coord < wall_pos
            )

            if goal_other_room:
                # --- go to the closest door that fits the agent ---
                number = env.variation_space.value["door"]["number"]
                positions = env.variation_space.value["door"]["position"]
                sizes = env.variation_space.value["door"]["size"]
                agent_radius = env.variation_space.value["agent"]["radius"].item()
                border_size = env.border_size

                best_center = None
                best_dist = float("inf")

                for pos_1d, size in zip(positions[:number], sizes[:number]):
                    # door must be wide enough
                    if size < 2.5 * agent_radius:
                        continue

                    # 1D center along the wall's orthogonal axis
                    center_t = pos_1d + size / 2.0

                    if wall_axis == 1:
                        # vertical wall at x = wall_pos, door spans along y
                        door_center = np.array(
                            [wall_pos, border_size + center_t],
                            dtype=np.float32,
                        )
                    else:
                        # horizontal wall at y = wall_pos, door spans along x
                        door_center = np.array(
                            [border_size + center_t, wall_pos],
                            dtype=np.float32,
                        )

                    dist = np.linalg.norm(door_center - agent_pos)
                    if dist < best_dist:
                        best_dist = dist
                        best_center = door_center

                if best_center is None:
                    # Fallback: aim for a point on the wall aligned with the goal
                    if wall_axis == 1:
                        target = np.array([wall_pos, goal_pos[1]], dtype=np.float32)
                    else:
                        target = np.array([goal_pos[0], wall_pos], dtype=np.float32)
                else:
                    target = best_center
            else:
                # --- already on the same side as the goal: go directly to goal ---
                target = goal_pos

            # --- turn target point into an action vector ---
            direction = target - agent_pos
            norm = np.linalg.norm(direction)

            if norm > 1e-8:
                if norm > max_norm:
                    direction = direction / norm * max_norm
            else:
                direction = np.zeros_like(direction)

            actions[i] = direction.astype(np.float32)

        actions += np.random.normal(0, self.action_noise, size=actions.shape)
        return actions
