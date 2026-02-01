import numpy as np
from stable_worldmodel.policy import BasePolicy


class ExpertPolicy(BasePolicy):
    """Expert Policy for Two Room Environment.

    Behavior:
      - If target is in other room: go to center of closest door that fits -> then go to target.
      - Otherwise: go directly to target.
    """

    def __init__(
        self,
        action_noise: float = 0.0,
        action_repeat_prob: float = 0.0,
        door_fit_margin: float = 1.10,
        door_reach_tol: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type = 'expert'
        self.action_noise = float(action_noise)
        self.door_fit_margin = float(door_fit_margin)
        # If None, will default to ~3*scale per-env at runtime
        self.door_reach_tol = door_reach_tol

    def set_env(self, env):
        self.env = env

    def get_action(self, info_dict, **kwargs):
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        assert 'pos_agent' in info_dict, (
            "'pos_agent' must be provided in info_dict"
        )
        assert 'pos_target' in info_dict, (
            "'pos_target' must be provided in info_dict"
        )

        base_env = self.env.unwrapped
        if hasattr(base_env, 'envs'):
            envs = [e.unwrapped for e in base_env.envs]
            is_vectorized = True
        else:
            envs = [base_env]
            is_vectorized = False

        actions = np.zeros(self.env.action_space.shape, dtype=np.float32)

        for i, env in enumerate(envs):
            if is_vectorized:
                agent_pos = np.asarray(
                    info_dict['pos_agent'][i], dtype=np.float32
                ).squeeze()
                pos_target = np.asarray(
                    info_dict['pos_target'][i], dtype=np.float32
                ).squeeze()
            else:
                agent_pos = np.asarray(
                    info_dict['pos_agent'], dtype=np.float32
                ).squeeze()
                pos_target = np.asarray(
                    info_dict['pos_target'], dtype=np.float32
                ).squeeze()

            # --- environment params (avoid env.variation_space.value; use .value fields) ---
            wall_axis = int(
                env.variation_space['wall']['axis'].value
            )  # 1 vertical, 0 horizontal
            wall_pos = float(env.wall_pos)  # must match env physics/collision

            # Room is determined by x if vertical wall, by y if horizontal wall
            room_idx = 0 if wall_axis == 1 else 1

            agent_side = agent_pos[room_idx] > wall_pos
            target_side = pos_target[room_idx] > wall_pos
            target_other_room = agent_side != target_side

            waypoint = None

            if target_other_room:
                # Doors: interpret consistently with env:
                # - position: center coordinate along wall direction (no border offset)
                # - size: half-extent along wall direction
                num = int(env.variation_space['door']['number'].value)
                door_pos = np.asarray(
                    env.variation_space['door']['position'].value,
                    dtype=np.float32,
                )[:num]
                door_size = np.asarray(
                    env.variation_space['door']['size'].value, dtype=np.float32
                )[:num]
                agent_radius = float(
                    env.variation_space['agent']['radius'].value.item()
                )

                # Choose closest door center that fits
                best = None
                best_dist = float('inf')

                for c_1d, s in zip(door_pos, door_size):
                    # Fit test: opening half-extent should exceed radius (with margin)
                    if float(s) < self.door_fit_margin * agent_radius:
                        continue

                    if wall_axis == 1:
                        # vertical wall at x=wall_pos, door center at y=c_1d
                        door_center = np.array(
                            [wall_pos, float(c_1d)], dtype=np.float32
                        )
                    else:
                        # horizontal wall at y=wall_pos, door center at x=c_1d
                        door_center = np.array(
                            [float(c_1d), wall_pos], dtype=np.float32
                        )

                    d = float(np.linalg.norm(door_center - agent_pos))
                    if d < best_dist:
                        best_dist = d
                        best = door_center

                if best is None:
                    # Fallback: go to the wall aligned with target (still better than nothing)
                    if wall_axis == 1:
                        waypoint = np.array(
                            [wall_pos, pos_target[1]], dtype=np.float32
                        )
                    else:
                        waypoint = np.array(
                            [pos_target[0], wall_pos], dtype=np.float32
                        )
                else:
                    # Two-stage: go to door center first; once close, go to target
                    tol = (
                        float(self.door_reach_tol)
                        if self.door_reach_tol is not None
                        else float(3.0 * env.scale)
                    )
                    if np.linalg.norm(best - agent_pos) > tol:
                        waypoint = best
                    else:
                        waypoint = pos_target
            else:
                waypoint = pos_target

            # --- convert waypoint to action direction (unit vector) ---
            direction = waypoint - agent_pos
            norm = float(np.linalg.norm(direction))
            if norm > 1e-8:
                direction = direction / norm
            else:
                direction = np.zeros_like(direction, dtype=np.float32)

            if is_vectorized:
                actions[i] = direction.astype(np.float32)
            else:
                actions = direction.astype(np.float32)

        if self.action_noise > 0:
            actions = actions + np.random.normal(
                0.0, self.action_noise, size=actions.shape
            ).astype(np.float32)

        # action repeat stochasticity
        self._last_action = getattr(self, '_last_action', None)
        if self._last_action is not None and self.action_repeat_prob > 0.0:
            repeat_mask = (
                np.random.uniform(
                    0.0, 1.0, size=(actions.shape[0],) if is_vectorized else ()
                )
                < self.action_repeat_prob
            )
            if is_vectorized:
                actions[repeat_mask] = self._last_action[repeat_mask]
            else:
                if repeat_mask:
                    actions = self._last_action

        # Keep within action space
        return np.clip(actions, -1.0, 1.0).astype(np.float32)
