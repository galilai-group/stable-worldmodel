import numpy as np
from stable_worldmodel.policy import BasePolicy


class DoorKeyExpertPolicy(BasePolicy):
    """Expert policy for the TwoRoomDoorKey environment.

    Behavior:
      - If the agent has not picked up the key yet, head straight to the key.
      - Once the key is picked up: if the target is in the other room, route
        through the closest door opening that fits the agent.
      - Otherwise (already in target's room), head straight to the target.
    """

    def __init__(
        self,
        action_noise: float = 0.0,
        action_repeat_prob: float = 0.0,
        door_fit_margin: float = 1.10,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type = 'expert'
        self.action_noise = float(action_noise)
        self.action_repeat_prob = float(action_repeat_prob)
        self.door_fit_margin = float(door_fit_margin)
        self.set_seed(seed)

    def set_seed(self, seed: int | None) -> None:
        """Set the random seed for action sampling.

        Args:
            seed: The seed value.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_env(self, env):
        self.env = env

    def get_action(self, info_dict, **kwargs):
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        assert 'state' in info_dict, "'state' must be provided in info_dict"
        assert 'goal_state' in info_dict, (
            "'goal_state' must be provided in info_dict"
        )
        assert 'key_position' in info_dict, (
            "'key_position' must be provided in info_dict"
        )
        assert 'has_key' in info_dict, (
            "'has_key' must be provided in info_dict"
        )

        if hasattr(self.env, 'envs'):
            envs = [e.unwrapped for e in self.env.envs]
            is_vectorized = True
        else:
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
                    info_dict['state'][i], dtype=np.float32
                ).squeeze()
                goal_pos = np.asarray(
                    info_dict['goal_state'][i], dtype=np.float32
                ).squeeze()
                key_pos = np.asarray(
                    info_dict['key_position'][i], dtype=np.float32
                ).squeeze()
                has_key = bool(
                    np.asarray(info_dict['has_key'][i]).squeeze() > 0.5
                )
            else:
                agent_pos = np.asarray(
                    info_dict['state'], dtype=np.float32
                ).squeeze()
                goal_pos = np.asarray(
                    info_dict['goal_state'], dtype=np.float32
                ).squeeze()
                key_pos = np.asarray(
                    info_dict['key_position'], dtype=np.float32
                ).squeeze()
                has_key = bool(
                    np.asarray(info_dict['has_key']).squeeze() > 0.5
                )

            # --- environment params (avoid env.variation_space.value; use .value fields) ---
            wall_axis = int(
                env.variation_space['wall']['axis'].value
            )  # 1 vertical, 0 horizontal
            wall_pos = float(env.wall_pos)  # must match env physics/collision
            wall_thickness = int(
                env.variation_space['wall']['thickness'].value
            )
            half_w = wall_thickness // 2
            near = wall_pos - half_w
            far = wall_pos + half_w

            # perp_idx = the axis the dot crosses to change rooms; par_idx
            # = the axis the wall runs along (and doors vary on).
            perp_idx = 0 if wall_axis == 1 else 1
            par_idx = 1 - perp_idx

            agent_side = agent_pos[perp_idx] > wall_pos
            target_side = goal_pos[perp_idx] > wall_pos
            target_other_room = agent_side != target_side

            waypoint = None
            best = None  # chosen door center, if any

            if not has_key:
                # Stage 1: pick up the key. Key is constrained to the agent's
                # room, so a straight line is always valid (no door routing).
                waypoint = key_pos
            elif target_other_room:
                # Stage 2: route through a fitting door, then to target.
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

                # Pick the door minimizing total path length
                # (agent -> door -> target). Avoids choosing a door close to
                # the agent that requires a diagonal exit clipping the
                # far doorpost.
                best_total = float('inf')
                for c_1d, s in zip(door_pos, door_size):
                    if float(s) < self.door_fit_margin * agent_radius:
                        continue
                    if wall_axis == 1:
                        door_center = np.array(
                            [wall_pos, float(c_1d)], dtype=np.float32
                        )
                    else:
                        door_center = np.array(
                            [float(c_1d), wall_pos], dtype=np.float32
                        )
                    total = float(
                        np.linalg.norm(door_center - agent_pos)
                    ) + float(np.linalg.norm(goal_pos - door_center))
                    if total < best_total:
                        best_total = total
                        best = door_center

                if best is None:
                    # Fallback: aim at the wall aligned with target.
                    if wall_axis == 1:
                        waypoint = np.array(
                            [wall_pos, goal_pos[1]], dtype=np.float32
                        )
                    else:
                        waypoint = np.array(
                            [goal_pos[0], wall_pos], dtype=np.float32
                        )
                else:
                    waypoint = best  # default: approach the chosen door
            else:
                # Stage 3: same room as target after pickup.
                waypoint = goal_pos

            # In-wall-band override (only relevant in Stage 2): once the agent
            # is inside the wall thickness, drive perpendicular straight
            # through, locking the par-axis to the chosen door's center.
            agent_perp = float(agent_pos[perp_idx])
            if best is not None and (near - 1.0) <= agent_perp <= (far + 1.0):
                exit_perp = (
                    far + 5.0 if goal_pos[perp_idx] > wall_pos else near - 5.0
                )
                waypoint = np.empty(2, dtype=np.float32)
                waypoint[perp_idx] = exit_perp
                waypoint[par_idx] = best[par_idx]

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
            actions = actions + self.rng.normal(
                0.0, self.action_noise, size=actions.shape
            ).astype(np.float32)

        # action repeat stochasticity
        self._last_action = getattr(self, '_last_action', None)
        if self._last_action is not None and self.action_repeat_prob > 0.0:
            repeat_mask = (
                self.rng.uniform(
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
