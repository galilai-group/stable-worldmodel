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
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type = 'expert'
        self.action_noise = float(action_noise)
        self.action_repeat_prob = float(action_repeat_prob)
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
            # pixel offsets below scale with render size (1.0 @ the 65px ref)
            scale = float(getattr(env, '_scale', 1.0))

            # perp_idx = the axis the dot crosses to change rooms; par_idx
            # = the axis the wall runs along (and doors vary on).
            perp_idx = 0 if wall_axis == 1 else 1
            par_idx = 1 - perp_idx

            agent_side = agent_pos[perp_idx] > wall_pos
            target_side = goal_pos[perp_idx] > wall_pos
            # "Still crossing" until the agent is clear of the wall slab on the
            # target's side. Treat being inside the wall band [near, far] as
            # not-yet-arrived: otherwise the moment the agent's perp passes the
            # centerline it declares "same room" and beelines diagonally toward
            # the target while still embedded in the doorway -> clips a doorpost.
            in_wall_band = near <= float(agent_pos[perp_idx]) <= far
            target_other_room = in_wall_band or (agent_side != target_side)

            if not has_key:
                # Stage 1: pick up the key. Key is constrained to the agent's
                # room, so a straight line is always valid (no door routing).
                waypoint = key_pos
            elif not target_other_room:
                # Stage 3: same room as target (and clear of the wall): straight.
                waypoint = goal_pos
            else:
                # Stage 2: two-phase door traversal (ALIGN, then CROSS). Aiming
                # straight at the door *center* and crossing diagonally jams the
                # agent against the SOLID wall whenever its par-coord is outside
                # the opening (the path reaches the wall before the agent slides
                # into the doorway), and the collision pushback then cancels its
                # motion -> a frozen fixed point. So first ALIGN to the door on
                # our own side (a path that never crosses the wall), then CROSS
                # straight through with par locked to the door center.
                num = int(env.variation_space['door']['number'].value)
                door_pos = np.asarray(
                    env.variation_space['door']['position'].value,
                    dtype=np.float32,
                )[:num]
                door_size = np.asarray(
                    env.variation_space['door']['size'].value, dtype=np.float32
                )[:num]

                # Pick the door minimizing total path length
                # (agent -> door -> target). The agent is a POINT, so a door is
                # usable as long as its opening — clipped to the playable area —
                # has positive width; route by the center of that usable slot
                # (aiming at a door's geometric center can land under the border
                # and jam the agent at the wall/border corner).
                low_play = float(env.BORDER_SIZE)
                high_play = float(env.IMG_SIZE - env.BORDER_SIZE)
                best_par = None
                best_half = 0.0
                best_total = float('inf')
                for c_1d, s in zip(door_pos, door_size):
                    slot_lo = max(float(c_1d) - float(s), low_play)
                    slot_hi = min(float(c_1d) + float(s), high_play)
                    if (slot_hi - slot_lo) <= 1.0 * scale:
                        continue  # opening lies entirely under the border
                    par_center = 0.5 * (slot_lo + slot_hi)
                    half_usable = 0.5 * (slot_hi - slot_lo)
                    if wall_axis == 1:
                        door_pt = np.array(
                            [wall_pos, par_center], dtype=np.float32
                        )
                    else:
                        door_pt = np.array(
                            [par_center, wall_pos], dtype=np.float32
                        )
                    total = float(
                        np.linalg.norm(door_pt - agent_pos)
                    ) + float(np.linalg.norm(goal_pos - door_pt))
                    if total < best_total:
                        best_total = total
                        best_par = par_center
                        best_half = half_usable

                waypoint = np.empty(2, dtype=np.float32)
                if best_par is None:
                    # No door fits the agent — degenerate; aim at the wall in
                    # line with the target so we at least try.
                    waypoint[perp_idx] = wall_pos
                    waypoint[par_idx] = goal_pos[par_idx]
                else:
                    # Collision is point-based (the env collides the dot's
                    # CENTER, with no agent-radius padding), so the crossing
                    # geometry uses small scale buffers, not the agent radius:
                    # clear the wall face by a hair, and require the center to be
                    # inside the usable opening (minus a small lip buffer) before
                    # committing to the straight-through crossing.
                    clear = half_w + 2.0 * scale
                    half_in = max(0.5 * scale, best_half - 1.0 * scale)
                    aligned = (
                        abs(float(agent_pos[par_idx]) - best_par) <= half_in
                    )
                    on_side = target_side if aligned else agent_side
                    waypoint[par_idx] = best_par
                    waypoint[perp_idx] = (
                        wall_pos + clear if on_side else wall_pos - clear
                    )

            # --- convert waypoint to an action ---
            # Proportional speed near the waypoint: scale the unit heading by
            # min(1, dist / speed) so the agent settles exactly onto staging
            # points and the door-center par instead of overshooting (bang-bang
            # oscillation). This is what lets it thread narrow / fast-speed
            # doorways rather than orbiting them.
            speed = float(env.variation_space['agent']['speed'].value.item())
            delta = waypoint - agent_pos
            norm = float(np.linalg.norm(delta))
            if norm > 1e-8:
                mag = min(1.0, norm / max(speed, 1e-6))
                direction = (delta / norm) * mag
            else:
                direction = np.zeros_like(delta, dtype=np.float32)

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
