import numpy as np
from stable_worldmodel.policy import BasePolicy


class ExpertPolicy(BasePolicy):
    """Expert policy for PiecewiseEnv (piecewise dynamics).

    Solves the per-step motion equation exactly:
        pos_next = pos + action * speed + bias[zone]
    → optimal action = (goal - pos - bias[zone]) / speed, clamped to [-1, 1].
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
            else:
                agent_pos = np.asarray(
                    info_dict['state'], dtype=np.float32
                ).squeeze()
                goal_pos = np.asarray(
                    info_dict['goal_state'], dtype=np.float32
                ).squeeze()

            speed = float(env.variation_space['agent']['speed'].value.item())
            zone_idx = _get_zone(
                agent_pos, env.grid_n, env.BORDER_SIZE, env.IMG_SIZE
            )
            bias = np.asarray(
                env.variation_space['zones'][f'bias_{zone_idx}'].value,
                dtype=np.float32,
            )

            # Invert pos_next = pos + action * speed + bias
            action = (goal_pos - agent_pos - bias) / speed

            if is_vectorized:
                actions[i] = action
            else:
                actions = action

        if self.action_noise > 0:
            actions = actions + self.rng.normal(
                0.0, self.action_noise, size=actions.shape
            ).astype(np.float32)

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

        actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
        self._last_action = actions
        return actions


def _get_zone(
    pos: np.ndarray, grid_n: int, border_size: int, img_size: int
) -> int:
    """Mirror of PiecewiseEnv._get_zone for numpy positions."""
    play_size = float(img_size - 2 * border_size)
    col = int((float(pos[0]) - border_size) / play_size * grid_n)
    row = int((float(pos[1]) - border_size) / play_size * grid_n)
    col = max(0, min(grid_n - 1, col))
    row = max(0, min(grid_n - 1, row))
    return row * grid_n + col
