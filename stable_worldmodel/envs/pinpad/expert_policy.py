import numpy as np
from stable_worldmodel.policy import BasePolicy


def compute_action_discrete(agent_position, goal_position):
    dx, dy = (goal_position - agent_position).tolist()
    if abs(dx) + abs(dy):
        # Gets directions we need to move in (in the transposed space)
        possible_actions = []
        if abs(dx):
            if dx > 0:
                possible_actions.append(3)  # right
            else:
                possible_actions.append(4)  # left
        if abs(dy):
            if dy > 0:
                possible_actions.append(1)  # up
            else:
                possible_actions.append(2)  # down

        # Alternates between horizontal and vertical moves
        if len(possible_actions) == 2:
            action = possible_actions[(abs(dx) + abs(dy)) % 2]
        else:
            action = possible_actions[0]
    else:
        action = 0
    return action


def compute_action_continuous(
    agent_position, goal_position, max_norm, add_noise, rng
):
    delta = goal_position - agent_position
    if add_noise:
        delta = delta + rng.normal(0, 1, delta.shape)

    if np.linalg.norm(delta) > max_norm:
        action = max_norm * delta / np.linalg.norm(delta)  # Clips norm
    else:
        action = delta
    return action


def get_action(info_dict, env, env_type, **kwargs):
    # Check if environment is vectorized
    base_env = env.unwrapped
    if hasattr(base_env, 'envs'):
        envs = [e.unwrapped for e in base_env.envs]
        is_vectorized = True
    else:
        envs = [base_env]
        is_vectorized = False

    # Computes actions for each environment
    actions = []
    dtype = np.int64 if env_type == 'discrete' else np.float64
    for i, env in enumerate(envs):
        if is_vectorized:
            agent_position = np.asarray(
                info_dict['agent_position'][i], dtype=dtype
            ).squeeze()
            goal_position = np.asarray(
                info_dict['goal_position'][i], dtype=dtype
            ).squeeze()
        else:
            agent_position = np.asarray(
                info_dict['agent_position'], dtype=dtype
            ).squeeze()
            goal_position = np.asarray(
                info_dict['goal_position'], dtype=dtype
            ).squeeze()

        if env_type == 'discrete':
            actions.append(
                compute_action_discrete(agent_position, goal_position)
            )
        elif env_type == 'continuous':
            actions.append(
                compute_action_continuous(
                    agent_position,
                    goal_position,
                    kwargs['max_norm'],
                    kwargs['add_noise'],
                    kwargs['rng'],
                )
            )
        else:
            raise ValueError(f'Invalid environment type: {env_type}')

    actions = np.array(actions)
    return actions if is_vectorized else actions[0]


class ExpertPolicyDiscrete(BasePolicy):
    """Expert policy for the PinPadDiscrete environment."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = 'expert'

    def get_action(self, info_dict, **kwargs):
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        assert 'agent_position' in info_dict, (
            'Agent position must be provided in info_dict'
        )
        assert 'goal_position' in info_dict, (
            'Goal position must be provided in info_dict'
        )

        return get_action(info_dict, self.env, 'discrete', **kwargs)


class ExpertPolicy(BasePolicy):
    """Expert policy for the PinPad environment."""

    def __init__(self, max_norm=1.0, add_noise=True, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.type = 'expert'
        self.max_norm = max_norm
        self.add_noise = add_noise
        self.rng = np.random.default_rng(seed)

    def get_action(self, info_dict, **kwargs):
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        assert 'agent_position' in info_dict, (
            'Agent position must be provided in info_dict'
        )
        assert 'goal_position' in info_dict, (
            'Goal position must be provided in info_dict'
        )

        kwargs['max_norm'] = self.max_norm
        kwargs['add_noise'] = self.add_noise
        kwargs['rng'] = self.rng
        return get_action(info_dict, self.env, 'continuous', **kwargs)
