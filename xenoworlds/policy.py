## -- Policy
### BasePolicy
### RandomPolicy
### OptimalPolicy (Expert)
### PlanningPolicy (wm, solver)

import numpy as np


class BasePolicy:
    """Base class for agent policies"""

    # a policy takes in an environment and a planner
    def __init__(self, env, horizon=1, **kwargs):
        self.env = env
        self.horizon = horizon

    def get_action(self, obs, goal_obs, **kwargs):
        """Get action from the policy given the observation"""
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def get_action(self, obs, goal_obs, **kwargs):
        action_seq = []
        for step in range(self.horizon):
            action = self.env.action_space.sample()
            action_seq.append(action)

        action_seq = np.stack(action_seq, axis=1)

        if action_seq.ndim == 2:
            action_seq = action_seq[:, np.newaxis, :]

        return action_seq


class OptimalPolicy(BasePolicy):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def get_action(self, obs, goal_obs, **kwargs):
        # Implement optimal policy logic here
        pass


class PlanningPolicy(BasePolicy):
    def __init__(self, env, planning_solver, **kwargs):
        super().__init__(env, **kwargs)
        self.solver = planning_solver

    def get_action(self, obs, goal_obs, **kwargs):
        return self.solver(obs, self.env.action_space, goal_obs)
