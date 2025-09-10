## -- Policy
### BasePolicy
### RandomPolicy
### OptimalPolicy (Expert)
### PlanningPolicy (wm, solver)

import numpy as np


class BasePolicy:
    """Base class for agent policies"""

    # a policy takes in an environment and a planner
    def __init__(self, **kwargs):
        self.env = None
        for arg, value in kwargs.items():
            setattr(self, arg, value)

    def get_action(self, obs, **kwargs):
        """Get action from the policy given the observation"""
        raise NotImplementedError

    def set_env(self, env):
        self.env = env


class RandomPolicy(BasePolicy):
    def get_action(self, obs, **kwargs):
        return self.env.action_space.sample()


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
