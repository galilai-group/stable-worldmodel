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
        self.type= "base"
        for arg, value in kwargs.items():
            setattr(self, arg, value)

    def get_action(self, obs, **kwargs):
        """Get action from the policy given the observation"""
        raise NotImplementedError

    def set_env(self, env):
        self.env = env


class RandomPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "random"

    def get_action(self, obs, **kwargs):
        return self.env.action_space.sample()


class ExpertPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "expert"

    def get_action(self, obs, goal_obs, **kwargs):
        # Implement expert policy logic here
        pass

class WorldModelPolicy(BasePolicy):
    def __init__(self, world_model, solver, **kwargs):
        super().__init__(**kwargs)
        # TODO add param like horizon
        # action chunk
        # mpc etc...
        self.type = "world_model"
        self.solver = solver
        self.world_model = world_model
        
    def get_action(self, obs, goal_obs, **kwargs):
        return self.solver(obs, self.env.action_space, goal_obs)

