## -- Policy
### BasePolicy
### RandomPolicy
### OptimalPolicy (Expert)
### PlanningPolicy (wm, solver)


class BasePolicy:
    """Base class for agent policies"""

    # a policy takes in an environment and a planner
    def __init__(self, env):
        raise NotImplementedError

    def get_action(self, states, **kwargs):
        """Get action from the policy given the state"""
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    def __init__(self, env):
        self.env = env

    def get_action(self, states, **kwargs):
        return self.env.action_space.sample()


class OptimalPolicy(BasePolicy):
    def __init__(self, env):
        self.env = env

    def get_action(self, states, **kwargs):
        # Implement optimal policy logic here
        pass


class PlanningPolicy(BasePolicy):
    def __init__(self, env, planning_solver):
        self.env = env
        self.solver = planning_solver  # leverage to determine the best action

    def get_action(self, states, goals, **kwargs):
        # Implement planning policy logic here
        pass
