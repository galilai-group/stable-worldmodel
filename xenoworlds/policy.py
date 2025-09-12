import numpy as np
from collections import deque

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

    def set_seed(self, seed):
        if self.env is not None:
            self.env.action_space.seed(seed)

class ExpertPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "expert"

    def get_action(self, obs, goal_obs, **kwargs):
        # Implement expert policy logic here
        pass

class WorldModelPolicy(BasePolicy):
    def __init__(self,
                world_model,
                solver,
                horizon=5,
                action_block=5,
                history_len=3,
                receding_horizon=10,
                **kwargs):
        super().__init__(**kwargs)

        self.type = "world_model"
        self.solver = solver
        self.world_model = world_model
        
        # planning horizon
        self.horizon = horizon     

        # number of actions to plan at once (frameskip)
        # e.g horizon=5,action_chunk=2 -> optimize 10 actions
        self.action_block = action_block

        # maximum history length for world model predictor
        self.history_len = history_len

        # receding horizon steps to take before re-planning (mpc)
        # rem: 1 <= receding_horizon <= action_block * horizon
        self.receding_horizon = receding_horizon

        self.action_buffer = deque(maxlen=self.receding_horizon)

    @property
    def plan_len(self):
        return self.horizon * self.action_block
    
    @property
    def action_dim(self):
        return np.prod(self.env.single_action_space.shape)

    def plan(self, obs, goal):
        # call the solver to get a plan
        #return self.solver(obs, self.env.action_space, goal)
        num_envs = self.env.num_envs
        return np.random.uniform(
            low=self.env.single_action_space.low,
            high=self.env.single_action_space.high,
            size=(num_envs, self.plan_len, self.action_dim)
        ).astype(np.float32)

    def get_action(self, obs, goal=None, **kwargs):
        assert hasattr(self, "env"), "Environment not set for the policy"

        # need to replan if action buffer is empty
        if len(self.action_buffer) == 0:
            plan = self.plan(obs, goal)
            # keep only the receding horizon steps
            for action in range(self.receding_horizon):
                self.action_buffer.append(plan[:, action])

        action = self.action_buffer.popleft()

        # TODO: reshape action shape into its original shape if needed
        return action # (num_envs, action_dim)
