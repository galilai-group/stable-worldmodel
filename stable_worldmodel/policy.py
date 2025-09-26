import numpy as np
from collections import deque
from typing import Protocol, runtime_checkable
import torch
import gymnasium as gym

from dataclasses import dataclass
from stable_worldmodel.solver import Solver


@dataclass(frozen=True)
class PlanConfig:
    """Configuration for the planning process."""

    horizon: int
    receding_horizon: int
    history_len: int = 1
    action_block: int = 1  # frameskip
    warm_start: bool = True  # use previous plan to warm start

    @property
    def plan_len(self):
        return self.horizon * self.action_block


class BasePolicy:
    """Base class for agent policies."""

    # a policy takes in an environment and a planner
    def __init__(self, **kwargs):
        self.env = None
        self.type = "base"
        for arg, value in kwargs.items():
            setattr(self, arg, value)

    def get_action(self, obs, **kwargs):
        """Get action from the policy given the observation."""
        raise NotImplementedError

    def set_env(self, env):
        self.env = env


class RandomPolicy(BasePolicy):
    """Random Policy."""

    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.type = "random"
        self.seed = seed

    def get_action(self, obs, **kwargs):
        return self.env.action_space.sample()

    def set_seed(self, seed):
        if self.env is not None:
            self.env.action_space.seed(seed)


class ExpertPolicy(BasePolicy):
    """Expert Policy."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "expert"

    def get_action(self, obs, goal_obs, **kwargs):
        # Implement expert policy logic here
        pass


class WorldModelPolicy(BasePolicy):
    """World Model Policy using a planning solver."""

    def __init__(
        self,
        solver: Solver,
        config: PlanConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.type = "world_model"
        self.cfg = config
        self.solver = solver
        self.action_buffer = deque(maxlen=self.cfg.receding_horizon)
        self._action_buffer = None
        self._next_init = None

    def set_env(self, env):
        self.env = env
        n_envs = getattr(env, "num_envs", 1)
        self.solver.configure(
            action_space=env.action_space, n_envs=n_envs, config=self.cfg
        )
        self._action_buffer = deque(maxlen=self.cfg.receding_horizon)

        assert isinstance(self.solver, Solver), (
            "Solver must implement the Solver protocol"
        )

    def get_action(self, info_dict, **kwargs):
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert "pixels" in info_dict, "'pixels' must be provided in info_dict"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"

        # need to replan if action buffer is empty
        if len(self._action_buffer) == 0:
            outputs = self.solver(info_dict, init_action=self._next_init)

            actions = outputs["actions"]  # (num_envs, horizon, action_dim)
            keep_horizon = self.cfg.receding_horizon

            plan = actions[:, :keep_horizon]
            rest = actions[:, keep_horizon:]

            self._next_init = rest if self.cfg.warm_start else None
            self._action_buffer.extend(plan.transpose(0, 1))

        action = self._action_buffer.popleft()

        action = action.reshape(*self.env.action_space.shape)

        return action.numpy()  # (num_envs, action_dim)


def AutoPolicy(torchscript_name, **kwargs):
    return  # TODO load the torchscript policy
