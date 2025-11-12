from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

import stable_worldmodel as swm
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


class Transformable(Protocol):
    """Protocol for input transformation."""

    def transform(x) -> torch.Tensor:  # pragma: no cover
        """Pre-process"""
        ...

    def inverse_transform(x) -> torch.Tensor:  # pragma: no cover
        """Revert pre-processed"""
        ...


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
        process: dict[str, Transformable] | None = None,
        transform: dict[str, callable] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.type = "world_model"
        self.cfg = config
        self.solver = solver
        self.action_buffer = deque(maxlen=self.flatten_receding_horizon)
        self.process = process or {}
        self.transform = transform or {}
        self._action_buffer = None
        self._next_init = None

    @property
    def flatten_receding_horizon(self):
        return self.cfg.receding_horizon * self.cfg.action_block

    def set_env(self, env):
        self.env = env
        n_envs = getattr(env, "num_envs", 1)
        self.solver.configure(action_space=env.action_space, n_envs=n_envs, config=self.cfg)
        self._action_buffer = deque(maxlen=self.flatten_receding_horizon)

        assert isinstance(self.solver, Solver), "Solver must implement the Solver protocol"

    def _prepare_info(self, info_dict):
        # pre-process and transform observations
        for k, v in info_dict.items():
            is_numpy = isinstance(v, (np.ndarray | np.generic))

            if k in self.process:
                if not is_numpy:
                    raise ValueError(f"Expected numpy array for key '{k}' in process, got {type(v)}")

                # flatten extra dimensions if needed
                shape = v.shape
                if len(shape) > 2:
                    v = v.reshape(-1, *shape[2:])

                # process and reshape back
                v = self.process[k].transform(v)
                v = v.reshape(shape)

            # collapse env and time dimensions for transform (e, t, ...) -> (e * t, ...)
            # then restore after transform
            if k in self.transform:
                shape = None
                if is_numpy or torch.is_tensor(v):
                    if v.ndim > 2:
                        shape = v.shape
                        v = v.reshape(-1, *shape[2:])

                v = torch.stack([self.transform[k](x) for x in v])
                is_numpy = isinstance(v, (np.ndarray | np.generic))

                if shape is not None:
                    v = v.reshape(*shape[:2], *v.shape[1:])

            if is_numpy and v.dtype != object:
                v = torch.from_numpy(v)

            info_dict[k] = v

        return info_dict

    def get_action(self, info_dict, **kwargs):
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert "pixels" in info_dict, "'pixels' must be provided in info_dict"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"

        info_dict = self._prepare_info(info_dict)

        # need to replan if action buffer is empty
        if len(self._action_buffer) == 0:
            outputs = self.solver(info_dict, init_action=self._next_init)

            actions = outputs["actions"]  # (num_envs, horizon, action_dim)
            keep_horizon = self.cfg.receding_horizon
            plan = actions[:, :keep_horizon]
            rest = actions[:, keep_horizon:]
            self._next_init = rest if self.cfg.warm_start else None

            # frameskip back to timestep
            plan = plan.reshape(self.env.num_envs, self.flatten_receding_horizon, -1)

            self._action_buffer.extend(plan.transpose(0, 1))

        action = self._action_buffer.popleft()
        action = action.reshape(*self.env.action_space.shape)
        action = action.numpy()

        # post-process action
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)

        return action  # (num_envs, action_dim)


def AutoCostModel(model_name, cache_dir=None):
    cache_dir = Path(cache_dir or swm.data.get_cache_dir())
    path = cache_dir / f"{model_name}_object.ckpt"
    assert path.exists(), f"World model named {model_name} not found. Should launch pretraining first."

    spt_module = torch.load(path, weights_only=False)

    def scan_module(module):
        if hasattr(module, "get_cost"):
            if isinstance(module.get_cost, torch.nn.Module):
                module = module.eval()
            return module
        for child in module.children():
            result = scan_module(child)
            if result is not None:
                return result
        return None

    result = scan_module(spt_module)
    if result is not None:
        return result

    raise RuntimeError("No cost model found in the loaded world model.")
