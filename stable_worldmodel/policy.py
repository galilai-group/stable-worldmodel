from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from loguru import logger as logging
from torchvision import tv_tensors

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


class Actionable(Protocol):
    """Protocol for model action computation."""

    def get_action(info) -> torch.Tensor:  # pragma: no cover
        """Compute action from observation and goal"""
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

    def _prepare_info(self, info_dict):
        # pre-process and transform observations
        for k, v in info_dict.items():
            is_numpy = isinstance(v, (np.ndarray | np.generic))

            if hasattr(self, "process") and k in self.process:
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
            if hasattr(self, "transform") and k in self.transform:
                shape = None
                if is_numpy or torch.is_tensor(v):
                    if v.ndim > 2:
                        shape = v.shape
                        v = v.reshape(-1, *shape[2:])
                if k.startswith("pixels") or k.startswith("goal"):
                    # permute channel first for transform
                    if is_numpy:
                        v = np.transpose(v, (0, 3, 1, 2))
                    else:
                        v = v.permute(0, 3, 1, 2)
                v = torch.stack([self.transform[k](tv_tensors.Image(x)) for x in v])
                is_numpy = isinstance(v, (np.ndarray | np.generic))

                if shape is not None:
                    v = v.reshape(*shape[:2], *v.shape[1:])

            if is_numpy and v.dtype.kind not in "USO":
                v = torch.from_numpy(v)

            info_dict[k] = v

        return info_dict


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


class FeedForwardPolicy(BasePolicy):
    """Feed-Forward Policy using a neural network model. Actions are computed via a single forward pass."""

    def __init__(
        self,
        model: Actionable,
        process: dict[str, Transformable] | None = None,
        transform: dict[str, callable] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type = "feed_forward"
        self.model = model.eval()
        self.process = process or {}
        self.transform = transform or {}

    def get_action(self, info_dict, **kwargs):
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"

        # Prepare the info dict (transforms and normalizes inputs)
        info_dict = self._prepare_info(info_dict)

        # Add goal_pixels key for GCBC model
        if "goal" in info_dict:
            info_dict["goal_pixels"] = info_dict["goal"]

        # Move all tensors to the model's device
        device = next(self.model.parameters()).device
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                info_dict[k] = v.to(device)

        # Get action from model
        with torch.no_grad():
            action = self.model.get_action(info_dict)

        # Convert to numpy
        if torch.is_tensor(action):
            action = action.cpu().detach().numpy()

        # post-process action
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)

        return action


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


def _load_model_with_attribute(run_name, attribute_name, cache_dir=None):
    """Helper function to load a model checkpoint and find a module with the specified attribute.

    Args:
        run_name: Path or name of the model run
        attribute_name: Name of the attribute to look for in the module (e.g., 'get_action', 'get_cost')
        cache_dir: Optional cache directory path

    Returns:
        The module with the specified attribute

    Raises:
        RuntimeError: If no module with the specified attribute is found
    """
    if Path(run_name).exists():
        run_path = Path(run_name)
    else:
        run_path = Path(cache_dir or swm.data.utils.get_cache_dir(), run_name)

    if run_path.is_dir():
        ckpt_files = list(run_path.glob("*_object.ckpt"))
        ckpt_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
        path = ckpt_files[0]
        logging.info(f"Loading model from checkpoint: {path}")
    else:
        path = Path(f"{run_path}_object.ckpt")
        assert path.exists(), "Checkpoint path does not exist: {path}. Launch pretraining first."

    spt_module = torch.load(path, weights_only=False, map_location="cpu")

    def scan_module(module):
        if hasattr(module, attribute_name):
            if isinstance(module, torch.nn.Module):
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

    raise RuntimeError(f"No module with '{attribute_name}' found in the loaded world model.")


def AutoActionableModel(run_name, cache_dir=None):
    return _load_model_with_attribute(run_name, "get_action", cache_dir)


def AutoCostModel(run_name, cache_dir=None):
    return _load_model_with_attribute(run_name, "get_cost", cache_dir)
