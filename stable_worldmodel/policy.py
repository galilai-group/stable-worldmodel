from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray
from torchvision import tv_tensors

from stable_worldmodel.planning.solver import Solver
from stable_worldmodel.protocols import Actionable, Transformable

if TYPE_CHECKING:
    from stable_worldmodel.world.env_pool import AsyncEnvMask


@dataclass(frozen=True)
class PlanConfig:
    """Configuration for the MPC planning loop.

    Attributes:
        horizon: Planning horizon in number of steps.
        receding_horizon: Number of steps to execute before re-planning.
        history_len: Number of past observations to consider.
        action_block: Number of times each action is repeated (frameskip).
        warm_start: Whether to use the previous plan to initialize the next one.
    """

    horizon: int
    receding_horizon: int
    history_len: int = 1
    action_block: int = 1
    warm_start: bool = True

    @property
    def plan_len(self) -> int:
        """Total plan length in environment steps."""
        return self.horizon * self.action_block


class BasePolicy:
    """Base class for agent policies.

    Attributes:
        env: The environment the policy is associated with.
        type: A string identifier for the policy type.
    """

    env: Any
    type: str

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the base policy.

        Args:
            **kwargs: Additional configuration parameters.
        """
        self.env = None
        self.type = 'base'
        for arg, value in kwargs.items():
            setattr(self, arg, value)

    def get_action(
        self,
        obs: Any,
        *,
        env_mask: AsyncEnvMask | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get action from the policy given the observation.

        Args:
            obs: The current observation from the environment.
            env_mask: Optional boolean mask over the full environment pool.
                When provided, ``obs`` contains only the selected rows, in
                ascending environment-index order. Stateful policies use the
                mask to map those rows back to their per-environment state.
            **kwargs: Additional parameters for action selection.

        Returns:
            Selected action as a numpy array.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    def on_reset(self, env_mask: AsyncEnvMask | None = None) -> None:
        """Notify the policy that selected environments were reset.

        Args:
            env_mask: Boolean mask over the full environment pool. ``None``
                means that every environment was reset.

        Stateless policies do not need to override this hook.
        """

    def set_env(self, env: Any) -> None:
        """Associate this policy with an environment.

        Args:
            env: The environment to associate.
        """
        self.env = env

    def _prepare_info(self, info_dict: dict) -> dict[str, torch.Tensor]:
        """Pre-process and transform observations.

        Applies preprocessing (via `self.process`) and transformations (via `self.transform`)
        to observation data. Used by subclasses like FeedForwardPolicy and WorldModelPolicy.
        Returns a new dict; the input is not mutated.

        Args:
            info_dict: Raw observation dictionary from the environment.

        Returns:
            A dictionary of processed tensors.

        Raises:
            ValueError: If an expected numpy array is missing for processing.
        """
        out = {}
        for k, v in info_dict.items():
            is_numpy = isinstance(v, (np.ndarray | np.generic))

            if hasattr(self, 'process') and k in self.process:
                if not is_numpy:
                    raise ValueError(
                        f"Expected numpy array for key '{k}' in process, got {type(v)}"
                    )

                # flatten extra dimensions if needed
                shape = v.shape
                if len(shape) > 2:
                    v = v.reshape(-1, *shape[2:])

                # process and reshape back
                v = self.process[k].transform(v)
                v = v.reshape(shape)

            # collapse env and time dimensions for transform (e, t, ...) -> (e * t, ...)
            # then restore after transform
            if hasattr(self, 'transform') and k in self.transform:
                shape = None
                if is_numpy or torch.is_tensor(v):
                    if v.ndim > 2:
                        shape = v.shape
                        v = v.reshape(-1, *shape[2:])
                if k.startswith('pixels') or k.startswith('goal'):
                    # permute channel first for transform
                    if is_numpy:
                        v = np.transpose(v, (0, 3, 1, 2))
                    else:
                        v = v.permute(0, 3, 1, 2)
                v = torch.stack(
                    [self.transform[k](tv_tensors.Image(x)) for x in v]
                )
                is_numpy = isinstance(v, (np.ndarray | np.generic))

                if shape is not None:
                    v = v.reshape(*shape[:2], *v.shape[1:])

            if is_numpy and v.dtype.kind not in 'USO':
                v = torch.from_numpy(v)

            out[k] = v

        return out


class RandomPolicy(BasePolicy):
    """Policy that samples random actions from the action space."""

    def __init__(self, seed: int | None = None, **kwargs: Any) -> None:
        """Initialize the random policy.

        Args:
            seed: Random seed applied to the action space when the environment
                is attached. If None, the action space uses its own default RNG,
                making action sampling non-deterministic across runs.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self.type = 'random'
        self.seed = seed

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        """Get a random action from the environment's action space.

        Args:
            obs: The current observation (ignored).
            **kwargs: Additional parameters (ignored).

        Returns:
            A randomly sampled action.
        """
        return self.env.action_space.sample()

    def set_env(self, env: Any) -> None:
        super().set_env(env)
        if self.seed is not None:
            self.set_seed(self.seed)

    def set_seed(self, seed: int) -> None:
        if self.env is not None:
            self.env.action_space.seed(seed)


class ExpertPolicy(BasePolicy):
    """Policy using expert demonstrations or heuristics."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the expert policy.

        Args:
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self.type = 'expert'

    def get_action(
        self, obs: Any, goal_obs: Any, **kwargs: Any
    ) -> np.ndarray | None:
        """Get action from the expert policy.

        Args:
            obs: The current observation.
            goal_obs: The goal observation.
            **kwargs: Additional parameters.

        Returns:
            The expert action, or None if not available.
        """
        # Implement expert policy logic here
        pass


class FeedForwardPolicy(BasePolicy):
    """Feed-Forward Policy using a neural network model.

    Actions are computed via a single forward pass through the model.
    Useful for imitation learning policies like Goal-Conditioned Behavioral Cloning (GCBC).

    Attributes:
        model: Neural network model implementing the Actionable protocol.
        process: Dictionary of data preprocessors for specific keys.
        transform: Dictionary of tensor transformations (e.g., image transforms).
    """

    def __init__(
        self,
        model: Actionable,
        process: dict[str, Transformable] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]]
        | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the feed-forward policy.

        Args:
            model: Neural network model with a `get_action` method.
            process: Dictionary of data preprocessors for specific keys.
            transform: Dictionary of tensor transformations (e.g., image transforms).
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self.type = 'feed_forward'
        self.model = model.eval()
        self.process = process or {}
        self.transform = transform or {}

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        """Get action via a forward pass through the neural network model.

        Args:
            info_dict: Current state information containing at minimum a 'goal' key.
            **kwargs: Additional parameters (unused).

        Returns:
            The selected action as a numpy array.

        Raises:
            AssertionError: If environment not set or 'goal' not in info_dict.
        """
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        assert 'goal' in info_dict, "'goal' must be provided in info_dict"

        # Prepare the info dict (transforms and normalizes inputs)
        info_dict = self._prepare_info(info_dict)

        # Add goal_pixels key for GCBC model
        if 'goal' in info_dict:
            info_dict['goal_pixels'] = info_dict['goal']

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
        if 'action' in self.process:
            action = self.process['action'].inverse_transform(action)

        return action


class WorldModelPolicy(BasePolicy):
    """Policy using a world model and planning solver for action selection."""

    def __init__(
        self,
        solver: Solver,
        config: PlanConfig,
        process: dict[str, Transformable] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]]
        | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the world model policy.

        Args:
            solver: The planning solver to use.
            config: MPC planning configuration.
            process: Dictionary of data preprocessors for specific keys.
            transform: Dictionary of tensor transformations (e.g., image transforms).
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)

        self.type = 'world_model'
        self.cfg = config
        self.solver = solver
        self.process = process or {}
        self.transform = transform or {}
        self._action_buffer: list[deque[torch.Tensor]] | None = None
        self._next_init: torch.Tensor | None = None

    @property
    def flatten_receding_horizon(self) -> int:
        """Receding horizon in environment steps (with frameskip)."""
        return self.cfg.receding_horizon * self.cfg.action_block

    def set_env(self, env: Any) -> None:
        """Configure the policy and solver for the given environment.

        Args:
            env: The environment to associate with the policy.
        """
        self.env = env
        n_envs = getattr(env, 'num_envs', 1)
        self.solver.configure(
            action_space=env.action_space, n_envs=n_envs, config=self.cfg
        )
        self._action_buffer = [
            deque(maxlen=self.flatten_receding_horizon) for _ in range(n_envs)
        ]

        assert isinstance(self.solver, Solver), (
            'Solver must implement the Solver protocol'
        )

    def get_action(
        self,
        info_dict: dict,
        *,
        env_mask: AsyncEnvMask | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get action via planning with the world model.

        Args:
            info_dict: Current state information from the environment.
            env_mask: Optional boolean mask over the full environment pool.
                ``info_dict`` contains one row per true entry, in ascending
                environment-index order. If omitted, all environments are
                assumed present (sync EnvPool).
            **kwargs: Additional parameters for planning.

        Returns:
            The selected action(s) as a numpy array.
        """
        assert hasattr(self, 'env'), 'Environment not set for the policy'

        n_envs = self.env.num_envs
        env_indices = _selected_env_indices(env_mask, n_envs)
        batch_size = len(env_indices)

        needs_flush = info_dict.get('_needs_flush')
        if needs_flush is not None:
            flush_rows = _bool_rows(needs_flush, batch_size, '_needs_flush')
            flush_mask = np.zeros(n_envs, dtype=bool)
            flush_mask[env_indices[flush_rows]] = True
            self.on_reset(flush_mask)

        info_dict = self._prepare_info(info_dict)
        info_dict.pop('_needs_flush', None)

        terminated = info_dict.get('terminated')
        dead = (
            _bool_rows(terminated, batch_size, 'terminated')
            if terminated is not None
            else np.zeros(batch_size, dtype=bool)
        )

        replan_rows = [
            row
            for row, env_i in enumerate(env_indices)
            if len(self._action_buffer[env_i]) == 0 and not dead[row]
        ]
        replan_envs = env_indices[replan_rows]

        if replan_rows:
            row_tensor = torch.as_tensor(replan_rows, dtype=torch.long)
            env_tensor = torch.as_tensor(replan_envs, dtype=torch.long)
            sliced = {}
            for k, v in info_dict.items():
                if torch.is_tensor(v):
                    sliced[k] = v[row_tensor]
                elif isinstance(v, np.ndarray):
                    sliced[k] = v[replan_rows]
                elif isinstance(v, list):
                    sliced[k] = [v[i] for i in replan_rows]
                else:
                    sliced[k] = v

            sliced_init = (
                self._next_init[env_tensor]
                if self._next_init is not None
                else None
            )

            outputs = self.solver(sliced, init_action=sliced_init)

            actions = outputs['actions']
            keep_horizon = self.cfg.receding_horizon
            plan = actions[:, :keep_horizon]
            rest = actions[:, keep_horizon:]

            if self.cfg.warm_start and rest.shape[1] > 0:
                if self._next_init is None:
                    self._next_init = torch.zeros(
                        n_envs, rest.shape[1], rest.shape[2], dtype=rest.dtype
                    )
                self._next_init[env_tensor] = rest
            elif not self.cfg.warm_start:
                self._next_init = None

            plan = plan.reshape(
                len(replan_rows), self.flatten_receding_horizon, -1
            )

            for row, env_i in enumerate(replan_envs):
                self._action_buffer[env_i].extend(plan[row])

        single_shape = self.env.single_action_space.shape
        is_discrete = 'Discrete' in type(self.env.single_action_space).__name__

        action = torch.full(
            (batch_size, *single_shape),
            fill_value=0 if is_discrete else float('nan'),
            dtype=torch.long if is_discrete else torch.float32,
        )

        for row, env_i in enumerate(env_indices):
            if not dead[row]:
                action[row] = self._action_buffer[env_i].popleft()

        if env_mask is None:
            action = action.reshape(*self.env.action_space.shape)
        action = action.numpy()

        if 'action' in self.process:
            action = self.process['action'].inverse_transform(action)

        return action

    def on_reset(self, env_mask: AsyncEnvMask | None = None) -> None:
        """Clear plans and warm starts for reset environment slots."""
        if self._action_buffer is None:
            return

        n_envs = len(self._action_buffer)
        env_indices = _selected_env_indices(env_mask, n_envs)
        for env_i in env_indices:
            self._action_buffer[env_i].clear()
            if self._next_init is not None:
                self._next_init[env_i] = 0


def _selected_env_indices(
    env_mask: AsyncEnvMask | None, n_envs: int
) -> NDArray[np.int64]:
    """Map an optional full-pool boolean mask to global env indices."""
    if env_mask is None:
        return np.arange(n_envs, dtype=np.int64)

    mask = np.asarray(env_mask)
    if mask.dtype != np.bool_ or mask.shape != (n_envs,):
        raise ValueError(
            f'env_mask must be boolean with shape ({n_envs},), '
            f'got dtype={mask.dtype} and shape={mask.shape}'
        )
    return np.flatnonzero(mask)


def _bool_rows(value: Any, batch_size: int, name: str) -> NDArray[np.bool_]:
    """Collapse a batched boolean value to one flag per input row."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    rows = np.asarray(value, dtype=bool)
    if rows.ndim == 0 or rows.shape[0] != batch_size:
        raise ValueError(
            f'{name} must have {batch_size} leading rows, got {rows.shape}'
        )
    return rows.reshape(batch_size, -1).any(axis=1)


# Alias for backward compatibility and type hinting
Policy = BasePolicy
