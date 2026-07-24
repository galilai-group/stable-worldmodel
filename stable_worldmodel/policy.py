from collections import deque
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

import numpy as np
import torch
from torchvision import tv_tensors

from stable_worldmodel.buffer import HistoryBuffer
from stable_worldmodel.planning.solver import Solver
from stable_worldmodel.protocols import Actionable, Transformable

# info-dict key under which WorldModelPolicy supplies the executed past
# action blocks (solver space, shape (B, history_len - 1, blocked_dim))
ACTION_HISTORY_KEY = 'action_history'


@dataclass(frozen=True)
class PlanConfig:
    """Configuration for the MPC planning loop.

    Attributes:
        horizon: Planning horizon in number of steps.
        receding_horizon: Number of steps to execute before re-planning.
        history_len: Number of observation frames (in ``action_block``
            timesteps, including the current frame) supplied to the world
            model at planning time. Values above 1 additionally supply the
            ``history_len - 1`` executed action blocks between those frames
            via ``info['action_history']``, and require a rollout-based
            ``Dynamics`` model (LeWM/PLDM/PreJEPA); Markovian ``Costable``
            models (TD-MPC2) only support the default of 1. **Warm-up
            behavior**: during the first ``(history_len - 1) *
            action_block`` env steps of each episode the context is
            simply shorter — it grows from 1 frame at the first plan up
            to ``history_len``, containing only real frames. Synthetic
            padding (copies of an env's oldest frame, with zero action
            blocks — as if the env had been stationary) is used only
            when a replan batch mixes envs at different fill levels
            (e.g. a freshly auto-reset env planning alongside envs with
            full histories), since their histories must stack into one
            tensor.
        history_max_len: Capacity (in env steps) of the per-env history
            buffer. ``None`` means derive
            ``(history_len - 1) * action_block + 1`` — the smallest size
            that yields ``history_len`` strided frames with a full action
            block between each consecutive pair. Set higher to retain
            more raw history than you sample.
        action_block: Number of times each action is repeated (frameskip).
        warm_start: Whether to use the previous plan to initialize the next one.
    """

    horizon: int
    receding_horizon: int
    history_len: int = 1
    history_max_len: int | None = None
    action_block: int = 1
    warm_start: bool = True

    def __post_init__(self) -> None:
        if self.history_len < 1:
            raise ValueError(
                f'history_len must be >= 1, got {self.history_len}'
            )

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

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        """Get action from the policy given the observation.

        Args:
            obs: The current observation from the environment.
            **kwargs: Additional parameters for action selection.

        Returns:
            Selected action as a numpy array.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

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
        """Attach the environment and seed its action space.

        Args:
            env: The environment to attach. If the policy was constructed
                with a seed, the environment's action space is seeded so
                action sampling is reproducible.
        """
        super().set_env(env)
        if self.seed is not None:
            self._set_seed(self.seed)

    def _set_seed(self, seed: int) -> None:
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
        history_keys: tuple[str, ...] = ('pixels',),
        **kwargs: Any,
    ) -> None:
        """Initialize the world model policy.

        Args:
            solver: The planning solver to use.
            config: MPC planning configuration.
            process: Dictionary of data preprocessors for specific keys.
            transform: Dictionary of tensor transformations (e.g., image transforms).
            history_keys: Observation keys stacked over the last
                ``config.history_len`` block timesteps when replanning
                (only used when ``history_len > 1``). The executed action
                blocks between those frames are supplied alongside under
                ``'action_history'``.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)

        self.type = 'world_model'
        self.cfg = config
        self.solver = solver
        self.process = process or {}
        self.transform = transform or {}
        self.history_keys = tuple(history_keys)
        self._action_buffer: list[deque[torch.Tensor]] | None = None
        self._next_init: torch.Tensor | None = None
        self._history_buffer: HistoryBuffer | None = None

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
        if self.cfg.history_len > 1:
            max_len = self.cfg.history_max_len
            if max_len is None:
                max_len = (
                    self.cfg.history_len - 1
                ) * self.cfg.action_block + 1
            self._history_buffer = HistoryBuffer(
                n_envs=n_envs,
                max_len=max_len,
                action_block=self.cfg.action_block,
                block_keys=('action',),
            )
        else:
            self._history_buffer = None

        assert isinstance(self.solver, Solver), (
            'Solver must implement the Solver protocol'
        )

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        """Get action via planning with the world model.

        Args:
            info_dict: Current state information from the environment.
            **kwargs: Additional parameters for planning.

        Returns:
            The selected action(s) as a numpy array.
        """
        assert hasattr(self, 'env'), 'Environment not set for the policy'

        n_envs = self.env.num_envs

        needs_flush = info_dict.pop('_needs_flush', None)
        if needs_flush is not None:
            for i in range(n_envs):
                if needs_flush[i]:
                    self._action_buffer[i].clear()
                    if self._next_init is not None:
                        self._next_init[i] = 0
            if self._history_buffer is not None:
                flush_ids = [i for i in range(n_envs) if bool(needs_flush[i])]
                if flush_ids:
                    self._history_buffer.reset(flush_ids)

        info_dict = self._prepare_info(info_dict)

        if self._history_buffer is not None:
            self._history_buffer.append(
                {k: info_dict[k] for k in (*self.history_keys, 'action')}
            )

        terminated = info_dict.get('terminated')
        dead = (
            np.asarray(terminated, dtype=bool)
            if terminated is not None
            else np.zeros(n_envs, dtype=bool)
        )

        replan_idx = [
            i
            for i in range(n_envs)
            if len(self._action_buffer[i]) == 0 and not dead[i]
        ]

        if replan_idx:
            idx_tensor = torch.as_tensor(replan_idx, dtype=torch.long)
            sliced = {}
            for k, v in info_dict.items():
                if torch.is_tensor(v):
                    sliced[k] = v[idx_tensor]
                elif isinstance(v, np.ndarray):
                    sliced[k] = v[replan_idx]
                elif isinstance(v, list):
                    sliced[k] = [v[i] for i in replan_idx]
                else:
                    sliced[k] = v

            if self._history_buffer is not None:
                # replan calls land on block boundaries, so the strided
                # history is the frames at the last history_len boundaries
                # plus the executed blocks between them (solver space).
                # Request only as many frames as the fullest env in the
                # batch holds: early in an episode the context grows
                # 1 -> history_len with no synthetic frames; padding
                # occurs only when the batch mixes envs at different
                # fill levels (histories must stack into one tensor).
                n_frames = min(
                    self.cfg.history_len,
                    max(self._history_buffer.num_strided(replan_idx)),
                )
                history = self._history_buffer.get(
                    n_frames, env_ids=replan_idx
                )
                for k in self.history_keys:
                    sliced[k] = history[k]
                if 'action' in history:
                    sliced[ACTION_HISTORY_KEY] = history['action']

            sliced_init = (
                self._next_init[idx_tensor]
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
                self._next_init[idx_tensor] = rest
            elif not self.cfg.warm_start:
                self._next_init = None

            plan = plan.reshape(
                len(replan_idx), self.flatten_receding_horizon, -1
            )

            for row, env_i in enumerate(replan_idx):
                self._action_buffer[env_i].extend(plan[row])

        single_shape = self.env.single_action_space.shape
        is_discrete = 'Discrete' in type(self.env.single_action_space).__name__

        action = torch.full(
            (n_envs, *single_shape),
            fill_value=0 if is_discrete else float('nan'),
            dtype=torch.long if is_discrete else torch.float32,
        )

        for i in range(n_envs):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()

        action = action.reshape(*self.env.action_space.shape)
        action = action.numpy()

        if 'action' in self.process:
            action = self.process['action'].inverse_transform(action)

        return action


# Alias for backward compatibility and type hinting
Policy = BasePolicy
