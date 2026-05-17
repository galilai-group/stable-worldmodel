from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from collections.abc import Callable

import numpy as np
import torch
from loguru import logger as logging
from torchvision import tv_tensors

import stable_worldmodel as swm
from stable_worldmodel.solver import Solver


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


class Transformable(Protocol):
    """Protocol for reversible data transformations (e.g., normalizers, scalers)."""

    def transform(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Apply preprocessing to input data.

        Args:
            x: Input data as a numpy array.

        Returns:
            Preprocessed data as a numpy array.
        """
        ...

    def inverse_transform(
        self, x: np.ndarray
    ) -> np.ndarray:  # pragma: no cover
        """Reverse the preprocessing transformation.

        Args:
            x: Preprocessed data as a numpy array.

        Returns:
            Original data as a numpy array.
        """
        ...


class Actionable(Protocol):
    """Protocol for model action computation."""

    def get_action(info) -> torch.Tensor:  # pragma: no cover
        """Compute action from observation and goal"""
        ...


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
            seed: Optional random seed for the action space.
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

    def set_seed(self, seed: int) -> None:
        """Set the random seed for action sampling.

        Args:
            seed: The seed value.
        """
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

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        """Get action via planning with the world model.

        Args:
            info_dict: Current state information from the environment.
            **kwargs: Additional parameters for planning.

        Returns:
            The selected action(s) as a numpy array.
        """
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        assert 'pixels' in info_dict, "'pixels' must be provided in info_dict"
        assert 'goal' in info_dict, "'goal' must be provided in info_dict"

        info_dict = self._prepare_info(info_dict)
        n_envs = self.env.num_envs

        needs_flush = info_dict.pop('_needs_flush', None)
        if needs_flush is not None:
            for i in range(n_envs):
                if needs_flush[i]:
                    self._action_buffer[i].clear()
                    if self._next_init is not None:
                        self._next_init[i] = 0

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

        action_dim = self.env.single_action_space.shape[-1]
        action = torch.full((n_envs, action_dim), float('nan'))
        for i in range(n_envs):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()

        action = action.reshape(*self.env.action_space.shape)
        action = action.float().numpy()

        if 'action' in self.process:
            action = self.process['action'].inverse_transform(action)

        return action


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
        run_path = Path(
            cache_dir
            or swm.data.utils.get_cache_dir(sub_folder='checkpoints'),
            run_name,
        )

    if run_path.is_dir():
        ckpt_files = list(run_path.glob('*_object.ckpt'))
        ckpt_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
        path = ckpt_files[0]
        logging.info(f'Loading model from checkpoint: {path}')
    else:
        path = Path(f'{run_path}_object.ckpt')
        assert path.exists(), (
            f'Checkpoint path does not exist: {path}. Launch pretraining first.'
        )

    spt_module = torch.load(path, weights_only=False, map_location='cpu')

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

    raise RuntimeError(
        f"No module with '{attribute_name}' found in the loaded world model."
    )


def AutoActionableModel(
    run_name: str, cache_dir: str | Path | None = None
) -> torch.nn.Module:
    """Load a model checkpoint and return the module with a `get_action` method.

    Automatically scans the checkpoint for a module implementing the Actionable
    protocol (i.e., has a `get_action` method).

    Args:
        run_name: Path or name of the model run/checkpoint.
        cache_dir: Optional cache directory path. Defaults to STABLEWM_HOME.

    Returns:
        The module with a `get_action` method, set to eval mode.

    Raises:
        RuntimeError: If no module with `get_action` is found in the checkpoint.
    """
    return _load_model_with_attribute(run_name, 'get_action', cache_dir)


def AutoCostModel(
    run_name: str, cache_dir: str | Path | None = None
) -> torch.nn.Module:
    """Load a model checkpoint and return the module with a `get_cost` method.

    Automatically scans the checkpoint for a module implementing a cost function
    (i.e., has a `get_cost` method) for use with planning solvers.

    Args:
        run_name: Path or name of the model run/checkpoint.
        cache_dir: Optional cache directory path. Defaults to STABLEWM_HOME.

    Returns:
        The module with a `get_cost` method, set to eval mode.

    Raises:
        RuntimeError: If no module with `get_cost` is found in the checkpoint.
    """
    return _load_model_with_attribute(run_name, 'get_cost', cache_dir)


# Alias for backward compatibility and type hinting
Policy = BasePolicy


# ─── Hierarchical World Model (HWM) planning ────────────────────────────────

import torch.nn.functional as F
from einops import rearrange


class HWMCostModel(torch.nn.Module):
    """L2 CEM cost model that searches in latent macro-action space.

    The HWM uses a non-standard action key (e.g. 'latent_action') that is
    incompatible with PreJEPA.rollout (which hardcodes 'action'). This class
    encodes the observation and goal once, then embeds latent action candidates
    and calls the predictor directly.

    Cost = MSE(predicted_pixels_emb[-1], goal_pixels_emb).
    """

    def __init__(self, model: torch.nn.Module, action_key: str = 'latent_action'):
        super().__init__()
        self.model = model
        self.action_key = action_key

    def parameters(self):
        return self.model.parameters()

    def criterion(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        return self.get_cost(info_dict, action_candidates)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            info_dict: CEM-expanded; each tensor has shape (B, N, T, ...).
                       Required keys: 'pixels', 'goal', 'proprio', 'id', 'step_idx'.
            action_candidates: (B, N, H, latent_dim) — latent macro-action candidates.

        Returns:
            cost: (B, N) — scalar cost per candidate.
        """
        B, N, H, latent_dim = action_candidates.shape
        device = action_candidates.device
        dtype = action_candidates.dtype

        # Keys that are encoded but are NOT the macro-action (e.g. 'proprio')
        emb_keys = [k for k in self.model.extra_encoders if k != self.action_key]

        # ── Cache obs encoding (invalidate on new episode step) ─────────────
        id_now = info_dict['id'][:, 0]
        step_now = info_dict['step_idx'][:, 0]
        need_obs = not (
            hasattr(self, '_obs_id')
            and self._obs_id.shape == id_now.shape
            and torch.equal(self._obs_id, id_now)
            and torch.equal(self._obs_step, step_now)
        )
        if need_obs:
            obs = {k: (v[:, 0].to(device=device, dtype=dtype)
                       if (torch.is_tensor(v) and v.is_floating_point())
                       else (v[:, 0].to(device) if torch.is_tensor(v) else v))
                   for k, v in info_dict.items()}
            obs = self.model.encode(obs, pixels_key='pixels', emb_keys=emb_keys, target='emb')
            self._obs_id = id_now.detach()
            self._obs_step = step_now.detach()
            self._pixels_emb = obs['pixels_emb'].detach()            # (B, 1, P, pixel_dim)
            self._extra_embs = {k: obs[f'{k}_emb'].detach() for k in emb_keys}

        pixel_dim = self._pixels_emb.shape[-1]
        n_patches = self._pixels_emb.shape[2]

        # ── Cache goal encoding ──────────────────────────────────────────────
        need_goal = not (
            hasattr(self, '_goal_id')
            and self._goal_id.shape == id_now.shape
            and torch.equal(self._goal_id, id_now)
        )
        if need_goal:
            goal = {k: (v[:, 0].to(device=device, dtype=dtype)
                        if (torch.is_tensor(v) and v.is_floating_point())
                        else (v[:, 0].to(device) if torch.is_tensor(v) else v))
                    for k, v in info_dict.items()}
            goal = self.model.encode(goal, pixels_key='goal', emb_keys=[], target='goal_emb')
            self._goal_id = id_now.detach()
            self._goal_pixels_emb = goal['pixels_goal_emb'].detach()  # (B, 1, P, pixel_dim)

        # ── Build base embedding: pixels + non-action extras (no action yet) ─
        pix_exp = self._pixels_emb.to(device, dtype).unsqueeze(1).expand(-1, N, -1, -1, -1)
        base_parts = [pix_exp]
        for key in emb_keys:
            key_emb = self._extra_embs[key].to(device, dtype)           # (B, 1, emb_dim)
            key_tiled = (key_emb
                         .unsqueeze(2).expand(-1, -1, n_patches, -1)    # (B, 1, P, emb_dim)
                         .unsqueeze(1).expand(-1, N, -1, -1, -1))       # (B, N, 1, P, emb_dim)
            base_parts.append(key_tiled)
        base_emb = torch.cat(base_parts, dim=-1)                        # (B, N, 1, P, D_base)

        # ── Embed all latent action candidates ──────────────────────────────
        act_enc = self.model.extra_encoders[self.action_key]
        act_flat = action_candidates.to(device, dtype).reshape(B * N, H, latent_dim)
        act_emb = act_enc(act_flat)                                     # (B*N, H, act_emb_dim)
        act_emb_dim = act_emb.shape[-1]
        act_emb = act_emb.reshape(B, N, H, act_emb_dim)

        # Offset of the action slot in the full embedding
        action_start = pixel_dim + sum(self._extra_embs[k].shape[-1] for k in emb_keys)

        # ── Multi-step predictor rollout ─────────────────────────────────────
        # h=0: build initial embedding = base + z_0 tiled across patches
        act_tiled_0 = (act_emb[:, :, 0:1]
                       .unsqueeze(3).expand(-1, -1, -1, n_patches, -1))  # (B, N, 1, P, act_emb_dim)
        z = torch.cat([base_emb, act_tiled_0], dim=-1)                   # (B, N, 1, P, D)
        z_flat = rearrange(z, 'b n t p d -> (b n) t p d')               # (B*N, 1, P, D)

        for h in range(H - 1):
            pred = self.model.predict(z_flat[:, -self.model.history_size:])[:, -1:]  # (B*N, 1, P, D)
            act_h1 = (act_emb[:, :, h + 1:h + 2]
                      .unsqueeze(3).expand(-1, -1, -1, n_patches, -1)
                      .reshape(B * N, 1, n_patches, act_emb_dim))        # (B*N, 1, P, act_emb_dim)
            pred_injected = torch.cat([
                pred[..., :action_start],
                act_h1,
                pred[..., action_start + act_emb_dim:],
            ], dim=-1)
            z_flat = torch.cat([z_flat, pred_injected], dim=1)

        # Final prediction — the waypoint arrived at after the last macro-action
        pred_final = self.model.predict(z_flat[:, -self.model.history_size:])[:, -1:]
        pred_final = pred_final.reshape(B, N, 1, n_patches, -1)         # (B, N, 1, P, D)

        # ── Cost: MSE over pixel embedding ───────────────────────────────────
        goal_exp = (self._goal_pixels_emb.to(device, dtype)
                    .unsqueeze(1).expand(-1, N, -1, -1, -1))             # (B, N, 1, P, pixel_dim)
        pred_pixels = pred_final[..., :pixel_dim]                        # (B, N, 1, P, pixel_dim)
        cost = F.mse_loss(pred_pixels, goal_exp, reduction='none')       # (B, N, 1, P, pixel_dim)
        return cost.mean(dim=tuple(range(2, cost.ndim)))                 # (B, N)


class _FixedGoalCostModel(torch.nn.Module):
    """L1 CEM cost model with a precomputed subgoal pixel embedding.

    Wraps a low-level PreJEPA (which has 'action' in extra_encoders and is
    therefore compatible with PreJEPA.rollout) but replaces the goal-image
    encoding step with a precomputed pixel embedding produced by the L2 planner.

    Cost = MSE(predicted_pixels_emb[-1], goal_pixels_emb).
    """

    def __init__(self, model: torch.nn.Module, goal_pixels_emb: torch.Tensor):
        """
        Args:
            model: Low-level PreJEPA; must have 'action' in extra_encoders.
            goal_pixels_emb: (B, P, pixel_dim) — pixel embedding of the subgoal.
        """
        super().__init__()
        self.model = model
        self.goal_pixels_emb = goal_pixels_emb   # (B, P, pixel_dim)

    def parameters(self):
        return self.model.parameters()

    def criterion(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        return self.get_cost(info_dict, action_candidates)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            info_dict: CEM-expanded; 'pixels' has shape (B, N, T, C, H, W).
            action_candidates: (B, N, horizon, primitive_dim * frameskip).

        Returns:
            cost: (B, N).
        """
        B, N = action_candidates.shape[:2]

        # Run low-level WM rollout (PreJEPA.rollout handles 'action' key)
        info_dict = self.model.rollout(info_dict, action_candidates)

        # predicted_pixels_emb: (B, N, T + n_steps + 1, P, pixel_dim)
        pred_pixels = info_dict['predicted_pixels_emb']
        pred_last = pred_pixels[:, :, -1:]                              # (B, N, 1, P, pixel_dim)

        goal = (self.goal_pixels_emb
                .unsqueeze(1).unsqueeze(2)
                .expand(-1, N, 1, -1, -1)
                .to(pred_last.device, pred_last.dtype))                 # (B, N, 1, P, pixel_dim)

        cost = F.mse_loss(pred_last, goal, reduction='none')            # (B, N, 1, P, pixel_dim)
        return cost.mean(dim=tuple(range(2, cost.ndim)))                # (B, N)


class HWMPolicy(BasePolicy):
    """Hierarchical World Model policy (L2 macro-planner + L1 primitive planner).

    L2 CEM searches in latent macro-action space (using HWMCostModel).
    The best macro-action sequence is rolled out through the HWM to extract
    subgoal pixel embeddings.  For each subgoal, L1 CEM plans a primitive
    action sequence using _FixedGoalCostModel, and the resulting actions fill
    a per-environment buffer that is drained one step at a time.

    Args:
        hwm_model: Trained HWM PreJEPA (extra_encoders has action_key).
        low_level_model: Frozen low-level PreJEPA (extra_encoders has 'action').
        l2_solver: CEMSolver whose model should be HWMCostModel(hwm_model).
        l1_solver: CEMSolver whose model is replaced per-subgoal with
                   _FixedGoalCostModel before each L1 solve.
        l2_config: PlanConfig for L2 (action_block=1, horizon=K, receding_horizon=M).
        l1_config: PlanConfig for L1 (action_block=frameskip, horizon=l1_horizon).
        macro_action_dim: Latent macro-action dimension (e.g. 4).
        process: Preprocessing dict (StandardScaler per key).
        transform: Image transform dict.
    """

    def __init__(
        self,
        hwm_model: torch.nn.Module,
        low_level_model: torch.nn.Module,
        l2_solver: Any,
        l1_solver: Any,
        l2_config: PlanConfig,
        l1_config: PlanConfig,
        macro_action_dim: int,
        process: dict[str, Any] | None = None,
        transform: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.type = 'hwm'
        self.hwm_model = hwm_model
        self.low_level_model = low_level_model
        self.l2_solver = l2_solver
        self.l1_solver = l1_solver
        self.l2_cfg = l2_config
        self.l1_cfg = l1_config
        self.macro_action_dim = macro_action_dim
        self.process = process or {}
        self.transform = transform or {}
        self._action_buffer: list[deque] | None = None

    def set_env(self, env: Any) -> None:
        """Attach the policy to an environment and configure both solvers."""
        import gymnasium as gym

        self.env = env
        n_envs = getattr(env, 'num_envs', 1)

        # L2 solver uses a fake Box action space of size macro_action_dim
        l2_action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_envs, self.macro_action_dim),
            dtype=np.float32,
        )
        self.l2_solver.configure(
            action_space=l2_action_space, n_envs=n_envs, config=self.l2_cfg
        )

        # L1 solver uses the real environment action space
        self.l1_solver.configure(
            action_space=env.action_space, n_envs=n_envs, config=self.l1_cfg
        )

        # Each buffer entry holds primitive actions for l2_receding subgoals
        l1_steps = self.l1_cfg.receding_horizon * self.l1_cfg.action_block
        capacity = l1_steps * self.l2_cfg.receding_horizon
        self._action_buffer = [deque(maxlen=capacity) for _ in range(n_envs)]

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        info_dict = self._prepare_info(info_dict)
        n_envs = self.env.num_envs

        needs_flush = info_dict.pop('_needs_flush', None)
        if needs_flush is not None:
            for i in range(n_envs):
                if needs_flush[i]:
                    self._action_buffer[i].clear()

        terminated = info_dict.get('terminated')
        dead = (
            np.asarray(terminated, dtype=bool)
            if terminated is not None
            else np.zeros(n_envs, dtype=bool)
        )

        replan_idx = [
            i for i in range(n_envs)
            if len(self._action_buffer[i]) == 0 and not dead[i]
        ]
        if replan_idx:
            self._replan(info_dict, replan_idx)

        action_dim = self.env.single_action_space.shape[-1]
        action = torch.full((n_envs, action_dim), float('nan'))
        for i in range(n_envs):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()

        action = action.reshape(*self.env.action_space.shape).float().numpy()
        if 'action' in self.process:
            action = self.process['action'].inverse_transform(action)
        return action

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _slice_info(self, info_dict: dict, replan_idx: list, idx_tensor: torch.Tensor) -> dict:
        sliced: dict = {}
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                sliced[k] = v[idx_tensor]
            elif isinstance(v, np.ndarray):
                sliced[k] = v[replan_idx]
            elif isinstance(v, list):
                sliced[k] = [v[i] for i in replan_idx]
            else:
                sliced[k] = v
        return sliced

    def _replan(self, info_dict: dict, replan_idx: list) -> None:
        idx_tensor = torch.as_tensor(replan_idx, dtype=torch.long)
        sliced = self._slice_info(info_dict, replan_idx, idx_tensor)

        device_l2 = next(self.hwm_model.parameters()).device
        dtype_l2 = next(self.hwm_model.parameters()).dtype

        # ── Step 1: L2 CEM — find best macro-action sequence ─────────────────
        # Clear HWM cost-model obs/goal caches so stale entries don't persist
        for attr in ('_obs_id', '_obs_step', '_goal_id'):
            if hasattr(self.l2_solver.model, attr):
                delattr(self.l2_solver.model, attr)

        outputs_l2 = self.l2_solver(sliced)
        mean_z = outputs_l2['actions'].to(device_l2, dtype_l2)  # (B', l2_horizon, latent_dim)

        # ── Step 2: Roll HWM forward to extract subgoal pixel embeddings ─────
        subgoal_pixels_embs = self._hwm_subgoal_rollout(sliced, mean_z, device_l2, dtype_l2)
        # subgoal_pixels_embs[k]: (B', P, pixel_dim)

        # ── Step 3: L1 CEM for each subgoal ──────────────────────────────────
        l1_info = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sliced.items()}

        device_l1 = next(self.low_level_model.parameters()).device
        dtype_l1 = next(self.low_level_model.parameters()).dtype

        for k in range(self.l2_cfg.receding_horizon):
            subgoal_emb = subgoal_pixels_embs[k].to(device_l1, dtype_l1)  # (B', P, pixel_dim)

            l1_actions = []
            solver_batch_size = max(1, int(getattr(self.l1_solver, 'batch_size', len(replan_idx))))
            for start in range(0, len(replan_idx), solver_batch_size):
                end = min(start + solver_batch_size, len(replan_idx))
                local_rows = list(range(start, end))
                local_idx = torch.arange(start, end, dtype=torch.long)
                l1_info_chunk = self._slice_info(l1_info, local_rows, local_idx)

                # Swap in a fixed-goal cost model whose subgoal batch matches
                # the info batch CEM is solving for.
                self.l1_solver.model = _FixedGoalCostModel(
                    self.low_level_model, subgoal_emb[start:end]
                )

                # Clear low-level WM observation cache so rollout re-encodes
                if hasattr(self.low_level_model, '_init_cached_info'):
                    del self.low_level_model._init_cached_info

                outputs_l1 = self.l1_solver(l1_info_chunk)
                l1_actions.append(outputs_l1['actions'])

            prim_actions = torch.cat(l1_actions, dim=0)  # (B', l1_horizon, prim_dim*frameskip)

            # Flatten to individual primitive env steps
            B_prime = prim_actions.shape[0]
            flat = prim_actions.reshape(
                B_prime,
                self.l1_cfg.receding_horizon * self.l1_cfg.action_block,
                -1,
            )  # (B', l1_steps, action_dim)

            for row, env_i in enumerate(replan_idx):
                self._action_buffer[env_i].extend(flat[row])

            # NOTE: for l2_receding > 1, l1_info is not advanced to the predicted
            # next state — all subgoals are planned from the same current observation.
            # This is a deliberate simplification; add WM-based state advancement here
            # if multi-step hierarchical open-loop planning is needed.

    def _hwm_subgoal_rollout(
        self,
        info_single: dict,
        mean_z: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Roll the HWM forward with the best macro-action and return per-step subgoal embeddings.

        Args:
            info_single: Observation dict (not CEM-expanded); B' × T × ...
            mean_z: (B', H, latent_dim) — the optimal macro-action sequence from L2 CEM.

        Returns:
            List of H tensors, each (B', P, pixel_dim) — predicted pixel embeddings
            after each macro-action step.  Only the first l2_receding entries are used.
        """
        model = self.hwm_model
        action_key = getattr(self.l2_solver.model, 'action_key', 'latent_action')
        emb_keys = [k for k in model.extra_encoders if k != action_key]
        H = mean_z.shape[1]

        with torch.no_grad():
            # Encode current observation (pixels + non-action extras)
            obs = {
                k: (v.to(device=device, dtype=dtype)
                    if (torch.is_tensor(v) and v.is_floating_point())
                    else (v.to(device) if torch.is_tensor(v) else v))
                for k, v in info_single.items()
            }
            obs = model.encode(obs, pixels_key='pixels', emb_keys=emb_keys, target='emb')

            pixels_emb = obs['pixels_emb']     # (B', 1, P, pixel_dim)
            pixel_dim = pixels_emb.shape[-1]
            n_patches = pixels_emb.shape[2]

            # Base embedding (without macro-action)
            base_parts = [pixels_emb]
            for key in emb_keys:
                tiled = (obs[f'{key}_emb']
                         .unsqueeze(2)
                         .expand(-1, -1, n_patches, -1))   # (B', 1, P, emb_dim)
                base_parts.append(tiled)
            base_emb = torch.cat(base_parts, dim=-1)        # (B', 1, P, D_base)

            act_enc = model.extra_encoders[action_key]
            action_start = pixel_dim + sum(obs[f'{k}_emb'].shape[-1] for k in emb_keys)

            current_emb = None   # tracks the most recent predicted state embedding
            subgoal_embs: list[torch.Tensor] = []

            for h in range(H):
                z_h = mean_z[:, h:h + 1].to(device, dtype)    # (B', 1, latent_dim)
                z_h_emb = act_enc(z_h)                         # (B', 1, act_emb_dim)
                act_emb_dim = z_h_emb.shape[-1]
                z_h_tiled = (z_h_emb
                             .unsqueeze(2)
                             .expand(-1, -1, n_patches, -1))   # (B', 1, P, act_emb_dim)

                if current_emb is None:
                    # First step: start from the encoded observation
                    full_emb = torch.cat([base_emb, z_h_tiled], dim=-1)   # (B', 1, P, D)
                else:
                    # Subsequent steps: inject new macro-action into the last prediction
                    full_emb = torch.cat([
                        current_emb[..., :action_start],
                        z_h_tiled,
                        current_emb[..., action_start + act_emb_dim:],
                    ], dim=-1)

                pred = model.predict(full_emb)[:, -1:]          # (B', 1, P, D)
                subgoal_embs.append(pred[:, 0, :, :pixel_dim])  # (B', P, pixel_dim)
                current_emb = pred

        return subgoal_embs[: self.l2_cfg.receding_horizon]
