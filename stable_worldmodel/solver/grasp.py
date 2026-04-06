"""GRASP: Gradient RelAxed Stochastic Planner.

Jointly optimises virtual intermediate states and actions via gradient
descent, using stop-gradient dynamics loss, goal shaping, Langevin-style
state noise, and a periodic full-rollout terminal-state sync.

Reference: https://arxiv.org/abs/2602.00475

The model passed to :class:`GRASPSolver` must expose:
* **encode** — encodes pixel observations into latent embeddings.
* **predict** / **action_encoder** — used for single-step differentiable
  latent predictions in the main GRASP optimisation loop.
* **get_cost** — used during periodic sync steps to ground the terminal
  state to the goal.

``info_dict`` passed to :meth:`GRASPSolver.solve` must contain:
* ``'pixels'``: ``(B, T, C, H, W)`` pixel observations.
* ``'goal'``:   ``(B, [T,] C, H, W)`` goal pixel observations.

Embeddings are computed automatically via ``model.encode()`` at the
start of each solve call.
"""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging

from .solver import Costable


class GRASPSolver:
    """GRASP: Gradient RelAxed Stochastic Planner.
    Reference: https://arxiv.org/abs/2602.00475

    Jointly optimises virtual states ``s_1 .. s_{T-1}`` and actions
    ``a_0 .. a_{T-1}`` by minimising:

    .. math::
        \\mathcal{L} = \\sum_{t=0}^{T-1}
            \\|F(\\bar{s}_t, a_t) - s_{t+1}\\|^2
          + \\sum_{t=0}^{T-1}
            \\|F(\\bar{s}_t, a_t) - g\\|^2

    where :math:`\\bar{s}_t` is the stop-gradient version of the virtual
    state at step *t*, ``s_0`` is fixed (observed), and ``s_T = g``.

    Every ``gd_interval`` steps a full-rollout terminal sync is run
    using ``model.get_cost``.
    """

    def __init__(
        self,
        model: Costable,
        n_steps: int = 200,
        batch_size: int | None = None,
        lr_s: float = 0.1,
        lr_a: float = 0.001,
        goal_weight: float = 1.0,
        state_noise_scale: float = 0.01,
        gd_interval: int = 50,
        gd_opt_steps: int = 10,
        gd_lr: float = 0.01,
        sync_mode: str = 'gd',
        cem_sync_samples: int = 64,
        cem_sync_topk: int = 10,
        cem_sync_var_scale: float = 1.0,
        cem_sync_var_min: float = 0.01,
        schedule_decay: bool = False,
        init_noise_scale: float = 0.1,
        min_noise_scale: float = 0.0,
        init_goal_weight: float = 2.0,
        min_goal_weight: float = 1.0,
        emb_key: str = 'emb',
        goal_emb_key: str = 'goal_emb',
        device: str | torch.device = 'cpu',
        seed: int = 1234,
    ) -> None:
        missing = []
        for attr in ('encode', 'predict', 'action_encoder', 'get_cost'):
            if not hasattr(model, attr):
                missing.append(attr)
        if missing:
            raise TypeError(
                f'GRASPSolver requires a model with {", ".join(missing)}, '
                f'got {type(model).__name__}.'
            )
        if sync_mode not in ('gd', 'cem'):
            raise ValueError(
                f"sync_mode must be 'gd' or 'cem', got {sync_mode!r}"
            )

        self.model = model
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.lr_s = lr_s
        self.lr_a = lr_a
        self.goal_weight = goal_weight
        self.state_noise_scale = state_noise_scale
        self.gd_interval = gd_interval
        self.gd_opt_steps = gd_opt_steps
        self.gd_lr = gd_lr
        self.sync_mode = sync_mode
        self.cem_sync_samples = cem_sync_samples
        self.cem_sync_topk = cem_sync_topk
        self.cem_sync_var_scale = cem_sync_var_scale
        self.cem_sync_var_min = cem_sync_var_min
        self.schedule_decay = schedule_decay
        self.init_noise_scale = init_noise_scale
        self.min_noise_scale = min_noise_scale
        self.init_goal_weight = init_goal_weight
        self.min_goal_weight = min_goal_weight
        self.emb_key = emb_key
        self.goal_emb_key = goal_emb_key
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)

        self._configured = False
        self._n_envs: int | None = None
        self._action_dim: int | None = None
        self._config: Any = None

    def configure(
        self,
        *,
        action_space: gym.Space,
        n_envs: int,
        config: Any,
    ) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

        if not isinstance(action_space, Box):
            logging.warning(
                f'Action space is discrete, got {type(action_space)}.'
                ' GRASPSolver may not work as expected.'
            )

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_virtual_states(
        self,
        emb_0: torch.Tensor,
        goal_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Linear interpolation between emb_0 and goal_emb.
        Returns (B, T-1, D) with requires_grad=True when T > 1.
        """
        num_virtual = self.horizon - 1
        if num_virtual > 0:
            t = torch.linspace(0, 1, num_virtual + 2, device=self.device)
            t = t[1:-1].view(1, -1, 1)  # (1, T-1, 1)
            virtual = emb_0.unsqueeze(1) + t * (
                goal_emb - emb_0
            ).unsqueeze(1)
        else:
            B, D = emb_0.shape
            virtual = torch.empty(B, 0, D, device=self.device)

        return virtual.detach().requires_grad_(virtual.numel() > 0)

    def _init_actions(
        self,
        actions: torch.Tensor | None,
        B: int,
    ) -> torch.Tensor:
        """Zero-pad actions to the full horizon, return with requires_grad."""
        if actions is None:
            a = torch.zeros(
                B, self.horizon, self.action_dim, device=self.device
            )
        else:
            a = actions.to(self.device)
            remaining = self.horizon - a.shape[1]
            if remaining > 0:
                pad = torch.zeros(
                    B, remaining, self.action_dim, device=self.device
                )
                a = torch.cat([a, pad], dim=1)

        return a.detach().requires_grad_(True)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        virtual_states: torch.Tensor,
        actions: torch.Tensor,
        emb_0: torch.Tensor,
        goal_emb: torch.Tensor,
        info_dict: dict,
        goal_weight: float | None = None,
    ) -> torch.Tensor:
        """Dynamics + goal loss with stop-gradient on state inputs.
        
        Evaluated entirely in parallel across time by flattening (B, T) 
        to avoid slow sequential for-loops, while maintaining strict 
        1-step independent predictions.
        """
        if goal_weight is None:
            goal_weight = self.goal_weight

        # Full state sequence: [s_0, s_1, ..., s_{T-1}, g] — (B, T+1, D)
        s_full = torch.cat(
            [
                emb_0.detach().unsqueeze(1),
                virtual_states,
                goal_emb.detach().unsqueeze(1),
            ],
            dim=1,
        )

        # 1. PARALLELIZATION FIX: Predict all independent T steps simultaneously
        s_t = s_full[:, :-1].detach()   # (B, T, D)
        s_next = s_full[:, 1:]          # (B, T, D)
        B, T = actions.shape[:2]

        # Flatten to (B*T, 1, D) to force 1-step isolated predictions 
        # (prevents sequence models from leaking information across time)
        s_t_flat = s_t.reshape(B * T, 1, -1)
        a_t_flat = actions.reshape(B * T, 1, -1)

        act_emb_flat = self.model.action_encoder(a_t_flat)                # (B*T, 1, act_D)
        pred_next_flat = self.model.predict(s_t_flat, act_emb_flat)[:, -1] # (B*T, D)
        
        # Reshape back to (B, T, D)
        pred_next = pred_next_flat.reshape(B, T, -1)

        # 2. MEANS FIX: Sum over features and time, take mean over batch only
        # Dynamics loss: \sum_t ||F(s_t, a_t) - s_{t+1}||^2
        dyn_loss = ((pred_next - s_next) ** 2).sum(dim=-1).sum(dim=1).mean()
        
        # Goal loss: \sum_t ||F(s_t, a_t) - g||^2
        goal_expanded = goal_emb.detach().unsqueeze(1)
        goal_loss = ((pred_next - goal_expanded) ** 2).sum(dim=-1).sum(dim=1).mean()

        return dyn_loss + goal_weight * goal_loss

    # ------------------------------------------------------------------
    # Sync helpers
    # ------------------------------------------------------------------

    def _compute_per_timestep_var(
        self,
        virtual_states: torch.Tensor,
        emb_0: torch.Tensor,
        goal_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Per-timestep CEM variance from virtual-state drift vs linear baseline.
        Returns ``(B, T, 1)``.
        """
        B, T = emb_0.shape[0], self.horizon

        if virtual_states.numel() > 0:
            num_virtual = T - 1
            t = torch.linspace(0, 1, num_virtual + 2, device=self.device)
            t = t[1:-1].view(1, -1, 1)
            expected = emb_0.detach().unsqueeze(1) + t * (
                goal_emb.detach() - emb_0.detach()
            ).unsqueeze(1)
            deviation = ((virtual_states.detach() - expected) ** 2).mean(dim=-1)
            mean_dev = deviation.mean(dim=-1, keepdim=True)
            deviation_full = torch.cat([mean_dev, deviation], dim=1)
            var_t = self.cem_sync_var_scale * deviation_full + self.cem_sync_var_min
        else:
            var_t = torch.full((B, T), self.cem_sync_var_min, device=self.device)

        return var_t.unsqueeze(-1)

    @staticmethod
    def _expand_info(info_dict: dict, num_samples: int) -> dict:
        """Add sample dimension to info_dict tensors for ``model.get_cost``.
        ``get_cost`` expects every tensor to be ``(B, S, ...)``.
        Uses ``.clone()`` to avoid autograd view-mutation errors across
        multiple optimiser steps.
        """
        expanded = {}
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                expanded[k] = v.unsqueeze(1).expand(
                    v.shape[0], num_samples, *v.shape[1:]
                ).clone()
            elif isinstance(v, np.ndarray):
                expanded[k] = np.repeat(v[:, None, ...], num_samples, axis=1)
            else:
                expanded[k] = v
        return expanded

    def _gd_sync(
        self,
        actions: torch.Tensor,
        info_dict: dict,
    ) -> torch.Tensor:
        """Full-rollout GD sync using ``model.get_cost``."""
        a_sync = actions.detach().clone().requires_grad_(True)
        sync_opt = torch.optim.Adam([a_sync], lr=self.gd_lr)

        for _ in range(self.gd_opt_steps):
            sync_opt.zero_grad()
            expanded = self._expand_info(info_dict, 1)
            cost = self.model.get_cost(expanded, a_sync.unsqueeze(1))
            
            # MEANS FIX: Use .sum() to preserve batch-independent gradients
            loss = cost.sum()
            loss.backward()
            sync_opt.step()

        return a_sync.detach()

    def _cem_sync(
        self,
        actions: torch.Tensor,
        virtual_states: torch.Tensor,
        emb_0: torch.Tensor,
        goal_emb: torch.Tensor,
        info_dict: dict,
    ) -> torch.Tensor:
        """Full-rollout CEM sync using ``model.get_cost``."""
        B, T, A = actions.shape
        topk = min(self.cem_sync_topk, self.cem_sync_samples)
        expanded = self._expand_info(info_dict, self.cem_sync_samples)

        with torch.no_grad():
            var_t = self._compute_per_timestep_var(
                virtual_states, emb_0, goal_emb
            )
            a_mean = actions.detach().clone()

            for _ in range(self.gd_opt_steps):
                noise = torch.randn(
                    B, self.cem_sync_samples, T, A,
                    device=self.device,
                    generator=self.torch_gen,
                )
                samples = a_mean.unsqueeze(1) + noise * var_t.unsqueeze(1).sqrt()

                costs = self.model.get_cost(expanded, samples)

                _, elite_idx = torch.topk(costs, topk, dim=1, largest=False)
                elite = torch.gather(
                    samples, 1,
                    elite_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, A),
                )

                a_mean = elite.mean(dim=1)
                var_t = (
                    elite.var(dim=1).mean(dim=-1, keepdim=True)
                    + self.cem_sync_var_min
                )

        return a_mean

    def _sync_step(
        self,
        actions: torch.Tensor,
        virtual_states: torch.Tensor,
        emb_0: torch.Tensor,
        goal_emb: torch.Tensor,
        info_dict: dict,
    ) -> torch.Tensor:
        """Dispatch to the configured sync strategy."""
        if self.sync_mode == 'cem':
            return self._cem_sync(
                actions, virtual_states, emb_0, goal_emb, info_dict
            )
        return self._gd_sync(actions, info_dict)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_observations(self, info_dict: dict) -> dict:
        """Encode pixels and goal into latent embeddings if not already present.
        Calls ``model.encode()`` to produce ``emb_key`` and ``goal_emb_key``
        entries in ``info_dict``.  If the embeddings are already present
        (e.g. from a previous solver call), this is a no-op.
        After encoding, both keys hold flat ``(B, D)`` tensors (last
        timestep of the history is kept).
        """
        device = self.device

        if self.emb_key not in info_dict:
            with torch.no_grad():
                obs_info = {'pixels': info_dict['pixels'].to(device).float()}
                obs_info = self.model.encode(obs_info)
                # (B, T, D) → (B, D): take last history frame
                emb = obs_info['emb']
                info_dict[self.emb_key] = emb[:, -1] if emb.ndim == 3 else emb

        if self.goal_emb_key not in info_dict:
            with torch.no_grad():
                goal_pixels = info_dict['goal'].to(device).float()
                # goal may lack the time dim — add one if needed
                if goal_pixels.ndim == 4:  # (B, C, H, W)
                    goal_pixels = goal_pixels.unsqueeze(1)
                goal_info = {'pixels': goal_pixels}
                goal_info = self.model.encode(goal_info)
                emb = goal_info['emb']
                info_dict[self.goal_emb_key] = (
                    emb[:, -1] if emb.ndim == 3 else emb
                )

        return info_dict

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        info_dict: dict,
        init_action: torch.Tensor | None = None,
    ) -> dict:
        """Solve the planning problem using GRASP."""
        start_time = time.time()

        # Encode observations → latent embeddings (no-op if already present)
        info_dict = self._encode_observations(info_dict)

        emb_0_full = info_dict[self.emb_key].to(self.device)
        goal_emb_full = info_dict[self.goal_emb_key].to(self.device)
        total_envs = emb_0_full.shape[0]
        batch_size = (
            self.batch_size if self.batch_size is not None else total_envs
        )

        # Phase length for scheduled decay
        phase_len = self.gd_interval if self.gd_interval > 0 else self.n_steps

        batch_actions_list: list[torch.Tensor] = []
        batch_vs_list: list[torch.Tensor] = []
        loss_history: list[list[float]] = []

        for start_idx in range(0, total_envs, batch_size):
            end_idx = min(start_idx + batch_size, total_envs)
            emb_0 = emb_0_full[start_idx:end_idx]
            goal_emb = goal_emb_full[start_idx:end_idx]
            B = emb_0.shape[0]

            # Slice info_dict for this batch
            batch_info: dict = {}
            for k, v in info_dict.items():
                if torch.is_tensor(v):
                    batch_info[k] = v[start_idx:end_idx].to(self.device)
                elif isinstance(v, np.ndarray):
                    batch_info[k] = v[start_idx:end_idx]
                else:
                    batch_info[k] = v

            batch_init = (
                init_action[start_idx:end_idx]
                if init_action is not None
                else None
            )

            virtual_states = self._init_virtual_states(emb_0, goal_emb)
            actions = self._init_actions(batch_init, B)

            param_groups: list[dict] = [{'params': [actions], 'lr': self.lr_a}]
            if virtual_states.numel() > 0:
                param_groups.append(
                    {'params': [virtual_states], 'lr': self.lr_s}
                )
            optim = torch.optim.Adam(param_groups)

            batch_loss_history: list[float] = []

            for k in range(self.n_steps):
                # Scheduled decay within each sync phase
                if self.schedule_decay:
                    phase_step = k % phase_len if self.gd_interval > 0 else k
                    decay_frac = phase_step / max(phase_len - 1, 1)
                    cur_noise = (
                        self.init_noise_scale
                        + (self.min_noise_scale - self.init_noise_scale)
                        * decay_frac
                    )
                    cur_goal_weight = (
                        self.init_goal_weight
                        + (self.min_goal_weight - self.init_goal_weight)
                        * decay_frac
                    )
                else:
                    cur_noise = self.state_noise_scale
                    cur_goal_weight = self.goal_weight

                optim.zero_grad()
                loss = self._compute_loss(
                    virtual_states,
                    actions,
                    emb_0,
                    goal_emb,
                    batch_info,
                    cur_goal_weight,
                )
                loss.backward()
                optim.step()
                batch_loss_history.append(loss.item())

                # Langevin-style noise
                if cur_noise > 0 and virtual_states.numel() > 0:
                    with torch.no_grad():
                        virtual_states.data += cur_noise * torch.randn(
                            virtual_states.shape,
                            generator=self.torch_gen,
                            device=self.device,
                        )

                # Periodic sync (skip k=0)
                need_sync = (
                    self.gd_interval > 0
                    and k > 0
                    and (k + 1) % self.gd_interval == 0
                )
                if need_sync:
                    synced = self._sync_step(
                        actions, virtual_states, emb_0, goal_emb, batch_info
                    )
                    actions.data.copy_(synced)
                    # Rebuild optimiser to reset momentum
                    optim = torch.optim.Adam(param_groups)

            batch_actions_list.append(actions.detach().cpu())
            batch_vs_list.append(virtual_states.detach().cpu())
            loss_history.append(batch_loss_history)

        logging.info(
            f'GRASPSolver.solve completed in'
            f' {time.time() - start_time:.4f} seconds.'
        )

        return {
            'actions': torch.cat(batch_actions_list, dim=0),
            'virtual_states': torch.cat(batch_vs_list, dim=0),
            'loss_history': loss_history,
        }