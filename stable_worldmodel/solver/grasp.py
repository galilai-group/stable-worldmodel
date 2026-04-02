"""GRASP: Gradient RelAxed Stochastic Planner for model-based planning.

Reference: https://arxiv.org/abs/2602.00475

This solver jointly optimises virtual intermediate latent states and actions
via stop-gradient dynamics, goal shaping, Langevin-style state noise, and a
periodic full-rollout sync that calls ``model.get_cost``.

Unlike the reference implementations, this version does **not** hard-code an
MSE loss in sync phases; it delegates cost computation entirely to
``model.get_cost`` from the stable-worldmodel :class:`.Costable` interface.

The model passed to :class:`GRASPSolver` must expose:

* **get_cost** (the :class:`.Costable` protocol) — used during periodic
  sync steps to ground the terminal state to the goal.
* **rollout** (the :class:`.Rollable` protocol) — used for single-step
  differentiable latent predictions in the main GRASP optimisation loop.

``info_dict`` passed to :meth:`GRASPSolver.solve` must contain:

* ``emb_key`` (default ``'emb'``): ``(B, D)`` initial latent embedding.
* ``goal_emb_key`` (default ``'goal_emb'``): ``(B, D)`` goal latent embedding.
* Any additional keys required by ``model.get_cost`` (e.g. pixel observations
  for the sync pass).
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

    Jointly optimises virtual intermediate latent states ``s_1 .. s_{T-1}``
    and actions ``a_0 .. a_{T-1}`` by minimising:

    .. math::

        \\mathcal{L} = \\sum_{t=0}^{T-1}
            \\|F(\\bar{s}_t,\\, a_t) - s_{t+1}\\|^2
            + \\lambda_{\\text{goal}}
            \\sum_{t=0}^{T-1} \\|F(\\bar{s}_t,\\, a_t) - g\\|^2

    where :math:`\\bar{s}_t` denotes the stop-gradient of the virtual state at
    step *t*, ``s_0`` is the encoded initial observation (fixed), and
    ``s_T = g`` (the goal embedding, also fixed).

    Every ``gd_interval`` gradient steps a full-rollout sync pass is
    triggered.  Two sync strategies are available via ``sync_mode``:

    * ``'gd'`` — short Adam optimisation on a cloned copy of the actions using
      ``model.get_cost`` (differentiable, uniform treatment across timesteps).
    * ``'cem'`` — gradient-free Cross Entropy Method using ``model.get_cost``
      (more robust in non-smooth cost landscapes; uses per-timestep adaptive
      variance derived from virtual-state drift relative to the linear
      interpolation baseline).

    Between gradient steps Langevin-style Gaussian noise is injected into the
    virtual states to escape local minima.  With ``schedule_decay=True``, both
    the noise scale and the goal weight anneal linearly from their initial
    values to their minimum values within each sync phase (or over all
    ``n_steps`` when ``gd_interval=0``), trading exploration for exploitation
    as optimisation proceeds.

    When ``n_envs`` is large and the full batch does not fit in GPU memory,
    set ``batch_size`` to chunk environments into sub-batches.  Each sub-batch
    runs the full ``n_steps`` optimisation independently.

    Args:
        model: World model implementing :class:`.Costable` (``get_cost``)
            and :class:`.Rollable` (``rollout``).
        n_steps: Total number of optimisation iterations per environment
            batch.
        batch_size: Number of environments to process in a single pass.
            ``None`` processes all environments together.
        lr_s: Learning rate for virtual-state optimisation.
        lr_a: Learning rate for action optimisation.
        goal_weight: Fixed weighting coefficient :math:`\\lambda_{\\text{goal}}`
            for the per-step goal loss (used when ``schedule_decay=False``).
        state_noise_scale: Fixed standard deviation of Langevin-style noise
            added to the virtual states (used when ``schedule_decay=False``).
        gd_interval: Run the sync step every this many iterations.
            ``0`` disables the sync entirely.
        gd_opt_steps: Number of optimisation steps inside each sync pass
            (gradient steps for ``sync_mode='gd'``; CEM iterations for
            ``sync_mode='cem'``).
        gd_lr: Learning rate for the GD sync-pass Adam optimiser (only used
            when ``sync_mode='gd'``).
        sync_mode: Sync strategy — ``'gd'`` (gradient descent, default) or
            ``'cem'`` (Cross Entropy Method).
        cem_sync_samples: Number of candidate action sequences sampled per CEM
            iteration (only used when ``sync_mode='cem'``).
        cem_sync_topk: Number of elite samples kept for the CEM update (only
            used when ``sync_mode='cem'``).
        cem_sync_var_scale: Scaling factor applied to the per-timestep adaptive
            variance in CEM sync (only used when ``sync_mode='cem'``).
        cem_sync_var_min: Minimum variance floor for CEM sync to prevent
            premature distribution collapse (only used when
            ``sync_mode='cem'``).
        schedule_decay: When ``True``, both the state noise scale and the goal
            loss weight decay linearly from their initial to minimum values
            within each sync phase (or over all ``n_steps`` when
            ``gd_interval=0``).
        init_noise_scale: Initial Langevin noise scale at the start of each
            phase (only used when ``schedule_decay=True``).
        min_noise_scale: Final Langevin noise scale at the end of each phase
            (only used when ``schedule_decay=True``).
        init_goal_weight: Initial goal loss weight at the start of each phase
            (only used when ``schedule_decay=True``).
        min_goal_weight: Final goal loss weight at the end of each phase
            (only used when ``schedule_decay=True``).
        emb_key: Key in ``info_dict`` for the initial latent embedding.
        goal_emb_key: Key in ``info_dict`` for the goal latent embedding.
        device: Device for tensor computations.
        seed: Random seed for reproducibility.
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
        min_goal_weight: float = 0.0,
        emb_key: str = 'emb',
        goal_emb_key: str = 'goal_emb',
        device: str | torch.device = 'cpu',
        seed: int = 1234,
    ) -> None:
        if not hasattr(model, 'rollout'):
            raise TypeError(
                f'GRASPSolver requires a model with a rollout method, '
                f'got {type(model).__name__}. '
                'Implement rollout(info_dict, action_sequence) -> dict '
                '(see the Rollable protocol).'
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
        """Configure the solver with environment specifications.

        Args:
            action_space: The action space of the environment.
            n_envs: Number of parallel environments.
            config: Planning configuration object (e.g. ``PlanConfig``).
        """
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

        if not isinstance(action_space, Box):
            logging.warning(
                f'Action space is discrete, got {type(action_space)}. '
                'GRASPSolver requires a continuous action space and may not '
                'work as expected.'
            )

    @property
    def n_envs(self) -> int:
        """Number of parallel environments."""
        return self._n_envs

    @property
    def action_dim(self) -> int:
        """Flattened action dimension including action_block grouping."""
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        """Planning horizon in timesteps."""
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        """Make solver callable, forwarding to :meth:`solve`."""
        return self.solve(*args, **kwargs)

    def _init_virtual_states(
        self,
        emb_0: torch.Tensor,
        goal_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Initialise virtual states by linear interpolation in latent space.

        Creates ``T-1`` intermediate virtual states along the straight line
        from ``emb_0`` to ``goal_emb`` and enables gradient computation on
        the returned tensor.

        Args:
            emb_0: ``(B, D)`` initial latent embedding (no grad).
            goal_emb: ``(B, D)`` goal latent embedding (no grad).

        Returns:
            ``(B, T-1, D)`` tensor of virtual states with
            ``requires_grad=True`` when ``T > 1``, otherwise an empty tensor.
        """
        num_virtual = self.horizon - 1
        if num_virtual > 0:
            t = torch.linspace(0, 1, num_virtual + 2, device=self.device)
            t = t[1:-1].view(1, -1, 1)  # (1, T-1, 1)
            virtual = emb_0.unsqueeze(1) + t * (goal_emb - emb_0).unsqueeze(
                1
            )  # (B, T-1, D)
        else:
            B, D = emb_0.shape
            virtual = torch.empty(B, 0, D, device=self.device)

        return virtual.detach().requires_grad_(virtual.numel() > 0)

    def _init_actions(
        self,
        actions: torch.Tensor | None,
        B: int,
    ) -> torch.Tensor:
        """Initialise the action tensor for optimisation.

        Zero-pads ``actions`` to the full planning horizon when shorter than
        ``horizon``, then returns the tensor with ``requires_grad=True``.

        Args:
            actions: Optional ``(B, H', action_dim)`` warm-start actions.
            B: Batch size for the current call.

        Returns:
            ``(B, T, action_dim)`` action tensor with ``requires_grad=True``.
        """
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

    def _compute_loss(
        self,
        virtual_states: torch.Tensor,
        actions: torch.Tensor,
        emb_0: torch.Tensor,
        goal_emb: torch.Tensor,
        info_dict: dict,
        goal_weight: float | None = None,
    ) -> torch.Tensor:
        """Compute the GRASP dynamics + goal loss.

        Builds the full virtual state sequence
        ``[s_0, s_1, ..., s_{T-1}, g]`` and for each timestep *t*:

        * Calls ``model.rollout`` with the stop-gradient state ``sg(s_t)``
          and single-step action ``a_t`` to obtain the predicted next state
          ``pred_{t+1}``.
        * Accumulates the dynamics loss
          ``||pred_{t+1} - s_{t+1}||^2`` (gradient flows through the virtual
          target ``s_{t+1}`` but *not* through the dynamics input ``s_t``).
        * Accumulates the goal loss
          ``goal_weight * ||pred_{t+1} - g||^2`` (encourages every predicted
          state to approach the goal).

        ``emb_0`` and ``goal_emb`` are always detached: they are fixed anchor
        states and must never receive gradients through this computation.

        Args:
            virtual_states: ``(B, T-1, D)`` optimisable intermediate states.
            actions: ``(B, T, action_dim)`` optimisable action sequence.
            emb_0: ``(B, D)`` initial latent state (fixed, no grad).
            goal_emb: ``(B, D)`` goal latent state (fixed, no grad).
            info_dict: Base info dict forwarded to ``model.rollout``.
            goal_weight: Override for the goal loss coefficient.  Defaults to
                ``self.goal_weight`` when ``None``.

        Returns:
            Scalar loss tensor (summed over timesteps, averaged over batch).
        """
        if goal_weight is None:
            goal_weight = self.goal_weight

        # Full state sequence: [s_0, s_1, ..., s_{T-1}, g] — (B, T+1, D)
        # emb_0 and goal_emb are always detached — they are fixed anchor states
        # and must never receive gradients through this computation.
        s_full = torch.cat(
            [
                emb_0.detach().unsqueeze(1),
                virtual_states,
                goal_emb.detach().unsqueeze(1),
            ],
            dim=1,
        )

        terms: list[torch.Tensor] = []

        for t in range(self.horizon):
            s_t = s_full[
                :, t
            ].detach()  # stop-gradient on dynamics input (B, D)
            s_next = s_full[:, t + 1]  # (B, D); grad flows for virtual states

            # Single-step, single-sample action: (B, S=1, T=1, action_dim)
            a_t = actions[:, t : t + 1].unsqueeze(1)

            # Call model.rollout with the current (detached) latent state
            step_info = {**info_dict, self.emb_key: s_t}
            self.model.rollout(step_info, a_t)

            # predicted_emb: (B, S=1, T=1, D) → (B, D)
            pred_next = step_info['predicted_emb'][:, 0, 0]

            terms.append(((pred_next - s_next) ** 2).mean())
            terms.append(
                goal_weight * ((pred_next - goal_emb.detach()) ** 2).mean()
            )

        return torch.stack(terms).sum()

    def _compute_per_timestep_var(
        self,
        virtual_states: torch.Tensor,
        emb_0: torch.Tensor,
        goal_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-timestep CEM variance from virtual-state drift.

        Measures the mean squared deviation of each virtual state from its
        position on the linear interpolation baseline between ``emb_0`` and
        ``goal_emb``.  Timesteps where the optimiser moved farthest from the
        linear prior receive higher exploration variance, reflecting where the
        relaxed dynamics were most uncertain.

        Args:
            virtual_states: ``(B, T-1, D)`` current virtual states (may be
                empty when ``horizon == 1``).
            emb_0: ``(B, D)`` initial latent state.
            goal_emb: ``(B, D)`` goal latent state.

        Returns:
            ``(B, T, 1)`` per-timestep variance tensor.
        """
        B, T = emb_0.shape[0], self.horizon

        if virtual_states.numel() > 0:
            num_virtual = T - 1
            t = torch.linspace(0, 1, num_virtual + 2, device=self.device)
            t = t[1:-1].view(1, -1, 1)  # (1, T-1, 1)
            expected = emb_0.detach().unsqueeze(1) + t * (
                goal_emb.detach() - emb_0.detach()
            ).unsqueeze(1)  # (B, T-1, D)

            # Per-timestep MSE vs. linear baseline: (B, T-1)
            deviation = ((virtual_states.detach() - expected) ** 2).mean(
                dim=-1
            )
            # Pad to cover all T timesteps; step 0 uses the mean deviation
            mean_dev = deviation.mean(dim=-1, keepdim=True)  # (B, 1)
            deviation_full = torch.cat([mean_dev, deviation], dim=1)  # (B, T)
            var_t = (
                self.cem_sync_var_scale * deviation_full
                + self.cem_sync_var_min
            )
        else:
            var_t = torch.full(
                (B, T), self.cem_sync_var_min, device=self.device
            )

        return var_t.unsqueeze(-1)  # (B, T, 1)

    def _gd_sync(
        self,
        actions: torch.Tensor,
        info_dict: dict,
    ) -> torch.Tensor:
        """Full-rollout GD sync using ``model.get_cost``.

        Runs ``gd_opt_steps`` Adam steps on a cloned copy of ``actions``,
        calling ``model.get_cost`` with the full action sequence (reshaped to
        include a sample dimension).  Only actions are updated; virtual states
        are unaffected.  This corrects for cumulative drift between the relaxed
        latent trajectory and the true model dynamics.

        Args:
            actions: ``(B, T, action_dim)`` current best actions.
            info_dict: Info dict passed directly to ``model.get_cost``.

        Returns:
            Refined actions ``(B, T, action_dim)``, detached.
        """
        a_sync = actions.detach().clone().requires_grad_(True)
        sync_opt = torch.optim.Adam([a_sync], lr=self.gd_lr)

        for _ in range(self.gd_opt_steps):
            sync_opt.zero_grad()
            # get_cost expects (B, S, T, action_dim) — add sample dim
            cost = self.model.get_cost(
                info_dict, a_sync.unsqueeze(1)
            )  # (B, 1)
            loss = cost.mean()
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
        """Full-rollout CEM sync using ``model.get_cost``.

        Gradient-free alternative to GD sync that is more robust in
        non-smooth cost landscapes.  Samples ``cem_sync_samples`` candidate
        action sequences with per-timestep adaptive variance (derived from
        virtual-state drift via :meth:`_compute_per_timestep_var`), evaluates
        them all via ``model.get_cost``, and updates the action mean using the
        top-``cem_sync_topk`` elite candidates.  Repeats for ``gd_opt_steps``
        CEM iterations.

        Args:
            actions: ``(B, T, action_dim)`` current best actions.
            virtual_states: ``(B, T-1, D)`` current virtual states.
            emb_0: ``(B, D)`` initial latent state.
            goal_emb: ``(B, D)`` goal latent state.
            info_dict: Info dict passed directly to ``model.get_cost``.

        Returns:
            Refined actions ``(B, T, action_dim)``, detached.
        """
        B, T, A = actions.shape
        topk = min(self.cem_sync_topk, self.cem_sync_samples)

        with torch.no_grad():
            var_t = self._compute_per_timestep_var(
                virtual_states, emb_0, goal_emb
            )  # (B, T, 1)
            a_mean = actions.detach().clone()

            for _ in range(self.gd_opt_steps):
                noise = torch.randn(
                    B,
                    self.cem_sync_samples,
                    T,
                    A,
                    device=self.device,
                    generator=self.torch_gen,
                )
                # (B, N, T, A) candidate action sequences
                samples = (
                    a_mean.unsqueeze(1) + noise * var_t.unsqueeze(1).sqrt()
                )

                # Evaluate all candidates in one call: (B, N)
                costs = self.model.get_cost(info_dict, samples)

                # Select elite candidates
                _, elite_idx = torch.topk(costs, topk, dim=1, largest=False)
                elite = torch.gather(
                    samples,
                    1,
                    elite_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, A),
                )  # (B, topk, T, A)

                # Update mean and recompute per-timestep variance from elites
                a_mean = elite.mean(dim=1)  # (B, T, A)
                var_t = (
                    elite.var(dim=1).mean(dim=-1, keepdim=True)
                    + self.cem_sync_var_min
                )  # (B, T, 1)

        return a_mean

    def _sync_step(
        self,
        actions: torch.Tensor,
        virtual_states: torch.Tensor,
        emb_0: torch.Tensor,
        goal_emb: torch.Tensor,
        info_dict: dict,
    ) -> torch.Tensor:
        """Dispatch to the configured sync strategy.

        Args:
            actions: ``(B, T, action_dim)`` current best actions.
            virtual_states: ``(B, T-1, D)`` current virtual states.
            emb_0: ``(B, D)`` initial latent state.
            goal_emb: ``(B, D)`` goal latent state.
            info_dict: Info dict forwarded to the sync method.

        Returns:
            Refined actions ``(B, T, action_dim)``, detached.
        """
        if self.sync_mode == 'cem':
            return self._cem_sync(
                actions, virtual_states, emb_0, goal_emb, info_dict
            )
        return self._gd_sync(actions, info_dict)

    def solve(
        self,
        info_dict: dict,
        init_action: torch.Tensor | None = None,
    ) -> dict:
        """Solve the planning problem using GRASP.

        Reads the initial latent embedding from ``info_dict[emb_key]``
        ``(B, D)`` and the goal embedding from ``info_dict[goal_emb_key]``
        ``(B, D)``, then runs joint optimisation over virtual latent states and
        actions.

        The full ``info_dict`` (including pixel observations when present) is
        forwarded unchanged to ``model.get_cost`` during each periodic sync
        pass, so it must contain every key that ``get_cost`` expects.

        When ``batch_size`` is set, environments are processed in consecutive
        sub-batches of that size.  Each sub-batch independently runs the full
        ``n_steps`` optimisation.  Results are concatenated in environment
        order.

        Args:
            info_dict: Environment information dict.  Must contain:

                * ``emb_key`` (default ``'emb'``): ``(B, D)`` initial latent
                  embedding.
                * ``goal_emb_key`` (default ``'goal_emb'``): ``(B, D)`` goal
                  latent embedding.
                * Any additional keys required by ``model.get_cost``.
            init_action: Optional ``(B, H', action_dim)`` warm-start actions.
                Zero-padded to the full horizon when ``H' < horizon``.

        Returns:
            Dictionary with:

            * ``'actions'``: ``(B, T, action_dim)`` optimised actions (CPU).
            * ``'virtual_states'``: ``(B, T-1, D)`` final virtual latent
              states (CPU).
            * ``'loss_history'``: ``list[list[float]]`` per-step scalar losses,
              one inner list per environment batch.
            * ``'cost'``: ``(B,)`` final cost for each environment evaluated
              via ``model.get_cost`` on the optimised actions (CPU).
        """
        start_time = time.time()

        emb_0_full = info_dict[self.emb_key].to(self.device)  # (B, D)
        goal_emb_full = info_dict[self.goal_emb_key].to(self.device)  # (B, D)
        total_envs = emb_0_full.shape[0]
        batch_size = (
            self.batch_size if self.batch_size is not None else total_envs
        )

        # Phase length for scheduled decay (one sync interval, or all steps)
        phase_len = self.gd_interval if self.gd_interval > 0 else self.n_steps

        batch_actions_list: list[torch.Tensor] = []
        batch_vs_list: list[torch.Tensor] = []
        loss_history: list[list[float]] = []

        for start_idx in range(0, total_envs, batch_size):
            end_idx = min(start_idx + batch_size, total_envs)
            emb_0 = emb_0_full[start_idx:end_idx]
            goal_emb = goal_emb_full[start_idx:end_idx]
            B = emb_0.shape[0]

            # Slice info_dict for this batch, forwarding all keys
            batch_info: dict = {}
            for key, v in info_dict.items():
                if torch.is_tensor(v):
                    batch_info[key] = v[start_idx:end_idx].to(self.device)
                elif isinstance(v, np.ndarray):
                    batch_info[key] = v[start_idx:end_idx]
                else:
                    batch_info[key] = v

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
                # Scheduled linear decay within each sync phase
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

                # Langevin-style noise injected into virtual states
                if cur_noise > 0 and virtual_states.numel() > 0:
                    with torch.no_grad():
                        virtual_states.data += cur_noise * torch.randn(
                            virtual_states.shape,
                            generator=self.torch_gen,
                            device=self.device,
                        )

                # Periodic sync step (skip k=0 to allow at least one gradient step)
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
                    # Rebuild optimiser to reset momentum after the in-place update
                    optim = torch.optim.Adam(param_groups)

            batch_actions_list.append(actions.detach().cpu())
            batch_vs_list.append(virtual_states.detach().cpu())
            loss_history.append(batch_loss_history)

        all_actions = torch.cat(
            batch_actions_list, dim=0
        )  # (B, T, action_dim)
        all_vs = torch.cat(batch_vs_list, dim=0)  # (B, T-1, D)

        # Compute final cost for every environment, chunked to match batch_size
        # and with info_dict tensors moved to self.device to avoid device mismatch.
        cost_chunks: list[torch.Tensor] = []
        for start_idx in range(0, total_envs, batch_size):
            end_idx = min(start_idx + batch_size, total_envs)
            chunk_info: dict = {}
            for key, v in info_dict.items():
                if torch.is_tensor(v):
                    chunk_info[key] = v[start_idx:end_idx].to(self.device)
                elif isinstance(v, np.ndarray):
                    chunk_info[key] = v[start_idx:end_idx]
                else:
                    chunk_info[key] = v
            with torch.no_grad():
                cost_chunks.append(
                    self.model.get_cost(
                        chunk_info,
                        all_actions[start_idx:end_idx]
                        .to(self.device)
                        .unsqueeze(1),
                    )
                    .squeeze(1)
                    .cpu()
                )
        final_cost = torch.cat(cost_chunks, dim=0)  # (B,)

        logging.info(
            f'GRASPSolver.solve completed in {time.time() - start_time:.4f} seconds.'
        )

        return {
            'actions': all_actions,
            'virtual_states': all_vs,
            'loss_history': loss_history,
            'cost': final_cost,  # (B,) — NEW
        }
