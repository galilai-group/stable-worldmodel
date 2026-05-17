import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from .module import (
    two_hot,
    two_hot_inv,
    log_std,
    gaussian_logprob,
    squash,
    SimNorm,
    NormedLinear,
    RunningScale,
    mlp,
    weight_init,
    zero_init,
)


class TDMPC2(nn.Module):
    """
    Main Neural Network Architecture for TD-MPC2.
    Handles dynamic encoding of modalities, latent dynamics, reward prediction, and action planning.

    Encoder takes observations only.

    Args:
        cfg: Configuration object containing model and training hyperparameters.
        extra_encoders: Optional pre-built ModuleDict of observation encoders.
            If provided, these are used directly instead of building default MLP
            encoders from cfg. Allows injecting custom encoder architectures
            (e.g. CNNs, transformers) without modifying this class.
            Output dims must match cfg.wm.encoding values.

    Assumptions:
        - Continuous Control: The algorithm assumes continuous action spaces.
        - Action Bounds: Actions are strictly assumed to be normalized to the range [-1.0, 1.0].
            The actor network and MPPI planner enforce this bound via Tanh and clamping.
        - Reward Scaling: Environment rewards and Q-values should fall roughly within the
            [vmin, vmax] range defined in the config, as they are discretized using two-hot encoding.
    """

    def __init__(self, cfg, extra_encoders: nn.ModuleDict | None = None):
        super().__init__()
        self.cfg = cfg
        self.scale = RunningScale(cfg.wm.tau)

        encoding_cfg = cfg.wm.get('encoding', {})
        self.use_pixels = 'pixels' in encoding_cfg
        self.latent_dim = 0

        if self.use_pixels:
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, 7, stride=2),
                nn.Mish(),
                nn.Conv2d(32, 32, 5, stride=2),
                nn.Mish(),
                nn.Conv2d(32, 32, 3, stride=2),
                nn.Mish(),
                nn.Conv2d(32, 32, 3, stride=1),
                nn.Mish(),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 3, cfg.image_size, cfg.image_size)
                cnn_out_dim = self.cnn(dummy).shape[1]

            pixel_dim = encoding_cfg['pixels']
            self.pixel_encoder = nn.Linear(cnn_out_dim, pixel_dim)
            self.latent_dim += pixel_dim

        if extra_encoders is not None:
            self.extra_encoders = extra_encoders
        else:
            # Default: build a two-layer MLP encoder for each non-pixel modality
            self.extra_encoders = nn.ModuleDict()
            for key, out_dim in encoding_cfg.items():
                if key == 'pixels':
                    continue
                in_dim = cfg.extra_dims[key]
                self.extra_encoders[key] = nn.Sequential(
                    NormedLinear(in_dim, cfg.wm.enc_dim),
                    nn.Linear(cfg.wm.enc_dim, out_dim),
                    nn.LayerNorm(out_dim),
                )

        # Accumulate latent dim from all non-pixel encoders
        for key, out_dim in encoding_cfg.items():
            if key != 'pixels':
                self.latent_dim += out_dim

        assert self.latent_dim > 0, (
            'Model must have pixels or at least one extra_encoder defined.'
        )

        self.sim_norm = SimNorm(cfg)

        # Latent dynamics model: predicts next latent state z' from (z, a)
        self.dynamics = mlp(
            self.latent_dim + cfg.action_dim, cfg.wm.mlp_dim,
            self.latent_dim, act=SimNorm(cfg),
        )

        # Reward predictor: predicts expected reward from (z, a) as a two-hot distribution
        self.reward = mlp(
            self.latent_dim + cfg.action_dim, cfg.wm.mlp_dim, cfg.wm.num_bins
        )

        # Policy prior (actor): outputs (mean, log_std) of a Gaussian over actions given z.
        # Used both to compute the policy loss and to warm-start CEM planning.
        self.pi = mlp(self.latent_dim, cfg.wm.mlp_dim, 2 * cfg.action_dim)

        # Ensemble of Q-functions: each predicts action-value from (z, a) as a two-hot
        # distribution. An ensemble is used for clipped double-Q to reduce overestimation.
        self.qs = nn.ModuleList(
            [
                mlp(
                    self.latent_dim + cfg.action_dim,
                    cfg.wm.mlp_dim,
                    cfg.wm.num_bins,
                    dropout=0.01,
                )
                for _ in range(cfg.wm.num_q)
            ]
        )
        self.target_qs = deepcopy(self.qs)
        for p in self.target_qs.parameters():
            p.requires_grad = False

        # Weight initialization (matches official TD-MPC2)
        self.apply(weight_init)
        zero_init([self.reward[-1].weight])
        for q in self.qs:
            zero_init([q[-1].weight])
        for q in self.target_qs:
            zero_init([q[-1].weight])

    def encode(self, obs_dict: dict) -> torch.Tensor:
        """Encode observations into a SimNorm-normalized latent state.

        Handles any number of leading dimensions without ndim checks, following
        the DINO-WM convention where the encoder owns the reshaping and callers
        pass observations as-is:

        - (B, dim)       → (B, latent_dim)          single-step inference
        - (B, T, dim)    → (B, T, latent_dim)        offline training sequence
        - (B, N, dim)    → (B, N, latent_dim)        CEM candidate expansion

        Args:
            obs_dict: Dictionary of observations with any leading shape.

        Returns:
            SimNorm-regularized latent state preserving all leading dimensions.
        """
        embeddings = []
        target_dtype = next(self.parameters()).dtype

        # Process primary vision modality — flatten all leading dims into batch
        if self.use_pixels:
            obs = obs_dict['pixels'].to(target_dtype)
            if obs.shape[-1] == 3:
                obs = obs.movedim(-1, -3)
            lead_dims = obs.shape[:-3]                      # e.g. (B,) or (B, T)
            obs_flat  = obs.reshape(-1, *obs.shape[-3:])    # (prod(lead), C, H, W)
            cnn_out   = self.cnn(obs_flat)
            z_pixels  = self.pixel_encoder(cnn_out).view(*lead_dims, -1)
            embeddings.append(z_pixels)

        # Process extra modalities (state, proprioception, etc.)
        for key, encoder in self.extra_encoders.items():
            obs      = obs_dict[key].to(target_dtype)       # (*lead, dim)
            lead     = obs.shape[:-1]
            obs_flat = obs.reshape(-1, obs.shape[-1])       # (prod(lead), dim)
            z        = encoder(obs_flat).view(*lead, -1)    # (*lead, enc_dim)
            embeddings.append(z)

        z_concat = torch.cat(embeddings, dim=-1)
        return self.sim_norm(z_concat)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One-step world model prediction.

        Given a latent state and action, predicts the next latent state via the
        dynamics model and the expected reward as a two-hot logit vector.

        Args:
            z: Current latent state of shape (B, latent_dim).
            action: Action of shape (B, action_dim).

        Returns:
            Tuple of (next_z, reward_logits) with shapes (B, latent_dim) and
            (B, num_bins) respectively.
        """
        z_a = torch.cat([z, action], dim=-1)
        return self.dynamics(z_a), self.reward(z_a)

    def rollout(
        self, z: torch.Tensor, horizon: int, num_trajs: int = 1
    ) -> torch.Tensor:
        """Roll out the actor policy from a latent state for a given horizon.

        Samples ``num_trajs`` stochastic trajectories and returns their mean.

        Args:
            z: Initial latent state of shape (B, latent_dim).
            horizon: Number of steps to unroll.
            num_trajs: Number of independent trajectories to average.

        Returns:
            Mean action sequence of shape (B, horizon, action_dim).
        """
        trajs = []
        for _ in range(num_trajs):
            curr_z, traj = z, []
            for _ in range(horizon):
                mean_raw, log_std_raw = self.pi(curr_z).chunk(2, dim=-1)
                act = torch.tanh(
                    mean_raw
                    + log_std(log_std_raw, low=-10, dif=12).exp()
                    * torch.randn_like(mean_raw)
                )
                traj.append(act)
                curr_z = self.dynamics(torch.cat([curr_z, act], dim=-1))
            trajs.append(torch.stack(traj, dim=1))  # (B, horizon, action_dim)
        return torch.stack(trajs).mean(0)            # (B, horizon, action_dim)

    def get_action(
        self,
        info_dict: dict,
        horizon: int = 1,
        prefix_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample an action sequence from the actor policy via latent rollout.

        Encodes the current observation into a latent state, optionally advances
        it through ``prefix_actions`` via the dynamics model, then calls
        ``rollout`` for ``horizon`` steps.

        Args:
            info_dict: Dictionary containing environment state information with
                shape (B, ...).
            horizon: Number of steps to plan.
            prefix_actions: Optional warm-start actions of shape
                (B, t, action_dim) with t < horizon. The latent state is
                advanced through these steps before the actor rollout.

        Returns:
            Action tensor of shape (B, horizon, action_dim).
        """
        device = next(self.parameters()).device
        encoding_keys = list(self.cfg.wm.get('encoding', {}).keys())

        obs_dict = {key: info_dict[key].to(device) for key in encoding_keys}
        z = self.encode(obs_dict)

        if prefix_actions is not None:
            for t in range(prefix_actions.shape[1]):
                z = self.dynamics(
                    torch.cat([z, prefix_actions[:, t].to(device)], dim=-1)
                )

        num_trajs = self.cfg.get('num_pi_trajs', 1)
        return self.rollout(z, horizon, num_trajs)  # (B, horizon, action_dim)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Evaluate the cost of candidate action trajectories.

        Rolls out the world model for each candidate, accumulates discounted
        rewards, and adds a terminal value estimate with an optional uncertainty
        penalty to favour conservative planning.

        Args:
            info_dict: Dictionary containing environment state with shape (B, N, ...).
            action_candidates: Candidate action sequences of shape (B, N, H, A).

        Returns:
            Cost tensor of shape (B, N). Lower is better.
        """
        device = action_candidates.device
        encoding_keys = list(self.cfg.wm.get('encoding', {}).keys())

        obs_dict = {key: info_dict[key].to(device) for key in encoding_keys}
        z = self.encode(obs_dict)

        B, N, H, A = action_candidates.shape

        if z.ndim == 2 and z.shape[0] == B:
            z = z.unsqueeze(1).repeat(1, N, 1).view(B * N, -1)
        elif z.ndim == 3 and z.shape[0] == B and z.shape[1] == N:
            z = z.view(B * N, -1)
        elif z.ndim == 2 and z.shape[0] == B * N:
            pass
        else:
            raise ValueError(f'Unexpected latent state shape: {z.shape}')

        actions = action_candidates.view(B * N, H, A)

        G, discount = 0, 1.0
        c = self.cfg.wm.get('uncertainty_penalty', 0.5)
        termination = torch.zeros(B * N, 1, dtype=torch.float32, device=z.device)

        for t in range(H):
            z_a = torch.cat([z, actions[:, t]], dim=-1)
            reward = two_hot_inv(self.reward(z_a), self.cfg)
            z = self.dynamics(z_a)
            G = G + discount * (1 - termination) * reward
            discount = discount * self.cfg.wm.get('discount', 0.99)

        mu = self.pi(z).chunk(2, dim=-1)[0]
        action = torch.tanh(mu)
        z_a_term = torch.cat([z, action], dim=-1)

        q_logits = torch.stack([q(z_a_term) for q in self.qs])
        q_values  = torch.stack([two_hot_inv(logits, self.cfg) for logits in q_logits])

        q_mean = q_values.mean(dim=0)
        q_std  = q_values.std(dim=0)

        penalty        = c * q_mean.abs() * q_std
        conservative_q = q_mean - penalty
        total_return   = G + discount * (1 - termination) * conservative_q

        return -total_return.view(B, N)


def tdmpc2_forward(self, batch, stage, cfg):
    """Forward pass and loss computation for TD-MPC2.

    Designed to be used as a Lightning ``training_step`` or called directly
    from an online training loop via a context object that implements
    ``self.model`` and ``self.log_dict``.

    Args:
        batch: Dict with keys matching cfg.wm.encoding plus 'action' and 'reward'.
        stage: 'train' or 'validate'. Controls target-network soft update.
        cfg: OmegaConf config with wm.* hyperparameters.

    Returns:
        The batch dict with 'loss' set to the total scalar loss.
    """
    encoding_keys = list(cfg.wm.get('encoding', {}).keys())
    B, T_plus_1 = batch['action'].shape[:2]

    flat_obs_dict = {}
    for key in encoding_keys:
        obs = batch[key]
        flat_obs_dict[key] = obs.reshape(-1, *obs.shape[2:])

    all_z = self.model.encode(flat_obs_dict).reshape(B, T_plus_1, -1)

    z = all_z[:, 0]
    target_zs = all_z[:, 1:]

    loss_consistency, loss_reward, loss_value, loss_pi = 0, 0, 0, 0
    discount = cfg.wm.get('discount', 0.99)
    entropy_coef = cfg.wm.get('entropy_coef', 1e-4)

    for t in range(cfg.wm.horizon):
        action = batch['action'][:, t]
        reward = batch['reward'][:, t]

        next_z_pred, reward_pred = self.model.forward(z, action)

        loss_consistency += F.mse_loss(
            next_z_pred, target_zs[:, t].detach()
        ) * (cfg.wm.rho**t)
        target_reward = two_hot(reward, cfg)
        loss_reward += -(
            target_reward * F.log_softmax(reward_pred, dim=-1)
        ).sum(-1).mean() * (cfg.wm.rho**t)

        with torch.no_grad():
            next_z_for_q = target_zs[:, t].detach()
            mean_raw, log_std_raw = self.model.pi(next_z_for_q).chunk(2, dim=-1)
            log_std_bounded = log_std(log_std_raw, low=-10, dif=12)
            eps = torch.randn_like(mean_raw)
            next_action_pred = torch.tanh(mean_raw + eps * log_std_bounded.exp())

            next_z_a = torch.cat([next_z_for_q, next_action_pred], dim=-1)
            q_indices = random.sample(range(cfg.wm.num_q), 2)
            next_qs = [
                two_hot_inv(self.model.target_qs[i](next_z_a), cfg)
                for i in q_indices
            ]
            next_q_min = torch.min(next_qs[0], next_qs[1])
            target_q = reward.unsqueeze(1) + discount * next_q_min
            target_q_two_hot = two_hot(target_q, cfg)

        z_a = torch.cat([z, action], dim=-1)
        for q in self.model.qs:
            loss_value += -(
                target_q_two_hot * F.log_softmax(q(z_a), dim=-1)
            ).sum(-1).mean() * (cfg.wm.rho**t)

        z_detached = z.detach()
        mean_raw, log_std_raw = self.model.pi(z_detached).chunk(2, dim=-1)
        log_std_bounded = log_std(log_std_raw, low=-10, dif=12)
        eps = torch.randn_like(mean_raw)
        log_prob = gaussian_logprob(eps, log_std_bounded)

        action_pi_raw = mean_raw + eps * log_std_bounded.exp()
        mu, action_pi, log_prob = squash(mean_raw, action_pi_raw, log_prob)

        scaled_entropy = -log_prob * cfg.action_dim

        z_pi = torch.cat([z_detached, action_pi], dim=-1)
        try:
            self.model.qs.requires_grad_(False)
            qs_pi = torch.stack(
                [two_hot_inv(q(z_pi), cfg) for q in self.model.qs], dim=0
            )
        finally:
            self.model.qs.requires_grad_(True)

        q_indices = random.sample(range(cfg.wm.num_q), 2)
        q_pi_avg = (qs_pi[q_indices[0]] + qs_pi[q_indices[1]]) / 2.0

        if t == 0:
            self.model.scale.update(q_pi_avg)
        q_pi_normalized = self.model.scale(q_pi_avg)

        step_pi_loss = -(entropy_coef * scaled_entropy + q_pi_normalized)
        loss_pi += step_pi_loss.mean() * (cfg.wm.rho**t)

        z = next_z_pred

    loss_consistency /= cfg.wm.horizon
    loss_reward /= cfg.wm.horizon
    loss_value /= cfg.wm.horizon * cfg.wm.num_q
    loss_pi /= cfg.wm.horizon

    total_loss = (
        cfg.wm.consistency_coef * loss_consistency
        + cfg.wm.reward_coef * loss_reward
        + cfg.wm.value_coef * loss_value
        + loss_pi
    )

    self.log_dict(
        {
            f'{stage}/loss': total_loss,
            f'{stage}/consist': loss_consistency,
            f'{stage}/reward': loss_reward,
            f'{stage}/value': loss_value,
            f'{stage}/policy': loss_pi,
        },
        on_step=True,
        sync_dist=False,
        prog_bar=True,
    )

    if stage == 'train':
        for q, t_q in zip(self.model.qs, self.model.target_qs):
            for p, p_t in zip(q.parameters(), t_q.parameters()):
                p_t.data.lerp_(p.data, cfg.wm.tau)

    batch['loss'] = total_loss
    return batch


__all__ = [
    'TDMPC2',
    'tdmpc2_forward',
    'two_hot',
    'two_hot_inv',
    'log_std',
    'gaussian_logprob',
    'squash',
]
