import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

# ---------------------------------------------------------------------------
# TD-MPC2 Math Utilities
# ---------------------------------------------------------------------------


def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg):
    if x.ndim == 0:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 1:
        x = x.unsqueeze(-1)

    bin_size = (cfg.wm.vmax - cfg.wm.vmin) / (cfg.wm.num_bins - 1)
    x = torch.clamp(symlog(x), cfg.wm.vmin, cfg.wm.vmax)

    indices = (x - cfg.wm.vmin) / bin_size
    bin_idx = indices.floor().long()
    bin_offset = indices - bin_idx

    bin_idx = bin_idx.clamp(0, cfg.wm.num_bins - 2)

    soft_two_hot = torch.zeros(
        x.shape[0], cfg.wm.num_bins, device=x.device, dtype=x.dtype
    )
    soft_two_hot.scatter_(1, bin_idx, 1 - bin_offset)
    soft_two_hot.scatter_(1, bin_idx + 1, bin_offset)

    return soft_two_hot


def two_hot_inv(logits, cfg):
    device = logits.device
    bin_values = torch.linspace(
        cfg.wm.vmin, cfg.wm.vmax, cfg.wm.num_bins, device=device
    )
    probs = F.softmax(logits, dim=-1)
    x = torch.sum(probs * bin_values, dim=-1, keepdim=True)
    return symexp(x)


def log_std(x, low=-10, dif=12):
    return low + 0.5 * dif * (torch.tanh(x) + 1)


def gaussian_logprob(eps, log_std):
    residual = -0.5 * eps.pow(2) - log_std
    log_prob = residual - 0.9189385175704956
    return log_prob.sum(-1, keepdim=True)


def squash(mu, pi, log_pi):
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    squashed_pi = torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
    log_pi = log_pi - squashed_pi.sum(-1, keepdim=True)
    return mu, pi, log_pi


# ---------------------------------------------------------------------------
# TD-MPC2 Building Blocks
# ---------------------------------------------------------------------------


class SimNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # The paper uses V=8 as the default simplex dimensionality
        self.simplex_dim = cfg.wm.get('simnorm_dim', 8)

    def forward(self, x):
        shp = x.shape
        # Group the last dimension into L simplices of size V
        x = x.view(*shp[:-1], -1, self.simplex_dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)


class NormedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.ln = nn.LayerNorm(out_features)

    def forward(self, x):
        return F.mish(self.ln(super().forward(x)))


class RunningScale(nn.Module):
    def __init__(self, tau=0.01):
        super().__init__()
        self.tau = tau
        self.register_buffer(
            'value', torch.ones(1, dtype=torch.float32) * 10.0
        )

    def update(self, x):
        with torch.no_grad():
            percentile_95 = torch.quantile(x.detach().float(), 0.95)
            percentile_05 = torch.quantile(x.detach().float(), 0.05)
            scale_val = torch.clamp(percentile_95 - percentile_05, min=1e-4)
            self.value.data.lerp_(scale_val.unsqueeze(0), self.tau)

    def forward(self, x, update=False):
        if update:
            self.update(x)
        return x / self.value


def mlp(in_dim, mlp_dim, out_dim, dropout=0.0):
    layers = [
        NormedLinear(in_dim, mlp_dim),
        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        NormedLinear(mlp_dim, mlp_dim),
        nn.Linear(mlp_dim, out_dim),
    ]
    return nn.Sequential(*layers)


class TDMPC2(nn.Module):
    """
    Main Neural Network Architecture for TD-MPC2.
    Handles dynamic encoding of modalities, latent dynamics, reward prediction, and action planning.

    Assumptions:
        - Continuous Control: The algorithm assumes continuous action spaces.
        - Action Bounds: Actions are strictly assumed to be normalized to the range [-1.0, 1.0].
          The actor network and MPPI planner enforce this bound via Tanh and clamping.
        - Reward Scaling: Environment rewards and Q-values should fall roughly within the
          [vmin, vmax] range defined in the config, as they are discretized using two-hot encoding.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.scale = RunningScale(cfg.wm.tau)

        # Determine modalities directly from the encoding dict
        encoding_cfg = cfg.wm.get('encoding', {})
        self.use_pixels = 'pixels' in encoding_cfg
        self.latent_dim = 0

        if self.use_pixels:
            self.cnn = nn.Sequential(
                nn.Conv2d(6, 32, 7, stride=2),
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
                dummy = torch.zeros(1, 6, cfg.image_size, cfg.image_size)
                cnn_out_dim = self.cnn(dummy).shape[1]

            pixel_dim = encoding_cfg['pixels']
            self.pixel_encoder = nn.Linear(cnn_out_dim, pixel_dim)
            self.latent_dim += pixel_dim

        self.extra_encoders = nn.ModuleDict()
        for key, out_dim in encoding_cfg.items():
            if key == 'pixels':
                continue  # Handled by primary backbone

            in_dim = cfg.extra_dims[key] * 2
            self.extra_encoders[key] = nn.Sequential(
                NormedLinear(in_dim, cfg.wm.enc_dim),
                NormedLinear(cfg.wm.enc_dim, out_dim),
            )
            self.latent_dim += out_dim

        assert self.latent_dim > 0, (
            'Model must have pixels or at least one extra_encoder defined.'
        )

        self.sim_norm = SimNorm(cfg)

        self.dynamics = nn.Sequential(
            NormedLinear(self.latent_dim + cfg.action_dim, cfg.wm.mlp_dim),
            NormedLinear(cfg.wm.mlp_dim, cfg.wm.mlp_dim),
            NormedLinear(cfg.wm.mlp_dim, self.latent_dim),
            SimNorm(cfg),
        )
        self.reward = mlp(
            self.latent_dim + cfg.action_dim, cfg.wm.mlp_dim, cfg.wm.num_bins
        )
        self.pi = mlp(self.latent_dim, cfg.wm.mlp_dim, 2 * cfg.action_dim)

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

    def encode(self, obs_dict, goal_dict):
        """
        Encodes primary vision and extra modalities separately, then concatenates them into a single latent state.

        Args:
            obs_dict (dict): Dictionary of current observations (e.g., 'pixels', 'state').
            goal_dict (dict): Dictionary of goal observations.

        Returns:
            torch.Tensor: The SimNorm-regularized latent representation of shape (B, latent_dim).
        """
        embeddings = []
        target_dtype = next(self.parameters()).dtype

        # Process Primary Vision
        if self.use_pixels:
            obs = obs_dict['pixels'].to(target_dtype)
            goal = goal_dict['pixels'].to(target_dtype)

            if obs.shape[-1] == 3:
                obs = obs.movedim(-1, -3)
                goal = goal.movedim(-1, -3)
            lead_dims = obs.shape[:-3]
            obs_flat = obs.reshape(-1, *obs.shape[-3:])
            goal_flat = goal.reshape(-1, *goal.shape[-3:])

            x_g = torch.cat([obs_flat, goal_flat], dim=-3)
            cnn_out = self.cnn(x_g)
            z_pixels = self.pixel_encoder(cnn_out).view(*lead_dims, -1)
            embeddings.append(z_pixels)

        # Process Extra Modalities (States, Proprio, etc.)
        for key, encoder in self.extra_encoders.items():
            obs = obs_dict[key].to(target_dtype)
            goal = goal_dict[key].to(target_dtype)
            x_g = torch.cat([obs, goal], dim=-1)
            z_extra = encoder(x_g)
            embeddings.append(z_extra)

        z_concat = torch.cat(embeddings, dim=-1)
        return self.sim_norm(z_concat)

    def forward(self, z, action):
        """
        Predicts the next latent state and expected reward given the current latent state and action.
        """
        z_a = torch.cat([z, action], dim=-1)
        return self.dynamics(z_a), self.reward(z_a)

    @torch.no_grad()
    def _estimate_value(self, z, actions, task=None):
        """
        Estimates the total expected return of an action sequence using latent dynamics.
        Implements Offline RL Test-Time Regularization (Appendix J) by applying an uncertainty penalty
        based on the standard deviation of the Q-value ensemble to avoid out-of-distribution actions.

        Args:
            z (torch.Tensor): Initial latent state of shape (B, latent_dim).
            actions (torch.Tensor): Sequence of actions of shape (B, horizon, action_dim).
            task (Any, optional): Task identifier (unused in single-task).

        Returns:
            torch.Tensor: The conservative value estimate of shape (B,).
        """
        G, discount = 0, 1
        horizon = actions.shape[1]
        num_samples = actions.shape[0]

        c = self.cfg.wm.get('uncertainty_penalty', 0.5)

        termination = torch.zeros(
            num_samples, 1, dtype=torch.float32, device=z.device
        )
        for t in range(horizon):
            z_a = torch.cat([z, actions[:, t]], dim=-1)
            reward = two_hot_inv(self.reward(z_a), self.cfg)
            z = self.dynamics(z_a)
            G = G + discount * (1 - termination) * reward
            discount = discount * self.cfg.wm.get('discount', 0.99)

        mu = self.pi(z).chunk(2, dim=-1)[0]
        action = torch.tanh(mu)
        z_a_term = torch.cat([z, action], dim=-1)

        q_logits = torch.stack([q(z_a_term) for q in self.qs])
        q_values = torch.stack(
            [two_hot_inv(logits, self.cfg) for logits in q_logits]
        )

        q_mean = q_values.mean(dim=0)
        q_std = q_values.std(dim=0)

        penalty = c * q_mean.abs() * q_std
        conservative_q = q_mean - penalty

        return G + discount * (1 - termination) * conservative_q

    @torch.no_grad()
    def _plan(self, obs_dict, goal_dict, step_idxs, eval_mode=False):
        """
        Performs batched Policy-Guided Model Predictive Path Integral (MPPI) planning in the latent space.

        Args:
            obs_dict (dict): Dictionary of current observations.
            goal_dict (dict): Dictionary of goal observations.
            step_idxs (torch.Tensor): Tensor of shape (B,) indicating the current episode step for warm-start resets.
            eval_mode (bool): Whether the agent is in evaluation mode.

        Returns:
            torch.Tensor: The optimal first action of the planned sequence, shape (B, action_dim).
        """
        z = self.encode(obs_dict, goal_dict)  # Shape: (B, latent_dim)
        B = z.shape[0]
        horizon = self.cfg.wm.horizon
        action_dim = self.cfg.action_dim

        num_pi_trajs = self.cfg.get('num_pi_trajs', 24)
        num_samples = self.cfg.get('num_samples', 512)
        num_elites = self.cfg.get('num_elites', 64)
        max_std = self.cfg.get('max_std', 2.0)
        min_std = self.cfg.get('min_std', 0.05)
        iterations = self.cfg.get('iterations', 6)
        temperature = self.cfg.get('temperature', 0.5)

        # Batched Temporal Warm-Start Initialization
        if not hasattr(self, '_prev_mean') or self._prev_mean.shape[0] != B:
            self._prev_mean = torch.zeros(
                B, horizon, action_dim, device=self.device
            )

        # Reset environments that just started an episode (t0)
        is_first_step = step_idxs == 0
        self._prev_mean[is_first_step] = 0.0

        if num_pi_trajs > 0:
            pi_actions = torch.empty(
                B, horizon, num_pi_trajs, action_dim, device=self.device
            )
            _z = (
                z.unsqueeze(1)
                .repeat(1, num_pi_trajs, 1)
                .view(B * num_pi_trajs, -1)
            )
            for t in range(horizon - 1):
                mu = self.pi(_z).chunk(2, dim=-1)[0]
                act = torch.tanh(mu)
                pi_actions[:, t] = act.view(B, num_pi_trajs, action_dim)
                _z = self.dynamics(torch.cat([_z, act], dim=-1))
            mu = self.pi(_z).chunk(2, dim=-1)[0]
            pi_actions[:, -1] = torch.tanh(mu).view(
                B, num_pi_trajs, action_dim
            )

        mean = torch.zeros(B, horizon, action_dim, device=self.device)
        mean[:, :-1] = self._prev_mean[:, 1:]  # Temporal shift
        std = torch.full(
            (B, horizon, action_dim),
            max_std,
            dtype=z.dtype,
            device=self.device,
        )

        z_expanded = (
            z.unsqueeze(1).repeat(1, num_samples, 1).view(B * num_samples, -1)
        )
        actions = torch.empty(
            B, horizon, num_samples, action_dim, device=self.device
        )

        if num_pi_trajs > 0:
            actions[:, :, :num_pi_trajs] = pi_actions

        for _ in range(iterations):
            # Sample noise
            r = torch.randn(
                B,
                horizon,
                num_samples - num_pi_trajs,
                action_dim,
                device=self.device,
            )

            # Apply batched mean/std
            actions_sample = mean.unsqueeze(2) + std.unsqueeze(2) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, :, num_pi_trajs:] = actions_sample

            # Reshape for _estimate_value: (B * num_samples, horizon, action_dim)
            flat_actions = actions.permute(0, 2, 1, 3).reshape(
                B * num_samples, horizon, action_dim
            )

            value = self._estimate_value(z_expanded, flat_actions).nan_to_num(
                0
            )
            value = value.view(
                B, num_samples
            )  # Unflatten back to batch format

            # TopK Elites along the sample dimension
            elite_idxs = torch.topk(
                value, num_elites, dim=1
            ).indices  # Shape: (B, num_elites)
            elite_value = torch.gather(value, 1, elite_idxs)

            # Gather elite actions
            idx_expanded = (
                elite_idxs.unsqueeze(1)
                .unsqueeze(3)
                .expand(-1, horizon, -1, action_dim)
            )
            elite_actions = torch.gather(
                actions, 2, idx_expanded
            )  # Shape: (B, horizon, num_elites, action_dim)

            # Update mean and std
            max_value = elite_value.max(dim=1, keepdim=True).values
            score = torch.exp(temperature * (elite_value - max_value))
            score = score / (score.sum(dim=1, keepdim=True) + 1e-9)

            score_expanded = score.unsqueeze(1).unsqueeze(
                3
            )  # Shape: (B, 1, num_elites, 1)

            mean = (score_expanded * elite_actions).sum(dim=2)
            variance = (
                score_expanded * (elite_actions - mean.unsqueeze(2)) ** 2
            ).sum(dim=2)
            std = variance.sqrt().clamp(min_std, max_std)

        best_idx = torch.argmax(score, dim=1)  # (B,)
        best_idx_expanded = best_idx.view(B, 1, 1, 1).expand(
            -1, 1, -1, action_dim
        )
        best_actions = (
            torch.gather(elite_actions[:, 0:1, :, :], 2, best_idx_expanded)
            .squeeze(1)
            .squeeze(1)
        )

        self._prev_mean.copy_(mean)
        return best_actions.clamp(-1, 1)  # Shape: (B, A)

    @torch.no_grad()
    def get_action(self, info_dict: dict) -> torch.Tensor:
        """
        Main entry point for action selection during evaluation or environment stepping.
        Extracts modalities from the observation dictionary and routes them to the MPPI planner or raw policy.

        Args:
            info_dict (dict): Dictionary containing environment observations and step information.

        Returns:
            torch.Tensor: The selected action of shape (B, action_dim).
        """
        encoding_keys = list(self.cfg.wm.get('encoding', {}).keys())

        # Get step_idxs safely based on the batch size of the first modality
        first_key = encoding_keys[0]
        step_idxs = info_dict.get(
            'step_idx',
            torch.zeros(
                info_dict[first_key].shape[0],
                device=info_dict[first_key].device,
            ),
        )

        obs_dict = {}
        goal_dict = {}

        for key in encoding_keys:
            obs_dict[key] = info_dict[key]

            eval_goal_key = f'goal_{key}'
            if eval_goal_key not in info_dict and 'goal' in info_dict:
                eval_goal_key = 'goal'

            goal_dict[key] = info_dict[eval_goal_key]

        use_mpc = self.cfg.get('mpc', True)
        if use_mpc:
            return self._plan(obs_dict, goal_dict, step_idxs=step_idxs)
        else:
            z = self.encode(obs_dict, goal_dict)
            mu = self.pi(z).chunk(2, dim=-1)[0]
            return torch.tanh(mu)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """
        Evaluates the cost of candidate action trajectories using the learned world model.
        Used by external trajectory optimizers (like CEM or external MPPI) to rank sequences.

        Args:
            info_dict (dict): Dictionary containing current observations and goals.
            action_candidates (torch.Tensor): Candidate action sequences of shape (B, N, H, A)
                                              where N is the number of samples and H is the horizon.

        Returns:
            torch.Tensor: The estimated cost (negative conservative return) of each trajectory, shape (B, N).
        """
        device = action_candidates.device
        encoding_keys = list(self.cfg.wm.get('encoding', {}).keys())

        obs_dict = {}
        goal_dict = {}

        for key in encoding_keys:
            obs = info_dict[key].to(device)

            eval_goal_key = f'goal_{key}'
            if eval_goal_key not in info_dict and 'goal' in info_dict:
                eval_goal_key = 'goal'
            goal = info_dict[eval_goal_key].to(device)

            if obs.ndim == 3 and key != 'pixels':
                obs = obs[:, -1, :]
            if goal.ndim == 3 and key != 'pixels':
                goal = goal[:, -1, :]

            obs_dict[key] = obs
            goal_dict[key] = goal

        z = self.encode(obs_dict, goal_dict)  # Shape: (B, latent_dim)

        B, N, H, A = action_candidates.shape

        # Expand z for every candidate trajectory
        z = z.unsqueeze(1).repeat(1, N, 1).view(B * N, -1)
        actions = action_candidates.view(B * N, H, A)

        G, discount = 0, 1.0
        c = self.cfg.wm.get('uncertainty_penalty', 0.5)

        termination = torch.zeros(
            B * N, 1, dtype=torch.float32, device=z.device
        )

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
        q_values = torch.stack(
            [two_hot_inv(logits, self.cfg) for logits in q_logits]
        )

        q_mean = q_values.mean(dim=0)
        q_std = q_values.std(dim=0)

        penalty = c * q_mean.abs() * q_std
        conservative_q = q_mean - penalty

        total_return = G + discount * (1 - termination) * conservative_q

        cost = -total_return.view(B, N)
        return cost
