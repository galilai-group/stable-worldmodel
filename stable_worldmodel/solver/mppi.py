import torch
import gymnasium as gym
from .solver import BasePlanner


"""
Adapted from the official implementation of TDMPC2: https://github.com/nicklashansen/tdmpc2/
"""


class MPPI(BasePlanner):
    def __init__(
        self,
        world_model: torch.nn.Module,
        action_space: gym.spaces.Box,
        horizon: int = 3,
        iterations: int = 6,
        num_samples: int = 512,
        num_elites: int = 64,
        min_std: float = 0.05,
        max_std: float = 2.0,
        temperature: float = 0.5,
        compile: bool = True,
    ):
        super().__init__(world_model, action_space)

        self.horizon = horizon
        self.iterations = iterations
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.min_std = min_std
        self.max_std = max_std
        self.temperature = temperature

        self.compile = compile

    @property
    def plan(self):
        if self.compile:
            plan = torch.compile(self._plan, mode="reduce-overhead")
        else:
            plan = self._plan
        self._plan_val = plan
        return self._plan_val

    @torch.no_grad()
    def _plan(self, obs, goal, t0):
        # Initialize state and parameters
        z = self.world_model.encode(
            obs
        )  # NOTE: the encode method should be the identity if the world model is not latent
        z_g = self.world_model.encode(goal)

        z = z.repeat(self.num_samples, 1)
        z_g = z_g.repeat(self.num_samples, 1)
        mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        std = torch.full(
            (self.horizon, self.action_dim),
            self.max_std,
            dtype=torch.float,
            device=self.device,
        )
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(
            self.horizon, self.num_samples, self.action_dim, device=self.device
        )

        # Iterate MPPI
        for _ in range(self.iterations):
            # Sample actions
            r = torch.randn(
                self.horizon, self.num_samples, self.action_dim, device=std.device
            )
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions = actions_sample

            # Compute elite actions
            value = self._evaluate_action_sequence(z, actions, z_g).nan_to_num(0)
            elite_idxs = torch.topk(value.squeeze(1), self.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0).values
            score = torch.exp(self.temperature * (elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (
                score.sum(0) + 1e-9
            )
            std = (
                (score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(
                    dim=1
                )
                / (score.sum(0) + 1e-9)
            ).sqrt()
            std = std.clamp(self.min_std, self.max_std)

        # Select action
        rand_idx = gumbel_softmax_sample(score.squeeze(1))
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
        a, std = actions[0], std[0]
        self._prev_mean.copy_(mean)
        return a.clamp(-1, 1)

    @torch.no_grad()
    def _evaluate_action_sequence(self, z, actions, z_g):
        """Estimate value of a trajectory starting at latent state z and executing given actions"""
        G = 0
        for t in range(self.cfg.horizon):
            z = self.world_model.next(z, actions[t])
            G = G + self.world_model.reward(z, z_g)
        return G


def gumbel_softmax_sample(p, temperature=1.0, dim=0):
    """Sample from the Gumbel-Softmax distribution."""
    logits = p.log()
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    return y_soft.argmax(-1)
