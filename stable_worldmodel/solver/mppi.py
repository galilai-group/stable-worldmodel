import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging

from .solver import Costable


class MPPISolver:
    """Model Predictive Path Integral Solver.

    proposed in https://arxiv.org/abs/1509.01149
    algorithm from: https://acdslab.github.io/mppi-generic-website/docs/mppi.html

    Note:
        The original MPPI compute the cost as a summation of costs along the trajectory.
        Here, we use the final cost only, which should be updated in future updates.
    """

    def __init__(
        self,
        model: Costable,
        num_samples,
        num_elites,
        var_scale,
        n_steps,
        use_elites=True,
        temperature=0.5,
        device="cpu",
    ):
        self.model = model
        self.var_scale = var_scale
        self.use_elites = use_elites
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.temperature = temperature
        self.n_steps = n_steps
        self.device = device

    def configure(self, *, action_space, n_envs: int, config) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

        # warning if action space is discrete
        if not isinstance(action_space, Box):
            logging.warning(f"Action space is discrete, got {type(action_space)}. GDSolver may not work as expected.")

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.solve(*args, **kwargs)

    def init_action_distrib(self, actions=None):
        """Initialize the action distribution params (mu, sigma) given the initial condition.

        Args:
            actions (n_envs, T, action_dim): initial actions, T <= horizon
        """
        var = self.var_scale * torch.ones([self.n_envs, self.horizon, self.action_dim])

        mean = torch.zeros([self.n_envs, 0, self.action_dim]) if actions is None else actions

        # -- fill remaining actions with random sample
        remaining = self.horizon - mean.shape[1]

        if remaining > 0:
            device = mean.device
            new_mean = torch.zeros([self.n_envs, remaining, self.action_dim])
            mean = torch.cat([mean, new_mean], dim=1).to(device)

        return mean, var

    def compute_trajectory_weights(self, costs: torch.Tensor) -> torch.Tensor:
        """Compute trajectory weights from costs using softmin with temperature.

        Args:
            costs (num_samples,): Tensor of trajectory costs.

        Returns:
            Tensor of trajectory weights.
        """
        stable_costs = costs - costs.min()
        weights = torch.softmax(-stable_costs / self.temperature, dim=0)
        return weights

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        outputs = {
            "costs": [],
            "mean": [],
            "var": [],
        }

        # -- initialize the action distribution
        mean, var = self.init_action_distrib(init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)
        n_envs = mean.shape[0]

        # -- optimization loop
        for step in range(self.n_steps):
            costs = []

            # TODO: could flatten the batch dimension and process all samples together
            # rem: need many memory and split before computing top k

            for traj in range(n_envs):
                expanded_infos = {}

                for k, v in info_dict.items():
                    v_traj = v[traj]
                    if torch.is_tensor(v):
                        v_traj = v_traj.unsqueeze(0).repeat_interleave(self.num_samples, dim=0)
                    elif isinstance(v, np.ndarray):
                        v_traj = np.repeat(v_traj[None, ...], self.num_samples, axis=0)

                    expanded_infos[k] = v_traj

                # sample noisy actions sequences
                candidates = torch.randn(self.num_samples, self.horizon, self.action_dim, device=self.device)
                candidates = candidates * var[traj] + mean[traj]

                # make the first action seq being mean
                candidates[0] = mean[traj]

                # get the costs
                cost = self.model.get_cost(expanded_infos, candidates)

                assert type(cost) is torch.Tensor, f"Expected cost to be a torch.Tensor, got {type(cost)}"
                assert cost.ndim == 1 and len(cost) == self.num_samples, (
                    f"Expected cost to be of shape num_samples ({self.num_samples},), got {cost.shape}"
                )

                if self.use_elites:
                    elite_idx = torch.argsort(cost)[: self.num_elites]
                    candidates = candidates[elite_idx]
                    cost = cost[elite_idx]

                costs.append(cost.max().item())

                # update nominal sequence with weighted candidates
                weights = self.compute_trajectory_weights(cost)
                mean[traj] = (weights * candidates).sum(dim=0)

            outputs["costs"].append(np.mean(costs))
            outputs["nominal_action"].append(mean.detach().cpu().clone())

            print(f"MMPI step {step + 1}/{self.n_steps}, cost: {outputs['costs'][-1]:.4f}")

        outputs["actions"] = mean.detach().cpu()

        return outputs
