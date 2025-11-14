import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging

from .solver import Costable


class CEMSolver:
    """Cross Entropy Method Solver.

    adapted from https://github.com/gaoyuezhou/dino_wm/blob/main/planning/cem.py
    """

    def __init__(self, model: Costable, num_samples, var_scale, n_steps, topk, device="cpu", seed: int = 1234):
        self.model = model
        self.var_scale = var_scale
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)

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

        for traj in range(n_envs):
            costs_per_traj = []

            # expand info for this trajectory once
            expanded_infos = {}
            for k, v in info_dict.items():
                v_traj = v[traj]
                if torch.is_tensor(v):
                    v_traj = v_traj.unsqueeze(0)  # add sample dim
                    v_traj = v_traj.expand(self.num_samples, *v_traj.shape[1:])
                    v_traj = v_traj.unsqueeze(0)  # add traj dim
                elif isinstance(v, np.ndarray):
                    v_traj = np.repeat(v_traj[None, None, ...], self.num_samples, axis=1)
                expanded_infos[k] = v_traj

            # -- optimization loop for this trajectory
            for step in range(self.n_steps):
                # sample action sequences candidation from normal distrib
                candidates = torch.randn(
                    self.num_samples, self.horizon, self.action_dim, generator=self.torch_gen, device=self.device
                )
                # scale and shift
                candidates = candidates * var[traj] + mean[traj]
                # make the first action seq being mean
                candidates[0] = mean[traj]
                candidates = candidates.unsqueeze(0)  # add traj dim

                # evaluate the candidates
                cost = self.model.get_cost(expanded_infos, candidates)[0]
                assert isinstance(cost, torch.Tensor), f"Expected cost to be a torch.Tensor, got {type(cost)}"
                assert cost.ndim == 1 and cost.shape[0] == self.num_samples, (
                    f"Expected cost to be of shape (num_samples), got {cost.shape}"
                )

                # select topk
                topk_idx = torch.argsort(cost)[: self.topk]
                topk_candidates = candidates[0, topk_idx]
                best_cost = cost[topk_idx[0]].item()
                costs_per_traj.append(best_cost)

                # update mean and variance
                mean[traj] = topk_candidates.mean(dim=0)
                var[traj] = topk_candidates.std(dim=0)

                print(f"  Traj {traj} | Step {step + 1}/{self.n_steps} | cost: {best_cost:.4f}")

            # store trajectory results
            outputs["costs"].append(np.mean(costs_per_traj))
            outputs["mean"].append(mean[traj].detach().cpu().clone())
            outputs["var"].append(var[traj].detach().cpu().clone())

        outputs["actions"] = mean.detach().cpu()
        return outputs
