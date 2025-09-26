import torch
import nevergrad as ng
import numpy as np
from .solver import Solver
from einops import rearrange, repeat
from torch.nn import functional as F


class CEMSolver(Solver):
    """Cross Entropy Method Solver

    adapted from https://github.com/gaoyuezhou/dino_wm/blob/main/planning/cem.py
    """

    def __init__(
        self,
        num_envs,
        horizon,
        action_dim,
        num_samples,
        var_scale,
        opt_steps,
        topk,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_envs = num_envs
        self.horizon = horizon
        self.var_scale = var_scale
        self.action_dim = action_dim
        self.num_samples = num_samples
        self.opt_steps = opt_steps
        self.topk = topk

    def init_action_distrib(self, actions=None):
        """Initialize the action distribution params (mu, sigma) given the initial condition.
        Args:
            actions (n_envs, T, action_dim): initial actions, T <= horizon
        rem: mean, var could be based on obs_0 but right now just used to extract n_envs
        """

        # ! should really note somewhere or make clear that action_dim is env_action_dim * frameskip
        var = self.var_scale * torch.ones(
            [self.num_envs, self.horizon, self.action_dim]
        )
        mean = (
            torch.zeros([self.num_envs, 0, self.action_dim])
            if actions is None
            else actions
        )

        # -- fill remaining actions with random sample
        remaining = self.horizon - mean.shape[1]

        if remaining > 0:
            device = mean.device
            new_mean = torch.zeros([self.num_envs, remaining, self.action_dim])
            mean = torch.cat([mean, new_mean], dim=1).to(device)

        return mean, var

    def solve(self, info_dict, init_action=None):
        # -- initialize the action distribution
        mean, var = self.init_action_distrib(init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)

        n_envs = mean.shape[0]

        # -- optimization loop
        for step in range(self.opt_steps):
            losses = []

            for traj in range(n_envs):
                env_info = {}

                # # duplicate the current observation for num_samples
                # cur_trans_obs_0 = {
                #     key: repeat(
                #         arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                #     )
                #     for key, arr in z_obs0.items()
                # }

                # # duplicate the current goal embedding for num_samples
                # cur_z_obs_g = {
                #     key: repeat(
                #         arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                #     )
                #     for key, arr in z_goal.items()
                # }

                # sample action sequences candidation from normal distrib
                candidates = torch.randn(
                    self.num_samples, self.horizon, self.action_dim, device=self.device
                )

                # scale and shift
                candidates = candidates * var[traj] + mean[traj]

                # make the first action seq being mean
                candidates[0] = mean[traj]

                # evaluate the candidates
                cost = self.model.get_cost(info_dict, candidates)

                assert type(cost) is torch.Tensor, (
                    f"Expected cost to be a torch.Tensor, got {type(cost)}"
                )
                assert cost.ndim == 0, (
                    f"Expected scalar tensor for cost, got shape {cost.shape}"
                )

                # -- get the elites
                topk_idx = torch.argsort(cost)[: self.topk]
                topk_candidates = candidates[topk_idx]
                losses.append(cost[topk_idx[0]].item())

                # -- update the mean and var
                mean[traj] = topk_candidates.mean(dim=0)
                var[traj] = topk_candidates.std(dim=0)

            if self.verbose:
                print(f"Losses at step {step}: {np.mean(losses)}")

        actions = mean.detach().cpu()

        return actions


# class CEMNevergrad(BaseSolver):
#     def __init__(
#         self,
#         world_model: torch.nn.Module,
#         n_steps: int,
#         action_space,
#         planning_horizon: int,
#     ):
#         super().__init__(world_model)
#         self.n_steps = n_steps
#         self.planning_horizon = planning_horizon
#         init = torch.from_numpy(
#             np.stack([action_space.sample() for _ in range(planning_horizon)], 0)
#         )
#         self.register_parameter("init", torch.nn.Parameter(init))

#     def solve(
#         self, states: torch.Tensor, action_space, goals: torch.Tensor
#     ) -> torch.Tensor:
#         """Solve the planning optimization problem using CEM."""
#         # Define the action space
#         with torch.no_grad():
#             init = torch.from_numpy(
#                 np.stack(
#                     [action_space.sample() for _ in range(self.planning_horizon)], 0
#                 )
#             )
#             self.init.copy_(init)
#         # Initialize the optimizer
#         optimizer = ng.optimizers.CMA(
#             parametrization=ng.p.Array(
#                 shape=self.init.shape,
#                 lower=np.stack(
#                     [action_space.low for _ in range(self.planning_horizon)], 0
#                 ),
#                 upper=np.stack(
#                     [action_space.high for _ in range(self.planning_horizon)], 0
#                 ),
#             ),
#             budget=self.n_steps,
#         )
#         # Run the optimization
#         for _ in range(self.n_steps):
#             candidate = optimizer.ask()
#             actions = torch.from_numpy(candidate.value.astype(np.float32))
#             rewards = self.evaluate_action_sequence(
#                 states, actions, goals
#             )  # todo how does it works? visual l2 distance with goals is enough? what about other metrics e.g SNR?
#             # Negate rewards to minimize
#             optimizer.tell(candidate, [-r for r in rewards])
#         # Get the best action sequence
#         best_action_sequence = optimizer.provide_recommendation().value
#         return torch.from_numpy(best_action_sequence.astype(np.float32))

#     def evaluate_action_sequence(self, states, actions, goals):
#         with torch.inference_mode():
#             preds = self.world_model(states, actions.unbind(0))
#             rewards = (preds - goals).square().mean(1)
#             return rewards
