import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging

from .solver import Costable


class CEMSolver:
    """Cross Entropy Method Solver.

    adapted from https://github.com/gaoyuezhou/dino_wm/blob/main/planning/cem.py
    """

    def __init__(
        self,
        model: Costable,
        num_samples,
        var_scale,
        n_steps,
        topk,
        device="cpu",
    ):
        self.model = model
        self.var_scale = var_scale
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
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
                        v_traj = v_traj.unsqueeze(0).repeat(self.num_samples)
                    elif isinstance(v, np.ndarray):
                        v_traj = np.repeat(v_traj[None, ...], self.num_samples, axis=0)

                    expanded_infos[k] = v_traj

                # sample action sequences candidation from normal distrib
                candidates = torch.randn(self.num_samples, self.horizon, self.action_dim, device=self.device)

                # scale and shift
                candidates = candidates * var[traj] + mean[traj]

                # make the first action seq being mean
                candidates[0] = mean[traj]

                # evaluate the candidates
                cost = self.model.get_cost(expanded_infos, candidates)

                assert type(cost) is torch.Tensor, f"Expected cost to be a torch.Tensor, got {type(cost)}"
                assert cost.ndim == 1 and len(cost) == self.num_samples, (
                    f"Expected cost to be of shape num_samples ({self.num_samples},), got {cost.shape}"
                )

                # -- get the elites
                topk_idx = torch.argsort(cost)[: self.topk]
                topk_candidates = candidates[topk_idx]
                costs.append(cost[topk_idx[0]].item())

                # -- update the mean and var
                mean[traj] = topk_candidates.mean(dim=0)
                var[traj] = topk_candidates.std(dim=0)

            outputs["costs"].append(np.mean(costs))
            outputs["mean"].append(mean.detach().cpu().clone())
            outputs["var"].append(var.detach().cpu().clone())

            # PRINT COST
            print(f"  CEM step {step + 1}/{self.n_steps}, cost: {outputs['costs'][-1]:.4f}")

        outputs["actions"] = mean.detach().cpu()

        return outputs


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
