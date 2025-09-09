import torch
import nevergrad as ng
import numpy as np
from .base import BaseSolver
from einops import rearrange, repeat
from torch.nn import functional as F


class CEMSolver(BaseSolver):
    """Cross Entropy Method Solver

    adapted from https://github.com/gaoyuezhou/dino_wm/blob/main/planning/cem.py
    """

    def __init__(
        self,
        world_model,
        horizon,
        action_dim,
        num_samples,
        var_scale,
        opt_steps,
        topk,
        criterion=F.mse_loss,
        device="cpu",
    ):
        super().__init__(world_model)
        self.horizon = horizon
        self.var_scale = var_scale
        self.action_dim = action_dim
        self.num_samples = num_samples
        self.opt_steps = opt_steps
        self.topk = topk

        self.criterion = criterion

        self.device = device

    def init_action_distrib(self, obs_0, actions=None):
        """Initialize the action distribution params (mu, sigma) given the initial condition.
        Args:
            actions (n_envs, T, action_dim): initial actions, T <= horizon
        rem: mean, var could be based on obs_0 but right now just used to extract n_envs
        """

        n_envs = obs_0["proprio"].shape[0]
        # ! should really note somewhere or make clear that action_dim is env_action_dim * frameskip
        var = self.var_scale * torch.ones([n_envs, self.horizon, self.action_dim])
        mean = torch.zeros([n_envs, 0, self.action_dim]) if actions is None else actions

        # -- fill remaining actions with random sample
        remaining = self.horizon - mean.shape[1]

        if remaining > 0:
            device = mean.device
            new_mean = torch.zeros([n_envs, remaining, self.action_dim])
            mean = torch.cat([mean, new_mean], dim=1).to(device)

        return mean, var

    def cost_function(self, z_pred, z_target, proprio_scale=1.0):
        """Compute the cost function for the optimization.
        Args:
            z_pred: Predicted latent observations dict .
            z_target: Target latent observations dict (goals).
        Returns:
            torch.Tensor: Computed cost. (B,)
        """

        z_pixel = z_pred["pixels"]
        z_proprio = z_pred["proprio"]

        z_goal_pixel = z_target["pixels"]
        z_goal_proprio = z_target["proprio"]

        # cost for the last prediction
        loss_pixel = self.criterion(
            z_pixel[:, -1:], z_goal_pixel, reduction="none"
        ).mean(dim=tuple(range(1, z_pixel.ndim)))

        # cost for the last proprioceptive observation
        loss_proprio = self.criterion(
            z_proprio[:, -1:], z_goal_proprio, reduction="none"
        ).mean(dim=tuple(range(1, z_proprio.ndim)))

        # Combine the losses
        loss = loss_pixel + proprio_scale * loss_proprio
        return loss

    def solve(self, obs_0: dict, action_space, goals: dict, init_action=None):
        # -- encode the goals
        z_goals = self.world_model.encode_obs(goals)
        z_goals = {k: v.detach() for k, v in z_goals.items()}

        # -- initialize the action distribution
        mean, var = self.init_action_distrib(obs_0, init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)

        n_envs = mean.shape[0]

        # -- optimization loop
        for step in range(self.opt_steps):
            losses = []
            for traj in range(n_envs):
                # duplicate the current observation for num_samples
                cur_trans_obs_0 = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in obs_0.items()
                }

                # duplicate the current goal embedding for num_samples
                cur_z_obs_g = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in z_goals.items()
                }

                # sample action sequences candidation from normal distrib
                candidates = torch.randn(
                    self.num_samples, self.horizon, self.action_dim, device=self.device
                )

                # scale and shift
                candidates = candidates * var[traj] + mean[traj]

                # make the first action seq being mean
                candidates[0] = mean[traj]

                # simulate the action sequences
                with torch.no_grad():
                    z_obs_i, z_i = self.world_model.rollout(cur_trans_obs_0, candidates)

                # compute the loss
                loss = self.cost_function(z_obs_i, cur_z_obs_g, proprio_scale=1.0)

                # -- get the elites
                topk_idx = torch.argsort(loss)[: self.topk]
                topk_candidates = candidates[topk_idx]
                losses.append(loss[topk_idx[0]].item())

                # -- update the mean and var
                mean[traj] = topk_candidates.mean(dim=0)
                var[traj] = topk_candidates.std(dim=0)

            print(f"Lossses at step {step}: {np.mean(losses)}")

        mpc_actions = mean.detach().cpu()

        mpc_actions = rearrange(
            mpc_actions,
            "b t (f d) -> b (t f) d",
            f=self.horizon,
        )

        mpc_actions = self.world_model.denormalize_actions(mpc_actions)

        return mpc_actions


class CEMNevergrad(BaseSolver):
    def __init__(
        self,
        world_model: torch.nn.Module,
        n_steps: int,
        action_space,
        planning_horizon: int,
    ):
        super().__init__(world_model)
        self.n_steps = n_steps
        self.planning_horizon = planning_horizon
        init = torch.from_numpy(
            np.stack([action_space.sample() for _ in range(planning_horizon)], 0)
        )
        self.register_parameter("init", torch.nn.Parameter(init))

    def solve(
        self, states: torch.Tensor, action_space, goals: torch.Tensor
    ) -> torch.Tensor:
        """Solve the planning optimization problem using CEM."""
        # Define the action space
        with torch.no_grad():
            init = torch.from_numpy(
                np.stack(
                    [action_space.sample() for _ in range(self.planning_horizon)], 0
                )
            )
            self.init.copy_(init)
        # Initialize the optimizer
        optimizer = ng.optimizers.CMA(
            parametrization=ng.p.Array(
                shape=self.init.shape,
                lower=np.stack(
                    [action_space.low for _ in range(self.planning_horizon)], 0
                ),
                upper=np.stack(
                    [action_space.high for _ in range(self.planning_horizon)], 0
                ),
            ),
            budget=self.n_steps,
        )
        # Run the optimization
        for _ in range(self.n_steps):
            candidate = optimizer.ask()
            actions = torch.from_numpy(candidate.value.astype(np.float32))
            rewards = self.evaluate_action_sequence(
                states, actions, goals
            )  # todo how does it works? visual l2 distance with goals is enough? what about other metrics e.g SNR?
            # Negate rewards to minimize
            optimizer.tell(candidate, [-r for r in rewards])
        # Get the best action sequence
        best_action_sequence = optimizer.provide_recommendation().value
        return torch.from_numpy(best_action_sequence.astype(np.float32))

    def evaluate_action_sequence(self, states, actions, goals):
        with torch.inference_mode():
            preds = self.world_model(states, actions.unbind(0))
            rewards = (preds - goals).square().mean(1)
            return rewards
