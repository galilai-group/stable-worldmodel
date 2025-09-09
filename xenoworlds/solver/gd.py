import torch

import numpy as np

from loguru import logger as logging
from torch.nn import functional as F
from einops import rearrange

from .base import BaseSolver


class GDSolver(BaseSolver):
    """Gradient Descent Solver"""

    def __init__(
        self,
        world_model,
        horizon: int,
        n_steps: int,
        action_space,
        init_action=None,
        criterion=F.mse_loss,
        action_noise=0.003,
        mpc_n_actions=1,
    ):
        super().__init__(world_model)
        self.horizon = horizon
        self.n_steps = n_steps
        self.criterion = criterion
        self.device = world_model.device
        self.action_noise = action_noise
        self.mpc_n_actions = mpc_n_actions

        # starting point for the optimization
        self.init_action(action_space, init_action)

    def init_action(self, action_space, initial_action=None):
        """Initialize the action tensor for the solver.
        set sel.init - initial action sequences (n_envs, horizon, action_dim)
        """
        action_dim = self.world_model.action_dim
        n_envs = action_space.shape[0]
        actions = initial_action
        # -- no actions provided, sample
        if actions is None:
            # (n_envs, 1, action_dim)
            n_envs = action_space.shape[0]
            actions = torch.zeros((n_envs, 0, action_dim))

        # fill remaining action with random sample!
        remaining = self.horizon - actions.shape[1]

        if remaining > 0:
            new_actions = torch.zeros(n_envs, remaining, action_dim)
            new_actions = rearrange(
                new_actions, "... (f d) -> ... f d", f=self.world_model.frameskip
            )
            new_actions = self.world_model.normalize_actions(new_actions)
            new_actions = rearrange(new_actions, "... f d -> ... (f d)")
            actions = torch.cat([actions, new_actions], dim=1)

        actions = actions.to(self.device)

        # -- reset the initial action
        if hasattr(self, "init"):
            self.init.copy_(actions)
        else:
            self.register_parameter(
                "init",
                torch.nn.Parameter(actions),
            )

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

    def solve(
        self, obs_0: dict, action_space, goals: dict, init_action=None
    ) -> torch.Tensor:
        """Solve the planning optimization problem using gradient descent."""

        # -- reinitialize the initial action and optim
        with torch.no_grad():
            self.init_action(action_space, init_action)
        optim = torch.optim.SGD([self.init], lr=1.0)

        # -- encode goal states
        z_goals = self.world_model.encode_obs(goals)
        z_goals = {k: v.detach() for k, v in z_goals.items()}

        # -- solve optimization
        for step_i in range(self.n_steps):
            z_obs_i, z = self.world_model.rollout(obs_0, self.init)
            loss = self.cost_function(z_obs_i, z_goals).sum()
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

            self.init.data += torch.randn_like(self.init) * self.action_noise

        # TODO add logger here
        # TODO break solving if finished
        logging.info(f"Final gradient solver loss: {loss.item()}")

        # ! ----- [DEBUG] Remove later
        self.DEBUG_save_imagined_trajectories(obs_0, self.init, video=True)

        # -- return the actions
        mpc_actions = self.init.detach().cpu()

        # # squeeze time dimension if needed
        # if mpc_actions.ndim == 3:
        #     mpc_actions = mpc_actions.squeeze(1)

        mpc_actions = rearrange(
            mpc_actions,
            "b t (f d) -> b (t f) d",
            f=self.horizon,
        )

        mpc_actions = self.world_model.denormalize_actions(mpc_actions)

        # # keep first action
        # mpc_actions = mpc_actions[:, 0]

        return mpc_actions

    def DEBUG_save_imagined_trajectories(self, obs_0, actions, video=False):
        import imageio

        if self.world_model is None:
            raise ValueError("World model is None, cannot debug imagined trajectories")

        # -- simulate the world under actions sequence
        z_obs_i, z = self.world_model.rollout(obs_0, actions)

        # -- decode obs
        decoded_obs, _ = self.world_model.decode_obs(z_obs_i)

        print("❤️ Decoded obs shape: ", decoded_obs["pixels"].shape)

        for traj in decoded_obs["pixels"]:
            frames = []
            for idx, frame in enumerate(traj):
                frame = rearrange(frame, "c w1 w2 -> w1 w2 c")
                frame = rearrange(frame, "w1 w2 c -> (w1) w2 c")
                frame = frame.detach().cpu().numpy()
                frames.append(frame)

            if video:
                video_writer = imageio.get_writer("videos/imagined.mp4", fps=12)

            for idx, frame in enumerate(frames):
                frame = frame * 2 - 1 if frame.min() >= 0 else frame
                frame = (((np.clip(frame, -1, 1) + 1) / 2) * 255).astype(np.uint8)
                if video:
                    video_writer.append_data(frame)
                else:
                    # save the image
                    imageio.imwrite(f"videos/imagined_{idx}.png", frame)

            if video:
                video_writer.close()
