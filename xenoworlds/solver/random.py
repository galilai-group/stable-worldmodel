import torch
import numpy as np
from einops import rearrange
from .base import BaseSolver


class RandomSolver(BaseSolver):
    """Random Solver"""

    def __init__(
        self,
        world_model,
        horizon: int,
    ):
        super().__init__(world_model)
        self.horizon = horizon
        self.device = world_model.device

    def sample_action_sequence(
        self,
        action_space,
        remaining_horizon,
    ):
        """Sample a random action sequence from the env action space."""
        env_action_dim = action_space.shape[-1]
        total_sequence = remaining_horizon * self.world_model.frameskip
        action_sequence = np.stack(
            [action_space.sample() for _ in range(total_sequence)], axis=1
        )
        action_sequence = torch.from_numpy(action_sequence)
        return action_sequence.view(
            -1, remaining_horizon, self.world_model.frameskip, env_action_dim
        )

    def solve(
        self, obs_0: dict, action_space, goals: dict, init_action=None
    ) -> torch.Tensor:
        """Solve the planning optimization problem using gradient descent."""

        action_dim = self.world_model.action_dim
        n_envs = action_space.shape[0]
        actions = init_action

        # -- no actions provided, sample
        if actions is None:
            n_envs = action_space.shape[0]
            actions = torch.zeros((n_envs, 0, action_dim))

        # fill remaining actions with random sample
        remaining = self.horizon - actions.shape[1]

        if remaining > 0:
            new_actions = self.sample_action_sequence(action_space, remaining)
            new_actions = self.world_model.normalize_actions(new_actions)
            new_actions = rearrange(
                new_actions, "... f d -> ... (f d)", f=self.world_model.frameskip
            )
            actions = torch.cat([actions, new_actions], dim=1)

        debug_actions = actions.clone().to(self.device)  # ! remove for debug

        actions = rearrange(
            actions,
            "b t (f d) -> b (t f) d",
            f=self.horizon,
        )

        actions = self.world_model.denormalize_actions(actions)

        # ! ===== [DEBUG] Remove later
        self.DEBUG_save_imagined_trajectories(obs_0, debug_actions, video=True)
        # ! =====
        return actions

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
