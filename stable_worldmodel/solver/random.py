import numpy as np
import torch
from .base import BaseSolver

class RandomSolver(BaseSolver):
    """Random Solver"""

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        frameskip: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.horizon = horizon
        self.action_dim = action_dim
        self.frameskip = frameskip

    def solve(
        self, z_obs0, z_goal, action_space, predict_fn=None, init_action=None
    ) -> torch.Tensor:
        """Solve the planning optimization problem using gradient descent."""

        action_dim = self.action_dim
        n_envs = action_space.shape[0]
        actions = init_action

        # -- no actions provided, sample
        if actions is None:
            n_envs = action_space.shape[0]
            actions = torch.zeros((n_envs, 0, action_dim))

        # fill remaining actions with random sample
        remaining = self.horizon - actions.shape[1]

        if remaining > 0:
            total_sequence = remaining * self.frameskip
            action_sequence = np.stack(
                [action_space.sample() for _ in range(total_sequence)], axis=1
            )
            new_action = torch.from_numpy(action_sequence)
            # new_action = action_sequence.view(
            #             -1, remaining, self.frameskip, action_dim
            #         )
    
            actions = torch.cat([actions, new_action], dim=1)

        return actions



      
