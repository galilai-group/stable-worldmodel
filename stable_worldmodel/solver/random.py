import numpy as np
import torch


class RandomSolver:
    """Random Solver."""

    def __init__(self):
        self._configured = False
        self._action_space = None
        self._n_envs = None
        self._action_dim = None
        self._config = None

    def configure(self, *, action_space, n_envs: int, config) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

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

    def solve(self, info_dict, init_action=None) -> torch.Tensor:
        """Solve the planning optimization problem using gradient descent."""
        outputs = {}
        actions = init_action

        # -- no actions provided, sample
        if actions is None:
            actions = torch.zeros((self.n_envs, 0, self.action_dim))

        # fill remaining actions with random sample
        remaining = self.horizon - actions.shape[1]

        if remaining > 0:
            total_sequence = remaining * self._config.action_block
            action_sequence = np.stack(
                [self._action_space.sample() for _ in range(total_sequence)], axis=1
            )

            new_action = torch.from_numpy(action_sequence).view(
                self.n_envs, remaining, self.action_dim
            )
            actions = torch.cat([actions, new_action], dim=1)

        outputs["actions"] = actions
        return outputs
