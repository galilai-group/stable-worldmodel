from copy import deepcopy

import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging

from .solver import Costable


class NevergradSolver:
    """Nevergrad Solver.

    supporting https://github.com/facebookresearch/nevergrad

    Attention:
        - CPU based optimizer (no GPU support)
        - It's your duty to ensure num_workers == n_envs for parallelization
    """

    def __init__(
        self,
        model: Costable,
        optimizer,
        n_steps: int,
        device="cpu",
    ):
        self.model = model
        self.optimizer = optimizer
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
            logging.warning(
                f"Action space is discrete, got {type(action_space)}. NevergradSolver may not work as expected."
            )

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

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        outputs = {
            "costs": [],
        }

        optimizer = deepcopy(self.optimizer)
        n_envs = info_dict[list(info_dict.keys())[0]].shape[0]

        for step in range(self.n_steps):
            costs = []

            candidates = [optimizer.ask() for _ in range(n_envs)]
            costs = [
                self.model.get_cost(info_dict, torch.tensor(candidates[traj].value)).item() for traj in range(n_envs)
            ]

            if not all(torch.is_tensor(c) for c in costs):
                raise ValueError("Costs must be torch tensors")

            outputs["costs"].append(np.mean(costs))

            for c, cost in zip(candidates, costs):
                optimizer.tell(c, cost)

            print(f"  Nevergrad step {step + 1}/{self.n_steps}, cost: {outputs['costs'][-1]:.4f}")

        best_action_sequence = optimizer.recommendation().value
        outputs["actions"] = torch.from_numpy(best_action_sequence)

        return outputs
