import time

import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging

from .solver import Costable


class GDSolver(torch.nn.Module):
    """Gradient Descent Solver."""

    def __init__(
        self,
        model: Costable,
        n_steps: int,
        action_noise=0.0,
        num_samples=1,
        device="cpu",
        seed: int = 1234,
    ):
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.num_samples = num_samples
        self.action_noise = action_noise
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)

        self._configured = False
        self._n_envs = None
        self._action_dim = None
        self._config = None

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

    def init_action(self, actions=None):
        """Initialize the action tensor for the solver.

        set self.init - initial action sequences (n_envs, horizon, action_dim)
        """
        if actions is None:
            actions = torch.zeros((self._n_envs, 0, self.action_dim))

        # fill remaining action
        remaining = self.horizon - actions.shape[1]

        if remaining > 0:
            new_actions = torch.zeros(self._n_envs, remaining, self.action_dim)
            actions = torch.cat([actions, new_actions], dim=1)

        actions = actions.unsqueeze(1).repeat_interleave(self.num_samples, dim=1)  # add sample dim
        actions[:, 1:] += (
            torch.randn(actions[:, 1:].shape, generator=self.torch_gen) * self.action_noise
        )  # add small noise to all samples except the first one
        actions = actions.to(self.device)

        # reset actions
        if hasattr(self, "init"):
            self.init.copy_(actions)
        else:
            self.register_parameter("init", torch.nn.Parameter(actions))

    def solve(self, info_dict, init_action=None) -> torch.Tensor:
        """Solve the planning optimization problem using gradient descent."""
        start_time = time.time()
        outputs = {
            "cost": [],
            "trajectory": [],
        }

        with torch.no_grad():
            self.init_action(init_action)

        optim = torch.optim.SGD([self.init], lr=1.0)

        expanded_infos = {}
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                v = v.unsqueeze(1)  # add sample dim
                v = v.expand(self.n_envs, self.num_samples, *v.shape[2:])
            elif isinstance(v, np.ndarray):
                v = np.repeat(v[:, None, ...], self.num_samples, axis=1)
            expanded_infos[k] = v

        # perform gradient descent
        for _ in range(self.n_steps):
            current_info = expanded_infos.copy()
            costs = self.model.get_cost(current_info, self.init)

            assert isinstance(costs, torch.Tensor), f"Got {type(costs)} cost, expect torch.Tensor"
            assert costs.ndim == 2 and costs.shape[0] == self.n_envs and costs.shape[1] == self.num_samples, (
                f"Cost should be of shape ({self.n_envs}, {self.num_samples}), got {costs.shape}"
            )
            assert costs.requires_grad, "Cost must requires_grad for GD solver."

            cost = costs.sum()  # independent cost for each env and each sample
            cost.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

            if self.action_noise > 0:
                self.init.data += torch.randn(self.init.shape, generator=self.torch_gen) * self.action_noise

            outputs["cost"].append(cost.item())
            outputs["trajectory"].extend([self.init.detach().cpu().clone()])

            print(f" GD step {_ + 1}/{self.n_steps}, cost: {outputs['cost'][-1]:.4f}")

        # TODO break solving if finished self.eval? done break

        # get the best actions to return
        top_idx = torch.argsort(costs, dim=1)[:, 0]
        batch_indices = torch.arange(self.init.size(0))
        top_actions = self.init[batch_indices, top_idx]
        outputs["actions"] = top_actions.detach().cpu()
        outputs["solve_time"] = time.time() - start_time
        print(f"GD solve time: {outputs['solve_time']:.4f} seconds")

        return outputs
