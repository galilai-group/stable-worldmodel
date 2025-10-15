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
        device="cpu",
    ):
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.action_noise = action_noise
        self.device = device

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

    def set_seed(self, seed: int) -> None:
        """Set random seed for deterministic behavior.

        Args:
            seed: Random seed to use for numpy and torch
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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

        actions = actions.to(self.device)

        # reset actions
        if hasattr(self, "init"):
            self.init.copy_(actions)
        else:
            self.register_parameter("init", torch.nn.Parameter(actions))

    def solve(self, info_dict, init_action=None) -> torch.Tensor:
        """Solve the planning optimization problem using gradient descent."""
        outputs = {
            "cost": [],
            "trajectory": [],
        }

        # Set model to eval mode to ensure deterministic behavior
        self.model.eval()

        with torch.no_grad():
            self.init_action(init_action)

        optim = torch.optim.SGD([self.init], lr=1.0)

        # perform gradient descent
        for _ in range(self.n_steps):
            # copy info dict to avoid in-place modification
            cost = self.model.get_cost(dict(info_dict), self.init)

            assert type(cost) is torch.Tensor, f"Got {type(cost)} cost, expect torch.Tensor"
            assert cost.ndim == 1 and len(cost) == self.n_envs, f"Cost should be of shape (n_envs,), got {cost.shape}"
            assert cost.requires_grad, "Cost must requires_grad for GD solver."

            cost = cost.sum()  # independent cost for each env
            cost.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

            if self.action_noise > 0:
                self.init.data += torch.randn_like(self.init) * self.action_noise

            outputs["cost"].append(cost.item())
            outputs["trajectory"].extend([self.init.detach().cpu().clone()])

            print(f" GD step {_ + 1}/{self.n_steps}, cost: {outputs['cost'][-1]:.4f}")

        # TODO break solving if finished self.eval? done break

        # get the actions to return
        outputs["actions"] = self.init.detach().cpu()

        return outputs
