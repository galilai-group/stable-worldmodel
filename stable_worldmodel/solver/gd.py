import torch
from loguru import logger as logging
from .solver import Solver


class GDSolver(Solver):
    """Gradient Descent Solver"""

    def __init__(
        self,
        horizon: int,
        n_steps: int,
        action_dim: int,
        action_noise=0.003,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.horizon = horizon
        self.n_steps = n_steps
        self.action_noise = action_noise
        self.action_dim = action_dim

        # starting point for the optimization
        # self.init_action(action_space, init_action)

    def init_action(self, action_space, initial_action=None):
        """Initialize the action tensor for the solver.
        set sel.init - initial action sequences (n_envs, horizon, action_dim)
        """

        n_envs = action_space.shape[0]
        actions = initial_action

        if actions is None:
            # (n_envs, 1, action_dim)
            n_envs = action_space.shape[0]
            actions = torch.zeros((n_envs, 0, self.action_dim))

        # fill remaining action
        remaining = self.horizon - actions.shape[1]

        if remaining > 0:
            new_actions = torch.zeros(n_envs, remaining, self.action_dim)
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

        # init action sequence
        with torch.no_grad():
            self.init_action(..., init_action)  # TODO need to find solution for that
        optim = torch.optim.SGD([self.init], lr=1.0)

        # perform gradient descent
        for _ in range(self.n_steps):
            cost = self.model.get_cost(info_dict, self.init)

            assert type(cost) is torch.Tensor, (
                f"Got {type(cost)} cost, expect torch.Tensor"
            )
            assert cost.ndim == 0, f"Cost should be a scalar, got shape {cost.shape}"
            assert cost.requires_grad, "Cost must requires_grad for GD solver."

            cost.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            self.init.data += torch.randn_like(self.init) * self.action_noise

            outputs["cost"].append(cost.item())
            outputs["trajectory"].extend([self.init.detach().cpu().clone()])

        # TODO add logger here
        # TODO break solving if finished

        # get the actions to return
        outputs["actions"] = self.init.detach().cpu()

        return outputs
