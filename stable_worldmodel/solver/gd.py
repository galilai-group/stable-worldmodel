import torch
from loguru import logger as logging
from .base import BaseSolver

class GDSolver(BaseSolver):
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
        #self.init_action(action_space, init_action)

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

    def solve(
        self, z_obs0, z_goal, action_space, predict_fn=None, init_action=None
    ) -> torch.Tensor:
        """Solve the planning optimization problem using gradient descent."""

        # init action sequence
        with torch.no_grad():
            self.init_action(action_space, init_action)
        optim = torch.optim.SGD([self.init], lr=1.0)

        # perform gradient descent
        for step_i in range(self.n_steps):
            z_preds = predict_fn(z_obs0, self.init)
            loss = self.cost_fn(z_preds, z_goal).sum()
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            self.init.data += torch.randn_like(self.init) * self.action_noise

        # TODO add logger here
        # TODO break solving if finished

        if self.verbose:
            logging.info(f"Final gradient solver loss: {loss.item()}")

        # get the actions to return
        actions = self.init.detach().cpu()

        return actions
