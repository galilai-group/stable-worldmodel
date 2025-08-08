"""Planning optimization solvers implementation."""

import torch
import nevergrad as ng
import numpy as np

from loguru import logger as logging
from torch.nn import functional as F

## -- Solver (For Planning)
### GradientSolver
### CEMSolver (navergrad, evotorch, mppi) MPC


class BaseSolver(torch.nn.Module):
    """Base class for planning solvers"""

    # the idea for solver is to implement different methods for solving planning optimization problems
    def __init__(self, world_model):
        super().__init__()

        # disable gradients for the world model
        self.world_model = world_model
        self.world_model.requires_grad_(False)

    def __call__(
        self, states: torch.Tensor, action_space, goals: torch.Tensor
    ) -> torch.Tensor:
        return self.solve(states, action_space, goals)

    def solve(
        self, states: torch.Tensor, action_space, goals: torch.Tensor
    ) -> torch.Tensor:
        """Solve the planning optimization problem given states, action space, and goals."""
        raise NotImplementedError("Solver must implement the solve method.")


class GDSolver(BaseSolver):
    """Gradient Descent Solver"""

    def __init__(
        self,
        world_model,
        n_steps: int,
        action_space,
        init_action=None,
        criterion=F.mse_loss,
    ):
        super().__init__(world_model)
        self.n_steps = n_steps
        self.criterion = criterion

        # starting point for the optimization
        init_action = init_action or torch.from_numpy(action_space.sample()).float()
        self.register_parameter("init", torch.nn.Parameter(init_action))

    def solve(
        self, states: torch.Tensor, action_space, goals: torch.Tensor, init_action=None
    ) -> torch.Tensor:
        """Solve the planning optimization problem using gradient descent."""

        # reinitialize the initial action if provided, otherwise sample from the action space
        with torch.no_grad():
            init_action = init_action or torch.from_numpy(action_space.sample()).float()
            self.init.copy_(init_action)

        # set up the optimizer
        # todo support any optimizer? and lr? maybe just provide a partial optim and just give the params
        optim = torch.optim.SGD([self.init], lr=1.0)

        for _ in range(self.n_steps):
            preds = self.world_model(states, self.init)
            loss = self.criterion(preds, goals, reduction="none").mean(1).sum()
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

        logging.info(f"Final gradient solver loss: {loss.item()}")
        return self.init.detach()


# TODO implement dino-wm cem solver
class CEMSolver(BaseSolver):
    """Cross Entropy Method Solver"""

    def __init__(self, world_model, n_steps: int, action_space, horizon: int):
        super().__init__(world_model)
        self.n_steps = n_steps
        self.action_space = action_space
        self.horizon = horizon


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
