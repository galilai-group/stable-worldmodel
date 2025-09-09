import torch


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
