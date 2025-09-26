import torch
from typing import Protocol


class Costable(Protocol):
    """Protocol for world model cost functions."""

    def get_cost(info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Compute cost for given action candidates based on info dictionary."""
        ...


class BaseSolver:
    """Base class for planning solvers."""

    # the idea for solver is to implement different methods for solving planning optimization problems
    def __init__(self, model: Costable, verbose=True, device="cpu"):
        self.model = model
        self.verbose = verbose
        self.device = device

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.solve(*args, **kwargs)

    def solve(self, info_dict, init_action=None) -> torch.Tensor:
        """Solve the planning optimization problem given states, action space, and goals."""
        raise NotImplementedError("Solver must implement the solve method.")
