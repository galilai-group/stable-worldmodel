import torch
import gymnasium as gym
from typing import Protocol, runtime_checkable


class Costable(Protocol):
    """Protocol for world model cost functions."""

    def get_cost(info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Compute cost for given action candidates based on info dictionary."""
        ...


@runtime_checkable
class Solver(Protocol):
    """Protocol for planning solvers."""

    def configure(self, *, action_space: gym.Space, n_envs: int, spec) -> None: ...
    @property
    def action_dim(self) -> int: ...
    @property
    def n_envs(self) -> int: ...
    @property
    def plan_len(self) -> int: ...
    def solve(self, info_dict, init_action=None) -> torch.Tensor: ...


# class BaseSolver:
#     """Base class for planning solvers."""

#     # the idea for solver is to implement different methods for solving planning optimization problems
#     def __init__(self, model: Costable, verbose=True, device="cpu"):
#         self.model = model
#         self.verbose = verbose
#         self.device = device

#     def __call__(self, *args, **kwargs) -> torch.Tensor:
#         return self.solve(*args, **kwargs)

#     def solve(self, info_dict, init_action=None) -> torch.Tensor:
#         """Solve the planning optimization problem given states, action space, and goals."""
#         raise NotImplementedError("Solver must implement the solve method.")
