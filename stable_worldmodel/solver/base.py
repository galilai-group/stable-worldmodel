import torch

class BaseSolver(torch.nn.Module):
    """Base class for planning solvers"""

    # the idea for solver is to implement different methods for solving planning optimization problems
    def __init__(self, cost_fn, verbose=True, device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.cost_fn = cost_fn        
        self.verbose = verbose
        self.device = device

    def __call__(
        self, z_obs0, z_goal, action_space, predict_fn=None, init_action=None
    ) -> torch.Tensor:
        return self.solve(z_obs0, z_goal, action_space, predict_fn, init_action)

    def solve(
        self, z_obs0, z_goal, action_space, predict_fn=None, init_action=None
    ) -> torch.Tensor:
        """Solve the planning optimization problem given states, action space, and goals."""
        raise NotImplementedError("Solver must implement the solve method.")
