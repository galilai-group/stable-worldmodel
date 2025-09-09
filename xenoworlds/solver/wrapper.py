"""Wrapper classes for solvers."""


class SolverWrapper:
    def __init__(self):
        return

    def solve(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)


class MPCWrapper(SolverWrapper):
    def __init__(self, solver, n_mpc_actions):
        super().__init__()
        self.solver = solver
        self.n_mpc_actions = n_mpc_actions

    def solve(self, *args, **kwargs):
        """Solve the environment using the MPC solver."""
        actions = self.solver(*args, **kwargs)
        return actions[:, : self.n_mpc_actions].numpy()
