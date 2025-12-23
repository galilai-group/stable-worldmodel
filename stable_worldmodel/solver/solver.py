from typing import Protocol, runtime_checkable

import gymnasium as gym
import torch


class Costable(Protocol):
    """Protocol for world model cost functions.

    This protocol defines the interface for models that can compute costs
    for planning and optimization. Models implementing this protocol can
    evaluate the quality of action sequences in a given environment state.

    Example:
        >>> class MyWorldModel(Costable):
        ...     def get_cost(self, info_dict, action_candidates):
        ...         # Compute cost based on predicted trajectories from action candidates
        ...         return costs
    """

    def criterion(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """
        Criterion used by get_cost to compute the cost. This method can be overridden to easily define a custom cost function.
        Args:
            info_dict: Dictionary containing environment state information. Typically contains info such as the goal and the results of rollouts.
            action_candidates: Tensor of shape (B, N, horizon, action_dim)
                containing action sequences to evaluate. B is the number of trajectories,
                N is the number of samples (action candidates).
        Returns:
            Tensor of shape (n_envs,) containing the cost of each environment action sequence.
            Lower costs indicate better action sequences.
        """

    def get_cost(info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """Compute cost for given action candidates based on info dictionary. Typically get_cost performs rollouts
        using the world model and then computes costs using the criterion.

        Args:
            info_dict: Dictionary containing environment state information.
                Typically includes keys like 'pixels', 'goal', 'proprio', 'predicted_states', etc.
                Note: the shape of tensors in info_dict should be (B, N, ...) where B is the number of trajectories
                and N is the number of samples (action candidates). This means that the tensors are repeated
                along the N dimension for each action candidate.
            action_candidates: Tensor of shape (B, N, horizon, action_dim)
                containing action sequences to evaluate. B and N should match the corresponding dimensions
                in info_dict.

        Returns:
            Tensor of shape (n_envs,) containing the cost of each environment action sequence
            .A lower costs indicate better action sequences.

        Note:
            The cost computation should be differentiable (`requires_grad=True`) with respect to
            action_candidates to enable gradient-based planning methods.
        """
        ...


@runtime_checkable
class Solver(Protocol):
    """Protocol for model-based planning solvers.

    This protocol defines the interface for optimization algorithms that plan
    action sequences by minimizing a cost function computed by a world model.
    Solvers receive the current environment state (observations, goals, proprioception)
    and output optimal action sequences that achieve desired behaviors.

    Planning Process:
        1. Receive current state via info_dict (pixels, goal, proprio, etc.)
        2. Initialize or warm-start action sequences
        3. Optimize actions using the world model's ``get_cost`` function
        4. Return optimized action sequences for execution

    The protocol supports various optimization methods including:
        - Gradient-based: GradientSolver (gradient descent)
        - Sampling-based: CEMSolver (cross-entropy method), MPPISolver
        - Random: RandomSolver (baseline)

    Key Concepts:
        - **Horizon**: Number of timesteps to plan ahead
        - **Action Block**: Number of actions grouped together due to frame skip.
        - **Receding Horizon**: Number of actions actually executed before replanning
        - **Warm Start**: Using previous solutions leftover to initialize new optimization

    Attributes:
        action_dim (int): Flattened action dimension including action_block grouping.
            Formula: base_action_dim * action_block
        n_envs (int): Number of parallel environments being optimized simultaneously.
        horizon (int): Planning horizon length in timesteps.

    Example:
        Basic usage with a world model:

        >>> # Setup world model and planning config
        >>> world_model = DINOWM(encoder, predictor, ...)
        >>> plan_config = PlanConfig(horizon=10, receding_horizon=5, action_block=2)
        >>>
        >>> # Create and configure solver
        >>> solver = GradientSolver(world_model, n_steps=10, device="cuda")
        >>> solver.configure(
        ...     action_space=env.action_space,
        ...     n_envs=4,
        ...     config=plan_config
        ... )
        >>>
        >>> # Solve for optimal actions given current state
        >>> info_dict = {'pixels': pixels, 'goal': goal, 'proprio': proprio}
        >>> outputs = solver.solve(info_dict, init_action=None)
        >>> actions = outputs["actions"]  # Shape: (4, 10, action_dim)
        >>>
        >>> # Warm-start next optimization with remaining actions
        >>> next_outputs = solver.solve(info_dict, init_action=outputs["actions"][:, 5:])

    See Also:
        - Costable: Protocol defining the world model cost interface
        - PlanConfig: Configuration dataclass for planning parameters
        - GradientSolver, CEMSolver, MPPISolver: Concrete solver implementations
    """

    def configure(self, *, action_space: gym.Space, n_envs: int, config) -> None:
        """Configure the solver with environment and planning specifications.

        This method initializes the solver's internal state based on the
        environment's action space and planning configuration. Must be called
        once after solver creation and before any solve() calls.

        Args:
            action_space (gym.Space): Environment's action space. For continuous
                control, this should be a Box space. The shape is typically
                (n_envs, action_dim) for vectorized environments.
            n_envs (int): Number of parallel environments to optimize for. The
                solver will produce n_envs independent action sequences.
            config (PlanConfig): Planning configuration containing:
                - horizon: Number of future timesteps to plan
                - receding_horizon: Number of planned actions to execute
                - action_block: Number of actions grouped together due to frame skip

        Note:
            This method should only be called once during initialization. The
            solver caches the configuration internally for use in solve().

        Raises:
            Warning: If action_space is not a Box (some solvers only support
                continuous actions).
        """
        ...

    @property
    def action_dim(self) -> int:
        """int: Flattened action dimension including action_block grouping.

        This is the total size of actions per timestep, computed as:
        base_action_dim * action_block

        The action_block groups multiple actions together for frame skipping.
        For example, if the environment has 2D actions and action_block=5,
        then action_dim=10 (the 2 action dimensions grouped 5 times).

        Returns:
            int: Total flattened action dimension used in optimization.
        """
        ...

    @property
    def n_envs(self) -> int:
        """int: Number of parallel environments being planned for.

        Returns:
            int: Number of independent action sequences the solver optimizes.
        """
        ...

    @property
    def horizon(self) -> int:
        """int: Planning horizon length in timesteps.

        This is the number of future timesteps the solver plans ahead.
        Note that this may differ from receding_horizon (the number of
        actions actually executed before replanning).

        Returns:
            int: Number of timesteps in the planning horizon.
        """
        ...

    def solve(self, info_dict, init_action=None) -> dict:
        """Solve the planning optimization problem to find optimal actions.

        This is the main method that performs trajectory optimization. It uses
        the world model to evaluate action sequences and finds actions that
        minimize the cost function. The optimization strategy is solver-specific
        (gradient descent, sampling, etc.).

        Typical workflow:
            1. Initialize action sequences (from init_action or zeros)
            2. Iteratively evaluate cost and update actions
            3. Return optimized actions and optimization statistics

        Args:
            info_dict (dict): Current environment state containing:
                - 'pixels' (np.ndarray): Current observation images, shape (n_envs, H, W, 3)
                - 'goal' (np.ndarray): Goal observation images, shape (n_envs, H, W, 3)
                - 'proprio' (np.ndarray, optional): Proprioceptive state, shape (n_envs, proprio_dim)
                - 'action' (np.ndarray, optional): Previous actions for history
                - Additional task-specific keys as needed

            init_action (torch.Tensor, optional): Warm-start action sequences with
                shape (n_envs, init_horizon, action_dim). Common use cases:
                - None: Initialize all actions to zero (cold start)
                - Partial sequence: Pad remaining horizon with zeros
                - Previous solution shifted: Warm-start from last optimization

        Returns:
            dict: Optimization results containing:
                - 'actions' (torch.Tensor): Optimized action sequences with shape
                  (n_envs, horizon, action_dim). These are the planned actions.
                - 'cost' (list[float]): Cost values during optimization. Format and
                  length depend on the solver implementation.
                - 'trajectory' (list[torch.Tensor]): Intermediate action sequences
                  during optimization (solver-dependent).
                - Additional solver-specific keys (e.g., 'elite_actions' for CEM)

        Note:
            The returned actions are typically in the solver's internal representation
            and may require denormalization or reshaping before execution in the
            environment. The WorldModelPolicy handles this transformation.

        Example:
            Cold start (zero initialization):
            >>> outputs = solver.solve(info_dict)

            Warm start with previous solution:
            >>> outputs1 = solver.solve(info_dict)
            >>> # Execute first 5 actions, keep rest for warm start
            >>> outputs2 = solver.solve(new_info_dict, init_action=outputs1["actions"][:, 5:])
        """
        ...
