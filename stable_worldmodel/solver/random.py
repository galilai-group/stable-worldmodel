"""Random action sampling solver for planning problems.

This module provides a baseline solver that samples random actions from the
action space without any optimization. It serves as a simple baseline for
comparison with more sophisticated planning algorithms like CEM, gradient
descent, or MPPI.

The RandomSolver is useful for:
    - Establishing performance baselines in model-based planning experiments
    - Testing environment and policy infrastructure without complex optimization
    - Quick debugging of planning pipelines
    - Ablation studies comparing random vs. optimized action selection

Classes:
    RandomSolver: Samples random actions uniformly from the action space.

Typical usage example:

    Basic usage with stable-worldmodel::

        import stable_worldmodel as swm

        # Create world and solver
        world = swm.World("swm/SimplePointMaze-v0", num_envs=4)
        solver = swm.solver.RandomSolver()

        # Configure planning
        config = swm.PlanConfig(horizon=10, receding_horizon=5, action_block=1)

        # Create policy and evaluate
        policy = swm.policy.WorldModelPolicy(solver=solver, config=config)
        world.set_policy(policy)
        results = world.evaluate(episodes=5)

    Direct solver usage::

        from stable_worldmodel.solver import RandomSolver
        import gymnasium as gym

        # Setup
        env = gym.make("Pendulum-v1")
        solver = RandomSolver()

        config = swm.PlanConfig(horizon=10, receding_horizon=5, action_block=1)
        solver.configure(action_space=env.action_space, n_envs=1, config=config)

        # Generate random actions
        result = solver.solve({})
        actions = result["actions"]  # Shape: (1, 10, action_dim)
"""

import numpy as np
import torch


class RandomSolver:
    """Random action sampling solver for model-based planning.

    This solver generates action sequences by uniformly sampling from the action
    space without any optimization or cost evaluation. Unlike optimization-based
    solvers (CEM, GD, MPPI), it does not require a world model or cost function,
    making it extremely fast and simple to use.

    The solver is primarily intended as a baseline for evaluating the performance
    gains of model-based planning. Random action selection typically performs
    poorly on complex tasks but can be surprisingly effective in simple or
    stochastic environments.

    Key features:
        - **Zero computation cost**: No forward passes through world models
        - **Parallel sampling**: Generates actions for multiple environments simultaneously
        - **Action blocking**: Supports repeating actions for temporal abstraction
        - **Warm-starting**: Can extend partial action sequences
        - **API compatible**: Works with WorldModelPolicy and other solver-based policies

    Attributes:
        n_envs (int): Number of parallel environments being planned for.
        action_dim (int): Total action dimensionality (base_dim × action_block).
        horizon (int): Number of planning steps in the action sequence.

    Example:
        Using with stable-worldmodel's World and Policy classes::

            import stable_worldmodel as swm

            # Create environment
            world = swm.World("swm/SimplePointMaze-v0", num_envs=8)

            # Setup random solver policy
            config = swm.PlanConfig(
                horizon=15,  # Plan 15 steps ahead
                receding_horizon=5,  # Replan every 5 steps
                action_block=1,  # No action repetition
            )
            solver = swm.solver.RandomSolver()
            policy = swm.policy.WorldModelPolicy(solver=solver, config=config)

            # Evaluate
            world.set_policy(policy)
            results = world.evaluate(episodes=10, seed=42)
            print(f"Baseline reward: {results['mean_reward']:.2f}")

        Standalone usage for custom planning loops::

            from stable_worldmodel.solver import RandomSolver
            import gymnasium as gym
            import torch

            env = gym.make("Hopper-v4", render_mode="rgb_array")
            solver = RandomSolver()

            # Configure
            config = swm.PlanConfig(horizon=20, receding_horizon=10, action_block=2)
            solver.configure(action_space=env.action_space, n_envs=1, config=config)

            # Generate and execute actions
            obs, info = env.reset()
            result = solver.solve(info_dict={})
            actions = result["actions"][0]  # Get first env's actions

            for i in range(config.receding_horizon):
                action = actions[i].numpy()
                obs, reward, done, truncated, info = env.step(action)
                if done or truncated:
                    break

    Note:
        This solver ignores the ``info_dict`` parameter in ``solve()`` since it
        doesn't use state information or world models. The parameter is kept for
        API consistency with optimization-based solvers.

    See Also:
        - :class:`~stable_worldmodel.solver.CEMSolver`: Cross-Entropy Method optimizer
        - :class:`~stable_worldmodel.solver.GradientSolver`: Gradient descent optimizer
        - :class:`~stable_worldmodel.solver.MPPISolver`: Model Predictive Path Integral optimizer
    """

    def __init__(self):
        """Initialize an unconfigured RandomSolver.

        Creates a solver instance that must be configured via ``configure()``
        before calling ``solve()``. This two-step initialization allows the
        policy framework to instantiate solvers before environment details
        are available.

        Example:
            Typical initialization pattern::

                solver = RandomSolver()  # Create
                solver.configure(...)  # Configure with env specs
                result = solver.solve({})  # Use
        """
        self._configured = False
        self._action_space = None
        self._n_envs = None
        self._action_dim = None
        self._config = None

    def configure(self, *, action_space, n_envs: int, config) -> None:
        """Configure the solver with environment and planning specifications.

        Must be called before ``solve()`` to set up the action space dimensions,
        number of parallel environments, and planning configuration (horizon,
        action blocking, etc.).

        Args:
            action_space: Gymnasium action space defining valid actions.
                Must have a ``sample()`` method and ``shape`` attribute.
                Typically ``env.action_space`` or ``env.single_action_space``.
            n_envs (int): Number of parallel environments to plan for.
                Action sequences will be generated for each environment
                independently. Must be ≥ 1.
            config: Planning configuration object (typically ``swm.PlanConfig``)
                with required attributes:
                    - ``horizon`` (int): Number of planning steps. Each step
                      corresponds to one action selection point.
                    - ``action_block`` (int): Number of environment steps per
                      planning step. Actions are repeated this many times.

        Raises:
            AttributeError: If config is missing required attributes.
            ValueError: If n_envs < 1 or horizon < 1.

        Example:
            Configure for vectorized environment::

                import stable_worldmodel as swm

                world = swm.World("swm/SimplePointMaze-v0", num_envs=8)
                solver = swm.solver.RandomSolver()

                config = swm.PlanConfig(horizon=10, receding_horizon=5, action_block=1)
                solver.configure(
                    action_space=world.envs.single_action_space,
                    n_envs=world.num_envs,
                    config=config,
                )

        Note:
            The solver extracts ``action_space.shape[1:]`` as the base action
            dimensionality, assuming the first dimension is the batch/environment
            dimension in vectorized action spaces.
        """
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

    @property
    def n_envs(self) -> int:
        """int: Number of parallel environments the solver plans for."""
        return self._n_envs

    @property
    def action_dim(self) -> int:
        """int: Total action dimensionality including action blocking.

        Equals base_action_dim x action_block. For example, if the environment
        has 3D continuous actions and action_block=5, this returns 15.
        """
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        """int: Planning horizon in steps (number of action selections)."""
        return self._config.horizon

    def __call__(self, *args, **kwargs) -> dict:
        """Make the solver callable, forwarding to solve().

        Allows using ``solver(info_dict)`` as shorthand for
        ``solver.solve(info_dict)``. This is the preferred calling convention
        in policy implementations.

        Args:
            *args: Positional arguments passed to solve().
            **kwargs: Keyword arguments passed to solve().

        Returns:
            dict: Dictionary containing 'actions' key with sampled action sequences.

        Example:
            Both calling styles are equivalent::

                result = solver(info_dict={})  # Callable style
                result = solver.solve(info_dict={})  # Explicit style
        """
        return self.solve(*args, **kwargs)

    def solve(self, info_dict, init_action=None) -> dict:
        """Generate random action sequences for the planning horizon.

        Samples random actions uniformly from the action space to create action
        sequences for each environment. If partial action sequences are provided
        via ``init_action``, only the remaining steps are sampled and concatenated.

        This method does not use ``info_dict`` since random sampling doesn't
        require state information, but the parameter is kept for API consistency
        with optimization-based solvers that do use environment state.

        Args:
            info_dict (dict): Environment state information dictionary. Not used
                by RandomSolver but required for solver API consistency. Other
                solvers may use fields like 'state', 'observation', 'latent', etc.
            init_action (torch.Tensor, optional): Partial action sequence to
                warm-start planning. Shape: ``(n_envs, k, action_dim)`` where
                ``k < horizon``. The solver samples actions for the remaining
                ``(horizon - k)`` steps and concatenates them. Useful for
                receding horizon planning where previous plans are reused.
                Defaults to None (sample full horizon).

        Returns:
            dict: Dictionary with a single key:
                - ``'actions'`` (torch.Tensor): Random action sequences with shape
                  ``(n_envs, horizon, action_dim)``. Values are sampled uniformly
                  from the action space bounds.

        Example:
            Generate full random action sequence::

                solver.configure(action_space=env.action_space, n_envs=4, config=config)
                result = solver.solve(info_dict={})
                actions = result["actions"]  # Shape: (4, horizon, action_dim)

            Warm-start with partial sequence (receding horizon planning)::

                # First planning step: full horizon
                result1 = solver.solve({})
                actions1 = result1["actions"]  # (4, 10, action_dim)

                # Execute first 5 actions, then replan
                executed = actions1[:, :5, :]
                remaining = actions1[:, 5:, :]  # Use as warm-start

                # Second planning step: extend remaining actions
                result2 = solver.solve({}, init_action=remaining)
                actions2 = result2["actions"]  # (4, 10, action_dim) - new last 5 steps

        Note:
            The sampling uses ``action_space.sample()`` which respects the space's
            bounds (e.g., Box low/high limits). For continuous spaces, this typically
            produces uniform distributions. For discrete spaces, it samples uniformly
            over valid discrete values.
        """
        outputs = {}
        actions = init_action

        # -- no actions provided, sample
        if actions is None:
            actions = torch.zeros((self.n_envs, 0, self.action_dim))

        # fill remaining actions with random sample
        remaining = self.horizon - actions.shape[1]

        if remaining > 0:
            total_sequence = remaining * self._config.action_block
            action_sequence = np.stack([self._action_space.sample() for _ in range(total_sequence)], axis=1)

            new_action = torch.from_numpy(action_sequence).view(self.n_envs, remaining, self.action_dim)
            actions = torch.cat([actions, new_action], dim=1)

        outputs["actions"] = actions
        return outputs
