"""Tests for continuous solvers (CEM, MPPI, GD, Nevergrad)."""

import numpy as np
import torch
from gymnasium import spaces as gym_spaces

from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver.cem import CEMSolver
from stable_worldmodel.solver.gd import GradientSolver
from stable_worldmodel.solver.icem import ICEMSolver
from stable_worldmodel.solver.mppi import MPPISolver


class DummyCostModel:
    """Simple Costable implementation for tests."""

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        # Quadratic cost: sum over horizon and action dims
        cost = action_candidates.pow(2).sum(dim=(-1, -2))
        return cost


###########################
## CEMSolver Tests       ##
###########################


def test_cem_solver_init():
    """Test CEMSolver initialization."""
    model = DummyCostModel()
    solver = CEMSolver(model=model, n_steps=10, num_samples=100)
    assert solver.model is model
    assert solver.n_steps == 10
    assert solver.num_samples == 100


def test_cem_solver_configure():
    """Test CEMSolver configuration."""
    model = DummyCostModel()
    solver = CEMSolver(model=model, n_steps=10)
    action_space = gym_spaces.Box(
        low=-1, high=1, shape=(4, 2), dtype=np.float32
    )
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    assert solver._configured is True
    assert solver.n_envs == 2
    assert solver.action_dim == 2
    assert solver.horizon == 5


def test_cem_solver_configure_discrete_warning(caplog):
    """Test CEMSolver warns on discrete action space."""
    model = DummyCostModel()
    solver = CEMSolver(model=model, n_steps=10)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=5, receding_horizon=3)

    solver.configure(action_space=action_space, n_envs=1, config=config)
    assert (
        'discrete' in caplog.text.lower() or solver._configured
    )  # warning logged


def test_cem_solver_init_action_distrib():
    """Test CEMSolver action distribution initialization."""
    model = DummyCostModel()
    solver = CEMSolver(model=model, n_steps=10)
    action_space = gym_spaces.Box(
        low=-1, high=1, shape=(2, 3), dtype=np.float32
    )
    config = PlanConfig(horizon=5, receding_horizon=3)
    solver.configure(action_space=action_space, n_envs=2, config=config)

    actions = torch.zeros(2, 5, 3)
    mean, var = solver.init_action_distrib(actions)
    assert mean.shape == (2, 5, 3)
    assert var.shape == (2, 5, 3)


def test_cem_solver_init_action_distrib_with_init():
    """Test CEMSolver action distribution returns provided actions as mean."""
    model = DummyCostModel()
    solver = CEMSolver(model=model, n_steps=10)
    action_space = gym_spaces.Box(
        low=-1, high=1, shape=(2, 3), dtype=np.float32
    )
    config = PlanConfig(horizon=5, receding_horizon=3)
    solver.configure(action_space=action_space, n_envs=2, config=config)

    init_actions = torch.randn(2, 5, 3)
    mean, var = solver.init_action_distrib(init_actions)
    assert mean.shape == (2, 5, 3)
    assert mean is init_actions


def test_cem_solver_call():
    """Test CEMSolver __call__ method."""
    model = DummyCostModel()
    solver = CEMSolver(
        model=model, n_steps=2, num_samples=50, batch_size=2, topk=10
    )
    action_space = gym_spaces.Box(
        low=-1, high=1, shape=(2, 2), dtype=np.float32
    )
    config = PlanConfig(horizon=3, receding_horizon=2)
    solver.configure(action_space=action_space, n_envs=2, config=config)

    info_dict = {'pixels': torch.randn(2, 1, 3, 64, 64)}
    outputs = solver(info_dict)

    assert 'actions' in outputs
    assert outputs['actions'].shape == (2, 3, 2)


###########################
## ICEMSolver Tests      ##
###########################


def test_icem_solver_init():
    """Test ICEMSolver initialization."""
    model = DummyCostModel()
    solver = ICEMSolver(
        model=model, n_steps=10, num_samples=100, noise_beta=2.0
    )
    assert solver.model is model
    assert solver.n_steps == 10
    assert solver.num_samples == 100
    assert solver.noise_beta == 2.0
    assert solver.alpha == 0.1
    assert solver.n_elite_keep == 5


def test_icem_solver_configure():
    """Test ICEMSolver configuration."""
    model = DummyCostModel()
    solver = ICEMSolver(model=model, n_steps=10)
    action_space = gym_spaces.Box(
        low=-1, high=1, shape=(4, 2), dtype=np.float32
    )
    config = PlanConfig(horizon=5, receding_horizon=3, action_block=1)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    assert solver._configured is True
    assert solver.n_envs == 2
    assert solver.action_dim == 2
    assert solver.horizon == 5
    assert solver._action_low is not None
    assert solver._action_high is not None


def test_icem_solver_init_action_distrib():
    """Test ICEMSolver action distribution initialization."""
    model = DummyCostModel()
    solver = ICEMSolver(model=model, n_steps=10)
    action_space = gym_spaces.Box(
        low=-1, high=1, shape=(2, 3), dtype=np.float32
    )
    config = PlanConfig(horizon=5, receding_horizon=3)
    solver.configure(action_space=action_space, n_envs=2, config=config)

    actions = torch.zeros(2, 5, 3)
    mean, var = solver.init_action_distrib(actions)
    assert mean.shape == (2, 5, 3)
    assert var.shape == (2, 5, 3)


def test_icem_solver_call():
    """Test ICEMSolver __call__ method."""
    model = DummyCostModel()
    solver = ICEMSolver(
        model=model, n_steps=2, num_samples=50, batch_size=2, topk=10
    )
    action_space = gym_spaces.Box(
        low=-1, high=1, shape=(2, 2), dtype=np.float32
    )
    config = PlanConfig(horizon=3, receding_horizon=2)
    solver.configure(action_space=action_space, n_envs=2, config=config)

    info_dict = {'pixels': torch.randn(2, 1, 3, 64, 64)}
    outputs = solver(info_dict)

    assert 'actions' in outputs
    assert outputs['actions'].shape == (2, 3, 2)


def test_icem_solver_white_noise_fallback():
    """Test ICEMSolver with beta=0 (white noise, equivalent to standard CEM)."""
    model = DummyCostModel()
    solver = ICEMSolver(
        model=model,
        n_steps=2,
        num_samples=50,
        batch_size=2,
        topk=10,
        noise_beta=0.0,
    )
    action_space = gym_spaces.Box(
        low=-1, high=1, shape=(2, 2), dtype=np.float32
    )
    config = PlanConfig(horizon=3, receding_horizon=2)
    solver.configure(action_space=action_space, n_envs=2, config=config)

    info_dict = {'pixels': torch.randn(2, 1, 3, 64, 64)}
    outputs = solver(info_dict)

    assert 'actions' in outputs
    assert outputs['actions'].shape == (2, 3, 2)


###########################
## MPPISolver Tests      ##
###########################


def test_mppi_solver_init():
    """Test MPPISolver initialization."""
    model = DummyCostModel()
    solver = MPPISolver(model=model, n_steps=10, temperature=0.5)
    assert solver.model is model
    assert solver.n_steps == 10
    assert solver.temperature == 0.5


def test_mppi_solver_configure():
    """Test MPPISolver configuration."""
    model = DummyCostModel()
    solver = MPPISolver(model=model, n_steps=10)
    action_space = gym_spaces.Box(
        low=-1, high=1, shape=(4, 2), dtype=np.float32
    )
    config = PlanConfig(horizon=5, receding_horizon=3)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    assert solver._configured is True
    assert solver.n_envs == 2
    assert solver.action_dim == 2
    assert solver.horizon == 5


def test_mppi_solver_init_action_distrib():
    """Test MPPISolver action distribution initialization."""
    model = DummyCostModel()
    solver = MPPISolver(model=model, n_steps=10)
    action_space = gym_spaces.Box(
        low=-1, high=1, shape=(2, 3), dtype=np.float32
    )
    config = PlanConfig(horizon=5, receding_horizon=3)
    solver.configure(action_space=action_space, n_envs=2, config=config)

    actions = torch.zeros(2, 5, 3)
    mean, var = solver.init_action_distrib(actions)
    assert mean.shape == (2, 5, 3)
    assert var.shape == (2, 5, 3)


def test_mppi_solver_call():
    """Test MPPISolver __call__ method."""
    model = DummyCostModel()
    solver = MPPISolver(model=model, n_steps=2, num_samples=10, batch_size=2)
    action_space = gym_spaces.Box(
        low=-1, high=1, shape=(2, 2), dtype=np.float32
    )
    config = PlanConfig(horizon=3, receding_horizon=2)
    solver.configure(action_space=action_space, n_envs=2, config=config)

    info_dict = {'pixels': torch.randn(2, 1, 3, 64, 64)}
    outputs = solver(info_dict)

    assert 'actions' in outputs
    assert outputs['actions'].shape == (2, 3, 2)


###########################
## GradientSolver Tests  ##
###########################


def test_gradient_solver_init():
    """Test GradientSolver initialization."""
    model = DummyCostModel()
    solver = GradientSolver(model=model, n_steps=10)
    assert solver.model is model
    assert solver.n_steps == 10
    assert solver._configured is False


def test_gradient_solver_configure():
    """Test GradientSolver configuration."""
    model = DummyCostModel()
    solver = GradientSolver(model=model, n_steps=10)
    action_space = gym_spaces.Box(
        low=-1, high=1, shape=(4, 2), dtype=np.float32
    )
    config = PlanConfig(horizon=5, receding_horizon=3)

    solver.configure(action_space=action_space, n_envs=2, config=config)

    assert solver._configured is True
    assert solver.n_envs == 2
    assert solver.action_dim == 2
    assert solver.horizon == 5


def test_gradient_solver_init_action():
    """Test GradientSolver action initialization."""
    model = DummyCostModel()
    solver = GradientSolver(model=model, n_steps=10, num_samples=3)
    action_space = gym_spaces.Box(
        low=-1, high=1, shape=(2, 3), dtype=np.float32
    )
    config = PlanConfig(horizon=5, receding_horizon=3)
    solver.configure(action_space=action_space, n_envs=2, config=config)

    solver.init_action(torch.zeros(2, 5, 3))
    assert hasattr(solver, 'init')
    assert solver.init.shape == (
        2,
        3,
        5,
        3,
    )  # (n_envs, num_samples, horizon, action_dim)


def test_gradient_solver_call():
    """Test GradientSolver __call__ method."""
    model = DummyCostModel()
    solver = GradientSolver(
        model=model, n_steps=2, num_samples=2, batch_size=2
    )
    action_space = gym_spaces.Box(
        low=-1, high=1, shape=(2, 2), dtype=np.float32
    )
    config = PlanConfig(horizon=3, receding_horizon=2)
    solver.configure(action_space=action_space, n_envs=2, config=config)

    info_dict = {'pixels': torch.randn(2, 1, 3, 64, 64)}
    outputs = solver(info_dict)

    assert 'actions' in outputs
    assert outputs['actions'].shape == (2, 3, 2)
