"""Tests for the GRASPSolver."""

import numpy as np
import pytest
import torch
from gymnasium import spaces as gym_spaces
from unittest.mock import patch

from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver.grasp import GRASPSolver
from stable_worldmodel.solver.solver import Rollable


# ---------------------------------------------------------------------------
# Dummy models
# ---------------------------------------------------------------------------


class DummyActionEncoder(torch.nn.Module):
    """Trivial action encoder: pads or truncates action to emb_dim."""

    def __init__(self, emb_dim: int = 4):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """action: (B, T, A) -> (B, T, emb_dim)"""
        A = action.shape[-1]
        if A >= self.emb_dim:
            return action[..., : self.emb_dim]
        return torch.nn.functional.pad(action, (0, self.emb_dim - A))


class DummyRollableModel:
    """Minimal model implementing Costable + predict/action_encoder for tests.

    action_encoder: pads/truncates actions to emb_dim.
    predict: returns emb + act_emb (trivially differentiable).
    rollout: used by get_cost sync path — encodes actions in (B,S,T) format.
    get_cost: returns the L2 norm of the full action sequence.
    """

    def __init__(self, emb_dim: int = 4):
        self.action_encoder = DummyActionEncoder(emb_dim)

    def predict(
        self, emb: torch.Tensor, act_emb: torch.Tensor
    ) -> torch.Tensor:
        """emb: (B, T, D), act_emb: (B, T, A_emb) -> (B, T, D)"""
        D = emb.shape[-1]
        A = act_emb.shape[-1]
        if A < D:
            act_emb = torch.nn.functional.pad(act_emb, (0, D - A))
        else:
            act_emb = act_emb[..., :D]
        return emb + act_emb

    def rollout(self, info_dict: dict, action_sequence: torch.Tensor) -> dict:
        """Used by get_cost sync path.

        info_dict['emb']:      (B, D)
        action_sequence:       (B, S, T, action_dim)
        writes predicted_emb:  (B, S, T, D)
        """
        emb = info_dict['emb']  # (B, D)
        B, S, T, A = action_sequence.shape
        D = emb.shape[-1]
        if A >= D:
            proj = action_sequence[..., :D]
        else:
            proj = torch.nn.functional.pad(action_sequence, (0, D - A))
        predicted = emb.detach().view(B, 1, 1, D) + proj  # (B, S, T, D)
        info_dict['predicted_emb'] = predicted
        return info_dict

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Quadratic cost over the full action sequence.

        action_candidates: (B, S, T, action_dim)
        Returns:           (B, S)
        """
        return action_candidates.pow(2).sum(dim=(-1, -2))


class DummyNonRollableModel:
    """Model without predict/action_encoder – tests the guard in __init__."""

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        return action_candidates.pow(2).sum(dim=(-1, -2))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_configured_solver(
    n_envs: int = 2,
    horizon: int = 3,
    action_dim: int = 2,
    emb_dim: int = 4,
    **kwargs,
) -> tuple[GRASPSolver, int]:
    model = DummyRollableModel()
    solver = GRASPSolver(model=model, **kwargs)
    action_space = gym_spaces.Box(
        low=-1, high=1, shape=(n_envs, action_dim), dtype=np.float32
    )
    config = PlanConfig(horizon=horizon, receding_horizon=horizon)
    solver.configure(action_space=action_space, n_envs=n_envs, config=config)
    return solver, emb_dim


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------


def test_init_stores_hyperparams():
    """Constructor stores all hyperparameters correctly."""
    model = DummyRollableModel()
    solver = GRASPSolver(
        model=model,
        n_steps=100,
        batch_size=4,
        lr_s=0.05,
        lr_a=0.002,
        goal_weight=2.0,
        state_noise_scale=0.1,
        gd_interval=25,
        gd_opt_steps=5,
        gd_lr=0.05,
        sync_mode='cem',
        cem_sync_samples=32,
        cem_sync_topk=8,
        cem_sync_var_scale=0.5,
        cem_sync_var_min=0.001,
        schedule_decay=True,
        init_noise_scale=0.2,
        min_noise_scale=0.01,
        init_goal_weight=3.0,
        min_goal_weight=0.5,
        emb_key='my_emb',
        goal_emb_key='my_goal',
        seed=42,
    )
    assert solver.model is model
    assert solver.n_steps == 100
    assert solver.batch_size == 4
    assert solver.lr_s == 0.05
    assert solver.lr_a == 0.002
    assert solver.goal_weight == 2.0
    assert solver.state_noise_scale == 0.1
    assert solver.gd_interval == 25
    assert solver.gd_opt_steps == 5
    assert solver.gd_lr == 0.05
    assert solver.sync_mode == 'cem'
    assert solver.cem_sync_samples == 32
    assert solver.cem_sync_topk == 8
    assert solver.cem_sync_var_scale == 0.5
    assert solver.cem_sync_var_min == 0.001
    assert solver.schedule_decay is True
    assert solver.init_noise_scale == 0.2
    assert solver.min_noise_scale == 0.01
    assert solver.init_goal_weight == 3.0
    assert solver.min_goal_weight == 0.5
    assert solver.emb_key == 'my_emb'
    assert solver.goal_emb_key == 'my_goal'
    assert solver._configured is False


def test_init_raises_without_rollout():
    """TypeError is raised when the model lacks predict/action_encoder."""
    model = DummyNonRollableModel()
    with pytest.raises(TypeError, match='predict'):
        GRASPSolver(model=model)


def test_init_raises_invalid_sync_mode():
    """ValueError is raised for an unrecognised sync_mode."""
    model = DummyRollableModel()
    with pytest.raises(ValueError, match='sync_mode'):
        GRASPSolver(model=model, sync_mode='bad_mode')


def test_rollable_protocol_check():
    """DummyRollableModel satisfies the Rollable structural protocol."""
    assert isinstance(DummyRollableModel(), Rollable)


def test_non_rollable_protocol_check():
    """DummyNonRollableModel does not satisfy the Rollable protocol."""
    assert not isinstance(DummyNonRollableModel(), Rollable)


# ---------------------------------------------------------------------------
# configure() tests
# ---------------------------------------------------------------------------


def test_configure_sets_attributes():
    """configure() sets n_envs, action_dim, horizon, and _configured flag."""
    solver, _ = _make_configured_solver(n_envs=3, horizon=5, action_dim=4)
    assert solver._configured is True
    assert solver.n_envs == 3
    assert solver.action_dim == 4  # action_block defaults to 1
    assert solver.horizon == 5


def test_configure_action_block():
    """action_dim is scaled by action_block from PlanConfig."""
    model = DummyRollableModel()
    solver = GRASPSolver(model=model)
    action_space = gym_spaces.Box(
        low=-1, high=1, shape=(2, 3), dtype=np.float32
    )
    config = PlanConfig(horizon=4, receding_horizon=4, action_block=2)
    solver.configure(action_space=action_space, n_envs=2, config=config)
    assert solver.action_dim == 6  # 3 * action_block=2


def test_configure_warns_on_discrete_space():
    """configure() logs a warning for discrete action spaces."""
    model = DummyRollableModel()
    solver = GRASPSolver(model=model)
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=3, receding_horizon=3)
    with patch('loguru.logger.warning') as mock_warning:
        solver.configure(action_space=action_space, n_envs=1, config=config)
    mock_warning.assert_called_once()
    warning_text = mock_warning.call_args[0][0].lower()
    assert 'discrete' in warning_text


# ---------------------------------------------------------------------------
# _init_virtual_states tests
# ---------------------------------------------------------------------------


def test_init_virtual_states_shape():
    """Virtual states have shape (B, T-1, D) after linear interpolation."""
    solver, D = _make_configured_solver(n_envs=2, horizon=4)
    emb_0 = torch.zeros(2, D)
    goal = torch.ones(2, D)
    vs = solver._init_virtual_states(emb_0, goal)
    assert vs.shape == (2, 3, D)  # T-1 = 3
    assert vs.requires_grad


def test_init_virtual_states_interpolation():
    """Virtual states lie on the straight line between emb_0 and goal."""
    solver, D = _make_configured_solver(n_envs=1, horizon=5)
    emb_0 = torch.zeros(1, D)
    goal = torch.full((1, D), 4.0)
    vs = solver._init_virtual_states(emb_0, goal)
    # T-1 = 4 intermediate points at t = 0.2, 0.4, 0.6, 0.8
    expected = torch.tensor([0.8, 1.6, 2.4, 3.2])
    assert torch.allclose(vs[0, :, 0], expected, atol=1e-5)


def test_init_virtual_states_horizon_one():
    """With horizon=1 there are no virtual states → empty tensor."""
    solver, D = _make_configured_solver(n_envs=2, horizon=1)
    emb_0 = torch.zeros(2, D)
    goal = torch.ones(2, D)
    vs = solver._init_virtual_states(emb_0, goal)
    assert vs.shape == (2, 0, D)
    assert not vs.requires_grad  # empty tensor has no grad


# ---------------------------------------------------------------------------
# _init_actions tests
# ---------------------------------------------------------------------------


def test_init_actions_from_none():
    """With no warm-start, actions are zero-initialised to (B, T, action_dim)."""
    solver, _ = _make_configured_solver(n_envs=2, horizon=3, action_dim=2)
    a = solver._init_actions(None, B=2)
    assert a.shape == (2, 3, 2)
    assert a.requires_grad
    assert a.sum().item() == 0.0


def test_init_actions_padding():
    """Short warm-start actions are zero-padded to the full horizon."""
    solver, _ = _make_configured_solver(n_envs=2, horizon=5, action_dim=2)
    warm = torch.randn(2, 2, 2)  # only 2 steps provided
    a = solver._init_actions(warm, B=2)
    assert a.shape == (2, 5, 2)
    assert torch.allclose(a[:, :2, :], warm)
    assert a[:, 2:, :].sum().item() == 0.0


def test_init_actions_full_horizon():
    """Full-horizon warm-start is not padded."""
    solver, _ = _make_configured_solver(n_envs=2, horizon=3, action_dim=2)
    warm = torch.randn(2, 3, 2)
    a = solver._init_actions(warm, B=2)
    assert a.shape == (2, 3, 2)
    assert torch.allclose(a, warm)


# ---------------------------------------------------------------------------
# _compute_loss tests
# ---------------------------------------------------------------------------


def test_compute_loss_scalar():
    """_compute_loss returns a scalar tensor."""
    solver, D = _make_configured_solver(n_envs=2, horizon=3, action_dim=2)
    B = 2
    emb_0 = torch.zeros(B, D)
    goal = torch.ones(B, D)
    vs = solver._init_virtual_states(emb_0, goal)
    actions = solver._init_actions(None, B)
    loss = solver._compute_loss(vs, actions, emb_0, goal, {})
    assert loss.ndim == 0


def test_compute_loss_has_grad():
    """Loss is differentiable w.r.t. actions and virtual states."""
    solver, D = _make_configured_solver(n_envs=2, horizon=3, action_dim=2)
    B = 2
    emb_0 = torch.zeros(B, D)
    goal = torch.ones(B, D)
    vs = solver._init_virtual_states(emb_0, goal)
    actions = solver._init_actions(None, B)
    loss = solver._compute_loss(vs, actions, emb_0, goal, {})
    loss.backward()
    assert actions.grad is not None
    if vs.numel() > 0:
        assert vs.grad is not None


def test_compute_loss_stop_gradient_on_state():
    """Dynamics input is stop-gradient: emb_0 receives no gradient."""
    _D = 4
    solver, D = _make_configured_solver(n_envs=1, horizon=2, action_dim=_D)
    emb_0 = torch.zeros(1, D, requires_grad=True)
    goal = torch.ones(1, D)
    vs = solver._init_virtual_states(emb_0.detach(), goal)
    actions = solver._init_actions(None, B=1)
    loss = solver._compute_loss(vs, actions, emb_0, goal, {})
    loss.backward()
    assert emb_0.grad is None


def test_compute_loss_uses_goal_weight_param():
    """_compute_loss respects the explicit goal_weight argument."""
    solver, D = _make_configured_solver(n_envs=1, horizon=2, action_dim=2)
    emb_0 = torch.zeros(1, D)
    goal = torch.ones(1, D) * 10.0
    vs = solver._init_virtual_states(emb_0, goal)
    actions = solver._init_actions(None, B=1)
    loss_low = solver._compute_loss(
        vs, actions, emb_0, goal, {}, goal_weight=0.0
    )
    loss_high = solver._compute_loss(
        vs, actions, emb_0, goal, {}, goal_weight=100.0
    )
    assert loss_high > loss_low


# ---------------------------------------------------------------------------
# _compute_per_timestep_var tests
# ---------------------------------------------------------------------------


def test_compute_per_timestep_var_shape():
    """Per-timestep variance has shape (B, T, 1)."""
    solver, D = _make_configured_solver(n_envs=2, horizon=4)
    emb_0 = torch.zeros(2, D)
    goal = torch.ones(2, D)
    vs = solver._init_virtual_states(emb_0, goal)
    var_t = solver._compute_per_timestep_var(vs, emb_0, goal)
    assert var_t.shape == (2, 4, 1)


def test_compute_per_timestep_var_on_linear_path():
    """Variance is at the floor when virtual states lie on the linear baseline."""
    solver, D = _make_configured_solver(
        n_envs=1, horizon=4, cem_sync_var_scale=1.0, cem_sync_var_min=0.01
    )
    emb_0 = torch.zeros(1, D)
    goal = torch.ones(1, D)
    # Virtual states exactly on the linear interpolation → deviation = 0
    vs = solver._init_virtual_states(emb_0, goal)
    var_t = solver._compute_per_timestep_var(vs, emb_0, goal)
    assert torch.allclose(var_t, torch.full_like(var_t, 0.01), atol=1e-5)


def test_compute_per_timestep_var_horizon_one():
    """With horizon=1 (no virtual states) variance equals the floor everywhere."""
    solver, D = _make_configured_solver(
        n_envs=2, horizon=1, cem_sync_var_min=0.05
    )
    emb_0 = torch.zeros(2, D)
    goal = torch.ones(2, D)
    vs = solver._init_virtual_states(emb_0, goal)
    var_t = solver._compute_per_timestep_var(vs, emb_0, goal)
    assert var_t.shape == (2, 1, 1)
    assert torch.allclose(var_t, torch.full_like(var_t, 0.05))


def test_compute_per_timestep_var_increases_with_drift():
    """Var is higher when virtual states deviate from the linear baseline."""
    solver, D = _make_configured_solver(n_envs=1, horizon=3)
    emb_0 = torch.zeros(1, D)
    goal = torch.ones(1, D)

    vs_on_path = solver._init_virtual_states(emb_0, goal)
    vs_off_path = vs_on_path.detach().clone() + 5.0  # large deviation

    var_on = solver._compute_per_timestep_var(vs_on_path, emb_0, goal)
    var_off = solver._compute_per_timestep_var(vs_off_path, emb_0, goal)
    assert var_off.mean() > var_on.mean()


# ---------------------------------------------------------------------------
# _gd_sync tests
# ---------------------------------------------------------------------------


def test_gd_sync_returns_correct_shape():
    """GD sync output has the same shape as the input actions."""
    solver, D = _make_configured_solver(
        n_envs=2, horizon=3, action_dim=2, gd_opt_steps=2
    )
    B = 2
    actions = torch.randn(B, 3, 2, requires_grad=True)
    info_dict = {'emb': torch.zeros(B, D), 'goal_emb': torch.ones(B, D)}
    synced = solver._gd_sync(actions, info_dict)
    assert synced.shape == (B, 3, 2)
    assert not synced.requires_grad


def test_gd_sync_reduces_cost():
    """After the GD sync the model cost is no higher than before."""
    model = DummyRollableModel()
    solver, D = _make_configured_solver(
        n_envs=2, horizon=3, action_dim=2, gd_opt_steps=50, gd_lr=0.1
    )
    B = 2
    actions = torch.randn(B, 3, 2)
    info_dict = {'emb': torch.zeros(B, D), 'goal_emb': torch.ones(B, D)}
    cost_before = model.get_cost(info_dict, actions.unsqueeze(1)).mean().item()
    synced = solver._gd_sync(actions, info_dict)
    cost_after = model.get_cost(info_dict, synced.unsqueeze(1)).mean().item()
    assert cost_after <= cost_before + 1e-4


# ---------------------------------------------------------------------------
# _cem_sync tests
# ---------------------------------------------------------------------------


def test_cem_sync_returns_correct_shape():
    """CEM sync output has the same shape as the input actions."""
    solver, D = _make_configured_solver(
        n_envs=2,
        horizon=3,
        action_dim=2,
        sync_mode='cem',
        cem_sync_samples=8,
        cem_sync_topk=4,
        gd_opt_steps=2,
    )
    B = 2
    actions = torch.randn(B, 3, 2)
    emb_0 = torch.zeros(B, D)
    goal = torch.ones(B, D)
    vs = solver._init_virtual_states(emb_0, goal)
    info_dict = {'emb': emb_0, 'goal_emb': goal}
    synced = solver._cem_sync(actions, vs, emb_0, goal, info_dict)
    assert synced.shape == (B, 3, 2)
    assert not synced.requires_grad


def test_cem_sync_reduces_cost():
    """After the CEM sync the model cost is no higher than before."""
    model = DummyRollableModel()
    solver, D = _make_configured_solver(
        n_envs=2,
        horizon=3,
        action_dim=2,
        sync_mode='cem',
        cem_sync_samples=32,
        cem_sync_topk=8,
        gd_opt_steps=20,
    )
    B = 2
    actions = torch.randn(B, 3, 2) * 2
    emb_0 = torch.zeros(B, D)
    goal = torch.ones(B, D)
    vs = solver._init_virtual_states(emb_0, goal)
    info_dict = {'emb': emb_0, 'goal_emb': goal}
    cost_before = model.get_cost(info_dict, actions.unsqueeze(1)).mean().item()
    synced = solver._cem_sync(actions, vs, emb_0, goal, info_dict)
    cost_after = model.get_cost(info_dict, synced.unsqueeze(1)).mean().item()
    assert cost_after <= cost_before + 1e-4


# ---------------------------------------------------------------------------
# _sync_step dispatch tests
# ---------------------------------------------------------------------------


def test_sync_step_dispatches_to_gd():
    """_sync_step calls _gd_sync when sync_mode='gd'."""
    solver, D = _make_configured_solver(
        n_envs=2, horizon=3, action_dim=2, gd_opt_steps=1
    )
    B = 2
    actions = torch.randn(B, 3, 2)
    emb_0 = torch.zeros(B, D)
    goal = torch.ones(B, D)
    vs = solver._init_virtual_states(emb_0, goal)
    info_dict = {'emb': emb_0, 'goal_emb': goal}
    synced = solver._sync_step(actions, vs, emb_0, goal, info_dict)
    assert synced.shape == (B, 3, 2)


def test_sync_step_dispatches_to_cem():
    """_sync_step calls _cem_sync when sync_mode='cem'."""
    solver, D = _make_configured_solver(
        n_envs=2,
        horizon=3,
        action_dim=2,
        sync_mode='cem',
        cem_sync_samples=4,
        cem_sync_topk=2,
        gd_opt_steps=1,
    )
    B = 2
    actions = torch.randn(B, 3, 2)
    emb_0 = torch.zeros(B, D)
    goal = torch.ones(B, D)
    vs = solver._init_virtual_states(emb_0, goal)
    info_dict = {'emb': emb_0, 'goal_emb': goal}
    synced = solver._sync_step(actions, vs, emb_0, goal, info_dict)
    assert synced.shape == (B, 3, 2)


# ---------------------------------------------------------------------------
# solve() tests
# ---------------------------------------------------------------------------


def test_solve_returns_required_keys():
    """solve() output contains 'actions', 'virtual_states', 'loss_history'."""
    solver, D = _make_configured_solver(
        n_envs=2, horizon=3, action_dim=2, n_steps=2
    )
    B = 2
    info_dict = {'emb': torch.zeros(B, D), 'goal_emb': torch.ones(B, D)}
    outputs = solver.solve(info_dict)
    assert 'actions' in outputs
    assert 'virtual_states' in outputs
    assert 'loss_history' in outputs


def test_solve_output_shapes():
    """solve() returns actions and virtual_states with correct shapes."""
    B, T, A, D = 3, 4, 2, 6
    solver, _ = _make_configured_solver(
        n_envs=B, horizon=T, action_dim=A, emb_dim=D, n_steps=2
    )
    info_dict = {'emb': torch.zeros(B, D), 'goal_emb': torch.ones(B, D)}
    outputs = solver.solve(info_dict)
    assert outputs['actions'].shape == (B, T, A)
    assert outputs['virtual_states'].shape == (B, T - 1, D)


def test_solve_loss_history_format():
    """loss_history is a list of lists; one inner list per batch."""
    n_steps = 7
    solver, D = _make_configured_solver(
        n_envs=2, horizon=3, action_dim=2, n_steps=n_steps
    )
    info_dict = {'emb': torch.zeros(2, D), 'goal_emb': torch.ones(2, D)}
    outputs = solver.solve(info_dict)
    history = outputs['loss_history']
    assert isinstance(history, list)
    assert len(history) == 1  # single batch (no batch_size set)
    assert len(history[0]) == n_steps


def test_solve_outputs_on_cpu():
    """solve() returns tensors on CPU regardless of solver device."""
    solver, D = _make_configured_solver(
        n_envs=2, horizon=3, action_dim=2, n_steps=2
    )
    info_dict = {'emb': torch.zeros(2, D), 'goal_emb': torch.ones(2, D)}
    outputs = solver.solve(info_dict)
    assert outputs['actions'].device.type == 'cpu'
    assert outputs['virtual_states'].device.type == 'cpu'


def test_solve_with_warm_start():
    """solve() accepts an init_action warm-start without errors."""
    B, T, A, D = 2, 4, 2, 4
    solver, _ = _make_configured_solver(
        n_envs=B, horizon=T, action_dim=A, emb_dim=D, n_steps=2
    )
    info_dict = {'emb': torch.zeros(B, D), 'goal_emb': torch.ones(B, D)}
    warm = torch.randn(B, 2, A)  # only 2 of 4 steps provided
    outputs = solver.solve(info_dict, init_action=warm)
    assert outputs['actions'].shape == (B, T, A)


def test_solve_horizon_one():
    """solve() works when horizon=1 (no virtual states)."""
    B, A, D = 2, 2, 4
    solver, _ = _make_configured_solver(
        n_envs=B, horizon=1, action_dim=A, emb_dim=D, n_steps=2
    )
    info_dict = {'emb': torch.zeros(B, D), 'goal_emb': torch.ones(B, D)}
    outputs = solver.solve(info_dict)
    assert outputs['actions'].shape == (B, 1, A)
    assert outputs['virtual_states'].shape == (B, 0, D)


def test_solve_loss_decreases():
    """Loss strictly decreases over a short optimisation run."""
    _D = 4
    solver, D = _make_configured_solver(
        n_envs=2,
        horizon=3,
        action_dim=4,
        emb_dim=_D,
        n_steps=20,
        lr_a=0.1,
        lr_s=0.1,
        state_noise_scale=0.0,  # disable noise for deterministic check
        gd_interval=0,  # disable sync
    )
    emb_0 = torch.zeros(2, D)
    goal = torch.ones(2, D) * 5.0
    info_dict = {'emb': emb_0, 'goal_emb': goal}
    outputs = solver.solve(info_dict)
    history = outputs['loss_history'][0]  # single batch
    assert history[0] > history[-1]


def test_solve_sync_fires():
    """Sync step is triggered at the expected iterations."""
    sync_calls: list[int] = []
    original_sync = GRASPSolver._sync_step

    def patched_sync(
        self, actions, virtual_states, emb_0, goal_emb, info_dict
    ):
        sync_calls.append(1)
        return original_sync(
            self, actions, virtual_states, emb_0, goal_emb, info_dict
        )

    GRASPSolver._sync_step = patched_sync

    try:
        solver, D = _make_configured_solver(
            n_envs=2,
            horizon=3,
            action_dim=2,
            n_steps=6,
            gd_interval=3,
            gd_opt_steps=1,
        )
        info_dict = {'emb': torch.zeros(2, D), 'goal_emb': torch.ones(2, D)}
        solver.solve(info_dict)
        # k=2: k>0 ✓, (k+1=3) % 3 == 0 ✓  → fires
        # k=5: k>0 ✓, (k+1=6) % 3 == 0 ✓  → fires
        assert len(sync_calls) == 2
    finally:
        GRASPSolver._sync_step = original_sync


def test_solve_sync_disabled():
    """No sync step fires when gd_interval=0."""
    sync_calls: list[int] = []
    original_sync = GRASPSolver._sync_step

    def patched_sync(
        self, actions, virtual_states, emb_0, goal_emb, info_dict
    ):
        sync_calls.append(1)
        return original_sync(
            self, actions, virtual_states, emb_0, goal_emb, info_dict
        )

    GRASPSolver._sync_step = patched_sync

    try:
        solver, D = _make_configured_solver(
            n_envs=2, horizon=3, action_dim=2, n_steps=10, gd_interval=0
        )
        info_dict = {'emb': torch.zeros(2, D), 'goal_emb': torch.ones(2, D)}
        solver.solve(info_dict)
        assert len(sync_calls) == 0
    finally:
        GRASPSolver._sync_step = original_sync


def test_solve_with_batch_size():
    """batch_size splits envs correctly; output shapes match single-pass solve."""
    B, T, A, D = 4, 3, 2, 4
    info_dict = {'emb': torch.zeros(B, D), 'goal_emb': torch.ones(B, D)}

    solver_full, _ = _make_configured_solver(
        n_envs=B, horizon=T, action_dim=A, emb_dim=D, n_steps=2, gd_interval=0
    )
    solver_batched, _ = _make_configured_solver(
        n_envs=B,
        horizon=T,
        action_dim=A,
        emb_dim=D,
        n_steps=2,
        batch_size=2,
        gd_interval=0,
    )

    out_full = solver_full.solve(info_dict)
    out_batched = solver_batched.solve(info_dict)

    assert out_batched['actions'].shape == out_full['actions'].shape
    assert (
        out_batched['virtual_states'].shape == out_full['virtual_states'].shape
    )
    # Two batches of size 2 → two inner loss lists
    assert len(out_batched['loss_history']) == 2
    assert len(out_batched['loss_history'][0]) == 2
    assert len(out_batched['loss_history'][1]) == 2


def test_solve_schedule_decay_runs():
    """solve() runs without errors when schedule_decay=True."""
    solver, D = _make_configured_solver(
        n_envs=2,
        horizon=3,
        action_dim=2,
        n_steps=6,
        gd_interval=3,
        schedule_decay=True,
        init_noise_scale=0.2,
        min_noise_scale=0.0,
        init_goal_weight=3.0,
        min_goal_weight=0.5,
    )
    info_dict = {'emb': torch.zeros(2, D), 'goal_emb': torch.ones(2, D)}
    out = solver.solve(info_dict)
    assert out['actions'].shape == (2, 3, 2)
    assert len(out['loss_history'][0]) == 6


def test_solve_cem_sync_runs():
    """solve() with sync_mode='cem' runs without errors."""
    solver, D = _make_configured_solver(
        n_envs=2,
        horizon=3,
        action_dim=2,
        n_steps=4,
        gd_interval=2,
        sync_mode='cem',
        cem_sync_samples=8,
        cem_sync_topk=4,
        gd_opt_steps=2,
    )
    info_dict = {'emb': torch.zeros(2, D), 'goal_emb': torch.ones(2, D)}
    out = solver.solve(info_dict)
    assert out['actions'].shape == (2, 3, 2)


def test_callable_calls_solve():
    """__call__ delegates to solve()."""
    solver, D = _make_configured_solver(
        n_envs=2, horizon=3, action_dim=2, n_steps=1
    )
    info_dict = {'emb': torch.zeros(2, D), 'goal_emb': torch.ones(2, D)}
    out_call = solver(info_dict)
    assert 'actions' in out_call


def test_solve_returns_cost_key():
    """solve() returns 'cost' with shape (B,) on CPU."""
    B, T, A, D = 3, 4, 2, 6
    solver, _ = _make_configured_solver(
        n_envs=B, horizon=T, action_dim=A, emb_dim=D, n_steps=2
    )
    info_dict = {'emb': torch.zeros(B, D), 'goal_emb': torch.ones(B, D)}
    outputs = solver.solve(info_dict)
    assert 'cost' in outputs
    assert outputs['cost'].shape == (B,)
    assert outputs['cost'].device.type == 'cpu'


def test_solve_sync_updates_actions():
    """Loss changes at the sync boundary, confirming synced actions are applied."""
    losses: list[float] = []
    original_loss_fn = GRASPSolver._compute_loss

    def recording_loss(self, *args, **kwargs):
        loss_val = original_loss_fn(self, *args, **kwargs)
        losses.append(loss_val.item())
        return loss_val

    GRASPSolver._compute_loss = recording_loss
    try:
        solver, D = _make_configured_solver(
            n_envs=2,
            horizon=3,
            action_dim=2,
            n_steps=6,
            gd_interval=3,
            gd_opt_steps=5,
            gd_lr=0.1,
            state_noise_scale=0.0,  # disable noise for a deterministic check
        )
        info_dict = {'emb': torch.zeros(2, D), 'goal_emb': torch.ones(2, D)}
        solver.solve(info_dict)
        # losses[2] is the last gradient step before the first sync (k=2).
        # losses[3] is the first gradient step after the sync (k=3).
        # If synced actions were silently discarded they would be equal.
        assert losses[2] != losses[3], (
            'Loss did not change across the sync boundary — '
            'synced actions may not have been applied to the optimiser.'
        )
    finally:
        GRASPSolver._compute_loss = original_loss_fn


def test_default_min_goal_weight():
    """Default min_goal_weight is 0.0 so the goal term fully anneals."""
    model = DummyRollableModel()
    assert GRASPSolver(model=model).min_goal_weight == 0.0
