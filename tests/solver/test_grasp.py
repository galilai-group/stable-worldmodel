"""Tests for GRASPSolver."""

import numpy as np
import pytest
import torch
from gymnasium import spaces as gym_spaces

from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver.grasp import GRASPSolver

EMB_DIM = 8
ACT_DIM = 4


class DummyGRASPModel:
    """Minimal model satisfying the GRASPSolver interface."""

    def encode(self, info_dict: dict) -> dict:
        pixels = info_dict["pixels"]
        B = pixels.shape[0]
        T = pixels.shape[1] if pixels.ndim == 5 else 1
        out = dict(info_dict)
        out["emb"] = torch.zeros(B, T, EMB_DIM)
        return out

    def action_encoder(self, actions: torch.Tensor) -> torch.Tensor:
        # actions: (N, 1, A) → (N, 1, EMB_DIM) (linear map for grad flow)
        N, S, A = actions.shape
        if A >= EMB_DIM:
            return actions[..., :EMB_DIM]
        # Pad with zeros while keeping actions in the graph so gradients flow
        pad = torch.zeros(N, S, EMB_DIM - A, device=actions.device)
        return torch.cat([actions, pad], dim=-1)

    def predict(self, s_t: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        # s_t: (N, 1, D), act_emb: (N, 1, D) → (N, 1, D)
        return s_t + act_emb * 0.0  # differentiable; output shape (N, 1, EMB_DIM)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        # action_candidates: (B, S, T, A) → cost (B, S)
        return action_candidates.pow(2).sum(dim=(-1, -2))


class DummyGRASPModelNDEmb:
    """Model that returns a 2-D embedding (no time dim)."""

    def encode(self, info_dict: dict) -> dict:
        pixels = info_dict["pixels"]
        B = pixels.shape[0]
        out = dict(info_dict)
        out["emb"] = torch.zeros(B, EMB_DIM)
        return out

    def action_encoder(self, actions: torch.Tensor) -> torch.Tensor:
        N, S, A = actions.shape
        return torch.zeros(N, S, EMB_DIM)

    def predict(self, s_t: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        return s_t

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        return action_candidates.pow(2).sum(dim=(-1, -2))


# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------

def _make_solver(**kwargs) -> GRASPSolver:
    defaults = dict(
        model=DummyGRASPModel(),
        n_steps=3,
        lr_s=0.01,
        lr_a=0.01,
        device="cpu",
        seed=0,
    )
    defaults.update(kwargs)
    return GRASPSolver(**defaults)


def _configure(solver: GRASPSolver, horizon: int = 3, n_envs: int = 2, action_shape=(1, ACT_DIM)):
    action_space = gym_spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32)
    config = PlanConfig(horizon=horizon, receding_horizon=1)
    solver.configure(action_space=action_space, n_envs=n_envs, config=config)


def _basic_info_dict(B: int = 2, T: int = 1, H: int = 8, C: int = 3) -> dict:
    return {
        "pixels": torch.randn(B, T, C, H, H),
        "goal": torch.randn(B, C, H, H),
    }


###############################################################################
# Initialisation
###############################################################################


def test_init_stores_hyperparameters():
    model = DummyGRASPModel()
    solver = GRASPSolver(
        model=model,
        n_steps=50,
        lr_s=0.05,
        lr_a=0.002,
        goal_weight=2.0,
        state_noise_scale=0.05,
        gd_interval=10,
        gd_opt_steps=5,
        gd_lr=0.005,
        sync_mode="gd",
        schedule_decay=True,
        init_noise_scale=0.5,
        min_noise_scale=0.0,
        init_goal_weight=3.0,
        min_goal_weight=1.0,
        device="cpu",
        seed=42,
    )
    assert solver.model is model
    assert solver.n_steps == 50
    assert solver.lr_s == 0.05
    assert solver.lr_a == 0.002
    assert solver.goal_weight == 2.0
    assert solver.state_noise_scale == 0.05
    assert solver.gd_interval == 10
    assert solver.gd_opt_steps == 5
    assert solver.gd_lr == 0.005
    assert solver.sync_mode == "gd"
    assert solver.schedule_decay is True
    assert solver.init_noise_scale == 0.5
    assert solver.min_noise_scale == 0.0
    assert solver.init_goal_weight == 3.0
    assert solver.min_goal_weight == 1.0
    assert solver._configured is False


def test_init_missing_encode_raises():
    class NoEncode:
        def predict(self, *a): ...
        def action_encoder(self, *a): ...
        def get_cost(self, *a): ...

    with pytest.raises(TypeError, match="encode"):
        GRASPSolver(model=NoEncode())


def test_init_missing_predict_raises():
    class NoPredict:
        def encode(self, *a): ...
        def action_encoder(self, *a): ...
        def get_cost(self, *a): ...

    with pytest.raises(TypeError, match="predict"):
        GRASPSolver(model=NoPredict())


def test_init_missing_action_encoder_raises():
    class NoActionEncoder:
        def encode(self, *a): ...
        def predict(self, *a): ...
        def get_cost(self, *a): ...

    with pytest.raises(TypeError, match="action_encoder"):
        GRASPSolver(model=NoActionEncoder())


def test_init_missing_get_cost_raises():
    class NoGetCost:
        def encode(self, *a): ...
        def predict(self, *a): ...
        def action_encoder(self, *a): ...

    with pytest.raises(TypeError, match="get_cost"):
        GRASPSolver(model=NoGetCost())


def test_init_invalid_sync_mode_raises():
    with pytest.raises(ValueError, match="sync_mode"):
        GRASPSolver(model=DummyGRASPModel(), sync_mode="bad_mode")


def test_init_cem_sync_mode_accepted():
    solver = GRASPSolver(model=DummyGRASPModel(), sync_mode="cem")
    assert solver.sync_mode == "cem"


###############################################################################
# configure
###############################################################################


def test_configure_sets_flags():
    solver = _make_solver()
    _configure(solver, horizon=5, n_envs=3)
    assert solver._configured is True
    assert solver.n_envs == 3
    assert solver.horizon == 5


def test_configure_action_dim_continuous():
    solver = _make_solver()
    _configure(solver, action_shape=(1, ACT_DIM))
    assert solver.action_dim == ACT_DIM


def test_configure_discrete_space_does_not_raise():
    """Discrete action space triggers a warning but must not crash."""
    solver = _make_solver()
    action_space = gym_spaces.Discrete(5)
    config = PlanConfig(horizon=3, receding_horizon=1)
    solver.configure(action_space=action_space, n_envs=1, config=config)
    assert solver._configured is True


###############################################################################
# _init_virtual_states
###############################################################################


def test_init_virtual_states_shape_multi_step():
    solver = _make_solver()
    _configure(solver, horizon=4)
    emb_0 = torch.zeros(2, EMB_DIM)
    goal_emb = torch.ones(2, EMB_DIM)
    vs = solver._init_virtual_states(emb_0, goal_emb)
    # horizon=4 → 3 virtual states
    assert vs.shape == (2, 3, EMB_DIM)
    assert vs.requires_grad is True


def test_init_virtual_states_horizon_1():
    solver = _make_solver()
    _configure(solver, horizon=1)
    emb_0 = torch.zeros(2, EMB_DIM)
    goal_emb = torch.ones(2, EMB_DIM)
    vs = solver._init_virtual_states(emb_0, goal_emb)
    assert vs.shape == (2, 0, EMB_DIM)
    assert vs.requires_grad is False


def test_init_virtual_states_linear_interpolation():
    solver = _make_solver()
    _configure(solver, horizon=3)   # → 2 virtual states at t=1/3, 2/3
    emb_0 = torch.zeros(1, EMB_DIM)
    goal_emb = torch.ones(1, EMB_DIM) * 3.0
    vs = solver._init_virtual_states(emb_0, goal_emb)  # (1, 2, D)
    # Midpoints at 1/3 and 2/3 of [0, 3]
    assert torch.allclose(vs[0, 0], torch.full((EMB_DIM,), 1.0), atol=1e-5)
    assert torch.allclose(vs[0, 1], torch.full((EMB_DIM,), 2.0), atol=1e-5)


###############################################################################
# _init_actions
###############################################################################


def test_init_actions_none_returns_zeros():
    solver = _make_solver()
    _configure(solver, horizon=3)
    a = solver._init_actions(None, B=2)
    assert a.shape == (2, 3, ACT_DIM)
    assert a.requires_grad is True
    assert a.abs().sum().item() == 0.0


def test_init_actions_with_shorter_init_pads():
    solver = _make_solver()
    _configure(solver, horizon=4)
    init = torch.ones(2, 2, ACT_DIM)  # only 2 of 4 steps provided
    a = solver._init_actions(init, B=2)
    assert a.shape == (2, 4, ACT_DIM)
    # First 2 steps from init_action, rest zero-padded
    assert torch.allclose(a[:, :2].detach(), torch.ones(2, 2, ACT_DIM))
    assert torch.allclose(a[:, 2:].detach(), torch.zeros(2, 2, ACT_DIM))


def test_init_actions_exact_length_no_padding():
    solver = _make_solver()
    _configure(solver, horizon=3)
    init = torch.ones(2, 3, ACT_DIM) * 0.5
    a = solver._init_actions(init, B=2)
    assert a.shape == (2, 3, ACT_DIM)
    assert torch.allclose(a.detach(), torch.ones(2, 3, ACT_DIM) * 0.5)


###############################################################################
# _compute_loss
###############################################################################


def test_compute_loss_is_scalar():
    solver = _make_solver()
    _configure(solver, horizon=3)
    B = 2
    emb_0 = torch.randn(B, EMB_DIM)
    goal_emb = torch.randn(B, EMB_DIM)
    vs = solver._init_virtual_states(emb_0, goal_emb)
    actions = solver._init_actions(None, B)
    loss = solver._compute_loss(vs, actions, emb_0, goal_emb, {})
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_compute_loss_is_differentiable():
    solver = _make_solver()
    _configure(solver, horizon=3)
    B = 2
    emb_0 = torch.randn(B, EMB_DIM)
    goal_emb = torch.randn(B, EMB_DIM)
    vs = solver._init_virtual_states(emb_0, goal_emb)
    actions = solver._init_actions(None, B)
    loss = solver._compute_loss(vs, actions, emb_0, goal_emb, {})
    loss.backward()
    # Gradients flow to virtual_states when horizon > 1
    assert vs.grad is not None or vs.numel() == 0


def test_compute_loss_respects_goal_weight():
    solver = _make_solver(goal_weight=0.0)
    _configure(solver, horizon=2)
    B = 1
    emb_0 = torch.zeros(B, EMB_DIM)
    goal_emb = torch.ones(B, EMB_DIM) * 10.0
    vs = solver._init_virtual_states(emb_0, goal_emb)
    actions = solver._init_actions(None, B)

    loss_no_goal = solver._compute_loss(vs, actions, emb_0, goal_emb, {}, goal_weight=0.0)
    loss_with_goal = solver._compute_loss(vs, actions, emb_0, goal_emb, {}, goal_weight=10.0)
    assert loss_with_goal.item() > loss_no_goal.item()


###############################################################################
# _compute_per_timestep_var
###############################################################################


def test_compute_per_timestep_var_shape_with_virtual():
    solver = _make_solver()
    _configure(solver, horizon=4)   # 3 virtual states
    B = 2
    emb_0 = torch.zeros(B, EMB_DIM)
    goal_emb = torch.ones(B, EMB_DIM)
    vs = solver._init_virtual_states(emb_0, goal_emb)
    var_t = solver._compute_per_timestep_var(vs, emb_0, goal_emb)
    assert var_t.shape == (B, 4, 1)   # horizon steps, 1 for broadcast


def test_compute_per_timestep_var_horizon_1():
    solver = _make_solver()
    _configure(solver, horizon=1)
    B = 2
    emb_0 = torch.zeros(B, EMB_DIM)
    goal_emb = torch.ones(B, EMB_DIM)
    vs = solver._init_virtual_states(emb_0, goal_emb)
    var_t = solver._compute_per_timestep_var(vs, emb_0, goal_emb)
    assert var_t.shape == (B, 1, 1)


def test_compute_per_timestep_var_all_positive():
    solver = _make_solver()
    _configure(solver, horizon=3)
    B = 2
    emb_0 = torch.zeros(B, EMB_DIM)
    goal_emb = torch.ones(B, EMB_DIM) * 5.0
    vs = solver._init_virtual_states(emb_0, goal_emb)
    var_t = solver._compute_per_timestep_var(vs, emb_0, goal_emb)
    assert (var_t > 0).all()


###############################################################################
# _expand_info
###############################################################################


def test_expand_info_tensor():
    info = {"obs": torch.zeros(2, 3, 4)}
    expanded = GRASPSolver._expand_info(info, num_samples=5)
    assert expanded["obs"].shape == (2, 5, 3, 4)


def test_expand_info_numpy():
    info = {"arr": np.zeros((2, 3))}
    expanded = GRASPSolver._expand_info(info, num_samples=4)
    assert expanded["arr"].shape == (2, 4, 3)


def test_expand_info_passthrough_non_tensor():
    info = {"scalar": 42, "text": "hello"}
    expanded = GRASPSolver._expand_info(info, num_samples=3)
    assert expanded["scalar"] == 42
    assert expanded["text"] == "hello"


###############################################################################
# _encode_observations
###############################################################################


def test_encode_observations_adds_emb_keys():
    solver = _make_solver()
    info = _basic_info_dict(B=2)
    result = solver._encode_observations(info)
    assert "emb" in result
    assert "goal_emb" in result


def test_encode_observations_emb_shape():
    solver = _make_solver()
    info = _basic_info_dict(B=3, T=2)
    result = solver._encode_observations(info)
    # Should be (B, D) — last time step kept
    assert result["emb"].shape == (3, EMB_DIM)
    assert result["goal_emb"].shape == (3, EMB_DIM)


def test_encode_observations_already_present_no_overwrite():
    solver = _make_solver()
    sentinel = torch.full((2, EMB_DIM), 99.0)
    info = _basic_info_dict(B=2)
    info["emb"] = sentinel
    info["goal_emb"] = sentinel.clone()
    result = solver._encode_observations(info)
    assert torch.allclose(result["emb"], sentinel)
    assert torch.allclose(result["goal_emb"], sentinel)


def test_encode_observations_goal_4d_pixels():
    """Goal pixels without time dim (B, C, H, W) should be handled."""
    solver = _make_solver()
    info = {
        "pixels": torch.randn(2, 1, 3, 8, 8),
        "goal": torch.randn(2, 3, 8, 8),   # 4-D, no time dim
    }
    result = solver._encode_observations(info)
    assert result["goal_emb"].shape == (2, EMB_DIM)


###############################################################################
# _gd_sync
###############################################################################


def test_gd_sync_returns_correct_shape():
    solver = _make_solver(gd_opt_steps=2, gd_lr=0.01)
    _configure(solver, horizon=3)
    B = 2
    actions = torch.zeros(B, 3, ACT_DIM)
    info = _basic_info_dict(B=B)
    result = solver._gd_sync(actions, info)
    assert result.shape == (B, 3, ACT_DIM)
    assert result.requires_grad is False


def test_gd_sync_detaches_from_graph():
    solver = _make_solver(gd_opt_steps=1)
    _configure(solver, horizon=2)
    B = 1
    actions = torch.zeros(B, 2, ACT_DIM, requires_grad=True)
    info = _basic_info_dict(B=B)
    result = solver._gd_sync(actions, info)
    assert not result.requires_grad


###############################################################################
# _cem_sync
###############################################################################


def test_cem_sync_returns_correct_shape():
    solver = _make_solver(
        cem_sync_samples=8, cem_sync_topk=4, gd_opt_steps=2
    )
    _configure(solver, horizon=3)
    B = 2
    emb_0 = torch.zeros(B, EMB_DIM)
    goal_emb = torch.ones(B, EMB_DIM)
    actions = torch.zeros(B, 3, ACT_DIM)
    vs = solver._init_virtual_states(emb_0, goal_emb)
    info = _basic_info_dict(B=B)
    result = solver._cem_sync(actions, vs, emb_0, goal_emb, info)
    assert result.shape == (B, 3, ACT_DIM)


###############################################################################
# solve — output format
###############################################################################


def test_solve_output_keys():
    solver = _make_solver(n_steps=2, gd_interval=0)
    _configure(solver, horizon=3)
    info = _basic_info_dict(B=2)
    out = solver.solve(info)
    assert "actions" in out
    assert "virtual_states" in out
    assert "loss_history" in out


def test_solve_actions_shape():
    solver = _make_solver(n_steps=2, gd_interval=0)
    _configure(solver, horizon=3, n_envs=2)
    info = _basic_info_dict(B=2)
    out = solver.solve(info)
    assert out["actions"].shape == (2, 3, ACT_DIM)


def test_solve_virtual_states_shape():
    solver = _make_solver(n_steps=2, gd_interval=0)
    _configure(solver, horizon=4, n_envs=2)
    info = _basic_info_dict(B=2)
    out = solver.solve(info)
    # horizon=4 → 3 virtual states
    assert out["virtual_states"].shape == (2, 3, EMB_DIM)


def test_solve_loss_history_length():
    solver = _make_solver(n_steps=5, gd_interval=0)
    _configure(solver, horizon=3, n_envs=2)
    info = _basic_info_dict(B=2)
    out = solver.solve(info)
    # Single batch → one inner list with n_steps entries
    assert len(out["loss_history"]) == 1
    assert len(out["loss_history"][0]) == 5


def test_solve_returns_cpu_tensors():
    solver = _make_solver(n_steps=2, gd_interval=0, device="cpu")
    _configure(solver, horizon=3, n_envs=2)
    info = _basic_info_dict(B=2)
    out = solver.solve(info)
    assert out["actions"].device.type == "cpu"
    assert out["virtual_states"].device.type == "cpu"


###############################################################################
# solve — batch_size splitting
###############################################################################


def test_solve_batch_splitting():
    """batch_size=1 should split 4 environments into 4 batches."""
    solver = _make_solver(n_steps=2, gd_interval=0, batch_size=1)
    _configure(solver, horizon=3, n_envs=4)
    info = _basic_info_dict(B=4)
    out = solver.solve(info)
    assert out["actions"].shape == (4, 3, ACT_DIM)
    assert len(out["loss_history"]) == 4


def test_solve_batch_size_larger_than_envs():
    solver = _make_solver(n_steps=2, gd_interval=0, batch_size=100)
    _configure(solver, horizon=3, n_envs=2)
    info = _basic_info_dict(B=2)
    out = solver.solve(info)
    assert out["actions"].shape == (2, 3, ACT_DIM)


###############################################################################
# solve — init_action warm-start
###############################################################################


def test_solve_with_init_action():
    solver = _make_solver(n_steps=2, gd_interval=0)
    _configure(solver, horizon=3, n_envs=2)
    info = _basic_info_dict(B=2)
    init_action = torch.ones(2, 2, ACT_DIM) * 0.5  # provide 2 of 3 steps
    out = solver.solve(info, init_action=init_action)
    assert out["actions"].shape == (2, 3, ACT_DIM)


###############################################################################
# solve — horizon=1 (no virtual states)
###############################################################################


def test_solve_horizon_1():
    solver = _make_solver(n_steps=2, gd_interval=0)
    _configure(solver, horizon=1, n_envs=2)
    info = _basic_info_dict(B=2)
    out = solver.solve(info)
    assert out["actions"].shape == (2, 1, ACT_DIM)
    assert out["virtual_states"].shape == (2, 0, EMB_DIM)


###############################################################################
# solve — periodic sync (gd_interval)
###############################################################################


def test_solve_with_gd_sync():
    solver = _make_solver(n_steps=4, gd_interval=2, gd_opt_steps=1)
    _configure(solver, horizon=3, n_envs=2)
    info = _basic_info_dict(B=2)
    out = solver.solve(info)
    assert out["actions"].shape == (2, 3, ACT_DIM)


def test_solve_with_cem_sync():
    solver = _make_solver(
        n_steps=4, gd_interval=2, gd_opt_steps=1,
        sync_mode="cem", cem_sync_samples=8, cem_sync_topk=4
    )
    _configure(solver, horizon=3, n_envs=2)
    info = _basic_info_dict(B=2)
    out = solver.solve(info)
    assert out["actions"].shape == (2, 3, ACT_DIM)


###############################################################################
# solve — schedule_decay
###############################################################################


def test_solve_schedule_decay():
    solver = _make_solver(
        n_steps=6,
        gd_interval=3,
        schedule_decay=True,
        init_noise_scale=0.5,
        min_noise_scale=0.0,
        init_goal_weight=3.0,
        min_goal_weight=1.0,
    )
    _configure(solver, horizon=3, n_envs=1)
    info = _basic_info_dict(B=1)
    out = solver.solve(info)
    assert out["actions"].shape == (1, 3, ACT_DIM)


###############################################################################
# solve — pre-encoded observations
###############################################################################


def test_solve_with_preencoded_embeddings():
    """If emb/goal_emb already in info_dict, encode is skipped."""
    solver = _make_solver(n_steps=2, gd_interval=0)
    _configure(solver, horizon=3, n_envs=2)
    B = 2
    info = {
        "emb": torch.zeros(B, EMB_DIM),
        "goal_emb": torch.ones(B, EMB_DIM),
    }
    out = solver.solve(info)
    assert out["actions"].shape == (B, 3, ACT_DIM)


###############################################################################
# __call__ delegation
###############################################################################


def test_call_delegates_to_solve():
    solver = _make_solver(n_steps=2, gd_interval=0)
    _configure(solver, horizon=3, n_envs=2)
    info = _basic_info_dict(B=2)
    out_call = solver(info)
    assert "actions" in out_call
    assert out_call["actions"].shape == (2, 3, ACT_DIM)


###############################################################################
# Model with 2-D (non-temporal) embeddings
###############################################################################


def test_solve_2d_embed_model():
    """Model returning (B, D) embeddings (no time dim) should work."""
    solver = GRASPSolver(
        model=DummyGRASPModelNDEmb(),
        n_steps=2,
        gd_interval=0,
        device="cpu",
        seed=0,
    )
    _configure(solver, horizon=3, n_envs=2)
    info = {
        "pixels": torch.randn(2, 1, 3, 8, 8),
        "goal": torch.randn(2, 3, 8, 8),
    }
    out = solver.solve(info)
    assert out["actions"].shape == (2, 3, ACT_DIM)


###############################################################################
# Reproducibility (seeded generator)
###############################################################################


def test_solve_deterministic_with_same_seed():
    def _run(seed):
        solver = _make_solver(
            n_steps=3, gd_interval=0, state_noise_scale=0.1, seed=seed
        )
        _configure(solver, horizon=3, n_envs=1)
        info = {
            "emb": torch.zeros(1, EMB_DIM),
            "goal_emb": torch.ones(1, EMB_DIM),
        }
        torch.manual_seed(seed)
        return solver.solve(info)["actions"]

    a1 = _run(77)
    a2 = _run(77)
    assert torch.allclose(a1, a2)
