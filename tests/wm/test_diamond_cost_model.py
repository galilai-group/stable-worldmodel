"""Tests for DiamondCostModel (cost model for solver integration)."""

import pytest
import torch

from stable_worldmodel.wm.diamond.module import (
    RewardTermModel,
    DiscreteActionEncoder,
    EDMModel,
)
from stable_worldmodel.wm.diamond.diamond import DiamondCostModel


def make_models(
    in_ch=3,
    base_ch=8,
    cond_dim=16,
    lstm_dim=16,
    action_dim=4,
    history_size=4,
    img_size=16,
):
    diffusion = EDMModel(
        in_ch=in_ch,
        base_ch=base_ch,
        cond_dim=cond_dim,
        history_frames=history_size,
    )
    reward_term = RewardTermModel(
        in_channels=in_ch,
        action_dim=action_dim,
        cond_dim=cond_dim,
        lstm_dim=lstm_dim,
    )
    action_encoder = DiscreteActionEncoder(
        num_actions=action_dim, emb_dim=cond_dim
    )
    cost_model = DiamondCostModel(
        diffusion=diffusion,
        reward_term=reward_term,
        action_encoder=action_encoder,
        history_size=history_size,
        n_denoise=2,
        device='cpu',
    )
    return cost_model


class TestDiamondCostModelInit:
    def test_init_stores_attributes(self):
        cost_model = make_models()
        assert cost_model.history_size == 4
        assert cost_model.n_denoise == 2
        assert cost_model.device == 'cpu'

    def test_criterion_delegates(self):
        """``criterion`` delegates to ``get_cost`` (satisfies Costable protocol)."""
        cost_model = make_models(img_size=16)
        B, N, L, C, H, W = 1, 4, 4, 3, 16, 16
        info_dict = {
            'pixels': torch.randn(B, N, L, C, H, W),
            'hx_rw': torch.randn(B, N, 16),
            'cx_rw': torch.randn(B, N, 16),
        }
        candidates = torch.randn(B, N, 5, 4)
        result = cost_model.criterion(info_dict, candidates)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (B, N)


class TestDiamondCostModelWarmup:
    def test_warmup_returns_correct_shapes(self):
        cost_model = make_models(img_size=16)
        B, L, C, H, W = 2, 4, 3, 16, 16
        obs_buf = torch.randn(B, L, C, H, W)
        act_buf = torch.randint(0, 4, (B, L))

        hx, cx = cost_model.warmup(obs_buf, act_buf)

        assert hx.shape == (B, 16)
        assert cx.shape == (B, 16)

    def test_warmup_batch_size_1(self):
        cost_model = make_models(img_size=16)
        B, L, C, H, W = 1, 4, 3, 16, 16
        obs_buf = torch.randn(B, L, C, H, W)
        act_buf = torch.randint(0, 4, (B, L))

        hx, cx = cost_model.warmup(obs_buf, act_buf)

        assert hx.shape == (1, 16)
        assert cx.shape == (1, 16)

    def test_warmup_different_history_size(self):
        cost_model = make_models(history_size=2, img_size=16)
        B, L, C, H, W = 1, 2, 3, 16, 16
        obs_buf = torch.randn(B, L, C, H, W)
        act_buf = torch.randint(0, 4, (B, L))

        hx, cx = cost_model.warmup(obs_buf, act_buf)

        assert hx.shape == (1, 16)
        assert cx.shape == (1, 16)


class TestDiamondCostModelGetCost:
    def test_get_cost_returns_correct_shape(self):
        cost_model = make_models(img_size=16)
        B, N, L, C, H, W = 2, 8, 4, 3, 16, 16
        H_plan = 5
        K = 4

        pixels = torch.randn(B, N, L, C, H, W)
        hx_rw = torch.randn(B, N, 16)
        cx_rw = torch.randn(B, N, 16)
        info_dict = {'pixels': pixels, 'hx_rw': hx_rw, 'cx_rw': cx_rw}

        candidates = torch.randn(B, N, H_plan, K)

        cost = cost_model.get_cost(info_dict, candidates)

        assert cost.shape == (B, N)
        assert cost.dtype == torch.float32

    def test_get_cost_batch_size_1(self):
        cost_model = make_models(img_size=16)
        B, N, L, C, H, W = 1, 4, 4, 3, 16, 16
        H_plan = 3
        K = 4

        pixels = torch.randn(B, N, L, C, H, W)
        hx_rw = torch.randn(B, N, 16)
        cx_rw = torch.randn(B, N, 16)
        info_dict = {'pixels': pixels, 'hx_rw': hx_rw, 'cx_rw': cx_rw}

        candidates = torch.randn(B, N, H_plan, K)

        cost = cost_model.get_cost(info_dict, candidates)

        assert cost.shape == (B, N)

    def test_get_cost_one_hot_actions(self):
        """One-hot action candidates should produce valid costs."""
        cost_model = make_models(img_size=16)
        B, N, L, C, H, W = 1, 4, 4, 3, 16, 16
        H_plan = 2
        K = 4

        pixels = torch.randn(B, N, L, C, H, W)
        hx_rw = torch.randn(B, N, 16)
        cx_rw = torch.randn(B, N, 16)
        info_dict = {'pixels': pixels, 'hx_rw': hx_rw, 'cx_rw': cx_rw}

        candidates = (
            torch.eye(K)
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(B, N, H_plan, K)
            .clone()
        )

        cost = cost_model.get_cost(info_dict, candidates)

        assert cost.shape == (B, N)
        assert not torch.isnan(cost).any()
        assert not torch.isinf(cost).any()

    def test_get_cost_no_nan(self):
        """Costs should be finite (not NaN or Inf)."""
        torch.manual_seed(0)
        cost_model = make_models(img_size=16)
        B, N, L, C, H, W = 1, 4, 4, 3, 16, 16
        H_plan = 2
        K = 4

        pixels = torch.randn(B, N, L, C, H, W)
        hx_rw = torch.randn(B, N, 16)
        cx_rw = torch.randn(B, N, 16)
        info_dict = {'pixels': pixels, 'hx_rw': hx_rw, 'cx_rw': cx_rw}
        candidates = torch.rand(B, N, H_plan, K)

        cost = cost_model.get_cost(info_dict, candidates)

        assert not torch.isnan(cost).any()
        assert not torch.isinf(cost).any()


class TestDiamondCostModelSolverIntegration:
    def test_solver_integration_shape(self):
        """End-to-end: info_dict from solver expansion should work."""
        import gymnasium as gym
        from stable_worldmodel.policy import PlanConfig
        from stable_worldmodel.solver.categorical_cem import (
            CategoricalCEMSolver,
        )

        cost_model = make_models(img_size=16)

        solver = CategoricalCEMSolver(
            model=cost_model,
            num_samples=4,
            n_steps=2,
            topk=2,
            device='cpu',
            seed=42,
        )
        config = PlanConfig(
            horizon=3, receding_horizon=1, history_len=4, action_block=1
        )
        solver.configure(
            action_space=gym.spaces.Discrete(4), n_envs=1, config=config
        )

        info_dict = {
            'pixels': torch.randn(1, 4, 3, 16, 16),
            'hx_rw': torch.randn(1, 16),
            'cx_rw': torch.randn(1, 16),
        }

        out = solver(info_dict)

        assert 'actions' in out
        assert out['actions'].shape == (1, 3, 1)
