"""Tests for DiamondAgent — imagination rollout and RL losses."""

import pytest
import torch

from stable_worldmodel.wm.diamond.module import (
    ActorCritic,
    RewardTermModel,
    DiscreteActionEncoder,
    EDMModel,
)
from stable_worldmodel.wm.diamond.diamond import (
    DiamondAgent,
    compute_lambda_returns,
    compute_rl_losses,
)


def make_agent(
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
    actor_critic = ActorCritic(
        in_channels=in_ch,
        action_dim=action_dim,
        lstm_dim=lstm_dim,
    )
    action_encoder = DiscreteActionEncoder(
        num_actions=action_dim, emb_dim=cond_dim
    )
    agent = DiamondAgent(
        diffusion_model=diffusion,
        reward_term_model=reward_term,
        actor_critic=actor_critic,
        action_encoder=action_encoder,
        history_size=history_size,
        device='cpu',
    )
    return agent


class TestImaginationRollout:
    def test_rollout_returns_expected_keys(self):
        agent = make_agent(img_size=16)
        B, L, C, H, W = 2, 4, 3, 16, 16
        init_obs = torch.randn(B, L, C, H, W)
        init_actions = torch.randint(0, 4, (B, L))

        traj = agent.imagination_rollout(
            init_obs, init_actions, H=5, n_denoise=2
        )

        expected_keys = {
            'frames',
            'actions',
            'rewards',
            'terminals',
            'values',
            'log_probs',
            'entropies',
        }
        assert set(traj.keys()) == expected_keys

    def test_rollout_shapes(self):
        agent = make_agent(img_size=16)
        B, L, C, H_im, W_im = 2, 4, 3, 16, 16
        H = 5
        init_obs = torch.randn(B, L, C, H_im, W_im)
        init_actions = torch.randint(0, 4, (B, L))

        traj = agent.imagination_rollout(
            init_obs, init_actions, H=H, n_denoise=2
        )

        assert traj['frames'].shape == (B, H + 1, C, H_im, W_im)
        assert traj['actions'].shape == (B, H, 1)
        assert traj['rewards'].shape == (B, H)
        assert traj['terminals'].shape == (B, H)
        assert traj['values'].shape == (B, H + 1)
        assert traj['log_probs'].shape == (B, H)
        assert traj['entropies'].shape == (B, H)

    def test_rollout_batch_size_1(self):
        agent = make_agent(img_size=16)
        B, L, C, H_im, W_im = 1, 4, 3, 16, 16
        H = 3
        init_obs = torch.randn(B, L, C, H_im, W_im)
        init_actions = torch.randint(0, 4, (B, L))

        traj = agent.imagination_rollout(
            init_obs, init_actions, H=H, n_denoise=2
        )

        assert traj['frames'].shape == (1, H + 1, C, H_im, W_im)
        assert traj['values'].shape == (1, H + 1)
        assert traj['rewards'].shape == (1, H)

    def test_rollout_different_history_size(self):
        agent = make_agent(history_size=2, img_size=16)
        B, L, C, H_im, W_im = 2, 2, 3, 16, 16
        H = 4
        init_obs = torch.randn(B, L, C, H_im, W_im)
        init_actions = torch.randint(0, 4, (B, L))

        traj = agent.imagination_rollout(
            init_obs, init_actions, H=H, n_denoise=2
        )

        assert traj['frames'].shape == (B, H + 1, C, H_im, W_im)
        assert traj['values'].shape == (B, H + 1)


class TestComputeLambdaReturns:
    def test_basic_shapes(self):
        B, H = 4, 5
        rewards = torch.randn(B, H)
        values = torch.randn(B, H + 1)
        terminated = torch.zeros(B, H)

        ret = compute_lambda_returns(rewards, values, terminated)
        assert ret.shape == (B, H)

    def test_terminal_stops_bootstrap(self):
        B, H = 2, 3
        rewards = torch.ones(B, H)
        values = torch.zeros(B, H + 1)
        terminated = torch.zeros(B, H)
        terminated[0, 1] = 1.0

        ret = compute_lambda_returns(
            rewards, values, terminated, gamma=0.9, lam=0.5
        )
        # After termination, bootstrap should be 0 since value is 0
        assert ret.shape == (B, H)

    def test_horizon_1(self):
        B, H = 2, 1
        rewards = torch.randn(B, H)
        values = torch.randn(B, H + 1)
        terminated = torch.zeros(B, H)

        ret = compute_lambda_returns(rewards, values, terminated)
        assert ret.shape == (B, H)


class TestComputeRLLosses:
    def test_losses_are_scalars(self):
        B, H = 4, 5
        trajectory = {
            'rewards': torch.randn(B, H),
            'values': torch.randn(B, H + 1),
            'terminals': torch.zeros(B, H),
            'log_probs': torch.randn(B, H),
            'entropies': torch.rand(B, H),
        }

        losses = compute_rl_losses(trajectory)

        assert 'rl_loss' in losses
        assert losses['rl_loss'].ndim == 0
        assert losses['rl_loss'].item() >= float('-inf')

    def test_value_loss_is_positive(self):
        B, H = 2, 3
        trajectory = {
            'rewards': torch.randn(B, H),
            'values': torch.randn(B, H + 1),
            'terminals': torch.zeros(B, H),
            'log_probs': torch.randn(B, H),
            'entropies': torch.rand(B, H),
        }

        losses = compute_rl_losses(trajectory)

        assert losses['value_loss'].item() >= 0

    def test_all_loss_keys_present(self):
        B, H = 2, 4
        trajectory = {
            'rewards': torch.randn(B, H),
            'values': torch.randn(B, H + 1),
            'terminals': torch.zeros(B, H),
            'log_probs': torch.randn(B, H),
            'entropies': torch.rand(B, H),
        }

        losses = compute_rl_losses(trajectory)

        expected_keys = {'policy_loss', 'value_loss', 'entropy', 'rl_loss'}
        assert set(losses.keys()) == expected_keys


class TestDiamondAgentTrainSteps:
    def test_train_diffusion_step(self):
        agent = make_agent(img_size=16)
        B, C, H, W = 2, 3, 16, 16
        batch = {
            'next_frame': torch.randn(B, C, H, W),
            'history': torch.randn(B, C * 4, H, W),
            'cond_vec': torch.randn(B, 16),
        }
        loss = agent.train_diffusion_step(batch)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_train_reward_term_step(self):
        agent = make_agent(img_size=16)
        B, T, C, H, W = 2, 5, 3, 16, 16
        batch = {
            'frames': torch.randn(B, T, C, H, W),
            'actions': torch.randint(0, 4, (B, T, 1)).float(),
            'rewards': torch.randn(B, T),
            'terminals': torch.randint(0, 2, (B, T)).float(),
        }
        loss = agent.train_reward_term_step(batch)
        assert loss.ndim == 0
        assert loss.item() >= 0
