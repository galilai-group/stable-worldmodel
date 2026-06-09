import torch
import torch.nn.functional as F
from torch import nn

from .module import (
    EDMModel,
    sample_euler,
    edm_loss_step,
    RewardTermModel,
    ActorCritic,
    DiscreteActionEncoder,
)


__all__ = [
    'DiamondCostModel',
    'DiamondAgent',
    'compute_lambda_returns',
    'compute_rl_losses',
]


def compute_lambda_returns(rewards, values, terminated, gamma=0.985, lam=0.95):
    B, H = rewards.shape
    returns = torch.zeros_like(rewards)
    g = torch.full((B,), gamma, device=rewards.device)
    for t in reversed(range(H)):
        if t == H - 1:
            returns[:, t] = values[:, t + 1]
        else:
            bootstrap = (1 - lam) * values[:, t + 1] + lam * returns[:, t + 1]
            returns[:, t] = (
                rewards[:, t] + g * (1 - terminated[:, t]) * bootstrap
            )
    return returns


def compute_rl_losses(trajectory, gamma=0.985, lam=0.95, eta=0.001):
    rewards = trajectory['rewards']
    values = trajectory['values']
    terminated = trajectory['terminals']
    log_probs = trajectory['log_probs']
    entropies = trajectory['entropies']

    lambda_returns = compute_lambda_returns(
        rewards, values, terminated, gamma, lam
    )
    lambda_returns = lambda_returns.detach()

    value_loss = F.mse_loss(
        values[:, :-1], lambda_returns, reduction='none'
    ).mean()

    advantages = lambda_returns - values[:, :-1].detach()
    policy_loss = -(log_probs * advantages).mean()

    entropy_bonus = entropies.mean()

    total_loss = policy_loss + value_loss - eta * entropy_bonus
    losses = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'entropy': entropy_bonus,
        'rl_loss': total_loss,
    }
    return losses


class DiamondCostModel(nn.Module):
    def __init__(
        self,
        diffusion,
        reward_term,
        action_encoder,
        history_size=4,
        n_denoise=3,
        device='cpu',
    ):
        super().__init__()
        self.diffusion = diffusion
        self.reward_term = reward_term
        self.action_encoder = action_encoder
        self.history_size = history_size
        self.n_denoise = n_denoise
        self.device = device

    @torch.no_grad()
    def warmup(self, obs_buf, act_buf):
        L = self.history_size
        B = obs_buf.shape[0]
        hx = torch.zeros(
            B, self.reward_term.lstm.hidden_size, device=self.device
        )
        cx = torch.zeros(
            B, self.reward_term.lstm.hidden_size, device=self.device
        )
        for t in range(L):
            frame = obs_buf[:, t]
            act = act_buf[:, t]
            act_emb = self.reward_term.action_embed(act.long())
            vis_emb = self.reward_term.encoder(frame, act_emb)
            inp = torch.cat([vis_emb, act_emb], dim=-1)
            hx, cx = self.reward_term.lstm(inp, (hx, cx))
        return hx, cx

    @torch.no_grad()
    def get_cost(self, info_dict, action_candidates):
        pixels = info_dict['pixels']
        hx_rw = info_dict['hx_rw']
        cx_rw = info_dict['cx_rw']
        B, N, L, C, H_im, W_im = pixels.shape
        H_plan = action_candidates.shape[2]
        K = action_candidates.shape[-1]

        BN = B * N

        obs = pixels.reshape(BN, L, C, H_im, W_im)
        hx = hx_rw.reshape(BN, -1)
        cx = cx_rw.reshape(BN, -1)

        act = torch.zeros(BN, L, dtype=torch.long, device=self.device)

        total_rewards = torch.zeros(BN, device=self.device)

        for t in range(H_plan):
            a_onehot = action_candidates[:, :, t]
            a_idx = a_onehot.argmax(dim=-1)
            a_t = a_idx.reshape(BN)

            new_act = torch.cat([act[:, 1:], a_t.unsqueeze(-1)], dim=1)
            act_emb = self.action_encoder(new_act)
            cond_vec = act_emb.mean(dim=1)

            history_flat = obs.reshape(BN, C * L, H_im, W_im)
            cond = {'history': history_flat, 'cond_vec': cond_vec}
            frame = sample_euler(
                self.diffusion,
                cond,
                (BN, C, H_im, W_im),
                self.device,
                n_steps=self.n_denoise,
            )

            a_emb = self.reward_term.action_embed(a_t)
            vis_emb = self.reward_term.encoder(frame, a_emb)
            inp = torch.cat([vis_emb, a_emb], dim=-1)
            hx, cx = self.reward_term.lstm(inp, (hx, cx))
            reward = self.reward_term.reward_head(hx).squeeze(-1)
            total_rewards += reward

            obs = torch.cat([obs[:, 1:], frame.unsqueeze(1)], dim=1)
            act = new_act

        cost = -total_rewards.view(B, N)
        return cost

    def criterion(self, info_dict, action_candidates):
        return self.get_cost(info_dict, action_candidates)


class DiamondAgent(nn.Module):
    def __init__(
        self,
        diffusion_model,
        reward_term_model,
        actor_critic,
        action_encoder,
        history_size=4,
        device='cpu',
    ):
        super().__init__()
        self.diffusion = diffusion_model
        self.reward_term = reward_term_model
        self.actor_critic = actor_critic
        self.action_encoder = action_encoder
        self.history_size = history_size
        self.device = device

    def encode_actions(self, actions):
        act_emb = self.action_encoder(actions.unsqueeze(-1))
        return act_emb.mean(dim=1)

    def imagination_rollout(self, init_obs, init_actions, H=15, n_denoise=3):
        B, L, C, H_im, W_im = init_obs.shape
        obs_buf = init_obs.clone()
        act_buf = init_actions.clone()
        hx_ac = torch.zeros(
            B, self.actor_critic.lstm.hidden_size, device=self.device
        )
        cx_ac = torch.zeros(
            B, self.actor_critic.lstm.hidden_size, device=self.device
        )
        hx_rw = torch.zeros(
            B, self.reward_term.lstm.hidden_size, device=self.device
        )
        cx_rw = torch.zeros(
            B, self.reward_term.lstm.hidden_size, device=self.device
        )

        for t in range(L):
            frame = obs_buf[:, t]
            act = init_actions[:, t]
            logits, val, (hx_ac, cx_ac) = self.actor_critic.forward(
                frame, (hx_ac, cx_ac)
            )
            with torch.no_grad():
                act_emb = self.reward_term.action_embed(act.long())
                vis_emb = self.reward_term.encoder(frame, act_emb)
                inp = torch.cat([vis_emb, act_emb], dim=-1)
                hx_rw, cx_rw = self.reward_term.lstm(inp, (hx_rw, cx_rw))

        frames = []
        actions = []
        rewards = []
        terminals = []
        values = []
        log_probs = []
        entropies = []

        frame = obs_buf[:, -1]
        for t in range(H):
            logits, value, (hx_ac, cx_ac) = self.actor_critic.forward(
                frame, (hx_ac, cx_ac)
            )
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            lp = dist.log_prob(action)
            ent = dist.entropy()

            act_onehot = action.unsqueeze(-1)
            with torch.no_grad():
                act_emb = self.reward_term.action_embed(action)
                vis_emb = self.reward_term.encoder(frame, act_emb)
                inp = torch.cat([vis_emb, act_emb], dim=-1)
                hx_rw, cx_rw = self.reward_term.lstm(inp, (hx_rw, cx_rw))
                reward = self.reward_term.reward_head(hx_rw)
                terminal = torch.sigmoid(self.reward_term.term_head(hx_rw))

            frames.append(frame)
            actions.append(act_onehot)
            rewards.append(reward)
            terminals.append(terminal)
            values.append(value)
            log_probs.append(lp.unsqueeze(-1))
            entropies.append(ent.unsqueeze(-1))

            obs_buf = torch.cat([obs_buf[:, 1:], frame.unsqueeze(1)], dim=1)
            act_buf = torch.cat([act_buf[:, 1:], act_onehot], dim=1)
            cond_vec = self.encode_actions(act_buf)
            history = obs_buf.reshape(B, C * L, H_im, W_im)
            cond = {'history': history, 'cond_vec': cond_vec}
            with torch.no_grad():
                frame = sample_euler(
                    self.diffusion,
                    cond,
                    (B, C, H_im, W_im),
                    self.device,
                    n_steps=n_denoise,
                )

        frames.append(frame)
        logits, value, _ = self.actor_critic.forward(frame, (hx_ac, cx_ac))
        values.append(value)

        trajectory = {
            'frames': torch.stack(frames, dim=1),
            'actions': torch.stack(actions, dim=1),
            'rewards': torch.stack(rewards, dim=1).squeeze(-1),
            'terminals': torch.stack(terminals, dim=1).squeeze(-1),
            'values': torch.stack(values, dim=1).squeeze(-1),
            'log_probs': torch.stack(log_probs, dim=1).squeeze(-1),
            'entropies': torch.stack(entropies, dim=1).squeeze(-1),
        }
        return trajectory

    def train_diffusion_step(self, batch):
        loss = edm_loss_step(self.diffusion, batch, self.device)
        return loss

    def train_reward_term_step(self, batch):
        frames = batch['frames'].to(self.device)
        actions = batch['actions'].to(self.device)
        target_rewards = batch['rewards'].to(self.device).squeeze(-1)
        target_terminals = batch['terminals'].to(self.device).squeeze(-1)
        pred_rewards, pred_terminals, _ = self.reward_term(frames, actions)
        rew_loss = F.mse_loss(pred_rewards.squeeze(-1), target_rewards)
        term_loss = F.binary_cross_entropy_with_logits(
            pred_terminals.squeeze(-1), target_terminals
        )
        return rew_loss + term_loss

    def collect_experience(self, env, steps=100, epsilon=0.01):
        obs, _ = env.reset()
        obs_buf = [obs]
        act_buf = []
        rew_buf = []
        term_buf = []
        for _ in range(steps):
            frame = (
                torch.tensor(obs, dtype=torch.float32, device=self.device)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            if torch.rand(1).item() < epsilon:
                action = env.action_space.sample()
            else:
                logits, _, _ = self.actor_critic(frame / 255.0)
                action = logits.argmax(dim=-1).item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            reward = torch.tensor(
                [max(-1, min(1, reward))], device=self.device
            )
            term = torch.tensor(
                [float(terminated or truncated)], device=self.device
            )
            obs_buf.append(next_obs)
            act_buf.append(torch.tensor([action], device=self.device))
            rew_buf.append(reward)
            term_buf.append(term)
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()
        return {
            'obs': torch.stack(
                [
                    torch.tensor(
                        o, dtype=torch.float32, device=self.device
                    ).permute(2, 0, 1)
                    for o in obs_buf
                ]
            ),
            'actions': torch.stack(act_buf),
            'rewards': torch.stack(rew_buf),
            'terminals': torch.stack(term_buf),
        }
