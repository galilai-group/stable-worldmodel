import torch
from torch import nn
import torch.nn.functional as F
from .unet import ResidualBlock


class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        block_channels=(32, 32, 64, 64),
        block_layers=(1, 1, 1, 1),
        cond_dim=None,
        num_groups=8,
    ):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, block_channels[0], 3, padding=1)

        stages = []
        prev_ch = block_channels[0]
        for out_ch, n_layers in zip(block_channels, block_layers):
            stage_blocks = []
            for _ in range(n_layers):
                stage_blocks.append(
                    ResidualBlock(prev_ch, out_ch, cond_dim, num_groups)
                )
                prev_ch = out_ch
            stages.append(nn.ModuleList(stage_blocks))
        self.stages = nn.ModuleList(stages)
        self.pool = nn.MaxPool2d(2)
        self.out_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, cond=None):
        h = self.in_conv(x)
        for stage in self.stages:
            for block in stage:
                h = block(h, cond)
            h = self.pool(h)
        h = self.out_pool(h)
        return h.flatten(1)


class RewardTermModel(nn.Module):
    def __init__(
        self,
        in_channels=3,
        action_dim=18,
        cond_dim=128,
        lstm_dim=512,
        num_groups=8,
    ):
        super().__init__()
        self.action_embed = nn.Embedding(action_dim, cond_dim)
        block_channels = (32, 32, 32, 32)
        self.encoder = ConvEncoder(
            in_channels=in_channels,
            block_channels=block_channels,
            block_layers=(2, 2, 2, 2),
            cond_dim=cond_dim,
            num_groups=num_groups,
        )
        enc_dim = block_channels[-1]
        self.lstm = nn.LSTMCell(enc_dim + cond_dim, lstm_dim)
        self.reward_head = nn.Linear(lstm_dim, 1)
        self.term_head = nn.Linear(lstm_dim, 1)

    def forward(self, frames, actions, state=None):
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        actions = actions.view(B * T, -1)
        act_emb = self.action_embed(actions.squeeze(-1).long())
        vis_emb = self.encoder(frames, act_emb)
        x = torch.cat([vis_emb, act_emb], dim=-1)

        hx = torch.zeros(B, self.lstm.hidden_size, device=frames.device)
        cx = torch.zeros(B, self.lstm.hidden_size, device=frames.device)
        if state is not None:
            hx, cx = state

        rewards = []
        terminals = []
        for t in range(T):
            inp = x[t::T] if T > 1 else x
            hx, cx = self.lstm(inp, (hx, cx))
            rewards.append(self.reward_head(hx))
            terminals.append(self.term_head(hx))

        rew = torch.stack(rewards, dim=1)
        term = torch.stack(terminals, dim=1)
        return rew, term, (hx, cx)


class ActorCritic(nn.Module):
    def __init__(
        self,
        in_channels=3,
        action_dim=18,
        lstm_dim=512,
        num_groups=8,
    ):
        super().__init__()
        self.action_dim = action_dim
        block_channels = (32, 32, 64, 64)
        self.encoder = ConvEncoder(
            in_channels=in_channels,
            block_channels=block_channels,
            block_layers=(1, 1, 1, 1),
            cond_dim=None,
            num_groups=num_groups,
        )
        enc_dim = block_channels[-1]
        self.lstm = nn.LSTMCell(enc_dim, lstm_dim)
        self.policy_head = nn.Linear(lstm_dim, action_dim)
        self.value_head = nn.Linear(lstm_dim, 1)

    def forward(self, frame, state=None):
        B, C, H, W = frame.shape
        vis_emb = self.encoder(frame)
        hx = torch.zeros(B, self.lstm.hidden_size, device=frame.device)
        cx = torch.zeros(B, self.lstm.hidden_size, device=frame.device)
        if state is not None:
            hx, cx = state
        hx, cx = self.lstm(vis_emb, (hx, cx))
        logits = self.policy_head(hx)
        value = self.value_head(hx)
        return logits, value, (hx, cx)

    def get_action(self, frame, state=None, deterministic=False):
        logits, value, state = self.forward(frame, state)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        return action, value, state, logits


__all__ = ['ConvEncoder', 'RewardTermModel', 'ActorCritic']
