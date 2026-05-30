import torch
from torch import nn
import torch.nn.functional as F


class DiffusionPredictor(nn.Module):
    """A lightweight embedding-space diffusion-like predictor.

    This module is designed to act as the `predictor` argument for the
    `Diamond` wrapper. It operates in embedding space (not pixel space)
    and provides a simple denoising-style training target: given current
    embeddings and action embeddings, predict the next-step embedding.

    Notes:
    - This is not a full image-space EDM U-Net. It is intentionally small
      and serves as an integration-friendly predictor that matches the
      predictor API expected by LeWM/PLDM/Diamond.
    - Later we can replace this with a full image-space diffusion model
      (EDM U-Net) and keep the same wrapper interface.
    """

    def __init__(self, emb_dim, act_emb_dim, hidden_dim=512, depth=3):
        super().__init__()
        self.emb_dim = emb_dim
        self.act_emb_dim = act_emb_dim
        self.hidden_dim = hidden_dim

        layers = []
        in_dim = emb_dim + act_emb_dim
        for i in range(depth - 1):
            layers.append(
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim)
            )
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim if depth > 1 else in_dim, emb_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, emb, act_emb):
        """
        emb: (B, T, D)
        act_emb: (B, T, A)
        returns: preds (B, T, D)
        """
        # Simple per-timestep prediction from concatenated features
        x = torch.cat([emb, act_emb], dim=-1)
        B, T, _ = x.shape
        x = x.view(B * T, -1)
        out = self.net(x)
        out = out.view(B, T, self.emb_dim)
        return out


class RewardTerminationHead(nn.Module):
    """A small model to predict scalar reward and binary termination from an embedding."""

    def __init__(self, emb_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.reward_head = nn.Linear(hidden, 1)
        self.terminal_head = nn.Linear(hidden, 1)

    def forward(self, emb):
        # emb: (B, T, D)
        B, T, D = emb.shape
        x = emb.view(B * T, D)
        h = self.net(x)
        reward = self.reward_head(h).view(B, T, 1)
        terminal_logits = self.terminal_head(h).view(B, T, 1)
        terminal = torch.sigmoid(terminal_logits)
        return reward, terminal


__all__ = ['DiffusionPredictor', 'RewardTerminationHead']


__all__ = ['DiffusionPredictor']
