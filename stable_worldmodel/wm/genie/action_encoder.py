"""Raw-action → action-embedding head for Genie.

Genie's dynamics conditions on LAM-extracted action codes (32-dim embeddings
from a small VQ). The LAM only consumes video, never raw actions. For
test-time control via a solver, the solver samples raw 2-D PushT actions and
needs a way to feed those into dyn. This module bridges the gap: a small MLP
trained to mimic LAM's output for known (frame_t, frame_t+1, action) triples.

During the joint training phase we add an auxiliary loss
    MSE(action_encoder(raw_action), LAM.extract_actions(video).detach())
which leaves LAM and dyn unaffected but teaches the action_encoder to project
raw actions into LAM's embedding space.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ActionEncoder(nn.Module):
    """MLP from raw actions to LAM-compatible action embeddings.

    Args:
        input_dim: Raw action dimensionality (e.g. 2 * frameskip for PushT).
        emb_dim: Output embedding dim, should match Genie's vq_embed_dim.
        hidden_dim: Width of the single hidden layer.
    """

    def __init__(self, input_dim: int, emb_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # (..., input_dim) -> (..., emb_dim); broadcasts over arbitrary leading dims.
        return self.net(x)
