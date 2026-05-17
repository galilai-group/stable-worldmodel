import warnings

import torch
import torch.nn.functional as F
from torch import nn


class TransformerActionEncoder(nn.Module):
    """Encode a fixed primitive-action segment into a latent macro-action."""

    def __init__(
        self,
        *,
        action_dim: int,
        horizon: int,
        latent_dim: int,
        hidden_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        pooling: str = 'cls',
    ):
        super().__init__()
        if pooling not in {'cls', 'mean'}:
            raise ValueError("pooling must be either 'cls' or 'mean'")

        self.action_dim = action_dim
        self.horizon = horizon
        self.latent_dim = latent_dim
        self.pooling = pooling

        self.input_proj = nn.Linear(action_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, horizon, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        # Keep dim_head explicit in config for parity with predictor-style
        # configs, even though PyTorch derives it from hidden_dim / heads.
        expected_dim_head = hidden_dim // heads
        if expected_dim_head != dim_head:
            raise ValueError(
                f'dim_head={dim_head} does not match hidden_dim / heads '
                f'({hidden_dim} / {heads} = {expected_dim_head})'
            )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """Return one latent action per primitive action segment.

        Args:
            actions: Tensor shaped ``(batch, horizon * action_dim)`` or
                ``(batch, horizon, action_dim)``.
        """
        if actions.ndim == 2:
            actions = actions.reshape(
                actions.shape[0], self.horizon, self.action_dim
            )
        elif actions.ndim != 3:
            raise ValueError(
                'actions must have shape (B, H * A) or (B, H, A), '
                f'got {tuple(actions.shape)}'
            )

        if actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(
                f'Expected action segment shape (*, {self.horizon}, '
                f'{self.action_dim}), got {tuple(actions.shape)}'
            )

        tokens = self.input_proj(actions)
        tokens = tokens + self.pos_embedding[:, : tokens.shape[1]]
        if self.pooling == 'cls':
            cls = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        tokens = self.transformer(tokens)
        if self.pooling == 'cls':
            pooled = tokens[:, 0]
        else:
            pooled = tokens.mean(dim=1)

        return self.output_proj(self.norm(pooled))


__all__ = ['TransformerActionEncoder']


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over non-padding sequence positions."""
    valid = (~mask).float()
    summed = (x * valid.unsqueeze(-1)).sum(dim=1)
    count = valid.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
    return summed / count


class PerStepMLP(nn.Module):
    """Project each action-block vector independently."""

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(x)


class SequenceEncoder(nn.Module):
    """Encode variable-length action-block chunks into latent macro-actions.

    Args:
        actions: Tensor shaped ``(B, T, L, input_dim)``.
        action_masks: Bool tensor shaped ``(B, T, L)`` where True is padding.
    """

    def __init__(
        self,
        output_dim: int = 4,
        input_dim: int = 10,
        d_model: int = 64,
        step_mlp: bool = True,
        nhead: int = 2,
        num_layers: int = 1,
        ff_mult: int = 2,
        dropout: float = 0.1,
        use_cls: bool = True,
        stochastic: bool = False,
        max_chunk: int = 15,
        norm_first: bool = True,
        uniform_input: bool = False,
    ):
        super().__init__()

        self.stochastic = stochastic
        self.use_cls = use_cls
        self.max_chunk = max_chunk
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.use_step_mlp = step_mlp
        self.uniform_input = uniform_input

        if self.use_step_mlp:
            self.step_mlp = PerStepMLP(input_dim=input_dim, d_model=d_model)
        else:
            if d_model != input_dim:
                raise ValueError(
                    'When step_mlp=False, d_model must equal input_dim.'
                )
            self.step_mlp = nn.Identity()

        self.pos = nn.Parameter(torch.zeros(max_chunk + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers
        )

        out_dim = 2 * output_dim if stochastic else output_dim
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
        )

        self.final_ln = nn.LayerNorm(
            output_dim, elementwise_affine=True, eps=1e-6
        )

    def forward(
        self,
        actions: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ):
        if (action_masks is None) != self.uniform_input:
            raise ValueError(
                'Invalid action_masks/self.uniform_input combination: '
                'expected action_masks is None exactly when '
                f'uniform_input=True. Got uniform_input={self.uniform_input}, '
                f'action_masks_is_none={action_masks is None}.'
            )

        bsz, num_chunks, length, channels = actions.shape
        if channels != self.input_dim:
            raise ValueError(
                f'Last dim of actions must be {self.input_dim}, '
                f'got {channels}'
            )

        if action_masks is None:
            action_masks = torch.zeros(
                bsz,
                num_chunks,
                length,
                dtype=torch.bool,
                device=actions.device,
            )

        if length > self.max_chunk:
            warnings.warn(
                f'chunk_t={length} exceeds max_chunk={self.max_chunk}; '
                'truncating to max_chunk'
            )
            actions = actions[:, :, : self.max_chunk]
            action_masks = action_masks[:, :, : self.max_chunk]
            length = self.max_chunk

        x = actions.reshape(bsz * num_chunks, length, channels)
        mask = action_masks.reshape(bsz * num_chunks, length)

        x = self.step_mlp(x)
        x = x + self.pos[1 : length + 1].unsqueeze(0)

        if self.use_cls:
            cls_tok = self.pos[0].unsqueeze(0).unsqueeze(0).expand(
                bsz * num_chunks, 1, -1
            )
            x = torch.cat([cls_tok, x], dim=1)
            pad_cls = torch.zeros(
                bsz * num_chunks, 1, dtype=torch.bool, device=x.device
            )
            src_key_padding_mask = torch.cat([pad_cls, mask], dim=1)
        else:
            src_key_padding_mask = mask

        y = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        h = y[:, 0, :] if self.use_cls else masked_mean(
            y, src_key_padding_mask
        )
        out = self.head(h)

        if self.stochastic:
            mu, log_std = torch.chunk(out, 2, dim=-1)
            std = F.softplus(log_std) + 1e-5
            return mu.view(bsz, num_chunks, -1), std.view(
                bsz, num_chunks, -1
            )

        z = self.final_ln(out)
        return z.view(bsz, num_chunks, -1)


__all__ = ['TransformerActionEncoder', 'SequenceEncoder']
