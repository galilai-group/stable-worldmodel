# Building blocks for the DreamerV4-style world model.
# Architecture adapted from https://github.com/nicklashansen/dreamer4 (MIT License)
# by Niklas Hansen, Haochen Shi, Jiuqi Wang (2025).
#
# Key idea: flow matching in latent space with shortcut forcing.
# The dynamics model predicts the clean latent z1 directly from a
# noisy interpolation z_tilde = (1-sigma)*noise + sigma*z1.
# Shortcut forcing trains variable-step (K=1,2,4,...,k_max) consistency
# so inference can run in as few as 1 function evaluation.

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Token layout helpers
# ---------------------------------------------------------------------------

class Modality(IntEnum):
    ACTION          = 1
    REGISTER        = 3
    SPATIAL         = 4
    SHORTCUT_SIGNAL = 5
    SHORTCUT_STEP   = 6


@dataclass(frozen=True)
class TokenLayout:
    segments: Tuple[Tuple[Modality, int], ...]

    def S(self) -> int:
        return sum(n for _, n in self.segments)

    def modality_ids(self) -> torch.Tensor:
        parts = [
            torch.full((n,), int(m), dtype=torch.int32)
            for m, n in self.segments if n > 0
        ]
        return torch.cat(parts, dim=0) if parts else torch.zeros((0,), dtype=torch.int32)

    def slices(self) -> Dict[Modality, slice]:
        out: Dict[Modality, slice] = {}
        idx = 0
        for m, n in self.segments:
            if n > 0 and m not in out:
                out[m] = slice(idx, idx + n)
            idx += n
        return out


# ---------------------------------------------------------------------------
# Positional embeddings
# ---------------------------------------------------------------------------

def sinusoid_table(n: int, d: int, base: float = 10000.0, device=None) -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    i   = torch.arange(d, device=device, dtype=torch.float32).unsqueeze(0)
    k   = torch.floor(i / 2.0)
    div = torch.exp(-(2.0 * k) / max(1.0, float(d)) * math.log(base))
    ang = pos * div
    return torch.where((i % 2) == 0, torch.sin(ang), torch.cos(ang))


def add_sinusoidal_positions(tokens: torch.Tensor) -> torch.Tensor:
    """tokens: (B, T, S, D)"""
    B, T, S, D = tokens.shape
    device = tokens.device
    pos_t = sinusoid_table(T, D, device=device)
    pos_s = sinusoid_table(S, D, device=device)
    pos = (pos_t[None, :, None, :] + pos_s[None, None, :, :]) * (1.0 / math.sqrt(D))
    return tokens + pos.to(dtype=tokens.dtype)


# ---------------------------------------------------------------------------
# Core transformer layers
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.scale / (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt())


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc_in  = nn.Linear(d_model, 2 * hidden)
        self.fc_out = nn.Linear(hidden, d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.fc_in(x).chunk(2, dim=-1)
        return self.drop(self.fc_out(self.drop(u * F.silu(v))))


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads    = n_heads
        self.head_dim   = d_model // n_heads
        self.dropout_p  = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor, *, attn_mask=None, is_causal: bool = False):
        N, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        drop = self.dropout_p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop, is_causal=is_causal)
        return self.out(y.transpose(1, 2).contiguous().view(N, L, D))


class SpaceSelfAttention(nn.Module):
    """Per-timestep attention with modality-aware masking."""

    def __init__(self, d_model: int, n_heads: int, modality_ids: torch.Tensor, dropout: float):
        super().__init__()
        self.register_buffer('modality_ids', modality_ids.to(torch.int32), persistent=False)
        S = int(modality_ids.numel())
        q_mod = modality_ids.unsqueeze(1)
        k_mod = modality_ids.unsqueeze(0)
        # full mixing for dynamics (all modalities can attend to all)
        allow = torch.ones((S, S), dtype=torch.bool)
        self.register_buffer('attn_mask', allow.unsqueeze(0).unsqueeze(0), persistent=False)
        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x.shape
        mask = self.attn_mask.expand(B * T, 1, S, S)
        y = self.attn(x.reshape(B * T, S, D), attn_mask=mask, is_causal=False)
        return y.reshape(B, T, S, D)


class TimeSelfAttention(nn.Module):
    """Causal temporal attention, applied to all spatial tokens."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x.shape
        x_nld = x.permute(0, 2, 1, 3).contiguous().view(B * S, T, D)
        out = self.attn(x_nld, is_causal=True)
        return out.view(B, S, T, D).permute(0, 2, 1, 3).contiguous()


class BlockCausalLayer(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, modality_ids: torch.Tensor,
        dropout: float, mlp_ratio: float, layer_index: int, time_every: int,
    ):
        super().__init__()
        self.do_time = ((layer_index + 1) % time_every == 0)
        self.norm1 = RMSNorm(d_model)
        self.space = SpaceSelfAttention(d_model, n_heads, modality_ids, dropout)
        self.drop1 = nn.Dropout(dropout)
        if self.do_time:
            self.norm2 = RMSNorm(d_model)
            self.time  = TimeSelfAttention(d_model, n_heads, dropout)
            self.drop2 = nn.Dropout(dropout)
        self.norm3 = RMSNorm(d_model)
        self.mlp   = MLP(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.space(self.norm1(x)))
        if self.do_time:
            x = x + self.drop2(self.time(self.norm2(x)))
        return x + self.mlp(self.norm3(x))


class BlockCausalTransformer(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, depth: int,
        modality_ids: torch.Tensor, dropout: float, mlp_ratio: float, time_every: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            BlockCausalLayer(
                d_model=d_model, n_heads=n_heads, modality_ids=modality_ids,
                dropout=dropout, mlp_ratio=mlp_ratio,
                layer_index=i, time_every=time_every,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Action encoder
# ---------------------------------------------------------------------------

class ActionEncoder(nn.Module):
    """Continuous actions -> (B, T, 1, d_model) token."""

    def __init__(self, d_model: int, action_dim: int = 3, hidden_mult: float = 2.0):
        super().__init__()
        hidden = int(d_model * hidden_mult)
        self.base = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.base, std=0.02)
        self.fc1 = nn.Linear(action_dim, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        nn.init.normal_(self.fc2.weight, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, actions: Optional[torch.Tensor], *, batch_time_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if actions is None:
            assert batch_time_shape is not None
            B, T = batch_time_shape
            out = self.base.view(1, 1, -1).expand(B, T, -1)
        else:
            out = self.fc2(F.silu(self.fc1(actions.clamp(-1, 1)))) + self.base
        return out[:, :, None, :]  # (B, T, 1, d_model)


# ---------------------------------------------------------------------------
# Dynamics model
# ---------------------------------------------------------------------------

class Dynamics(nn.Module):
    """Block-causal transformer dynamics model with shortcut forcing.

    Inputs  per timestep: noisy spatial tokens + action token + shortcut conditioning.
    Output per timestep: predicted clean spatial tokens (x1_hat).
    """

    def __init__(
        self,
        *,
        d_model: int,
        d_spatial: int,
        n_spatial: int,
        n_register: int = 4,
        n_heads: int,
        depth: int,
        k_max: int,
        action_dim: int = 3,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 1,
    ):
        super().__init__()
        assert (k_max & (k_max - 1)) == 0, 'k_max must be a power of 2'
        self.d_model    = d_model
        self.d_spatial  = d_spatial
        self.n_spatial  = n_spatial
        self.n_register = n_register
        self.k_max      = k_max

        self.spatial_proj    = nn.Linear(d_spatial, d_model)
        self.register_tokens = nn.Parameter(torch.empty(n_register, d_model))
        nn.init.normal_(self.register_tokens, std=0.02)

        self.action_encoder = ActionEncoder(d_model=d_model, action_dim=action_dim)

        # shortcut conditioning: step_idx ∈ {0,...,log2(k_max)}, signal_idx ∈ {0,...,k_max}
        n_step_bins = int(math.log2(k_max)) + 1
        self.step_embed   = nn.Embedding(n_step_bins, d_model)
        self.signal_embed = nn.Embedding(k_max + 1,   d_model)

        segments = (
            (Modality.ACTION,          1),
            (Modality.SHORTCUT_SIGNAL, 1),
            (Modality.SHORTCUT_STEP,   1),
            (Modality.SPATIAL,         n_spatial),
            (Modality.REGISTER,        n_register),
        )
        self.layout = TokenLayout(segments=segments)
        sl = self.layout.slices()
        self.spatial_slice = sl[Modality.SPATIAL]
        modality_ids = self.layout.modality_ids()

        self.transformer = BlockCausalTransformer(
            d_model=d_model, n_heads=n_heads, depth=depth,
            modality_ids=modality_ids, dropout=dropout,
            mlp_ratio=mlp_ratio, time_every=time_every,
        )

        self.flow_head = nn.Linear(d_model, d_spatial)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def forward(
        self,
        actions: Optional[torch.Tensor],     # (B, T, action_dim) or None
        step_idxs: torch.Tensor,             # (B, T) long
        signal_idxs: torch.Tensor,           # (B, T) long
        spatial_tokens: torch.Tensor,        # (B, T, n_spatial, d_spatial)
    ) -> torch.Tensor:
        B, T = spatial_tokens.shape[:2]

        s_tok  = self.spatial_proj(spatial_tokens)                          # (B,T,n_spatial,d)
        a_tok  = self.action_encoder(actions, batch_time_shape=(B, T))      # (B,T,1,d)
        reg    = self.register_tokens.view(1, 1, self.n_register, -1).expand(B, T, -1, -1)
        st_tok = self.step_embed(step_idxs.to(torch.long))[:, :, None, :]  # (B,T,1,d)
        si_tok = self.signal_embed(signal_idxs.to(torch.long))[:, :, None, :]  # (B,T,1,d)

        tokens = torch.cat([a_tok, si_tok, st_tok, s_tok, reg], dim=2)     # (B,T,S,d)
        tokens = add_sinusoidal_positions(tokens)
        x = self.transformer(tokens)

        spatial_out = x[:, :, self.spatial_slice, :]                        # (B,T,n_spatial,d)
        return self.flow_head(spatial_out)                                   # (B,T,n_spatial,d_spatial)


# ---------------------------------------------------------------------------
# Training loss: flow matching + shortcut bootstrap
# ---------------------------------------------------------------------------

def _emax(k_max: int) -> int:
    e = int(round(math.log2(k_max)))
    assert (1 << e) == k_max
    return e


def _sample_coarse_step(device, B: int, T: int, k_max: int):
    emax = _emax(k_max)
    step_idx = torch.randint(0, max(1, emax), (B, T), device=device, dtype=torch.long)
    return step_idx


def _sample_tau(device, B: int, T: int, k_max: int, step_idx: torch.Tensor):
    K = (1 << step_idx).to(torch.long)
    j = torch.floor(torch.rand((B, T), device=device) * K.float()).to(torch.long)
    tau      = j.float() / K.float()
    scale    = k_max // K
    tau_idx  = j * scale
    return tau, tau_idx


def dynamics_loss(
    dynamics: nn.Module,
    *,
    z1: torch.Tensor,                        # (B, T, n_spatial, d_spatial) clean targets
    actions: Optional[torch.Tensor],         # (B, T, action_dim) or None
    k_max: int,
    self_fraction: float = 0.25,
    bootstrap_start: int = 5_000,
    global_step: int = 0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Flow matching loss with shortcut bootstrap.

    Empirical rows (B_emp): train finest-step prediction (standard flow matching).
    Self rows (B_self): train shortcut consistency (enables K=1 inference after warmup).
    """
    device = z1.device
    B, T   = z1.shape[:2]
    emax   = _emax(k_max)

    B_self = min(max(0, int(round(self_fraction * B))), B - 1)
    B_emp  = B - B_self

    # step indices
    step_emp  = torch.full((B_emp, T), emax, device=device, dtype=torch.long)
    if B_self > 0:
        step_self = _sample_coarse_step(device, B_self, T, k_max)
    else:
        step_self = torch.zeros((0, T), device=device, dtype=torch.long)
    step_full = torch.cat([step_emp, step_self], dim=0)

    sigma_full, sigma_idx_full = _sample_tau(device, B, T, k_max, step_full)
    sigma_emp  = sigma_full[:B_emp]
    sigma_self = sigma_full[B_emp:]
    sigma_idx_self = sigma_idx_full[B_emp:]

    # corrupt: z_tilde = (1-sigma)*noise + sigma*z_clean
    z0_full     = torch.randn_like(z1)
    z_tilde     = (1.0 - sigma_full)[..., None, None] * z0_full + sigma_full[..., None, None] * z1
    z_tilde_self = z_tilde[B_emp:]

    # importance weights
    w_emp  = 0.9 * sigma_emp  + 0.1
    w_self = 0.9 * sigma_self + 0.1

    acts_emp  = actions[:B_emp] if actions is not None else None
    acts_self = actions[B_emp:] if actions is not None else None

    # main forward
    x1_hat_full = dynamics(actions, step_full, sigma_idx_full, z_tilde)
    x1_hat_emp  = x1_hat_full[:B_emp]
    x1_hat_self = x1_hat_full[B_emp:]

    flow_per = (x1_hat_emp.float() - z1[:B_emp].float()).pow(2).mean(dim=(2, 3))  # (B_emp,T)
    loss_emp = (flow_per * w_emp).mean()

    boot_mse   = z1.new_zeros(())
    loss_self  = z1.new_zeros(())

    do_boot = (B_self > 0) and (global_step >= bootstrap_start)
    if do_boot:
        d_self    = 1.0 / (1 << step_self).float()
        d_half    = d_self / 2.0
        step_half = step_self + 1
        sigma_plus     = sigma_self + d_half
        sigma_idx_plus = sigma_idx_self + (k_max * d_half).to(torch.long)

        # half-step 1: from sigma_self using step_half
        x1_h1 = dynamics(acts_self, step_half, sigma_idx_self, z_tilde_self)
        denom1 = (1.0 - sigma_self).clamp_min(1e-6)[..., None, None]
        b1 = (x1_h1.float() - z_tilde_self.float()) / denom1
        z_prime = (z_tilde_self.float() + b1 * d_half[..., None, None]).to(z_tilde_self.dtype)

        # half-step 2: from sigma_plus using step_half
        x1_h2 = dynamics(acts_self, step_half, sigma_idx_plus, z_prime)
        denom2 = (1.0 - sigma_plus).clamp_min(1e-6)[..., None, None]
        b2 = (x1_h2.float() - z_prime.float()) / denom2

        # target for single full step
        vbar = ((b1 + b2) / 2.0).detach()

        # self prediction
        denom_s = (1.0 - sigma_self).clamp_min(1e-6)[..., None, None]
        v_self = (x1_hat_self.float() - z_tilde_self.float()) / denom_s

        boot_per  = (1.0 - sigma_self).pow(2) * (v_self - vbar).pow(2).mean(dim=(2, 3))
        loss_self = (boot_per * w_self).mean()
        boot_mse  = boot_per.mean()

    loss = ((loss_emp * B_emp) + (loss_self * B_self)) / B

    aux = {
        'flow_mse':      flow_per.mean().detach(),
        'bootstrap_mse': boot_mse.detach(),
        'loss_emp':      loss_emp.detach(),
        'loss_self':     loss_self.detach(),
        'sigma_mean':    sigma_full.mean().detach(),
    }
    return loss, aux


# ---------------------------------------------------------------------------
# Inference: flow integration (shortcut sampling)
# ---------------------------------------------------------------------------

def make_schedule(k_max: int, K: int) -> Dict:
    """Build a shortcut sampling schedule with K integration steps."""
    assert (k_max & (k_max - 1)) == 0 and (K & (K - 1)) == 0
    assert K <= k_max
    e     = int(round(math.log2(K)))
    scale = k_max // K
    tau   = [i / K for i in range(K)] + [1.0]
    tau_idx = [i * scale for i in range(K)] + [k_max]
    return dict(K=K, e=e, dt=1.0 / K, tau=tau, tau_idx=tau_idx)


@torch.no_grad()
def sample_next_frame(
    dynamics: Dynamics,
    *,
    past_packed: torch.Tensor,           # (B, t, n_spatial, d_spatial)
    sched: Dict,
    actions: Optional[torch.Tensor],     # (B, t+1, action_dim) or None
    k_max: int,
) -> torch.Tensor:
    """Sample one future timestep autoregressively using flow integration."""
    device = past_packed.device
    dtype  = past_packed.dtype
    B, t   = past_packed.shape[:2]
    n_s, d_s = past_packed.shape[2], past_packed.shape[3]

    K, e  = int(sched['K']), int(sched['e'])
    tau   = sched['tau']
    tau_idx = sched['tau_idx']
    dt    = float(sched['dt'])
    emax  = _emax(k_max)

    z = torch.randn((B, 1, n_s, d_s), device=device, dtype=dtype)

    step_full = torch.full((B, t + 1), emax, device=device, dtype=torch.long)
    step_full[:, -1] = e
    sig_full  = torch.full((B, t + 1), k_max - 1, device=device, dtype=torch.long)

    for i in range(K):
        sig_full[:, -1] = int(tau_idx[i])
        seq = torch.cat([past_packed, z], dim=1)  # (B, t+1, n_s, d_s)

        if actions is None:
            acts_in = None
        elif actions.shape[1] >= t + 1:
            acts_in = actions[:, :t + 1]
        else:
            # pad with zeros if future action is unavailable
            pad = torch.zeros(B, t + 1 - actions.shape[1], *actions.shape[2:],
                              device=device, dtype=actions.dtype)
            acts_in = torch.cat([actions, pad], dim=1)
        x1_hat  = dynamics(acts_in, step_full, sig_full, seq)[:, -1:]  # (B,1,n_s,d_s)

        tau_i = float(tau[i])
        denom = max(1e-4, 1.0 - tau_i)
        b     = (x1_hat.float() - z.float()) / denom
        z     = (z.float() + b * dt).to(dtype)

    return z[:, 0]  # (B, n_spatial, d_spatial)


__all__ = [
    'Modality', 'TokenLayout',
    'RMSNorm', 'MLP', 'MultiheadSelfAttention',
    'SpaceSelfAttention', 'TimeSelfAttention',
    'BlockCausalLayer', 'BlockCausalTransformer',
    'ActionEncoder', 'Dynamics',
    'dynamics_loss', 'make_schedule', 'sample_next_frame',
]
