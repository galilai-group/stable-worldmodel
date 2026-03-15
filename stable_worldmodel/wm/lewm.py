"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange, einsum
from torch import nn


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class JEPA(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(
            pixels, 'b t ... -> (b t) ...'
        )  # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info['emb'] = rearrange(emb, '(b t) d -> b t d', b=b)

        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, 'b t d -> (b t) d'))
        preds = rearrange(preds, '(b t) d -> b t d', b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert 'pixels' in info, 'pixels not in info_dict'
        H = info['pixels'].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info['action'] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info['emb'] = _init['emb'].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, 'b s ... -> (b s) ...').clone()
        act = rearrange(act_0, 'b s ... -> (b s) ...')
        act_future = rearrange(act_future, 'b s ... -> (b s) ...')

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, '(b s) ... -> b s ...', b=B, s=S)
        info['predicted_emb'] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict['predicted_emb']  # (B,S, T-1, dim)
        goal_emb = info_dict['goal_emb']  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction='none',
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Compute the cost of action candidates given an info dict with goal and initial state."""

        assert 'goal' in info_dict, 'goal not in info_dict'

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal['pixels'] = goal['goal']

        for k in info_dict:
            if k.startswith('goal_'):
                goal[k[len('goal_') :]] = goal.pop(k)

        goal.pop('action')
        goal = self.encode(goal)

        info_dict['goal_emb'] = goal['emb']
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)

        return cost


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(
            3, dim=-1
        )  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (
            rearrange(t, 'b t (h d) -> b h t d', h=self.heads) for t in qkv
        )
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=drop, is_causal=causal
        )
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):
        x = self.input_proj(x)

        if c is not None:
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)
        x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(
            input_dim, smoothed_dim, kernel_size=1, stride=1
        )
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, input_dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


#####################
## Training Losses ##
#####################


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer('t', t)
        self.register_buffer('phi', window)
        self.register_buffer('weights', weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device='cuda')
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(
            -3
        ).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()  # average over projections and time


class VCReg(torch.nn.Module):
    """Variance-Covariance Regularizer"""

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def _std_loss(self, z):
        z = z.transpose(0, 1)  # (T, B, D)
        std = (z.var(dim=1) + self.eps).sqrt()  # (T, D)
        std_loss = torch.mean(F.relu(1 - std), dim=-1)  # (T,)
        return std_loss

    def _cov_loss(self, z):
        B, T, D = z.shape
        z = z.transpose(0, 1)  # (T, B, D)
        cov = einsum(z, z, 't b i, t b j -> t i j') / (B - 1)  # (T, D, D)
        diag = einsum(cov, 't i i -> t i').pow(2).sum(dim=-1)  # (T,)
        cov_loss = (cov.pow(2).sum(dim=[-1, -2]) - diag).div(D**2 - D)  # (T,)
        return cov_loss

    def forward(self, z):
        """
        z: (..., D)
        """

        if z.dim() == 2:
            D = z.size(-1)
            z = z.view(-1, D)

        z = z - z.mean(
            dim=0, keepdim=True
        )  # mean for each dim across batch samples

        return {
            'std_loss': self._std_loss(z).mean(),
            'std_t_loss': self._std_loss(z.transpose(0, 1)).mean(),
            'cov_loss': self._cov_loss(z).mean(),
            'cov_t_loss': self._cov_loss(z.transpose(0, 1)).mean(),
        }


class PLDM(torch.nn.Module):
    """VCReg anti-collapse + Temporal Alignment + Inverse Dynamics Modeling losses"""

    def __init__(self):
        super().__init__()

        self.vc_reg = VCReg()

    def forward(self, z, a_pred=None, a_target=None):
        """
        z: (B, T, D)
        a_pred: (B, T-1, A)
        a_target: (B, T-1, A)
        """

        output = {}
        if a_pred is not None and a_target is not None:
            output['idm_loss'] = F.mse_loss(a_pred, a_target)

        output['temp_align_loss'] = F.mse_loss(z[:, :-1], z[:, 1:])  # detach?
        output.update(self.vc_reg(z))

        return output


#######################
###  Geodesic Loss  ###
#######################


class PathStraighteningLoss(torch.nn.Module):
    """Path Straightening Loss Module (Pairwise Negative Cosine Similarity)"""

    def __init__(self):
        super().__init__()
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        v = x[:, 1:] - x[:, :-1]  # velocities
        sim = self.cos_sim(v[:, :-1], v[:, 1:])
        return -sim.mean()
