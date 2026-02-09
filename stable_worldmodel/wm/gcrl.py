import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn


# TODO encode is very similar to the one in pyro.py - consider refactoring
# TODO models are very similar to the ones in pyro.py - consider refactoring


class GCRL(torch.nn.Module):
    def __init__(
        self,
        encoder,
        action_predictor,
        value_predictor,
        critic_predictor=None,
        extra_encoders=None,
        history_size=3,
        interpolate_pos_encoding=True,
        log_std_min=-5.0,
        log_std_max=2.0,
    ):
        super().__init__()

        self.encoder = encoder
        self.value_predictor = value_predictor
        self.action_predictor = action_predictor
        self.critic_predictor = critic_predictor
        self.extra_encoders = extra_encoders or {}
        self.history_size = history_size

        self.interpolate_pos_encoding = interpolate_pos_encoding

        # Learnable log_stds for action distribution (state-independent)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        action_dim = action_predictor.out_proj.out_features
        self.log_stds = nn.Parameter(torch.zeros(action_dim))

    def encode(
        self,
        info,
        pixels_key='pixels',
        emb_keys=None,
        prefix=None,
        target='embed',
        is_video=False,
    ):
        assert target not in info, f'{target} key already in info_dict'
        emb_keys = emb_keys or self.extra_encoders.keys()
        prefix = prefix or ''

        encode_fn = self._encode_video if is_video else self._encode_image
        pixels_embed = encode_fn(info[pixels_key].float())  # (B, T, 3, H, W)

        # == improve the embedding
        n_patches = pixels_embed.shape[2]
        embedding = pixels_embed
        info[f'pixels_{target}'] = pixels_embed

        for key in emb_keys:
            extr_enc = self.extra_encoders[key]
            extra_input = info[f'{prefix}{key}'].float()  # (B, T, dim)
            extra_embed = extr_enc(
                extra_input
            )  # (B, T, dim) -> (B, T, emb_dim)
            info[f'{key}_{target}'] = extra_embed

            # copy extra embedding across patches for each time step
            extra_tiled = repeat(
                extra_embed.unsqueeze(2), 'b t 1 d -> b t p d', p=n_patches
            )

            # concatenate along feature dimension
            embedding = torch.cat([embedding, extra_tiled], dim=3)

        info[target] = embedding  # (B, T, P, d)

        return info

    def _encode_image(self, pixels):
        # == pixels embedding
        B = pixels.shape[0]
        pixels = rearrange(pixels, 'b t ... -> (b t) ...')

        kwargs = (
            {'interpolate_pos_encoding': True}
            if self.interpolate_pos_encoding
            else {}
        )
        pixels_embed = self.encoder(pixels, **kwargs)

        if hasattr(pixels_embed, 'last_hidden_state'):
            pixels_embed = pixels_embed.last_hidden_state
            pixels_embed = pixels_embed[:, 1:, :]  # drop cls token
        else:
            pixels_embed = pixels_embed.logits.unsqueeze(
                1
            )  # (B*T, 1, emb_dim)

        pixels_embed = rearrange(pixels_embed, '(b t) p d -> b t p d', b=B)

        return pixels_embed

    def _encode_video(self, pixels):
        B, T, C, H, W = pixels.shape
        kwargs = (
            {'interpolate_pos_encoding': True}
            if self.interpolate_pos_encoding
            else {}
        )

        pixels_embeddings = []

        # roll the embedding computation over time
        for t in range(T):
            padding = max(T - (t + 1), 0)  # number of frames to pad
            past_frames = pixels[:, : t + 1, :, :, :]  # (B, t+1, C, H, W)

            # repeat last frame to pad
            pad_frames = past_frames[:, -1:, :, :, :].repeat(
                1, padding, 1, 1, 1
            )  # (B, padding, C, H, W)
            frames = torch.cat(
                [past_frames, pad_frames], dim=1
            )  # (B, T, C, H, W)

            frame_embed = self.encoder(frames, **kwargs)  # (B, 1, P, emb_dim)
            frame_embed = frame_embed.last_hidden_state
            pixels_embeddings.append(frame_embed)

        pixels_embed = torch.stack(
            pixels_embeddings, dim=1
        )  # (B, T, P, emb_dim)

        return pixels_embed

    def predict_actions(self, embedding, embedding_goal, temperature=1.0):
        """predict action distribution per frame
        Args:
            embedding: (B, T, P, d)
            embedding_goal: (B, 1, P, d)
            temperature: scaling factor for the standard deviation
        Returns:
            means: (B, T, action_dim) - action means
            stds: (action_dim,) - action standard deviations (broadcasted)
        """

        embedding = rearrange(embedding, 'b t p d -> b (t p) d')
        embedding_goal = rearrange(embedding_goal, 'b t p d -> b (t p) d')
        means = self.action_predictor(embedding, embedding_goal)

        # Clip log_stds and compute scale
        log_stds = torch.clamp(
            self.log_stds, self.log_std_min, self.log_std_max
        )
        stds = torch.exp(log_stds) * temperature

        return means, stds

    def predict_values(self, embedding, embedding_goal):
        """predict values per frame
        Args:
            embedding: (B, T, P, d)
            embedding_goal: (B, 1, P, d)
        Returns:
            preds: (B, T, 1)
        """

        embedding = rearrange(embedding, 'b t p d -> b (t p) d')
        embedding_goal = rearrange(embedding_goal, 'b t p d -> b (t p) d')
        preds = self.value_predictor(embedding, embedding_goal)

        return preds

    def get_action(self, info, sample=False, temperature=1.0):
        """Get action given observation and goal (uses last frame's prediction).

        Args:
            info: dict containing 'pixels' and 'goal' keys
            sample: if True, sample from distribution; if False, return mean
            temperature: scaling factor for std when sampling
        Returns:
            actions: (B, action_dim)
        """
        # first encode observation
        info = self.encode(info, pixels_key='pixels', target='embed')
        # encode goal
        info = self.encode(
            info,
            pixels_key='goal',
            prefix='goal_',
            target='goal_embed',
        )
        # then predict action distribution
        means, stds = self.predict_actions(
            info['embed'], info['goal_embed'], temperature=temperature
        )
        # get last frame's action prediction
        means = means[:, -1, :]

        if sample:
            # Sample from Normal distribution
            actions = means + stds * torch.randn_like(means)
        else:
            actions = means

        return actions


class Embedder(torch.nn.Module):
    def __init__(
        self,
        num_frames=1,
        tubelet_size=1,
        in_chans=8,
        emb_dim=10,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.emb_dim = emb_dim
        self.patch_embed = torch.nn.Conv1d(
            in_chans, emb_dim, kernel_size=tubelet_size, stride=tubelet_size
        )

    def forward(self, x):
        with torch.amp.autocast(enabled=False, device_type=x.device.type):
            x = x.permute(0, 2, 1)  # (B, T, B) -> (B, D, T)
            x = self.patch_embed(x)
            x = x.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)
        return x


class Predictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        out_dim,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        causal=True,
        non_positive_output=False,
    ):
        super().__init__()

        self.num_patches = num_patches
        self.num_frames = num_frames
        self.non_positive_output = non_positive_output

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * (num_patches), dim)
        )  # dim for the pos encodings
        self.pos_embedding_goal = nn.Parameter(
            torch.randn(1, (num_patches), dim)
        )  # dim for the pos encodings of goal (assumed single image)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            num_patches,
            num_frames,
            causal=causal,
        )
        self.out_proj = nn.Linear(dim, out_dim)

    def forward(self, x, g):
        """
        Args:
            x: (B, T*P, dim) - observation embeddings
            g: (B, P, dim) - goal embeddings
        Returns:
            out: (B, T, out_dim) - per-frame predictions
        """
        # prepare input for transformer
        x = x + self.pos_embedding[:, : x.shape[1]]
        g = g + self.pos_embedding_goal[:, : g.shape[1]]
        x = self.dropout(x)
        # transformer forward - returns (B, T, dim), one embedding per frame
        x = self.transformer(x, g)
        # project to output dimension
        x = self.out_proj(x)
        # apply non-positive constraint for value functions (rewards <= 0)
        if self.non_positive_output:
            x = -F.softplus(x)
        return x


class FeedForward(nn.Module):
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
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        num_patches=1,
        num_frames=1,
        att_type='self',
        causal=False,
    ):
        super().__init__()
        assert att_type in {'self', 'cross', 'frame_agg'}, (
            'attention type must be self, cross, or frame_agg'
        )
        self.att_type = att_type
        self.causal = causal and att_type in {'self', 'frame_agg'}
        self.num_patches = num_patches
        self.num_frames = num_frames

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        if self.att_type == 'cross':
            self.norm_c = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        # Frame aggregation: one learnable query token per frame
        if self.att_type == 'frame_agg':
            self.frame_tokens = nn.Parameter(
                0.02 * torch.randn(1, num_frames, dim)
            )

        # Register causal mask buffer
        if self.causal:
            if self.att_type == 'self':
                mask = self._generate_causal_mask(num_patches, num_frames)
            elif self.att_type == 'frame_agg':
                mask = self._generate_frame_agg_causal_mask(
                    num_patches, num_frames
                )
            self.register_buffer('causal_mask', mask)

    def _generate_causal_mask(self, num_patches, num_frames):
        """Generate block-causal mask: tokens in frame t can attend to frames 0..t."""
        total_tokens = num_patches * num_frames
        mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool)

        for t in range(num_frames):
            row_start = t * num_patches
            row_end = (t + 1) * num_patches
            col_end = (
                t + 1
            ) * num_patches  # Can attend up to and including frame t
            mask[row_start:row_end, :col_end] = True

        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T*P, T*P)

    def _generate_frame_agg_causal_mask(self, num_patches, num_frames):
        """Generate causal mask for frame aggregation: query t attends to patches from frames 0..t."""
        mask = torch.zeros(
            num_frames, num_frames * num_patches, dtype=torch.bool
        )

        for t in range(num_frames):
            col_end = (t + 1) * num_patches
            mask[t, :col_end] = True

        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T*P)

    def forward(self, x, c=None):
        B, N, C = x.size()
        x = self.norm(x)
        if self.att_type == 'cross':
            c = self.norm_c(c)
            q_in = x
            kv_in = c
        elif self.att_type == 'frame_agg':
            # Compute actual number of frames from input (supports variable-length sequences)
            actual_frames = N // self.num_patches
            q_in = self.frame_tokens[:, :actual_frames, :].expand(
                B, -1, -1
            )  # (B, actual_frames, dim)
            kv_in = x  # (B, actual_frames*P, dim)
        else:  # self.att_type == "self"
            q_in = x
            kv_in = x

        # q, k, v: (B, heads, T, dim_head)
        q = self.to_q(q_in)
        k, v = self.to_kv(kv_in).chunk(2, dim=-1)
        q, k, v = (
            rearrange(t, 'b n (h d) -> b h n d', h=self.heads)
            for t in (q, k, v)
        )

        # Apply causal mask if enabled
        if self.causal:
            attn_mask = self.causal_mask
            if self.att_type == 'self':
                attn_mask = attn_mask[:, :, :N, :N]
            elif self.att_type == 'frame_agg':
                actual_frames = N // self.num_patches
                attn_mask = attn_mask[:, :, :actual_frames, :N]
        else:
            attn_mask = None

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class Transformer(nn.Module):
    """
    Goal-conditioned Transformer with causal masking and per-frame outputs.
    Alternates between self-attention and cross-attention, ends with frame aggregation.
    """

    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        num_patches=1,
        num_frames=1,
        causal=True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for i in range(depth):
            if i == depth - 1:  # last layer: frame-wise aggregation (T*P -> T)
                att_type = 'frame_agg'
            elif i % 2 == 0:
                att_type = 'self'
            else:
                att_type = 'cross'
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            num_patches=num_patches,
                            num_frames=num_frames,
                            att_type=att_type,
                            causal=causal,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x, g):
        """
        Args:
            x: (B, T*P, dim)
            g: (B, P, dim)
        Returns:
            out: (B, T, dim) - one embedding per frame
        """
        for i, (attn, ff) in enumerate(self.layers):
            if (
                i == len(self.layers) - 1
            ):  # frame aggregation layer - no residual (dimension changes)
                x = attn(x)
                x = ff(x)
            elif i % 2 == 0:  # self-attention with causal masking
                x = attn(x) + x
                x = ff(x) + x
            else:  # cross-attention goal conditioning
                x = attn(x, g) + x
                x = ff(x) + x

        return self.norm(x)


class SelfAttentionTransformer(nn.Module):
    """
    Transformer with only self-attention (no cross-attention).
    Used for embedding state and goal separately in metric-based value functions.
    """

    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        num_patches=1,
        num_frames=1,
        causal=True,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.norm = nn.LayerNorm(dim)

        self.layers = nn.ModuleList([])
        for i in range(depth):
            if i == depth - 1:  # last layer: frame-wise aggregation (T*P -> T)
                att_type = 'frame_agg'
            else:
                att_type = 'self'
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            num_patches=num_patches,
                            num_frames=num_frames,
                            att_type=att_type,
                            causal=causal,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        """
        Process state sequence with causal self-attention.
        Args:
            x: (B, T*P, dim)
        Returns:
            out: (B, T, dim) - one embedding per frame
        """
        for i, (attn, ff) in enumerate(self.layers):
            if i == len(self.layers) - 1:  # frame aggregation layer
                x = attn(x)
                x = ff(x)
            else:  # self-attention
                x = attn(x) + x
                x = ff(x) + x

        return self.norm(x)


class MetricValuePredictor(nn.Module):
    """
    Value predictor using L2 distance in learned embedding space.

    V(s, g) = -||φ(s) - φ(g)||₂

    This architecture embeds state and goal separately (no cross-attention)
    and computes value as the negative L2 distance between embeddings.
    """

    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        embed_dim=64,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        causal=True,
    ):
        super().__init__()

        self.num_patches = num_patches
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # Positional embeddings for state (multiple frames)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * num_patches, dim)
        )
        # Positional embeddings for goal (single frame)
        self.pos_embedding_goal = nn.Parameter(
            torch.randn(1, num_patches, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)

        # Self-attention transformer (shared for state and goal)
        self.transformer = SelfAttentionTransformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            num_patches,
            num_frames,
            causal=causal,
        )

        # Project to learned embedding space
        self.embed_proj = nn.Linear(dim, embed_dim)

    def forward(self, x, g):
        """
        Compute value as negative L2 distance in embedding space.

        Args:
            x: (B, T*P, dim) - observation embeddings
            g: (B, P, dim) - goal embeddings
        Returns:
            value: (B, T, 1) - negative L2 distance per frame
        """
        # Add positional embeddings
        x = x + self.pos_embedding[:, : x.shape[1]]
        g = g + self.pos_embedding[:, : g.shape[1]]
        x = self.dropout(x)
        g = self.dropout(g)

        # Embed state and goal separately through transformer
        x_embed = self.transformer(x)  # (B, T, dim)
        g_embed = self.transformer(g)  # (B, 1, dim)

        # Project to learned embedding space
        x_embed = self.embed_proj(x_embed)  # (B, T, embed_dim)
        g_embed = self.embed_proj(g_embed)  # (B, 1, embed_dim)

        # Compute negative L2 distance: V(s, g) = -||φ(s) - φ(g)||
        diff = x_embed - g_embed
        squared_dist = (diff**2).sum(dim=-1, keepdim=True)
        value = -torch.sqrt(torch.clamp(squared_dist, min=1e-6))  # (B, T, 1)

        return value


class DoubleValuePredictor(nn.Module):
    """
    General double value predictor wrapper for double Q-learning style training.

    Wraps two independent copies of any value predictor network to enable:
    - Minimum of target values for computing Q targets (reduces overestimation)
    - Separate expectile losses for each network

    Usage with TeacherStudentWrapper:
        The wrapper will create EMA copies of both networks automatically.
        forward_student() and forward_teacher() will return (v1, v2) tuples.

    Args:
        value_predictor_cls: The class of the value predictor to wrap (e.g., Predictor)
        **kwargs: Arguments passed to both value predictor instances
    """

    def __init__(self, value_predictor_cls, **kwargs):
        super().__init__()
        self.v1 = value_predictor_cls(**kwargs)
        self.v2 = value_predictor_cls(**kwargs)

    def forward(self, x, g):
        """
        Compute values from both networks.

        Args:
            x: (B, T*P, dim) - observation embeddings
            g: (B, P, dim) - goal embeddings
        Returns:
            (v1, v2): tuple of value tensors from each network
        """
        return self.v1(x, g), self.v2(x, g)


class ExpectileLoss(nn.Module):
    def __init__(self, tau=0.9):
        super().__init__()
        self.tau = tau

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        residual = targets - preds
        # expectile weights
        weight = torch.abs(self.tau - (residual < 0).float())
        loss = (weight * residual.pow(2)).mean()
        return loss


class DoubleExpectileLoss(nn.Module):
    """
    Expectile loss for double value networks.

    Computes the loss as done in https://github.com/seohongpark/HILP/blob/master/hilp_gcrl/src/agents/hilp.py
        adv = q - v_target_avg
        loss = expectile_loss(adv, q1 - v1) + expectile_loss(adv, q2 - v2)
    """

    def __init__(self, tau=0.9):
        super().__init__()
        self.tau = tau

    def expectile_loss(self, adv, residual):
        """Compute expectile loss with advantage-based weights."""
        weight = torch.abs(self.tau - (adv < 0).float())
        return (weight * residual.pow(2)).mean()

    def forward(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor,
        q1: torch.Tensor,
        q2: torch.Tensor,
        adv: torch.Tensor,
    ):
        """
        Compute double expectile loss.

        Args:
            v1, v2: (B, T, 1) predicted values from student networks
            q1, q2: (B, T, 1) target Q values
            adv: (B, T, 1) advantage = q - v_target_avg

        Returns:
            total_loss, (loss1, loss2)
        """
        loss1 = self.expectile_loss(adv, q1 - v1)
        loss2 = self.expectile_loss(adv, q2 - v2)
        return loss1 + loss2, (loss1, loss2)
