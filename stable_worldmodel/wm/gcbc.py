import torch
import torch.nn.functional as F

# from torchvision import transforms
from einops import rearrange, repeat
from torch import nn


# TODO encode is very similar to the one in prejepa.py - consider refactoring
# TODO models are very similar to the ones in prejepa.py - consider refactoring


class GCBC(torch.nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        extra_encoders=None,
        history_size=3,
        num_pred=1,
        interpolate_pos_encoding=True,
    ):
        super().__init__()

        self.backbone = encoder
        self.predictor = predictor
        self.extra_encoders = extra_encoders or {}
        self.history_size = history_size
        self.num_pred = num_pred

        self.interpolate_pos_encoding = interpolate_pos_encoding

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
        pixels_embed = self.backbone(pixels, **kwargs)

        if hasattr(pixels_embed, 'last_hidden_state'):
            pixels_embed = pixels_embed.last_hidden_state
            pixels_embed = pixels_embed[:, 1:, :]  # drop cls token
        else:
            pixels_embed = pixels_embed.logits.unsqueeze(
                1
            )  # (B*T, 1, emb_dim)

        pixels_embed = rearrange(
            pixels_embed.detach(), '(b t) p d -> b t p d', b=B
        )

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

            frame_embed = self.backbone(frames, **kwargs)  # (B, 1, P, emb_dim)
            frame_embed = frame_embed.last_hidden_state
            pixels_embeddings.append(frame_embed)

        pixels_embed = torch.stack(
            pixels_embeddings, dim=1
        )  # (B, T, P, emb_dim)

        return pixels_embed

    def predict(self, embedding, embedding_goal):
        """predict actions
        Args:
            embedding: (B, T, P, d)
            embedding_goal: (B, 1, P, d)
        Returns:
            preds: (B, effective_action_dim)
        """

        embedding = rearrange(embedding, 'b t p d -> b (t p) d')
        embedding_goal = rearrange(embedding_goal, 'b t p d -> b (t p) d')
        preds = self.predictor(embedding, embedding_goal)

        return preds

    def get_action(self, info):
        """Get action given observation and goal."""
        # first encode observation
        info = self.encode(
            info, pixels_key='pixels', emb_keys=['proprio'], target='embed'
        )
        # encode goal
        info = self.encode(
            info,
            pixels_key='goal',
            emb_keys=['proprio'],
            prefix='goal_',
            target='goal_embed',
        )
        # then predict action
        actions = self.predict(info['embed'], info['goal_embed'])
        actions = actions.squeeze(1)
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
        action_dim,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()

        self.num_patches = num_patches
        self.num_frames = num_frames

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
        )
        self.action_proj = nn.Linear(dim, action_dim)

    def forward(
        self, x, g
    ):  # x: (b, window_size * H/patch_size * W/patch_size, 384)
        # prepare input for transformer
        x = x + self.pos_embedding[:, : x.shape[1]]
        g = g + self.pos_embedding_goal[:, : g.shape[1]]
        x = self.dropout(x)
        # transformer forward
        x = self.transformer(x, g)
        # project to action dim
        x = self.action_proj(x)
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
    ):
        super().__init__()
        assert att_type in {'self', 'cross', 'agg'}, (
            'attention type must be either self, cross, or agg (aggregation)'
        )
        self.att_type = att_type
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
        if self.att_type == 'agg':
            self.out_token = nn.Parameter(0.02 * torch.randn(1, 1, dim))

        # self.register_buffer("bias", self.generate_mask_matrix(num_patches, num_frames))

    def forward(self, x, c=None):
        B, T, C = x.size()
        x = self.norm(x)
        if self.att_type == 'cross':
            c = self.norm_c(c)
            q_in = x
            kv_in = c
        elif self.att_type == 'agg':
            q_in = self.out_token.repeat(B, 1, 1)
            kv_in = x
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

        # attn_mask = self.bias[:, :, :T, :T] == 1  # bool mask
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

    def generate_mask_matrix(self, npatch, nwindow):
        zeros = torch.zeros(npatch, npatch)
        ones = torch.ones(npatch, npatch)
        if self.att_type == 'cross':
            mask = torch.zeros(npatch, npatch).unsqueeze(0).unsqueeze(0)
        else:
            rows = []
            for i in range(nwindow):
                row = torch.cat(
                    [ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1
                )
                rows.append(row)
            mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
        return mask


class Transformer(nn.Module):
    """
    Goal-conditioned Transformer that alternates between self-attention and cross-attention blocks.
    The last layer is an attention-based aggregation block.
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
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for i in range(depth):
            if i == depth - 1:  # last layer is an aggregation layer
                att_type = 'agg'
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
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x, g):
        for i, (attn, ff) in enumerate(self.layers):
            if i == len(self.layers) - 1:  # last layer is an aggregation layer
                x = attn(x)
                x = ff(x)
            elif i % 2 == 0:  # self-attention
                x = attn(x) + x
                x = ff(x) + x
            else:  # cross-attention goal conditioning
                x = attn(x, g) + x
                x = ff(x) + x

        return self.norm(x)
