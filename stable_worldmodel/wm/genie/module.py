import math
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


def cosine_schedule(u):
    if isinstance(u, Tensor):
        return torch.cos(u * math.pi / 2)
    return math.cos(u * math.pi / 2)


def init_weights(modules: Iterator[nn.Module]):
    std = 0.02
    for module in modules:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_codes: int,
        embed_dim: int,
        commitment_beta: float = 0.25,
        usage_decay: float = 0.99,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.embed_dim = embed_dim
        self.commitment_beta = commitment_beta
        self.usage_decay = usage_decay

        self.codebook = nn.Embedding(num_codes, embed_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        self.register_buffer('usage_count', torch.zeros(num_codes))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        orig_shape = x.shape
        x_ND = x.reshape(-1, self.embed_dim)

        codebook_KD = self.codebook.weight
        x_sq_N1 = x_ND.pow(2).sum(dim=1, keepdim=True)
        codebook_sq_K = codebook_KD.pow(2).sum(dim=1)
        xc_NK = x_ND @ codebook_KD.t()
        dists_NK = x_sq_N1 - 2 * xc_NK + codebook_sq_K

        indices_N = dists_NK.argmin(dim=1)
        quantized_ND = self.codebook(indices_N)

        codebook_loss = F.mse_loss(quantized_ND, x_ND.detach())
        commitment_loss = F.mse_loss(x_ND, quantized_ND.detach())
        loss = codebook_loss + self.commitment_beta * commitment_loss

        quantized_ND = x_ND + (quantized_ND - x_ND).detach()

        quantized = quantized_ND.reshape(orig_shape)
        indices = indices_N.reshape(orig_shape[:-1])

        if self.training:
            with torch.no_grad():
                one_hot_NK = F.one_hot(indices_N, self.num_codes).to(
                    self.usage_count.dtype
                )
                self.usage_count.mul_(self.usage_decay).add_(
                    one_hot_NK.sum(dim=0)
                )

        return quantized, indices, loss

    @torch.no_grad()
    def reinit_dead_codes(self, x: Tensor, threshold: float = 1.0) -> int:
        x_flat = x.reshape(-1, self.embed_dim)
        dead = self.usage_count < threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0
        sampled = torch.randint(0, x_flat.size(0), (n_dead,), device=x.device)
        self.codebook.weight.data[dead] = x_flat[sampled]
        self.usage_count[dead] = float(threshold)
        return n_dead


class SelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        qkv_use_bias: bool = False,
        proj_use_bias: bool = True,
        qk_use_norm: bool = True,
        qk_use_mup: bool = True,
        attn_drop: float = 0.0,
        causal: bool = True,
        max_seq_len: int | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 8 / self.head_dim if qk_use_mup else self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_use_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model, bias=proj_use_bias)
        self.qk_norm = qk_use_norm
        self.causal = causal
        if self.qk_norm:
            self.norm = nn.LayerNorm(self.head_dim, eps=1e-5)
        if self.causal:
            assert max_seq_len is not None, 'causal=True requires max_seq_len'
            mask = torch.triu(
                torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
                diagonal=1,
            )
            self.register_buffer(
                'causal_mask', mask.view(1, 1, max_seq_len, max_seq_len)
            )

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        qkv = self.qkv(x)
        qkv = rearrange(
            qkv, 'B N (three H D) -> three B H N D', three=3, H=self.num_heads
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = self.norm(q).to(dtype=v.dtype)
            k = self.norm(k).to(dtype=v.dtype)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=causal,
            scale=self.scale,
        )
        out = rearrange(out, 'B H N D -> B N (H D)')
        return self.proj(out)


class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        ratio: float = 4.0,
        use_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = int(d_model * ratio)
        self._model = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=use_bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model, bias=use_bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)


class ST_Block(nn.Module):
    """Spatio-temporal transformer block: spatial attn → temporal (causal) attn → MLP,
    each with pre-LayerNorm and residual connections.
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        temporal_dim: int,
        proj_use_bias: bool = True,
        qkv_use_bias: bool = False,
        qk_use_norm: bool = True,
        qk_use_mup: bool = True,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        mlp_use_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-5)

        attn_args = dict(
            num_heads=num_heads,
            d_model=d_model,
            qkv_use_bias=qkv_use_bias,
            proj_use_bias=proj_use_bias,
            qk_use_norm=qk_use_norm,
            qk_use_mup=qk_use_mup,
            attn_drop=attn_dropout,
        )
        self.spatial_attn = SelfAttention(**attn_args, causal=False)
        self.temporal_attn = SelfAttention(
            **attn_args, causal=True, max_seq_len=temporal_dim
        )
        self.mlp = MLP(
            d_model=d_model,
            ratio=mlp_ratio,
            use_bias=mlp_use_bias,
            dropout=mlp_dropout,
        )

    def forward(self, x_TSC: Tensor) -> Tensor:
        T, S = x_TSC.size(1), x_TSC.size(2)
        x_SC = rearrange(x_TSC, 'B T S C -> (B T) S C')
        x_SC = x_SC + self.spatial_attn(self.norm1(x_SC), causal=False)
        x_TC = rearrange(x_SC, '(B T) S C -> (B S) T C', T=T)
        x_TC = x_TC + self.temporal_attn(self.norm2(x_TC), causal=True)
        x_TC = x_TC + self.mlp(self.norm3(x_TC))
        return rearrange(x_TC, '(B S) T C -> B T S C', S=S)


class ST_Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        temporal_dim: int,
        qkv_use_bias: bool = False,
        proj_use_bias: bool = True,
        qk_use_norm: bool = True,
        qk_use_mup: bool = True,
        attn_dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        mlp_use_bias: bool = True,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        block_args = dict(
            num_heads=num_heads,
            d_model=d_model,
            temporal_dim=temporal_dim,
            qkv_use_bias=qkv_use_bias,
            proj_use_bias=proj_use_bias,
            qk_use_norm=qk_use_norm,
            qk_use_mup=qk_use_mup,
            attn_dropout=attn_dropout,
            mlp_ratio=mlp_ratio,
            mlp_use_bias=mlp_use_bias,
            mlp_dropout=mlp_dropout,
        )
        self.blocks = nn.ModuleList(
            [ST_Block(**block_args) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class ST_MaskGIT(nn.Module):
    """MaskGIT dynamics head: predicts image tokens from partially-masked context,
    conditioned on latent-action embeddings.
    """

    def __init__(
        self,
        decoder: ST_Transformer,
        spatial_dim: int,
        temporal_dim: int,
        d_model: int,
        image_vocab_size: int,
        action_embed_dim: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.h = self.w = math.isqrt(spatial_dim)
        assert self.h * self.w == spatial_dim, 'spatial_dim must be a square'

        self.decoder = decoder
        self.pos_embed_TSC = nn.Parameter(
            torch.zeros(1, temporal_dim, spatial_dim, d_model)
        )
        self.token_embed_VD = nn.Embedding(image_vocab_size, d_model)
        self.mask_token_embed_1D = nn.Parameter(torch.zeros(1, d_model))
        self.image_vocab_size = image_vocab_size
        self.mask_token_id = image_vocab_size
        self.out_x_proj = nn.Linear(d_model, image_vocab_size)
        self.action_proj_ED = nn.Linear(action_embed_dim, d_model)
        init_weights(self.modules())

    def compute_logits(
        self,
        input_ids_TS: Tensor,
        actions_T: Tensor | None = None,
    ) -> Tensor:
        T = input_ids_TS.size(1)
        is_mask = input_ids_TS == self.mask_token_id
        safe_ids = input_ids_TS.masked_fill(is_mask, 0)
        x_TSC = self.token_embed_VD(safe_ids)
        x_TSC = torch.where(
            is_mask.unsqueeze(-1), self.mask_token_embed_1D, x_TSC
        )
        x_TSC = x_TSC + self.pos_embed_TSC

        if actions_T is not None:
            a_TC = self.action_proj_ED(actions_T)
            if a_TC.size(1) == T - 1:
                pad = torch.zeros_like(a_TC[:, :1])
                a_TC = torch.cat([a_TC, pad], dim=1)
            x_TSC = x_TSC + a_TC.unsqueeze(2)

        x_TSC = self.decoder(x_TSC)
        logits_TSV = self.out_x_proj(x_TSC)
        return rearrange(
            logits_TSV, 'B T (H W) C -> B C T H W', H=self.h, W=self.w
        )

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor,
        actions_T: Tensor | None = None,
    ) -> dict:
        logits_BCTHW = self.compute_logits(input_ids, actions_T)
        labels_THW = rearrange(
            labels, 'B T (H W) -> B T H W', H=self.h, W=self.w
        )
        input_THW = rearrange(
            input_ids, 'B T (H W) -> B T H W', H=self.h, W=self.w
        )

        logits_BCTHW = logits_BCTHW[:, :, 1:]
        labels_THW = labels_THW[:, 1:]
        input_THW = input_THW[:, 1:]

        loss_THW = F.cross_entropy(logits_BCTHW, labels_THW, reduction='none')
        correct_THW = logits_BCTHW.argmax(dim=1) == labels_THW
        mask_THW = input_THW == self.mask_token_id
        num_masked = mask_THW.sum().clamp(min=1)

        loss = (loss_THW * mask_THW).sum() / num_masked
        acc = (correct_THW * mask_THW).float().sum() / num_masked
        return {'loss': loss, 'acc': acc, 'logits': logits_BCTHW}

    @torch.no_grad()
    def sample_frame(
        self,
        context_TS: Tensor,
        frame_idx: int,
        num_steps: int,
        actions_T: Tensor | None = None,
        temperature: float = 1.0,
        unmask_mode: str = 'random',
    ) -> Tensor:
        assert frame_idx >= 1
        assert torch.all(context_TS[:, frame_idx:] == self.mask_token_id)

        B, T, S = context_TS.shape
        working = context_TS.clone()
        unmasked = torch.zeros(
            B, S, dtype=torch.bool, device=context_TS.device
        )

        for step in range(num_steps):
            logits_BCTHW = self.compute_logits(working, actions_T)
            logits_BCS = rearrange(
                logits_BCTHW[:, :, frame_idx], 'B C H W -> B C (H W)'
            )

            if temperature <= 1e-8:
                samples_S = logits_BCS.argmax(dim=1)
                probs_BCS = torch.softmax(logits_BCS, dim=1)
            else:
                probs_BCS = torch.softmax(logits_BCS / temperature, dim=1)
                samples_S = torch.distributions.Categorical(
                    probs=rearrange(probs_BCS, 'B C S -> B S C')
                ).sample()

            confidences_S = torch.gather(
                probs_BCS, 1, samples_S.unsqueeze(1)
            ).squeeze(1)
            samples_S = torch.where(unmasked, working[:, frame_idx], samples_S)

            if step != num_steps - 1:
                n = math.ceil(cosine_schedule((step + 1) / num_steps) * S)
                if unmask_mode == 'greedy':
                    scores_S = confidences_S
                else:
                    scores_S = torch.rand_like(confidences_S)
                scores_S = scores_S.masked_fill(unmasked, torch.inf)
                order = torch.argsort(scores_S, dim=1)
                unmasked = unmasked.scatter(1, order[:, n:], True)
                samples_S = samples_S.scatter(
                    1, order[:, :n], self.mask_token_id
                )

            working[:, frame_idx] = samples_S

        return samples_S

    @torch.no_grad()
    def rollout(
        self,
        prompt_TS: Tensor,
        num_new_frames: int,
        num_steps: int,
        actions_T: Tensor | None = None,
        temperature: float = 1.0,
        unmask_mode: str = 'random',
    ) -> Tensor:
        B, T_prompt, S = prompt_TS.shape
        T_total = T_prompt + num_new_frames
        T_model = self.pos_embed_TSC.size(1)
        assert T_total == T_model

        masked_tail = torch.full(
            (B, num_new_frames, S),
            self.mask_token_id,
            dtype=prompt_TS.dtype,
            device=prompt_TS.device,
        )
        full_TS = torch.cat([prompt_TS, masked_tail], dim=1)

        for frame_idx in range(T_prompt, T_total):
            sample_S = self.sample_frame(
                full_TS,
                frame_idx=frame_idx,
                num_steps=num_steps,
                actions_T=actions_T,
                temperature=temperature,
                unmask_mode=unmask_mode,
            )
            full_TS[:, frame_idx] = sample_S
        return full_TS


class ST_ViViT(nn.Module):
    """Video tokenizer: ST-Transformer encoder → VQ → ST-Transformer decoder.

    Encodes (B, T, H, W, c) pixel video into discrete tokens of shape (B, T, S)
    where S = (H/patch_size) * (W/patch_size).
    """

    def __init__(
        self,
        encoder: ST_Transformer,
        decoder: ST_Transformer,
        vq: VectorQuantizer,
        patch_size: int,
        height: int,
        width: int,
        channels: int,
        temporal_dim: int,
        d_model: int,
    ):
        super().__init__()
        assert height % patch_size == 0 and width % patch_size == 0
        self.encoder = encoder
        self.decoder = decoder
        self.vq = vq
        self.patch_size = patch_size
        self.h_grid = height // patch_size
        self.w_grid = width // patch_size
        self.channels = channels
        S = self.h_grid * self.w_grid

        self.patchify = nn.Conv2d(
            channels, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.enc_pos_embed_TSC = nn.Parameter(
            torch.zeros(1, temporal_dim, S, d_model)
        )
        self.dec_pos_embed_TSC = nn.Parameter(
            torch.zeros(1, temporal_dim, S, d_model)
        )
        self.pre_vq_proj = nn.Linear(d_model, vq.embed_dim)
        self.post_vq_proj = nn.Linear(vq.embed_dim, d_model)
        self.unpatchify = nn.Linear(
            d_model, patch_size * patch_size * channels
        )
        init_weights(self.modules())

    def encode(self, video: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, T = video.size(0), video.size(1)
        video_NcHW = rearrange(video, 'B T H W c -> (B T) c H W')
        patches_NChw = self.patchify(video_NcHW)
        patches_TSC = rearrange(
            patches_NChw, '(B T) C h w -> B T (h w) C', B=B, T=T
        )

        z_TSC = patches_TSC + self.enc_pos_embed_TSC
        z_TSC = self.encoder(z_TSC)
        z_TSE = self.pre_vq_proj(z_TSC)
        z_q_TSE, indices_TS, vq_loss = self.vq(z_TSE)
        z_q_TSC = self.post_vq_proj(z_q_TSE)
        return indices_TS, z_q_TSC, vq_loss

    def decode(self, z_q_TSC: Tensor) -> Tensor:
        z_TSC = z_q_TSC + self.dec_pos_embed_TSC
        z_TSC = self.decoder(z_TSC)
        patches_TSP = self.unpatchify(z_TSC)
        return rearrange(
            patches_TSP,
            'B T (h w) (p1 p2 c) -> B T (h p1) (w p2) c',
            h=self.h_grid,
            w=self.w_grid,
            p1=self.patch_size,
            p2=self.patch_size,
        )

    def decode_from_indices(self, indices_TS: Tensor) -> Tensor:
        z_q_TSE = self.vq.codebook(indices_TS)
        z_q_TSC = self.post_vq_proj(z_q_TSE)
        return self.decode(z_q_TSC)

    def forward(self, video: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        indices_TS, z_q_TSC, vq_loss = self.encode(video)
        video_hat = self.decode(z_q_TSC)
        return video_hat, indices_TS, vq_loss


class LAM(nn.Module):
    """Latent Action Model — a VQ-VAE over (video to discrete action id per frame
    transition to video prediction). At inference only `vq.codebook` is used;
    the encoder is discarded and action codes come from the planner.
    """

    def __init__(
        self,
        encoder: ST_Transformer,
        decoder: ST_Transformer,
        vq: VectorQuantizer,
        patch_size: int,
        height: int,
        width: int,
        channels: int,
        temporal_dim: int,
        d_model: int,
    ):
        super().__init__()
        assert height % patch_size == 0 and width % patch_size == 0
        self.encoder = encoder
        self.decoder = decoder
        self.vq = vq
        self.patch_size = patch_size
        self.h_grid = height // patch_size
        self.w_grid = width // patch_size
        self.channels = channels
        self.d_model = d_model
        S = self.h_grid * self.w_grid

        self.enc_patchify = nn.Conv2d(
            channels, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.dec_patchify = nn.Conv2d(
            channels, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.enc_pos_embed_TSC = nn.Parameter(
            torch.zeros(1, temporal_dim, S, d_model)
        )
        self.dec_pos_embed_TSC = nn.Parameter(
            torch.zeros(1, temporal_dim - 1, S, d_model)
        )
        self.pre_vq_proj = nn.Linear(d_model, vq.embed_dim)
        self.action_to_dmodel = nn.Linear(vq.embed_dim, d_model)
        self.unpatchify = nn.Linear(
            d_model, patch_size * patch_size * channels
        )
        init_weights(self.modules())

    def extract_actions(self, video: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, T = video.size(0), video.size(1)
        video_NcHW = rearrange(video, 'B T H W c -> (B T) c H W')
        patches_NChw = self.enc_patchify(video_NcHW)
        patches_TSC = rearrange(
            patches_NChw, '(B T) C h w -> B T (h w) C', B=B, T=T
        )

        z_TSC = patches_TSC + self.enc_pos_embed_TSC
        z_TSC = self.encoder(z_TSC)
        z_TC = z_TSC.mean(dim=2)
        action_TC = z_TC[:, 1:, :]

        action_TE = self.pre_vq_proj(action_TC)
        action_q_TE, action_ids, vq_loss = self.vq(action_TE)
        return action_ids, action_q_TE, vq_loss

    def predict_next_frames(
        self, prev_video: Tensor, action_q_TE: Tensor
    ) -> Tensor:
        B, Tm1 = prev_video.size(0), prev_video.size(1)
        video_NcHW = rearrange(prev_video, 'B T H W c -> (B T) c H W')
        patches_NChw = self.dec_patchify(video_NcHW)
        patches_TSC = rearrange(
            patches_NChw, '(B T) C h w -> B T (h w) C', B=B, T=Tm1
        )

        action_TC = self.action_to_dmodel(action_q_TE)
        patches_TSC = patches_TSC + action_TC.unsqueeze(2)

        z_TSC = patches_TSC + self.dec_pos_embed_TSC
        z_TSC = self.decoder(z_TSC)
        patches_TSP = self.unpatchify(z_TSC)
        return rearrange(
            patches_TSP,
            'B T (h w) (p1 p2 c) -> B T (h p1) (w p2) c',
            h=self.h_grid,
            w=self.w_grid,
            p1=self.patch_size,
            p2=self.patch_size,
        )

    def forward(self, video: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        action_ids, action_q_TE, vq_loss = self.extract_actions(video)
        next_pred = self.predict_next_frames(video[:, :-1], action_q_TE)
        return next_pred, action_ids, vq_loss


class ActionEncoder(nn.Module):
    """MLP from raw env actions to LAM-compatible action embeddings.

    Trained as an auxiliary head during the joint phase against
    `LAM.extract_actions(video).detach()`; at inference the planner can sample
    raw env actions and project them through this MLP for use with the
    dynamics model (i.e. Genie(action_mode="raw")).
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
        return self.net(x)
