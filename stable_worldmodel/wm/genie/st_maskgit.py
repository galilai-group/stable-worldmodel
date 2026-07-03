from einops import rearrange
import torch
from torch import nn, Tensor
from typing import Optional
import math
import torch.nn.functional as F
from stable_worldmodel.wm.genie.utils import init_weights, cosine_schedule

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
    ) -> None:
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
            # use qk normalization using fp32 as in LN
            self.norm = nn.LayerNorm(self.head_dim, eps=1e-05)

        if self.causal:
            assert max_seq_len is not None, 'causal=True requires max_seq_len'
            mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
            self.register_buffer('causal_mask', mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        qkv: Tensor = self.qkv(x)
        qkv = rearrange(qkv, 'B N (three H D) -> three B H N D', three=3, H=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            # cast back from fp32 to v.dtype (bf16 etc)
            q = self.norm(q).to(dtype=v.dtype)
            k = self.norm(k).to(dtype=v.dtype)

        # PyTorch SDPA (Flash / mem-efficient on Ampere+) — O(N) memory, ~2x speed.
        # `is_causal=True` applies the standard upper-tri causal mask internally.
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=causal,
            scale=self.scale,
        )

        x = rearrange(out, 'B H N D -> B N (H D)')
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        ratio: float = 4.0,
        use_bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
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
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-05)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-05)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-05)

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
        self.temporal_attn = SelfAttention(**attn_args, causal=True, max_seq_len=temporal_dim)

        self.mlp = MLP(d_model=d_model, ratio=mlp_ratio, use_bias=mlp_use_bias, dropout=mlp_dropout)


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
        st_block_args = dict(
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
        self.blocks = nn.ModuleList([ST_Block(**st_block_args) for _ in range(num_layers)])

    def forward(self, x: Tensor) -> Tensor:
        
        for blk in self.blocks:
            x = blk(x)

        return x


class ST_MaskGIT(nn.Module):
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
        assert self.h * self.w == spatial_dim, "spatial_dim must be a square"

        self.decoder = decoder
        self.pos_embed_TSC = nn.Parameter(torch.zeros(1, temporal_dim, spatial_dim, d_model))
        self.token_embed_VD = nn.Embedding(image_vocab_size, d_model)
        self.mask_token_embed_1D = nn.Parameter(torch.zeros(1, d_model))
        self.image_vocab_size = image_vocab_size
        self.mask_token_id = image_vocab_size
        self.out_x_proj = nn.Linear(d_model, image_vocab_size)
        self.action_proj_ED = nn.Linear(action_embed_dim, d_model)
        init_weights(self.modules())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        actions_T: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Training forward pass.

        Args:
            input_ids: (B, T, S) token ids with some positions replaced by self.mask_token_id
            labels:    (B, T, S) unmasked ground-truth token ids
            actions_T: optional (B, T) or (B, T-1) action ids

        Returns:
            dict with keys 'loss', 'acc', 'logits'
            - loss:   scalar, mean cross-entropy over masked positions (excluding frame 0)
            - acc:    scalar, fraction of masked positions predicted correctly
            - logits: (B, C, T-1, H, W) logits for frames 1..T-1
        """
        logits_BCTHW = self.compute_logits(input_ids, actions_T)

        labels_THW = rearrange(labels, "B T (H W) -> B T H W", H=self.h, W=self.w)
        input_THW = rearrange(input_ids, "B T (H W) -> B T H W", H=self.h, W=self.w)

        # Frame 0 is the prompt — never masked, never predicted. Drop it.
        logits_BCTHW = logits_BCTHW[:, :, 1:]
        labels_THW = labels_THW[:, 1:]
        input_THW = input_THW[:, 1:]

        # Per-position CE; only count where the input was actually masked.
        loss_THW = F.cross_entropy(logits_BCTHW, labels_THW, reduction="none")
        correct_THW = logits_BCTHW.argmax(dim=1) == labels_THW

        mask_THW = input_THW == self.mask_token_id
        num_masked = mask_THW.sum().clamp(min=1)

        loss = (loss_THW * mask_THW).sum() / num_masked
        acc = (correct_THW * mask_THW).float().sum() / num_masked

        return {"loss": loss, "acc": acc, "logits": logits_BCTHW}

    def compute_logits(
        self,
        input_ids_TS: torch.Tensor,
        actions_T: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        T = input_ids_TS.size(1)
        is_mask = input_ids_TS == self.mask_token_id
        safe_ids = input_ids_TS.masked_fill(is_mask, 0)
        x_TSC = self.token_embed_VD(safe_ids)
        x_TSC = torch.where(is_mask.unsqueeze(-1), self.mask_token_embed_1D, x_TSC)
        x_TSC = x_TSC + self.pos_embed_TSC

        # broadcast to spatial dim
        if actions_T is not None:
            # actions_T: (B, T, E) or (B, T-1, E) quantized embeddings from LAM (caller has .detach()'d)
            a_TC = self.action_proj_ED(actions_T)
            if a_TC.size(1) == T - 1:
                pad = torch.zeros_like(a_TC[:, :1])
                a_TC = torch.cat([a_TC, pad], dim=1)
            x_TSC = x_TSC + a_TC.unsqueeze(2)

        x_TSC = self.decoder(x_TSC)
        logits_TSV = self.out_x_proj(x_TSC)

        return rearrange(logits_TSV, "B T (H W) C -> B C T H W", H=self.h, W=self.w)

    @torch.no_grad()
    def sample_frame(
        self,
        context_TS: torch.Tensor,
        frame_idx: int,
        num_steps: int,
        actions_T: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        unmask_mode: str = "random",
    ) -> torch.Tensor:
        """
        Fill in the tokens of frame `frame_idx` via MaskGIT iterative refinement.
        Does not mutate `context_TS`.

        Args:
            context_TS: (B, T, S) token ids; frame_idx and later must be fully masked
            frame_idx:  which frame to fill in (>= 1)
            num_steps:  number of MaskGIT iterations
            actions_T:  optional (B, T) or (B, T-1) action ids
            temperature: <= 1e-8 means greedy (argmax)
            unmask_mode: "random" (default, as in 1xgpt) or "greedy"

        Returns:
            sample_S: (B, S) predicted token ids for frame_idx
        """
        assert frame_idx >= 1, "frame 0 is the prompt; cannot sample it"
        assert torch.all(context_TS[:, frame_idx:] == self.mask_token_id), \
            f"frame {frame_idx} and later must be fully masked"

        B, T, S = context_TS.shape
        working = context_TS.clone()
        unmasked = torch.zeros(B, S, dtype=torch.bool, device=context_TS.device)

        for step in range(num_steps):
            logits_BCTHW = self.compute_logits(working, actions_T)
            logits_BCS = rearrange(logits_BCTHW[:, :, frame_idx], "B C H W -> B C (H W)")

            if temperature <= 1e-8:
                samples_S = logits_BCS.argmax(dim=1)
                probs_BCS = torch.softmax(logits_BCS, dim=1)
            else:
                probs_BCS = torch.softmax(logits_BCS / temperature, dim=1)
                samples_S = torch.distributions.Categorical(
                    probs=rearrange(probs_BCS, "B C S -> B S C")
                ).sample()

            confidences_S = torch.gather(probs_BCS, 1, samples_S.unsqueeze(1)).squeeze(1)

            # Preserve positions that were settled in earlier steps
            samples_S = torch.where(unmasked, working[:, frame_idx], samples_S)

            if step != num_steps - 1:
                # cosine schedule: how many positions to keep masked for the next round
                n = math.ceil(cosine_schedule((step + 1) / num_steps) * S)

                if unmask_mode == "greedy":
                    scores_S = confidences_S
                elif unmask_mode == "random":
                    scores_S = torch.rand_like(confidences_S)
                else:
                    raise ValueError(f"unmask_mode must be 'random' or 'greedy', got {unmask_mode!r}")

                # Already-settled positions get +inf so they're never re-masked
                scores_S = scores_S.masked_fill(unmasked, torch.inf)
                order = torch.argsort(scores_S, dim=1)         # ascending
                unmasked = unmasked.scatter(1, order[:, n:], True)
                samples_S = samples_S.scatter(1, order[:, :n], self.mask_token_id)

            working[:, frame_idx] = samples_S

        return samples_S

    @torch.no_grad()
    def rollout(
        self,
        prompt_TS: torch.Tensor,
        num_new_frames: int,
        num_steps: int,
        actions_T: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        unmask_mode: str = "random",
    ) -> torch.Tensor:
        """
        Autoregressively generate `num_new_frames` frames after the prompt.

        Args:
            prompt_TS:       (B, T_prompt, S) prompt token ids
            num_new_frames:  how many frames to generate
            num_steps:       MaskGIT iterations per frame
            actions_T:       optional action ids for the full context
            temperature:     sampling temperature
            unmask_mode:     "random" or "greedy"

        Returns:
            full_TS: (B, T_prompt + num_new_frames, S) prompt followed by generated frames
        """
        B, T_prompt, S = prompt_TS.shape
        T_total = T_prompt + num_new_frames
        T_model = self.pos_embed_TSC.size(1)
        assert T_total == T_model, \
            f"T_prompt ({T_prompt}) + num_new_frames ({num_new_frames}) " \
            f"must equal model temporal_dim ({T_model})"

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