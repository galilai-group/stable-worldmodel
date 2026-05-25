from einops import rearrange
import torch
from torch import nn, Tensor


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
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 8 / self.head_dim if qk_use_mup else self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_use_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model, bias=proj_use_bias)
        self.qk_norm = qk_use_norm

        if self.qk_norm:
            # use qk normalization using fp32 as in LN
            self.norm = nn.LayerNorm(self.head_dim, eps=1e-05)
        
        if self.causal:
            self.causal_mask = self._generate_causal_mask(d_model)
            self.register_buffer('causal_mask', self.causal_mask)

    def _generate_causal_mask(self, n_tokens: int) -> Tensor:
        mask = torch.tril(torch.ones(n_tokens, n_tokens))
        return rearrange(mask, 'N N -> 1 1 N N')

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        qkv: Tensor = self.qkv(x)
        qkv = rearrange(qkv, 'B N (3 H D) -> 3 B H N D', H=self.num_heads)
        q,k,v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            # cast back from fp32 to v.dtype (bf16 etc)
            q: Tensor = self.norm(q).to(dtype=v.dtype)
            k: Tensor = self.norm(k).to(dtype=v.dtype)

        q *= self.scale
        attn = q @ rearrange(k, 'B H N D -> B H D N')

        if causal:
            attn = attn.masked_fill(self.causal_mask, -torch.finfo(attn.dtype).max)

        attn = attn.softmax(dim=-1)

        x = rearrange((attn @ v), 'B H N D -> B N (H D)')
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



class STBlock(nn.Module):
    # See Figure 4 of https://arxiv.org/pdf/2402.15391.pdf
    def __init__(
        self,
        num_heads: int,
        d_model: int,
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
        self.norm1 = nn.Identity() if qk_use_norm else nn.LayerNorm(d_model, eps=1e-05)
        
        attn_args = dict(
            num_heads=num_heads,
            d_model=d_model,
            qkv_use_bias=qkv_use_bias,
            proj_use_bias=proj_use_bias,
            qk_use_norm=qk_use_norm,
            qk_use_mup=qk_use_mup,
            attn_drop=attn_dropout,
        )

        self.spatial_attn = SelfAttention(**attn_args)
        self.temporal_attn = SelfAttention(**attn_args)
        self.causal_mask = SelfAttention._generate_causal_mask(d_model)
        
        self.norm2 = nn.Identity() if qk_use_norm else nn.LayerNorm(d_model, eps=1e-05)
        self.mlp = MLP(d_model=d_model, ratio=mlp_ratio, use_bias=mlp_use_bias, dropout=mlp_dropout)


    def forward(self, x_TSC: Tensor) -> Tensor:
        # spatial attn (no ffw mlp)
        T, S = x_TSC.size(1), x_TSC.size(2)
        x_SC = rearrange(x_TSC, 'B T S C -> (B T) S C')
        x_SC = x_SC + self.spatial_attn(self.norm1(x_SC), causal=False)

        # temporal attn (causal)
        x_TC = rearrange(x_SC, '(B T) S C -> (B S) T C', T=T)
        x_TC = x_TC + self.temporal_attn(x_TC, causal=True)

        # mlp
        x_TC = x_TC + self.mlp(self.norm2(x_TC))
        x_TSC = rearrange(x_TC, '(B S) T C -> B T S C', S=S)
        return x_TSC


class ST_TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
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
            qkv_use_bias=qkv_use_bias,
            proj_use_bias=proj_use_bias,
            qk_use_norm=qk_use_norm,
            qk_use_mup=qk_use_mup,
            attn_dropout=attn_dropout,
            mlp_ratio=mlp_ratio,
            mlp_use_bias=mlp_use_bias,
            mlp_dropout=mlp_dropout,
        )
        self.blocks = nn.ModuleList([STBlock(**st_block_args) for _ in range(num_layers)])

    def forward(self, x: Tensor) -> Tensor:
        
        for blk in self.blocks:
            x = blk(x)

        return x