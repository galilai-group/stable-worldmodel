import torch

from torch import nn
from torch.nn import functional as F
from einops import rearrange


# fmt: off
BACKBONE_ALIASES = {
    # DINOv2
    "dinov2_small":        "facebook/dinov2-small",
    "dinov2_base":         "facebook/dinov2-base",
    "dinov2_large":        "facebook/dinov2-large",
    "dinov2_giant":        "facebook/dinov2-giant",
    # DINOv3
    "dinov3_small":        "facebook/dinov3-vits16-pretrain-lvd1689m",
    # DINO (v1)
    "dino_vits16":         "facebook/dino-vits16",
    "dino_vits8":          "facebook/dino-vits8",
    "dino_vitb16":         "facebook/dino-vitb16",
    "dino_vitb8":          "facebook/dino-vitb8",
    # MAE
    "mae_base":            "facebook/vit-mae-base",
    "mae_large":           "facebook/vit-mae-large",
    "mae_huge":            "facebook/vit-mae-huge",
    # I-JEPA
    "ijepa_huge":          "facebook/ijepa-huge-patch14-224",
    # VJEPA2
    "vjepa2_large":        "facebook/vjepa2-vit-l",
    "vjepa2_huge":         "facebook/vjepa2-vit-h",
    # WebSSL
    "webssl_large":        "facebook/webssl-vith14-5b",
    # ViT
    "vit_base_patch16":    "google/vit-base-patch16-224",
    "vit_large_patch16":   "google/vit-large-patch16-224",
    # SigLIP2
    "siglip2_base_224":    "google/siglip2-base-patch16-224",
    "siglip2_large_256":   "google/siglip2-large-patch16-256",
    # ResNet
    "resnet_50":           "microsoft/resnet-50",
    "resnet_101":          "microsoft/resnet-101",
}
# fmt: on


def create_backbone(name: str) -> nn.Module:
    """Load a pretrained HuggingFace vision encoder.

    ``name`` can be either a short alias from ``BACKBONE_ALIASES``
    (e.g. ``dinov2_small``) or a full HuggingFace model id.
    """
    from transformers import AutoModel, AutoModelForImageClassification

    name = BACKBONE_ALIASES.get(name, name)

    _SPECIAL_CASES = {
        'microsoft/resnet-': {
            'model_class': AutoModelForImageClassification,
            'post_init': lambda m: setattr(
                m.classifier, '1', nn.LayerNorm(m.config.hidden_sizes[-1])
            ),
        },
    }

    case = next(
        (v for prefix, v in _SPECIAL_CASES.items() if name.startswith(prefix)),
        {},
    )
    backbone = case.get('model_class', AutoModel).from_pretrained(name)

    if hasattr(backbone, 'vision_model'):
        backbone = backbone.vision_model
    if 'post_init' in case:
        case['post_init'](backbone)

    return backbone


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


class CausalPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        pool='cls',
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        assert pool in {'cls', 'mean'}, (
            'pool type must be either cls (cls token) or mean (mean pooling)'
        )

        self.num_patches = num_patches
        self.num_frames = num_frames

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * (num_patches), dim)
        )  # dim for the pos encodings
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
        self.pool = pool

    def forward(
        self, x
    ):  # x: (b, window_size * H/patch_size * W/patch_size, 384)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
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
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        self.register_buffer(
            'bias', self.generate_mask_matrix(num_patches, num_frames)
        )

    def forward(self, x):
        B, T, C = x.size()
        x = self.norm(x)

        # q, k, v: (B, heads, T, dim_head)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            rearrange(t, 'b n (h d) -> b h n d', h=self.heads) for t in qkv
        )

        attn_mask = self.bias[:, :, :T, :T] == 1  # bool mask

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
        rows = []
        for i in range(nwindow):
            row = torch.cat(
                [ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1
            )
            rows.append(row)
        mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
        return mask


class Transformer(nn.Module):
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
        for _ in range(depth):
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
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
