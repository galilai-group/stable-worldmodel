import torch
from torch import nn
from einops import rearrange
from stable_worldmodel.wm.genie.st_maskgit import ST_Transformer
from stable_worldmodel.wm.genie.vq import VectorQuantizer
from stable_worldmodel.wm.genie.utils import init_weights

class ST_ViViT(nn.Module):
    """
    ST-ViViT: a spatial-temporal VQ-VAE video tokenizer.

    Encodes (B, T, H, W, channels) pixel video into discrete tokens of shape
    (B, T, S) where S = (H/patch_size) * (W/patch_size).
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

        # Per-frame patchify (Conv2d with stride = patch_size)
        self.patchify = nn.Conv2d(channels, d_model, kernel_size=patch_size, stride=patch_size)

        # Separate pos embeds for encoder and decoder
        self.enc_pos_embed_TSC = nn.Parameter(torch.zeros(1, temporal_dim, S, d_model))
        self.dec_pos_embed_TSC = nn.Parameter(torch.zeros(1, temporal_dim, S, d_model))

        # d_model <> embed_dim bottleneck around the VQ
        self.pre_vq_proj = nn.Linear(d_model, vq.embed_dim)
        self.post_vq_proj = nn.Linear(vq.embed_dim, d_model)

        # Project each output token back to one patch's worth of pixels
        self.unpatchify = nn.Linear(d_model, patch_size * patch_size * channels)

        init_weights(self.modules())

    def encode(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            video: (B, T, H, W, c) pixel video
        Returns:
            indices_TS: (B, T, S) codebook ids
            z_q_TSC:    (B, T, S, d_model) post-VQ embeddings (after post_vq_proj)
            vq_loss:    scalar VQ loss
        """
        B, T = video.size(0), video.size(1)

        # Patchify per frame
        video_NcHW = rearrange(video, "B T H W c -> (B T) c H W")
        patches_NChw = self.patchify(video_NcHW)
        patches_TSC = rearrange(patches_NChw, "(B T) C h w -> B T (h w) C", B=B, T=T)

        # Encode with pos embed
        z_TSC = patches_TSC + self.enc_pos_embed_TSC
        z_TSC = self.encoder(z_TSC)

        # Bottleneck through the VQ
        z_TSE = self.pre_vq_proj(z_TSC)
        z_q_TSE, indices_TS, vq_loss = self.vq(z_TSE)
        z_q_TSC = self.post_vq_proj(z_q_TSE)

        return indices_TS, z_q_TSC, vq_loss

    def decode(self, z_q_TSC: torch.Tensor) -> torch.Tensor:
        """
        Decode d_model embeddings back to pixels.

        Args:
            z_q_TSC: (B, T, S, d_model) embeddings (already after post_vq_proj)
        Returns:
            video_hat: (B, T, H, W, c) reconstructed pixels
        """
        z_TSC = z_q_TSC + self.dec_pos_embed_TSC
        z_TSC = self.decoder(z_TSC)

        patches_TSP = self.unpatchify(z_TSC)
        video_hat = rearrange(
            patches_TSP,
            "B T (h w) (p1 p2 c) -> B T (h p1) (w p2) c",
            h=self.h_grid, w=self.w_grid,
            p1=self.patch_size, p2=self.patch_size,
        )
        return video_hat

    def decode_from_indices(self, indices_TS: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete token ids to pixels. Used at inference, e.g. after
        the dynamics model samples a token sequence.

        Args:
            indices_TS: (B, T, S) codebook ids
        Returns:
            video_hat: (B, T, H, W, c) decoded pixels
        """
        z_q_TSE = self.vq.codebook(indices_TS)
        z_q_TSC = self.post_vq_proj(z_q_TSE)
        return self.decode(z_q_TSC)

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full encode → quantize → decode pass.

        Args:
            video: (B, T, H, W, c) input video
        Returns:
            video_hat:  (B, T, H, W, c) reconstructed video
            indices_TS: (B, T, S) codebook ids
            vq_loss:    scalar
        """
        indices_TS, z_q_TSC, vq_loss = self.encode(video)
        video_hat = self.decode(z_q_TSC)
        return video_hat, indices_TS, vq_loss