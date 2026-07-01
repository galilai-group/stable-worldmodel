import torch
from torch import nn
from einops import rearrange
from stable_worldmodel.wm.genie.st_maskgit import ST_Transformer
from stable_worldmodel.wm.genie.vq import VectorQuantizer
from stable_worldmodel.wm.genie.utils import init_weights


class LAM(nn.Module):
    """
    Latent Action Model.

    A VQ-VAE on raw pixels where the "tokens" are discrete latent actions.
    Trained as:
      encoder: video -> action vector per frame transition (T-1 of them)
               -> VQ -> discrete action ids `a` in [0, num_actions]
      decoder: (prev frames, `a`) -> predicted next frames -> reconstruction loss

    At inference everything except `vq.codebook` is discarded.
    User-supplied action ids feed straight into the dynamics model's own action embedding.

    Suffix conventions: B=batch, T=time, H,W=pixel dims, h,w=patch-grid dims,
                        S=h*w, C=d_model, E=vq.embed_dim, c=RGB channels.
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
        temporal_dim: int,   # T
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

        # Separate patchify for encoder and decoder
        self.enc_patchify = nn.Conv2d(channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.dec_patchify = nn.Conv2d(channels, d_model, kernel_size=patch_size, stride=patch_size)

        # Encoder sees T positions, decoder sees T-1 (we drop the last frame)
        self.enc_pos_embed_TSC = nn.Parameter(torch.zeros(1, temporal_dim, S, d_model))
        self.dec_pos_embed_TSC = nn.Parameter(torch.zeros(1, temporal_dim - 1, S, d_model))

        # VQ bottleneck for actions: d_model → embed_dim before VQ,
        # then embed_dim → d_model on the decoder side
        self.pre_vq_proj = nn.Linear(d_model, vq.embed_dim)
        self.action_to_dmodel = nn.Linear(vq.embed_dim, d_model)

        # Decoder output → patch worth of pixels
        self.unpatchify = nn.Linear(d_model, patch_size * patch_size * channels)

        init_weights(self.modules())

    def extract_actions(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            video: (B, T, H, W, c) pixel video
        Returns:
            action_ids: (B, T-1) discrete action ids
            action_q_TE: (B, T-1, embed_dim) post-VQ action embeddings
            vq_loss: scalar
        """
        B, T = video.size(0), video.size(1)

        video_NcHW = rearrange(video, "B T H W c -> (B T) c H W")
        patches_NChw = self.enc_patchify(video_NcHW)
        patches_TSC = rearrange(patches_NChw, "(B T) C h w -> B T (h w) C", B=B, T=T)

        z_TSC = patches_TSC + self.enc_pos_embed_TSC
        z_TSC = self.encoder(z_TSC)

        # Pool spatial → one vector per frame
        z_TC = z_TSC.mean(dim=2)                            # (B, T, C)

        # Position t of the encoder has seen frames 0..t, so position t+1 carries
        # enough info to describe the transition from frame t to t+1. Drop position 0.
        action_TC = z_TC[:, 1:, :]                          # (B, T-1, C)

        action_TE = self.pre_vq_proj(action_TC)
        action_q_TE, action_ids, vq_loss = self.vq(action_TE)
        return action_ids, action_q_TE, vq_loss

    def predict_next_frames(
        self,
        prev_video: torch.Tensor,
        action_q_TE: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prev_video:  (B, T-1, H, W, c) frames x_0..x_{T-2}
            action_q_TE: (B, T-1, embed_dim) post-VQ actions
        Returns:
            next_pred: (B, T-1, H, W, c) predictions for x_1..x_{T-1}
        """
        B, Tm1 = prev_video.size(0), prev_video.size(1)

        video_NcHW = rearrange(prev_video, "B T H W c -> (B T) c H W")
        patches_NChw = self.dec_patchify(video_NcHW)
        patches_TSC = rearrange(patches_NChw, "(B T) C h w -> B T (h w) C", B=B, T=Tm1)

        # Project actions to d_model, broadcast across spatial, add (per paper)
        action_TC = self.action_to_dmodel(action_q_TE)      # (B, T-1, C)
        patches_TSC = patches_TSC + action_TC.unsqueeze(2)

        z_TSC = patches_TSC + self.dec_pos_embed_TSC
        z_TSC = self.decoder(z_TSC)

        patches_TSP = self.unpatchify(z_TSC)
        next_pred = rearrange(
            patches_TSP,
            "B T (h w) (p1 p2 c) -> B T (h p1) (w p2) c",
            h=self.h_grid, w=self.w_grid,
            p1=self.patch_size, p2=self.patch_size,
        )
        return next_pred

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward: encode actions, predict next frames.

        Args:
            video: (B, T, H, W, c) input video
        Returns:
            next_pred:  (B, T-1, H, W, c) predicted frames x_1..x_{T-1}
            action_ids: (B, T-1) discrete action ids
            vq_loss:    scalar
        """
        action_ids, action_q_TE, vq_loss = self.extract_actions(video)
        next_pred = self.predict_next_frames(video[:, :-1], action_q_TE)
        return next_pred, action_ids, vq_loss