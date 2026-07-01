import torch
from torch import nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Gradient-trained vector quantizer.

    Forward takes continuous vectors (..., D) and returns:
      - quantized vectors of the same shape (with straight-through gradient)
      - integer indices of shape (...) for downstream token prediction
      - a scalar loss to add to the total training loss

    Codebook is updated via gradients on the codebook loss. Dead-code reinit
    is exposed as a separate method to be called periodically by training code.
    """
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

        # ema of how often each code is used for dead-code detection
        self.register_buffer("usage_count", torch.zeros(num_codes))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (..., embed_dim) continuous vectors from the encoder
        Returns:
            quantized: (..., embed_dim) — straight-through, same shape as x
            indices:   (...,) long tensor of nearest-code ids
            loss:      scalar, codebook + commitment_beta * commitment

        Suffix convention: N = flattened batch, D = embed_dim, K = num_codes.
        """
        orig_shape = x.shape
        x_ND = x.reshape(-1, self.embed_dim)

        # squared l2 distance to every code
        codebook_KD = self.codebook.weight
        x_sq_N1 = x_ND.pow(2).sum(dim=1, keepdim=True)
        codebook_sq_K = codebook_KD.pow(2).sum(dim=1)
        xc_NK = x_ND @ codebook_KD.t()
        dists_NK = x_sq_N1 - 2 * xc_NK + codebook_sq_K

        indices_N = dists_NK.argmin(dim=1)
        quantized_ND = self.codebook(indices_N)

        codebook_loss = F.mse_loss(quantized_ND, x_ND.detach()) # pull codes -> encoder
        commitment_loss = F.mse_loss(x_ND, quantized_ND.detach()) # pull encoder -> codes
        loss = codebook_loss + self.commitment_beta * commitment_loss

        # straight through estimator: forward returns quantized, backward acts like identity
        quantized_ND = x_ND + (quantized_ND - x_ND).detach()

        quantized = quantized_ND.reshape(orig_shape)
        indices = indices_N.reshape(orig_shape[:-1])

        # ema-track usage so reinit_dead_codes can find inactive entries
        if self.training:
            with torch.no_grad():
                one_hot_NK = F.one_hot(indices_N, self.num_codes).to(self.usage_count.dtype)
                self.usage_count.mul_(self.usage_decay).add_(one_hot_NK.sum(dim=0))

        return quantized, indices, loss

    @torch.no_grad()
    def reinit_dead_codes(self, x: torch.Tensor, threshold: float = 1.0) -> int:
        """
        Replace codes whose EMA usage is below `threshold` with random encoder
        outputs sampled from `x`. Call every N training steps.

        Args:
            x: (..., embed_dim) recent encoder outputs to sample replacements from
            threshold: codes with usage_count < threshold are considered dead
        Returns:
            number of codes reinitialized
        """
        x_flat = x.reshape(-1, self.embed_dim)
        dead = self.usage_count < threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0

        sampled = torch.randint(0, x_flat.size(0), (n_dead,), device=x.device)
        self.codebook.weight.data[dead] = x_flat[sampled]
        self.usage_count[dead] = float(threshold)
        return n_dead