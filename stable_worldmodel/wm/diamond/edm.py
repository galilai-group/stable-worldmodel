import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import SimpleUNet
from .diffusion import RewardTerminationHead
import torch.nn as nn


def sigma_sampling(batch_size, device, mean=-0.4, std=1.2):
    return torch.exp(torch.randn(batch_size, device=device) * std + mean)


def c_preconditioners(sigma, sigma_data=0.5):
    # sigma: (B,)
    sigma1 = sigma.view(-1, 1, 1, 1)
    c_in = 1.0 / (sigma1**2 + sigma_data**2).sqrt()
    c_out = (sigma1 * sigma_data / (sigma1**2 + sigma_data**2)).sqrt()
    c_skip = (sigma_data**2) / (sigma_data**2 + sigma1**2)
    c_noise = 0.25 * torch.log(sigma1.squeeze())
    return c_in, c_out, c_skip, c_noise


class EDMModel(nn.Module):
    """EDM-like wrapper around a U-Net.

    This implements the network preconditioning used in Karras et al. (2022)
    and provides a `predict` method to produce denoised images from noisy
    inputs and conditioning (frame-stack + actions embedding).
    """

    def __init__(
        self, in_ch=3, base_ch=64, cond_dim=256, sigma_data=0.5, emb_dim=256
    ):
        super().__init__()
        self.unet = SimpleUNet(
            in_channels=in_ch * 2, base_ch=base_ch, cond_dim=cond_dim
        )
        self.sigma_data = sigma_data
        # embedder: map predicted frames to embeddings for reward/termination
        self.embedder = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch, emb_dim),
        )
        # reward/termination head attached by default
        self.rhead = RewardTerminationHead(emb_dim)

    def forward(self, x_noisy, sigma, cond):
        # x_noisy: (B, C, H, W)
        # sigma: (B,)
        # cond: (B, cond_dim)
        c_in, c_out, c_skip, c_noise = c_preconditioners(
            sigma, self.sigma_data
        )

        x_in = c_in * x_noisy
        # Concatenate historical frames channel-wise if present; otherwise
        # duplicate the noisy input to match the U-Net expected channels.
        if 'history' in cond and cond['history'] is not None:
            inp = torch.cat([x_in, cond['history']], dim=1)
        else:
            inp = torch.cat([x_in, x_in], dim=1)

        cond_vec = cond.get('cond_vec', None)
        out = self.unet(inp, cond_vec)
        denoised = c_out * out + c_skip * x_noisy
        return denoised

    def predict(self, x_noisy, sigma, cond):
        return self.forward(x_noisy, sigma, cond)

    def score(self, x_noisy, sigma, cond):
        """Estimate score = grad_x log p_sigma(x) using the relation
        score = (x0_hat - x) / sigma^2 where x0_hat is predicted clean image.
        """
        x0_hat = self.predict(x_noisy, sigma, cond)
        sigma1 = sigma.view(-1, 1, 1, 1)
        score = (x0_hat - x_noisy) / (sigma1**2)
        return score


__all__ = ['EDMModel', 'sigma_sampling']
