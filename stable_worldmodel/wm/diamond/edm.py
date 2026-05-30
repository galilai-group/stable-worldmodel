import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import SimpleUNet


def sigma_sampling(batch_size, device, mean=-0.4, std=1.2):
    return torch.exp(torch.randn(batch_size, device=device) * std + mean)


def c_preconditioners(sigma, sigma_data=0.5):
    # sigma: (B,)
    sigma = sigma.view(-1, 1, 1, 1)
    c_in = 1.0 / (sigma**2 + sigma_data**2).sqrt()
    c_out = (sigma * sigma_data / (sigma**2 + sigma_data**2)).sqrt()
    c_skip = (sigma_data**2) / (sigma_data**2 + sigma**2)
    c_noise = 0.25 * torch.log(sigma)
    return c_in, c_out, c_skip, c_noise


class EDMModel(nn.Module):
    """EDM-like wrapper around a U-Net.

    This implements the network preconditioning used in Karras et al. (2022)
    and provides a `predict` method to produce denoised images from noisy
    inputs and conditioning (frame-stack + actions embedding).
    """

    def __init__(self, in_ch=3, base_ch=64, cond_dim=256, sigma_data=0.5):
        super().__init__()
        self.unet = SimpleUNet(
            in_channels=in_ch * 2, base_ch=base_ch, cond_dim=cond_dim
        )
        self.sigma_data = sigma_data

    def forward(self, x_noisy, sigma, cond):
        # x_noisy: (B, C, H, W)
        # sigma: (B,)
        # cond: (B, cond_dim)
        c_in, c_out, c_skip, c_noise = c_preconditioners(
            sigma, self.sigma_data
        )

        x_in = c_in * x_noisy
        inp = (
            torch.cat([x_in, cond['history']], dim=1)
            if 'history' in cond
            else x_in
        )
        out = self.unet(inp, cond['cond_vec'])
        denoised = c_out * out + c_skip * x_noisy
        return denoised

    def predict(self, x_noisy, sigma, cond):
        return self.forward(x_noisy, sigma, cond)


__all__ = ['EDMModel', 'sigma_sampling']
