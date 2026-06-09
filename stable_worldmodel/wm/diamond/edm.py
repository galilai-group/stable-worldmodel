import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import SimpleUNet


def sigma_sampling(batch_size, device, mean=-0.4, std=1.2):
    return torch.exp(torch.randn(batch_size, device=device) * std + mean)


def c_preconditioners(sigma, sigma_data=0.5):
    # sigma: (B,)
    sigma1 = sigma.view(-1, 1, 1, 1)
    c_in = 1.0 / (sigma1**2 + sigma_data**2).sqrt()
    c_out = (sigma1 * sigma_data) / (sigma1**2 + sigma_data**2).sqrt()
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
        self,
        in_ch=3,
        base_ch=64,
        cond_dim=256,
        sigma_data=0.5,
        history_frames=4,
    ):
        super().__init__()
        # U-Net input: noisy next frame (in_ch) + history frames stacked (in_ch * history_frames)
        # U-Net output: only the denoised frame (in_ch), not the history
        # cond_dim + 1 to account for noise level appended at runtime
        self.unet = SimpleUNet(
            in_channels=in_ch * (history_frames + 1),
            out_channels=in_ch,
            base_ch=base_ch,
            cond_dim=cond_dim + 1,
        )
        self.sigma_data = sigma_data

    def forward(self, x_noisy, sigma, cond):
        # x_noisy: (B, C, H, W)
        # sigma: (B,)
        # cond: dict with keys 'history' (B, C*L, H, W) and 'cond_vec' (B, D)
        c_in, c_out, c_skip, c_noise = c_preconditioners(
            sigma, self.sigma_data
        )

        x_in = c_in * x_noisy
        history = cond.get('history', None)
        if history is not None:
            inp = torch.cat([x_in, history], dim=1)
        else:
            # zero-pad to match expected U-Net input channels
            pad_channels = self.unet.inc.in_channels - x_in.size(1)
            if pad_channels > 0:
                pad = torch.zeros_like(x_in[:, :1]).expand(
                    -1, pad_channels, -1, -1
                )
                inp = torch.cat([x_in, pad], dim=1)
            else:
                inp = x_in

        cond_vec = cond.get('cond_vec', None)
        if cond_vec is not None and isinstance(cond_vec, torch.Tensor):
            # Broadcast c_noise to match cond_vec batch dimension
            c_noise_vec = c_noise.view(-1, 1).expand(cond_vec.size(0), -1)
            cond_vec = torch.cat([cond_vec, c_noise_vec], dim=-1)
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
