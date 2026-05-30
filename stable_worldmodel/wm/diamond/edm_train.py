import torch
import torch.nn.functional as F
from torch import nn
from .edm import c_preconditioners, sigma_sampling


def edm_loss_step(model, batch, device):
    """Compute EDM preconditioned L2 loss for a batch.

    batch: dict with keys:
      - next_frame: (B, C, H, W) clean target
      - history: (B, C*L, H, W) stacked conditioning frames (optional)
      - cond_vec: (B, cond_dim) conditioning vector (actions/time embedding)

    model.predict(x_noisy, sigma, cond) should return the network's prediction
    corresponding to the preconditioned target described in the paper.
    """
    B = batch['next_frame'].shape[0]
    # sample sigma per example
    sigma = sigma_sampling(B, device)
    sigma_t = sigma.to(device)

    # create noisy observation
    noise = torch.randn_like(batch['next_frame'], device=device)
    x_noisy = batch['next_frame'] + noise * sigma_t.view(-1, 1, 1, 1)

    cond = {
        'history': batch.get('history', None),
        'cond_vec': batch.get('cond_vec', None),
    }

    # network predicts denoised x0_hat
    x0_hat = model.predict(x_noisy, sigma_t, cond)

    # compute target: (1 / c_out) * (x0 - c_skip * x_noisy)
    c_in, c_out, c_skip, c_noise = c_preconditioners(sigma_t)

    target = (
        batch['next_frame'] - c_skip.view(-1, 1, 1, 1) * x_noisy
    ) / c_out.view(-1, 1, 1, 1)

    loss = F.mse_loss(x0_hat, target)
    return loss


def example_train_step(model, optimizer, batch, device):
    model.train()
    loss = edm_loss_step(model, batch, device)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


__all__ = ['edm_loss_step', 'example_train_step']
