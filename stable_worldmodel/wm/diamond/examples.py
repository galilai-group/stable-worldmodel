import torch

from .edm import EDMModel, sigma_sampling
from .edm_sampling import sample_euler, sample_heun
from .diffusion import RewardTerminationHead


def example_infer(
    model: EDMModel, history, cond_vec, device='cpu', method='euler'
):
    """Produce a single next-frame prediction from model.

    history: (B, C*L, H, W)
    cond_vec: (B, cond_dim)
    """
    model.eval()
    B = history.shape[0]
    # sample sigma and build cond dict
    sigma = sigma_sampling(B, device)
    cond = {'history': history, 'cond_vec': cond_vec}
    shape = (
        B,
        history.shape[1] // (history.shape[1] // 3),
        history.shape[2],
        history.shape[3],
    )
    # The shape above is a placeholder; using history channels to infer C is messy here.
    # Instead, we reconstruct target shape as (B, 3, H, W) assuming RGB.
    shape = (B, 3, history.shape[2], history.shape[3])

    if method == 'euler':
        out = sample_euler(model, cond, shape, device)
    else:
        out = sample_heun(model, cond, shape, device)

    return out


def example_train_step_wrapper(model, optimizer, batch, device='cpu'):
    from .edm_train import example_train_step

    return example_train_step(model, optimizer, batch, device)


__all__ = ['example_infer', 'example_train_step_wrapper']
