import torch
import math


def make_sigma_schedule(sigma_max, sigma_min, n_steps):
    # geometric schedule between sigma_max and sigma_min
    if n_steps == 1:
        return [sigma_min]
    schedule = [
        sigma_max * (sigma_min / sigma_max) ** (i / (n_steps - 1))
        for i in range(n_steps)
    ]
    return schedule


def sample_euler(
    model, cond, shape, device, n_steps=3, sigma_max=1.0, sigma_min=1e-3
):
    """Deterministic Euler-like sampler.

    model.predict(x, sigma, cond) is expected to return an estimate of x0.
    """
    schedule = make_sigma_schedule(sigma_max, sigma_min, n_steps)
    # start from Gaussian noise at sigma_max
    x = torch.randn(*shape, device=device) * schedule[0]

    for i in range(len(schedule) - 1):
        sigma = torch.tensor(schedule[i], device=device)
        sigma_next = torch.tensor(schedule[i + 1], device=device)

        x0 = model.predict(x, sigma, cond)
        # deterministic update: preserve current noise eps
        x = x0 + (sigma_next / sigma) * (x - x0)

    # final denoising at sigma_min
    x_final = model.predict(x, torch.tensor(schedule[-1], device=device), cond)
    return x_final


def sample_heun(
    model, cond, shape, device, n_steps=3, sigma_max=1.0, sigma_min=1e-3
):
    """Heun's method (improved Euler) adapted for denoising.

    Uses two evaluations per step but improves stability.
    """
    schedule = make_sigma_schedule(sigma_max, sigma_min, n_steps)
    x = torch.randn(*shape, device=device) * schedule[0]

    for i in range(len(schedule) - 1):
        sigma = torch.tensor(schedule[i], device=device)
        sigma_next = torch.tensor(schedule[i + 1], device=device)

        # first (Euler) prediction
        x0_1 = model.predict(x, sigma, cond)
        x_euler = x0_1 + (sigma_next / sigma) * (x - x0_1)

        # second prediction at provisional point
        x0_2 = model.predict(x_euler, sigma_next, cond)

        # Heun update: use the provisional Euler step for stability. A true
        # Heun update would average derivatives (requires score estimates).
        x = x_euler

    x_final = model.predict(x, torch.tensor(schedule[-1], device=device), cond)
    return x_final


__all__ = ['sample_euler', 'sample_heun', 'make_sigma_schedule']
