import torch
from stable_worldmodel.wm.diamond.module import (
    c_preconditioners,
    sigma_sampling,
    EDMModel,
    sample_euler,
    sample_heun,
    make_sigma_schedule,
    edm_loss_step,
)


def test_preconditioners_shapes():
    sig = torch.tensor([0.1, 1.0])
    c_in, c_out, c_skip, c_noise = c_preconditioners(sig)
    assert c_in.shape[0] == 2
    assert c_out.shape[0] == 2


def test_sigma_sampling():
    s = sigma_sampling(4, device='cpu')
    assert s.shape[0] == 4


def test_unet_forward_and_predict():
    model = EDMModel(in_ch=3, base_ch=16, cond_dim=32)
    B, C, H, W = 2, 3, 32, 32
    x = torch.randn(B, C, H, W)
    history = torch.randn(B, C * 4, H, W)
    cond_vec = torch.randn(B, 32)
    out = model.predict(
        x,
        torch.tensor([0.5, 0.5])[:1],
        {'history': history, 'cond_vec': cond_vec},
    )
    assert out.shape == x.shape


def test_edm_loss_step_basic():
    model = EDMModel(in_ch=3, base_ch=16, cond_dim=32)
    B, C, H, W = 2, 3, 32, 32
    next_frame = torch.randn(B, C, H, W)
    history = torch.randn(B, C * 4, H, W)
    cond_vec = torch.randn(B, 32)
    batch = {
        'next_frame': next_frame,
        'history': history,
        'cond_vec': cond_vec,
    }
    loss = edm_loss_step(model, batch, device='cpu')
    assert loss.item() >= 0


def test_samplers_shapes():
    model = EDMModel(in_ch=3, base_ch=16, cond_dim=32)
    B, C, H, W = 1, 3, 32, 32
    history = torch.randn(B, C * 4, H, W)
    cond_vec = torch.randn(B, 32)
    cond = {'history': history, 'cond_vec': cond_vec}
    shape = (B, C, H, W)
    out = sample_euler(model, cond, shape, device='cpu', n_steps=2)
    assert out.shape == shape
    out2 = sample_heun(model, cond, shape, device='cpu', n_steps=2)
    assert out2.shape == shape
