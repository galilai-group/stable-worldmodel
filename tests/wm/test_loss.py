"""Tests for stable_worldmodel.wm.loss.SIGReg.

SIGReg is a thin, device-agnostic, multi-GPU-capable adapter over
``stable_pretraining.methods.lejepa.SlicedEppsPulley`` (the canonical LeJEPA
SIGReg, arXiv:2511.08544). The distributed correctness of the underlying
statistic is covered by stable_pretraining's own DDP tests; here we only test
the adapter contract:

* it constructs and runs on CPU (no hard-coded ``"cuda"`` device);
* it faithfully delegates to ``SlicedEppsPulley`` with the documented kwarg
  mapping (``knots -> n_points``, ``num_proj -> num_slices``);
* it pools all leading axes into the sample axis and returns a finite scalar
  through which gradients flow.

Determinism note: ``SlicedEppsPulley`` seeds its random projection from an
internal ``global_step`` buffer (initialised to 0), not the global torch RNG,
so two freshly constructed modules draw the *same* first projection and can be
compared directly.

The adapter requires the optional ``train`` dependency (stable_pretraining);
these tests skip when it is not installed (CI installs it via
``uv sync --all-extras``).
"""

import pytest
import torch


# SIGReg's forward path delegates to stable_pretraining; skip cleanly when the
# optional training dependency is absent.
spt_lejepa = pytest.importorskip('stable_pretraining.methods.lejepa')

from stable_worldmodel.wm.loss import SIGReg  # noqa: E402


# ---------------------------------------------------------------------------
# device-agnostic: runs on CPU (the old version hard-coded device="cuda")
# ---------------------------------------------------------------------------


def test_sigreg_runs_on_cpu():
    reg = SIGReg(knots=17, num_proj=64)
    proj = torch.randn(4, 8, 16)  # (T, B, D) on CPU

    out = reg(proj)

    assert out.shape == ()  # scalar
    assert torch.isfinite(out)
    assert out.device.type == 'cpu'


def test_sigreg_gradient_flows():
    reg = SIGReg(knots=17, num_proj=64)
    proj = torch.randn(4, 8, 16, requires_grad=True)

    reg(proj).backward()

    assert proj.grad is not None
    assert torch.isfinite(proj.grad).all()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS device not available'
)
def test_sigreg_runs_on_mps():
    reg = SIGReg(knots=17, num_proj=64).to('mps')
    proj = torch.randn(4, 8, 16, device='mps')

    out = reg(proj)

    assert torch.isfinite(out)
    assert out.device.type == 'mps'


# ---------------------------------------------------------------------------
# faithful delegation + kwarg mapping
# ---------------------------------------------------------------------------


def test_sigreg_kwarg_mapping():
    reg = SIGReg(knots=9, num_proj=32)

    assert isinstance(reg.sliced_ep, spt_lejepa.SlicedEppsPulley)
    assert reg.sliced_ep.num_slices == 32  # num_proj -> num_slices
    # knots -> n_points (Epps-Pulley quadrature nodes); the 'phi' buffer has
    # length n_points.
    assert reg.sliced_ep.ep.phi.numel() == 9


def test_sigreg_matches_sliced_epps_pulley():
    """SIGReg(proj) == SlicedEppsPulley(proj.reshape(-1, D)).

    Both modules are freshly constructed, so both draw the same first
    projection (global_step == 0).
    """
    proj = torch.randn(4, 8, 16)

    out_adapter = SIGReg(knots=17, num_proj=64)(proj)
    out_direct = spt_lejepa.SlicedEppsPulley(num_slices=64, n_points=17)(
        proj.reshape(-1, proj.size(-1))
    )

    torch.testing.assert_close(out_adapter, out_direct)


def test_sigreg_pools_leading_axes():
    """Flattening (T, B, D) -> (T*B, D) feeds N=T*B samples to the EP test, so
    a pre-reshaped input yields the same statistic."""
    proj = torch.randn(4, 8, 16)

    flat = SIGReg(knots=17, num_proj=64)(proj)
    explicit = SIGReg(knots=17, num_proj=64)(proj.reshape(-1, 16))

    torch.testing.assert_close(flat, explicit)
