"""Regression tests for a device-mismatch crash when warm-starting a solver.

THE BUG
-------
Every solver below pads a caller-supplied ``init_action``/``actions`` tensor
up to the full planning horizon with a freshly-allocated zero tensor. That
zero tensor was created without a ``device=`` argument, so it always lands on
torch's *default* device (CPU) regardless of where the tensor being padded
lives. When ``solve()`` is called with ``init_action=None`` the freshly
created "actions so far" tensor is also default-device, so the two match and
nothing crashes. But a real receding-horizon MPC caller warm-starts from the
*previous* solve's leftover plan -- typically already moved to the model's
device (e.g. CUDA) -- and ``torch.cat`` between that GPU tensor and the
CPU-default pad tensor raises::

    RuntimeError: Expected all tensors to be on the same device, but found
    at least two devices, cuda:0 and cpu!

This was found via a downstream project's warm-started MPC evaluation loop,
which was the first caller to exercise the "non-None, GPU-resident
init_action" path.

Of the seven solvers carrying this pattern, ``CEMSolver``, ``MPPISolver``,
``ICEMSolver``, ``GradientSolver`` and ``LagrangianSolver`` route
``init_action`` through ``planning/solver/utils.py::prepare_init_action``
first, which already pads to the full horizon with the correct device -- so
in current ``solve()`` call paths the buggy branch below is normally dead
code, but it is still reachable if that method is called directly, or if the
model's own padding (the ``Actionable`` branch) ever returns short. Only
``PredictiveSamplingSolver`` and ``PGDSolver`` still hit this live, since
they never call ``prepare_init_action``.

RUNNING THIS ON A GPU (e.g. in a cluster)
------------------------------------------
No flags needed. The "foreign device" fixture below is ``'meta'`` (always
available, no hardware required) plus ``'cuda'`` whenever
``torch.cuda.is_available()`` is true. On a CUDA box this automatically adds
a `cuda` case to every parametrized test, reproducing the exact
``RuntimeError`` from the bug report with a real ``cuda:0`` tensor:

    pytest tests/planning/solver/test_warm_start_device_mismatch.py -v

``test_cem_solve_end_to_end_warm_start_cuda`` additionally replays the full
``solve()`` -> leftover -> ``solve()`` warm-start loop from the original bug
report, but only runs when CUDA is actually available.
"""

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Discrete

from stable_worldmodel.planning.solver.cem import CEMSolver
from stable_worldmodel.planning.solver.gd import GradientSolver
from stable_worldmodel.planning.solver.icem import ICEMSolver
from stable_worldmodel.planning.solver.lagrangian import LagrangianSolver
from stable_worldmodel.planning.solver.mppi import MPPISolver
from stable_worldmodel.planning.solver.pgd import PGDSolver
from stable_worldmodel.planning.solver.predictive_sampling import (
    PredictiveSamplingSolver,
)
from stable_worldmodel.planning.solver.utils import prepare_init_action
from stable_worldmodel.policy import PlanConfig


class DummyCost:
    """Minimal Costable stand-in. The tests below never call get_cost()."""


def _foreign_devices() -> list[str]:
    devices = ['meta']
    if torch.cuda.is_available():
        devices.append('cuda')
    return devices


FOREIGN_DEVICES = _foreign_devices()


def _make_solver(solver_cls, device: str, **kwargs):
    """Build ``solver_cls`` to plan on ``device``.

    The constructor seeds a ``torch.Generator`` on its device, and
    ``init_action`` draws noise with that generator on ``self.device``, so a
    real device must be passed to the constructor. ``torch.Generator``
    rejects ``'meta'``, so a meta solver is built on the CPU and only its
    ``.device`` (a plain attribute read later) is moved.
    """
    if device == 'meta':
        solver = solver_cls(cost=DummyCost(), **kwargs)
        solver.device = device
        return solver
    return solver_cls(cost=DummyCost(), device=device, **kwargs)


def _box_setup(n_envs: int, horizon: int, action_dim: int):
    action_space = Box(low=-1, high=1, shape=(1, action_dim), dtype=np.float32)
    config = PlanConfig(
        horizon=horizon, receding_horizon=horizon, action_block=1
    )
    return action_space, config


def _discrete_setup(n_envs: int, horizon: int, n_categories: int):
    action_space = Discrete(n_categories)
    config = PlanConfig(
        horizon=horizon, receding_horizon=horizon, action_block=1
    )
    return action_space, config


###########################################################
## init_action_distrib: CEMSolver / MPPISolver / ICEMSolver
###########################################################


@pytest.mark.parametrize('other_device', FOREIGN_DEVICES)
@pytest.mark.parametrize('solver_cls', [CEMSolver, MPPISolver, ICEMSolver])
def test_init_action_distrib_warm_start_on_foreign_device(
    solver_cls, other_device
):
    """Warm-starting with a short, foreign-device tensor must not crash.

    Before the fix, the zero-pad tensor allocated inside
    ``init_action_distrib`` had no ``device=``, so it defaulted to CPU and
    ``torch.cat`` against a foreign-device ``actions`` tensor raised
    RuntimeError.
    """
    n_envs, horizon, action_dim = 2, 5, 4
    action_space, config = _box_setup(n_envs, horizon, action_dim)

    solver = solver_cls(cost=DummyCost(), batch_size=n_envs, num_samples=8)
    solver.configure(action_space=action_space, n_envs=n_envs, config=config)

    # Leftover plan from a previous solve, shorter than the horizon and
    # resident on a device other than the CPU default -- exactly what a
    # warm-started receding-horizon MPC caller passes in.
    warm_start = torch.zeros(
        n_envs, horizon - 1, solver.action_dim, device=other_device
    )

    mean, var = solver.init_action_distrib(n_envs, actions=warm_start)

    assert mean.shape == (n_envs, horizon, solver.action_dim)
    assert mean.device.type == torch.device(other_device).type
    assert var.shape == (n_envs, horizon, solver.action_dim)


###########################################################
## init_nominal: PredictiveSamplingSolver
###########################################################


@pytest.mark.parametrize('other_device', FOREIGN_DEVICES)
def test_predictive_sampling_init_nominal_warm_start_on_foreign_device(
    other_device,
):
    n_envs, horizon, action_dim = 2, 5, 4
    action_space, config = _box_setup(n_envs, horizon, action_dim)

    solver = PredictiveSamplingSolver(
        cost=DummyCost(), batch_size=n_envs, num_samples=8
    )
    solver.configure(action_space=action_space, n_envs=n_envs, config=config)

    warm_start = torch.zeros(
        n_envs, horizon - 1, solver.action_dim, device=other_device
    )

    nominal = solver.init_nominal(n_envs, actions=warm_start)

    assert nominal.shape == (n_envs, horizon, solver.action_dim)
    assert nominal.device.type == torch.device(other_device).type


###########################################################
## init_action: GradientSolver / LagrangianSolver (Box actions)
###########################################################


@pytest.mark.parametrize('other_device', FOREIGN_DEVICES)
@pytest.mark.parametrize(
    'solver_cls,ctor_kwargs',
    [
        (GradientSolver, {'n_steps': 3}),
        (LagrangianSolver, {'n_steps': 3}),
    ],
)
def test_gradient_based_init_action_warm_start_on_foreign_device(
    solver_cls, ctor_kwargs, other_device
):
    n_envs, horizon, action_dim = 2, 5, 4
    action_space, config = _box_setup(n_envs, horizon, action_dim)

    # Solver device matches the warm-start device here so init_action's
    # trailing `.to(self.device)` is a same-device no-op: on 'meta' that
    # avoids trying to materialize real data out of a data-less tensor (a
    # limitation of 'meta' as a CPU-only device stand-in, not something the
    # real bug depends on); on a real 'cuda' node this exercises exactly the
    # warm-started-on-GPU scenario from the bug report.
    solver = _make_solver(
        solver_cls, other_device, num_samples=3, **ctor_kwargs
    )
    solver.configure(action_space=action_space, n_envs=n_envs, config=config)

    warm_start = torch.zeros(
        n_envs, horizon - 1, solver.action_dim, device=other_device
    )

    if solver_cls is GradientSolver:
        solver.init_action(n_envs, actions=warm_start)
    else:
        solver.init_action(actions=warm_start)

    assert solver.init.shape[2] == horizon
    assert solver.init.device.type == torch.device(other_device).type


###########################################################
## init_action: PGDSolver (discrete/simplex actions)
###########################################################


@pytest.mark.parametrize('other_device', FOREIGN_DEVICES)
def test_pgd_init_action_warm_start_on_foreign_device(other_device):
    n_envs, horizon, n_categories = 2, 5, 4
    action_space, config = _discrete_setup(n_envs, horizon, n_categories)

    solver = _make_solver(PGDSolver, other_device, n_steps=3, num_samples=3)
    solver.configure(action_space=action_space, n_envs=n_envs, config=config)

    warm_start = torch.zeros(
        n_envs,
        horizon - 1,
        solver._action_simplex_dim,
        device=other_device,
    )

    solver.init_action(actions=warm_start)

    assert solver.init.shape[2] == horizon
    assert solver.init.device.type == torch.device(other_device).type


###########################################################
## prepare_init_action: solver-device placement
###########################################################


@pytest.mark.parametrize('other_device', FOREIGN_DEVICES)
def test_prepare_init_action_none_non_actionable_respects_device(other_device):
    """``init_action=None`` + non-Actionable model must honor ``device=``.

    Before the fix, this path had no reference tensor to borrow a device
    from and hardcoded CPU, so solvers configured with ``device='cuda'``
    received a CPU plan and crashed on the first device-mixing op.
    """
    n_envs, horizon, action_dim = 2, 5, 4

    result = prepare_init_action(
        DummyCost(),
        {'dummy': torch.zeros(n_envs)},
        None,
        horizon,
        n_envs=n_envs,
        action_dim=action_dim,
        device=other_device,
    )

    assert result.shape == (n_envs, horizon, action_dim)
    assert result.device.type == torch.device(other_device).type


@pytest.mark.parametrize('other_device', FOREIGN_DEVICES)
def test_prepare_init_action_partial_cpu_warm_start_moved_to_device(
    other_device,
):
    """A partial CPU warm start must be relocated onto the requested device."""
    n_envs, horizon, action_dim = 2, 5, 4
    warm_start = torch.zeros(n_envs, horizon - 2, action_dim)  # CPU

    result = prepare_init_action(
        DummyCost(),
        {'dummy': torch.zeros(n_envs)},
        warm_start,
        horizon,
        n_envs=n_envs,
        action_dim=action_dim,
        device=other_device,
    )

    assert result.shape == (n_envs, horizon, action_dim)
    assert result.device.type == torch.device(other_device).type


def test_prepare_init_action_no_device_keeps_legacy_behavior():
    """Without ``device=``, the pre-existing device semantics are unchanged."""
    n_envs, horizon, action_dim = 2, 5, 4

    result = prepare_init_action(
        DummyCost(),
        {'dummy': torch.zeros(n_envs)},
        None,
        horizon,
        n_envs=n_envs,
        action_dim=action_dim,
    )

    assert result.shape == (n_envs, horizon, action_dim)
    assert result.device.type == 'cpu'


###########################################################
## init_action: full-horizon warm start on a foreign solver device
###########################################################


@pytest.mark.parametrize(
    'solver_cls,ctor_kwargs',
    [
        (GradientSolver, {'n_steps': 3}),
        (LagrangianSolver, {'n_steps': 3}),
    ],
)
def test_gradient_based_init_action_full_horizon_moved_to_solver_device(
    solver_cls, ctor_kwargs
):
    """A full-horizon CPU plan must still land on the solver's device.

    ``prepare_init_action`` always returns a full-horizon tensor, so the
    ``remaining > 0`` fill branch never triggers from ``solve()``. Before the
    fix, the only ``.to(self.device)`` lived inside that branch, leaving
    full-horizon actions stranded on their original device.
    """
    n_envs, horizon, action_dim = 2, 5, 4
    action_space, config = _box_setup(n_envs, horizon, action_dim)

    solver = solver_cls(cost=DummyCost(), num_samples=3, **ctor_kwargs)
    solver.device = 'meta'
    solver.configure(action_space=action_space, n_envs=n_envs, config=config)

    warm_start = torch.zeros(n_envs, horizon, solver.action_dim)  # CPU, full

    if solver_cls is GradientSolver:
        solver.init_action(n_envs, actions=warm_start)
    else:
        solver.init_action(actions=warm_start)

    assert solver.init.shape[2] == horizon
    assert solver.init.device.type == 'meta'


###########################################################
## End-to-end reproduction of the original bug report (CUDA only)
###########################################################


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires a CUDA device'
)
def test_cem_solve_end_to_end_warm_start_cuda():
    """Exact repro from the bug report: solve() -> leftover -> solve().

    Before the fix this raised:
        RuntimeError: Expected all tensors to be on the same device, but
        found at least two devices, cuda:0 and cpu!
    on the second `solve()` call, because the leftover plan from the first
    solve was moved to CUDA (as any warm-started MPC caller would do) while
    `init_action_distrib`'s zero-pad tensor stayed on CPU.
    """

    class DummyCostModel:
        def get_cost(self, info_dict, action_candidates):
            b, n = action_candidates.shape[:2]
            return torch.rand(b, n, device=action_candidates.device)

    device = 'cuda'
    solver = CEMSolver(
        cost=DummyCostModel(),
        batch_size=1,
        num_samples=16,
        n_steps=1,
        topk=4,
        device=device,
    )

    action_space = Box(low=-1, high=1, shape=(1, 4), dtype=np.float32)
    config = PlanConfig(horizon=5, receding_horizon=5, action_block=1)
    solver.configure(action_space=action_space, n_envs=1, config=config)

    info_dict = {'dummy': torch.zeros(1, device=device)}

    out1 = solver.solve(info_dict, init_action=None)
    assert out1['actions'].shape[1] == 5

    # Leftover horizon from solve 1, moved to GPU -- what a real MPC caller
    # does when carrying a previous solve's tail forward.
    leftover = out1['actions'][:, 1:].to(device)
    assert leftover.is_cuda

    out2 = solver.solve(info_dict, init_action=leftover)
    assert out2['actions'].shape[1] == 5


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires a CUDA device'
)
def test_gradient_solve_end_to_end_cold_start_cuda():
    """Cold-start repro: init_action=None + non-Actionable cost + CUDA solver.

    Before the fix this raised the device-mismatch RuntimeError on the very
    first solve() call: prepare_init_action returned a CPU zero plan, and
    with the full horizon already filled, init_action() never relocated it
    onto the solver's device. ``action_noise > 0`` additionally exercises the
    in-loop noise injection, which allocated its randn on CPU despite the
    CUDA generator.
    """

    class DiffCost:
        def get_cost(self, info_dict, action_candidates):
            return action_candidates.pow(2).sum(dim=(-2, -1))

    solver = GradientSolver(
        cost=DiffCost(),
        n_steps=2,
        num_samples=3,
        action_noise=0.1,
        device='cuda',
    )

    action_space = Box(low=-1, high=1, shape=(1, 4), dtype=np.float32)
    config = PlanConfig(horizon=5, receding_horizon=5, action_block=1)
    solver.configure(action_space=action_space, n_envs=1, config=config)

    info_dict = {'dummy': torch.zeros(1, device='cuda')}

    out = solver.solve(info_dict, init_action=None)
    assert out['actions'].shape == (1, 5, solver.action_dim)
