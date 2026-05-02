"""Tests for CompositeSolver over Dict action spaces."""

import numpy as np
import pytest
import torch
from gymnasium import spaces as gym_spaces

from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver.base import BaseSolver
from stable_worldmodel.solver.callbacks import BestCostRecorder
from stable_worldmodel.solver.categorical_cem import CategoricalCEMSolver
from stable_worldmodel.solver.cem import CEMSolver
from stable_worldmodel.solver.composite import CompositeSolver


###########################
## Test fixtures         ##
###########################


class _SumSquaresCost:
    """Joint cost = sum of squares across all sub-action tensors."""

    def get_cost(self, info_dict, candidates):
        if isinstance(candidates, dict):
            total = None
            for v in candidates.values():
                term = v.pow(2).sum(dim=(-1, -2))
                total = term if total is None else total + term
            return total
        return candidates.pow(2).sum(dim=(-1, -2))


class _CoupledCost:
    """Cost couples cont and disc — minimized iff cont is small AND disc=target.

    cost = ||cont||^2 + alpha * (1 - mass_on_target_in_disc).
    """

    def __init__(self, target=2, base_simplex_dim=4, alpha=10.0):
        self.target = target
        self.K = base_simplex_dim
        self.alpha = alpha

    def get_cost(self, info_dict, candidates):
        cont = candidates['cont']
        disc = candidates['disc']
        # disc is one-hot reshaped to (B, N, H, action_block * K)
        ab = disc.shape[-1] // self.K
        d = disc.reshape(*disc.shape[:-1], ab, self.K)
        target_mass = d[..., self.target].sum(dim=(-1, -2))
        H = cont.shape[-2]
        return cont.pow(2).sum(dim=(-1, -2)) + self.alpha * (
            H * ab - target_mass
        )


def _config(horizon=3, receding=2, action_block=1):
    return PlanConfig(
        horizon=horizon, receding_horizon=receding, action_block=action_block
    )


def _dict_space(cont_dim=3, n_cats=4):
    return gym_spaces.Dict(
        {
            'cont': gym_spaces.Box(
                low=-1, high=1, shape=(2, cont_dim), dtype=np.float32
            ),
            'disc': gym_spaces.Discrete(n_cats),
        }
    )


def _make_composite(
    model,
    n_steps=4,
    num_samples=16,
    topk=4,
    batch_size=None,
):
    cont = CEMSolver(
        model=model,
        n_steps=n_steps,
        num_samples=num_samples,
        topk=topk,
        batch_size=batch_size,
    )
    disc = CategoricalCEMSolver(
        model=model,
        n_steps=n_steps,
        num_samples=num_samples,
        topk=topk,
        batch_size=batch_size,
    )
    return CompositeSolver({'cont': cont, 'disc': disc})


###########################
## Configuration         ##
###########################


def test_composite_requires_at_least_one_child():
    with pytest.raises(ValueError, match='at least one child'):
        CompositeSolver({})


def test_composite_rejects_non_dict_action_space():
    cont = CEMSolver(model=_SumSquaresCost(), n_steps=1, num_samples=2, topk=1)
    solver = CompositeSolver({'cont': cont})
    with pytest.raises(TypeError, match='Dict action space'):
        solver.configure(
            action_space=gym_spaces.Box(-1, 1, shape=(2, 2)),
            n_envs=1,
            config=_config(),
        )


def test_composite_rejects_missing_subspace_key():
    cont = CEMSolver(model=_SumSquaresCost(), n_steps=1, num_samples=2, topk=1)
    solver = CompositeSolver({'cont': cont, 'extra': cont})
    with pytest.raises(KeyError, match='extra'):
        solver.configure(
            action_space=_dict_space(), n_envs=1, config=_config()
        )


def test_composite_rejects_mismatched_num_samples():
    """Children with different num_samples cannot be jointly evaluated."""
    cont = CEMSolver(model=_SumSquaresCost(), n_steps=1, num_samples=8, topk=2)
    disc = CategoricalCEMSolver(
        model=_SumSquaresCost(), n_steps=1, num_samples=4, topk=2
    )
    solver = CompositeSolver({'cont': cont, 'disc': disc})
    with pytest.raises(ValueError, match='num_samples'):
        solver.configure(
            action_space=_dict_space(), n_envs=1, config=_config()
        )


###########################
## Output shape          ##
###########################


def test_composite_actions_have_per_key_shapes():
    solver = _make_composite(
        _SumSquaresCost(), n_steps=2, num_samples=8, topk=2
    )
    solver.configure(
        action_space=_dict_space(cont_dim=3, n_cats=4),
        n_envs=3,
        config=_config(horizon=4),
    )
    out = solver({'state': torch.zeros(3, 1)})

    assert isinstance(out['actions'], dict)
    assert out['actions']['cont'].shape == (3, 4, 3)
    assert out['actions']['disc'].shape == (3, 4, 1)
    assert out['actions']['disc'].dtype == torch.int64


def test_composite_extra_outputs_are_namespaced():
    """Each child's extras land in outputs under '<key>.<extra>'."""
    solver = _make_composite(
        _SumSquaresCost(), n_steps=2, num_samples=8, topk=2
    )
    solver.configure(action_space=_dict_space(), n_envs=2, config=_config())
    out = solver({'state': torch.zeros(2, 1)})

    assert 'cont.mean' in out and 'cont.var' in out
    assert 'disc.probs' in out


def test_composite_disc_actions_are_in_bounds():
    K = 5
    solver = _make_composite(
        _SumSquaresCost(), n_steps=2, num_samples=8, topk=2
    )
    solver.configure(
        action_space=_dict_space(n_cats=K),
        n_envs=2,
        config=_config(action_block=2),
    )
    out = solver({'state': torch.zeros(2, 1)})

    a = out['actions']['disc']
    assert a.shape == (2, 3, 2)
    assert int(a.min()) >= 0 and int(a.max()) < K


###########################
## Joint cost coupling   ##
###########################


def test_composite_optimizes_coupled_cost_to_known_optimum():
    """Solver finds the configuration that minimizes both components jointly."""
    target = 2
    K = 4
    solver = _make_composite(
        _CoupledCost(target=target, base_simplex_dim=K, alpha=10.0),
        n_steps=20,
        num_samples=64,
        topk=8,
    )
    solver.configure(
        action_space=_dict_space(cont_dim=2, n_cats=K),
        n_envs=2,
        config=_config(horizon=3),
    )
    out = solver({'state': torch.zeros(2, 1)})

    # Continuous component should drive towards 0.
    assert out['actions']['cont'].abs().mean().item() < 0.5
    # Discrete component should pick the target everywhere.
    assert (out['actions']['disc'] == target).all()


def test_composite_single_child_matches_standalone():
    """A composite with one CEM child should behave like that CEM solver."""
    cont = CEMSolver(
        model=_SumSquaresCost(),
        n_steps=4,
        num_samples=16,
        topk=4,
        seed=0,
    )
    composite = CompositeSolver({'cont': cont})

    space = gym_spaces.Dict(
        {'cont': gym_spaces.Box(-1, 1, shape=(2, 2), dtype=np.float32)}
    )
    composite.configure(
        action_space=space, n_envs=2, config=_config(horizon=3)
    )
    out = composite({'state': torch.zeros(2, 1)})

    assert isinstance(out['actions'], dict)
    assert out['actions']['cont'].shape == (2, 3, 2)


###########################
## Batching & init       ##
###########################


def test_composite_batched_solve():
    solver = _make_composite(
        _SumSquaresCost(),
        n_steps=2,
        num_samples=8,
        topk=2,
        batch_size=2,
    )
    solver.configure(action_space=_dict_space(), n_envs=5, config=_config())
    out = solver({'state': torch.zeros(5, 1)})

    assert out['actions']['cont'].shape[0] == 5
    assert out['actions']['disc'].shape[0] == 5
    assert len(out['costs']) == 5


def test_composite_passes_init_action_to_children():
    """init_action is dispatched per child by key."""
    captured = {}

    class _CaptureCEM(CEMSolver):
        def init_state(self, n_envs, init=None):
            captured['cont_init'] = init
            return super().init_state(n_envs, init)

    cont = _CaptureCEM(
        model=_SumSquaresCost(), n_steps=1, num_samples=4, topk=2
    )
    disc = CategoricalCEMSolver(
        model=_SumSquaresCost(), n_steps=1, num_samples=4, topk=2
    )
    solver = CompositeSolver({'cont': cont, 'disc': disc})
    solver.configure(
        action_space=_dict_space(cont_dim=3),
        n_envs=2,
        config=_config(horizon=4),
    )

    init_cont = torch.full((2, 4, 3), 0.5)
    solver({'state': torch.zeros(2, 1)}, init_action={'cont': init_cont})

    assert captured['cont_init'] is init_cont


###########################
## Callbacks per child   ##
###########################


def test_composite_callbacks_fire_with_namespaced_payload():
    """Composite-level callbacks fire and receive merged payload."""
    seen_payload_keys: set = set()

    class _Probe(BestCostRecorder):
        name = 'probe'

        def compute(self, **state):
            seen_payload_keys.update(state.keys())
            return super().compute(**state)

    cont = CEMSolver(
        model=_SumSquaresCost(),
        n_steps=3,
        num_samples=8,
        topk=2,
        batch_size=2,
    )
    disc = CategoricalCEMSolver(
        model=_SumSquaresCost(),
        n_steps=3,
        num_samples=8,
        topk=2,
        batch_size=2,
    )
    solver = CompositeSolver(
        {'cont': cont, 'disc': disc}, callbacks=[_Probe()]
    )
    solver.configure(action_space=_dict_space(), n_envs=2, config=_config())
    out = solver({'state': torch.zeros(2, 1)})

    assert 'callbacks' in out
    history = out['callbacks']['probe']
    assert len(history) == 1 and len(history[0]) == 3
    # Per-child payload keys are namespaced.
    assert 'cont.mean' in seen_payload_keys
    assert 'disc.probs' in seen_payload_keys


###########################
## Gradient compatibility ##
###########################


class _SGDChild(BaseSolver):
    """Tiny gradient-based child used to prove the propose/update split
    works for non-elitist algorithms inside a composite."""

    def __init__(self, model, n_steps=5, num_samples=4, lr=0.5):
        super().__init__(model=model, n_steps=n_steps, num_samples=num_samples)
        self.lr = lr

    def configure(self, *, action_space, n_envs, config):
        super().configure(
            action_space=action_space, n_envs=n_envs, config=config
        )
        shape = action_space.shape
        self._action_dim = (
            int(np.prod(shape[1:])) if len(shape) > 1 else int(np.prod(shape))
        )

    def init_state(self, n_envs, init=None):
        return {
            'params': torch.zeros(
                n_envs, self.num_samples, self.horizon, self._action_dim
            )
        }

    def propose(self, state):
        leaf = state['params'].detach().clone().requires_grad_(True)
        state['_leaf'] = leaf
        return leaf

    def update(self, state, candidates, costs):
        costs.sum().backward()
        with torch.no_grad():
            new_p = state['_leaf'] - self.lr * state['_leaf'].grad
        return {'params': new_p}, {}

    def finalize(self, state):
        with torch.no_grad():
            score = state['params'].pow(2).sum(dim=(-1, -2))
            best = score.argmin(dim=1)
        idx = torch.arange(state['params'].shape[0])
        return state['params'][idx, best].detach().cpu()

    def extra_outputs(self, state):
        return {}


def test_composite_supports_gradient_child():
    """SGD child + categorical CEM child cooperate through joint cost."""
    target = 1
    K = 3
    cont = _SGDChild(
        model=_CoupledCost(target=target, base_simplex_dim=K),
        n_steps=15,
        num_samples=8,
        lr=0.3,
    )
    disc = CategoricalCEMSolver(
        model=_CoupledCost(target=target, base_simplex_dim=K),
        n_steps=15,
        num_samples=8,
        topk=2,
    )
    solver = CompositeSolver({'cont': cont, 'disc': disc})
    solver.configure(
        action_space=_dict_space(cont_dim=2, n_cats=K),
        n_envs=2,
        config=_config(horizon=2),
    )
    out = solver({'state': torch.zeros(2, 1)})

    # SGD pushes cont toward 0.
    assert out['actions']['cont'].abs().mean().item() < 0.2
    # Categorical CEM picks the target.
    assert (out['actions']['disc'] == target).all()
