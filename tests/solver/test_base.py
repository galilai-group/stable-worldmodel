"""Tests for BaseSolver template-method orchestration."""

import numpy as np
import pytest
import torch
from gymnasium import spaces as gym_spaces

from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver.base import BaseSolver
from stable_worldmodel.solver.callbacks import BestCostRecorder


class _QuadraticCost:
    def get_cost(self, info_dict, candidates):
        return candidates.pow(2).sum(dim=(-1, -2))


class _DummySolver(BaseSolver):
    """Minimal subclass: returns identity mean, no real search."""

    def __init__(
        self, model, n_steps=2, num_samples=4, batch_size=None, callbacks=None
    ):
        super().__init__(
            model=model,
            n_steps=n_steps,
            num_samples=num_samples,
            batch_size=batch_size,
            callbacks=callbacks,
        )
        self.calls = {
            'init_state': 0,
            'propose': 0,
            'update': 0,
            'finalize': 0,
        }

    def configure(self, *, action_space, n_envs, config):
        super().configure(
            action_space=action_space, n_envs=n_envs, config=config
        )
        self._action_dim = int(np.prod(action_space.shape[1:])) or 1

    def init_state(self, n_envs, init=None):
        self.calls['init_state'] += 1
        return {
            'mean': torch.zeros(
                n_envs, self.horizon, self._action_dim, dtype=self.dtype
            )
        }

    def propose(self, state):
        self.calls['propose'] += 1
        bs = state['mean'].shape[0]
        # Replicate the mean as N candidates plus tiny perturbation (deterministic).
        return (
            state['mean']
            .unsqueeze(1)
            .expand(bs, self.num_samples, *state['mean'].shape[1:])
            .clone()
        )

    def update(self, state, candidates, costs):
        self.calls['update'] += 1
        new_mean = candidates.mean(dim=1)
        return {'mean': new_mean}, {
            'mean': new_mean,
            'prev_mean': state['mean'],
        }

    def finalize(self, state):
        self.calls['finalize'] += 1
        return state['mean'].detach().cpu()

    def extra_outputs(self, state):
        return {'mean': [state['mean'].detach().cpu()]}


def _config(horizon=3, receding=2, action_block=1):
    return PlanConfig(
        horizon=horizon, receding_horizon=receding, action_block=action_block
    )


def _box_space(dim=2):
    return gym_spaces.Box(low=-1, high=1, shape=(2, dim), dtype=np.float32)


###########################
## Hook lifecycle        ##
###########################


def test_base_solver_calls_all_hooks():
    """init_state, propose, update, finalize each called the expected number of times."""
    solver = _DummySolver(model=_QuadraticCost(), n_steps=4, num_samples=8)
    solver.configure(action_space=_box_space(), n_envs=2, config=_config())

    info = {'state': torch.zeros(2, 1)}
    solver(info)

    assert solver.calls['init_state'] == 1
    # 1 batch (default batch_size = total_envs), n_steps=4 propose/update calls.
    assert solver.calls['propose'] == 4
    assert solver.calls['update'] == 4
    assert solver.calls['finalize'] == 1


def test_base_solver_required_hooks_raise_not_implemented():
    """Direct instantiation cannot solve — hooks are abstract."""
    solver = BaseSolver(model=_QuadraticCost(), n_steps=1, num_samples=1)
    solver.configure(action_space=_box_space(), n_envs=1, config=_config())
    with pytest.raises(NotImplementedError):
        solver({'state': torch.zeros(1, 1)})


###########################
## Output structure      ##
###########################


def test_base_solver_outputs_actions_and_extras():
    solver = _DummySolver(model=_QuadraticCost(), n_steps=2)
    solver.configure(
        action_space=_box_space(dim=3), n_envs=2, config=_config()
    )

    out = solver({'state': torch.zeros(2, 1)})

    assert 'actions' in out and out['actions'].shape == (2, 3, 3)
    assert 'mean' in out and out['mean'][0].shape == (2, 3, 3)
    assert 'costs' in out and len(out['costs']) == 2


def test_base_solver_costs_extends_across_batches():
    """With batch_size < n_envs, outputs['costs'] gets extended per-batch."""
    solver = _DummySolver(model=_QuadraticCost(), n_steps=2, batch_size=2)
    solver.configure(action_space=_box_space(), n_envs=5, config=_config())

    out = solver({'state': torch.zeros(5, 1)})

    assert len(out['costs']) == 5


###########################
## Slicing & write-back  ##
###########################


def test_base_solver_state_slicing_writes_back_correctly():
    """A child that mutates state per-batch produces consistent global state."""

    class _Mutate(_DummySolver):
        def update(self, state, candidates, costs):
            new_mean = state['mean'] + 1.0  # deterministic global +1
            return {'mean': new_mean}, {
                'mean': new_mean,
                'prev_mean': state['mean'],
            }

    solver = _Mutate(model=_QuadraticCost(), n_steps=3, batch_size=2)
    solver.configure(action_space=_box_space(), n_envs=4, config=_config())

    out = solver({'state': torch.zeros(4, 1)})

    # 3 steps of +1 across 4 envs in 2 batches → all envs end at 3.
    assert torch.allclose(out['actions'], torch.full_like(out['actions'], 3.0))


def test_base_solver_handles_tensor_state_directly():
    """Subclasses returning a bare tensor state still slice and write back."""

    class _TensorState(_DummySolver):
        def init_state(self, n_envs, init=None):
            return torch.zeros(
                n_envs, self.horizon, self._action_dim, dtype=self.dtype
            )

        def propose(self, state):
            bs = state.shape[0]
            return (
                state.unsqueeze(1)
                .expand(bs, self.num_samples, *state.shape[1:])
                .clone()
            )

        def update(self, state, candidates, costs):
            return state + 1.0, {}

        def finalize(self, state):
            return state.detach().cpu()

        def extra_outputs(self, state):
            return {}

    solver = _TensorState(model=_QuadraticCost(), n_steps=2, batch_size=2)
    solver.configure(action_space=_box_space(), n_envs=3, config=_config())

    out = solver({'state': torch.zeros(3, 1)})

    assert torch.allclose(out['actions'], torch.full_like(out['actions'], 2.0))


###########################
## Validation            ##
###########################


def test_base_solver_validates_cost_shape():
    """Non-2D cost tensor should raise."""

    class _BadCost:
        def get_cost(self, info, cands):
            return cands.sum()  # scalar

    solver = _DummySolver(model=_BadCost(), n_steps=1)
    solver.configure(action_space=_box_space(), n_envs=2, config=_config())
    with pytest.raises(ValueError, match='cost shape'):
        solver({'state': torch.zeros(2, 1)})


def test_base_solver_validates_cost_type():
    """Non-Tensor cost should raise TypeError."""

    class _BadCost:
        def get_cost(self, info, cands):
            return [0.0] * cands.shape[0]

    solver = _DummySolver(model=_BadCost(), n_steps=1)
    solver.configure(action_space=_box_space(), n_envs=2, config=_config())
    with pytest.raises(TypeError, match='Tensor'):
        solver({'state': torch.zeros(2, 1)})


def test_base_solver_validates_cost_n_matches_candidates():
    """Cost N must match candidate N."""

    class _MismatchedCost:
        def get_cost(self, info, cands):
            # Drop a sample on purpose
            return cands.pow(2).sum(dim=(-1, -2))[:, :-1]

    solver = _DummySolver(model=_MismatchedCost(), n_steps=1, num_samples=4)
    solver.configure(action_space=_box_space(), n_envs=2, config=_config())
    with pytest.raises(ValueError, match='N='):
        solver({'state': torch.zeros(2, 1)})


###########################
## Callbacks integration ##
###########################


def test_base_solver_fires_callbacks_with_payload():
    """Callbacks receive step + candidates + costs + payload kwargs."""
    cb = BestCostRecorder()
    solver = _DummySolver(model=_QuadraticCost(), n_steps=3, callbacks=[cb])
    solver.configure(action_space=_box_space(), n_envs=2, config=_config())

    out = solver({'state': torch.zeros(2, 1)})

    assert 'callbacks' in out
    history = out['callbacks']['BestCostRecorder']
    # 1 batch x 3 steps
    assert len(history) == 1 and len(history[0]) == 3


def test_base_solver_no_callbacks_omits_key():
    solver = _DummySolver(model=_QuadraticCost(), n_steps=2)
    solver.configure(action_space=_box_space(), n_envs=2, config=_config())
    out = solver({'state': torch.zeros(2, 1)})
    assert 'callbacks' not in out


###########################
## Info expansion        ##
###########################


def test_base_solver_expands_info_to_num_samples():
    """get_cost should see info with a num_samples dim inserted at axis 1."""
    captured = {}

    class _CaptureCost:
        def get_cost(self, info, cands):
            captured['shape'] = info['state'].shape
            return cands.pow(2).sum(dim=(-1, -2))

    solver = _DummySolver(model=_CaptureCost(), n_steps=1, num_samples=7)
    solver.configure(action_space=_box_space(), n_envs=3, config=_config())
    solver({'state': torch.zeros(3, 4)})

    assert captured['shape'] == (3, 7, 4)
