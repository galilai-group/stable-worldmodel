"""Tests for EnvPool — self-contained, no real envs needed."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from stable_worldmodel.world.env_pool import (
    EnvPool,
    _broadcast_arg,
    _stack_fresh,
    _write_env_info,
)


# -- Minimal test env -----------------------------------------------------


class CounterEnv(gym.Env):
    """Trivial env that counts steps and terminates after `max_steps`.

    Info contains numpy, torch, and scalar values for stacking tests.
    """

    def __init__(self, max_steps: int = 5):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 1, shape=(4,))
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,))
        self._max_steps = max_steps
        self._step_count = 0
        self._seed = None

    def reset(self, *, seed=None, options=None):
        self._step_count = 0
        self._seed = seed
        obs = np.zeros(4, dtype=np.float32)
        info = self._make_info()
        return obs, info

    def step(self, action):
        self._step_count += 1
        obs = np.full(4, self._step_count, dtype=np.float32)
        reward = float(self._step_count)
        terminated = self._step_count >= self._max_steps
        info = self._make_info()
        return obs, reward, terminated, False, info

    def _make_info(self):
        return {
            'pixels': np.full((3, 3), self._step_count, dtype=np.uint8),
            'state': np.array([self._step_count], dtype=np.float32),
            'tensor_val': torch.tensor([float(self._step_count)]),
            'label': f'step_{self._step_count}',
        }


def _make_pool(n: int = 3, max_steps: int = 5) -> EnvPool:
    return EnvPool([lambda ms=max_steps: CounterEnv(ms) for _ in range(n)])


# -- EnvPool properties ---------------------------------------------------


def test_num_envs():
    pool = _make_pool(4)
    assert pool.num_envs == 4
    pool.close()


def test_spaces():
    pool = _make_pool(2)
    assert pool.action_space.shape == (2, 2)
    assert pool.single_action_space.shape == (2,)
    assert pool.observation_space.shape == (2, 4)
    assert pool.single_observation_space.shape == (4,)
    pool.close()


def test_variation_space_none():
    pool = _make_pool(2)
    assert pool.variation_space is None
    assert pool.single_variation_space is None
    pool.close()


# -- Reset ----------------------------------------------------------------


def test_reset_all():
    pool = _make_pool(3)
    _, infos = pool.reset()

    assert isinstance(infos['pixels'], np.ndarray)
    assert infos['pixels'].shape == (3, 1, 3, 3)
    assert isinstance(infos['tensor_val'], torch.Tensor)
    assert infos['tensor_val'].shape == (3, 1, 1)
    assert isinstance(infos['label'], list)
    assert len(infos['label']) == 3
    pool.close()


def test_reset_with_seed_int():
    pool = _make_pool(3)
    pool.reset(seed=42)
    assert pool.envs[0]._seed == 42
    assert pool.envs[1]._seed == 43
    assert pool.envs[2]._seed == 44
    pool.close()


def test_reset_with_seed_list():
    pool = _make_pool(3)
    pool.reset(seed=[10, 20, 30])
    assert pool.envs[0]._seed == 10
    assert pool.envs[1]._seed == 20
    assert pool.envs[2]._seed == 30
    pool.close()


def test_reset_masked():
    pool = _make_pool(3)
    # full reset first to initialize stacked infos
    pool.reset(seed=0)

    # step all envs once so info changes
    actions = np.zeros((3, 2))
    pool.step(actions)

    # now partial reset — only env 1
    mask = np.array([False, True, False])
    _, infos = pool.reset(seed=[None, 99, None], mask=mask)

    # env 1 was reset (step_count back to 0)
    assert pool.envs[1]._seed == 99
    assert pool.envs[1]._step_count == 0
    assert infos['state'][1] == 0.0

    # env 0 and 2 were NOT reset (still at step 1 from the step above)
    assert pool.envs[0]._step_count == 1
    assert pool.envs[2]._step_count == 1
    assert infos['state'][0] == 1.0
    assert infos['state'][2] == 1.0
    pool.close()


def test_reset_masked_preserves_stacked_type():
    pool = _make_pool(2)
    pool.reset()

    mask = np.array([True, False])
    _, infos = pool.reset(mask=mask)

    # types should be preserved
    assert isinstance(infos['pixels'], np.ndarray)
    assert isinstance(infos['tensor_val'], torch.Tensor)
    assert isinstance(infos['label'], list)
    pool.close()


# -- Step -----------------------------------------------------------------


def test_step_all():
    pool = _make_pool(3)
    pool.reset()

    actions = np.ones((3, 2))
    _, rewards, terms, truncs, infos = pool.step(actions)

    assert rewards.shape == (3,)
    assert np.all(rewards == 1.0)
    assert terms.shape == (3,)
    assert not np.any(terms)
    assert truncs.shape == (3,)
    assert not np.any(truncs)
    # all envs at step 1
    np.testing.assert_array_equal(infos['state'], [[[1.0]], [[1.0]], [[1.0]]])
    pool.close()


def test_step_masked():
    pool = _make_pool(3)
    pool.reset()

    actions = np.ones((3, 2))
    mask = np.array([True, False, True])
    _, rewards, terms, truncs, infos = pool.step(actions, mask=mask)

    # only envs 0 and 2 stepped
    assert rewards[0] == 1.0
    assert rewards[1] == 0.0  # not stepped
    assert rewards[2] == 1.0
    assert not terms[1]  # not stepped

    # env 1 should have its reset info (step_count=0)
    assert infos['state'][1] == 0.0
    # envs 0 and 2 advanced
    assert infos['state'][0] == 1.0
    assert infos['state'][2] == 1.0
    pool.close()


def test_step_termination():
    pool = _make_pool(2, max_steps=2)
    pool.reset()

    actions = np.zeros((2, 2))
    pool.step(actions)  # step 1
    _, _, terms, _, _ = pool.step(actions)  # step 2 → terminates

    assert terms[0]
    assert terms[1]
    pool.close()


def test_step_masked_no_termination_for_skipped():
    pool = _make_pool(2, max_steps=2)
    pool.reset()

    actions = np.zeros((2, 2))
    pool.step(actions)  # step 1 for both

    # only step env 0 (which will terminate)
    mask = np.array([True, False])
    _, _, terms, _, _ = pool.step(actions, mask=mask)

    assert terms[0]  # stepped to completion
    assert not terms[1]  # not stepped, stays at step 1
    pool.close()


# -- In-place updates -----------------------------------------------------


def test_step_updates_inplace():
    """Verify that step writes into pre-allocated arrays, not new ones."""
    pool = _make_pool(2)
    pool.reset()

    actions = np.zeros((2, 2))
    _, _, _, _, infos1 = pool.step(actions)
    pixels_id = id(infos1['pixels'])
    state_id = id(infos1['state'])
    tensor_id = id(infos1['tensor_val'])

    _, _, _, _, infos2 = pool.step(actions)
    assert id(infos2['pixels']) == pixels_id
    assert id(infos2['state']) == state_id
    assert id(infos2['tensor_val']) == tensor_id
    pool.close()


def test_masked_reset_updates_inplace():
    pool = _make_pool(2)
    _, infos1 = pool.reset()
    pixels_id = id(infos1['pixels'])

    mask = np.array([True, False])
    _, infos2 = pool.reset(mask=mask)
    assert id(infos2['pixels']) == pixels_id
    pool.close()


def test_full_reset_reallocates():
    pool = _make_pool(2)
    _, infos1 = pool.reset()
    pixels_id = id(infos1['pixels'])

    _, infos2 = pool.reset()
    # full reset rebuilds arrays
    assert id(infos2['pixels']) != pixels_id
    pool.close()


# -- Multiple steps with mask transitions ---------------------------------


def test_mask_transitions():
    """Simulate DONE mode: envs drop out one by one."""
    pool = _make_pool(3, max_steps=3)
    pool.reset()

    alive = np.ones(3, dtype=bool)
    actions = np.zeros((3, 2))

    for t in range(1, 6):
        mask = alive if not alive.all() else None
        _, _, terms, _, infos = pool.step(actions, mask=mask)

        for i in range(3):
            if alive[i] and terms[i]:
                alive[i] = False

        if not alive.any():
            break

    # all envs should have terminated after 3 steps
    assert not alive.any()
    assert t == 3


# -- Helper functions -----------------------------------------------------


class TestBroadcastArg:
    def test_none(self):
        assert _broadcast_arg(None, 3) == [None, None, None]

    def test_list_passthrough(self):
        arg = [1, 2, 3]
        assert _broadcast_arg(arg, 3) is arg

    def test_ndarray(self):
        result = _broadcast_arg(np.array([10, 20]), 2)
        assert result == [10, 20]

    def test_int_increment(self):
        assert _broadcast_arg(5, 3, increment=True) == [5, 6, 7]

    def test_int_no_increment(self):
        assert _broadcast_arg(5, 3, increment=False) == [5, 5, 5]

    def test_dict_broadcast(self):
        d = {'key': 'val'}
        result = _broadcast_arg(d, 3)
        assert result == [d, d, d]


class TestStackFresh:
    def test_numpy(self):
        infos = [
            {'a': np.array([1, 2])},
            {'a': np.array([3, 4])},
        ]
        stacked = _stack_fresh(infos)
        assert isinstance(stacked['a'], np.ndarray)
        np.testing.assert_array_equal(stacked['a'], [[[1, 2]], [[3, 4]]])

    def test_torch(self):
        infos = [
            {'t': torch.tensor([1.0])},
            {'t': torch.tensor([2.0])},
        ]
        stacked = _stack_fresh(infos)
        assert isinstance(stacked['t'], torch.Tensor)
        assert stacked['t'].shape == (2, 1, 1)

    def test_scalar(self):
        infos = [
            {'s': 'hello'},
            {'s': 'world'},
        ]
        stacked = _stack_fresh(infos)
        assert isinstance(stacked['s'], list)
        assert stacked['s'] == [['hello'], ['world']]

    def test_mixed_types(self):
        infos = [
            {'np': np.array([1]), 't': torch.tensor([1.0]), 'x': 42},
            {'np': np.array([2]), 't': torch.tensor([2.0]), 'x': 43},
        ]
        stacked = _stack_fresh(infos)
        assert isinstance(stacked['np'], np.ndarray)
        assert isinstance(stacked['t'], torch.Tensor)
        assert isinstance(stacked['x'], np.ndarray)


class TestWriteEnvInfo:
    def test_numpy_inplace(self):
        stacked = {'a': np.zeros((3, 1, 2))}
        _write_env_info(stacked, 1, {'a': np.array([5.0, 6.0])})
        np.testing.assert_array_equal(stacked['a'][1, 0], [5.0, 6.0])
        # other slots untouched
        np.testing.assert_array_equal(stacked['a'][0, 0], [0.0, 0.0])

    def test_torch_inplace(self):
        stacked = {'t': torch.zeros(3, 1, 2)}
        _write_env_info(stacked, 2, {'t': torch.tensor([7.0, 8.0])})
        assert torch.equal(stacked['t'][2, 0], torch.tensor([7.0, 8.0]))
        assert torch.equal(stacked['t'][0, 0], torch.zeros(2))

    def test_list_inplace(self):
        stacked = {'l': [['a'], ['b'], ['c']]}
        _write_env_info(stacked, 0, {'l': 'z'})
        assert stacked['l'] == [['z'], ['b'], ['c']]

    def test_unknown_key_ignored(self):
        stacked = {'a': np.zeros(3)}
        _write_env_info(stacked, 0, {'b': 999})
        assert 'b' not in stacked


# -- Edge cases -----------------------------------------------------------


def test_single_env():
    pool = _make_pool(1)
    _, infos = pool.reset(seed=7)

    assert infos['pixels'].shape == (1, 1, 3, 3)
    assert infos['tensor_val'].shape == (1, 1, 1)
    assert pool.envs[0]._seed == 7

    actions = np.zeros((1, 2))
    _, rewards, terms, truncs, infos = pool.step(actions)
    assert rewards.shape == (1,)
    assert infos['state'].shape == (1, 1, 1)
    pool.close()


def test_step_before_reset():
    """Step without reset should fail — _stacked_infos is None."""
    pool = _make_pool(2)
    actions = np.zeros((2, 2))
    with pytest.raises((TypeError, AttributeError)):
        pool.step(actions)
    pool.close()


def test_all_false_mask_step():
    """Stepping with all-False mask should change nothing."""
    pool = _make_pool(3)
    pool.reset()

    actions = np.zeros((3, 2))
    # step once to get to step_count=1
    pool.step(actions)

    mask = np.zeros(3, dtype=bool)
    _, rewards, terms, truncs, infos = pool.step(actions, mask=mask)

    # nothing stepped
    np.testing.assert_array_equal(rewards, [0, 0, 0])
    np.testing.assert_array_equal(terms, [False, False, False])
    # infos unchanged from previous step (step_count=1)
    np.testing.assert_array_equal(infos['state'], [[[1]], [[1]], [[1]]])
    pool.close()


def test_all_true_mask_step():
    """All-True mask should behave identically to mask=None."""
    pool = _make_pool(2)
    pool.reset()
    actions = np.zeros((2, 2))

    mask = np.ones(2, dtype=bool)
    _, rewards, terms, _, infos = pool.step(actions, mask=mask)

    assert rewards[0] == 1.0
    assert rewards[1] == 1.0
    assert infos['state'][0] == 1.0
    assert infos['state'][1] == 1.0
    pool.close()


def test_all_false_mask_reset():
    """Reset with all-False mask should change nothing."""
    pool = _make_pool(2)
    pool.reset(seed=10)

    # step to change state
    actions = np.zeros((2, 2))
    pool.step(actions)

    mask = np.zeros(2, dtype=bool)
    _, infos = pool.reset(seed=99, mask=mask)

    # seeds unchanged
    assert pool.envs[0]._seed == 10
    assert pool.envs[1]._seed == 11
    # step counts unchanged
    assert pool.envs[0]._step_count == 1
    assert pool.envs[1]._step_count == 1
    pool.close()


def test_all_true_mask_reset():
    """All-True mask should reset all envs (but uses in-place path)."""
    pool = _make_pool(2)
    pool.reset(seed=10)

    # step to change state
    actions = np.zeros((2, 2))
    pool.step(actions)
    assert pool.envs[0]._step_count == 1

    mask = np.ones(2, dtype=bool)
    _, infos = pool.reset(seed=50, mask=mask)

    assert pool.envs[0]._seed == 50
    assert pool.envs[1]._seed == 51
    assert pool.envs[0]._step_count == 0
    assert pool.envs[1]._step_count == 0
    pool.close()


def test_seed_as_numpy_array():
    pool = _make_pool(3)
    seeds = np.array([100, 200, 300])
    pool.reset(seed=seeds)
    assert pool.envs[0]._seed == 100
    assert pool.envs[1]._seed == 200
    assert pool.envs[2]._seed == 300
    pool.close()


def test_options_as_list():
    pool = _make_pool(2)
    opts = [{'a': 1}, {'a': 2}]
    # CounterEnv ignores options, but this tests the broadcast path
    pool.reset(options=opts)
    pool.close()


def test_options_as_dict_broadcast():
    pool = _make_pool(2)
    pool.reset(options={'a': 1})
    pool.close()


def test_consecutive_masked_steps_accumulate():
    """Verify state progresses correctly over multiple masked steps."""
    pool = _make_pool(3, max_steps=10)
    pool.reset()
    actions = np.zeros((3, 2))

    # step all once
    pool.step(actions)

    # step only env 0 two more times
    mask = np.array([True, False, False])
    pool.step(actions, mask=mask)
    pool.step(actions, mask=mask)

    assert pool.envs[0]._step_count == 3
    assert pool.envs[1]._step_count == 1
    assert pool.envs[2]._step_count == 1

    _, _, _, _, infos = pool.step(actions)  # step all
    # env 0 is at step 4, others at step 2
    assert infos['state'][0] == 4.0
    assert infos['state'][1] == 2.0
    assert infos['state'][2] == 2.0
    pool.close()


def test_masked_step_then_masked_reset():
    """Partial step followed by partial reset — verify no cross-contamination."""
    pool = _make_pool(3)
    pool.reset(seed=0)
    actions = np.zeros((3, 2))

    # step only env 1
    step_mask = np.array([False, True, False])
    pool.step(actions, mask=step_mask)
    assert pool.envs[1]._step_count == 1

    # reset only env 1
    reset_mask = np.array([False, True, False])
    _, infos = pool.reset(seed=[None, 42, None], mask=reset_mask)

    assert pool.envs[1]._seed == 42
    assert pool.envs[1]._step_count == 0
    # others untouched
    assert pool.envs[0]._step_count == 0
    assert pool.envs[2]._step_count == 0
    # infos consistent
    assert infos['state'][1] == 0.0
    pool.close()


def test_close_is_idempotent():
    pool = _make_pool(2)
    pool.reset()
    pool.close()
    pool.close()  # should not raise


def test_tensor_inplace_after_masked_step():
    """Torch tensors in stacked infos update in-place on masked steps."""
    pool = _make_pool(3)
    pool.reset()
    actions = np.zeros((3, 2))

    _, _, _, _, infos = pool.step(actions)
    tensor_id = id(infos['tensor_val'])

    mask = np.array([False, True, False])
    _, _, _, _, infos2 = pool.step(actions, mask=mask)

    # same tensor object
    assert id(infos2['tensor_val']) == tensor_id
    # only env 1 advanced (step 2), others stayed (step 1)
    assert infos2['tensor_val'][0].item() == 1.0
    assert infos2['tensor_val'][1].item() == 2.0
    assert infos2['tensor_val'][2].item() == 1.0
    pool.close()


def test_label_list_after_masked_step():
    """Non-array info (list of strings) updates correctly on masked steps."""
    pool = _make_pool(2)
    pool.reset()
    actions = np.zeros((2, 2))

    pool.step(actions)  # both at step 1
    mask = np.array([True, False])
    _, _, _, _, infos = pool.step(actions, mask=mask)

    assert infos['label'][0][0] == 'step_2'
    assert infos['label'][1][0] == 'step_1'  # unchanged
    pool.close()


def test_interleaved_termination_and_reset():
    """Simulate autoreset: env terminates, gets reset, continues."""
    pool = _make_pool(2, max_steps=2)
    pool.reset(seed=0)
    actions = np.zeros((2, 2))

    pool.step(actions)  # step 1
    _, _, terms, _, _ = pool.step(actions)  # step 2 → both terminate

    assert terms[0] and terms[1]

    # reset only env 0
    mask = np.array([True, False])
    _, infos = pool.reset(seed=[50, None], mask=mask)

    assert pool.envs[0]._step_count == 0
    assert pool.envs[0]._seed == 50
    # env 1 still at terminated state
    assert pool.envs[1]._step_count == 2

    # step only env 0
    _, rewards, terms, _, infos = pool.step(actions, mask=mask)
    assert rewards[0] == 1.0
    assert rewards[1] == 0.0  # not stepped
    assert infos['state'][0] == 1.0
    assert infos['state'][1] == 2.0  # stale from before
    pool.close()
