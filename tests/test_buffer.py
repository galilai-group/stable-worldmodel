"""Tests for HistoryBuffer."""

from collections import deque

import numpy as np
import pytest
import torch

from stable_worldmodel.buffer import (
    HistoryBuffer,
    _merge_time,
    _slice_env,
    _stack_envs,
)


class TestConstruction:
    def test_basic(self):
        buf = HistoryBuffer(n_envs=3, max_len=5)
        assert buf.n_envs == 3
        assert buf.max_len == 5
        assert buf.action_block == 1
        assert len(buf) == 0

    def test_action_block(self):
        buf = HistoryBuffer(n_envs=2, max_len=5, action_block=3)
        assert buf.action_block == 3

    def test_invalid_n_envs(self):
        with pytest.raises(ValueError, match='n_envs'):
            HistoryBuffer(n_envs=0, max_len=5)
        with pytest.raises(ValueError, match='n_envs'):
            HistoryBuffer(n_envs=-1, max_len=5)

    def test_invalid_max_len(self):
        with pytest.raises(ValueError, match='max_len'):
            HistoryBuffer(n_envs=2, max_len=0)
        with pytest.raises(ValueError, match='max_len'):
            HistoryBuffer(n_envs=2, max_len=-3)

    def test_invalid_action_block(self):
        with pytest.raises(ValueError, match='action_block'):
            HistoryBuffer(n_envs=2, max_len=5, action_block=0)
        with pytest.raises(ValueError, match='action_block'):
            HistoryBuffer(n_envs=2, max_len=5, action_block=-1)

    def test_internal_deques(self):
        buf = HistoryBuffer(n_envs=3, max_len=4)
        assert len(buf._buffers) == 3
        for b in buf._buffers:
            assert isinstance(b, deque)
            assert b.maxlen == 4


class TestAppend:
    def test_single_append(self):
        buf = HistoryBuffer(n_envs=2, max_len=5)
        buf.append({'x': np.array([[1.0], [2.0]])})
        assert len(buf) == 1
        np.testing.assert_array_equal(buf._buffers[0][0]['x'], [1.0])
        np.testing.assert_array_equal(buf._buffers[1][0]['x'], [2.0])

    def test_multiple_appends_increase_length(self):
        buf = HistoryBuffer(n_envs=2, max_len=5)
        for i in range(3):
            buf.append({'x': np.full((2, 1), i, dtype=np.float32)})
        assert len(buf) == 3
        for env_buf in buf._buffers:
            assert len(env_buf) == 3

    def test_max_len_eviction(self):
        buf = HistoryBuffer(n_envs=2, max_len=2)
        for i in range(5):
            buf.append({'x': np.full((2, 1), i, dtype=np.float32)})
        assert len(buf) == 2
        np.testing.assert_array_equal(buf._buffers[0][0]['x'], [3.0])
        np.testing.assert_array_equal(buf._buffers[0][1]['x'], [4.0])

    def test_torch_values(self):
        buf = HistoryBuffer(n_envs=2, max_len=5)
        buf.append({'t': torch.tensor([[1.0, 2.0], [3.0, 4.0]])})
        assert torch.equal(buf._buffers[0][0]['t'], torch.tensor([1.0, 2.0]))
        assert torch.equal(buf._buffers[1][0]['t'], torch.tensor([3.0, 4.0]))

    def test_scalar_value_passes_through(self):
        """Non-array values are stored as-is for every env (no slicing)."""
        buf = HistoryBuffer(n_envs=2, max_len=5)
        buf.append({'tag': 'hello', 'arr': np.array([10.0, 20.0])})
        assert buf._buffers[0][0]['tag'] == 'hello'
        assert buf._buffers[1][0]['tag'] == 'hello'
        assert buf._buffers[0][0]['arr'] == 10.0
        assert buf._buffers[1][0]['arr'] == 20.0

    def test_mixed_types(self):
        buf = HistoryBuffer(n_envs=2, max_len=5)
        buf.append(
            {
                'np': np.array([[1.0], [2.0]]),
                't': torch.tensor([[3.0], [4.0]]),
                'lbl': 'x',
            }
        )
        e0 = buf._buffers[0][0]
        e1 = buf._buffers[1][0]
        assert isinstance(e0['np'], np.ndarray)
        assert isinstance(e0['t'], torch.Tensor)
        assert e0['lbl'] == 'x'
        np.testing.assert_array_equal(e0['np'], [1.0])
        np.testing.assert_array_equal(e1['np'], [2.0])


class TestGet:
    def test_n_zero_returns_none(self):
        buf = HistoryBuffer(n_envs=2, max_len=5)
        buf.append({'x': np.array([[1.0], [2.0]])})
        assert buf.get(0) is None

    def test_n_negative_returns_none(self):
        buf = HistoryBuffer(n_envs=2, max_len=5)
        buf.append({'x': np.array([[1.0], [2.0]])})
        assert buf.get(-1) is None

    def test_empty_returns_none(self):
        buf = HistoryBuffer(n_envs=2, max_len=5)
        assert buf.get(1) is None

    def test_warmup_returns_partial(self):
        """During warm-up, get(n) returns the largest k <= n that fits."""
        buf = HistoryBuffer(n_envs=2, max_len=5)
        buf.append({'x': np.array([[[1.0]], [[2.0]]])})
        out = buf.get(3)
        assert out is not None
        assert out['x'].shape == (2, 1, 1)
        np.testing.assert_array_equal(out['x'][0, :, 0], [1.0])
        np.testing.assert_array_equal(out['x'][1, :, 0], [2.0])

    def test_one_env_empty_returns_none(self):
        """If any single env has zero entries, get returns None."""
        buf = HistoryBuffer(n_envs=2, max_len=5)
        buf._buffers[0].append({'x': np.array([1.0])})
        assert buf.get(1) is None

    def test_chronological_order(self):
        buf = HistoryBuffer(n_envs=2, max_len=5)
        for i in range(3):
            buf.append({'x': np.full((2, 1, 1), i, dtype=np.float32)})

        out = buf.get(3)
        assert out is not None
        assert out['x'].shape == (2, 3, 1)
        np.testing.assert_array_equal(out['x'][0, :, 0], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(out['x'][1, :, 0], [0.0, 1.0, 2.0])

    def test_action_block_stride(self):
        buf = HistoryBuffer(n_envs=1, max_len=10, action_block=2)
        for i in range(3):
            buf.append({'x': np.full((1, 1, 1), i, dtype=np.float32)})

        out = buf.get(2)
        np.testing.assert_array_equal(out['x'][0, :, 0], [0.0, 2.0])

    def test_action_block_warmup(self):
        """get(n) auto-grows: with k=(min_len-1)//action_block + 1 strides."""
        buf = HistoryBuffer(n_envs=1, max_len=10, action_block=2)
        for i in range(2):
            buf.append({'x': np.full((1, 1, 1), i, dtype=np.float32)})
        out = buf.get(2)
        assert out is not None
        assert out['x'].shape == (1, 1, 1)
        np.testing.assert_array_equal(out['x'][0, :, 0], [1.0])

    def test_warmup_only_oldest(self):
        """With one entry, get(3) returns just that entry (k=1)."""
        buf = HistoryBuffer(n_envs=1, max_len=10)
        buf.append({'x': np.array([[[5.0]]])})
        out = buf.get(3)
        assert out is not None
        assert out['x'].shape == (1, 1, 1)
        np.testing.assert_array_equal(out['x'][0, :, 0], [5.0])

    def test_warmup_partial(self):
        """With two entries, get(3) returns the two in chronological order."""
        buf = HistoryBuffer(n_envs=1, max_len=10)
        buf.append({'x': np.array([[[1.0]]])})
        buf.append({'x': np.array([[[2.0]]])})
        out = buf.get(3)
        assert out['x'].shape == (1, 2, 1)
        np.testing.assert_array_equal(out['x'][0, :, 0], [1.0, 2.0])

    def test_per_env_independent(self):
        buf = HistoryBuffer(n_envs=2, max_len=5)
        for i in range(3):
            arr = np.array([[[float(i)]], [[float(i * 10)]]])
            buf.append({'x': arr})
        out = buf.get(3)
        np.testing.assert_array_equal(out['x'][0, :, 0], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(out['x'][1, :, 0], [0.0, 10.0, 20.0])

    def test_torch_preserved(self):
        buf = HistoryBuffer(n_envs=2, max_len=5)
        for i in range(2):
            buf.append({'t': torch.tensor([[[float(i)]], [[float(i + 5)]]])})
        out = buf.get(2)
        assert isinstance(out['t'], torch.Tensor)
        assert out['t'].shape == (2, 2, 1)
        torch.testing.assert_close(out['t'][0, :, 0], torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(out['t'][1, :, 0], torch.tensor([5.0, 6.0]))

    def test_scalar_per_env_shape(self):
        """Per-step (n_envs,) input yields output (n_envs, n)."""
        buf = HistoryBuffer(n_envs=2, max_len=5)
        for i in range(3):
            buf.append({'r': np.array([float(i), float(i + 100)])})
        out = buf.get(3)
        assert out['r'].shape == (2, 3)
        np.testing.assert_array_equal(out['r'][0], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(out['r'][1], [100.0, 101.0, 102.0])

    def test_multi_step_time_dim(self):
        """Per-step (n_envs, T, ...) with T>1 yields (n_envs, n*T, ...)."""
        buf = HistoryBuffer(n_envs=2, max_len=5)
        for i in range(3):
            arr = np.full((2, 2, 4), i, dtype=np.float32)
            buf.append({'x': arr})
        out = buf.get(3)
        assert out['x'].shape == (2, 6, 4)
        expected = np.array([0, 0, 1, 1, 2, 2], dtype=np.float32)
        np.testing.assert_array_equal(out['x'][0, :, 0], expected)

    def test_get_after_eviction(self):
        buf = HistoryBuffer(n_envs=1, max_len=2)
        for i in range(5):
            buf.append({'x': np.full((1, 1, 1), i, dtype=np.float32)})
        out = buf.get(2)
        np.testing.assert_array_equal(out['x'][0, :, 0], [3.0, 4.0])

    def test_mixed_keys_preserve_types(self):
        buf = HistoryBuffer(n_envs=2, max_len=5)
        for i in range(2):
            buf.append(
                {
                    'np': np.full((2, 1, 3), i, dtype=np.float32),
                    't': torch.full((2, 1, 3), float(i)),
                }
            )
        out = buf.get(2)
        assert isinstance(out['np'], np.ndarray)
        assert isinstance(out['t'], torch.Tensor)
        assert out['np'].shape == (2, 2, 3)
        assert out['t'].shape == (2, 2, 3)


class TestBlockKeys:
    def test_blocked_shape(self):
        """Block key output is (n_envs, k, action_block * D)."""
        buf = HistoryBuffer(
            n_envs=2, max_len=10, action_block=2, block_keys=('a',)
        )
        for i in range(4):
            buf.append({'a': np.full((2, 1, 3), i, dtype=np.float32)})
        out = buf.get(2)
        assert out['a'].shape == (2, 2, 6)

    def test_blocked_window_chronological(self):
        """Each block concatenates raw entries oldest -> newest."""
        buf = HistoryBuffer(
            n_envs=1, max_len=10, action_block=2, block_keys=('a',)
        )
        for i in range(4):
            buf.append({'a': np.array([[[float(i)]]])})
        out = buf.get(2)
        assert out['a'].shape == (1, 2, 2)
        np.testing.assert_array_equal(out['a'][0, 0], [0.0, 1.0])
        np.testing.assert_array_equal(out['a'][0, 1], [2.0, 3.0])

    def test_blocked_warmup_requires_full_window(self):
        """k for blocked keys is min_len // action_block (full windows)."""
        buf = HistoryBuffer(
            n_envs=1, max_len=10, action_block=2, block_keys=('a',)
        )
        buf.append({'a': np.array([[[1.0]]])})
        assert buf.get(2) is None
        buf.append({'a': np.array([[[2.0]]])})
        out = buf.get(2)
        assert out is not None
        assert out['a'].shape == (1, 1, 2)
        np.testing.assert_array_equal(out['a'][0, 0], [1.0, 2.0])

    def test_blocked_only_when_action_block_gt_one(self):
        """With action_block=1, block_keys behaves like a normal key."""
        buf = HistoryBuffer(
            n_envs=1, max_len=10, action_block=1, block_keys=('a',)
        )
        for i in range(2):
            buf.append({'a': np.array([[[float(i)]]])})
        out = buf.get(2)
        assert out['a'].shape == (1, 2, 1)
        np.testing.assert_array_equal(out['a'][0, :, 0], [0.0, 1.0])

    def test_blocked_mixed_with_regular_keys(self):
        """Block keys and regular keys coexist with consistent k."""
        buf = HistoryBuffer(
            n_envs=1, max_len=10, action_block=2, block_keys=('a',)
        )
        for i in range(4):
            buf.append(
                {
                    'a': np.array([[[float(i)]]]),
                    'o': np.array([[[float(i * 10)]]]),
                }
            )
        out = buf.get(2)
        assert out['a'].shape == (1, 2, 2)
        assert out['o'].shape == (1, 2, 1)
        np.testing.assert_array_equal(out['o'][0, :, 0], [10.0, 30.0])

    def test_blocked_torch(self):
        buf = HistoryBuffer(
            n_envs=1, max_len=10, action_block=2, block_keys=('a',)
        )
        for i in range(4):
            buf.append({'a': torch.tensor([[[float(i)]]])})
        out = buf.get(2)
        assert isinstance(out['a'], torch.Tensor)
        assert out['a'].shape == (1, 2, 2)
        torch.testing.assert_close(out['a'][0, 0], torch.tensor([0.0, 1.0]))


class TestReset:
    def test_reset_all(self):
        buf = HistoryBuffer(n_envs=3, max_len=5)
        for i in range(3):
            buf.append({'x': np.full((3, 1), i, dtype=np.float32)})
        buf.reset()
        assert len(buf) == 0
        for env_buf in buf._buffers:
            assert len(env_buf) == 0

    def test_reset_subset(self):
        buf = HistoryBuffer(n_envs=3, max_len=5)
        for i in range(3):
            buf.append({'x': np.full((3, 1), i, dtype=np.float32)})

        buf.reset([1])
        assert len(buf._buffers[0]) == 3
        assert len(buf._buffers[1]) == 0
        assert len(buf._buffers[2]) == 3

    def test_reset_empty_list_is_noop(self):
        buf = HistoryBuffer(n_envs=2, max_len=5)
        buf.append({'x': np.array([[1.0], [2.0]])})
        buf.reset([])
        assert len(buf) == 1

    def test_reset_then_append(self):
        buf = HistoryBuffer(n_envs=2, max_len=5)
        buf.append({'x': np.array([[[1.0]], [[2.0]]])})
        buf.reset()
        buf.append({'x': np.array([[[9.0]], [[10.0]]])})
        out = buf.get(1)
        np.testing.assert_array_equal(out['x'][0, :, 0], [9.0])
        np.testing.assert_array_equal(out['x'][1, :, 0], [10.0])


class TestLen:
    def test_empty(self):
        buf = HistoryBuffer(n_envs=3, max_len=5)
        assert len(buf) == 0

    def test_uniform(self):
        buf = HistoryBuffer(n_envs=3, max_len=5)
        for _ in range(2):
            buf.append({'x': np.zeros((3, 1))})
        assert len(buf) == 2

    def test_min_after_partial_reset(self):
        buf = HistoryBuffer(n_envs=3, max_len=5)
        for _ in range(4):
            buf.append({'x': np.zeros((3, 1))})
        buf.reset([0])
        assert len(buf) == 0


class TestSliceEnv:
    def test_numpy(self):
        v = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(_slice_env(v, 1), [3, 4])

    def test_torch(self):
        v = torch.tensor([[1.0], [2.0]])
        torch.testing.assert_close(_slice_env(v, 0), torch.tensor([1.0]))

    def test_passthrough_str(self):
        assert _slice_env('hello', 0) == 'hello'

    def test_passthrough_int(self):
        assert _slice_env(42, 1) == 42


class TestMergeTime:
    def test_torch_zero_dim(self):
        out = _merge_time([torch.tensor(1.0), torch.tensor(2.0)])
        assert out.shape == (2,)
        torch.testing.assert_close(out, torch.tensor([1.0, 2.0]))

    def test_torch_multi_dim(self):
        out = _merge_time([torch.zeros(2, 3), torch.ones(2, 3)])
        assert out.shape == (4, 3)

    def test_numpy_zero_dim(self):
        out = _merge_time([np.array(1.0), np.array(2.0)])
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)

    def test_numpy_multi_dim(self):
        out = _merge_time([np.zeros((2, 3)), np.ones((2, 3))])
        assert out.shape == (4, 3)

    def test_fallback_for_scalars(self):
        out = _merge_time([1, 2, 3])
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, [1, 2, 3])


class TestStackEnvs:
    def test_torch(self):
        out = _stack_envs([torch.tensor([1.0]), torch.tensor([2.0])])
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 1)

    def test_numpy(self):
        out = _stack_envs([np.array([1.0]), np.array([2.0])])
        assert isinstance(out, np.ndarray)
        assert out.shape == (2, 1)

    def test_fallback(self):
        out = _stack_envs([1, 2, 3])
        assert isinstance(out, np.ndarray)
