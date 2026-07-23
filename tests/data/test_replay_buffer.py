"""Tests for ReplayBuffer: in-memory ring storage that doubles as a
Dataset and a Writer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stable_worldmodel.data import (
    FolderDataset,
    HDF5Dataset,
    ReplayBuffer,
)
from stable_worldmodel.data.dataset import Dataset


def _episode(n_steps: int, *, seed: int = 0, with_pixels: bool = True) -> dict:
    """Build an in-memory episode with bulk per-step arrays.

    Note: ReplayBuffer accepts arrays *or* lists for write_episode; tests
    deliberately exercise the array path because rollouts often produce
    bulk arrays that we want to be a no-op to ingest.
    """
    rng = np.random.default_rng(seed)
    ep = {
        'action': rng.standard_normal((n_steps, 2)).astype(np.float32),
        'proprio': rng.standard_normal((n_steps, 4)).astype(np.float32),
        'reward': rng.standard_normal(n_steps).astype(np.float32),
    }
    if with_pixels:
        ep['pixels'] = rng.integers(
            0, 256, size=(n_steps, 8, 8, 3), dtype=np.uint8
        )
    return ep


@pytest.fixture
def filled_buf():
    """A small filled buffer used by several read-side tests."""
    buf = ReplayBuffer(max_steps=200, history_len=4)
    for s in range(4):
        buf.write_episode(_episode(20, seed=s))
    return buf


class TestInit:
    def test_subclasses_dataset(self):
        # ReplayBuffer must be substitutable wherever a Dataset is expected.
        buf = ReplayBuffer(max_steps=10)
        assert isinstance(buf, Dataset)

    def test_default_state_is_empty(self):
        buf = ReplayBuffer(max_steps=10, history_len=2, frameskip=1)
        assert len(buf) == 0
        assert buf.num_episodes == 0
        assert buf.num_steps_stored == 0
        assert buf.num_valid_ends() == 0
        assert buf.column_names == []

    def test_history_len_doubles_as_num_steps(self):
        buf = ReplayBuffer(max_steps=10, history_len=4)
        assert buf.num_steps == 4
        assert buf.span == 4 * 1

    def test_span_includes_frameskip(self):
        buf = ReplayBuffer(max_steps=10, history_len=4, frameskip=3)
        assert buf.span == 12

    @pytest.mark.parametrize(
        'kw',
        [
            {'max_steps': 0},
            {'max_steps': -1},
            {'max_steps': 10, 'history_len': 0},
            {'max_steps': 10, 'history_len': -2},
            {'max_steps': 10, 'frameskip': 0},
            {'max_steps': 10, 'frameskip': -1},
        ],
    )
    def test_rejects_non_positive(self, kw):
        with pytest.raises(ValueError, match='must be positive'):
            ReplayBuffer(**kw)


class TestWriteEpisode:
    def test_lazy_allocation_from_first_episode(self):
        buf = ReplayBuffer(max_steps=50)
        # No columns until something is written.
        assert buf._cols == {}
        buf.write_episode(_episode(5, seed=0))
        assert set(buf.column_names) == {
            'pixels',
            'proprio',
            'action',
            'reward',
        }
        # Each ring array is sized to max_steps.
        assert buf._cols['pixels'].shape == (50, 8, 8, 3)
        assert buf._cols['proprio'].shape == (50, 4)
        assert buf._cols['action'].shape == (50, 2)
        assert buf._cols['reward'].shape == (50,)

    def test_dtype_inferred_from_first_episode(self):
        buf = ReplayBuffer(max_steps=10)
        buf.write_episode(_episode(3, seed=0))
        assert buf._cols['pixels'].dtype == np.uint8
        assert buf._cols['action'].dtype == np.float32
        assert buf._cols['reward'].dtype == np.float32

    def test_accepts_per_step_lists(self):
        # World.collect produces dict[col, list[step_arr]] — must work without coercion.
        n = 5
        ep_list = {
            'pixels': [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(n)],
            'action': [np.zeros(2, dtype=np.float32) for _ in range(n)],
        }
        buf = ReplayBuffer(max_steps=10)
        buf.write_episode(ep_list)
        assert buf.num_episodes == 1
        assert buf.num_steps_stored == n

    def test_empty_dict_is_noop(self):
        buf = ReplayBuffer(max_steps=10)
        buf.write_episode({})
        assert buf.num_episodes == 0
        assert buf.num_steps_stored == 0

    def test_zero_length_episode_is_noop(self):
        buf = ReplayBuffer(max_steps=10)
        buf.write_episode({'action': np.zeros((0, 2), dtype=np.float32)})
        assert buf.num_episodes == 0
        assert buf.num_steps_stored == 0

    def test_rejects_scalar_column(self):
        buf = ReplayBuffer(max_steps=10)
        with pytest.raises(ValueError, match='scalar'):
            buf.write_episode({'action': 42})

    def test_rejects_oversize_episode(self):
        buf = ReplayBuffer(max_steps=10)
        with pytest.raises(ValueError, match='exceeds max_steps'):
            buf.write_episode(_episode(20, seed=0))

    def test_rejects_inconsistent_lengths(self):
        buf = ReplayBuffer(max_steps=10)
        with pytest.raises(ValueError, match='expected'):
            buf.write_episode(
                {
                    'action': np.zeros((5, 2), dtype=np.float32),
                    'proprio': np.zeros((4, 4), dtype=np.float32),
                }
            )

    def test_rejects_schema_drift_missing_column(self):
        buf = ReplayBuffer(max_steps=20)
        buf.write_episode(_episode(5, seed=0))
        with pytest.raises(ValueError, match='schema mismatch'):
            buf.write_episode(
                {'pixels': np.zeros((3, 8, 8, 3), dtype=np.uint8)}
            )

    def test_rejects_schema_drift_extra_column(self):
        buf = ReplayBuffer(max_steps=20)
        buf.write_episode(_episode(5, seed=0))
        ep = _episode(5, seed=1)
        ep['extra'] = np.zeros((5, 3), dtype=np.float32)
        with pytest.raises(ValueError, match='schema mismatch'):
            buf.write_episode(ep)

    def test_rejects_per_step_shape_drift(self):
        buf = ReplayBuffer(max_steps=20)
        buf.write_episode(_episode(5, seed=0))
        ep = _episode(5, seed=1)
        ep['proprio'] = np.zeros((5, 9), dtype=np.float32)  # wrong feature dim
        with pytest.raises(ValueError, match='per-step'):
            buf.write_episode(ep)

    def test_write_episodes_iterates(self):
        buf = ReplayBuffer(max_steps=100)
        buf.write_episodes(_episode(10, seed=s) for s in range(3))
        assert buf.num_episodes == 3
        assert buf.num_steps_stored == 30

    def test_context_manager_compatible(self):
        # Must work as a Writer in a `with`-statement (matching the protocol
        # used by World.collect).
        buf = ReplayBuffer(max_steps=20)
        with buf as w:
            assert w is buf
            w.write_episode(_episode(5, seed=0))
        assert buf.num_episodes == 1


class TestEviction:
    def test_evicts_oldest_to_fit(self):
        buf = ReplayBuffer(max_steps=50, history_len=1)
        buf.write_episode(_episode(30, seed=0))
        buf.write_episode(_episode(15, seed=1))
        # Adding 20 → 50 - 20 = 30 free. Need to drop oldest (30) → fits.
        buf.write_episode(_episode(20, seed=2))
        assert buf.num_episodes == 2
        assert buf.num_steps_stored == 35

    def test_evicts_multiple_when_needed(self):
        buf = ReplayBuffer(max_steps=30, history_len=1)
        for s in range(3):  # 3 × 10 = 30 fills exactly
            buf.write_episode(_episode(10, seed=s))
        # Adding 20 → must evict two of the 10-step episodes to fit.
        buf.write_episode(_episode(20, seed=99))
        assert buf.num_episodes == 2
        assert buf.num_steps_stored == 30

    def test_evicted_data_no_longer_visible(self):
        buf = ReplayBuffer(max_steps=20, history_len=1)
        ep0 = _episode(10, seed=10)
        ep1 = _episode(10, seed=11)
        ep2 = _episode(10, seed=12)  # forces eviction of ep0
        buf.write_episode(ep0)
        buf.write_episode(ep1)
        buf.write_episode(ep2)

        # The oldest still-stored episode is now ep1. Sample its first step.
        first = buf[0]
        np.testing.assert_array_equal(first['pixels'][0], ep1['pixels'][0])

    def test_at_capacity_no_unnecessary_eviction(self):
        # Episode fits exactly with 0 slack — should still write without
        # evicting anything that didn't need to go.
        buf = ReplayBuffer(max_steps=10, history_len=1)
        buf.write_episode(_episode(10, seed=0))
        assert buf.num_episodes == 1
        # Now full. Next write of any size must evict the existing episode.
        buf.write_episode(_episode(5, seed=1))
        assert buf.num_episodes == 1
        assert buf.num_steps_stored == 5


class TestSample:
    def test_default_uniform_sampler_returns_correct_shapes(self, filled_buf):
        out = filled_buf.sample(batch_size=8, history_len=4)
        assert out['pixels'].shape == (8, 4, 8, 8, 3)
        assert out['proprio'].shape == (8, 4, 4)
        assert out['action'].shape == (8, 4, 2)
        assert out['reward'].shape == (8, 4)

    def test_uniform_sampler_raises_on_empty(self):
        buf = ReplayBuffer(max_steps=20, history_len=4)
        with pytest.raises(RuntimeError, match='no clips'):
            buf.sample(batch_size=4)

    def test_uniform_sampler_raises_when_episodes_too_short(self):
        # All stored episodes are shorter than history_len → no valid clips.
        buf = ReplayBuffer(max_steps=20, history_len=10)
        buf.write_episode(_episode(5, seed=0))
        assert buf.num_valid_ends() == 0
        with pytest.raises(RuntimeError, match='no clips'):
            buf.sample(batch_size=2)

    def test_history_len_default_is_constructor_value(self, filled_buf):
        out = filled_buf.sample(batch_size=2)
        # filled_buf has history_len=4
        assert out['action'].shape == (2, 4, 2)

    def test_history_len_override(self, filled_buf):
        out = filled_buf.sample(batch_size=2, history_len=8)
        assert out['action'].shape == (2, 8, 2)

    def test_step_counter_auto_increments(self, filled_buf):
        seen = []

        def sampler(step, buffer, batch_size, history_len):
            seen.append(step)
            return np.zeros(batch_size, dtype=np.int64)

        filled_buf.sampler = sampler
        filled_buf.sample(2)
        filled_buf.sample(2)
        filled_buf.sample(2)
        assert seen == [0, 1, 2]

    def test_explicit_step_overrides_counter(self, filled_buf):
        seen = []

        def sampler(step, buffer, batch_size, history_len):
            seen.append(step)
            return np.zeros(batch_size, dtype=np.int64)

        filled_buf.sampler = sampler
        filled_buf.sample(2, step=42)
        filled_buf.sample(2, step=100)
        assert seen == [42, 100]

    def test_explicit_step_does_not_advance_counter(self, filled_buf):
        seen = []

        def sampler(step, buffer, batch_size, history_len):
            seen.append(step)
            return np.zeros(batch_size, dtype=np.int64)

        filled_buf.sampler = sampler
        filled_buf.sample(2, step=999)
        filled_buf.sample(2)  # back to internal counter
        assert seen == [999, 0]

    def test_sampler_receives_buffer_reference(self, filled_buf):
        captured = {}

        def sampler(step, buffer, batch_size, history_len):
            captured['buffer'] = buffer
            captured['history_len'] = history_len
            return np.arange(batch_size, dtype=np.int64)

        filled_buf.sampler = sampler
        filled_buf.sample(3, history_len=2)
        assert captured['buffer'] is filled_buf
        assert captured['history_len'] == 2

    def test_recency_biased_sampler(self):
        """A sampler that always picks the last K clips returns data from
        the most recent episode."""
        buf = ReplayBuffer(max_steps=200, history_len=2)
        episodes = [_episode(20, seed=s) for s in range(4)]
        for ep in episodes:
            buf.write_episode(ep)

        def recent(step, buffer, batch_size, history_len):
            n = buffer.num_valid_ends(history_len)
            return np.full(batch_size, n - 1, dtype=np.int64)

        buf.sampler = recent
        out = buf.sample(batch_size=3, history_len=2)
        # Last clip's pixels[-1] must equal last step of last-written episode.
        np.testing.assert_array_equal(
            out['pixels'][0, -1], episodes[-1]['pixels'][-1]
        )

    def test_sampler_returning_wrong_shape_raises(self, filled_buf):
        def bad(step, buffer, batch_size, history_len):
            return np.zeros(batch_size + 1, dtype=np.int64)

        filled_buf.sampler = bad
        with pytest.raises(ValueError, match='returned shape'):
            filled_buf.sample(batch_size=4)

    def test_sampler_returning_out_of_range_raises(self, filled_buf):
        def bad(step, buffer, batch_size, history_len):
            return np.full(batch_size, 10**9, dtype=np.int64)

        filled_buf.sampler = bad
        with pytest.raises(IndexError, match='out of range'):
            filled_buf.sample(batch_size=4)

    @pytest.mark.parametrize('batch_size', [0, -1, -10])
    def test_rejects_non_positive_batch_size(self, filled_buf, batch_size):
        with pytest.raises(ValueError, match='batch_size'):
            filled_buf.sample(batch_size=batch_size)

    @pytest.mark.parametrize('history_len', [0, -1])
    def test_rejects_non_positive_history_len(self, filled_buf, history_len):
        with pytest.raises(ValueError, match='history_len'):
            filled_buf.sample(batch_size=4, history_len=history_len)

    def test_step_conditioned_curriculum(self):
        """A real curriculum: early steps prefer the most recent 5 clips,
        later steps draw uniformly. Confirms the step is plumbed through.
        """
        decisions = []

        def curriculum(step, buffer, batch_size, history_len):
            n = buffer.num_valid_ends(history_len)
            if step < 10:
                decisions.append('recent')
                return np.full(batch_size, n - 1, dtype=np.int64)
            decisions.append('uniform')
            return np.zeros(batch_size, dtype=np.int64)

        buf = ReplayBuffer(max_steps=100, history_len=2, sampler=curriculum)
        for s in range(3):
            buf.write_episode(_episode(20, seed=s))

        for _ in range(15):
            buf.sample(2)

        assert decisions[:10] == ['recent'] * 10
        assert decisions[10:] == ['uniform'] * 5


class TestFrameskip:
    def test_observation_columns_strided(self):
        buf = ReplayBuffer(max_steps=200, history_len=4, frameskip=2)
        ep = _episode(20, seed=0)
        buf.write_episode(ep)

        item = buf[0]
        # 4 strided observations from positions {0, 2, 4, 6}.
        np.testing.assert_array_equal(
            item['pixels'], ep['pixels'][[0, 2, 4, 6]]
        )
        np.testing.assert_array_equal(
            item['proprio'], ep['proprio'][[0, 2, 4, 6]]
        )

    def test_action_kept_dense_and_reshaped(self):
        buf = ReplayBuffer(max_steps=200, history_len=4, frameskip=2)
        ep = _episode(20, seed=0)
        buf.write_episode(ep)

        item = buf[0]
        # action shape = (history_len, frameskip * action_dim) = (4, 4)
        assert item['action'].shape == (4, 4)
        expected = ep['action'][0:8].reshape(4, 4)
        np.testing.assert_array_equal(item['action'], expected)

    def test_sample_with_frameskip(self):
        buf = ReplayBuffer(max_steps=200, history_len=3, frameskip=4)
        for s in range(3):
            buf.write_episode(_episode(40, seed=s))

        out = buf.sample(batch_size=5)
        # Observations: (B, history_len, ...)
        assert out['pixels'].shape == (5, 3, 8, 8, 3)
        # Action: (B, history_len, frameskip * action_dim) = (5, 3, 8)
        assert out['action'].shape == (5, 3, 8)

    def test_valid_clip_count_with_frameskip(self):
        # length=10, frameskip=2, history=4 → span=8 → 10 - 8 + 1 = 3 valid clips
        buf = ReplayBuffer(max_steps=20, history_len=4, frameskip=2)
        buf.write_episode(_episode(10, seed=0))
        assert buf.num_valid_ends() == 3

    def test_clips_too_long_for_episode_yield_zero(self):
        buf = ReplayBuffer(max_steps=20, history_len=10, frameskip=2)
        buf.write_episode(_episode(10, seed=0))
        # span = 20 > length = 10 → no valid clips
        assert buf.num_valid_ends() == 0


class TestDatasetInterface:
    def test_len_equals_num_valid_ends(self, filled_buf):
        assert len(filled_buf) == filled_buf.num_valid_ends()

    def test_getitem_zero_returns_first_clip(self, filled_buf):
        item = filled_buf[0]
        assert item['pixels'].shape == (4, 8, 8, 3)
        assert item['action'].shape == (4, 2)

    def test_getitem_negative_indexes_from_end(self, filled_buf):
        last_pos = filled_buf[len(filled_buf) - 1]
        last_neg = filled_buf[-1]
        for k in last_pos:
            np.testing.assert_array_equal(last_pos[k], last_neg[k])

    def test_getitem_out_of_range_raises(self, filled_buf):
        with pytest.raises(IndexError):
            _ = filled_buf[len(filled_buf)]

    def test_getitem_empty_buffer_raises(self):
        buf = ReplayBuffer(max_steps=10, history_len=2)
        with pytest.raises(IndexError, match='empty'):
            _ = buf[0]

    def test_transform_applied_in_dataset_path(self):
        def to_tensor(clip):
            return {k: torch.as_tensor(np.asarray(v)) for k, v in clip.items()}

        buf = ReplayBuffer(max_steps=50, history_len=2, transform=to_tensor)
        buf.write_episode(_episode(10, seed=0))
        item = buf[0]
        for v in item.values():
            assert isinstance(v, torch.Tensor)

    def test_dataloader_iteration(self, filled_buf):
        from torch.utils.data import DataLoader

        loader = DataLoader(filled_buf, batch_size=4, shuffle=False)
        batches = list(loader)
        # Coverage: every clip seen exactly once across the epoch.
        total_seen = sum(b['pixels'].shape[0] for b in batches)
        assert total_seen == len(filled_buf)
        # Each batch entry has the right per-item shape.
        assert batches[0]['pixels'].shape[1:] == (4, 8, 8, 3)
        assert batches[0]['action'].shape[1:] == (4, 2)

    def test_load_slice_returns_episode_window(self, filled_buf):
        out = filled_buf._load_slice(0, 0, 5)
        assert out['pixels'].shape == (5, 8, 8, 3)
        assert out['action'].shape == (5, 2)

    def test_load_slice_validates_bounds(self, filled_buf):
        with pytest.raises(IndexError):
            filled_buf._load_slice(99, 0, 5)
        with pytest.raises(IndexError):
            filled_buf._load_slice(0, 0, 999)


class TestClipCache:
    def test_cache_built_lazily(self):
        buf = ReplayBuffer(max_steps=50, history_len=2)
        buf.write_episode(_episode(10, seed=0))
        # Empty before first read.
        assert buf._clip_starts is None
        _ = buf[0]
        assert buf._clip_starts is not None

    def test_reads_reuse_cache_object(self, filled_buf):
        _ = filled_buf[0]
        cache = filled_buf._clip_starts
        for _ in range(5):
            _ = filled_buf[0]
            _ = filled_buf.num_valid_ends()
        assert filled_buf._clip_starts is cache

    def test_write_invalidates(self, filled_buf):
        _ = filled_buf[0]
        assert filled_buf._clip_starts is not None
        filled_buf.write_episode(_episode(10, seed=42))
        assert filled_buf._clip_starts is None

    def test_eviction_invalidates(self):
        buf = ReplayBuffer(max_steps=30, history_len=2)
        buf.write_episode(_episode(20, seed=0))
        _ = buf[0]
        cached = buf._clip_starts
        # Force eviction.
        buf.write_episode(_episode(15, seed=1))
        assert buf._clip_starts is None
        _ = buf[0]
        assert buf._clip_starts is not cached

    def test_clear_invalidates(self, filled_buf):
        _ = filled_buf[0]
        assert filled_buf._clip_starts is not None
        filled_buf.clear()
        assert filled_buf._clip_starts is None
        assert filled_buf.num_episodes == 0

    def test_different_span_re_keys(self):
        buf = ReplayBuffer(max_steps=200, history_len=4)
        for s in range(3):
            buf.write_episode(_episode(20, seed=s))
        _ = buf.num_valid_ends(4)
        cache_h4 = buf._clip_starts
        _ = buf.num_valid_ends(2)
        assert buf._clip_starts is not cache_h4

    def test_cache_values_are_correct(self, filled_buf):
        # Three episodes of length 20, history_len=4, frameskip=1 → span=4.
        # Valid clips per episode = 17 → starts = [0, 17, 34, 51, 68].
        # filled_buf has 4 episodes of length 20.
        starts = filled_buf._get_clip_starts(4)
        np.testing.assert_array_equal(starts, [0, 17, 34, 51, 68])


class TestRingStorage:
    def test_data_correct_after_wraparound(self):
        # Force the head to wrap by evicting an episode that started at 0.
        buf = ReplayBuffer(max_steps=20, history_len=1)
        ep0 = _episode(8, seed=10)
        ep1 = _episode(8, seed=11)
        ep2 = _episode(8, seed=12)  # write at position 16, wraps to 4
        buf.write_episode(ep0)
        buf.write_episode(ep1)
        buf.write_episode(ep2)

        assert buf.num_episodes == 2
        # ep1 starts at position 8 (untouched); first stored clip is ep1[0].
        np.testing.assert_array_equal(buf[0]['pixels'][0], ep1['pixels'][0])
        # ep2 wraps; last clip is its last step.
        np.testing.assert_array_equal(
            buf[len(buf) - 1]['pixels'][0], ep2['pixels'][-1]
        )

    def test_clip_does_not_cross_episode_boundary(self):
        # Clip with history_len=3 spanning positions [N-1..N+1] would cross
        # an episode boundary — must never happen.
        buf = ReplayBuffer(max_steps=50, history_len=3)
        ep_a = _episode(10, seed=20)
        ep_b = _episode(10, seed=21)
        buf.write_episode(ep_a)
        buf.write_episode(ep_b)

        # Iterate every clip: each must fall entirely within one episode.
        for i in range(len(buf)):
            clip = buf[i]
            # Clip's pixels[0] must equal a row of ep_a or ep_b but not span.
            in_a = any(
                np.array_equal(clip['pixels'][0], ep_a['pixels'][j])
                for j in range(10)
            )
            in_b = any(
                np.array_equal(clip['pixels'][0], ep_b['pixels'][j])
                for j in range(10)
            )
            assert in_a or in_b


class TestProperties:
    def test_lengths_matches_writes(self, filled_buf):
        np.testing.assert_array_equal(
            filled_buf.lengths, np.array([20, 20, 20, 20], dtype=np.int32)
        )

    def test_offsets_are_logical_episode_starts(self, filled_buf):
        np.testing.assert_array_equal(
            filled_buf.offsets, np.array([0, 20, 40, 60], dtype=np.int64)
        )

    def test_lengths_empty_buffer(self):
        buf = ReplayBuffer(max_steps=10)
        assert buf.lengths.shape == (0,)
        assert buf.offsets.shape == (0,)


class TestDump:
    def test_dump_to_folder_and_reload(self, tmp_path):
        buf = ReplayBuffer(max_steps=100, history_len=1)
        for s in range(3):
            buf.write_episode(_episode(10, seed=s))

        out = tmp_path / 'dumped'
        buf.dump(out, format='folder')

        ds = FolderDataset(path=out, num_steps=4)
        np.testing.assert_array_equal(ds.lengths, [10, 10, 10])

    def test_dump_round_trips_data(self, tmp_path):
        buf = ReplayBuffer(max_steps=50, history_len=1)
        ep = _episode(10, seed=999)
        buf.write_episode(ep)

        out = tmp_path / 'rt'
        buf.dump(out, format='folder')
        ds = FolderDataset(path=out, num_steps=10)

        # Pull the whole episode and compare.
        item = ds[0]
        # Tabular columns are stored exactly as .npz — round-trip is byte-exact.
        np.testing.assert_array_equal(item['proprio'].numpy(), ep['proprio'])
        np.testing.assert_array_equal(item['action'].numpy(), ep['action'])
        np.testing.assert_array_equal(item['reward'].numpy(), ep['reward'])
        # Pixels go through JPEG (lossy); only check shape/dtype survived.
        pixels = item['pixels'].permute(0, 2, 3, 1).numpy()
        assert pixels.shape == ep['pixels'].shape
        assert pixels.dtype == ep['pixels'].dtype

    def test_dump_to_hdf5(self, tmp_path):
        buf = ReplayBuffer(max_steps=50, history_len=1)
        for s in range(2):
            buf.write_episode(_episode(8, seed=s))

        out = tmp_path / 'rb.h5'
        buf.dump(out, format='hdf5')
        ds = HDF5Dataset(path=out, num_steps=4)
        assert len(ds.lengths) == 2

    def test_dump_overwrites_by_default(self, tmp_path):
        out = tmp_path / 'overwrite_target'
        buf = ReplayBuffer(max_steps=50, history_len=1)
        buf.write_episode(_episode(10, seed=0))
        buf.dump(out, format='folder')

        # Dump again with a different episode count — overwrite is the default.
        buf.clear()
        buf.write_episode(_episode(5, seed=1))
        buf.dump(out, format='folder')

        ds = FolderDataset(path=out, num_steps=1)
        np.testing.assert_array_equal(ds.lengths, [5])

    def test_dump_after_clear_writes_nothing(self, tmp_path):
        buf = ReplayBuffer(max_steps=50, history_len=1)
        buf.write_episode(_episode(10, seed=0))
        buf.clear()

        out = tmp_path / 'empty'
        buf.dump(out, format='folder')
        ds = FolderDataset(path=out, num_steps=1)
        assert len(ds.lengths) == 0

    def test_episodes_iterator_yields_writer_format(self, filled_buf):
        eps = list(filled_buf.episodes())
        assert len(eps) == filled_buf.num_episodes
        # World.collect / writers expect dict[col, list[step_arr]].
        for ep in eps:
            assert isinstance(ep['action'], list)
            assert all(isinstance(x, np.ndarray) for x in ep['action'])


class TestClear:
    def test_clear_resets_state(self, filled_buf):
        filled_buf.clear()
        assert filled_buf.num_episodes == 0
        assert filled_buf.num_steps_stored == 0
        assert len(filled_buf) == 0

    def test_clear_keeps_column_allocations(self, filled_buf):
        cols_before = {k: id(v) for k, v in filled_buf._cols.items()}
        filled_buf.clear()
        cols_after = {k: id(v) for k, v in filled_buf._cols.items()}
        # Same arrays — clear() reuses them, doesn't reallocate.
        assert cols_before == cols_after

    def test_can_refill_after_clear(self, filled_buf):
        filled_buf.clear()
        filled_buf.write_episode(_episode(10, seed=42))
        assert filled_buf.num_episodes == 1
        assert filled_buf.num_steps_stored == 10
