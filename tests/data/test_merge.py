"""Tests for merging/concatenating datasets into a single dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stable_worldmodel.data import (
    LanceDataset,
    LanceWriter,
    detect_format,
    merge,
)


def _write_shard(
    out: Path,
    *,
    ep_lengths: tuple[int, ...],
    marker: int,
    cols: tuple[str, ...] = ('pixels', 'action', 'proprio'),
) -> None:
    """Write a small lance shard. Each step's ``action`` is
    ``[marker + ep_local_idx, step_idx]`` so merged content can be traced back
    to its source shard and episode."""
    rng = np.random.default_rng(marker)
    with LanceWriter(out) as w:
        for ep_local, ep_len in enumerate(ep_lengths):
            ep: dict[str, list] = {}
            if 'pixels' in cols:
                ep['pixels'] = [
                    rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
                    for _ in range(ep_len)
                ]
            if 'action' in cols:
                ep['action'] = [
                    np.array([marker + ep_local, s], dtype=np.float32)
                    for s in range(ep_len)
                ]
            if 'proprio' in cols:
                ep['proprio'] = [
                    rng.standard_normal(3).astype(np.float32)
                    for _ in range(ep_len)
                ]
            w.write_episode(ep)


def test_merge_concatenates_episodes_contiguously(tmp_path):
    a = tmp_path / 'a.lance'
    b = tmp_path / 'b.lance'
    _write_shard(a, ep_lengths=(3, 4), marker=100)
    _write_shard(b, ep_lengths=(5,), marker=200)

    out = tmp_path / 'merged.lance'
    merge([str(a), str(b)], str(out), dest_format='lance', progress=False)

    ds = LanceDataset(path=out)
    # Episode lengths are the concatenation of both shards, in order, and the
    # contiguous offsets [0, 3, 7] confirm episode_idx was renumbered gap-free
    # across the shard boundary (the reader rejects a non-contiguous
    # episode_idx, so a successful read is itself the collision check).
    assert ds.lengths.tolist() == [3, 4, 5]
    assert ds.offsets.tolist() == [0, 3, 7]
    assert len(ds.lengths) == 3
    assert sorted(ds.column_names) == ['action', 'pixels', 'proprio']


def test_merge_preserves_source_order_and_content(tmp_path):
    a = tmp_path / 'a.lance'
    b = tmp_path / 'b.lance'
    _write_shard(a, ep_lengths=(3, 4), marker=100)  # markers 100, 101
    _write_shard(b, ep_lengths=(5,), marker=200)  # marker 200

    out = tmp_path / 'merged.lance'
    merge([str(a), str(b)], str(out), dest_format='lance', progress=False)

    ds = LanceDataset(path=out)
    # Merged episode 2 (the third) is shard B's first episode: marker 200.
    ep2 = np.asarray(ds.load_episode(2)['action'])
    assert ep2.shape[0] == 5
    assert int(ep2[0, 0]) == 200
    # Merged episode 1 is shard A's second episode: marker 101.
    ep1 = np.asarray(ds.load_episode(1)['action'])
    assert int(ep1[0, 0]) == 101


def test_merge_column_mismatch_raises(tmp_path):
    a = tmp_path / 'a.lance'
    bad = tmp_path / 'bad.lance'
    _write_shard(a, ep_lengths=(3,), marker=100)
    _write_shard(bad, ep_lengths=(2,), marker=200, cols=('pixels', 'action'))

    with pytest.raises(ValueError, match='column mismatch'):
        merge(
            [str(a), str(bad)],
            str(tmp_path / 'm.lance'),
            dest_format='lance',
            progress=False,
        )


def test_merge_requires_two_sources(tmp_path):
    a = tmp_path / 'a.lance'
    _write_shard(a, ep_lengths=(3,), marker=100)
    with pytest.raises(ValueError, match='at least two'):
        merge([str(a)], str(tmp_path / 'm.lance'), dest_format='lance')


def test_merge_mode_error_refuses_existing(tmp_path):
    a = tmp_path / 'a.lance'
    b = tmp_path / 'b.lance'
    _write_shard(a, ep_lengths=(3,), marker=100)
    _write_shard(b, ep_lengths=(4,), marker=200)
    out = tmp_path / 'merged.lance'

    merge([str(a), str(b)], str(out), dest_format='lance', progress=False)

    # A second merge with the default mode='error' must refuse to clobber.
    with pytest.raises(FileExistsError):
        merge([str(a), str(b)], str(out), dest_format='lance', progress=False)

    # overwrite starts fresh — same result, no error.
    merge(
        [str(a), str(b)],
        str(out),
        dest_format='lance',
        mode='overwrite',
        progress=False,
    )
    assert LanceDataset(path=out).lengths.tolist() == [3, 4]


def test_merge_append_extends_existing(tmp_path):
    a = tmp_path / 'a.lance'
    b = tmp_path / 'b.lance'
    _write_shard(a, ep_lengths=(3, 4), marker=100)
    _write_shard(b, ep_lengths=(5,), marker=200)
    out = tmp_path / 'merged.lance'

    merge([str(a), str(b)], str(out), dest_format='lance', progress=False)
    # Appending the same sources again extends the dataset contiguously.
    merge(
        [str(a), str(b)],
        str(out),
        dest_format='lance',
        mode='append',
        progress=False,
    )
    ds = LanceDataset(path=out)
    assert ds.lengths.tolist() == [3, 4, 5, 3, 4, 5]


def test_merge_transcodes_to_another_format(tmp_path):
    """Merge can change format in one pass (lance shards → folder)."""
    a = tmp_path / 'a.lance'
    b = tmp_path / 'b.lance'
    _write_shard(a, ep_lengths=(3, 4), marker=100)
    _write_shard(b, ep_lengths=(5,), marker=200)

    out = tmp_path / 'merged_folder'
    merge([str(a), str(b)], str(out), dest_format='folder', progress=False)

    assert (out / 'ep_len.npz').exists()
    ep_len = np.load(out / 'ep_len.npz')['arr_0']
    assert ep_len.tolist() == [3, 4, 5]
    assert detect_format(out) is not None
