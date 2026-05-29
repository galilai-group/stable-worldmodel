"""Tests for the lance_video format: MP4-blob round-trip, detection, parity.

The image columns are decoded by torchcodec and the writer needs imageio's
ffmpeg muxer; skip the whole module if either is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('imageio')

# Importing torchcodec eagerly loads libtorchcodec, which raises RuntimeError
# (not ImportError) when the environment has no matching FFmpeg shared
# libraries — common on CI runners. importorskip only catches ImportError, so
# guard the import and skip the whole module on any failure.
try:
    from torchcodec.decoders import VideoDecoder  # noqa: F401
except Exception as exc:  # noqa: BLE001
    pytest.skip(f'torchcodec unavailable ({exc})', allow_module_level=True)

from stable_worldmodel.data import (  # noqa: E402
    LanceDataset,
    LanceVideoDataset,
    LanceVideoWriter,
    LanceWriter,
    detect_format,
    get_format,
    list_formats,
)


def _episodes(ep_lengths=(8, 6, 10), h=32, w=32):
    rng = np.random.default_rng(0)
    for ep_len in ep_lengths:
        yield {
            # Smoothly varying frames so the h264 codec stays near-lossless.
            'pixels': [
                np.full((h, w, 3), 10 + 3 * t, dtype=np.uint8)
                for t in range(ep_len)
            ],
            'action': [
                rng.standard_normal(2).astype(np.float32)
                for _ in range(ep_len)
            ],
            'proprio': [
                rng.standard_normal(3).astype(np.float32)
                for _ in range(ep_len)
            ],
        }


def _write(out: Path) -> list[dict]:
    eps = [{k: list(v) for k, v in ep.items()} for ep in _episodes()]
    with LanceVideoWriter(out) as w:
        w.write_episodes([dict(ep) for ep in eps])
    return eps


def test_lance_video_registered():
    assert 'lance_video' in list_formats()


def test_two_table_layout(tmp_path):
    out = tmp_path / 'set'
    _write(out)
    assert (out / 'set.lance').is_dir()
    assert (out / 'set_videos.lance').is_dir()


def test_detect_prefers_video_over_lance(tmp_path):
    out = tmp_path / 'set'
    _write(out)
    # A `*_videos.lance` sibling must route the dir to lance_video, not lance.
    assert detect_format(out).name == 'lance_video'


def test_roundtrip_shapes_and_alignment(tmp_path):
    out = tmp_path / 'set'
    _write(out)
    ds = LanceVideoDataset(
        out,
        num_steps=3,
        frameskip=2,
        keys_to_load=['pixels', 'action', 'proprio'],
    )
    assert len(ds) > 0
    s = ds[0]
    assert tuple(s['pixels'].shape) == (3, 3, 32, 32)
    assert s['pixels'].dtype.is_floating_point is False  # uint8
    assert s['proprio'].shape[0] == 3
    # Episode 0 frames 0,2,4 encode levels ~10,16,22 — monotone, near-exact.
    levels = [float(s['pixels'][k].float().mean()) for k in range(3)]
    assert levels[0] < levels[1] < levels[2]
    assert abs(levels[0] - 10) < 4 and abs(levels[2] - 22) < 4


def test_tabular_parity_with_frame_lance(tmp_path):
    """Tabular columns must round-trip bit-identically to the frame format."""
    eps = _write(tmp_path / 'vid')
    with LanceWriter(tmp_path / 'frm' / 'frm.lance') as w:
        w.write_episodes([dict(ep) for ep in eps])

    common = dict(
        num_steps=3,
        frameskip=2,
        keys_to_load=['pixels', 'action', 'proprio'],
    )
    vid = LanceVideoDataset(tmp_path / 'vid', **common)
    frm = LanceDataset(tmp_path / 'frm', image_columns=['pixels'], **common)
    assert len(vid) == len(frm)
    for i in (0, len(vid) // 2, len(vid) - 1):
        a, b = frm[i], vid[i]
        assert np.allclose(a['action'].numpy(), b['action'].numpy())
        assert np.allclose(a['proprio'].numpy(), b['proprio'].numpy())
        assert a['pixels'].shape == b['pixels'].shape


def test_batch_parity_with_frame_lance(tmp_path):
    """`__getitems__` must match the frame format in keys, shapes and dtypes
    for an arbitrary batch — incl. cross-episode, out-of-order, duplicate
    indices (exercises the dedupe/gather + per-episode decode grouping)."""
    eps = _write(tmp_path / 'vid')
    with LanceWriter(tmp_path / 'frm' / 'frm.lance') as w:
        w.write_episodes([dict(ep) for ep in eps])

    common = dict(
        num_steps=3,
        frameskip=2,
        keys_to_load=['pixels', 'action', 'proprio'],
    )
    vid = LanceVideoDataset(tmp_path / 'vid', **common)
    frm = LanceDataset(tmp_path / 'frm', image_columns=['pixels'], **common)
    assert len(vid) == len(frm)

    n = len(vid)
    batch = [0, n - 1, n // 2, n // 3, n // 2, 1]  # out-of-order + duplicate
    vb = vid.__getitems__(batch)
    fb = frm.__getitems__(batch)
    assert len(vb) == len(fb) == len(batch)
    for vs, fs in zip(vb, fb):
        assert set(vs) == set(fs)
        for k in fs:
            assert vs[k].shape == fs[k].shape, k
            assert vs[k].dtype == fs[k].dtype, k
        assert np.allclose(
            vs['action'].numpy(), fs['action'].numpy(), equal_nan=True
        )
        assert np.allclose(
            vs['proprio'].numpy(), fs['proprio'].numpy(), equal_nan=True
        )


def test_batch_matches_per_item(tmp_path):
    """Batched `__getitems__` must agree with single-item `__getitem__`."""
    _write(tmp_path / 'vid')
    ds = LanceVideoDataset(
        tmp_path / 'vid',
        num_steps=3,
        frameskip=2,
        keys_to_load=['pixels', 'action', 'proprio'],
    )
    idxs = [2, 0, len(ds) - 1, 2]
    batched = ds.__getitems__(idxs)
    for pos, i in enumerate(idxs):
        single = ds[i]
        assert set(batched[pos]) == set(single)
        for k in single:
            assert batched[pos][k].shape == single[k].shape, k
            assert np.array_equal(
                batched[pos][k].numpy(), single[k].numpy()
            ), k


def test_select_video_keys(tmp_path):
    out = tmp_path / 'set'
    _write(out)
    ds = LanceVideoDataset(
        out, num_steps=1, frameskip=1, keys_to_load=['action', 'proprio']
    )
    sample = ds[0]
    assert 'pixels' not in sample
    assert 'action' in sample and 'proprio' in sample


def test_open_via_format_registry(tmp_path):
    out = tmp_path / 'set'
    _write(out)
    fmt = get_format('lance_video')
    ds = fmt.open_reader(
        out, num_steps=1, frameskip=1, keys_to_load=['pixels']
    )
    assert tuple(ds[0]['pixels'].shape) == (1, 3, 32, 32)
