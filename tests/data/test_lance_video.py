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

import lancedb  # noqa: E402
import pyarrow as pa  # noqa: E402

from stable_worldmodel.data import (  # noqa: E402
    EPISODE_DATA_KEY,
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


def test_load_episode_returns_full_video(tmp_path):
    """`load_episode` must decode every frame of the episode, aligned with the
    tabular columns. This is the path `convert`/`merge` rely on: opened with
    default `num_steps=1`, a full-episode read must not truncate the video to a
    single frame."""
    out = tmp_path / 'set'
    eps = _write(out)
    ds = LanceVideoDataset(out, keys_to_load=['pixels', 'action', 'proprio'])
    for ep_idx, ep in enumerate(eps):
        ep_len = len(ep['pixels'])
        sample = ds.load_episode(ep_idx)
        assert sample['pixels'].shape[0] == ep_len, (
            f'episode {ep_idx}: got {sample["pixels"].shape[0]} video frames, '
            f'expected {ep_len}'
        )
        assert sample['action'].shape[0] == ep_len
        assert sample['proprio'].shape[0] == ep_len


def test_select_video_keys(tmp_path):
    out = tmp_path / 'set'
    _write(out)
    ds = LanceVideoDataset(
        out, num_steps=1, frameskip=1, keys_to_load=['action', 'proprio']
    )
    sample = ds[0]
    assert 'pixels' not in sample
    assert 'action' in sample and 'proprio' in sample


def test_append_across_writer_sessions(tmp_path):
    """Reopening the writer to append more episodes must not crash.

    Data collection writes in chunks: each chunk opens a fresh
    ``LanceVideoWriter`` in the default ``mode='append'``. The first chunk
    creates the table (running ``_init_schema``, which builds ``_rename_map``);
    every later chunk hits ``_load_existing_state`` instead, which recovers the
    schema from disk but used to leave ``_rename_map`` empty — so
    ``_frames_batch`` raised ``StopIteration`` on the second chunk. Splitting
    one logical dataset across two writer sessions reproduces that path.
    """
    out = tmp_path / 'set'
    eps = [{k: list(v) for k, v in ep.items()} for ep in _episodes()]
    first, rest = eps[:1], eps[1:]

    with LanceVideoWriter(out) as w:
        w.write_episodes([dict(ep) for ep in first])
    # Second session appends — previously crashed with StopIteration here.
    with LanceVideoWriter(out) as w:
        w.write_episodes([dict(ep) for ep in rest])

    ds = LanceVideoDataset(out, keys_to_load=['pixels', 'action', 'proprio'])
    # Every episode survived the append, with its video rows intact (an empty
    # rename map would also have silently dropped the appended video frames).
    for ep_idx, ep in enumerate(eps):
        ep_len = len(ep['pixels'])
        sample = ds.load_episode(ep_idx)
        assert sample['pixels'].shape[0] == ep_len, (
            f'episode {ep_idx}: got {sample["pixels"].shape[0]} video frames, '
            f'expected {ep_len}'
        )
        assert sample['action'].shape[0] == ep_len


def test_open_via_format_registry(tmp_path):
    out = tmp_path / 'set'
    _write(out)
    fmt = get_format('lance_video')
    ds = fmt.open_reader(
        out, num_steps=1, frameskip=1, keys_to_load=['pixels']
    )
    assert tuple(ds[0]['pixels'].shape) == (1, 3, 32, 32)


def test_ranged_decode_parity_with_full_bytes(tmp_path):
    """Decoding through the ranged blob reader must be bit-identical to
    decoding from fully materialized blob bytes (the previous behavior)."""
    import torch

    import lance

    _write(tmp_path / 'd')
    ds = LanceVideoDataset(tmp_path / 'd', num_steps=4)
    item = ds[0]

    videos_dir = next((tmp_path / 'd').glob('*_videos.lance'))
    v = lance.dataset(str(videos_dir))
    blob = v.take_blobs(blob_column='video_bytes', indices=[0])[0]
    data = blob.readall()
    blob.close()
    ref = VideoDecoder(data, seek_mode='approximate')
    expected = ref.get_frames_at(indices=[0, 1, 2, 3]).data
    assert torch.equal(item['pixels'], expected)


def test_ranged_decode_reads_partial_blob(tmp_path, monkeypatch):
    """A small window from a long episode must not download the whole MP4."""
    import stable_worldmodel.data.formats.lance_video as lv

    rng = np.random.default_rng(1)
    ep = {
        'pixels': [
            rng.integers(0, 255, (64, 64, 3)).astype(np.uint8)
            for _ in range(300)
        ],
        'action': [
            rng.standard_normal(2).astype(np.float32) for _ in range(300)
        ],
    }
    with LanceVideoWriter(tmp_path / 'd') as w:
        w.write_episodes([ep])

    fetched = []
    orig = lv._SeekableBlob.readinto

    def counting(self, b):
        n = orig(self, b)
        fetched.append(n)
        return n

    monkeypatch.setattr(lv._SeekableBlob, 'readinto', counting)

    ds = LanceVideoDataset(tmp_path / 'd', num_steps=4, blob_buffer_size=4096)
    item = ds[0]
    assert item['pixels'].shape[0] == 4

    import lance

    videos_dir = next((tmp_path / 'd').glob('*_videos.lance'))
    v = lance.dataset(str(videos_dir))
    blob = v.take_blobs(blob_column='video_bytes', indices=[0])[0]
    blob.seek(0, 2)
    blob_size = blob.tell()
    blob.close()
    assert sum(fetched) < blob_size, (
        f'ranged reader fetched {sum(fetched)} of {blob_size} bytes'
    )


def _eps_with_data(ep_lengths=(8, 6, 10)):
    eps = [{k: list(v) for k, v in ep.items()} for ep in _episodes(ep_lengths)]
    for i, ep in enumerate(eps):
        ep[EPISODE_DATA_KEY] = {
            'model_xml': f'<scene {i}/>',
            'states': np.arange(3 + i, dtype=np.float32),
        }
    return eps


def test_episode_data_roundtrip(tmp_path):
    """Episode data written through the lance_video writer lands in the
    ``<table>_episodes`` side table and reads back via the inherited API."""
    out = tmp_path / 'set'
    with LanceVideoWriter(out) as w:
        w.write_episodes(_eps_with_data())

    assert (out / 'set_episodes.lance').is_dir()
    ds = LanceVideoDataset(out)
    assert ds.lengths.tolist() == [8, 6, 10]
    assert ds.episode_column_names == ['model_xml', 'states']
    data = ds.get_episode_data()
    assert data['model_xml'] == ['<scene 0/>', '<scene 1/>', '<scene 2/>']
    assert [v.shape for v in data['states']] == [(3,), (4,), (5,)]
    assert ds.get_episode_data([2, 0])['model_xml'] == [
        '<scene 2/>',
        '<scene 0/>',
    ]


def test_episode_data_append_across_sessions(tmp_path):
    out = tmp_path / 'set'
    eps = _eps_with_data()
    with LanceVideoWriter(out) as w:
        w.write_episodes(eps[:1])
    with LanceVideoWriter(out) as w:  # mode='append' default
        w.write_episodes(eps[1:])

    ds = LanceVideoDataset(out)
    assert ds.lengths.tolist() == [8, 6, 10]
    assert ds.get_episode_data()['model_xml'] == [
        '<scene 0/>',
        '<scene 1/>',
        '<scene 2/>',
    ]


def test_episode_data_key_mismatch_raises(tmp_path):
    out = tmp_path / 'set'
    eps = _eps_with_data(ep_lengths=(4, 4))
    eps[1][EPISODE_DATA_KEY] = {'other_key': 'x'}
    with pytest.raises(ValueError, match='episode-data key mismatch'):
        with LanceVideoWriter(out) as w:
            w.write_episodes(eps)


def test_episode_data_overwrite_drops_side_table(tmp_path):
    out = tmp_path / 'set'
    with LanceVideoWriter(out) as w:
        w.write_episodes(_eps_with_data(ep_lengths=(4,)))
    plain = [{k: list(v) for k, v in ep.items()} for ep in _episodes((6,))]
    with LanceVideoWriter(out, mode='overwrite') as w:
        w.write_episodes(plain)

    assert not (out / 'set_episodes.lance').exists()
    ds = LanceVideoDataset(out)
    assert ds.lengths.tolist() == [6]
    assert ds.episode_column_names == []


def test_convert_lance_to_lance_video_carries_episode_data(tmp_path):
    from stable_worldmodel.data import EPISODE_DATA_KEY as EDK, convert

    src = tmp_path / 'src.lance'
    rng = np.random.default_rng(0)
    with LanceWriter(src) as w:
        for i in range(2):
            w.write_episode(
                {
                    'pixels': [
                        np.full((32, 32, 3), 10 + 3 * t, dtype=np.uint8)
                        for t in range(5)
                    ],
                    'action': [
                        rng.standard_normal(2).astype(np.float32)
                        for _ in range(5)
                    ],
                    EDK: {'model_xml': f'<scene {i}/>'},
                }
            )

    dest = tmp_path / 'vidset'
    convert(str(src), str(dest), dest_format='lance_video', progress=False)

    ds = LanceVideoDataset(dest)
    assert ds.lengths.tolist() == [5, 5]
    assert ds.get_episode_data()['model_xml'] == ['<scene 0/>', '<scene 1/>']


def test_resolver_ignores_episodes_sibling(tmp_path):
    """A ``<table>_episodes.lance`` sibling must not confuse frames-table
    resolution, and the inherited reader picks up the episode data."""
    out = tmp_path / 'set'
    _write(out)

    db = lancedb.connect(str(out))
    db.create_table(
        'set_episodes',
        data=pa.table(
            {
                'episode_idx': pa.array([0, 1, 2], type=pa.int32()),
                'model_xml': pa.array(['<a/>', '<b/>', '<c/>']),
            }
        ),
    )

    assert detect_format(out).name == 'lance_video'
    ds = LanceVideoDataset(out)
    assert ds.lengths.tolist() == [8, 6, 10]
    assert ds.episode_column_names == ['model_xml']
    assert ds.get_episode_data([2, 0])['model_xml'] == ['<c/>', '<a/>']
