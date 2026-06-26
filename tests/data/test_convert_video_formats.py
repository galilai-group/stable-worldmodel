"""Round-trip ``convert`` tests across the compressed-image formats.

``swm convert`` (:func:`stable_worldmodel.data.convert`) reads each episode
through the source format's reader and re-writes it through the destination
writer. The three formats that store image columns as *compressed* media —

  * ``video``        : one MP4 per episode on disk (FolderDataset layout),
  * ``lance``        : per-frame JPEG blobs in a Lance ``binary`` column,
  * ``lance_video``  : per-episode MP4 blobs in a Lance ``large_binary`` column,

exercise the trickiest part of the conversion adapter
(:func:`stable_worldmodel.data.utils._episode_to_step_lists`): the
``(N, C, H, W) -> (N, H, W, C)`` transpose plus a full image encode/decode
round-trip. This module covers every directed pair among the three, *including*
each format to itself (e.g. video -> video), since an identical-format
round-trip is both a real use and where a reader/writer column-name mismatch
surfaces. The simpler hdf5/folder pairs already live in ``test_format.py`` /
``test_lance.py``.

The image column is named ``pixels`` (not ``video``) on purpose: ``convert``
cannot forward reader kwargs to the *source*, so the ``VideoDataset`` reader
must auto-detect its video column from the on-disk layout rather than assuming
a fixed name. Using a non-default name here guards that — a reader that
hardcoded ``video_keys=['video']`` would drop the column on read.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# Both the video and lance_video writers encode MP4 via imageio's ffmpeg muxer.
pytest.importorskip('imageio')

from stable_worldmodel.data import (  # noqa: E402
    LanceVideoWriter,
    LanceWriter,
    VideoWriter,
    convert,
    list_formats,
    load_dataset,
)

IMG_KEY = 'pixels'
EP_LENGTHS = (6, 8)
HW = 32
COLUMNS = {IMG_KEY, 'action', 'proprio'}

def _pair(src: str, dst: str):
    return pytest.param(src, dst, id=f'{src}->{dst}')


# Every directed pair among the three compressed-image formats, including each
# format to itself — an identical-format round-trip is a real use and is where
# a reader/writer column-name mismatch (e.g. the video reader's old hardcoded
# 'video' key) would bite.
_IMG_FORMATS = ('video', 'lance', 'lance_video')
PAIRS = [_pair(src, dst) for src in _IMG_FORMATS for dst in _IMG_FORMATS]


def _episodes() -> list[dict]:
    """Two episodes with flat (per-step constant) colour frames.

    Flat fields keep both h264 and JPEG near-lossless, and the colour ramps
    with the step (and episode) index so frames stay distinguishable rather
    than collapsing to one value the codec could trivially reproduce.
    """
    rng = np.random.default_rng(0)
    eps = []
    for ep_i, ep_len in enumerate(EP_LENGTHS):
        eps.append(
            {
                IMG_KEY: [
                    np.full(
                        (HW, HW, 3), 10 + 5 * ep_i + 6 * t, dtype=np.uint8
                    )
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
        )
    return eps


def _copy(eps: list[dict]) -> list[dict]:
    """Fresh per-write copy — writers may consume their input iterables."""
    return [{k: list(v) for k, v in ep.items()} for ep in eps]


def _require_backends(*fmts: str) -> None:
    """Skip when a format's optional decode backend is unavailable on CI."""
    if 'lance_video' in fmts:
        try:
            import torchcodec.decoders  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            # torchcodec raises RuntimeError (not ImportError) when FFmpeg
            # shared libs are missing, so importorskip would not catch it.
            pytest.skip(f'torchcodec unavailable ({exc})')
    if 'video' in fmts:
        if (
            importlib.util.find_spec('decord') is None
            and importlib.util.find_spec('av') is None
        ):
            pytest.skip('no video decode backend (decord / av) available')


def _write_source(fmt: str, root: Path, eps: list[dict]) -> Path:
    """Write ``eps`` to disk with ``fmt``'s writer; return the loadable path."""
    if fmt == 'video':
        p = root / 'video_src'
        with VideoWriter(p, fps=10) as w:
            w.write_episodes(_copy(eps))
        return p
    if fmt == 'lance':
        p = root / 'lance_src.lance'
        with LanceWriter(p) as w:
            w.write_episodes(_copy(eps))
        return p
    if fmt == 'lance_video':
        p = root / 'lance_video_src'
        with LanceVideoWriter(p, fps=10) as w:
            w.write_episodes(_copy(eps))
        return p
    raise ValueError(f'unknown format {fmt!r}')


def _dest_path(fmt: str, root: Path) -> Path:
    return {
        'video': root / 'video_dst',
        'lance': root / 'lance_dst.lance',
        'lance_video': root / 'lance_video_dst',
    }[fmt]


def _to_np(x) -> np.ndarray:
    return x.numpy() if hasattr(x, 'numpy') else np.asarray(x)


def _read_episodes(path: Path, fmt: str, n: int) -> list[dict]:
    ds = load_dataset(str(path), format=fmt)
    assert len(ds.lengths) == n, (
        f'{fmt}: expected {n} episodes, got {len(ds.lengths)}'
    )
    return [ds.load_episode(i) for i in range(n)]


@pytest.mark.parametrize('src_fmt,dst_fmt', PAIRS)
def test_convert_roundtrip(tmp_path, src_fmt, dst_fmt):
    """convert(src -> dst) preserves episode structure, tabular data, and images.

    Tabular columns (float32) are stored losslessly by every format, so they
    must match within float tolerance. Image columns pass through a lossy
    codec on at least one leg, so they are compared on shape/dtype exactly and
    on per-frame mean within a few intensity levels (the frames are flat
    fields, which both h264 and JPEG reproduce closely).
    """
    _require_backends(src_fmt, dst_fmt)
    assert src_fmt in list_formats() and dst_fmt in list_formats()

    eps = _episodes()
    src_path = _write_source(src_fmt, tmp_path, eps)
    dst_path = _dest_path(dst_fmt, tmp_path)

    convert(
        str(src_path),
        str(dst_path),
        source_format=src_fmt,
        dest_format=dst_fmt,
        progress=False,
    )

    src_eps = _read_episodes(src_path, src_fmt, len(eps))
    dst_eps = _read_episodes(dst_path, dst_fmt, len(eps))

    for i, (a, b) in enumerate(zip(src_eps, dst_eps)):
        ep_len = EP_LENGTHS[i]
        assert set(a) == COLUMNS, f'ep {i}: source columns {set(a)}'
        assert set(b) == COLUMNS, f'ep {i}: dest columns {set(b)}'

        # Tabular columns survive losslessly through every writer.
        for col in ('action', 'proprio'):
            xa = _to_np(a[col]).reshape(ep_len, -1)
            xb = _to_np(b[col]).reshape(ep_len, -1)
            np.testing.assert_allclose(
                xa, xb, atol=1e-4, err_msg=f'ep {i}: column {col!r} mismatch'
            )

        # Image column: identical (T, C, H, W) uint8 layout, near-equal pixels.
        ia = _to_np(a[IMG_KEY])
        ib = _to_np(b[IMG_KEY])
        assert ia.shape == (ep_len, 3, HW, HW), (
            f'ep {i}: source image shape {ia.shape}'
        )
        assert ib.shape == (ep_len, 3, HW, HW), (
            f'ep {i}: dest image shape {ib.shape}'
        )
        assert ia.dtype == np.uint8 and ib.dtype == np.uint8

        per_frame_mean_diff = np.abs(
            ia.astype(np.int16).mean(axis=(1, 2, 3))
            - ib.astype(np.int16).mean(axis=(1, 2, 3))
        )
        assert per_frame_mean_diff.max() < 8.0, (
            f'ep {i}: per-frame mean drift {per_frame_mean_diff} exceeds '
            f'tolerance for {src_fmt} -> {dst_fmt}'
        )
