"""Parse MP4 sample tables into a GOP-level byte-range index.

The index answers "which byte ranges of this file do I need to decode
frames [a, b)?" without touching media data: the container header
(``ftyp``+``moov``) plus the sample tables inside ``moov`` (``stss``,
``stsz``, ``stsc``, ``stco``/``co64``) are enough. Parsing needs only
small ranged reads, so an index can be built from object storage for a
few MB per file regardless of file size.

Used by :class:`LanceVideoWriter` (index at write time, bytes already in
memory) and by backfill jobs (ranged reads over existing blobs). The
reader consumes the resulting columns; it never parses MP4s itself.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from collections.abc import Callable


#: Version tag stored in column metadata of the index columns.
VIDEO_INDEX_VERSION = '1'
VIDEO_INDEX_META_KEY = 'swm:video_index_version'
INDEX_COLUMNS = ('moov_range', 'gop_frame_idx', 'gop_byte_offset')

_HEAD_PROBE = 64 * 1024


@dataclass
class Mp4Index:
    """GOP-level decode index for one MP4 file.

    Attributes:
        moov_range: ``[start, end)`` byte range of the ``moov`` box.
        gop_frame_idx: frame index of each GOP start (0-based, sorted;
            first entry is 0). One entry per sync sample.
        gop_byte_offset: byte offset of each GOP's first sample, plus a
            final sentinel one past the last sample — so GOP ``k`` spans
            ``[gop_byte_offset[k], gop_byte_offset[k + 1])`` and
            ``len(gop_byte_offset) == len(gop_frame_idx) + 1``.
    """

    moov_range: tuple[int, int]
    gop_frame_idx: list[int]
    gop_byte_offset: list[int]

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def __post_init__(self) -> None:
        self._n_frames = 0

    def ranges_for_frames(
        self, start: int, stop: int
    ) -> list[tuple[int, int]]:
        """Byte ranges (media data only) covering frames ``[start, stop)``."""
        import bisect

        lo = bisect.bisect_right(self.gop_frame_idx, start) - 1
        hi = bisect.bisect_left(self.gop_frame_idx, stop)
        lo = max(lo, 0)
        return [(self.gop_byte_offset[lo], self.gop_byte_offset[hi])]


def parse_mp4_index(
    read_range: Callable[[int, int], bytes], file_size: int
) -> Mp4Index:
    """Build an :class:`Mp4Index` using only ranged reads.

    Args:
        read_range: ``f(start, end) -> bytes`` for ``[start, end)``.
        file_size: total file size in bytes.
    """
    moov_start, moov_end = _find_box(read_range, file_size, b'moov')
    moov = read_range(moov_start, moov_end)
    stbl = _video_stbl(moov)

    sizes = _parse_stsz(stbl)
    offsets = _sample_offsets(stbl, sizes)
    sync = _parse_stss(stbl, n_samples=len(sizes))

    gop_frame_idx = [s - 1 for s in sync]  # stss is 1-based
    gop_byte_offset = [offsets[i] for i in gop_frame_idx]
    gop_byte_offset.append(offsets[-1] + sizes[-1])

    idx = Mp4Index(
        moov_range=(moov_start, moov_end),
        gop_frame_idx=gop_frame_idx,
        gop_byte_offset=gop_byte_offset,
    )
    idx._n_frames = len(sizes)
    return idx


def parse_mp4_index_bytes(data: bytes) -> Mp4Index:
    """Convenience wrapper for fully materialized MP4 bytes."""
    return parse_mp4_index(lambda a, b: data[a:b], len(data))


def _find_box(read_range, file_size: int, fourcc: bytes) -> tuple[int, int]:
    """Locate a top-level box by walking box headers (headers only)."""
    pos = 0
    probe = read_range(0, min(_HEAD_PROBE, file_size))
    while pos < file_size:
        if pos + 16 <= len(probe):
            header = probe[pos : pos + 16]
        else:
            header = read_range(pos, min(pos + 16, file_size))
        if len(header) < 8:
            break
        size = struct.unpack('>I', header[:4])[0]
        btype = header[4:8]
        body_off = 8
        if size == 1:
            size = struct.unpack('>Q', header[8:16])[0]
            body_off = 16
        elif size == 0:  # box extends to EOF
            size = file_size - pos
        if btype == fourcc:
            return pos, pos + size
        if size < body_off:
            raise ValueError(f'corrupt MP4: box {btype!r} size {size}')
        pos += size
    raise ValueError(f'MP4 box {fourcc!r} not found')


def _walk_children(data: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size = struct.unpack('>I', data[pos : pos + 4])[0]
        btype = data[pos + 4 : pos + 8]
        body = pos + 8
        if size == 1:
            size = struct.unpack('>Q', data[pos + 8 : pos + 16])[0]
            body = pos + 16
        elif size == 0:
            size = end - pos
        yield btype, body, pos + size
        pos += size


def _find_child(data: bytes, start: int, end: int, fourcc: bytes):
    for btype, body, box_end in _walk_children(data, start, end):
        if btype == fourcc:
            return body, box_end
    return None


def _video_stbl(moov: bytes) -> bytes:
    """Extract the sample-table box of the (first) video track."""
    for btype, body, box_end in _walk_children(moov, 8, len(moov)):
        if btype != b'trak':
            continue
        mdia = _find_child(moov, body, box_end, b'mdia')
        if mdia is None:
            continue
        hdlr = _find_child(moov, mdia[0], mdia[1], b'hdlr')
        if hdlr is None or moov[hdlr[0] + 8 : hdlr[0] + 12] != b'vide':
            continue
        minf = _find_child(moov, mdia[0], mdia[1], b'minf')
        stbl = _find_child(moov, minf[0], minf[1], b'stbl')
        return moov[stbl[0] - 8 : stbl[1]]
    raise ValueError('no video track found')


def _stbl_child(stbl: bytes, fourcc: bytes):
    return _find_child(stbl, 8, len(stbl), fourcc)


def _parse_stsz(stbl: bytes) -> list[int]:
    loc = _stbl_child(stbl, b'stsz')
    if loc is None:
        raise ValueError('stsz box missing')
    body = loc[0]
    uniform, count = struct.unpack('>II', stbl[body + 4 : body + 12])
    if uniform:
        return [uniform] * count
    off = body + 12
    return list(struct.unpack(f'>{count}I', stbl[off : off + 4 * count]))


def _parse_stss(stbl: bytes, n_samples: int) -> list[int]:
    loc = _stbl_child(stbl, b'stss')
    if loc is None:  # no sync table: every sample is a sync sample
        return list(range(1, n_samples + 1))
    body = loc[0]
    count = struct.unpack('>I', stbl[body + 4 : body + 8])[0]
    off = body + 8
    return list(struct.unpack(f'>{count}I', stbl[off : off + 4 * count]))


def _sample_offsets(stbl: bytes, sizes: list[int]) -> list[int]:
    """Per-sample file offsets from stsc x stco/co64."""
    loc = _stbl_child(stbl, b'stco')
    width = 4
    if loc is None:
        loc = _stbl_child(stbl, b'co64')
        width = 8
    if loc is None:
        raise ValueError('stco/co64 box missing')
    body = loc[0]
    n_chunks = struct.unpack('>I', stbl[body + 4 : body + 8])[0]
    off = body + 8
    fmt = '>' + ('I' if width == 4 else 'Q') * n_chunks
    chunk_offsets = list(
        struct.unpack(fmt, stbl[off : off + width * n_chunks])
    )

    loc = _stbl_child(stbl, b'stsc')
    body = loc[0]
    n_ent = struct.unpack('>I', stbl[body + 4 : body + 8])[0]
    ent = [
        struct.unpack('>III', stbl[body + 8 + 12 * i : body + 20 + 12 * i])
        for i in range(n_ent)
    ]  # (first_chunk 1-based, samples_per_chunk, sample_desc_idx)

    offsets: list[int] = []
    sample = 0
    for i, (first_chunk, per_chunk, _) in enumerate(ent):
        last_chunk = ent[i + 1][0] - 1 if i + 1 < len(ent) else n_chunks
        for chunk in range(first_chunk, last_chunk + 1):
            pos = chunk_offsets[chunk - 1]
            for _ in range(per_chunk):
                if sample >= len(sizes):
                    return offsets
                offsets.append(pos)
                pos += sizes[sample]
                sample += 1
    return offsets
