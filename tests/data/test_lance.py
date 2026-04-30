"""Tests for the Lance format: writer round-trip, detection, batched fetch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from stable_worldmodel.data import (
    LanceDataset,
    LanceWriter,
    convert,
    detect_format,
    get_format,
    list_formats,
)


def _write_demo(
    out: Path,
    *,
    ep_lengths: tuple[int, ...] = (5, 4, 6),
    extra_cols: dict[str, tuple[int, ...]] | None = None,
    pixels_key: str = 'pixels',
    image_codec: str = 'jpeg',
) -> None:
    rng = np.random.default_rng(0)
    extra_cols = extra_cols or {}
    with LanceWriter(out, image_codec=image_codec) as w:
        for ep_len in ep_lengths:
            ep = {
                pixels_key: [
                    rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
                    for _ in range(ep_len)
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
            for col, shape in extra_cols.items():
                ep[col] = [
                    rng.standard_normal(shape).astype(np.float32)
                    for _ in range(ep_len)
                ]
            w.write_episode(ep)


def test_lance_registered():
    assert 'lance' in list_formats()
    assert get_format('lance').name == 'lance'


def test_lance_detect(tmp_path):
    out = tmp_path / 'demo.lance'
    _write_demo(out)
    fmt = detect_format(out)
    assert fmt is not None and fmt.name == 'lance'

    # Parent directory holding the .lance also detects.
    fmt_parent = detect_format(tmp_path)
    assert fmt_parent is not None and fmt_parent.name == 'lance'


def test_writer_roundtrip(tmp_path):
    out = tmp_path / 'demo.lance'
    _write_demo(out, ep_lengths=(5, 4, 6))

    ds = LanceDataset(path=out)
    assert ds.lengths.tolist() == [5, 4, 6]
    assert ds.offsets.tolist() == [0, 5, 9]
    assert sorted(ds.column_names) == ['action', 'pixels', 'proprio']
    assert len(ds) == sum([5, 4, 6])

    sample = ds[0]
    assert set(sample) == {'action', 'pixels', 'proprio'}
    assert sample['pixels'].shape == (1, 3, 8, 8)
    assert sample['action'].shape == (1, 2)
    assert sample['proprio'].shape == (1, 3)


def test_image_column_autodetected(tmp_path):
    out = tmp_path / 'demo.lance'
    _write_demo(out)
    ds = LanceDataset(path=out)
    assert ds.image_columns == {'pixels'}


def test_dot_rename(tmp_path):
    """HDF5-style dotted column names are renamed to underscores transparently."""
    rng = np.random.default_rng(0)
    out = tmp_path / 'multi.lance'
    with LanceWriter(out) as w:
        for ep_len in (3, 4):
            w.write_episode(
                {
                    'pixels.top': [
                        rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
                        for _ in range(ep_len)
                    ],
                    'pixels.wrist': [
                        rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
                        for _ in range(ep_len)
                    ],
                    'observation.state': [
                        rng.standard_normal(4).astype(np.float32)
                        for _ in range(ep_len)
                    ],
                }
            )

    ds = LanceDataset(path=out)
    assert set(ds.column_names) == {
        'pixels_top',
        'pixels_wrist',
        'observation_state',
    }
    assert ds.image_columns == {'pixels_top', 'pixels_wrist'}


def test_getitems_batched(tmp_path):
    out = tmp_path / 'demo.lance'
    _write_demo(out)
    ds = LanceDataset(path=out)
    indices = [0, 5, 10]
    batch = ds.__getitems__(indices)
    assert len(batch) == 3
    for i, idx in enumerate(indices):
        single = ds[idx]
        np.testing.assert_array_equal(
            batch[i]['pixels'].numpy(), single['pixels'].numpy()
        )
        np.testing.assert_allclose(
            batch[i]['proprio'].numpy(), single['proprio'].numpy()
        )


def test_get_col_and_row(tmp_path):
    out = tmp_path / 'demo.lance'
    _write_demo(out)
    ds = LanceDataset(path=out)

    proprio = ds.get_col_data('proprio')
    assert proprio.shape == (sum([5, 4, 6]), 3)

    rows = ds.get_row_data([0, 1, 2])
    assert rows['proprio'].shape == (3, 3)


def test_merge_col(tmp_path):
    out = tmp_path / 'demo.lance'
    _write_demo(out, extra_cols={'velocity': (3,)})
    ds = LanceDataset(path=out)
    ds.merge_col(['proprio', 'velocity'], 'state')
    assert 'state' in ds.column_names
    assert ds.get_col_data('state').shape == (sum([5, 4, 6]), 6)


def test_error_mode_raises_for_existing_table(tmp_path):
    out = tmp_path / 'demo.lance'
    _write_demo(out)
    with pytest.raises(FileExistsError):
        with LanceWriter(out, mode='error') as w:
            w.write_episode(
                {
                    'pixels': [
                        np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)
                    ],
                    'action': [
                        np.zeros(2, dtype=np.float32) for _ in range(2)
                    ],
                }
            )


def test_overwrite_replaces_table(tmp_path):
    out = tmp_path / 'demo.lance'
    _write_demo(out, ep_lengths=(5, 4, 6))
    with LanceWriter(out, mode='overwrite') as w:
        for ep_len in (3, 3):
            w.write_episode(
                {
                    'pixels': [
                        np.zeros((8, 8, 3), dtype=np.uint8)
                        for _ in range(ep_len)
                    ],
                    'action': [
                        np.zeros(2, dtype=np.float32) for _ in range(ep_len)
                    ],
                }
            )
    ds = LanceDataset(path=out)
    assert ds.lengths.tolist() == [3, 3]


def test_append_extends_existing_table(tmp_path):
    out = tmp_path / 'demo.lance'
    _write_demo(out, ep_lengths=(5, 4))

    with LanceWriter(out) as w:  # mode='append' is the default
        for ep_len in (3, 6):
            w.write_episode(
                {
                    'pixels': [
                        np.zeros((8, 8, 3), dtype=np.uint8)
                        for _ in range(ep_len)
                    ],
                    'action': [
                        np.zeros(2, dtype=np.float32) for _ in range(ep_len)
                    ],
                    'proprio': [
                        np.zeros(3, dtype=np.float32) for _ in range(ep_len)
                    ],
                }
            )

    ds = LanceDataset(path=out)
    assert ds.lengths.tolist() == [5, 4, 3, 6]
    assert ds.offsets.tolist() == [0, 5, 9, 12]


def test_append_schema_mismatch_raises(tmp_path):
    out = tmp_path / 'demo.lance'
    _write_demo(out, ep_lengths=(3,))

    with pytest.raises(ValueError, match='schema mismatch'):
        with LanceWriter(out) as w:
            w.write_episode(
                {
                    'pixels': [
                        np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)
                    ],
                    'action': [
                        np.zeros(2, dtype=np.float32) for _ in range(2)
                    ],
                    # missing 'proprio'; new key 'extra'
                    'extra': [np.zeros(2, dtype=np.float32) for _ in range(2)],
                }
            )


def test_append_dim_mismatch_raises(tmp_path):
    out = tmp_path / 'demo.lance'
    _write_demo(out, ep_lengths=(3,))

    with pytest.raises(ValueError, match='dimension mismatch'):
        with LanceWriter(out) as w:
            w.write_episode(
                {
                    'pixels': [
                        np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)
                    ],
                    'action': [
                        np.zeros(2, dtype=np.float32) for _ in range(2)
                    ],
                    'proprio': [
                        np.zeros(99, dtype=np.float32) for _ in range(2)
                    ],
                }
            )


def test_invalid_mode_raises(tmp_path):
    out = tmp_path / 'demo.lance'
    with pytest.raises(ValueError, match='write mode'):
        LanceWriter(out, mode='nope')


def test_invalid_image_codec_raises(tmp_path):
    out = tmp_path / 'demo.lance'
    with pytest.raises(ValueError, match='image_codec'):
        LanceWriter(out, image_codec='nonsense')


def test_codec_raw_roundtrip(tmp_path):
    """codec=raw — pixels stored as pa.large_binary with shape + compression
    metadata so Lance applies general compression. Reader decodes via
    np.frombuffer + reshape (no JPEG decode)."""
    out = tmp_path / 'demo.lance'
    _write_demo(out, image_codec='raw')

    import pyarrow as pa
    import lancedb

    schema = lancedb.connect(str(tmp_path)).open_table('demo').schema
    pixel_field = schema.field('pixels')
    assert pa.types.is_large_binary(pixel_field.type)
    md = pixel_field.metadata or {}
    assert md.get(b'image_shape') == b'8,8,3'
    # Compression hint must be present for Lance to engage zstd.
    assert md.get(b'lance-encoding:compression') == b'zstd'

    ds = LanceDataset(path=out)
    assert ds.image_columns == {'pixels'}
    assert ds._raw_image_shapes['pixels'] == (8, 8, 3)
    sample = ds[0]
    assert sample['pixels'].shape == (1, 3, 8, 8)
    assert sample['pixels'].dtype == torch.uint8


def test_codec_raw_legacy_fixed_size_list_still_readable(tmp_path):
    """The reader auto-detects the old fixed_size_list<uint8> raw layout —
    not produced by the writer any more, but datasets out in the wild
    still work."""
    import pyarrow as pa
    import lancedb

    rng = np.random.default_rng(0)
    H = W = 8
    C = 3
    flat = rng.integers(0, 255, (3, H * W * C), dtype=np.uint8).reshape(-1)
    schema = pa.schema(
        [
            pa.field('episode_idx', pa.int32()),
            pa.field('step_idx', pa.int32()),
            pa.field(
                'pixels',
                pa.list_(pa.uint8(), H * W * C),
                metadata={b'image_shape': f'{H},{W},{C}'.encode()},
            ),
        ]
    )
    arr = pa.FixedSizeListArray.from_arrays(
        pa.array(flat, type=pa.uint8()), H * W * C
    )
    table = pa.table(
        [
            pa.array([0, 0, 0], type=pa.int32()),
            pa.array([0, 1, 2], type=pa.int32()),
            arr,
        ],
        schema=schema,
    )
    lancedb.connect(str(tmp_path)).create_table(
        'legacy_raw', data=table, schema=schema, mode='overwrite'
    )

    ds = LanceDataset(path=tmp_path / 'legacy_raw.lance')
    assert ds._raw_image_shapes['pixels'] == (8, 8, 3)
    assert ds[0]['pixels'].shape == (1, 3, 8, 8)
    assert ds[0]['pixels'].dtype == torch.uint8


def test_codec_jpeg_roundtrip(tmp_path):
    """codec=jpeg — pixels stored as pa.binary, decoded on read (existing path)."""
    out = tmp_path / 'demo.lance'
    _write_demo(out, image_codec='jpeg')

    import pyarrow as pa
    import lancedb

    schema = lancedb.connect(str(tmp_path)).open_table('demo').schema
    assert pa.types.is_binary(schema.field('pixels').type)

    ds = LanceDataset(path=out)
    assert ds.image_columns == {'pixels'}
    assert 'pixels' not in ds._raw_image_shapes
    sample = ds[0]
    assert sample['pixels'].shape == (1, 3, 8, 8)


def test_codec_both_roundtrip(tmp_path):
    """codec=both — pixels (raw) AND pixels_jpeg (binary) coexist; reader
    picks each up as its own image column when projected."""
    out = tmp_path / 'demo.lance'
    _write_demo(out, image_codec='both')

    import pyarrow as pa
    import lancedb

    schema = lancedb.connect(str(tmp_path)).open_table('demo').schema
    assert pa.types.is_large_binary(schema.field('pixels').type)
    assert pa.types.is_binary(schema.field('pixels_jpeg').type)

    # Default reader projects both; both auto-detected as images.
    ds = LanceDataset(path=out)
    assert ds.image_columns == {'pixels', 'pixels_jpeg'}
    assert 'pixels' in ds._raw_image_shapes
    assert 'pixels_jpeg' not in ds._raw_image_shapes

    # Project only the raw col → no JPEG decode at all.
    ds_raw = LanceDataset(
        path=out, keys_to_load=['pixels', 'action', 'proprio']
    )
    assert ds_raw.image_columns == {'pixels'}
    assert ds_raw[0]['pixels'].shape == (1, 3, 8, 8)

    # Project only the jpeg col → goes through decode path.
    ds_jpeg = LanceDataset(
        path=out, keys_to_load=['pixels_jpeg', 'action', 'proprio']
    )
    assert ds_jpeg.image_columns == {'pixels_jpeg'}
    assert ds_jpeg[0]['pixels_jpeg'].shape == (1, 3, 8, 8)


def test_codec_both_multi_camera(tmp_path):
    """Multi-camera + codec='both': each camera gets a raw col AND a jpeg
    companion; reader auto-detects all four as image columns."""
    rng = np.random.default_rng(0)
    out = tmp_path / 'multi.lance'
    with LanceWriter(out, image_codec='both') as w:
        for ep_len in (3, 4):
            w.write_episode(
                {
                    'pixels.top': [
                        rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
                        for _ in range(ep_len)
                    ],
                    'pixels.wrist': [
                        rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
                        for _ in range(ep_len)
                    ],
                    'observation.state': [
                        rng.standard_normal(4).astype(np.float32)
                        for _ in range(ep_len)
                    ],
                }
            )

    ds = LanceDataset(path=out)
    assert set(ds.column_names) == {
        'pixels_top',
        'pixels_top_jpeg',
        'pixels_wrist',
        'pixels_wrist_jpeg',
        'observation_state',
    }
    assert ds.image_columns == {
        'pixels_top',
        'pixels_top_jpeg',
        'pixels_wrist',
        'pixels_wrist_jpeg',
    }
    # The two `_jpeg` companions go through the JPEG decode path.
    assert set(ds._raw_image_shapes) == {'pixels_top', 'pixels_wrist'}
    sample = ds[0]
    for k in (
        'pixels_top',
        'pixels_top_jpeg',
        'pixels_wrist',
        'pixels_wrist_jpeg',
    ):
        assert sample[k].shape == (1, 3, 8, 8)
        assert sample[k].dtype == torch.uint8


def test_codec_jpeg_append_to_jpeg_table(tmp_path):
    """Append works against a legacy jpeg-codec table even when the writer's
    default has flipped to raw — codec is inferred from existing schema."""
    out = tmp_path / 'demo.lance'
    _write_demo(out, ep_lengths=(3, 4), image_codec='jpeg')

    # Default-constructed writer (codec='raw') should still append in jpeg.
    with LanceWriter(out) as w:
        w.write_episode(
            {
                'pixels': [
                    np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)
                ],
                'action': [np.zeros(2, dtype=np.float32) for _ in range(2)],
                'proprio': [np.zeros(3, dtype=np.float32) for _ in range(2)],
            }
        )

    ds = LanceDataset(path=out)
    assert ds.lengths.tolist() == [3, 4, 2]
    assert ds.image_columns == {'pixels'}
    assert 'pixels' not in ds._raw_image_shapes  # still jpeg on disk


def test_convert_lance_to_folder_and_back(tmp_path):
    src = tmp_path / 'src.lance'
    _write_demo(src, ep_lengths=(3, 4))

    folder_out = tmp_path / 'folder_out'
    convert(
        str(src), str(folder_out), source_format='lance', dest_format='folder'
    )
    assert (folder_out / 'ep_len.npz').exists()
    assert (folder_out / 'pixels').is_dir()

    back = tmp_path / 'roundtrip.lance'
    convert(
        str(folder_out),
        str(back),
        source_format='folder',
        dest_format='lance',
    )
    ds = LanceDataset(path=back)
    assert ds.lengths.tolist() == [3, 4]
