"""Blob storage compatibility, independent of optional video decoders."""

import numpy as np
import pytest

pytest.importorskip('lancedb')
pytest.importorskip('imageio')
pytest.importorskip('imageio_ffmpeg')

import lance  # noqa: E402
import lancedb  # noqa: E402
import pyarrow as pa  # noqa: E402

from stable_worldmodel.data import LanceVideoWriter  # noqa: E402


@pytest.mark.parametrize('existing_version', [None, '2.0', '2.1'])
def test_video_blob_storage_survives_append(tmp_path, existing_version):
    path = tmp_path / 'video'

    def write(mode):
        with LanceVideoWriter(path, mode=mode) as writer:
            writer.write_episode(
                {
                    'pixels': [
                        np.full((16, 16, 3), i * 20, dtype=np.uint8)
                        for i in range(3)
                    ],
                    'action': [np.array([0.5], dtype=np.float32)] * 3,
                }
            )

    def payloads():
        dataset = lance.dataset(str(path / 'video_videos.lance'))
        assert dataset.data_storage_version == (existing_version or '2.1')
        assert dataset.schema.field('video_bytes').metadata == {
            b'lance-encoding:blob': b'true'
        }
        values = []
        for blob in dataset.take_blobs(
            'video_bytes', indices=list(range(dataset.count_rows()))
        ):
            values.append(blob.readall())
            blob.close()
        return values

    write('error')
    if existing_version is not None:
        # Recreate the videos table in a legacy format before appending,
        # preserving the original schema and bytes.
        db = lancedb.connect(str(path))
        table = db.open_table('video_videos')
        dataset = table.to_lance()
        blob = dataset.take_blobs('video_bytes', indices=[0])[0]
        content = blob.readall()
        blob.close()
        db.create_table(
            'video_videos',
            data=pa.table(
                {
                    'episode_idx': [0],
                    'video_key': ['pixels'],
                    'video_bytes': [content],
                },
                schema=table.schema,
            ),
            mode='overwrite',
            storage_options={
                'new_table_data_storage_version': existing_version
            },
        )
    original = payloads()
    assert len(original) == 1
    assert original[0][4:8] == b'ftyp'
    write('append')
    assert payloads() == original * 2
    frames = lance.dataset(str(path / 'video.lance')).to_table()
    assert frames['episode_idx'].to_pylist() == [0, 0, 0, 1, 1, 1]
