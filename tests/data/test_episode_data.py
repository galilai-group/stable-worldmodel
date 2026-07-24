"""Tests for episode-scoped data plumbing: the split helper, wrapper
forwarding (Merge/Concat), and convert carry-through."""

from __future__ import annotations

import numpy as np

from stable_worldmodel.data import (
    EPISODE_DATA_KEY,
    ConcatDataset,
    LanceDataset,
    LanceWriter,
    MergeDataset,
    convert,
    split_episode_data,
)


class _FakeEpisodeDataset:
    """Minimal dataset stub exposing the episode-data API plus the bits the
    composition wrappers need (``lengths``, ``column_names``, ``__len__``)."""

    def __init__(self, n_episodes: int, columns: list[str], tag: str):
        self.lengths = np.full(n_episodes, 5, dtype=np.int64)
        self.column_names = [f'{tag}_col']
        self._columns = list(columns)
        self._tag = tag

    def __len__(self) -> int:
        return 3

    @property
    def episode_column_names(self) -> list[str]:
        return list(self._columns)

    def get_episode_data(self, episodes_idx=None) -> dict[str, list]:
        if episodes_idx is None:
            episodes_idx = range(len(self.lengths))
        idxs = [int(i) for i in episodes_idx]
        return {
            c: [f'{self._tag}/{c}/{i}' for i in idxs] for c in self._columns
        }


class _PlainDataset:
    """Dataset without any episode-data API (pre-existing custom readers)."""

    def __init__(self, n_episodes: int):
        self.lengths = np.full(n_episodes, 5, dtype=np.int64)
        self.column_names = ['plain_col']

    def __len__(self) -> int:
        return 3


def test_split_episode_data_non_mutating():
    src = {'action': [1, 2], EPISODE_DATA_KEY: {'x': 1}}
    per_step, extra = split_episode_data(src)
    assert per_step == {'action': [1, 2]}
    assert extra == {'x': 1}
    # The source dict keeps its key and the returned mapping is a copy.
    assert EPISODE_DATA_KEY in src
    extra['y'] = 2
    assert 'y' not in src[EPISODE_DATA_KEY]


def test_split_episode_data_absent_key():
    src = {'action': [1]}
    per_step, extra = split_episode_data(src)
    assert per_step is src
    assert extra == {}


def test_split_episode_data_empty_mapping():
    src = {'action': [1], EPISODE_DATA_KEY: {}}
    per_step, extra = split_episode_data(src)
    assert per_step == {'action': [1]}
    assert extra == {}


def test_merge_dataset_union_first_wins():
    a = _FakeEpisodeDataset(3, ['xml', 'shared'], tag='a')
    b = _FakeEpisodeDataset(3, ['meta', 'shared'], tag='b')
    merged = MergeDataset([a, b])

    assert merged.episode_column_names == ['xml', 'shared', 'meta']
    data = merged.get_episode_data([1, 0])
    assert data['xml'] == ['a/xml/1', 'a/xml/0']
    assert data['meta'] == ['b/meta/1', 'b/meta/0']
    # Collision: the first dataset wins, mirroring per-step columns.
    assert data['shared'] == ['a/shared/1', 'a/shared/0']


def test_merge_dataset_without_episode_data():
    merged = MergeDataset([_PlainDataset(3), _PlainDataset(3)])
    assert merged.episode_column_names == []
    assert merged.get_episode_data() == {}


def test_concat_dataset_global_to_local_mapping():
    a = _FakeEpisodeDataset(2, ['xml'], tag='a')
    b = _FakeEpisodeDataset(3, ['xml'], tag='b')
    cat = ConcatDataset([a, b])

    assert cat.episode_column_names == ['xml']
    data = cat.get_episode_data([0, 3, 1, 4])
    assert data['xml'] == ['a/xml/0', 'b/xml/1', 'a/xml/1', 'b/xml/2']
    # None selects every episode in global order.
    assert cat.get_episode_data()['xml'] == [
        'a/xml/0',
        'a/xml/1',
        'b/xml/0',
        'b/xml/1',
        'b/xml/2',
    ]


def test_concat_dataset_none_fill_for_missing_keys():
    a = _FakeEpisodeDataset(2, ['xml'], tag='a')
    b = _PlainDataset(2)
    cat = ConcatDataset([a, b])

    assert cat.episode_column_names == ['xml']
    data = cat.get_episode_data([0, 2, 1])
    assert data['xml'] == ['a/xml/0', None, 'a/xml/1']


def _write_lance_with_episode_data(out, n_eps: int = 2) -> None:
    rng = np.random.default_rng(0)
    with LanceWriter(out) as w:
        for i in range(n_eps):
            w.write_episode(
                {
                    'pixels': [
                        rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
                        for _ in range(3)
                    ],
                    'action': [
                        rng.standard_normal(2).astype(np.float32)
                        for _ in range(3)
                    ],
                    EPISODE_DATA_KEY: {'model_xml': f'<scene {i}/>'},
                }
            )


def test_convert_lance_to_lance_carries_episode_data(tmp_path):
    src = tmp_path / 'src.lance'
    _write_lance_with_episode_data(src)

    dest = tmp_path / 'dst' / 'dst.lance'
    convert(str(src), str(dest), dest_format='lance', progress=False)

    ds = LanceDataset(path=dest)
    assert ds.episode_column_names == ['model_xml']
    assert ds.get_episode_data()['model_xml'] == ['<scene 0/>', '<scene 1/>']


def test_convert_to_folder_drops_episode_data_with_warning(tmp_path):
    from loguru import logger

    src = tmp_path / 'src.lance'
    _write_lance_with_episode_data(src)

    dest = tmp_path / 'folder_out'
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level='WARNING')
    try:
        convert(str(src), str(dest), dest_format='folder', progress=False)
    finally:
        logger.remove(sink_id)
    assert any('does not support episode data' in m for m in messages)

    # Round the folder back into lance: the episode data is gone for good
    # (proving the folder writer never received the reserved key).
    back = tmp_path / 'back' / 'back.lance'
    convert(str(dest), str(back), dest_format='lance', progress=False)
    ds = LanceDataset(path=back)
    assert ds.lengths.tolist() == [3, 3]
    assert ds.episode_column_names == []
    assert ds.get_episode_data() == {}


def test_base_dataset_defaults():
    from stable_worldmodel.data.dataset import Dataset

    ds = Dataset(
        lengths=np.array([3]), offsets=np.array([0]), frameskip=1, num_steps=1
    )
    assert ds.episode_column_names == []
    assert ds.get_episode_data() == {}
    assert ds.get_episode_data([0]) == {}


def test_format_flag():
    from stable_worldmodel.data import Format, get_format, list_formats

    assert Format.supports_episode_data is False  # base default
    supported = {'lance', 'lance_video'}
    for name in list_formats():
        assert get_format(name).supports_episode_data is (name in supported)
