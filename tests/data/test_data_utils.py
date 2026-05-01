"""Tests for stable_worldmodel.data.utils."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from stable_worldmodel.data.utils import (
    _download,
    _hf_find_dataset_entry,
    _hf_list_tree,
    _hf_walk_files,
    _resolve_dataset,
    _resolve_dataset_hf,
    ensure_dir_exists,
    get_cache_dir,
    load_dataset,
)
from stable_worldmodel.utils import DEFAULT_CACHE_DIR, HF_BASE_URL


# ─── get_cache_dir ────────────────────────────────────────────────────────────


def test_get_cache_dir_default_no_env(monkeypatch):
    monkeypatch.delenv('STABLEWM_HOME', raising=False)
    result = get_cache_dir()
    assert result == Path(DEFAULT_CACHE_DIR)


def test_get_cache_dir_uses_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv('STABLEWM_HOME', str(tmp_path))
    result = get_cache_dir()
    assert result == tmp_path


def test_get_cache_dir_override_root(tmp_path):
    result = get_cache_dir(override_root=tmp_path)
    assert result == tmp_path


def test_get_cache_dir_override_root_ignores_env(monkeypatch, tmp_path):
    monkeypatch.setenv('STABLEWM_HOME', '/some/other/path')
    result = get_cache_dir(override_root=tmp_path)
    assert result == tmp_path


def test_get_cache_dir_with_sub_folder(tmp_path):
    result = get_cache_dir(override_root=tmp_path, sub_folder='datasets')
    assert result == tmp_path / 'datasets'


def test_get_cache_dir_sub_folder_env(monkeypatch, tmp_path):
    monkeypatch.setenv('STABLEWM_HOME', str(tmp_path))
    result = get_cache_dir(sub_folder='models')
    assert result == tmp_path / 'models'


# ─── ensure_dir_exists ────────────────────────────────────────────────────────


def test_ensure_dir_exists_creates_new_dir(tmp_path):
    new_dir = tmp_path / 'a' / 'b' / 'c'
    assert not new_dir.exists()
    ensure_dir_exists(new_dir)
    assert new_dir.exists()


def test_ensure_dir_exists_existing_dir(tmp_path):
    ensure_dir_exists(tmp_path)  # should not raise
    assert tmp_path.exists()


# ─── _resolve_dataset ─────────────────────────────────────────────────────────


def test_resolve_dataset_explicit_h5_file(tmp_path):
    h5 = tmp_path / 'data.h5'
    h5.touch()
    result = _resolve_dataset(str(h5), tmp_path)
    assert result == h5


def test_resolve_dataset_explicit_h5_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resolve_dataset(str(tmp_path / 'missing.h5'), tmp_path)


def test_resolve_dataset_explicit_hdf5_file(tmp_path):
    h5 = tmp_path / 'data.hdf5'
    h5.touch()
    result = _resolve_dataset(str(h5), tmp_path)
    assert result == h5


def test_resolve_dataset_directory(tmp_path):
    sub = tmp_path / 'subdir'
    sub.mkdir()
    (sub / 'data.h5').touch()
    # _resolve_dataset returns the path as-is; format detection happens later.
    result = _resolve_dataset(str(sub), tmp_path)
    assert result == sub


def test_resolve_dataset_hf_repo(tmp_path):
    with patch('stable_worldmodel.data.utils._resolve_dataset_hf') as mock_hf:
        mock_hf.return_value = tmp_path / 'user--repo'
        result = _resolve_dataset('user/repo', tmp_path)
        mock_hf.assert_called_once_with('user/repo', tmp_path)
        assert result == tmp_path / 'user--repo'


def test_resolve_dataset_invalid_name_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match='Cannot resolve'):
        _resolve_dataset('not_a_valid_name', tmp_path)


# ─── _resolve_dataset_hf ──────────────────────────────────────────────────────


def test_resolve_dataset_hf_uses_cache(tmp_path):
    repo_id = 'user/repo'
    local_dir = tmp_path / 'user--repo'
    local_dir.mkdir()
    (local_dir / 'dataset.h5').touch()

    result = _resolve_dataset_hf(repo_id, tmp_path)
    assert result == local_dir


def test_resolve_dataset_hf_downloads_h5_file(tmp_path):
    repo_id = 'user/repo'
    expected_dir = tmp_path / 'user--repo'

    def fake_download(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    with (
        patch(
            'stable_worldmodel.data.utils._hf_find_dataset_entry',
            return_value={'path': 'dataset.h5', 'type': 'file'},
        ),
        patch(
            'stable_worldmodel.data.utils._download', side_effect=fake_download
        ),
    ):
        result = _resolve_dataset_hf(repo_id, tmp_path)

    assert result == expected_dir
    assert (expected_dir / 'dataset.h5').exists()


def test_resolve_dataset_hf_downloads_lance_directory(tmp_path):
    repo_id = 'user/repo'
    expected_dir = tmp_path / 'user--repo'

    def fake_download(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    with (
        patch(
            'stable_worldmodel.data.utils._hf_find_dataset_entry',
            return_value={'path': 'foo.lance', 'type': 'directory'},
        ),
        patch(
            'stable_worldmodel.data.utils._hf_walk_files',
            return_value=[
                'foo.lance/data/0.lance',
                'foo.lance/_versions/1.manifest',
            ],
        ),
        patch(
            'stable_worldmodel.data.utils._download', side_effect=fake_download
        ),
    ):
        result = _resolve_dataset_hf(repo_id, tmp_path)

    assert result == expected_dir
    assert (expected_dir / 'foo.lance' / 'data' / '0.lance').exists()
    assert (expected_dir / 'foo.lance' / '_versions' / '1.manifest').exists()


def test_resolve_dataset_hf_constructs_correct_url_for_file(tmp_path):
    repo_id = 'myorg/mydata'
    expected_url = f'{HF_BASE_URL}/datasets/{repo_id}/resolve/main/dataset.h5'
    captured = {}

    def fake_download(url, dest):
        captured['url'] = url
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    with (
        patch(
            'stable_worldmodel.data.utils._hf_find_dataset_entry',
            return_value={'path': 'dataset.h5', 'type': 'file'},
        ),
        patch(
            'stable_worldmodel.data.utils._download', side_effect=fake_download
        ),
    ):
        _resolve_dataset_hf(repo_id, tmp_path)

    assert captured['url'] == expected_url


# ─── _hf_find_dataset_entry ──────────────────────────────────────────────────


def _mock_tree_response(entries):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(entries).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def test_hf_find_dataset_entry_returns_h5_file():
    entries = [
        {'path': 'README.md', 'type': 'file'},
        {'path': 'data.h5', 'type': 'file'},
    ]
    with patch(
        'urllib.request.urlopen', return_value=_mock_tree_response(entries)
    ):
        entry = _hf_find_dataset_entry('user/repo')
    assert entry['path'] == 'data.h5'
    assert entry['type'] == 'file'


def test_hf_find_dataset_entry_returns_lance_directory():
    entries = [{'path': 'foo.lance', 'type': 'directory'}]
    with patch(
        'urllib.request.urlopen', return_value=_mock_tree_response(entries)
    ):
        entry = _hf_find_dataset_entry('user/repo')
    assert entry['path'] == 'foo.lance'


def test_hf_find_dataset_entry_prefers_lance_over_h5():
    entries = [
        {'path': 'data.h5', 'type': 'file'},
        {'path': 'foo.lance', 'type': 'directory'},
    ]
    with patch(
        'urllib.request.urlopen', return_value=_mock_tree_response(entries)
    ):
        entry = _hf_find_dataset_entry('user/repo')
    assert entry['path'] == 'foo.lance'


def test_hf_find_dataset_entry_raises_when_not_found():
    entries = [
        {'path': 'README.md', 'type': 'file'},
        {'path': 'config.json', 'type': 'file'},
    ]
    with patch(
        'urllib.request.urlopen', return_value=_mock_tree_response(entries)
    ):
        with pytest.raises(FileNotFoundError, match='No dataset found'):
            _hf_find_dataset_entry('user/repo')


def test_hf_list_tree_uses_datasets_api_url():
    entries = [{'path': 'data.h5', 'type': 'file'}]
    captured = {}

    def fake_urlopen(url):
        captured['url'] = url
        return _mock_tree_response(entries)

    with patch('urllib.request.urlopen', side_effect=fake_urlopen):
        _hf_list_tree('myorg/myrepo')

    assert '/api/datasets/myorg/myrepo/tree/main' in captured['url']


def test_hf_walk_files_recurses_into_subdirs():
    """Two-level walk: top → subdir → leaf files."""
    pages = {
        'foo.lance': [
            {'path': 'foo.lance/data', 'type': 'directory'},
            {'path': 'foo.lance/manifest', 'type': 'file'},
        ],
        'foo.lance/data': [
            {'path': 'foo.lance/data/0.lance', 'type': 'file'},
        ],
    }

    def fake_urlopen(url):
        for sub, entries in pages.items():
            if url.endswith(f'/tree/main/{sub}'):
                return _mock_tree_response(entries)
        raise AssertionError(f'unexpected url {url}')

    with patch('urllib.request.urlopen', side_effect=fake_urlopen):
        files = _hf_walk_files('user/repo', 'foo.lance')

    assert sorted(files) == [
        'foo.lance/data/0.lance',
        'foo.lance/manifest',
    ]


# ─── _download ────────────────────────────────────────────────────────────────


def test_download_writes_content(tmp_path):
    dest = tmp_path / 'file.bin'
    content = b'hello world'

    mock_response = MagicMock()
    mock_response.headers.get.return_value = str(len(content))
    mock_response.read.side_effect = [content, b'']

    with patch('urllib.request.urlopen', return_value=mock_response):
        _download('http://example.com/file', dest)

    assert dest.read_bytes() == content


def test_download_handles_no_content_length(tmp_path):
    dest = tmp_path / 'file.bin'
    content = b'data'

    mock_response = MagicMock()
    mock_response.headers.get.return_value = (
        '0'  # zero → treated as None by `or None`
    )
    mock_response.read.side_effect = [content, b'']

    with patch('urllib.request.urlopen', return_value=mock_response):
        _download('http://example.com/file', dest)

    assert dest.read_bytes() == content


# ─── load_dataset ─────────────────────────────────────────────────────────────


def _make_h5(path: Path):
    """Create a minimal valid HDF5 dataset file."""
    with h5py.File(path, 'w') as f:
        f.create_dataset('ep_len', data=np.array([5]))
        f.create_dataset('ep_offset', data=np.array([0]))
        f.create_dataset(
            'observation', data=np.random.rand(5, 4).astype(np.float32)
        )
        f.create_dataset(
            'action', data=np.random.rand(5, 2).astype(np.float32)
        )


def test_load_dataset_from_local_h5(tmp_path):
    """load_dataset autodetects HDF5 from a .h5 path and returns a working reader."""
    from stable_worldmodel.data import HDF5Dataset

    datasets_dir = tmp_path / 'datasets'
    datasets_dir.mkdir()
    h5 = datasets_dir / 'mydata.h5'
    _make_h5(h5)

    ds = load_dataset(str(h5), cache_dir=str(tmp_path))
    assert isinstance(ds, HDF5Dataset)
    assert ds.h5_path == h5


def test_load_dataset_from_directory(tmp_path):
    """load_dataset autodetects HDF5 from a directory containing one .h5 file."""
    from stable_worldmodel.data import HDF5Dataset

    datasets_dir = tmp_path / 'datasets'
    datasets_dir.mkdir()
    sub = datasets_dir / 'mydata'
    sub.mkdir()
    h5 = sub / 'dataset.h5'
    _make_h5(h5)

    ds = load_dataset(str(sub), cache_dir=str(tmp_path))
    assert isinstance(ds, HDF5Dataset)
    assert ds.h5_path == h5


def test_load_dataset_missing_file_raises(tmp_path):
    datasets_dir = tmp_path / 'datasets'
    datasets_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        load_dataset(str(tmp_path / 'missing.h5'), cache_dir=str(tmp_path))
