"""Tests for stable_worldmodel.data.utils."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from stable_worldmodel.data.utils import (
    _download,
    _extract_zst_tar,
    _resolve_dataset,
    _resolve_dataset_folder,
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


# ─── _resolve_dataset_folder ──────────────────────────────────────────────────


def test_resolve_dataset_folder_single_h5(tmp_path):
    h5 = tmp_path / 'data.h5'
    h5.touch()
    result = _resolve_dataset_folder(tmp_path)
    assert result == h5


def test_resolve_dataset_folder_single_hdf5(tmp_path):
    h5 = tmp_path / 'data.hdf5'
    h5.touch()
    result = _resolve_dataset_folder(tmp_path)
    assert result == h5


def test_resolve_dataset_folder_no_h5_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resolve_dataset_folder(tmp_path)


def test_resolve_dataset_folder_multiple_h5_raises(tmp_path):
    (tmp_path / 'a.h5').touch()
    (tmp_path / 'b.h5').touch()
    with pytest.raises(ValueError, match='Ambiguous'):
        _resolve_dataset_folder(tmp_path)


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
    h5 = sub / 'data.h5'
    h5.touch()
    result = _resolve_dataset(str(sub), tmp_path)
    assert result == h5


def test_resolve_dataset_hf_repo(tmp_path):
    with patch('stable_worldmodel.data.utils._resolve_dataset_hf') as mock_hf:
        mock_hf.return_value = tmp_path / 'data.h5'
        result = _resolve_dataset('user/repo', tmp_path)
        mock_hf.assert_called_once_with('user/repo', tmp_path)
        assert result == tmp_path / 'data.h5'


def test_resolve_dataset_invalid_name_raises(tmp_path):
    with pytest.raises(ValueError, match="Cannot resolve"):
        _resolve_dataset('not_a_valid_name', tmp_path)


# ─── _resolve_dataset_hf ──────────────────────────────────────────────────────


def test_resolve_dataset_hf_uses_cache(tmp_path):
    repo_id = 'user/repo'
    local_dir = tmp_path / 'user--repo'
    local_dir.mkdir()
    h5 = local_dir / 'dataset.h5'
    h5.touch()

    result = _resolve_dataset_hf(repo_id, tmp_path)
    assert result == h5


def test_resolve_dataset_hf_downloads_when_not_cached(tmp_path):
    repo_id = 'user/repo'
    expected_h5 = tmp_path / 'user--repo' / 'dataset.h5'

    def fake_download(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    def fake_extract(archive, dest):
        # leave archive in place — _resolve_dataset_hf will unlink it
        (dest / 'dataset.h5').touch()

    with (
        patch('stable_worldmodel.data.utils._download', side_effect=fake_download),
        patch('stable_worldmodel.data.utils._extract_zst_tar', side_effect=fake_extract),
    ):
        result = _resolve_dataset_hf(repo_id, tmp_path)

    assert result == expected_h5


def test_resolve_dataset_hf_constructs_correct_url(tmp_path):
    repo_id = 'myorg/mydata'
    expected_url = f'{HF_BASE_URL}/{repo_id}/resolve/main/dataset.tar.zst'
    captured = {}

    def fake_download(url, dest):
        captured['url'] = url
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    def fake_extract(archive, dest):
        # leave archive in place — _resolve_dataset_hf will unlink it
        (dest / 'dataset.h5').touch()

    with (
        patch('stable_worldmodel.data.utils._download', side_effect=fake_download),
        patch('stable_worldmodel.data.utils._extract_zst_tar', side_effect=fake_extract),
    ):
        _resolve_dataset_hf(repo_id, tmp_path)

    assert captured['url'] == expected_url


# ─── _extract_zst_tar ─────────────────────────────────────────────────────────


def test_extract_zst_tar_success(tmp_path):
    archive = tmp_path / 'data.tar.zst'
    archive.touch()

    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr='')
        _extract_zst_tar(archive, tmp_path)

    cmd = mock_run.call_args[0][0]
    assert '--use-compress-program=unzstd' in cmd
    assert str(archive) in cmd
    assert str(tmp_path) in cmd


def test_extract_zst_tar_failure_raises(tmp_path):
    archive = tmp_path / 'data.tar.zst'
    archive.touch()

    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr='extraction failed')
        with pytest.raises(RuntimeError, match='Failed to extract'):
            _extract_zst_tar(archive, tmp_path)


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
    mock_response.headers.get.return_value = '0'  # zero → treated as None by `or None`
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
        f.create_dataset('observation', data=np.random.rand(5, 4).astype(np.float32))
        f.create_dataset('action', data=np.random.rand(5, 2).astype(np.float32))


def test_load_dataset_from_local_h5(tmp_path):
    # load_dataset resolves the h5, computes a relative name, and delegates to HDF5Dataset
    datasets_dir = tmp_path / 'datasets'
    datasets_dir.mkdir()
    h5 = datasets_dir / 'mydata.h5'
    _make_h5(h5)

    with patch('stable_worldmodel.data.dataset.HDF5Dataset') as mock_cls:
        load_dataset(str(h5), cache_dir=str(tmp_path))
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs['name'] == 'mydata'


def test_load_dataset_from_directory(tmp_path):
    datasets_dir = tmp_path / 'datasets'
    datasets_dir.mkdir()
    sub = datasets_dir / 'mydata'
    sub.mkdir()
    h5 = sub / 'dataset.h5'
    _make_h5(h5)

    with patch('stable_worldmodel.data.dataset.HDF5Dataset') as mock_cls:
        load_dataset(str(sub), cache_dir=str(tmp_path))
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs['name'] == 'mydata/dataset'


def test_load_dataset_missing_file_raises(tmp_path):
    datasets_dir = tmp_path / 'datasets'
    datasets_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        load_dataset(str(tmp_path / 'missing.h5'), cache_dir=str(tmp_path))
