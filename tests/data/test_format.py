"""Tests for the format registry, writers, and conversion utility."""

from __future__ import annotations

import importlib
from pathlib import Path

import h5py
import numpy as np
import pytest

import stable_worldmodel as swm
from stable_worldmodel.data import (
    FORMATS,
    Format,
    FolderDataset,
    HDF5Dataset,
    VideoDataset,
    convert,
    detect_format,
    get_format,
    list_formats,
    register_format,
)
from stable_worldmodel.data.formats.folder import Folder, FolderWriter
from stable_worldmodel.data.formats.hdf5 import HDF5, HDF5Writer
from stable_worldmodel.data.formats.lerobot import LeRobot
from stable_worldmodel.data.formats.video import Video, VideoWriter


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _episode(n_steps: int, with_pixels: bool = True) -> dict:
    """Build an in-memory episode in writer input shape."""
    rng = np.random.default_rng(0)
    ep = {
        'action': [
            rng.standard_normal(2).astype(np.float32) for _ in range(n_steps)
        ],
        'proprio': [
            rng.standard_normal(4).astype(np.float32) for _ in range(n_steps)
        ],
    }
    if with_pixels:
        ep['pixels'] = [
            rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
            for _ in range(n_steps)
        ]
    return ep


@pytest.fixture
def two_episodes():
    return [_episode(5), _episode(7)]


# ─── Registry ─────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_builtins_registered(self):
        assert {'hdf5', 'folder', 'video', 'lerobot'}.issubset(
            set(list_formats())
        )

    def test_get_format_returns_class(self):
        assert get_format('hdf5') is HDF5
        assert get_format('folder') is Folder
        assert get_format('video') is Video
        assert get_format('lerobot') is LeRobot

    def test_get_format_unknown_raises(self):
        with pytest.raises(ValueError, match='unknown format'):
            get_format('does_not_exist')

    def test_register_format_adds_to_registry(self):
        class _Tmp(Format):
            name = '_tmp_test_register'

            @classmethod
            def detect(cls, path):
                return False

        try:
            register_format(_Tmp)
            assert FORMATS['_tmp_test_register'] is _Tmp
            assert '_tmp_test_register' in list_formats()
        finally:
            FORMATS.pop('_tmp_test_register', None)

    def test_register_format_rejects_missing_name(self):
        class _NoName(Format):
            pass

        with pytest.raises(ValueError, match='non-empty'):
            register_format(_NoName)

    def test_register_format_rejects_duplicate(self):
        class _Dup(Format):
            name = 'hdf5'  # already taken

            @classmethod
            def detect(cls, path):
                return False

        with pytest.raises(ValueError, match='already registered'):
            register_format(_Dup)

    def test_register_format_returns_class_for_decorator_use(self):
        # @register_format must return the class so it can be used as a decorator.
        class _Echo(Format):
            name = '_echo_test'

            @classmethod
            def detect(cls, path):
                return False

        try:
            assert register_format(_Echo) is _Echo
        finally:
            FORMATS.pop('_echo_test', None)

    def test_format_base_methods_raise(self, tmp_path):
        # Subclasses that don't override these should still get a clean
        # NotImplementedError (not AttributeError).
        class _Bare(Format):
            name = '_bare_test'

        with pytest.raises(NotImplementedError):
            _Bare.detect(tmp_path)
        with pytest.raises(NotImplementedError):
            _Bare.open_reader(tmp_path)
        with pytest.raises(NotImplementedError):
            _Bare.open_writer(tmp_path)


# ─── Detection ────────────────────────────────────────────────────────────────


class TestDetect:
    def test_hdf5_matches_h5_file(self, tmp_path):
        # detect() is a path shape check, not a file existence check.
        f = tmp_path / 'data.h5'
        f.touch()
        assert HDF5.detect(f) is True
        assert HDF5.detect(tmp_path / 'whatever.hdf5') is True
        assert HDF5.detect(tmp_path / 'whatever.txt') is False

    def test_hdf5_matches_dir_with_h5(self, tmp_path):
        (tmp_path / 'a.h5').touch()
        assert HDF5.detect(tmp_path) is True

    def test_hdf5_rejects_unrelated(self, tmp_path):
        (tmp_path / 'data.txt').touch()
        assert HDF5.detect(tmp_path / 'data.txt') is False

    def test_folder_requires_ep_len(self, tmp_path):
        assert Folder.detect(tmp_path) is False
        (tmp_path / 'ep_len.npz').touch()
        assert Folder.detect(tmp_path) is True

    def test_folder_loses_to_video_when_mp4_present(self, tmp_path):
        (tmp_path / 'ep_len.npz').touch()
        sub = tmp_path / 'video'
        sub.mkdir()
        (sub / 'ep_0.mp4').touch()
        assert Folder.detect(tmp_path) is False
        assert Video.detect(tmp_path) is True

    def test_video_rejects_dir_without_mp4(self, tmp_path):
        (tmp_path / 'ep_len.npz').touch()
        assert Video.detect(tmp_path) is False

    def test_lerobot_matches_scheme(self):
        assert LeRobot.detect('lerobot://lerobot/pusht') is True
        assert LeRobot.detect('lerobot/pusht') is False
        assert LeRobot.detect('/some/path') is False
        assert LeRobot.detect(Path('/some/path')) is False

    def test_detect_format_dispatches(self, tmp_path):
        h5 = tmp_path / 'a.h5'
        h5.touch()
        assert detect_format(h5) is HDF5

        folder = tmp_path / 'folder_ds'
        folder.mkdir()
        (folder / 'ep_len.npz').touch()
        assert detect_format(folder) is Folder

    def test_detect_format_returns_none_for_unknown(self, tmp_path):
        unknown = tmp_path / 'nothing.bin'
        unknown.touch()
        assert detect_format(unknown) is None


# ─── HDF5 writer ──────────────────────────────────────────────────────────────


class TestHDF5Writer:
    def test_roundtrip(self, tmp_path, two_episodes):
        out = tmp_path / 'data.h5'
        with HDF5Writer(out) as w:
            for ep in two_episodes:
                w.write_episode(ep)

        ds = HDF5Dataset(path=out)
        assert list(ds.lengths) == [5, 7]
        assert list(ds.offsets) == [0, 5]
        assert set(ds.column_names) == {'action', 'proprio', 'pixels'}

        ep0 = ds.load_episode(0)
        assert ep0['pixels'].shape == (5, 3, 8, 8)  # NCHW
        assert ep0['proprio'].shape == (5, 4)

    def test_open_writer_via_format(self, tmp_path, two_episodes):
        out = tmp_path / 'data.h5'
        with HDF5.open_writer(out) as w:
            w.write_episode(two_episodes[0])
        assert out.exists()
        assert isinstance(HDF5.open_reader(out), HDF5Dataset)

    def test_outside_with_block_raises(self, tmp_path, two_episodes):
        w = HDF5Writer(tmp_path / 'data.h5')
        with pytest.raises(RuntimeError, match='outside of a `with` block'):
            w.write_episode(two_episodes[0])

    def test_ep_offsets_are_cumulative(self, tmp_path, two_episodes):
        out = tmp_path / 'data.h5'
        with HDF5Writer(out) as w:
            for ep in two_episodes:
                w.write_episode(ep)
        with h5py.File(out, 'r') as f:
            assert list(f['ep_len'][:]) == [5, 7]
            assert list(f['ep_offset'][:]) == [0, 5]
            assert f['action'].shape[0] == 12

    def test_per_step_shape_mismatch_raises(self, tmp_path):
        # Schema is locked after the first episode — a column whose per-step
        # shape changes in a later episode must fail loudly.
        out = tmp_path / 'data.h5'
        rng = np.random.default_rng(0)
        ep_a = {
            'action': [
                rng.standard_normal(2).astype(np.float32) for _ in range(3)
            ]
        }
        ep_b = {
            'action': [
                rng.standard_normal(7).astype(np.float32) for _ in range(3)
            ]
        }
        with pytest.raises((ValueError, TypeError, OSError)):
            with HDF5Writer(out) as w:
                w.write_episode(ep_a)
                w.write_episode(ep_b)

    def test_append_extends_existing_file(self, tmp_path, two_episodes):
        out = tmp_path / 'data.h5'
        with HDF5Writer(out) as w:
            w.write_episode(two_episodes[0])
        with HDF5Writer(out) as w:  # default mode='append'
            w.write_episode(two_episodes[1])
        ds = HDF5Dataset(path=out)
        assert list(ds.lengths) == [5, 7]
        assert list(ds.offsets) == [0, 5]

    def test_error_mode_raises_for_existing_file(self, tmp_path, two_episodes):
        out = tmp_path / 'data.h5'
        with HDF5Writer(out) as w:
            w.write_episode(two_episodes[0])
        with pytest.raises(FileExistsError):
            HDF5Writer(out, mode='error').__enter__()

    def test_overwrite_truncates_existing_file(self, tmp_path, two_episodes):
        out = tmp_path / 'data.h5'
        with HDF5Writer(out) as w:
            w.write_episode(two_episodes[0])
            w.write_episode(two_episodes[1])
        with HDF5Writer(out, mode='overwrite') as w:
            w.write_episode(two_episodes[1])
        ds = HDF5Dataset(path=out)
        assert list(ds.lengths) == [7]

    def test_append_schema_mismatch_raises(self, tmp_path, two_episodes):
        out = tmp_path / 'data.h5'
        with HDF5Writer(out) as w:
            w.write_episode(two_episodes[0])
        ep_extra = dict(two_episodes[1])
        ep_extra['unexpected'] = [
            np.zeros(2, np.float32) for _ in range(len(ep_extra['action']))
        ]
        with pytest.raises(ValueError, match='schema mismatch'):
            with HDF5Writer(out) as w:
                w.write_episode(ep_extra)

    def test_append_per_step_shape_mismatch_raises(
        self, tmp_path, two_episodes
    ):
        out = tmp_path / 'data.h5'
        with HDF5Writer(out) as w:
            w.write_episode(two_episodes[0])
        ep_bad = {
            'action': [np.zeros(99, np.float32) for _ in range(3)],
            'proprio': [np.zeros(4, np.float32) for _ in range(3)],
            'pixels': [np.zeros((8, 8, 3), np.uint8) for _ in range(3)],
        }
        with pytest.raises(ValueError, match='shape mismatch'):
            with HDF5Writer(out) as w:
                w.write_episode(ep_bad)

    def test_invalid_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match='write mode'):
            HDF5Writer(tmp_path / 'data.h5', mode='nope')


class TestHDF5OpenReader:
    def test_directory_with_single_h5(self, tmp_path, two_episodes):
        sub = tmp_path / 'data_dir'
        sub.mkdir()
        with HDF5Writer(sub / 'data.h5') as w:
            w.write_episode(two_episodes[0])
        ds = HDF5.open_reader(sub)
        assert isinstance(ds, HDF5Dataset)

    def test_directory_with_no_h5_raises(self, tmp_path):
        sub = tmp_path / 'empty_dir'
        sub.mkdir()
        with pytest.raises(FileNotFoundError, match='No .h5/.hdf5'):
            HDF5.open_reader(sub)

    def test_directory_with_multiple_h5_raises(self, tmp_path):
        sub = tmp_path / 'ambiguous'
        sub.mkdir()
        (sub / 'a.h5').touch()
        (sub / 'b.h5').touch()
        with pytest.raises(ValueError, match='Ambiguous'):
            HDF5.open_reader(sub)


# ─── Folder writer ────────────────────────────────────────────────────────────


class TestFolderWriter:
    def test_roundtrip(self, tmp_path, two_episodes):
        out = tmp_path / 'folder_ds'
        with FolderWriter(out) as w:
            for ep in two_episodes:
                w.write_episode(ep)

        # Layout assertions.
        assert (out / 'ep_len.npz').exists()
        assert (out / 'ep_offset.npz').exists()
        assert (out / 'action.npz').exists()
        assert (out / 'pixels').is_dir()
        assert (out / 'pixels' / 'ep_0_step_0.jpeg').exists()
        assert (out / 'pixels' / 'ep_1_step_6.jpeg').exists()

        # Reader sees the same shape.
        ds = FolderDataset(path=out, folder_keys=['pixels'])
        assert list(ds.lengths) == [5, 7]
        assert set(ds.column_names) == {'action', 'proprio', 'pixels'}

        ep0 = ds.load_episode(0)
        assert ep0['pixels'].shape == (5, 3, 8, 8)  # NCHW after reader permute
        assert ep0['proprio'].shape == (5, 4)

    def test_format_open_writer_and_reader(self, tmp_path, two_episodes):
        out = tmp_path / 'fmt_folder'
        with Folder.open_writer(out) as w:
            for ep in two_episodes:
                w.write_episode(ep)
        ds = Folder.open_reader(out, folder_keys=['pixels'])
        assert isinstance(ds, FolderDataset)
        assert len(ds.lengths) == 2

    def test_autodetects_folder_keys_from_subdirs(
        self, tmp_path, two_episodes
    ):
        # Without folder_keys, FolderDataset should pick up subdirectories
        # (image/video columns) automatically from the on-disk layout.
        out = tmp_path / 'autodetect'
        with FolderWriter(out) as w:
            for ep in two_episodes:
                w.write_episode(ep)
        ds = FolderDataset(path=out)  # no folder_keys passed
        assert ds.folder_keys == ['pixels']
        ep0 = ds.load_episode(0)
        assert ep0['pixels'].shape == (5, 3, 8, 8)

    def test_explicit_empty_folder_keys_disables_subdir_loading(
        self, tmp_path, two_episodes
    ):
        # Passing folder_keys=[] explicitly should NOT auto-detect — let the
        # caller treat all subdirs as nonexistent.
        out = tmp_path / 'no_folder_keys'
        with FolderWriter(out) as w:
            w.write_episode(two_episodes[0])
        ds = FolderDataset(
            path=out,
            folder_keys=[],
            keys_to_load=['action', 'proprio'],
        )
        assert ds.folder_keys == []
        ep0 = ds.load_episode(0)
        assert 'pixels' not in ep0

    def test_append_extends_existing_folder(self, tmp_path, two_episodes):
        out = tmp_path / 'folder_ds'
        with FolderWriter(out) as w:
            w.write_episode(two_episodes[0])
        with FolderWriter(out) as w:  # default mode='append'
            w.write_episode(two_episodes[1])
        ds = FolderDataset(path=out, folder_keys=['pixels'])
        assert list(ds.lengths) == [5, 7]
        assert (out / 'pixels' / 'ep_0_step_0.jpeg').exists()
        assert (out / 'pixels' / 'ep_1_step_6.jpeg').exists()

    def test_error_mode_raises_for_existing_folder(
        self, tmp_path, two_episodes
    ):
        out = tmp_path / 'folder_ds'
        with FolderWriter(out) as w:
            w.write_episode(two_episodes[0])
        with pytest.raises(FileExistsError):
            FolderWriter(out, mode='error').__enter__()

    def test_overwrite_clears_existing_folder(self, tmp_path, two_episodes):
        out = tmp_path / 'folder_ds'
        with FolderWriter(out) as w:
            w.write_episode(two_episodes[0])
            w.write_episode(two_episodes[1])
        with FolderWriter(out, mode='overwrite') as w:
            w.write_episode(two_episodes[1])
        ds = FolderDataset(path=out, folder_keys=['pixels'])
        assert list(ds.lengths) == [7]
        # First-write image files from the discarded session must be gone.
        assert not (out / 'pixels' / 'ep_1_step_0.jpeg').exists()

    def test_append_schema_mismatch_raises(self, tmp_path, two_episodes):
        out = tmp_path / 'folder_ds'
        with FolderWriter(out) as w:
            w.write_episode(two_episodes[0])
        ep_extra = dict(two_episodes[1])
        ep_extra['unexpected'] = [
            np.zeros(2, np.float32) for _ in range(len(ep_extra['action']))
        ]
        with pytest.raises(ValueError, match='schema mismatch'):
            with FolderWriter(out) as w:
                w.write_episode(ep_extra)

    def test_append_per_step_shape_mismatch_raises(
        self, tmp_path, two_episodes
    ):
        out = tmp_path / 'folder_ds'
        with FolderWriter(out) as w:
            w.write_episode(two_episodes[0])
        ep_bad = {
            'action': [np.zeros(99, np.float32) for _ in range(3)],
            'proprio': [np.zeros(4, np.float32) for _ in range(3)],
            'pixels': [np.zeros((8, 8, 3), np.uint8) for _ in range(3)],
        }
        with pytest.raises(ValueError, match='shape mismatch'):
            with FolderWriter(out) as w:
                w.write_episode(ep_bad)

    def test_invalid_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match='write mode'):
            FolderWriter(tmp_path / 'folder_ds', mode='nope')


# ─── Video writer ─────────────────────────────────────────────────────────────


class TestVideoWriter:
    @pytest.fixture(autouse=True)
    def _skip_if_no_imageio(self):
        if importlib.util.find_spec('imageio') is None:
            pytest.skip('imageio not available')

    @staticmethod
    def _video_episode(n_steps: int, hw: int = 32) -> dict:
        rng = np.random.default_rng(0)
        return {
            'action': [
                rng.standard_normal(2).astype(np.float32)
                for _ in range(n_steps)
            ],
            'pixels': [
                rng.integers(0, 255, size=(hw, hw, 3), dtype=np.uint8)
                for _ in range(n_steps)
            ],
        }

    def test_writes_expected_layout(self, tmp_path):
        out = tmp_path / 'video_ds'
        eps = [self._video_episode(8), self._video_episode(10)]
        with VideoWriter(out, fps=25) as w:
            for ep in eps:
                w.write_episode(ep)

        assert (out / 'ep_len.npz').exists()
        assert (out / 'pixels' / 'ep_0.mp4').exists()
        assert (out / 'pixels' / 'ep_1.mp4').exists()
        assert (out / 'action.npz').exists()
        np.testing.assert_array_equal(
            np.load(out / 'ep_len.npz')['arr_0'], np.array([8, 10], np.int32)
        )

    def test_decord_roundtrip(self, tmp_path):
        if importlib.util.find_spec('decord') is None:
            pytest.skip('decord not available')

        out = tmp_path / 'video_ds'
        eps = [self._video_episode(8), self._video_episode(10)]
        with VideoWriter(out, fps=25) as w:
            for ep in eps:
                w.write_episode(ep)

        ds = VideoDataset(path=out, video_keys=['pixels'])
        assert list(ds.lengths) == [8, 10]
        ep0 = ds.load_episode(0)
        assert ep0['pixels'].shape[0] == 8
        assert ep0['pixels'].shape[1] == 3  # CHW after reader permute

    def test_append_extends_existing_video_dir(self, tmp_path):
        out = tmp_path / 'video_ds'
        with VideoWriter(out, fps=25) as w:
            w.write_episode(self._video_episode(8))
        with VideoWriter(out, fps=25) as w:  # default mode='append'
            w.write_episode(self._video_episode(10))
        np.testing.assert_array_equal(
            np.load(out / 'ep_len.npz')['arr_0'], np.array([8, 10], np.int32)
        )
        assert (out / 'pixels' / 'ep_0.mp4').exists()
        assert (out / 'pixels' / 'ep_1.mp4').exists()

    def test_error_mode_raises_for_existing_video_dir(self, tmp_path):
        out = tmp_path / 'video_ds'
        with VideoWriter(out, fps=25) as w:
            w.write_episode(self._video_episode(4))
        with pytest.raises(FileExistsError):
            VideoWriter(out, mode='error').__enter__()

    def test_overwrite_clears_existing_video_dir(self, tmp_path):
        out = tmp_path / 'video_ds'
        with VideoWriter(out, fps=25) as w:
            w.write_episode(self._video_episode(8))
            w.write_episode(self._video_episode(10))
        with VideoWriter(out, fps=25, mode='overwrite') as w:
            w.write_episode(self._video_episode(5))
        np.testing.assert_array_equal(
            np.load(out / 'ep_len.npz')['arr_0'], np.array([5], np.int32)
        )
        assert not (out / 'pixels' / 'ep_1.mp4').exists()

    def test_append_schema_mismatch_raises(self, tmp_path):
        out = tmp_path / 'video_ds'
        with VideoWriter(out, fps=25) as w:
            w.write_episode(self._video_episode(4))
        ep_extra = self._video_episode(4)
        ep_extra['unexpected'] = [np.zeros(2, np.float32) for _ in range(4)]
        with pytest.raises(ValueError, match='schema mismatch'):
            with VideoWriter(out, fps=25) as w:
                w.write_episode(ep_extra)

    def test_append_per_step_shape_mismatch_raises(self, tmp_path):
        out = tmp_path / 'video_ds'
        with VideoWriter(out, fps=25) as w:
            w.write_episode(self._video_episode(4))
        ep_bad = {
            'action': [np.zeros(99, np.float32) for _ in range(3)],
            'pixels': [np.zeros((32, 32, 3), np.uint8) for _ in range(3)],
        }
        with pytest.raises(ValueError, match='shape mismatch'):
            with VideoWriter(out, fps=25) as w:
                w.write_episode(ep_bad)

    def test_invalid_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match='write mode'):
            VideoWriter(tmp_path / 'video_ds', mode='nope')


# ─── Convert ──────────────────────────────────────────────────────────────────


class TestConvert:
    def test_hdf5_to_folder(self, tmp_path, two_episodes):
        # Seed source HDF5.
        src = tmp_path / 'src.h5'
        with HDF5Writer(src) as w:
            for ep in two_episodes:
                w.write_episode(ep)

        dst = tmp_path / 'folder_ds'
        convert(src, dst, dest_format='folder', progress=False)

        assert (dst / 'ep_len.npz').exists()
        assert (dst / 'pixels' / 'ep_0_step_0.jpeg').exists()

        ds = FolderDataset(path=dst, folder_keys=['pixels'])
        assert list(ds.lengths) == [5, 7]
        assert set(ds.column_names) == {'action', 'proprio', 'pixels'}

    def test_folder_to_hdf5(self, tmp_path, two_episodes):
        src = tmp_path / 'folder_ds'
        with FolderWriter(src) as w:
            for ep in two_episodes:
                w.write_episode(ep)

        dst = tmp_path / 'out.h5'
        convert(src, dst, dest_format='hdf5', progress=False)
        ds = HDF5Dataset(path=dst)
        assert list(ds.lengths) == [5, 7]

    def test_tabular_only_dataset(self, tmp_path):
        # No image columns — exercises the non-image branch of the
        # episode-shape adapter inside `convert`.
        rng = np.random.default_rng(0)
        eps = [
            {
                'action': [
                    rng.standard_normal(3).astype(np.float32) for _ in range(4)
                ],
                'reward': [
                    rng.standard_normal(1).astype(np.float32) for _ in range(4)
                ],
            }
            for _ in range(2)
        ]
        src = tmp_path / 'src.h5'
        with HDF5Writer(src) as w:
            for ep in eps:
                w.write_episode(ep)

        dst = tmp_path / 'dst.h5'
        convert(src, dst, dest_format='hdf5', progress=False)

        a = HDF5Dataset(path=src)
        b = HDF5Dataset(path=dst)
        assert list(a.lengths) == list(b.lengths)
        np.testing.assert_array_equal(
            a.get_col_data('action'), b.get_col_data('action')
        )
        np.testing.assert_array_equal(
            a.get_col_data('reward'), b.get_col_data('reward')
        )

    def test_hdf5_to_hdf5_identity(self, tmp_path, two_episodes):
        src = tmp_path / 'src.h5'
        with HDF5Writer(src) as w:
            for ep in two_episodes:
                w.write_episode(ep)

        dst = tmp_path / 'dst.h5'
        convert(src, dst, dest_format='hdf5', progress=False)

        a = HDF5Dataset(path=src)
        b = HDF5Dataset(path=dst)
        assert list(a.lengths) == list(b.lengths)
        assert set(a.column_names) == set(b.column_names)

        np.testing.assert_array_equal(
            a.get_col_data('action'), b.get_col_data('action')
        )
        np.testing.assert_array_equal(
            a.get_col_data('pixels'), b.get_col_data('pixels')
        )


# ─── load_dataset autodetect ──────────────────────────────────────────────────


class TestLoadDatasetAutodetect:
    def test_autodetects_hdf5(self, tmp_path, two_episodes):
        out = tmp_path / 'datasets' / 'data.h5'
        out.parent.mkdir()
        with HDF5Writer(out) as w:
            w.write_episode(two_episodes[0])
        ds = swm.data.load_dataset(str(out), cache_dir=str(tmp_path))
        assert isinstance(ds, HDF5Dataset)

    def test_autodetects_folder(self, tmp_path, two_episodes):
        out = tmp_path / 'datasets' / 'folder_ds'
        out.parent.mkdir()
        with FolderWriter(out) as w:
            w.write_episode(two_episodes[0])
        ds = swm.data.load_dataset(
            str(out), cache_dir=str(tmp_path), folder_keys=['pixels']
        )
        assert isinstance(ds, FolderDataset)

    def test_explicit_format_overrides_detection(self, tmp_path, two_episodes):
        # Write a folder, but force-load it as folder explicitly.
        out = tmp_path / 'datasets' / 'folder_ds'
        out.parent.mkdir()
        with FolderWriter(out) as w:
            w.write_episode(two_episodes[0])
        ds = swm.data.load_dataset(
            str(out),
            cache_dir=str(tmp_path),
            format='folder',
            folder_keys=['pixels'],
        )
        assert isinstance(ds, FolderDataset)

    def test_unknown_format_raises(self, tmp_path):
        unknown = tmp_path / 'datasets' / 'mystery.weird'
        unknown.parent.mkdir()
        unknown.touch()
        with pytest.raises(ValueError, match='No format detected'):
            swm.data.load_dataset(str(unknown), cache_dir=str(tmp_path))


# ─── World.collect with non-default format ────────────────────────────────────


class TestCollectFormat:
    def test_collect_to_folder(self, tmp_path):
        from stable_worldmodel import World
        from stable_worldmodel.policy import RandomPolicy

        world = World(
            env_name='swm/PushT-v1',
            num_envs=2,
            image_shape=(32, 32),
            max_episode_steps=10,
        )
        world.set_policy(RandomPolicy())

        out = tmp_path / 'folder_collected'
        world.collect(out, episodes=2, seed=0, format='folder')
        world.envs.close()

        assert (out / 'ep_len.npz').exists()
        assert (out / 'pixels').is_dir()
        # Reload via auto-detection.
        ds = swm.data.load_dataset(
            str(out), cache_dir=str(tmp_path), folder_keys=['pixels']
        )
        assert isinstance(ds, FolderDataset)
        assert len(ds.lengths) == 2

    def test_collect_unknown_format_raises(self, tmp_path):
        from stable_worldmodel import World
        from stable_worldmodel.policy import RandomPolicy

        world = World(
            env_name='swm/PushT-v1',
            num_envs=1,
            image_shape=(32, 32),
            max_episode_steps=5,
        )
        world.set_policy(RandomPolicy())
        try:
            with pytest.raises(ValueError, match='unknown format'):
                world.collect(tmp_path / 'x', episodes=1, format='nonexistent')
        finally:
            world.envs.close()
