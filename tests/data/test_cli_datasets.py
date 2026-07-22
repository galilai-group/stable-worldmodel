"""CLI dataset discovery: ``swm datasets`` / ``swm inspect`` must list every
on-disk format the registry knows how to write.

These commands used to hand-roll their own globbing, which silently dropped any
format whose layout the globs didn't anticipate (e.g. a ``lance_video`` dataset
written into a per-shard subdirectory). They now delegate to
:func:`detect_format`, so this test writes a tiny dataset in each writable
format's real on-disk layout and asserts both commands see it. A new format that
forgets to wire up detection — or a layout the CLI can't address by name — fails
here instead of vanishing from the listing.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from typer.testing import CliRunner

from stable_worldmodel.cli import app
from stable_worldmodel.data import detect_format, list_formats


runner = CliRunner()


def _episode(n_steps: int) -> dict:
    rng = np.random.default_rng(0)
    return {
        'action': [
            rng.standard_normal(2).astype(np.float32) for _ in range(n_steps)
        ],
        'proprio': [
            rng.standard_normal(4).astype(np.float32) for _ in range(n_steps)
        ],
        'pixels': [
            rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
            for _ in range(n_steps)
        ],
    }


def _needs(modname: str) -> None:
    if importlib.util.find_spec(modname) is None:
        pytest.skip(f'{modname} not available')


# Each builder writes a two-episode dataset named ``name`` into ``datasets_dir``
# using that format's real on-disk layout (the same shapes the writers and
# World.collect produce), and returns the expected listed format label. The
# lance/lance_video builders deliberately nest the table inside a per-dataset
# directory — the exact layout that used to slip past the listing.


def _build_hdf5(datasets_dir, name):
    _needs('h5py')
    from stable_worldmodel.data.formats.hdf5 import HDF5Writer

    with HDF5Writer(datasets_dir / f'{name}.h5') as w:
        for ep in (_episode(4), _episode(5)):
            w.write_episode(ep)
    return 'HDF5'


def _build_folder(datasets_dir, name):
    from stable_worldmodel.data.formats.folder import FolderWriter

    with FolderWriter(datasets_dir / name) as w:
        for ep in (_episode(4), _episode(5)):
            w.write_episode(ep)
    # Episodes carry a `pixels` image column stored as JPEGs, so the folder
    # format refines its label to 'Image' (the Folder/Image nuance the CLI
    # keeps for npz-folder datasets).
    return 'Image'


def _build_video(datasets_dir, name):
    _needs('imageio')
    from stable_worldmodel.data.formats.video import VideoWriter

    with VideoWriter(datasets_dir / name, fps=10) as w:
        for ep in (_episode(4), _episode(5)):
            w.write_episode(ep)
    return 'Video'


def _build_lance(datasets_dir, name):
    _needs('lancedb')
    from stable_worldmodel.data.formats.lance import LanceWriter

    with LanceWriter(datasets_dir / name / f'{name}.lance') as w:
        for ep in (_episode(4), _episode(5)):
            w.write_episode(ep)
    return 'Lance'


def _build_lance_video(datasets_dir, name):
    _needs('lancedb')
    _needs('imageio')
    from stable_worldmodel.data.formats.lance_video import LanceVideoWriter

    with LanceVideoWriter(datasets_dir / name / f'{name}.lance', fps=10) as w:
        for ep in (_episode(4), _episode(5)):
            w.write_episode(ep)
    return 'Lance Video'


_BUILDERS = {
    'hdf5': _build_hdf5,
    'folder': _build_folder,
    'video': _build_video,
    'lance': _build_lance,
    'lance_video': _build_lance_video,
}


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """Point the CLI's cache lookup at a throwaway dir and return its
    ``datasets`` subfolder (created on demand by the writers)."""
    monkeypatch.setenv('STABLEWM_HOME', str(tmp_path))
    datasets = tmp_path / 'datasets'
    datasets.mkdir()
    return datasets


@pytest.mark.parametrize('fmt_name', sorted(_BUILDERS))
def test_datasets_lists_every_format(fmt_name, cache_root):
    """A dataset written in each format is discovered and labeled correctly."""
    if fmt_name not in list_formats():
        pytest.skip(f'format {fmt_name!r} not registered')
    name = f'ds_{fmt_name}'
    expected_label = _BUILDERS[fmt_name](cache_root, name)

    result = runner.invoke(app, ['datasets'])
    assert result.exit_code == 0, result.output
    assert 'No datasets found' not in result.output
    # The name can wrap across lines in the rendered table, so match on a
    # stable, unbroken substring rather than the full name.
    assert fmt_name in result.output
    assert expected_label in result.output


@pytest.mark.parametrize('fmt_name', sorted(_BUILDERS))
def test_inspect_finds_every_format(fmt_name, cache_root):
    """``swm inspect <name>`` resolves and describes each format by name."""
    if fmt_name not in list_formats():
        pytest.skip(f'format {fmt_name!r} not registered')
    name = f'ds_{fmt_name}'
    expected_label = _BUILDERS[fmt_name](cache_root, name)

    result = runner.invoke(app, ['inspect', name])
    assert result.exit_code == 0, result.output
    assert 'Dataset not found' not in result.output
    assert expected_label in result.output


@pytest.mark.parametrize('fmt_name', sorted(_BUILDERS))
def test_convert_finds_every_format(fmt_name, cache_root):
    """``swm convert <name>`` must resolve a source written in any format by
    name. Like the listing, this used to hand-roll path heuristics (probing for
    a top-level ``_versions``/``ep_len.npz`` sentinel) that didn't match the
    ``lance_video`` layout — where ``_versions`` lives inside nested ``.lance``
    subtables — so convert reported 'Dataset not found' for datasets that
    ``swm datasets`` happily listed. It now resolves the source via
    detect_format, so a layout the CLI can't address by name fails here.

    Scope is source resolution only: we assert the command gets past the
    'Dataset not found' guard into the conversion stage. Whether a particular
    source→dest round-trip then succeeds is the concern of
    ``test_convert_video_formats.py``, not this test."""
    if fmt_name not in list_formats():
        pytest.skip(f'format {fmt_name!r} not registered')
    _needs('imageio')
    name = f'ds_{fmt_name}'
    _BUILDERS[fmt_name](cache_root, name)

    result = runner.invoke(
        app, ['convert', name, f'{name}_out', '-f', 'video']
    )
    assert 'Dataset not found' not in result.output
    assert f'Converting {name}' in result.output


def test_convert_missing_dataset_errors(cache_root):
    result = runner.invoke(app, ['convert', 'does_not_exist', 'out'])
    assert result.exit_code == 1
    assert 'not found' in result.output.lower()


def test_merge_combines_shards(cache_root):
    """``swm merge`` concatenates two shards into a single dataset that the
    listing and inspect then see, with the summed episode/step counts."""
    _needs('lancedb')
    _build_lance(cache_root, 'shard0')  # 2 episodes (4 + 5 steps)
    _build_lance(cache_root, 'shard1')  # 2 episodes (4 + 5 steps)

    result = runner.invoke(
        app, ['merge', 'shard0', 'shard1', '-o', 'combined']
    )
    assert result.exit_code == 0, result.output
    assert 'Done' in result.output

    # The merged dataset resolves by name and reports the combined totals:
    # 4 episodes, (4 + 5) * 2 = 18 steps.
    inspect = runner.invoke(app, ['inspect', 'combined'])
    assert inspect.exit_code == 0, inspect.output
    assert 'Dataset not found' not in inspect.output
    assert 'Lance' in inspect.output
    assert 'Episodes: 4' in inspect.output
    assert 'Steps:    18' in inspect.output


def test_merge_missing_source_errors(cache_root):
    _needs('lancedb')
    _build_lance(cache_root, 'shard0')
    result = runner.invoke(
        app, ['merge', 'shard0', 'does_not_exist', '-o', 'combined']
    )
    assert result.exit_code == 1
    assert 'not found' in result.output.lower()


def test_merge_requires_two_sources(cache_root):
    _needs('lancedb')
    _build_lance(cache_root, 'shard0')
    result = runner.invoke(app, ['merge', 'shard0', '-o', 'combined'])
    assert result.exit_code == 1
    assert 'at least two' in result.output.lower()


def test_merge_existing_output_errors_then_overwrite(cache_root):
    """Default mode refuses to clobber an existing output; --overwrite replaces
    it."""
    _needs('lancedb')
    _build_lance(cache_root, 'shard0')
    _build_lance(cache_root, 'shard1')

    first = runner.invoke(app, ['merge', 'shard0', 'shard1', '-o', 'combined'])
    assert first.exit_code == 0, first.output

    again = runner.invoke(app, ['merge', 'shard0', 'shard1', '-o', 'combined'])
    assert again.exit_code == 1
    assert 'already exists' in again.output.lower()

    forced = runner.invoke(
        app, ['merge', 'shard0', 'shard1', '-o', 'combined', '--overwrite']
    )
    assert forced.exit_code == 0, forced.output
    assert 'Done' in forced.output


def test_detect_format_recognizes_every_layout(cache_root):
    """Guard the CLI's foundation directly: detect_format must classify each
    written layout, so the listing built on it can never silently drop one."""
    for fmt_name, build in sorted(_BUILDERS.items()):
        if fmt_name not in list_formats():
            continue
        name = f'probe_{fmt_name}'
        build(cache_root, name)
        # The entry as it appears in the cache dir (dir for most, .h5 file for
        # hdf5) is what `swm datasets` iterates over.
        entry = next(
            p for p in cache_root.iterdir() if p.name.startswith(name)
        )
        fmt = detect_format(entry)
        assert fmt is not None, f'{fmt_name}: {entry} not detected'
        assert fmt.name == fmt_name


@pytest.mark.parametrize('fmt_name', ['lance', 'lance_video'])
def test_inspect_reports_column_shapes(fmt_name, cache_root):
    """Lance inspect reports a per-column (n_steps, ...) shape and dtype, not
    just the raw Arrow type — including decoded image/video frame dims."""
    if fmt_name not in list_formats():
        pytest.skip(f'format {fmt_name!r} not registered')
    name = f'ds_{fmt_name}'
    _BUILDERS[fmt_name](cache_root, name)  # 4 + 5 = 9 steps

    result = runner.invoke(app, ['inspect', name])
    assert result.exit_code == 0, result.output
    # Tabular column: fixed_size_list<float32>[4] over 9 steps.
    assert '(9, 4)' in result.output  # proprio
    assert 'float32' in result.output
    # Image/video column decoded to a uint8 frame shape (16x16x3 episodes).
    assert 'uint8' in result.output
    # The lance_video layout stores frames as MP4 blobs, so recovering the
    # decoded frame dims requires torchcodec; without it inspect falls back to
    # an ellipsis shape. The plain lance layout decodes JPEGs via PIL instead.
    if fmt_name == 'lance_video':
        try:
            from torchcodec.decoders import VideoDecoder  # noqa: F401
        except Exception as exc:
            pytest.skip(f'torchcodec unavailable ({exc})')
    assert '16' in result.output


def test_datasets_empty_cache_reports_none(cache_root):
    result = runner.invoke(app, ['datasets'])
    assert result.exit_code == 0, result.output
    assert 'No datasets found' in result.output


def test_inspect_missing_dataset_errors(cache_root):
    result = runner.invoke(app, ['inspect', 'does_not_exist'])
    assert result.exit_code == 1
    assert 'not found' in result.output.lower()
