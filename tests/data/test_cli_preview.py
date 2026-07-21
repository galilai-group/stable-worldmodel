"""``swm preview`` — sample random episodes from a dataset and render them to
MP4 for quick inspection, without converting the whole dataset.

These tests reuse the format builders / fixture from ``test_cli_datasets`` (which
write tiny datasets in each format's real on-disk layout) and drive the command
through Typer's ``CliRunner``, asserting the expected ``ep<idx>.mp4`` files land
in the output directory.
"""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from stable_worldmodel.cli import app
from stable_worldmodel.data import list_formats

from test_cli_datasets import _episode, _needs


runner = CliRunner()


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """Point the CLI's cache lookup at a throwaway dir and return its
    ``datasets`` subfolder (mirrors the fixture in ``test_cli_datasets``)."""
    monkeypatch.setenv('STABLEWM_HOME', str(tmp_path))
    datasets = tmp_path / 'datasets'
    datasets.mkdir()
    return datasets


def _build_multi_episode(datasets_dir, name, n_eps):
    """A lance dataset with ``n_eps`` episodes (each carries a ``pixels``
    image column), for sampling / reproducibility assertions."""
    _needs('lancedb')
    from stable_worldmodel.data.formats.lance import LanceWriter

    with LanceWriter(datasets_dir / name / f'{name}.lance') as w:
        for i in range(n_eps):
            w.write_episode(_episode(4 + i))


def _build_multiview(datasets_dir, name):
    """A lance dataset whose episodes carry two image columns
    (``pixels`` + ``pixels_top``) to exercise the side-by-side panel path."""
    _needs('lancedb')
    from stable_worldmodel.data.formats.lance import LanceWriter

    rng = np.random.default_rng(0)

    def ep(n):
        base = _episode(n)
        base['pixels_top'] = [
            rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
            for _ in range(n)
        ]
        return base

    with LanceWriter(datasets_dir / name / f'{name}.lance') as w:
        w.write_episode(ep(4))
        w.write_episode(ep(5))


def _mp4_names(directory):
    return {p.name for p in directory.glob('*.mp4')}


@pytest.fixture(autouse=True)
def _require_imageio():
    _needs('imageio')


def test_preview_writes_mp4_per_episode(cache_root):
    if 'lance' not in list_formats():
        pytest.skip('lance format not registered')
    _build_multi_episode(cache_root, 'ds', n_eps=2)
    out = cache_root.parent / 'out'

    result = runner.invoke(
        app, ['preview', 'ds', '-n', '2', '--seed', '0', '-o', str(out)]
    )
    assert result.exit_code == 0, result.output
    assert _mp4_names(out) == {'ep0.mp4', 'ep1.mp4'}


def test_preview_seed_reproducible(cache_root):
    if 'lance' not in list_formats():
        pytest.skip('lance format not registered')
    _build_multi_episode(cache_root, 'ds', n_eps=8)
    out_a = cache_root.parent / 'a'
    out_b = cache_root.parent / 'b'
    out_c = cache_root.parent / 'c'

    common = ['preview', 'ds', '-n', '3']
    r_a = runner.invoke(app, [*common, '--seed', '7', '-o', str(out_a)])
    r_b = runner.invoke(app, [*common, '--seed', '7', '-o', str(out_b)])
    assert r_a.exit_code == 0 and r_b.exit_code == 0, r_a.output + r_b.output
    assert _mp4_names(out_a) == _mp4_names(out_b)
    assert len(_mp4_names(out_a)) == 3

    # A different seed is allowed to (and here does) pick a different subset.
    r_c = runner.invoke(app, [*common, '--seed', '1', '-o', str(out_c)])
    assert r_c.exit_code == 0, r_c.output
    assert _mp4_names(out_c) != _mp4_names(out_a)


def test_preview_explicit_episodes(cache_root):
    if 'lance' not in list_formats():
        pytest.skip('lance format not registered')
    _build_multi_episode(cache_root, 'ds', n_eps=5)
    out = cache_root.parent / 'out'

    result = runner.invoke(
        app, ['preview', 'ds', '--episodes', '0,3', '-o', str(out)]
    )
    assert result.exit_code == 0, result.output
    assert _mp4_names(out) == {'ep0.mp4', 'ep3.mp4'}


def test_preview_explicit_episodes_out_of_range(cache_root):
    if 'lance' not in list_formats():
        pytest.skip('lance format not registered')
    _build_multi_episode(cache_root, 'ds', n_eps=2)

    result = runner.invoke(app, ['preview', 'ds', '--episodes', '99'])
    assert result.exit_code == 1
    assert 'out of range' in result.output.lower()


def test_preview_num_exceeds_count(cache_root):
    if 'lance' not in list_formats():
        pytest.skip('lance format not registered')
    _build_multi_episode(cache_root, 'ds', n_eps=2)
    out = cache_root.parent / 'out'

    result = runner.invoke(app, ['preview', 'ds', '-n', '999', '-o', str(out)])
    assert result.exit_code == 0, result.output
    assert _mp4_names(out) == {'ep0.mp4', 'ep1.mp4'}
    assert 'sampling all' in result.output.lower()


def test_preview_default_output_location(cache_root):
    """Without -o, videos land under <cache>/previews/<name>."""
    if 'lance' not in list_formats():
        pytest.skip('lance format not registered')
    _build_multi_episode(cache_root, 'ds', n_eps=2)

    result = runner.invoke(app, ['preview', 'ds', '--episodes', '0'])
    assert result.exit_code == 0, result.output
    out = cache_root.parent / 'previews' / 'ds'
    assert _mp4_names(out) == {'ep0.mp4'}


def test_preview_multiview_single_file_per_episode(cache_root):
    """Multiple image columns compose into one ep<idx>.mp4, not one per view."""
    if 'lance' not in list_formats():
        pytest.skip('lance format not registered')
    _build_multiview(cache_root, 'ds')
    out = cache_root.parent / 'out'

    result = runner.invoke(
        app, ['preview', 'ds', '--episodes', '0,1', '-o', str(out)]
    )
    assert result.exit_code == 0, result.output
    assert _mp4_names(out) == {'ep0.mp4', 'ep1.mp4'}


def test_preview_key_filter(cache_root):
    if 'lance' not in list_formats():
        pytest.skip('lance format not registered')
    _build_multiview(cache_root, 'ds')
    out = cache_root.parent / 'out'

    result = runner.invoke(
        app,
        [
            'preview',
            'ds',
            '--episodes',
            '0',
            '--key',
            'pixels',
            '-o',
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert _mp4_names(out) == {'ep0.mp4'}


def test_preview_key_filter_no_match(cache_root):
    if 'lance' not in list_formats():
        pytest.skip('lance format not registered')
    _build_multi_episode(cache_root, 'ds', n_eps=2)

    result = runner.invoke(
        app, ['preview', 'ds', '--episodes', '0', '--key', 'nonexistent']
    )
    assert result.exit_code == 1
    assert 'no videos written' in result.output.lower()


def test_preview_missing_dataset_errors(cache_root):
    result = runner.invoke(app, ['preview', 'does_not_exist'])
    assert result.exit_code == 1
    assert 'not found' in result.output.lower()
