"""Stable World Model CLI commands."""

from importlib.metadata import version as pkg_version

from typing import Annotated

import typer
from rich import print
from rich.table import Table

from stable_worldmodel.envs import WORLDS


app = typer.Typer()


def _version_callback(value: bool):
    if value:
        typer.echo(
            f'stable-worldmodel version: {pkg_version("stable-worldmodel")}'
        )
        raise typer.Exit()


def _detect_folder_format(folder) -> str:
    for sub in sorted(folder.iterdir()):
        if sub.is_dir():
            if any(sub.glob('*.mp4')):
                return 'Video'
            if any(sub.glob('*.jpeg')) or any(sub.glob('*.jpg')):
                return 'Image'
    return 'Folder'


def _format_size(n_bytes: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n_bytes < 1024:
            return f'{n_bytes:.1f} {unit}'
        n_bytes /= 1024
    return f'{n_bytes:.1f} PB'


def _inspect_hdf5_dataset(path) -> None:
    import h5py

    with h5py.File(path, 'r') as f:
        ep_len = f['ep_len'][:]
        columns = {
            k: (f[k].shape, str(f[k].dtype))
            for k in sorted(f.keys())
            if k not in ('ep_len', 'ep_offset')
        }

    size = _format_size(path.stat().st_size)
    print(f'[bold]Name:[/bold]     {path.stem}')
    print('[bold]Format:[/bold]   HDF5')
    print(f'[bold]Path:[/bold]     {path}')
    print(f'[bold]Size:[/bold]     {size}')
    print(f'[bold]Episodes:[/bold] {len(ep_len)}')
    print(f'[bold]Steps:[/bold]    {int(ep_len.sum())}')
    print(f'[bold]Ep length:[/bold] {int(ep_len.min())} – {int(ep_len.max())}')

    table = Table(title='Columns')
    table.add_column('Column', style='cyan', no_wrap=True)
    table.add_column('Shape', style='yellow')
    table.add_column('Dtype', style='magenta')
    for col, (shape, dtype) in columns.items():
        table.add_row(col, str(shape), dtype)
    print(table)


def _inspect_folder_dataset(path) -> None:
    import numpy as np

    ep_len = np.load(path / 'ep_len.npz')['arr_0']
    fmt = _detect_folder_format(path)
    npz_size = sum(p.stat().st_size for p in path.glob('*.npz'))

    print(f'[bold]Name:[/bold]     {path.name}')
    print(f'[bold]Format:[/bold]   {fmt}')
    print(f'[bold]Path:[/bold]     {path}')
    print(f'[bold]Size:[/bold]     {_format_size(npz_size)} (metadata only)')
    print(f'[bold]Episodes:[/bold] {len(ep_len)}')
    print(f'[bold]Steps:[/bold]    {int(ep_len.sum())}')
    print(f'[bold]Ep length:[/bold] {int(ep_len.min())} – {int(ep_len.max())}')

    table = Table(title='Columns')
    table.add_column('Column', style='cyan', no_wrap=True)
    table.add_column('Shape', style='yellow')
    table.add_column('Dtype', style='magenta')

    for p in sorted(path.iterdir()):
        if p.suffix == '.npz' and p.stem not in ('ep_len', 'ep_offset'):
            arr = np.load(p)['arr_0']
            table.add_row(p.stem, str(arr.shape), str(arr.dtype))

    for p in sorted(path.iterdir()):
        if p.is_dir():
            table.add_row(p.name, '(folder)', 'image/video')

    print(table)


def _format_space(space) -> tuple[str, str, str]:
    """Return (type_label, range_str, init_str) for a leaf space."""
    from stable_worldmodel import spaces as swm_spaces

    init = space.init_value if hasattr(space, 'init_value') else None
    init_str = str(init) if init is not None else '-'

    if isinstance(space, swm_spaces.RGBBox):
        return 'RGBBox', '[0,255]^3', init_str
    if isinstance(space, swm_spaces.Box):
        low = space.low.flat[0] if space.low.size == 1 else space.low.tolist()
        high = (
            space.high.flat[0] if space.high.size == 1 else space.high.tolist()
        )
        shape = '' if space.shape == () else f' shape={list(space.shape)}'
        return 'Box', f'[{low}, {high}]{shape}', init_str
    if isinstance(space, swm_spaces.Discrete):
        end = space.start + space.n - 1
        return 'Discrete', f'[{space.start}, {end}]', init_str
    return type(space).__name__, '-', init_str


def _get_space_at_path(variation_space, dotted_path: str):
    space = variation_space
    for part in dotted_path.split('.'):
        space = space.spaces[part]
    return space


@app.command()
def datasets():
    """List all datasets in the cache directory."""
    from stable_worldmodel.data.utils import get_cache_dir

    cache_dir = get_cache_dir()
    table = Table(title=f'Datasets in {cache_dir}')
    table.add_column('Name', justify='left', style='cyan', no_wrap=True)
    table.add_column('Format', justify='left', style='magenta')
    table.add_column('Size', justify='right', style='yellow')

    rows = []

    for h5_path in sorted(cache_dir.glob('*.h5')):
        size = _format_size(h5_path.stat().st_size)
        rows.append((h5_path.stem, 'HDF5', size))

    for folder in sorted(cache_dir.iterdir()):
        if not folder.is_dir() or not (folder / 'ep_len.npz').exists():
            continue
        npz_size = sum(p.stat().st_size for p in folder.glob('*.npz'))
        rows.append(
            (
                folder.name,
                _detect_folder_format(folder),
                _format_size(npz_size),
            )
        )

    if not rows:
        print(f'No datasets found in {cache_dir}')
    else:
        for row in rows:
            table.add_row(*row)
        print(table)


@app.command()
def inspect(
    name: Annotated[str, typer.Argument(help='Dataset name to inspect.')],
):
    """Show detailed info for a dataset."""
    from stable_worldmodel.data.utils import get_cache_dir

    cache_dir = get_cache_dir()
    h5_path = cache_dir / f'{name}.h5'
    folder_path = cache_dir / name

    if h5_path.exists():
        _inspect_hdf5_dataset(h5_path)
    elif folder_path.is_dir() and (folder_path / 'ep_len.npz').exists():
        _inspect_folder_dataset(folder_path)
    else:
        print(f'[red]Dataset not found: {name}[/red]')
        print('Run [cyan]swm datasets[/cyan] to see available datasets.')
        raise typer.Exit(1)


@app.command()
def envs():
    """List all registered environments."""
    table = Table(title='Registered SWM Environments')
    table.add_column(
        'Environment ID', justify='left', style='cyan', no_wrap=True
    )
    table.add_column('Type', justify='left', style='magenta', no_wrap=True)

    continuous = sorted(e for e in WORLDS if 'Discrete' not in e)
    discrete = sorted(e for e in WORLDS if 'Discrete' in e)

    for env_id in continuous:
        table.add_row(env_id, 'Continuous')
    if discrete:
        table.add_section()
        for env_id in discrete:
            table.add_row(env_id, 'Discrete')

    print(table)


@app.command()
def fovs(
    env: Annotated[
        str, typer.Argument(help='Environment ID (e.g. PushT-v1).')
    ],
):
    """List factors of variation for the given environment."""
    import gymnasium as gym

    if '/' not in env:
        env = f'swm/{env}'

    if env not in WORLDS:
        print(f'[red]Unknown environment: {env}[/red]')
        print('Run [cyan]swm envs[/cyan] to see available environments.')
        raise typer.Exit(1)

    try:
        environment = gym.make(env)
        unwrapped = environment.unwrapped
    except Exception as e:
        print(f'[red]Failed to instantiate {env}: {e}[/red]')
        raise typer.Exit(1)

    if not hasattr(unwrapped, 'variation_space'):
        print(f'[yellow]{env} has no variation_space.[/yellow]')
        raise typer.Exit()

    vs = unwrapped.variation_space
    names = vs.names()

    table = Table(title=f'Factors of Variation — {env}')
    table.add_column('Factor', style='cyan', no_wrap=True)
    table.add_column('Type', style='magenta')
    table.add_column('Range', style='yellow')
    table.add_column('Default', style='green')

    for name in names:
        space = _get_space_at_path(vs, name)
        type_label, range_str, init_str = _format_space(space)
        table.add_row(name, type_label, range_str, init_str)

    print(table)
    environment.close()


@app.command()
def checkpoints(
    filter: Annotated[
        str | None,
        typer.Argument(
            help='Optional substring to filter by run or checkpoint name.',
            show_default=False,
        ),
    ] = None,
):
    """List model checkpoints available in the cache directory."""
    from stable_worldmodel.data.utils import get_cache_dir

    cache_dir = get_cache_dir()
    table = Table(title=f'Checkpoints in {cache_dir}')
    table.add_column('Run', justify='left', style='cyan', no_wrap=True)
    table.add_column('Checkpoint', justify='left', style='magenta')

    def _ckpt_name(p):
        return p.stem.removesuffix('_object')

    def _by_mtime(p):
        return p.stat().st_mtime

    groups: list[tuple[str, list[str]]] = []

    import re

    pattern = re.compile(filter) if filter else None

    def _matches(run: str, ckpt: str) -> bool:
        if pattern is None:
            return True
        return bool(pattern.search(ckpt) or pattern.search(run))

    # Root-level checkpoints (directly in cache_dir)
    root_files = sorted(cache_dir.glob('*_object.ckpt'), key=_by_mtime)
    if root_files:
        names = [
            _ckpt_name(p) for p in root_files if _matches('', _ckpt_name(p))
        ]
        if names:
            groups.append(('', names))

    # Per-directory checkpoints
    for folder in sorted(cache_dir.iterdir()):
        if not folder.is_dir():
            continue
        ckpt_files = sorted(folder.glob('*_object.ckpt'), key=_by_mtime)
        if not ckpt_files:
            continue
        run_name = folder.name
        names = [
            _ckpt_name(p)
            for p in ckpt_files
            if _matches(run_name, _ckpt_name(p))
        ]
        if not names:
            continue
        groups.append((run_name, names))

    if not groups:
        msg = f'No checkpoints found in {cache_dir}'
        if filter:
            msg += f' matching pattern [bold]{filter}[/bold]'
        print(msg)
    else:
        first = True
        for run_name, ckpt_names in groups:
            if not first:
                table.add_section()
            first = False
            for i, ckpt in enumerate(ckpt_names):
                table.add_row(run_name if i == 0 else '', ckpt)
        print(table)


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            '--version',
            '-v',
            callback=_version_callback,
            is_eager=True,
            help='Show installed version.',
        ),
    ] = None,
):
    """Stable World Model - World Model Research Made Simple."""


if __name__ == '__main__':
    app()
