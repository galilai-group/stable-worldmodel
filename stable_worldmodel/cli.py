"""Stable World Model CLI commands."""

from importlib.metadata import version as pkg_version

from typing import Annotated

import typer
from rich import print
from rich.table import Table


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


def _entry_size(path) -> int:
    """Bytes on disk for a dataset entry — a file's own size, or the recursive
    sum of every file under a dataset directory (Lance tables, MP4 blobs, npz)."""
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


# Pretty labels for the registry format keys. Anything not listed falls back to
# the registry name itself, so a newly-registered format still lists cleanly.
_FORMAT_LABELS = {
    'lance': 'Lance',
    'lance_video': 'Lance Video',
    'hdf5': 'HDF5',
    'video': 'Video',
    'folder': 'Folder',
    'lerobot': 'LeRobot',
}


def _format_label(path, fmt) -> str:
    """Human-readable label for a detected format. The ``folder`` format keeps
    its Image/Folder nuance (it has no separate registry entry)."""
    if fmt.name == 'folder':
        return _detect_folder_format(path)
    return _FORMAT_LABELS.get(fmt.name, fmt.name)


def _iter_datasets(cache_dir):
    """Yield ``(path, format_cls)`` for every cache entry a registered format
    recognizes. Detection is delegated to :func:`detect_format` so the listing
    stays in lockstep with the format registry — every format, present and
    future, is discovered the same way instead of via per-format globs here."""
    from stable_worldmodel.data import detect_format

    if not cache_dir.is_dir():
        return
    for entry in sorted(cache_dir.iterdir()):
        fmt = detect_format(entry)
        if fmt is not None:
            yield entry, fmt


def _arrow_scalar_dtype(t) -> str:
    """Map a pyarrow scalar/value type to a numpy-style dtype label."""
    import pyarrow as pa

    _MAP = {
        pa.float16(): 'float16',
        pa.float32(): 'float32',
        pa.float64(): 'float64',
        pa.int8(): 'int8',
        pa.int16(): 'int16',
        pa.int32(): 'int32',
        pa.int64(): 'int64',
        pa.uint8(): 'uint8',
        pa.uint16(): 'uint16',
        pa.uint32(): 'uint32',
        pa.uint64(): 'uint64',
        pa.bool_(): 'bool',
    }
    return _MAP.get(t, str(t))


def _sample_video_frame_shapes(videos_uri, video_keys) -> dict:
    """Decode the first frame of one MP4 blob per video key to learn its
    ``(C, H, W)`` shape — the shape the lance_video reader yields per step.

    Best-effort: any key whose blob can't be read or decoded (e.g. torchcodec
    missing) maps to ``None`` so the caller falls back to an ellipsis shape.
    """
    shapes = {k: None for k in video_keys}
    try:
        import lance
        from torchcodec.decoders import VideoDecoder
    except Exception:
        return shapes

    try:
        vds = lance.dataset(videos_uri)
        keys = (
            vds.to_table(columns=['video_key']).column('video_key').to_pylist()
        )
    except Exception:
        return shapes

    # First physical row holding each key — matches how the reader maps
    # (episode, key) to a blob row offset.
    first_row: dict[str, int] = {}
    for i, k in enumerate(keys):
        first_row.setdefault(str(k), i)

    for vkey in video_keys:
        row = first_row.get(vkey)
        if row is None:
            continue
        try:
            blob = vds.take_blobs(blob_column='video_bytes', indices=[row])[0]
            try:
                data = blob.readall()
            finally:
                blob.close()
            frames = (
                VideoDecoder(data, seek_mode='approximate')
                .get_frames_at(indices=[0])
                .data
            )  # (1, C, H, W)
            shapes[vkey] = tuple(int(d) for d in frames.shape[1:])
        except Exception:
            shapes[vkey] = None
    return shapes


def _resolve_lance_table(path):
    """Return the ``.lance`` frames table for a lance/lance_video dataset.

    ``path`` may be the ``.lance`` table itself (flat layout) or a dataset
    directory holding one ``<name>.lance`` frames table (plus, for lance_video,
    a ``<name>_videos.lance`` sibling which is excluded here). Mirrors the
    readers' directory-based resolution.
    """
    from pathlib import Path

    p = Path(path)
    if p.suffix == '.lance':
        return p
    cands = [
        t for t in sorted(p.glob('*.lance')) if not t.stem.endswith('_videos')
    ]
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        raise ValueError(
            f'Ambiguous Lance dataset in {p}: {[c.name for c in cands]}.'
        )
    raise FileNotFoundError(f'No Lance frames table found in {p}.')


def _inspect_lance_dataset(path) -> None:
    import io
    from pathlib import Path

    import numpy as np
    import pyarrow as pa
    import lancedb

    path = Path(path)
    table_path = _resolve_lance_table(path)
    parent, stem = table_path.parent, table_path.stem
    db = lancedb.connect(str(parent) or '.')
    table = db.open_table(stem)
    schema = table.schema
    lance_tbl = table.to_lance()
    ep_arr = (
        lance_tbl.to_table(columns=['episode_idx']).column('episode_idx')
    ).to_numpy()
    n_steps = len(ep_arr)
    if n_steps:
        n_episodes = int(ep_arr[-1]) + 1
        ep_lengths = [int((ep_arr == i).sum()) for i in range(n_episodes)]
    else:
        n_episodes = 0
        ep_lengths = []

    # Binary columns in the frames table hold one JPEG per frame (the plain
    # `lance` format). Decode a single sample to recover the image shape.
    binary_cols = [
        f.name
        for f in schema
        if pa.types.is_binary(f.type) or pa.types.is_large_binary(f.type)
    ]
    binary_shapes: dict[str, tuple | None] = {}
    if binary_cols and n_steps > 0:
        sample = lance_tbl.to_table(columns=binary_cols).slice(0, 1)
        for col in binary_cols:
            try:
                from PIL import Image

                raw = sample.column(col)[0].as_py()
                binary_shapes[col] = np.array(
                    Image.open(io.BytesIO(raw))
                ).shape
            except Exception:
                binary_shapes[col] = None

    # A `_videos` sibling means this is the lance_video layout: image columns
    # live there as MP4 blobs, not in the frames schema above. Decode a frame
    # per key to report its (C, H, W) shape.
    videos_name = f'{stem}_videos'
    is_video = videos_name in db.list_tables().tables
    video_shapes: dict[str, tuple | None] = {}
    if is_video:
        vkeys = sorted(
            {
                str(v)
                for v in db.open_table(videos_name)
                .to_lance()
                .to_table(columns=['video_key'])
                .column('video_key')
                .to_pylist()
            }
        )
        video_shapes = _sample_video_frame_shapes(
            f'{parent}/{videos_name}.lance', vkeys
        )

    size = _format_size(_entry_size(path))
    print(f'[bold]Name:[/bold]     {path.stem}')
    print(f'[bold]Format:[/bold]   {"Lance Video" if is_video else "Lance"}')
    print(f'[bold]Path:[/bold]     {path}')
    print(f'[bold]Size:[/bold]     {size}')
    print(f'[bold]Episodes:[/bold] {n_episodes}')
    print(f'[bold]Steps:[/bold]    {n_steps}')
    if ep_lengths:
        print(f'[bold]Ep length:[/bold] {min(ep_lengths)} – {max(ep_lengths)}')

    cols = Table(title='Columns')
    cols.add_column('Column', style='cyan', no_wrap=True)
    cols.add_column('Shape', style='yellow')
    cols.add_column('Dtype', style='magenta')

    for f in schema:
        if f.name in ('episode_idx', 'step_idx'):
            continue
        t = f.type
        if pa.types.is_fixed_size_list(t):
            list_size = t.list_size
            dtype = _arrow_scalar_dtype(t.value_type)
            shape = (
                f'({n_steps},)'
                if list_size == 1
                else f'({n_steps}, {list_size})'
            )
        elif pa.types.is_binary(t) or pa.types.is_large_binary(t):
            img_shape = binary_shapes.get(f.name)
            if img_shape:
                inner = ', '.join(str(d) for d in img_shape)
                shape = f'({n_steps}, {inner})'
            else:
                shape = f'({n_steps}, ...)'
            dtype = 'uint8'
        else:
            shape = f'({n_steps},)'
            dtype = _arrow_scalar_dtype(t)
        cols.add_row(f.name, shape, dtype)

    for vkey in sorted(video_shapes):
        frame_shape = video_shapes[vkey]
        if frame_shape:
            inner = ', '.join(str(d) for d in frame_shape)
            shape = f'({n_steps}, {inner})'
        else:
            shape = f'({n_steps}, ...)'
        cols.add_row(vkey, shape, 'uint8')

    print(cols)


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

    cache_dir = get_cache_dir(sub_folder='datasets')
    table = Table(title=f'Datasets in {cache_dir}')
    table.add_column('Name', justify='left', style='cyan', no_wrap=True)
    table.add_column('Format', justify='left', style='magenta')
    table.add_column('Size', justify='right', style='yellow')

    rows = [
        (path.stem, _format_label(path, fmt), _format_size(_entry_size(path)))
        for path, fmt in _iter_datasets(cache_dir)
    ]

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
    from stable_worldmodel.data import detect_format
    from stable_worldmodel.data.utils import get_cache_dir

    cache_dir = get_cache_dir(sub_folder='datasets')
    # A dataset is addressed by name; its on-disk entry is either a directory
    # (lance/lance_video/folder/video) or a file (hdf5). Try each candidate and
    # let the registry say what it is, so inspect covers the same formats the
    # listing does.
    candidates = [
        cache_dir / name,
        cache_dir / f'{name}.lance',
        cache_dir / f'{name}.h5',
        cache_dir / f'{name}.hdf5',
    ]
    for path in candidates:
        if not path.exists():
            continue
        fmt = detect_format(path)
        if fmt is None:
            continue
        if fmt.name in ('lance', 'lance_video'):
            _inspect_lance_dataset(path)
        elif fmt.name == 'hdf5':
            _inspect_hdf5_dataset(path)
        elif fmt.name in ('folder', 'video'):
            _inspect_folder_dataset(path)
        else:
            print(f'[bold]Name:[/bold]   {path.stem}')
            print(f'[bold]Format:[/bold] {_format_label(path, fmt)}')
            print(f'[bold]Path:[/bold]   {path}')
            print(f'[bold]Size:[/bold]   {_format_size(_entry_size(path))}')
        return

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

    from stable_worldmodel.envs import DISCRETE_WORLDS, WORLDS

    continuous = sorted(WORLDS - DISCRETE_WORLDS)
    discrete = sorted(DISCRETE_WORLDS)

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

    from stable_worldmodel.envs import WORLDS

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
def convert(
    name: Annotated[str, typer.Argument(help='Source dataset name.')],
    output: Annotated[
        str | None,
        typer.Argument(
            help='Output dataset name. Defaults to <name>-<dest-format>.',
            show_default=False,
        ),
    ] = None,
    dest_format: Annotated[
        str,
        typer.Option('--dest-format', '-f', help='Destination format.'),
    ] = 'video',
    source_format: Annotated[
        str | None,
        typer.Option(
            '--source-format', help='Force source format (skip detection).'
        ),
    ] = None,
):
    """Convert a dataset to another format (e.g. HDF5 → video)."""
    from stable_worldmodel.data import convert as data_convert
    from stable_worldmodel.data import detect_format
    from stable_worldmodel.data.utils import get_cache_dir

    cache_dir = get_cache_dir(sub_folder='datasets')

    # Resolve the source by name through the format registry, mirroring
    # `inspect`/`datasets`, so every format (lance, lance_video, folder, video,
    # hdf5) is found the same way instead of via per-format path heuristics.
    candidates = [
        cache_dir / name,
        cache_dir / f'{name}.lance',
        cache_dir / f'{name}.h5',
        cache_dir / f'{name}.hdf5',
    ]
    source_path = None
    for path in candidates:
        if path.exists() and detect_format(path) is not None:
            source_path = path
            break
    if source_path is None:
        print(f'[red]Dataset not found: {name}[/red]')
        print('Run [cyan]swm datasets[/cyan] to see available datasets.')
        raise typer.Exit(1)

    dest_name = output if output is not None else f'{name}-{dest_format}'
    dest_path = cache_dir / dest_name

    print(
        f'Converting [cyan]{name}[/cyan] → [magenta]{dest_format}[/magenta] as [cyan]{dest_name}[/cyan]'
    )
    data_convert(
        source_path,
        dest_path,
        dest_format=dest_format,
        source_format=source_format,
    )
    print(f'[green]Done.[/green] Output: {dest_path}')


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

    cache_dir = get_cache_dir(sub_folder='checkpoints')
    table = Table(title=f'Checkpoints in {cache_dir}')
    table.add_column('Run', justify='left', style='cyan', no_wrap=True)
    table.add_column('Checkpoint', justify='left', style='magenta')

    def _ckpt_name(p):
        return p.stem

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
    root_files = sorted(cache_dir.glob('*.pt'), key=_by_mtime)
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
        ckpt_files = sorted(folder.glob('*.pt'), key=_by_mtime)
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
