import json
import os
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import torch
from loguru import logger as logging
from tqdm import tqdm

from stable_worldmodel.utils import DEFAULT_CACHE_DIR, HF_BASE_URL


def get_cache_dir(
    override_root: Path | None = None,
    sub_folder: str | None = None,
) -> Path:
    base = override_root
    if override_root is None:
        base = os.getenv('STABLEWM_HOME', str(DEFAULT_CACHE_DIR))

    cache_path = (
        Path(base, sub_folder) if sub_folder is not None else Path(base)
    )

    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def ensure_dir_exists(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def load_dataset(
    name: str,
    cache_dir: str = None,
    format: str | None = None,
    **kwargs,
):
    """Resolve a dataset name to a local path and dispatch to the matching
    format reader from the registry.

    Supported names:

    1. **Local path** — file or directory.
    2. **HuggingFace repo** (``<user>/<repo>``) — downloaded and cached under
       ``<cache_dir>/datasets/<user>--<repo>/``.
    3. **Format scheme** (e.g. ``lerobot://lerobot/pusht``) — passed through
       to the matching format unchanged.

    The format is auto-detected via :func:`detect_format` unless ``format`` is
    provided explicitly. To register a new format, decorate a
    :class:`~stable_worldmodel.data.format.Format` subclass with
    :func:`~stable_worldmodel.data.format.register_format`.

    Args:
        name: Local path, HF repo id, or scheme-prefixed identifier.
        cache_dir: Root cache directory. Defaults to ``STABLEWM_HOME`` or
            ``~/.stable_worldmodel``.
        format: Explicit format name (skips detection).
        **kwargs: Forwarded to the format's reader.

    Returns:
        A reader instance (typically a
        :class:`~stable_worldmodel.data.dataset.Dataset` subclass).
    """
    from stable_worldmodel.data.format import (
        FORMATS,
        detect_format,
        get_format,
    )

    name = str(name)

    # Scheme-prefixed identifiers (e.g. lerobot://...) bypass path resolution.
    if '://' in name:
        if format is None:
            for fmt in FORMATS.values():
                if fmt.detect(name):
                    return fmt.open_reader(name, **kwargs)
            raise ValueError(f'No format detected for {name!r}')
        return get_format(format).open_reader(name, **kwargs)

    datasets_dir = get_cache_dir(cache_dir, sub_folder='datasets')
    ensure_dir_exists(datasets_dir)
    path = _resolve_dataset(name, datasets_dir)

    if format is not None:
        return get_format(format).open_reader(path, **kwargs)

    fmt = detect_format(path)
    if fmt is None:
        raise ValueError(
            f'No format detected for {path!r}; pass format= explicitly.'
        )
    return fmt.open_reader(path, **kwargs)


def _resolve_dataset(name: str, datasets_dir: Path) -> Path:
    """Resolve *name* (local path or HF repo id) to a local path.

    Returns whatever exists on disk — file or directory. Format detection
    happens after this in :func:`load_dataset`.
    """
    local = Path(name)
    if not local.is_absolute():
        local = datasets_dir / local

    if local.exists():
        return local

    # HuggingFace repo: <user>/<repo>
    if '/' in name and not name.startswith(('.', '/')):
        return _resolve_dataset_hf(name, datasets_dir)

    raise FileNotFoundError(
        f'Cannot resolve {name!r}: not a local path or HF repo id.'
    )


def _resolve_dataset_folder(folder: Path) -> Path:
    """Return the single HDF5 file inside *folder*."""
    h5_files = list(folder.glob('*.h5')) + list(folder.glob('*.hdf5'))
    if not h5_files:
        raise FileNotFoundError(f'No .h5 / .hdf5 file found in {folder}')
    if len(h5_files) > 1:
        raise ValueError(
            f'Ambiguous dataset: multiple HDF5 files in {folder}. '
            'Specify the file directly.'
        )
    logging.info(f'Using dataset at {h5_files[0]}')
    return h5_files[0]


def _hf_dataset_find_archive(repo_id: str) -> str:
    """Return the filename of the first .h5.zst or .tar.zst in a HF dataset repo."""
    api_url = f'{HF_BASE_URL}/api/datasets/{repo_id}/tree/main'
    with urllib.request.urlopen(api_url) as resp:
        entries = json.loads(resp.read())
    for entry in entries:
        name = entry.get('path', '')
        if name.endswith('.h5.zst') or name.endswith('.tar.zst'):
            return name
    raise FileNotFoundError(
        f'No .h5.zst or .tar.zst file found in HF dataset repo {repo_id}'
    )


def _resolve_dataset_hf(repo_id: str, datasets_dir: Path) -> Path:
    """Resolve a HF repo id, downloading and extracting when not cached.

    Local layout: ``<datasets_dir>/<user>--<repo>/dataset.h5``
    The archive fetched from HF must be a ``.h5.zst`` or ``.tar.zst`` file.
    """
    local_dir = datasets_dir / repo_id.replace('/', '--')

    if local_dir.is_dir():
        h5_files = list(local_dir.glob('*.h5')) + list(
            local_dir.glob('*.hdf5')
        )
        if h5_files:
            logging.info(f'Using cached dataset for {repo_id} at {local_dir}')
            return _resolve_dataset_folder(local_dir)

    logging.info(f'Downloading dataset {repo_id} from HuggingFace...')
    local_dir.mkdir(parents=True, exist_ok=True)

    archive_name = _hf_dataset_find_archive(repo_id)
    url = f'{HF_BASE_URL}/datasets/{repo_id}/resolve/main/{archive_name}'
    archive_path = local_dir / archive_name

    logging.info(f'Fetching {url}')
    _download(url, archive_path)

    logging.info(f'Extracting {archive_path} into {local_dir}')
    if archive_name.endswith('.tar.zst'):
        _extract_zst_tar(archive_path, local_dir)
    else:
        _extract_zst(archive_path)
    archive_path.unlink()

    return _resolve_dataset_folder(local_dir)


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a tqdm progress bar."""
    response = urllib.request.urlopen(url)
    total = int(response.headers.get('Content-Length', 0)) or None
    with (
        open(dest, 'wb') as f,
        tqdm(total=total, unit='B', unit_scale=True, desc=dest.name) as bar,
    ):
        chunk = response.read(8192)
        while chunk:
            f.write(chunk)
            bar.update(len(chunk))
            chunk = response.read(8192)


def _extract_zst_tar(archive: Path, dest: Path) -> None:
    """Extract a ``.tar.zst`` archive into *dest* using the system ``tar`` command."""
    result = subprocess.run(
        [
            'tar',
            '--use-compress-program=unzstd',
            '-xf',
            str(archive),
            '-C',
            str(dest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'Failed to extract {archive}:\n{result.stderr.strip()}'
        )


def _extract_zst(archive: Path) -> None:
    """Decompress a plain ``.zst`` file in-place using ``unzstd``."""
    result = subprocess.run(
        ['unzstd', str(archive), '-o', str(archive.with_suffix(''))],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'Failed to decompress {archive}:\n{result.stderr.strip()}'
        )


def convert(
    source,
    dest,
    *,
    source_format: str | None = None,
    dest_format: str = 'hdf5',
    cache_dir: str | None = None,
    progress: bool = True,
    **dest_kwargs,
) -> None:
    """Convert a dataset from one registered format to another.

    Reads each episode from *source* and writes it through the writer of
    *dest_format*. Format detection follows the same rules as
    :func:`load_dataset` — autodetect by default, or pass ``source_format``
    explicitly.

    Args:
        source: Path or identifier accepted by :func:`load_dataset`.
        dest: Output path for the destination writer.
        source_format: Force a source format (skips detection).
        dest_format: Registered writer name (default ``'hdf5'``).
        cache_dir: Forwarded to the source loader for HF/local resolution.
        progress: Show a progress bar over episodes.
        **dest_kwargs: Forwarded to the destination writer.

    Example::

        from stable_worldmodel.data import convert
        convert('data.h5', 'data_video/', dest_format='video', fps=30)
    """
    from stable_worldmodel.data.format import get_format

    src = load_dataset(source, cache_dir=cache_dir, format=source_format)
    writer_cls = get_format(dest_format)

    iterator = range(len(src.lengths))
    if progress:
        iterator = tqdm(iterator, desc=f'Converting → {dest_format}')

    with writer_cls.open_writer(dest, **dest_kwargs) as writer:
        for ep_idx in iterator:
            ep = src.load_episode(ep_idx)
            writer.write_episode(
                _episode_to_step_lists(ep, int(src.lengths[ep_idx]))
            )


def _episode_to_step_lists(ep: dict, ep_len: int) -> dict[str, list]:
    """Adapt an episode dict from a reader to the ``{col: [step_arr, ...]}``
    shape that writers consume.

    Specifically:
      - Tensors → NumPy arrays.
      - Image arrays in ``(N, C, H, W)`` are transposed back to ``(N, H, W, C)``.
      - Scalars (e.g. flattened string columns) are repeated ``ep_len`` times.
    """
    out: dict[str, list] = {}
    for col, val in ep.items():
        if isinstance(val, torch.Tensor):
            arr = val.detach().cpu().numpy()
        elif isinstance(val, np.ndarray):
            arr = val
        else:
            out[col] = [val] * ep_len
            continue

        if arr.ndim == 4 and arr.shape[1] in (1, 3):
            arr = arr.transpose(0, 2, 3, 1)
        out[col] = list(arr)
    return out


__all__ = ['load_dataset', 'convert', 'get_cache_dir', 'ensure_dir_exists']
