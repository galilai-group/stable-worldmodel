import json
import os
import subprocess
import urllib.request
from pathlib import Path

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


def load_dataset(name: str, cache_dir: str = None, **kwargs):
    """Resolve a dataset and return an instantiated :class:`HDF5Dataset`.

    Supported formats for `name`:

    1. **Local path** — absolute or relative path to a ``.h5`` / ``.hdf5`` file
       or a directory containing exactly one such file.

    2. **HuggingFace repo** (``<user>/<repo>``) — downloaded from HF and cached
       under ``<cache_dir>/datasets/<user>--<repo>/``.  The repo must expose a
       single ``dataset.tar.zst`` archive containing one HDF5 file.

    Args:
        name: Local path or HF repo id (``user/repo``).
        cache_dir: Root cache directory.  Defaults to ``STABLEWM_HOME`` or
            ``~/.stable_worldmodel``.
        **kwargs: Extra arguments passed to ``HDF5Dataset``.

    Returns:
        An :class:`~stable_worldmodel.data.dataset.HDF5Dataset` instance.

    Example::

        ds = load_dataset('my-org/my-robot-data', num_steps=4, frameskip=2)
    """
    from stable_worldmodel.data.dataset import HDF5Dataset

    datasets_dir = get_cache_dir(cache_dir, sub_folder='datasets')
    ensure_dir_exists(datasets_dir)
    h5_path = _resolve_dataset(name, datasets_dir)
    rel_name = str(h5_path.relative_to(datasets_dir).with_suffix(''))
    return HDF5Dataset(name=rel_name, cache_dir=cache_dir, **kwargs)


def _resolve_dataset(name: str, datasets_dir: Path) -> Path:
    local = Path(name)
    if not local.is_absolute():
        local = datasets_dir / local

    # format 1a: explicit HDF5 file path
    if local.suffix in ('.h5', '.hdf5'):
        if local.exists():
            return local
        raise FileNotFoundError(f'Dataset file not found: {local}')

    # format 1b: directory containing a single HDF5 file
    if local.is_dir():
        return _resolve_dataset_folder(local)

    # format 2: HuggingFace repo (<user>/<repo>)
    if '/' in name:
        return _resolve_dataset_hf(name, datasets_dir)

    raise ValueError(
        f"Cannot resolve '{name}': not a .h5 file, a folder, or a HF repo id."
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


__all__ = ['load_dataset', 'get_cache_dir', 'ensure_dir_exists']
