import json
import os
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
    happens after this in :func:`load_dataset`. Local layout for cached
    datasets is a directory under ``<datasets_dir>/<name>/``; the directory
    may hold a ``foo.lance/`` table, a ``foo.h5`` file, or any other
    layout a registered format can detect.
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


# Suffixes we recognise on HF: a `.lance` directory (preferred) or a
# `.h5` / `.hdf5` file. Each format is downloaded in its native shape — no
# tar/zst wrapping.
_HF_FILE_SUFFIXES: tuple[str, ...] = ('.h5', '.hdf5')
_HF_DIR_SUFFIXES: tuple[str, ...] = ('.lance',)


def _hf_list_tree(repo_id: str, sub_path: str = '') -> list[dict]:
    """One HF API call: list entries at ``<repo>/tree/main/<sub_path>``."""
    suffix = f'/{sub_path}' if sub_path else ''
    api_url = f'{HF_BASE_URL}/api/datasets/{repo_id}/tree/main{suffix}'
    with urllib.request.urlopen(api_url) as resp:
        return json.loads(resp.read())


def _hf_find_dataset_entry(repo_id: str) -> dict:
    """Return the first top-level entry that looks like a dataset.

    Preference: a ``*.lance`` directory wins over an ``*.h5`` file when
    both are present, since lance is the default format.
    """
    entries = _hf_list_tree(repo_id)

    for entry in entries:
        path = entry.get('path', '')
        if entry.get('type') == 'directory' and path.endswith(
            _HF_DIR_SUFFIXES
        ):
            return entry
    for entry in entries:
        path = entry.get('path', '')
        if entry.get('type') == 'file' and path.endswith(_HF_FILE_SUFFIXES):
            return entry

    raise FileNotFoundError(
        f'No dataset found in HF repo {repo_id}: expected a top-level '
        f'`*.lance` directory or `*.h5`/`*.hdf5` file.'
    )


def _hf_walk_files(repo_id: str, sub_path: str) -> list[str]:
    """Recursively list every *file* path under ``sub_path`` on HF."""
    out: list[str] = []
    stack = [sub_path]
    while stack:
        current = stack.pop()
        for entry in _hf_list_tree(repo_id, current):
            path = entry.get('path', '')
            if entry.get('type') == 'directory':
                stack.append(path)
            else:
                out.append(path)
    return out


def _resolve_dataset_hf(repo_id: str, datasets_dir: Path) -> Path:
    """Resolve a HF repo id, downloading on first use.

    Local layout: ``<datasets_dir>/<user>--<repo>/`` — the directory is
    returned as-is and format detection picks up whatever lives inside
    (``*.lance``, ``*.h5``, …).
    """
    local_dir = datasets_dir / repo_id.replace('/', '--')

    if local_dir.is_dir() and any(local_dir.iterdir()):
        logging.info(f'Using cached dataset for {repo_id} at {local_dir}')
        return local_dir

    logging.info(f'Downloading dataset {repo_id} from HuggingFace...')
    local_dir.mkdir(parents=True, exist_ok=True)

    entry = _hf_find_dataset_entry(repo_id)
    entry_path = entry['path']

    if entry.get('type') == 'directory':
        files = _hf_walk_files(repo_id, entry_path)
        if not files:
            raise FileNotFoundError(
                f"HF repo {repo_id}: directory '{entry_path}' is empty."
            )
        for remote in tqdm(files, desc=f'Fetching {entry_path}'):
            url = f'{HF_BASE_URL}/datasets/{repo_id}/resolve/main/{remote}'
            dest = local_dir / remote
            dest.parent.mkdir(parents=True, exist_ok=True)
            _download(url, dest)
    else:
        url = f'{HF_BASE_URL}/datasets/{repo_id}/resolve/main/{entry_path}'
        dest = local_dir / entry_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f'Fetching {url}')
        _download(url, dest)

    return local_dir


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


def convert(
    source,
    dest,
    *,
    source_format: str | None = None,
    dest_format: str = 'lance',
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
        dest_format: Registered writer name (default ``'lance'``).
        cache_dir: Forwarded to the source loader for HF/local resolution.
        progress: Show a progress bar over episodes.
        **dest_kwargs: Forwarded to the destination writer.

    Example::

        from stable_worldmodel.data import convert
        convert('data.lance', 'data_video', dest_format='video')
    """
    from stable_worldmodel.data.format import get_format

    src = load_dataset(source, cache_dir=cache_dir, format=source_format)
    writer_cls = get_format(dest_format)

    iterator = range(len(src.lengths))
    if progress:
        iterator = tqdm(iterator, desc=f'Converting → {dest_format}')

    def episodes():
        for ep_idx in iterator:
            ep = src.load_episode(ep_idx)
            yield _episode_to_step_lists(ep, int(src.lengths[ep_idx]))

    with writer_cls.open_writer(dest, **dest_kwargs) as writer:
        writer.write_episodes(episodes())


def _episode_to_step_lists(ep: dict, ep_len: int) -> dict[str, list]:
    """Adapt an episode dict from a reader to the ``{col: [step_arr, ...]}``
    shape that writers consume.

    Specifically:
      - Tensors → NumPy arrays.
      - Image arrays in ``(N, C, H, W)`` are transposed back to ``(N, H, W, C)``.
      - Image arrays in float dtypes (e.g. LeRobot's ``ToTensor``-normalised
        ``[0, 1]`` floats) are rescaled to ``uint8 [0, 255]`` so downstream
        writers (Lance JPEG encode, Video MP4 encode, HDF5 fixed-dtype
        datasets) receive a consistent display-range integer image.
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

        # Float image → uint8. LeRobot's ToTensor pipeline produces float32
        # in [0, 1]; HDF5 / Lance / Video tworoom-style readers all assume
        # uint8 HxWxC. Detect by shape (3D HWC or 4D NHWC with 1/3 channels)
        # and float dtype, then clip-and-scale.
        if (
            arr.dtype.kind == 'f'
            and arr.ndim in (3, 4)
            and arr.shape[-1] in (1, 3)
        ):
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)

        out[col] = list(arr)
    return out


class ZScoreNormalizer:
    """Picklable z-score normalizer: returns ``((x - mean) / std).float()``.

    Lives at module scope (not a closure) so it survives pickle. Required
    when the dataset is consumed by a DataLoader using the ``spawn`` start
    method — which LanceDataset forces on Linux for fork-safety.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def column_normalizer(dataset, source: str, target: str):
    """Build a per-column z-score :class:`WrapTorchTransform` from dataset
    stats. Picklable end-to-end (uses :class:`ZScoreNormalizer` rather
    than a local closure)."""
    # Lazy import — stable_pretraining is a training-only dep.
    from stable_pretraining.data.transforms import WrapTorchTransform

    data = torch.from_numpy(np.array(dataset.get_col_data(source)))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return WrapTorchTransform(
        ZScoreNormalizer(mean, std), source=source, target=target
    )


__all__ = [
    'load_dataset',
    'convert',
    'get_cache_dir',
    'ensure_dir_exists',
    'ZScoreNormalizer',
    'column_normalizer',
]
