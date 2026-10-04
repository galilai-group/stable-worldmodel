import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

import torch
from tqdm import tqdm

from stable_worldmodel.data import ensure_dir_exists, get_cache_dir
from stable_worldmodel.data.normalization import (
    deserialize_scaler,
    serialize_scaler,
)
from stable_worldmodel.utils import HF_BASE_URL

logger = logging.getLogger(__name__)

_PREPROCESSING_FILENAME = 'preprocessing.json'
_PREPROCESSING_VERSION = 1


def _serialize_preprocessing(process: dict) -> dict:
    return {
        'version': _PREPROCESSING_VERSION,
        'process': {
            key: serialize_scaler(scaler) for key, scaler in process.items()
        },
    }


def _deserialize_preprocessing(spec: dict) -> dict:
    version = spec.get('version')
    if version != _PREPROCESSING_VERSION:
        raise ValueError(
            f'Unsupported preprocessing version: {version!r}. '
            f'Expected {_PREPROCESSING_VERSION}.'
        )
    process = spec.get('process')
    if not isinstance(process, dict):
        raise TypeError(
            "Serialized preprocessing must contain a 'process' mapping."
        )
    return {key: deserialize_scaler(value) for key, value in process.items()}


def save_pretrained(
    model: torch.nn.Module,
    run_name: str,
    config: dict | None = None,
    config_key: str | None = None,
    filename: str = 'weights.pt',
    cache_dir: str = None,
    preprocessing: dict | None = None,
):
    from omegaconf import OmegaConf

    ckpt_dir = get_cache_dir(cache_dir, sub_folder='checkpoints') / run_name
    ensure_dir_exists(ckpt_dir)

    checkpoint_path = ckpt_dir / filename
    torch.save(model.state_dict(), checkpoint_path)

    if preprocessing is not None:
        preprocessing_path = ckpt_dir / _PREPROCESSING_FILENAME
        with open(preprocessing_path, 'w') as f:
            json.dump(_serialize_preprocessing(preprocessing), f, indent=2)

    if config is None:
        logger.warning('No config! Loading will have to be done manually.')
        return

    if config_key is not None and config_key in config:
        config = config[config_key]

    config_path = ckpt_dir / 'config.json'

    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f'📦📦📦 Model saved to {checkpoint_path} 📦📦📦')

    return


def load_pretrained(name: str, cache_dir: str = None, extra_args=None):
    """Load a model from a local checkpoint or a HuggingFace repository.

    Supported formats for `name`:

    1. **`.pt` file** — path to a specific checkpoint file.
       A `config.json` must live in the same directory.

        ```python
        model = load_pretrained('my_run/weights_epoch_10.pt')
        ```

    2. **Folder** — path to a directory containing exactly one `.pt` file
       and a `config.json`.

        ```python
        model = load_pretrained('my_run/')
        ```

    3. **HuggingFace repo** (`<user>/<repo>`) — loaded from the local cache
       if already present, otherwise fetched from HF.

        ```python
        model = load_pretrained('nice-user/my-worldmodel')
        ```

    All local paths are resolved relative to `<cache_dir>/checkpoints/`.
    """
    from hydra.utils import instantiate

    cache_dir = get_cache_dir(cache_dir, sub_folder='checkpoints')
    ensure_dir_exists(cache_dir)
    checkpoint_path, config = _resolve(name, cache_dir)
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    # assume keys with the dotted notation
    if extra_args is not None:
        for key, value in extra_args.items():
            parts = key.split('.')
            d = config
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value

    model = instantiate(config)
    model.load_state_dict(state_dict)
    return model


def load_preprocessing(name: str, cache_dir: str | None = None) -> dict | None:
    """Load fitted preprocessing saved alongside a checkpoint.

    Returns ``None`` for legacy checkpoints that do not contain a
    ``preprocessing.json`` artifact.
    """
    cache_dir = get_cache_dir(cache_dir, sub_folder='checkpoints')
    ensure_dir_exists(cache_dir)
    checkpoint_path, _ = _resolve(name, cache_dir)
    preprocessing_path = checkpoint_path.parent / _PREPROCESSING_FILENAME

    hf_cache_name = f'models--{name.replace("/", "--")}'
    if (
        not preprocessing_path.exists()
        and checkpoint_path.parent.name == hf_cache_name
    ):
        url = f'{HF_BASE_URL}/{name}/resolve/main/{_PREPROCESSING_FILENAME}'
        _download_optional(url, preprocessing_path)

    if not preprocessing_path.exists():
        return None

    with open(preprocessing_path) as f:
        spec = json.load(f)
    return _deserialize_preprocessing(spec)


def _resolve(name: str, cache_dir: Path) -> tuple[Path, dict]:
    """Return ``(checkpoint_path, config_dict)`` for *name*.

    Resolution order:
      1. ``<cache_dir>/<name>``  as a ``.pt`` file
      2. ``<cache_dir>/<name>``  as a folder
      3. HuggingFace repo (cached locally under ``<cache_dir>/<user>/<repo>/``)
    """
    local = cache_dir / name

    # format 1: explicit .pt file
    if local.suffix == '.pt':
        if not local.exists():
            raise FileNotFoundError(f'Checkpoint not found: {local}')
        return local, _load_config(local.parent)

    # format 2: folder containing a .pt and config.json
    # (skip if it has no .pt — likely a sibling output dir, e.g. eval videos —
    # and fall through to HF resolution when name looks like a repo id)
    if local.is_dir() and list(local.glob('*.pt')):
        return _resolve_folder(local)

    # format 3: HuggingFace repo (<user>/<repo>)
    if '/' in name:
        return _resolve_hf(name, cache_dir)

    raise ValueError(
        f"Cannot resolve '{name}': not a .pt file, a folder, or a HF repo id."
    )


def _resolve_folder(folder: Path) -> tuple[Path, dict]:
    """Load from a folder containing one ``.pt`` file and a ``config.json``."""
    pt_files = list(folder.glob('*.pt'))
    if not pt_files:
        raise FileNotFoundError(f'No .pt file found in {folder}')
    if len(pt_files) > 1:
        raise ValueError(
            f'Ambiguous checkpoint: multiple .pt files in {folder}. '
            'Specify the file directly.'
        )
    logger.info(f'Loading checkpoint from folder {folder}...')
    return pt_files[0], _load_config(folder)


def _resolve_hf(repo_id: str, cache_dir: Path) -> tuple[Path, dict]:
    """Resolve a HuggingFace repo id, using a local cache when available.

    Local layout: ``<cache_dir>/models--<user>--<repo>/``
    """
    local_dir = cache_dir / f'models--{repo_id.replace("/", "--")}'

    if local_dir.is_dir():
        logger.info(f'Loading {repo_id} from local cache...')
        return _resolve_folder(local_dir)

    logger.info(f'Downloading {repo_id} from HuggingFace...')
    local_dir.mkdir(parents=True, exist_ok=True)
    for filename in ('config.json', 'weights.pt'):
        url = f'{HF_BASE_URL}/{repo_id}/resolve/main/{filename}'
        dest = local_dir / filename
        logger.info(f'Fetching {url}')
        _download(url, dest)

    return _resolve_folder(local_dir)


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a tqdm progress bar."""
    response = urllib.request.urlopen(url)
    total = int(response.headers.get('Content-Length', 0)) or None
    with (
        open(dest, 'wb') as f,
        tqdm(total=total, unit='B', unit_scale=True, desc=dest.name) as bar,
    ):
        while chunk := response.read(8192):
            f.write(chunk)
            bar.update(len(chunk))


def _download_optional(url: str, dest: Path) -> bool:
    try:
        _download(url, dest)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        dest.unlink(missing_ok=True)
        return False
    return True


def _load_config(folder: Path) -> dict:
    config_path = folder / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'config.json not found in {folder}')
    with open(config_path) as f:
        return json.load(f)


__all__ = ['load_preprocessing', 'load_pretrained', 'save_pretrained']
