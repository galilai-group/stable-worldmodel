import json
import urllib.request
from pathlib import Path
import numpy as np
import torch

from loguru import logger as logging
from tqdm import tqdm

from stable_worldmodel.utils import HF_BASE_URL
from stable_worldmodel.data import get_cache_dir, ensure_dir_exists


class ZScoreNormalizer(torch.nn.Module):
    """Picklable z-score normaliser for use in dataset transforms.

    The training scripts previously closed over ``mean``/``std`` from a local
    function.  That works under the default ``fork`` multiprocessing start
    method (child inherits memory), but breaks under ``spawn`` — which
    ``LanceDataset`` forces on Linux, and which DDP also requires — because
    local closures are not picklable.

    Subclassing ``torch.nn.Module`` and storing the stats as buffers gives us
    both picklability and automatic device movement.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer('mean', mean.clone())
        self.register_buffer('std', std.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / self.std).float()


def column_normalizer(dataset, source: str, target: str):
    """Build a picklable column normaliser wrapped for use as a dataset transform.

    Computes z-score stats once over ``dataset.get_col_data(source)`` (skipping
    any rows with NaN), and returns a ``WrapTorchTransform`` that applies them.
    Replaces the closure-based pattern previously repeated across the training
    scripts, which broke under ``spawn`` DataLoader workers.
    """
    from stable_pretraining.data.transforms import WrapTorchTransform

    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.asarray(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    normaliser = ZScoreNormalizer(
        mean=data.mean(0, keepdim=True),
        std=data.std(0, keepdim=True),
    )
    return WrapTorchTransform(normaliser, source=source, target=target)


def save_pretrained(
    model: torch.nn.Module,
    run_name: str,
    config: dict | None = None,
    config_key: str | None = None,
    filename: str = 'weights.pt',
    cache_dir: str = None,
):
    from omegaconf import OmegaConf

    ckpt_dir = get_cache_dir(cache_dir, sub_folder='checkpoints') / run_name
    ensure_dir_exists(ckpt_dir)

    checkpoint_path = ckpt_dir / filename
    torch.save(model.state_dict(), checkpoint_path)

    if config is None:
        logging.warning('No config! Loading will have to be done manually.')
        return

    if config_key is not None and config_key in config:
        config = config[config_key]

    config_path = ckpt_dir / 'config.json'

    config = OmegaConf.to_container(config, resolve=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logging.info(f'📦📦📦 Model saved to {checkpoint_path} 📦📦📦')

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
    if local.is_dir():
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
    logging.info(f'Loading checkpoint from folder {folder}...')
    return pt_files[0], _load_config(folder)


def _resolve_hf(repo_id: str, cache_dir: Path) -> tuple[Path, dict]:
    """Resolve a HuggingFace repo id, using a local cache when available.

    Local layout: ``<cache_dir>/models--<user>--<repo>/``
    """
    local_dir = cache_dir / f'models--{repo_id.replace("/", "--")}'

    if local_dir.is_dir():
        logging.info(f'Loading {repo_id} from local cache...')
        return _resolve_folder(local_dir)

    logging.info(f'Downloading {repo_id} from HuggingFace...')
    local_dir.mkdir(parents=True, exist_ok=True)
    for filename in ('config.json', 'weights.pt'):
        url = f'{HF_BASE_URL}/{repo_id}/resolve/main/{filename}'
        dest = local_dir / filename
        logging.info(f'Fetching {url}')
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


def _load_config(folder: Path) -> dict:
    config_path = folder / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'config.json not found in {folder}')
    with open(config_path) as f:
        return json.load(f)


__all__ = ['load_pretrained', 'save_pretrained']
