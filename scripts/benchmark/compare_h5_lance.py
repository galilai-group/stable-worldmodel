"""Benchmark HDF5 vs Lance datasets across local and S3 sources.

Configure via Hydra — edit configs/benchmark.yaml or pass CLI overrides:

  python compare_h5_lance.py benchmark.batch_size=32
  python compare_h5_lance.py --config-name=my_run

Each entry under ``datasets:`` in the config becomes one benchmark row
(two rows when ``keys_to_cache`` is set: one no-cache, one cached).
For S3 datasets set ``source: s3`` and provide ``aws_region`` (plus
``aws_access_key_id`` / ``aws_secret_access_key`` if not using an IAM role).
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from stable_worldmodel.data import HDF5Dataset, LanceDataset
from stable_worldmodel.data.utils import get_cache_dir

try:
    from stable_worldmodel.data import VideoDataset
except ImportError:
    VideoDataset = None


# ---- Storage-options helpers ------------------------------------------------


def _lance_storage_opts(ds_cfg: DictConfig) -> dict:
    region = ds_cfg.get('aws_region', 'us-east-2')
    opts: dict = {'region': region, 'virtual_hosted_style_request': 'true'}
    if ds_cfg.get('aws_access_key_id'):
        opts['aws_access_key_id'] = ds_cfg.aws_access_key_id
        opts['aws_secret_access_key'] = ds_cfg.aws_secret_access_key
    return opts


def _hdf5_storage_opts(ds_cfg: DictConfig) -> dict:
    region = ds_cfg.get('aws_region', 'us-east-2')
    opts: dict = {'client_kwargs': {'region_name': region}}
    if ds_cfg.get('aws_access_key_id'):
        opts['key'] = ds_cfg.aws_access_key_id
        opts['secret'] = ds_cfg.aws_secret_access_key
    return opts


# ---- Size measurement -------------------------------------------------------


def _local_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())


def _fmt_bytes(n: int) -> str:
    if n <= 0:
        return '—'
    for unit, thresh in [
        ('TB', 1 << 40),
        ('GB', 1 << 30),
        ('MB', 1 << 20),
        ('KB', 1 << 10),
    ]:
        if n >= thresh:
            return f'{n / thresh:.2f} {unit}'
    return f'{n} B'


# ---- Dataset construction ---------------------------------------------------


def _make_dataset(ds_cfg: DictConfig, keys_to_cache: list[str], common: dict):
    fmt = ds_cfg.format.lower()
    source = ds_cfg.get('source', 'local')
    raw_path = Path(ds_cfg.path)
    if source == 'local' and not raw_path.is_absolute():
        raw_path = get_cache_dir(sub_folder='datasets') / raw_path
    path = str(raw_path)

    if fmt == 'lance':
        kwargs: dict = {}
        if source == 's3':
            kwargs['connect_kwargs'] = {
                'storage_options': _lance_storage_opts(ds_cfg)
            }
        return LanceDataset(
            path=path, keys_to_cache=keys_to_cache, **kwargs, **common
        )

    if fmt == 'hdf5':
        kwargs = {}
        if source == 's3':
            kwargs['storage_options'] = _hdf5_storage_opts(ds_cfg)
        return HDF5Dataset(
            path=path, keys_to_cache=keys_to_cache, **kwargs, **common
        )

    if fmt == 'video':
        if VideoDataset is None:
            raise ImportError(
                'VideoDataset not available (install decord/imageio)'
            )
        return VideoDataset(path=path, video_keys=['pixels'], **common)

    raise ValueError(f'Unknown format: {fmt!r}')


# ---- Benchmark loop ---------------------------------------------------------


def _bench_one(label: str, ds, b_cfg: DictConfig) -> tuple[float, float]:
    loader = DataLoader(
        ds,
        batch_size=b_cfg.batch_size,
        num_workers=b_cfg.num_workers,
        pin_memory=False,
        prefetch_factor=2 if b_cfg.num_workers > 0 else None,
    )
    it = iter(loader)
    for _ in range(b_cfg.warmup):
        b = next(it, None)
        if b is None:
            it = iter(loader)
            b = next(it)
        _ = b['pixels'].shape

    n, t0 = 0, time.perf_counter()
    for _ in range(b_cfg.steps):
        b = next(it, None)
        if b is None:
            it = iter(loader)
            b = next(it)
        n += b['pixels'].shape[0]
    dt = time.perf_counter() - t0
    sps = n / dt
    ms_per_step = dt / b_cfg.steps * 1e3
    print(
        f'{label:<54} {sps:9.1f} samples/s   ({ms_per_step:7.1f} ms/step)',
        flush=True,
    )
    return sps, ms_per_step


# ---- Main -------------------------------------------------------------------


@hydra.main(
    version_base=None, config_path='./configs', config_name='benchmark'
)
def main(cfg: DictConfig) -> None:
    b_cfg = cfg.benchmark
    print(
        f'workers={b_cfg.num_workers} batch={b_cfg.batch_size} steps={b_cfg.steps}\n'
    )

    results: list[tuple[str, str, str, str, float, float, int]] = []

    for ds_cfg in cfg.datasets:
        name = ds_cfg.name
        fmt = ds_cfg.format.upper()
        source = ds_cfg.get('source', 'local')
        keys_to_load = list(ds_cfg.keys_to_load)
        keys_to_cache_full = list(ds_cfg.get('keys_to_cache', []))

        common = dict(
            num_steps=b_cfg.num_steps,
            frameskip=b_cfg.frameskip,
            keys_to_load=keys_to_load,
        )

        cache_variants: list[tuple[str, list[str]]] = [('no-cache', [])]
        if keys_to_cache_full:
            cache_variants.append(('cached', keys_to_cache_full))

        for cache_label, keys_to_cache in cache_variants:
            label = f'{name:<14} {fmt:<7} {source:<8} {cache_label:<8}'
            try:
                ds = _make_dataset(ds_cfg, keys_to_cache, common)
            except Exception as e:
                print(f'  (skipping {label.strip()}: {e})', flush=True)
                continue
            sps, ms_step = _bench_one(label, ds, b_cfg)
            raw_path = Path(ds_cfg.path)
            if not raw_path.is_absolute():
                raw_path = get_cache_dir(sub_folder='datasets') / raw_path
            size = _local_size(raw_path) if source == 'local' else 0
            results.append(
                (name, fmt, source, cache_label, sps, ms_step, size)
            )

    print('\n## Throughput\n')
    print(
        '| Dataset        | Format  | Source   | Cache    | samples/s | ms/step  | Storage    |'
    )
    print(
        '|----------------|---------|----------|----------|-----------|----------|------------|'
    )
    for name, fmt, source, cache, sps, ms_step, size in results:
        print(
            f'| {name:<14} | {fmt:<7} | {source:<8} | {cache:<8} | '
            f'{sps:9.1f} | {ms_step:8.1f} | {_fmt_bytes(size):>10} |'
        )


if __name__ == '__main__':
    import multiprocessing as mp

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
