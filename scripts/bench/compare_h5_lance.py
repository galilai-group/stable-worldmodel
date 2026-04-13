"""Compare HDF5 vs Lance throughput for the full Tworoom dataset."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from stable_worldmodel.data import HDF5Dataset, LanceDataset

H5_NAME = 'quentinll--lewm-tworooms/tworoom'
DEFAULT_CACHE_DIR = Path('~/.stable_worldmodel').expanduser()
DEFAULT_COLUMNS = ['pixels', 'action', 'proprio']
CACHE_COLS = ['action', 'proprio']


def _make_loader(dataset: torch.utils.data.Dataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        prefetch_factor=2,
    )


def _benchmark(label: str, dataset: torch.utils.data.Dataset, args: argparse.Namespace) -> tuple[str, float, float]:
    loader = _make_loader(dataset, args)
    it = iter(loader)
    for _ in range(args.warmup):
        batch = next(it, None)
        if batch is None:
            it = iter(loader)
            batch = next(it)
        _ = batch['pixels'].shape

    samples = 0
    start = time.perf_counter()
    for _ in range(args.steps):
        batch = next(it, None)
        if batch is None:
            it = iter(loader)
            batch = next(it)
        samples += batch['pixels'].shape[0]
    elapsed = time.perf_counter() - start
    sps = samples / elapsed
    print(f"{label:<32} {sps:8.1f} samples/s")
    return label, sps, elapsed / args.steps


def _connect_kwargs_from_env() -> dict:
    storage = {}
    if os.environ.get('AWS_ACCESS_KEY_ID'):
        storage['aws_access_key_id'] = os.environ['AWS_ACCESS_KEY_ID']
    if os.environ.get('AWS_SECRET_ACCESS_KEY'):
        storage['aws_secret_access_key'] = os.environ['AWS_SECRET_ACCESS_KEY']
    if os.environ.get('AWS_SESSION_TOKEN'):
        storage['aws_session_token'] = os.environ['AWS_SESSION_TOKEN']
    if os.environ.get('AWS_DEFAULT_REGION'):
        storage['region'] = os.environ['AWS_DEFAULT_REGION']
    if os.environ.get('AWS_ENDPOINT_URL'):
        storage['endpoint_url'] = os.environ['AWS_ENDPOINT_URL']
    return {'storage_options': storage} if storage else {}


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark HDF5 vs Lance throughput for Tworoom.')
    parser.add_argument('--num-steps', type=int, default=4)
    parser.add_argument('--frameskip', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--lance-local', default='./tworoom_lance_full')
    parser.add_argument('--lance-s3', default='s3://lancedb-datasets-dev-us-east-2-devrel/training/stableworldmodel/tworoom_lance')
    parser.add_argument('--table-name', default='lewm_tworoom')
    args = parser.parse_args()
    args.lance_local = str(Path(args.lance_local).resolve())
    print(f"Local Lance URI: {args.lance_local}")

    common = dict(
        num_steps=args.num_steps,
        frameskip=args.frameskip,
        keys_to_load=DEFAULT_COLUMNS,
    )

    h5_local_cached = HDF5Dataset(
        H5_NAME,
        cache_dir=str(DEFAULT_CACHE_DIR),
        keys_to_cache=CACHE_COLS,
        **common,
    )
    h5_local_nocache = HDF5Dataset(
        H5_NAME,
        cache_dir=str(DEFAULT_CACHE_DIR),
        keys_to_cache=[],
        **common,
    )

    lance_local = LanceDataset(
        uri=args.lance_local,
        table_name=args.table_name,
        image_columns=['pixels'],
        keys_to_cache=[],
        **common,
    )
    lance_local_cached = LanceDataset(
        uri=args.lance_local,
        table_name=args.table_name,
        image_columns=['pixels'],
        keys_to_cache=CACHE_COLS,
        **common,
    )

    connect_kwargs = _connect_kwargs_from_env()
    lance_s3 = LanceDataset(
        uri=args.lance_s3,
        table_name=args.table_name,
        image_columns=['pixels'],
        keys_to_cache=[],
        connect_kwargs=connect_kwargs,
        **common,
    )
    lance_s3_cached = LanceDataset(
        uri=args.lance_s3,
        table_name=args.table_name,
        image_columns=['pixels'],
        keys_to_cache=CACHE_COLS,
        connect_kwargs=connect_kwargs,
        **common,
    )

    print('Warming up / benchmarking...')
    results = []
    for label, dataset in [
        ('HDF5 local (cached)', h5_local_cached),
        ('HDF5 local (no cache)', h5_local_nocache),
        ('Lance local (no cache)', lance_local),
        ('Lance local (cached)', lance_local_cached),
        ('Lance S3 (no cache)', lance_s3),
        ('Lance S3 (cached)', lance_s3_cached),
    ]:
        results.append(_benchmark(label, dataset, args))

    print('\nSummary (samples/sec):')
    for label, sps, _ in results:
        print(f"  {label:<28} {sps:8.1f}")


if __name__ == '__main__':
    import multiprocessing as mp

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
