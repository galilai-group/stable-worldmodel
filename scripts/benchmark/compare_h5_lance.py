from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from stable_worldmodel.data import HDF5Dataset, LanceDataset

try:
    from stable_worldmodel.data import VideoDataset
except ImportError:  # decord/imageio missing — Video row will be skipped
    VideoDataset = None


# Set these on Colab/laptop. Leave blank on EC2 with an IAM instance role
AWS_ACCESS_KEY_ID = ''
AWS_SECRET_ACCESS_KEY = ''
AWS_REGION = 'us-east-2'

S3_BUCKET = 'lancedb-datasets-dev-us-east-2-devrel'
S3_LANCE_PREFIX = f's3://{S3_BUCKET}/training/stableworldmodel/tworoom_lance'
S3_LANCE_TABLE = 'lewm_tworoom'
S3_HDF5_URI = f's3://{S3_BUCKET}/training/stableworldmodel/tworoom/tworoom.h5'

LOCAL_LANCE_DIR = Path('./tworoom_lance_local').resolve()
LOCAL_HDF5_PATH = Path('./tworoom.h5').resolve()
LOCAL_VIDEO_DIR = Path('./tworoom.video').resolve()

DEFAULT_COLUMNS = ['pixels', 'action', 'proprio']
CACHE_COLS = ['action', 'proprio']
LEROBOT_REPO = 'lerobot/pusht_image'


def _lance_opts() -> dict:
    opts = {'region': AWS_REGION, 'virtual_hosted_style_request': 'true'}
    if AWS_ACCESS_KEY_ID:
        opts['aws_access_key_id'] = AWS_ACCESS_KEY_ID
        opts['aws_secret_access_key'] = AWS_SECRET_ACCESS_KEY
    return opts


def _hdf5_opts() -> dict:
    opts = {'client_kwargs': {'region_name': AWS_REGION}}
    if AWS_ACCESS_KEY_ID:
        opts['key'] = AWS_ACCESS_KEY_ID
        opts['secret'] = AWS_SECRET_ACCESS_KEY
    return opts


def _ensure_local_h5() -> bool:
    if LOCAL_HDF5_PATH.exists():
        return True
    print(f'downloading {S3_HDF5_URI} → {LOCAL_HDF5_PATH} ...', flush=True)
    r = subprocess.run(
        [
            'aws',
            's3',
            'cp',
            S3_HDF5_URI,
            str(LOCAL_HDF5_PATH),
            '--region',
            AWS_REGION,
            '--no-progress',
        ]
    )
    return r.returncode == 0


def _ensure_local_lance() -> bool:
    table_dir = LOCAL_LANCE_DIR / f'{S3_LANCE_TABLE}.lance'
    if table_dir.exists() and any(table_dir.iterdir()):
        return True
    print(
        f'syncing {S3_LANCE_PREFIX}/{S3_LANCE_TABLE}.lance/ → {table_dir} ...',
        flush=True,
    )
    table_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
            'aws',
            's3',
            'sync',
            f'{S3_LANCE_PREFIX}/{S3_LANCE_TABLE}.lance/',
            str(table_dir),
            '--region',
            AWS_REGION,
            '--no-progress',
        ]
    )
    return r.returncode == 0


def _bench(label, ds, args):
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    it = iter(loader)
    for _ in range(args.warmup):
        b = next(it, None) or next(iter(loader))
        _ = b['pixels'].shape

    n, t0 = 0, time.perf_counter()
    for _ in range(args.steps):
        b = next(it, None)
        if b is None:
            it = iter(loader)
            b = next(it)
        n += b['pixels'].shape[0]
    dt = time.perf_counter() - t0
    sps = n / dt
    print(
        f'{label:<32} {sps:8.1f} samples/s   ({dt / args.steps * 1e3:5.1f} ms/step)',
        flush=True,
    )
    return label, sps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-steps', type=int, default=4)
    p.add_argument('--frameskip', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument(
        '--no-local', action='store_true', help='skip local rows (no download)'
    )
    p.add_argument('--no-s3', action='store_true', help='skip S3 rows')
    p.add_argument('--include-lerobot', action='store_true')
    p.add_argument('--lerobot-repo', default=LEROBOT_REPO)
    args = p.parse_args()

    common = dict(
        num_steps=args.num_steps,
        frameskip=args.frameskip,
        keys_to_load=DEFAULT_COLUMNS,
    )

    datasets: list[tuple[str, torch.utils.data.Dataset]] = []

    if not args.no_local:
        if _ensure_local_h5():
            datasets.append(
                (
                    'HDF5 local (no cache)',
                    HDF5Dataset(
                        path=str(LOCAL_HDF5_PATH), keys_to_cache=[], **common
                    ),
                )
            )
            datasets.append(
                (
                    'HDF5 local (cached)',
                    HDF5Dataset(
                        path=str(LOCAL_HDF5_PATH),
                        keys_to_cache=CACHE_COLS,
                        **common,
                    ),
                )
            )
        else:
            print('(skipping local HDF5: download failed)')
        if _ensure_local_lance():
            datasets.append(
                (
                    'Lance local (no cache)',
                    LanceDataset(
                        path=str(LOCAL_LANCE_DIR),
                        table_name=S3_LANCE_TABLE,
                        keys_to_cache=[],
                        **common,
                    ),
                )
            )
            datasets.append(
                (
                    'Lance local (cached)',
                    LanceDataset(
                        path=str(LOCAL_LANCE_DIR),
                        table_name=S3_LANCE_TABLE,
                        keys_to_cache=CACHE_COLS,
                        **common,
                    ),
                )
            )
        else:
            print('(skipping local Lance: sync failed)')
        if VideoDataset is None:
            print('(skipping local Video: decord/imageio not installed)')
        elif not LOCAL_VIDEO_DIR.exists():
            print(
                f'(skipping local Video: {LOCAL_VIDEO_DIR} not found — '
                'run scripts/benchmark/convert.py first, or convert your '
                "local .h5 with `convert(src.h5, dst.video, dest_format='video')`)"
            )
        else:
            # FolderDataset/VideoDataset eagerly load tabular .npz columns
            # into RAM at init, so action/proprio are already "cached" — no
            # keys_to_cache knob; one row suffices.
            try:
                datasets.append(
                    (
                        'Video local',
                        VideoDataset(
                            path=str(LOCAL_VIDEO_DIR),
                            video_keys=['pixels'],
                            **common,
                        ),
                    )
                )
            except Exception as e:
                print(f'(skipping local Video: {e})')

    if not args.no_s3:
        lance_opts = {'storage_options': _lance_opts()}
        h5_opts = _hdf5_opts()
        datasets.append(
            (
                'Lance S3 (no cache)',
                LanceDataset(
                    path=S3_LANCE_PREFIX,
                    table_name=S3_LANCE_TABLE,
                    keys_to_cache=[],
                    connect_kwargs=lance_opts,
                    **common,
                ),
            )
        )
        datasets.append(
            (
                'Lance S3 (cached)',
                LanceDataset(
                    path=S3_LANCE_PREFIX,
                    table_name=S3_LANCE_TABLE,
                    keys_to_cache=CACHE_COLS,
                    connect_kwargs=lance_opts,
                    **common,
                ),
            )
        )
        try:
            datasets.append(
                (
                    'HDF5 S3 (no cache)',
                    HDF5Dataset(
                        path=S3_HDF5_URI,
                        storage_options=h5_opts,
                        keys_to_cache=[],
                        **common,
                    ),
                )
            )
            datasets.append(
                (
                    'HDF5 S3 (cached)',
                    HDF5Dataset(
                        path=S3_HDF5_URI,
                        storage_options=h5_opts,
                        keys_to_cache=CACHE_COLS,
                        **common,
                    ),
                )
            )
        except Exception as e:
            print(f'(skipping HDF5 S3: {e})')

    if args.include_lerobot:
        try:
            from stable_worldmodel.data import LeRobotAdapter

            datasets.append(
                (
                    f'LeRobot ({args.lerobot_repo})',
                    LeRobotAdapter(args.lerobot_repo, **common),
                )
            )
        except Exception as e:
            print(f'(skipping LeRobot: {e})')

    print(
        f'\nworkers={args.num_workers} batch={args.batch_size} steps={args.steps}\n',
        flush=True,
    )
    results = [_bench(label, ds, args) for label, ds in datasets]
    print('\nSummary (samples/sec):')
    for label, sps in results:
        print(f'  {label:<32} {sps:8.1f}')


if __name__ == '__main__':
    import multiprocessing as mp

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    os.environ.setdefault('AWS_DEFAULT_REGION', AWS_REGION)
    main()
