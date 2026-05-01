"""Comprehensive benchmark — tworoom + pusht across HDF5/Lance/Video.

Throughput in samples/s, on-disk storage in MB. Auto-downloads any missing
local copies from S3 first, then runs every applicable combination, then
prints a markdown table.

What's measured per (dataset x format x source x cache-mode):
  - HDF5 local/s3 (cached/no-cache), Lance local/s3 (cached/no-cache),
    Video local. 8 rows per dataset.
  - Storage cost — local file/dir size and S3 prefix size.

Run on EC2 with an IAM instance role attached, or set AWS creds in env.
Pass ``--no-s3`` to skip the network rows, ``--no-local`` to skip the
local rows.

Run scripts/benchmark/convert.py first to produce + upload the source
data this script reads from.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from torch.utils.data import DataLoader

from stable_worldmodel.data import HDF5Dataset, LanceDataset

try:
    from stable_worldmodel.data import VideoDataset
except ImportError:  # decord/imageio missing — Video rows skipped
    VideoDataset = None


# ---- Configuration (must match scripts/benchmark/convert.py PLAN) ----------

S3_BUCKET = 'lancedb-datasets-dev-us-east-2-devrel'
S3_BASE = f's3://{S3_BUCKET}/training/stableworldmodel'
S3_REGION = 'us-east-2'

# Set on Colab/laptop. Leave blank on EC2 with an IAM instance role.
AWS_ACCESS_KEY_ID = ''
AWS_SECRET_ACCESS_KEY = ''

DEFAULT_COLUMNS = ['pixels', 'action', 'proprio']
CACHE_COLS = ['action', 'proprio']

# Tworoom-only. Pusht (96×96) made per-format throughput differences
# noisy; tworoom (224×224) is now the bench's source of truth.
DATASETS = {
    'tworoom': {
        'image_size': 224,
        'h5_local': './tworoom.h5',
        'h5_s3': f'{S3_BASE}/tworoom/tworoom.h5',
        'lance_local': './tworoom.lance',
        'lance_s3': f'{S3_BASE}/tworoom/tworoom.lance',
        'video_local': './tworoom.video',
    },
    'pusht': {
        'image_size': 224,
        'h5_local': './pusht.h5',
        'h5_s3': f'{S3_BASE}/pusht/pusht.h5',
        'lance_local': './pusht.lance',
        'lance_s3': f'{S3_BASE}/pusht/pusht.lance',
        'video_local': './pusht.video',
    },
}


# ---- Storage-options helpers (creds-aware) ---------------------------------


def _lance_storage_opts() -> dict:
    opts = {'region': S3_REGION, 'virtual_hosted_style_request': 'true'}
    if AWS_ACCESS_KEY_ID:
        opts['aws_access_key_id'] = AWS_ACCESS_KEY_ID
        opts['aws_secret_access_key'] = AWS_SECRET_ACCESS_KEY
    return opts


def _hdf5_storage_opts() -> dict:
    opts = {'client_kwargs': {'region_name': S3_REGION}}
    if AWS_ACCESS_KEY_ID:
        opts['key'] = AWS_ACCESS_KEY_ID
        opts['secret'] = AWS_SECRET_ACCESS_KEY
    return opts


# ---- Auto-download from S3 -------------------------------------------------


def _aws(*args) -> int:
    return subprocess.run(['aws', *args, '--region', S3_REGION]).returncode


def _ensure_local_h5(local: Path, s3_uri: str) -> bool:
    if local.exists():
        return True
    print(f'  downloading {s3_uri} → {local}', flush=True)
    return _aws('s3', 'cp', s3_uri, str(local), '--no-progress') == 0


def _ensure_local_dir(local_dir: Path, s3_uri: str) -> bool:
    """Sync an S3 directory prefix (Lance / Video) to a local dir."""
    if local_dir.exists() and any(local_dir.iterdir()):
        return True
    print(f'  syncing {s3_uri}/ -> {local_dir}', flush=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    return (
        _aws(
            's3',
            'sync',
            s3_uri.rstrip('/') + '/',
            str(local_dir),
            '--no-progress',
        )
        == 0
    )


# ---- Storage measurement ---------------------------------------------------


def _local_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())


def _s3_size(uri: str) -> int:
    """Total bytes under an S3 URI via `aws s3 ls --recursive --summarize`."""
    try:
        out = subprocess.check_output(
            [
                'aws',
                's3',
                'ls',
                '--recursive',
                '--summarize',
                uri,
                '--region',
                S3_REGION,
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return 0
    for line in out.splitlines():
        if 'Total Size' in line:
            return int(line.split(':', 1)[1].strip())
    return 0


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


# ---- Bench loop ------------------------------------------------------------


def _bench_one(label, ds, args):
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    it = iter(loader)
    for _ in range(args.warmup):
        b = next(it, None)
        if b is None:
            it = iter(loader)
            b = next(it)
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
    ms_per_step = dt / args.steps * 1e3
    print(
        f'{label:<42} {sps:9.1f} samples/s   ({ms_per_step:7.1f} ms/step)',
        flush=True,
    )
    return sps, ms_per_step


# ---- Main ------------------------------------------------------------------


def _build_rows(ds_name, cfg, args, common):
    """Return list of (fmt, source, cache, dataset, storage_uri)."""
    rows: list[tuple[str, str, str, object, str]] = []
    h5_local = Path(cfg['h5_local']).resolve()
    lance_local = Path(cfg['lance_local']).resolve()
    video_local = Path(cfg['video_local']).resolve()

    if not args.no_local:
        if _ensure_local_h5(h5_local, cfg['h5_s3']):
            rows.append(
                (
                    'HDF5',
                    'local',
                    'no-cache',
                    HDF5Dataset(
                        path=str(h5_local), keys_to_cache=[], **common
                    ),
                    str(h5_local),
                )
            )
            rows.append(
                (
                    'HDF5',
                    'local',
                    'cached',
                    HDF5Dataset(
                        path=str(h5_local), keys_to_cache=CACHE_COLS, **common
                    ),
                    str(h5_local),
                )
            )
        if _ensure_local_dir(lance_local, cfg['lance_s3']):
            rows.append(
                (
                    'Lance',
                    'local',
                    'no-cache',
                    LanceDataset(
                        path=str(lance_local), keys_to_cache=[], **common
                    ),
                    str(lance_local),
                )
            )
            rows.append(
                (
                    'Lance',
                    'local',
                    'cached',
                    LanceDataset(
                        path=str(lance_local),
                        keys_to_cache=CACHE_COLS,
                        **common,
                    ),
                    str(lance_local),
                )
            )
        if VideoDataset is not None and video_local.exists():
            try:
                rows.append(
                    (
                        'Video',
                        'local',
                        '-',
                        VideoDataset(
                            path=str(video_local),
                            video_keys=['pixels'],
                            **common,
                        ),
                        str(video_local),
                    )
                )
            except Exception as e:
                print(f'  (skipping {ds_name} Video: {e})')

    if not args.no_s3:
        lance_opts = {'storage_options': _lance_storage_opts()}
        h5_opts = _hdf5_storage_opts()
        rows.append(
            (
                'Lance',
                's3',
                'no-cache',
                LanceDataset(
                    path=cfg['lance_s3'],
                    keys_to_cache=[],
                    connect_kwargs=lance_opts,
                    **common,
                ),
                cfg['lance_s3'],
            )
        )
        rows.append(
            (
                'Lance',
                's3',
                'cached',
                LanceDataset(
                    path=cfg['lance_s3'],
                    keys_to_cache=CACHE_COLS,
                    connect_kwargs=lance_opts,
                    **common,
                ),
                cfg['lance_s3'],
            )
        )
        rows.append(
            (
                'HDF5',
                's3',
                'no-cache',
                HDF5Dataset(
                    path=cfg['h5_s3'],
                    storage_options=h5_opts,
                    keys_to_cache=[],
                    **common,
                ),
                cfg['h5_s3'],
            )
        )
        rows.append(
            (
                'HDF5',
                's3',
                'cached',
                HDF5Dataset(
                    path=cfg['h5_s3'],
                    storage_options=h5_opts,
                    keys_to_cache=CACHE_COLS,
                    **common,
                ),
                cfg['h5_s3'],
            )
        )

    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-steps', type=int, default=4)
    p.add_argument('--frameskip', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--no-local', action='store_true')
    p.add_argument('--no-s3', action='store_true')
    args = p.parse_args()

    common = dict(
        num_steps=args.num_steps,
        frameskip=args.frameskip,
        keys_to_load=DEFAULT_COLUMNS,
    )

    print(
        f'workers={args.num_workers} batch={args.batch_size} steps={args.steps}\n'
    )

    # Each result: (ds_name, fmt, source, cache, storage_uri, sps, ms_step, size_bytes)
    results: list[tuple[str, str, str, str, str, float, float, int]] = []

    for ds_name, cfg in DATASETS.items():
        rows = _build_rows(ds_name, cfg, args, common)

        # Storage costs first; cache by URI so we don't re-list S3 prefixes.
        storage_cache: dict[str, int] = {}
        for fmt, source, cache, ds, storage_uri in rows:
            if storage_uri not in storage_cache:
                storage_cache[storage_uri] = (
                    _s3_size(storage_uri)
                    if storage_uri.startswith('s3://')
                    else _local_size(Path(storage_uri))
                )

        for fmt, source, cache, ds, storage_uri in rows:
            label = f'{ds_name:<7}  {fmt:<7} {source:<8} {cache:<8}'
            sps, ms_step = _bench_one(label, ds, args)
            results.append(
                (
                    ds_name,
                    fmt,
                    source,
                    cache,
                    storage_uri,
                    sps,
                    ms_step,
                    storage_cache[storage_uri],
                )
            )

    # ---- Markdown summary --------------------------------------------------
    print('\n## Throughput\n')
    print(
        '| Dataset | Format  | Source   | Cache    | samples/s | ms/step  | Storage    |'
    )
    print(
        '|---------|---------|----------|----------|-----------|----------|------------|'
    )
    for ds_name, fmt, source, cache, _uri, sps, ms_step, size in results:
        print(
            f'| {ds_name:<7} | {fmt:<7} | {source:<8} | {cache:<8} | '
            f'{sps:9.1f} | {ms_step:8.1f} | {_fmt_bytes(size):>10} |'
        )

    print('\n## Storage size per dataset/format (local)\n')
    print('| Dataset | Format  | Local size |')
    print('|---------|---------|------------|')
    seen: set[tuple[str, str]] = set()
    for ds_name, fmt, source, cache, uri, *_ in results:
        if source != 'local':
            continue
        key = (ds_name, fmt)
        if key in seen:
            continue
        seen.add(key)
        size = _local_size(Path(uri))
        print(f'| {ds_name:<7} | {fmt:<7} | {_fmt_bytes(size):>10} |')


if __name__ == '__main__':
    import multiprocessing as mp

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    os.environ.setdefault('AWS_DEFAULT_REGION', S3_REGION)
    main()
