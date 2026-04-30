"""Convert + upload tworoom and pusht datasets across HDF5/Lance/Video formats.

Two datasets, three target formats each — the bench's source of truth.
Idempotent: skips conversion if the local output already exists; skips
upload by checking the S3 prefix. Pass ``--force`` to redo from scratch.

Defaults:
    python convert.py            # convert + upload both datasets
    python convert.py --no-upload         # convert only
    python convert.py --datasets pusht    # one dataset only
    python convert.py --force             # ignore existing local outputs

Run on EC2 with an IAM instance role attached, or set AWS creds in env.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from stable_worldmodel.data import convert, get_format
from stable_worldmodel.data.utils import _episode_to_step_lists


def _patch_lerobot_version_cache() -> None:
    """Cache HF version lookup so the LeRobot conversion doesn't fire one
    HuggingFace API call per episode (which otherwise times out around
    episode ~80 on slower networks). See:
    https://github.com/huggingface/lerobot/blob/v0.5.1/src/lerobot/datasets/utils.py

    Lazy: only patches when convert_pusht() actually runs, so this script
    can be imported on machines without lerobot installed.
    """
    import lerobot.datasets.utils as _lru

    if getattr(_lru, '_swm_version_cache_installed', False):
        return
    _orig = _lru.get_safe_version
    _cache: dict[tuple, str] = {}

    def _cached(repo_id, version):
        key = (repo_id, version)
        if key not in _cache:
            _cache[key] = _orig(repo_id, version)
        return _cache[key]

    _lru.get_safe_version = _cached
    _lru._swm_version_cache_installed = True


# ---- Configuration ---------------------------------------------------------

S3_BUCKET = 'lancedb-datasets-dev-us-east-2-devrel'
S3_BASE = f's3://{S3_BUCKET}/training/stableworldmodel'
S3_REGION = 'us-east-2'

# Pusht: from LeRobot Hub. Aliasing every native column so HDF5/Lance
# carry the full schema, not just pixels/action/proprio.
PUSHT_REPO = 'lerobot/pusht_image'
PUSHT_KEY_ALIASES = {
    'observation.image': 'pixels',
    'observation.state': 'proprio',
    'action': 'action',
    'timestamp': 'timestamp',
    'frame_index': 'frame_index',
    'next.reward': 'reward',
    'next.done': 'done',
    'index': 'global_index',
}

# Tworoom: original .h5 lives on S3 (no public HF mirror). We download
# it once with `aws s3 cp` and then derive the lance + video versions.
TWOROOM_S3_URI = f'{S3_BASE}/tworoom/tworoom.h5'

# Default Lance image codec: write BOTH cols (raw uint8 `pixels` + jpeg
# `pixels_jpeg`) so the bench can read either path against the same data.
LANCE_CODEC = 'both'

# Each entry: (local_path, s3_subpath_relative_to_S3_BASE).
# All formats live under one prefix per dataset for cleanliness.
PLAN = {
    'pusht': {
        'h5': ('pusht.h5', 'pusht/pusht.h5'),
        'lance': ('pusht.lance', 'pusht/pusht.lance/'),
        'video': ('pusht.video', 'pusht/pusht.video/'),
    },
    'tworoom': {
        'h5': ('tworoom.h5', 'tworoom/tworoom.h5'),
        'lance': ('tworoom.lance', 'tworoom/tworoom.lance/'),
        'video': ('tworoom.video', 'tworoom/tworoom.video/'),
    },
}


# ---- Helpers ---------------------------------------------------------------


def _is_done(local_path: Path) -> bool:
    """A conversion target is considered done if its non-empty on disk."""
    p = Path(local_path)
    if not p.exists():
        return False
    if p.is_file():
        return p.stat().st_size > 0
    return any(p.iterdir())


def _aws(*args) -> int:
    return subprocess.run(['aws', *args, '--region', S3_REGION]).returncode


def _upload(local: Path, s3_subpath: str) -> None:
    s3_uri = f'{S3_BASE}/{s3_subpath}'
    if local.is_file():
        print(f'  upload {local} → {s3_uri}', flush=True)
        rc = _aws('s3', 'cp', str(local), s3_uri, '--no-progress')
    else:
        print(f'  sync {local}/ → {s3_uri}', flush=True)
        rc = _aws(
            's3',
            'sync',
            f'{str(local).rstrip("/")}/',
            s3_uri,
            '--delete',
            '--no-progress',
        )
    if rc != 0:
        raise SystemExit(f'aws s3 upload failed (exit {rc}) for {local}')


def _wipe(local: Path) -> None:
    if local.is_file():
        local.unlink()
    elif local.is_dir():
        shutil.rmtree(local)


# ---- Conversions -----------------------------------------------------------


def convert_pusht(force: bool) -> None:
    """LeRobot pusht_image → {h5, lance, video} from a single source pass."""
    print('\n=== pusht ===')
    targets = []
    for fmt, (local, _) in PLAN['pusht'].items():
        local_p = Path(local)
        if force and local_p.exists():
            _wipe(local_p)
        if not _is_done(local_p):
            targets.append((fmt, local))
    if not targets:
        print('  all formats already converted; skipping')
        return

    _patch_lerobot_version_cache()
    from stable_worldmodel.data import LeRobotAdapter

    print(f'  source: lerobot://{PUSHT_REPO}')
    src = LeRobotAdapter(PUSHT_REPO, key_aliases=PUSHT_KEY_ALIASES)
    n_eps = len(src.lengths)

    for fmt, dest in targets:
        print(f'  → {fmt}: {dest} ({n_eps} episodes)', flush=True)
        kw = {'mode': 'overwrite'}
        if fmt == 'lance':
            kw['image_codec'] = LANCE_CODEC
        writer_cls = get_format(fmt)
        with writer_cls.open_writer(dest, **kw) as w:

            def gen():
                for i in range(n_eps):
                    yield _episode_to_step_lists(
                        src.load_episode(i), int(src.lengths[i])
                    )

            w.write_episodes(gen())


def convert_tworoom(force: bool) -> None:
    """Tworoom S3 .h5 → {h5 (local copy), lance, video}.

    The original tworoom .h5 lives on S3 (not HF), so we ``aws s3 cp`` it
    once into ./tworoom.h5 then derive the lance + video versions from
    that local file via the load_dataset/convert pipeline.
    """
    print('\n=== tworoom ===')
    h5_local, _ = PLAN['tworoom']['h5']
    lance_local, _ = PLAN['tworoom']['lance']
    video_local, _ = PLAN['tworoom']['video']

    h5_p = Path(h5_local)
    if force and h5_p.exists():
        _wipe(h5_p)
    if not _is_done(h5_p):
        print(f'  downloading {TWOROOM_S3_URI} → {h5_p}', flush=True)
        rc = _aws('s3', 'cp', TWOROOM_S3_URI, str(h5_p), '--no-progress')
        if rc != 0:
            raise SystemExit('tworoom h5 download failed')

    targets: list[tuple[str, str]] = []
    for fmt, dest in [('lance', lance_local), ('video', video_local)]:
        dest_p = Path(dest)
        if force and dest_p.exists():
            _wipe(dest_p)
        if _is_done(dest_p):
            print(f'  {fmt}: {dest} already exists; skipping')
            continue
        targets.append((fmt, dest))
    if not targets:
        return

    for fmt, dest in targets:
        print(f'  → {fmt}: {dest}', flush=True)
        kw = {'mode': 'overwrite'}
        if fmt == 'lance':
            kw['image_codec'] = LANCE_CODEC
        convert(str(h5_p), dest, dest_format=fmt, **kw)


# ---- Main ------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--datasets',
        nargs='+',
        choices=['pusht', 'tworoom'],
        default=['pusht', 'tworoom'],
    )
    p.add_argument(
        '--no-upload', action='store_true', help='skip the S3 upload step'
    )
    p.add_argument(
        '--force', action='store_true', help='re-convert even if outputs exist'
    )
    args = p.parse_args()

    if 'pusht' in args.datasets:
        convert_pusht(args.force)
    if 'tworoom' in args.datasets:
        convert_tworoom(args.force)

    if args.no_upload:
        print('\n--no-upload: leaving S3 untouched.')
        return

    print('\n=== upload ===')
    for ds_name in args.datasets:
        for fmt, (local, s3_sub) in PLAN[ds_name].items():
            local_p = Path(local)
            if not local_p.exists():
                print(f'  skip {ds_name}/{fmt}: {local_p} not found')
                continue
            _upload(local_p, s3_sub)


if __name__ == '__main__':
    main()
