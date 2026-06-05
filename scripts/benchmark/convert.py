"""Convert + upload tworoom + pusht across HDF5/Lance/Video formats.

Both datasets are 224x224 LeWorldModel sources. Idempotent: skips
conversion if the local output already exists; skips upload by checking
the S3 prefix. Pass ``--force`` to redo from scratch.

Defaults:
    python convert.py            # convert + upload all formats
    python convert.py --no-upload         # convert only
    python convert.py --force             # ignore existing local outputs

Source resolution:
    tworoom: ``{S3_BASE}/tworoom/tworoom.h5`` (no public HF mirror).
    pusht:   tries ``{S3_BASE}/pusht/pusht.h5`` first; on miss, falls
             back to ``quentinll/lewm-pusht`` on HF and decompresses
             the ``.zst`` blob via the system ``zstd`` CLI.

Run on EC2 with an IAM instance role attached, or set AWS creds in env.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from stable_worldmodel.data import HDF5Dataset, get_format
from stable_worldmodel.data.utils import _episode_to_step_lists


# ---- Configuration ---------------------------------------------------------

S3_BUCKET = 'lancedb-datasets-dev-us-east-2-devrel'
S3_BASE = f's3://{S3_BUCKET}/training/stableworldmodel'
S3_REGION = 'us-east-2'

# tworoom: only available on S3.
TWOROOM_S3_URI = f'{S3_BASE}/tworoom/tworoom.h5'

# pusht: try S3 first, fall back to HF + zstd decompress on miss.
PUSHT_S3_URI = f'{S3_BASE}/pusht/pusht.h5'
PUSHT_HF_REPO = 'quentinll/lewm-pusht'
PUSHT_HF_FILE = 'pusht_expert_train.h5.zst'

# Each entry: (local_path, s3_subpath_relative_to_S3_BASE).
PLAN = {
    'tworoom': {
        'hdf5': ('tworoom.h5', 'tworoom/tworoom.h5'),
        'lance': ('tworoom.lance', 'tworoom/tworoom.lance/'),
        'video': ('tworoom.video', 'tworoom/tworoom.video/'),
    },
    'pusht': {
        'hdf5': ('pusht.h5', 'pusht/pusht.h5'),
        'lance': ('pusht.lance', 'pusht/pusht.lance/'),
        'video': ('pusht.video', 'pusht/pusht.video/'),
    },
}


# ---- Helpers ---------------------------------------------------------------


def _is_done(local_path: Path) -> bool:
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
        print(f'  upload {local} -> {s3_uri}', flush=True)
        rc = _aws('s3', 'cp', str(local), s3_uri, '--no-progress')
    else:
        print(f'  sync {local}/ -> {s3_uri}', flush=True)
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


# ---- Source fetchers -------------------------------------------------------


def _fetch_tworoom_h5(dest: Path) -> None:
    print(f'  downloading {TWOROOM_S3_URI} -> {dest}', flush=True)
    rc = _aws('s3', 'cp', TWOROOM_S3_URI, str(dest), '--no-progress')
    if rc != 0:
        raise SystemExit('tworoom h5 download failed')


def _fetch_pusht_h5(dest: Path) -> None:
    """S3 first; fall back to HF + zstd decompress."""
    print(f'  trying {PUSHT_S3_URI} -> {dest}', flush=True)
    rc = _aws('s3', 'cp', PUSHT_S3_URI, str(dest), '--no-progress')
    if rc == 0:
        return

    print(
        f'  S3 miss; downloading {PUSHT_HF_REPO}/{PUSHT_HF_FILE} from HF',
        flush=True,
    )
    from huggingface_hub import hf_hub_download

    zst = hf_hub_download(
        repo_id=PUSHT_HF_REPO,
        filename=PUSHT_HF_FILE,
        repo_type='dataset',
    )
    print(f'  decompressing {zst} -> {dest}', flush=True)
    rc = subprocess.run(['zstd', '-d', '-f', '-o', str(dest), zst]).returncode
    if rc != 0:
        raise SystemExit(
            'pusht: zstd decompression failed (install zstd if missing)'
        )


# ---- Conversions -----------------------------------------------------------


def _derive_lance_video(name: str, h5_p: Path, force: bool) -> None:
    """From a local source .h5, derive lance + video outputs in PLAN[name]."""
    plan = PLAN[name]
    targets: list[tuple[str, str]] = []
    for fmt in ('lance', 'video'):
        dest = plan[fmt][0]
        dest_p = Path(dest)
        if force and dest_p.exists():
            _wipe(dest_p)
        if _is_done(dest_p):
            print(f'  {fmt}: {dest} already exists; skipping')
            continue
        targets.append((fmt, dest))
    if not targets:
        return

    src = HDF5Dataset(path=str(h5_p))
    n_eps = len(src.lengths)
    for fmt, dest in targets:
        print(f'  -> {fmt}: {dest} ({n_eps} episodes)', flush=True)
        writer_cls = get_format(fmt)
        with writer_cls.open_writer(dest, mode='overwrite') as w:

            def gen():
                for i in range(n_eps):
                    yield _episode_to_step_lists(
                        src.load_episode(i), int(src.lengths[i])
                    )

            w.write_episodes(gen())


def convert_tworoom(force: bool) -> None:
    print('\n=== tworoom ===')
    h5_p = Path(PLAN['tworoom']['hdf5'][0])
    if not _is_done(h5_p):
        _fetch_tworoom_h5(h5_p)
    else:
        print(f'  source: {h5_p} (already present)')
    _derive_lance_video('tworoom', h5_p, force)


def convert_pusht(force: bool) -> None:
    print('\n=== pusht ===')
    h5_p = Path(PLAN['pusht']['hdf5'][0])
    if not _is_done(h5_p):
        _fetch_pusht_h5(h5_p)
    else:
        print(f'  source: {h5_p} (already present)')
    _derive_lance_video('pusht', h5_p, force)


# ---- Main ------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--no-upload', action='store_true', help='skip the S3 upload step'
    )
    p.add_argument(
        '--force',
        action='store_true',
        help='re-convert even if outputs exist',
    )
    args = p.parse_args()

    convert_tworoom(args.force)
    convert_pusht(args.force)

    if args.no_upload:
        print('\n--no-upload: leaving S3 untouched.')
        return

    print('\n=== upload ===')
    for ds_name, plan in PLAN.items():
        for fmt, (local, s3_sub) in plan.items():
            local_p = Path(local)
            if not local_p.exists():
                print(f'  skip {ds_name}/{fmt}: {local_p} not found')
                continue
            _upload(local_p, s3_sub)


if __name__ == '__main__':
    main()
