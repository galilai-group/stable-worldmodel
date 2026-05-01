"""Convert + upload tworoom across HDF5/Lance/Video/LeRobot formats.

Tworoom (224×224) is the bench's source of truth — pusht's 96×96 frames
were too small to make per-format throughput differences land cleanly.

Idempotent: skips conversion if the local output already exists; skips
upload by checking the S3 prefix. Pass ``--force`` to redo from scratch.

Defaults:
    python convert.py            # convert + upload all formats
    python convert.py --no-upload         # convert only
    python convert.py --force             # ignore existing local outputs

Run on EC2 with an IAM instance role attached, or set AWS creds in env.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np

from stable_worldmodel.data import HDF5Dataset, get_format
from stable_worldmodel.data.utils import _episode_to_step_lists


# ---- Configuration ---------------------------------------------------------

S3_BUCKET = 'lancedb-datasets-dev-us-east-2-devrel'
S3_BASE = f's3://{S3_BUCKET}/training/stableworldmodel'
S3_REGION = 'us-east-2'

# Tworoom: original .h5 lives on S3 (no public HF mirror). We download
# it once with `aws s3 cp` and then derive the lance/video/lerobot versions.
TWOROOM_S3_URI = f'{S3_BASE}/tworoom/tworoom.h5'

# LeRobot needs an FPS and per-frame `task` string. Tworoom doesn't
# carry either natively — pick reasonable values; the bench only cares
# about read throughput, not these labels.
LEROBOT_REPO_ID = 'local/tworoom'
LEROBOT_FPS = 10
LEROBOT_TASK = 'tworoom-navigation'

# Each entry: (local_path, s3_subpath_relative_to_S3_BASE).
PLAN = {
    'tworoom': {
        'hdf5': ('tworoom.h5', 'tworoom/tworoom.h5'),
        'lance': ('tworoom.lance', 'tworoom/tworoom.lance/'),
        'video': ('tworoom.video', 'tworoom/tworoom.video/'),
        'lerobot': ('tworoom.lerobot', 'tworoom/tworoom.lerobot/'),
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


def _hdf5_to_lerobot(src: HDF5Dataset, dest: Path, n_eps: int) -> None:
    """Stream tworoom HDF5 episodes into a LeRobotDataset on disk via
    `LeRobotDataset.create()` + per-frame `add_frame()` + `save_episode()`.

    Schema: pixels → ``observation.images.main`` (video), proprio →
    ``observation.state``, action → ``action``. Per-frame ``task`` is a
    static string — bench doesn't read it.

    `streaming_encoding=True` keeps a single SVT-AV1 encoder warm across
    every episode instead of spinning one up per save. Without it,
    LeRobot writes per-frame PNGs then re-reads + encodes them at
    save_episode() — the dominant cost on a 1500+ episode dataset.
    `VideoEncodingManager` flushes the pending stream on exit; `.finalize()`
    is also called explicitly afterwards to be safe.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import VideoEncodingManager

    probe = _episode_to_step_lists(src.load_episode(0), int(src.lengths[0]))
    pixels0 = np.asarray(probe['pixels'][0])
    proprio0 = np.asarray(probe['proprio'][0])
    action0 = np.asarray(probe['action'][0])

    h, w = pixels0.shape[:2]
    features = {
        'observation.images.main': {
            'dtype': 'video',
            'shape': (h, w, 3),
            'names': ['height', 'width', 'channels'],
        },
        'observation.state': {
            'dtype': 'float32',
            'shape': (int(proprio0.size),),
        },
        'action': {
            'dtype': 'float32',
            'shape': (int(action0.size),),
        },
    }

    lr_ds = LeRobotDataset.create(
        repo_id=LEROBOT_REPO_ID,
        fps=LEROBOT_FPS,
        features=features,
        root=dest,
        use_videos=True,
        streaming_encoding=True,
        encoder_threads=4,
    )

    with VideoEncodingManager(lr_ds):
        for i in range(n_eps):
            ep_len = int(src.lengths[i])
            ep = _episode_to_step_lists(src.load_episode(i), ep_len)
            for t in range(ep_len):
                lr_ds.add_frame(
                    {
                        'observation.images.main': np.asarray(
                            ep['pixels'][t], dtype=np.uint8
                        ),
                        'observation.state': np.asarray(
                            ep['proprio'][t], dtype=np.float32
                        ).reshape(-1),
                        'action': np.asarray(
                            ep['action'][t], dtype=np.float32
                        ).reshape(-1),
                        'task': LEROBOT_TASK,
                    }
                )
            lr_ds.save_episode()


def convert_tworoom(force: bool) -> None:
    """Tworoom S3 .h5 → {h5 (canonical local copy), lance, video, lerobot}.

    The original tworoom .h5 lives on S3 (not HF). We ``aws s3 cp`` it
    once into ./tworoom.h5 — ``--force`` does NOT re-download the source
    since it's the canonical input — then read it directly via
    HDF5Dataset and stream into the lance/video/lerobot writers.
    """
    print('\n=== tworoom ===')
    h5_local, _ = PLAN['tworoom']['hdf5']
    lance_local, _ = PLAN['tworoom']['lance']
    video_local, _ = PLAN['tworoom']['video']
    lerobot_local, _ = PLAN['tworoom']['lerobot']

    # Source: download only if missing. --force never wipes the source.
    h5_p = Path(h5_local)
    if not _is_done(h5_p):
        print(f'  downloading {TWOROOM_S3_URI} → {h5_p}', flush=True)
        rc = _aws('s3', 'cp', TWOROOM_S3_URI, str(h5_p), '--no-progress')
        if rc != 0:
            raise SystemExit('tworoom h5 download failed')
    else:
        print(f'  source: {h5_p} (already present)')

    # Derived outputs: --force wipes & re-derives.
    targets: list[tuple[str, str]] = []
    for fmt, dest in [
        ('lance', lance_local),
        ('video', video_local),
        ('lerobot', lerobot_local),
    ]:
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
        print(f'  → {fmt}: {dest} ({n_eps} episodes)', flush=True)
        if fmt == 'lerobot':
            _hdf5_to_lerobot(src, Path(dest), n_eps)
            continue
        writer_cls = get_format(fmt)
        with writer_cls.open_writer(dest, mode='overwrite') as w:

            def gen():
                for i in range(n_eps):
                    yield _episode_to_step_lists(
                        src.load_episode(i), int(src.lengths[i])
                    )

            w.write_episodes(gen())


# ---- Main ------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--no-upload', action='store_true', help='skip the S3 upload step'
    )
    p.add_argument(
        '--force', action='store_true', help='re-convert even if outputs exist'
    )
    args = p.parse_args()

    convert_tworoom(args.force)

    if args.no_upload:
        print('\n--no-upload: leaving S3 untouched.')
        return

    print('\n=== upload ===')
    for fmt, (local, s3_sub) in PLAN['tworoom'].items():
        local_p = Path(local)
        if not local_p.exists():
            print(f'  skip tworoom/{fmt}: {local_p} not found')
            continue
        _upload(local_p, s3_sub)


if __name__ == '__main__':
    main()
