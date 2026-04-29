"""Convert a LeRobot dataset into HDF5 + Lance for the throughput benchmark.

Pre-step for `compare_h5_lance.py`. Reads ``lerobot/pusht_image`` (or any
LeRobot HF dataset) via :class:`LeRobotAdapter`, exposing every native
column through ``key_aliases``, and writes the same data to two formats
locally:

  - ``pusht.h5``      (HDF5, single file)
  - ``pusht.lance``   (Lance, directory)

Both end up with identical schemas and identical data, so the bench can
read the same rows out of all three formats (LeRobot / HDF5 / Lance) for
an apples-to-apples comparison.

The leading monkey-patch caches LeRobot's ``get_safe_version`` lookup so
the conversion doesn't fire one HuggingFace API call per episode (which
otherwise times out around episode ~80 on slower networks). See:
https://github.com/huggingface/lerobot/blob/v0.5.1/src/lerobot/datasets/utils.py
"""

from __future__ import annotations

import lerobot.datasets.utils as _lru


_orig_get_safe_version = _lru.get_safe_version
_version_cache: dict[tuple, str] = {}


def _cached_get_safe_version(repo_id, version):
    key = (repo_id, version)
    if key not in _version_cache:
        _version_cache[key] = _orig_get_safe_version(repo_id, version)
    return _version_cache[key]


_lru.get_safe_version = _cached_get_safe_version


from stable_worldmodel.data import LeRobotAdapter, get_format
from stable_worldmodel.data.utils import _episode_to_step_lists


REPO_ID = 'lerobot/pusht_image'
KEY_ALIASES = {
    'observation.image': 'pixels',
    'observation.state': 'proprio',
    'action': 'action',
    'timestamp': 'timestamp',
    'frame_index': 'frame_index',
    'next.reward': 'reward',
    'next.done': 'done',
    'index': 'global_index',
}
TARGETS = [
    ('hdf5', 'pusht.h5'),
    ('lance', 'pusht.lance'),
]


def main() -> None:
    src = LeRobotAdapter(REPO_ID, key_aliases=KEY_ALIASES)
    n_eps = len(src.lengths)

    for fmt, dest in TARGETS:
        print(f'→ {fmt}: {dest}')
        writer_cls = get_format(fmt)
        with writer_cls.open_writer(dest, mode='overwrite') as w:
            def gen():
                for i in range(n_eps):
                    yield _episode_to_step_lists(
                        src.load_episode(i), int(src.lengths[i])
                    )
            w.write_episodes(gen())


if __name__ == '__main__':
    main()
