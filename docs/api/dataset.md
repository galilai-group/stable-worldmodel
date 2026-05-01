title: Dataset
summary: Dataset handling
---

`stable_worldmodel` ships a small, pluggable data layer built around a
**format registry**. A *format* is a recipe for reading and writing a
particular on-disk layout (HDF5, a folder of frames, MP4 episodes, …). All
built-in datasets — and any custom one you write — go through the same
registry, so the rest of the library (e.g. `World.collect`, `swm.data.load_dataset`,
`swm.data.convert`) doesn't care which backend you pick.

```text
                  ┌──────────────────────────────────────────────┐
                  │              FORMAT REGISTRY                 │
                  │ lance │ hdf5 │ folder │ video │ lerobot │ …  │
                  └──────────────────────────────────────────────┘
                          ▲                       ▲
              detect ─────┘                       └───── @register_format
       open_reader / open_writer
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
     load_dataset(...)          World.collect(..., format=)
     convert(src, dst)
```

## **[ Quick Tour ]**

```python
import stable_worldmodel as swm

# 1) Record some episodes — World picks the writer registered under `format`.
world = swm.World('swm/PushT-v1', num_envs=4, image_shape=(64, 64))
world.set_policy(swm.policy.RandomPolicy(seed=0))
world.collect('data/pusht.lance', episodes=20, seed=0)           # lance (default)
world.collect('data/pusht_video', episodes=20, format='video')   # mp4 + npz

# 2) Load any dataset — format is autodetected from the path.
ds = swm.data.load_dataset('data/pusht.lance', num_steps=4, frameskip=1)
sample = ds[0]
print(sample['pixels'].shape, sample['action'].shape)   # (4, 3, H, W) (4, A)

# 3) Switch backends without changing your collection code.
swm.data.convert('data/pusht.lance', 'data/pusht_video', dest_format='video', fps=30)
```

`load_dataset` accepts:

  - a **local path** (file or directory),
  - a **HuggingFace dataset repo** (`<user>/<repo>`) — auto-downloaded and
    cached under `$STABLEWM_HOME/datasets/`,
  - a **scheme-prefixed identifier** (e.g. `lerobot://lerobot/pusht`) handed
    straight to the matching format.

```python
ds = swm.data.load_dataset('lerobot://lerobot/pusht',
                           primary_camera_key='observation.images.top',
                           num_steps=8)
```

## **[ Storage Formats ]**

All built-in formats expose the same `Dataset` API: each item is a dict of
tensors stacked across `num_steps`, with image columns transposed to
`(T, C, H, W)`.

/// tab | Lance (default)
The **`lance`** format stores episodes as a single LanceDB table laid out
in episode-contiguous flat rows. Image columns are kept as JPEG blobs in
`pa.binary` columns; tabular columns become fixed-size lists of `float32`.
Two index columns (`episode_idx`, `step_idx`) let the reader recover
episode boundaries by scanning a single column. It's the default for
`World.collect`.

**Why this is the default for WM training.** World-model and VLA workloads
mix large image trajectories with shuffled, random-access reads — a pattern
that suffers under one-file-per-episode formats. This backend was chosen
for two properties:

- **Column-projected reads.** `keys_to_load=['pixels', 'action']` fetches
  only those columns; unused fields (`state`, `reward`, additional cameras)
  are never read or decoded.
- **Streaming from object storage.** `path` accepts an `s3://`, `gs://`,
  or `az://` URI (credentials via `connect_kwargs`); the reader pulls byte
  ranges on demand, so a training run does not need a local copy of a
  multi-TB dataset.

Combined with row-level random access via `episode_idx` / `step_idx`,
shuffled DataLoader workers stay fed without per-episode file-open
overhead — which is the bottleneck at the dataset sizes typical of
modern VLA training.

**Layout:**

```text
dataset.lance/         # LanceDB table directory
├── _versions/         # transaction log
├── data/              # column fragments (Arrow IPC)
└── _indices/          # secondary indexes (none by default)
```

**Read:**

```python
import stable_worldmodel as swm

ds = swm.data.load_dataset('data/pusht.lance', num_steps=16, frameskip=1,
                           keys_to_load=['pixels', 'action'])
```

**Write:**

```python
from stable_worldmodel.data import LanceWriter

with LanceWriter('data/pusht.lance') as w:           # mode='append' default
    for ep in episodes:
        w.write_episode(ep)
```

`World.collect(path, episodes=...)` defaults to this format.

!!! info ""
    Field names with `.` are renamed to `_` on disk because Lance reserves
    `.` as a struct-path separator (`pixels.top` → `pixels_top`). The
    reader still references columns by their renamed name.
///

/// tab | HDF5
The **`hdf5`** format stores everything in a single `.h5` file with one
dataset per column plus the `ep_len`/`ep_offset` index. Useful when you
want a portable single-file artifact.

**Layout:**

```text
dataset.h5
├── pixels      # (Total_Steps, H, W, C) uint8
├── action      # (Total_Steps, Action_Dim) float32
├── reward      # (Total_Steps,) float32
├── ep_len      # (Num_Episodes,) int32
└── ep_offset   # (Num_Episodes,) int64
```

**Read:**

```python
import stable_worldmodel as swm

ds = swm.data.load_dataset('data/pusht.h5', num_steps=16, frameskip=1,
                           keys_to_load=['pixels', 'action'])
```

**Write:**

```python
from stable_worldmodel.data import HDF5Writer

with HDF5Writer('data/pusht.h5') as w:
    for ep in episodes:                      # ep = {col: [step_arr, ...]}
        w.write_episode(ep)
```

To use this format with `World.collect`, pass `format='hdf5'` explicitly.
///

/// tab | Folder
The **`folder`** format keeps tabular columns as `.npz` arrays and image
columns as one JPEG per step. It's great when you want to inspect frames
on disk or stream a few keys without paying HDF5's open cost.

**Layout:**

```text
dataset/
├── ep_len.npz              # (N,)  int32
├── ep_offset.npz           # (N,)  int64
├── action.npz              # (Total_Steps, A)
├── reward.npz              # (Total_Steps,)
└── pixels/                 # one image per step
    ├── ep_0_step_0.jpeg
    ├── ep_0_step_1.jpeg
    └── ...
```

**Read:** image columns are inferred from subdirectories, so `folder_keys`
is rarely needed.

```python
ds = swm.data.load_dataset('data/pusht_folder/', num_steps=4)
```

**Write:** any uint8 `(H, W, 3)` or `(H, W, 1)` array is auto-detected
as an image column and saved as JPEG.

```python
from stable_worldmodel.data import FolderWriter

with FolderWriter('data/pusht_folder') as w:
    w.write_episode({'pixels': frames, 'action': actions})
```
///

/// tab | Video
The **`video`** format is identical to `folder` for tabular columns, but
encodes each image column as one MP4 per episode. Frames are decoded with
[`decord`](https://github.com/dmlc/decord), which makes it a good fit for
long episodes where storing raw JPEGs is wasteful.

**Layout:**

```text
dataset/
├── ep_len.npz, ep_offset.npz, action.npz, ...
└── video/
    ├── ep_0.mp4
    └── ep_1.mp4
```

**Read / Write:**

```python
ds = swm.data.load_dataset('data/pusht_video/', num_steps=8)

# direct write
from stable_worldmodel.data import VideoWriter
with VideoWriter('data/pusht_video', fps=30, codec='libx264') as w:
    w.write_episode(episode)

# or via World
world.collect('data/pusht_video', episodes=100, format='video')
```

!!! info ""
    `video` requires the optional `decord` dependency for reading and
    `imageio` (with an FFmpeg backend) for writing.
///

/// tab | LeRobot
The **`lerobot`** format is a read-only adapter over
[`lerobot.datasets.LeRobotDataset`](https://github.com/huggingface/lerobot).
It's identified by the `lerobot://` scheme and exposes the same episode-based
API as the native SWM datasets: by default the primary camera is mapped to
`pixels`, `action` to `action`, and `observation.state` to `proprio`.

```python
ds = swm.data.load_dataset(
    'lerobot://lerobot/pusht',
    primary_camera_key='observation.images.top',  # → 'pixels'
    num_steps=8,
    keys_to_load=['pixels', 'action', 'proprio', 'ep_idx', 'step_idx'],
    keys_to_cache=['action', 'proprio', 'ep_idx', 'step_idx'],
)
```

!!! info ""
    LeRobot support is feature-gated to **Python 3.12+** because the upstream
    `lerobot` package requires it. Install with
    `pip install 'stable-worldmodel[format]'`. There is no `lerobot` writer —
    mapping arbitrary `World` info dicts onto LeRobot's schema is not supported.
///

/// tab | Goal-Conditioned
**`GoalDataset`** wraps any of the formats above to add a sampled goal
observation per item, for goal-conditioned learning. Goals are drawn from
one of four buckets (random, geometric future, uniform future, current)
according to a probability vector.

```python
from stable_worldmodel.data import GoalDataset

base = swm.data.load_dataset('data/pusht.h5', num_steps=4)
goal = GoalDataset(
    base,
    goal_probabilities=(0.3, 0.5, 0.0, 0.2),  # random, geom. future, uniform future, current
    gamma=0.99,
    seed=42,
)
item = goal[0]                                  # adds 'goal_pixels', 'goal_proprio'
```
///

## **[ Write Modes ]**

Every built-in writer accepts a standard `mode` kwarg with three values
(default: `'append'`):

| mode          | when target exists                            | when target is missing |
|---------------|-----------------------------------------------|------------------------|
| `'append'`    | extend with new episodes (validates schema)   | create from scratch    |
| `'overwrite'` | drop existing data, then write fresh          | create from scratch    |
| `'error'`     | raise `FileExistsError`                       | create from scratch    |

`'append'` is the default so that re-running a collection script naturally
extends the dataset rather than failing or wiping prior work. Lengths,
offsets, and episode indexes are resumed from the on-disk state. Image
columns continue to use the next available `ep_idx` (or row offset) so
appended frames don't clobber existing files.

Schema is validated against the on-disk dataset before any new bytes are
written. A column added, removed, retyped (image vs. tabular), or with a
mismatched per-step shape raises a clear `ValueError`:

```python
from stable_worldmodel.data import HDF5Writer

with HDF5Writer('data/pusht.h5') as w:               # extends existing
    w.write_episode(ep)

with HDF5Writer('data/pusht.h5', mode='overwrite') as w:  # starts fresh
    w.write_episode(ep)

with HDF5Writer('data/pusht.h5', mode='error') as w:      # raises if it exists
    w.write_episode(ep)
```

The same `mode` kwarg works for `FolderWriter`, `VideoWriter`, and
`LanceWriter`, and is forwarded by `World.collect(...)` and
`swm.data.convert(...)` via their writer-kwarg passthrough.

## **[ Converting Between Formats ]**

`convert()` walks each episode of a source dataset and writes it through the
writer of `dest_format`. Source format is autodetected unless you pass
`source_format=`.

```python
from stable_worldmodel.data import convert

# HDF5 → MP4 directory (fps forwarded to VideoWriter)
convert('data/pusht.h5', 'data/pusht_video',
        dest_format='video', fps=30)

# Folder → HDF5 (good for shrinking many JPEGs into one file)
convert('data/pusht_folder', 'data/pusht.h5', dest_format='hdf5')
```

This composes with `load_dataset`'s resolution rules, so you can convert
straight from a HuggingFace repo or a `lerobot://` URL:

```python
convert('lerobot://lerobot/pusht', 'data/pusht_local',
        source_format='lerobot', dest_format='video',
        primary_camera_key='observation.images.top')
```

## **[ Registering a Custom Format ]**

A format is just a class with three classmethods. Decorate it with
`@register_format` and the rest of the stack picks it up.

```python
from stable_worldmodel.data import Format, register_format
from stable_worldmodel.data.dataset import Dataset

@register_format
class Parquet(Format):
    name = 'parquet'

    @classmethod
    def detect(cls, path):
        from pathlib import Path
        return Path(path).suffix == '.parquet'

    @classmethod
    def open_reader(cls, path, **kw):
        return ParquetDataset(path, **kw)        # subclass of Dataset

    @classmethod
    def open_writer(cls, path, **kw):
        return ParquetWriter(path, **kw)          # __enter__/__exit__/write_episode
```

Once imported, your format is usable everywhere:

```python
swm.data.load_dataset('foo.parquet')                   # reader
world.collect('foo.parquet', episodes=10, format='parquet')  # writer
swm.data.list_formats()         # ['hdf5', 'folder', 'video', 'lerobot', 'parquet']
```

Read-only formats simply omit `open_writer`; write-only formats omit
`open_reader`. Both calls raise a clear error by default. If your writer
should participate in the standard mode contract, accept a `mode` kwarg and
delegate validation to `validate_write_mode`:

```python
from stable_worldmodel.data.format import validate_write_mode, WRITE_MODES

class ParquetWriter:
    def __init__(self, path, *, mode: str = 'append'):
        validate_write_mode(mode)             # rejects values outside WRITE_MODES
        self.mode = mode
        ...
```

## **[ Base Class ]**

::: stable_worldmodel.data.dataset.Dataset
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.data.dataset.Dataset.__getitem__
::: stable_worldmodel.data.dataset.Dataset.load_episode
::: stable_worldmodel.data.dataset.Dataset.load_chunk

## **[ Implementations ]**

::: stable_worldmodel.data.HDF5Dataset
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.data.FolderDataset
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.data.VideoDataset
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.data.ImageDataset
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.data.LeRobotAdapter
    options:
        heading_level: 3
        members: false
        show_source: false

## **[ Wrappers ]**

::: stable_worldmodel.data.GoalDataset
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.data.MergeDataset
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.data.ConcatDataset
    options:
        heading_level: 3
        members: false
        show_source: false

## **[ Format Registry ]**

::: stable_worldmodel.data.format.Format
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.data.format.register_format
::: stable_worldmodel.data.format.list_formats
::: stable_worldmodel.data.format.get_format
::: stable_worldmodel.data.format.detect_format

## **[ Top-Level Helpers ]**

::: stable_worldmodel.data.load_dataset
::: stable_worldmodel.data.convert
