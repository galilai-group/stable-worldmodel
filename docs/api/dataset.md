title: Dataset
summary: Dataset handling
---

`stable_worldmodel` provides a flexible dataset API that supports HDF5 files, LanceDB tables (local or remote), and folder-based layouts.

## **[ Storage Formats ]**

/// tab | HDF5 Format (Recommended)
The **`HDF5Dataset`** stores all data in a single `.h5` file. This is the default format for recording rollouts using `World.record_dataset`.

**File Structure:**
```
dataset_name.h5
├── pixels          # (Total_Steps, C, H, W) or (Total_Steps, H, W, C)
├── action          # (Total_Steps, Action_Dim)
├── reward          # (Total_Steps,)
├── terminated      # (Total_Steps,)
├── ep_len          # (Num_Episodes,) - Length of each episode
└── ep_offset       # (Num_Episodes,) - Start index of each episode
```

**Usage:**
```python
from stable_worldmodel.data import HDF5Dataset

dataset = HDF5Dataset(
    name="my_dataset",
    frameskip=1,
    num_steps=50  # Sequence length for training
)
```
///

/// tab | LanceDB Format
The **`LanceDataset`** streams directly from a Lance table (local path, `hf://` dataset, or any `s3://`/object-store URI). Each row represents a timestep and only the requested columns are scanned, making it ideal for training straight from cloud storage.

**Usage:**
```python
from stable_worldmodel.data import LanceDataset

dataset = LanceDataset(
    # Point at the full .lance table path — table name is inferred.
    uri='s3://my-bucket/lewm/lewm_pusht.lance',  # or ./lewm_lance/lewm_pusht.lance, hf://datasets/…
    num_steps=4,
    frameskip=5,
    keys_to_load=['pixels', 'action', 'proprio'],
    connect_kwargs={'aws_region': 'us-east-2'},  # optional credentials/region
)
# Image columns are auto-detected from the Arrow schema (any pa.binary column).
# Pass image_columns=[...] only to override that default.
# Equivalent two-argument form (if you want to share a database URI across tables):
# LanceDataset(uri='s3://my-bucket/lewm', table_name='lewm_pusht', ...)
```

!!! tip
    Lance tables store JPEG-compressed frames by default. Decoding happens inside each PyTorch DataLoader worker and the dataset never calls `.to_arrow()`—only `Permutation` and column scanners—so you can safely read from very large remote datasets without materializing them.
///

/// tab | Folder Format
The **`FolderDataset`** stores metadata in `.npz` files and heavy media (images) as individual files.

**File Structure:**
```
dataset_name/
├── ep_len.npz      # Contains 'arr_0': Array of episode lengths
├── ep_offset.npz   # Contains 'arr_0': Array of episode start offsets
├── action.npz      # Contains 'arr_0': Full array of actions
├── reward.npz      # Contains 'arr_0': Full array of rewards
└── pixels/         # Folder for image data
    ├── ep_0_step_0.jpg
    ├── ep_0_step_1.jpg
    └── ...
```

**Usage:**
```python
from stable_worldmodel.data import FolderDataset

dataset = FolderDataset(
    name="my_image_dataset",
    folder_keys=["pixels"]  # Keys to load from folders instead of .npz
)
```
///

/// tab | Video Format
The **`VideoDataset`** is a specialized `FolderDataset` that reads frames directly from MP4 files using `decord`. This saves significant disk space compared to storing individual images.

**File Structure:**
```
dataset_name/
├── ep_len.npz
├── ep_offset.npz
├── action.npz
└── video/          # Folder for video files
    ├── ep_0.mp4
    ├── ep_1.mp4
    └── ...
```

**Usage:**
```python
from stable_worldmodel.data import VideoDataset

dataset = VideoDataset(
    name="my_video_dataset",
    video_keys=["video"]
)
```
///

/// tab | Image Format
The **`ImageDataset`** is a convenience alias for `FolderDataset` with image defaults. It assumes 'pixels' is stored as individual image files.

**File Structure:**
```
dataset_name/
├── ep_len.npz
├── ep_offset.npz
├── action.npz
└── pixels/         # Folder for image files
    ├── ep_0_step_0.jpeg
    ├── ep_0_step_1.jpeg
    └── ...
```

**Usage:**
```python
from stable_worldmodel.data import ImageDataset

dataset = ImageDataset(
    name="my_image_dataset",
    image_keys=["pixels"]  # Default
)
```
///

/// tab | Goal-Conditioned
The **`GoalDataset`** wraps any dataset to add goal observations for goal-conditioned learning. Goals are sampled from random states, future states in the same episode, or the current state.

**Usage:**
```python
from stable_worldmodel.data import HDF5Dataset, GoalDataset

# Wrap any base dataset
base_dataset = HDF5Dataset(name="my_dataset", num_steps=50)
goal_dataset = GoalDataset(
    base_dataset,
    goal_probabilities=(0.3, 0.5, 0.2),  # (random, future, current)
    gamma=0.99,  # Discount for future sampling
    seed=42
)

# Items now include goal_pixels and goal_proprio keys
item = goal_dataset[0]
```
///

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

::: stable_worldmodel.data.LanceDataset
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

## **[ Wrappers ]**

::: stable_worldmodel.data.GoalDataset
    options:
        heading_level: 3
        members: false
        show_source: false

## **[ Utilities ]**

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

::: stable_worldmodel.data.create_dataset
    options:
        heading_level: 3
        members: false
        show_source: false

## **[ Converting to Lance ]**

Use the bundled CLI to convert any supported HDF5 dataset (local file, HuggingFace repo, or the built-in presets) into a LanceDB table:

```bash
python scripts/data/convert_to_lance.py \
  --dataset pusht \
  --lance-uri ./lewm_lance \
  --max-episodes 2 \
  --overwrite
```

Custom datasets can be converted by providing `--source /path/to/dataset.h5 --columns pixels action proprio` and an explicit `--table-name`.

Hydra configs stay unchanged: override Lance-specific keys at launch time, e.g.

```bash
python scripts/train/lewm.py \
  +data.dataset.uri=./lewm_lance/lewm_pusht.lance \
  data.dataset.keys_to_load='[pixels,action,proprio,state]'
```

Providing `uri` (or the explicit `uri`+`table_name` pair) automatically instantiates `LanceDataset`, so existing HDF5 configs remain backward compatible. The `.../foo.lance` shorthand infers the table name from the suffix — use a separate `table_name` only when you want to share a database URI across multiple tables.

---

## **[ Why Lance ]**

### No RAM cache required

`HDF5Dataset` needs `keys_to_cache` to hold columns like `action` and `proprio` in RAM, because random seeks into a compressed HDF5 file are too slow per-sample. This caps usable dataset size to available RAM.

`LanceDataset` reaches higher throughput without caching. `__getitems__` (PyTorch DataLoader ≥ 2.0) coalesces the entire batch into one columnar `take()` call — only the requested columns and rows are read, with no decompression of unneeded data. Datasets that do not fit in RAM work out of the box.

### Train directly from object storage

HDF5 on S3 is slow enough to be unusable without pre-downloading the file: every random-access seek triggers a separate HTTP range request. Lance on S3 issues **one** vectorised read per batch (one round-trip instead of `batch_size` sequential round-trips) and its columnar fragment layout minimises bytes transferred.

Benchmark on the Tworoom dataset, 4 DataLoader workers, batch size 64, local Mac hardware:

| Backend | Notes | samples / s |
|---|---|---|
| HDF5 local | action + proprio cached | 1 615 |
| HDF5 local | no cache | 1 434 |
| **Lance local** | **no cache** | **3 758** |
| Lance local | action + proprio cached | 3 897 |
| **Lance S3** | **no cache** | **2 261** |
| Lance S3 | action + proprio cached | 2 208 |
| HDF5 S3 | action + proprio cached | 832 |
| HDF5 S3 | no cache | 24 |

Note: these results might vary depending on your bucket location.
Lance local is **2.3× faster than HDF5 local** (cached). Lance S3 (no cache, no download) is **94× faster than HDF5 S3** (no cache) and still **40% faster than HDF5 local cached** — meaning you can start training immediately on a spot instance with no local disk and no data download step. 

### Zero-copy dataset evolution

Lance versions the table on every write. Common curation tasks cost no extra storage:

- **Add a column** (e.g. a new reward signal): `table.add_columns(...)` writes only the new fragment files.
- **Delete episodes** (e.g. failed rollouts): `table.delete("episode_idx = 42")` records a deletion in a new version file; no rows are moved.
- **Time-travel**: `db.open_table("name", version=N)` reopens any prior snapshot of the dataset.
