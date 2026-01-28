title: Dataset
---

# Dataset Handling

`stable_worldmodel` provides a flexible dataset API that supports both HDF5-based storage (for speed and compactness) and Folder-based storage.

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

## **[ Base Classes ]**

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
