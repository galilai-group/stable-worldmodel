title: Collect Dataset
summary: Create a small PushT dataset, inspect it, load it for training, and convert it to another format.
sidebar_title: New Dataset
---

This tutorial walks through the first stage of a world-model experiment:
recording trajectories into a dataset that can be loaded by PyTorch training
code.

You will:

- create a vectorized `World`,
- collect PushT trajectories with a weak expert policy,
- randomize selected factors of variation,
- inspect the dataset from the CLI,
- load fixed-length clips for training,
- convert the dataset to another storage format.

## Install

Use the full extra for this tutorial because it needs environment and dataset
format dependencies:

```bash
pip install 'stable-worldmodel[all]'
```

Datasets are stored under `$STABLEWM_HOME/datasets`. If the variable is not
set, the default root is `~/.stable_worldmodel`.

```bash
export STABLEWM_HOME=$PWD/.stablewm
mkdir -p "$STABLEWM_HOME/datasets"
```

## Inspect the environment

List the environments registered by `stable_worldmodel`:

```bash
swm envs
```

Inspect PushT's factors of variation:

```bash
swm fovs swm/PushT-v1
```

Factors of variation are reset-time controls such as object colors, object
positions, sizes, shapes, and physics parameters. Passing a factor name in
`options={"variation": ...}` samples that factor for each episode.

## Collect trajectories

Create `collect_pusht_tutorial.py`:

```python
from pathlib import Path
import os

import stable_worldmodel as swm
from stable_worldmodel.envs.pusht import WeakPolicy


root = Path(os.environ.get('STABLEWM_HOME', Path.home() / '.stable_worldmodel'))
dataset_path = root / 'datasets' / 'tutorial_pusht.lance'

world = swm.World(
    'swm/PushT-v1',
    num_envs=8,
    image_shape=(96, 96),
    max_episode_steps=100,
)

world.set_policy(WeakPolicy(dist_constraint=100, seed=0))

world.collect(
    dataset_path,
    episodes=64,
    seed=0,
    options={
        'variation': [
            'agent.start_position',
            'block.start_position',
            'block.angle',
            'agent.color',
            'block.color',
        ],
    },
)

world.close()
print(f'wrote {dataset_path}')
```

Run it:

```bash
python collect_pusht_tutorial.py
```

The default dataset format is `lance`, which is the recommended format for
training workloads with shuffled image trajectories. To write HDF5 instead,
pass `format='hdf5'` and use a `.h5` path.

```python
world.collect(
    root / 'datasets' / 'tutorial_pusht.h5',
    episodes=64,
    seed=0,
    format='hdf5',
)
```

## Inspect the dataset

Datasets in `$STABLEWM_HOME/datasets` can be inspected by name:

```bash
swm inspect tutorial_pusht
```

The output includes the detected format, episode count, step count, and stored
columns. All numeric entries from `world.infos` become columns. The standard
columns include:

| Column | Meaning |
|---|---|
| `pixels` | Rendered RGB frame resized by `World(image_shape=...)`. |
| `action` | Action applied at the current step. |
| `reward` | Environment reward. |
| `terminated` / `truncated` | Episode-end flags. |
| `state` / `proprio` | Low-dimensional state when the environment provides it. |
| `step_idx` | Step index within the episode. |
| `variation.*` | Variation values requested in reset options. |

For Lance datasets, dots in field names are normalized to underscores on disk
because Lance reserves `.` for struct paths.

## Load training clips

Load fixed-length clips with `swm.data.load_dataset()`:

```python
import stable_worldmodel as swm

dataset = swm.data.load_dataset(
    'tutorial_pusht.lance',
    num_steps=8,
    frameskip=1,
    keys_to_load=['pixels', 'action', 'state'],
)

sample = dataset[0]
print(sample.keys())
print(sample['pixels'].shape)  # (T, C, H, W)
print(sample['action'].shape)  # (T, action_dim)
```

Then use the dataset with a standard PyTorch `DataLoader`:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

batch = next(iter(loader))
print(batch['pixels'].shape)  # (B, T, C, H, W)
```

`num_steps` is the temporal window returned by `__getitem__`. `frameskip`
controls the stride between observation frames while keeping action sequences
dense.

## Convert formats

Convert the Lance dataset to a video-backed dataset:

```bash
swm convert tutorial_pusht tutorial_pusht_video --dest-format video
```

Or convert in Python:

```python
from pathlib import Path
import os

import stable_worldmodel as swm

root = Path(os.environ.get('STABLEWM_HOME', Path.home() / '.stable_worldmodel'))

swm.data.convert(
    root / 'datasets' / 'tutorial_pusht.lance',
    root / 'datasets' / 'tutorial_pusht_video',
    dest_format='video',
    fps=30,
)
```

Use `video` when you want compact, easy-to-watch rollouts. Use `lance` when
you want fast random access during training.

## Common checks

- If `swm inspect tutorial_pusht` cannot find the dataset, confirm
  `$STABLEWM_HOME` is the same in both shells.
- If a policy returns the wrong shape, check `world.envs.action_space.shape`.
  Vectorized worlds expect actions shaped like the batched action space.
- If you need exact reproducibility for OOD experiments, include every factor
  you randomize in `options['variation']`; only watched variation values are
  written into the dataset.
- If you collect outside `$STABLEWM_HOME/datasets`, pass an absolute path to
  `swm.data.load_dataset()`.
