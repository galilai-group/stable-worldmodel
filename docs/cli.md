---
title: CLI Reference
summary: swm command-line interface reference
sidebar_title: CLI
---

After installing `stable-worldmodel`, the `swm` command is available to inspect datasets, environments, and checkpoints without writing any Python code.

## `swm datasets`

List all datasets stored in your cache directory (`$STABLEWM_HOME`, defaults to `~/.stable_worldmodel/`).

```bash
swm datasets
```

```
               Datasets in ~/.stable_worldmodel/
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃ Name                   ┃ Format ┃    Size ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│ pusht_expert_train     │ Lance  │ 812.3MB │
│ pusht_expert_val       │ Lance  │  81.3MB │
└────────────────────────┴────────┴─────────┘
```

## `swm inspect <name>`

Show detailed metadata for a dataset: number of episodes, step counts, episode length distribution, and all stored columns with their shapes and dtypes.

```bash
swm inspect pusht_expert_train
```

```
Name:      pusht_expert_train
Format:    Lance
Path:      ~/.stable_worldmodel/pusht_expert_train.lance
Size:      812.3 MB
Episodes:  2000
Steps:     297806
Ep length: 100 – 200

           Columns
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Column   ┃ Type                         ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ action   │ fixed_size_list<float>[2]    │
│ pixels   │ binary                       │
│ state    │ fixed_size_list<float>[5]    │
└──────────┴──────────────────────────────┘
```

## `swm preview <name>`

Render a random sample of episodes to MP4 for a quick look at a dataset — without transcoding the whole thing the way `swm convert -f video` does. Each episode is loaded on its own, so the cost stays low even for very large datasets.

```bash
swm preview pusht_expert_train              # 4 random episodes
swm preview pusht_expert_train -n 8 --open  # sample 8, then open the folder
swm preview pusht_expert_train --episodes 0,3,17
```

```
Previewing pusht_expert_train — episodes [12, 340, 1201, 1777] → ~/.stable_worldmodel/previews/pusht_expert_train
Wrote 4 video(s) to ~/.stable_worldmodel/previews/pusht_expert_train
```

One `ep<idx>.mp4` is written per sampled episode. For a multi-view dataset (several image columns, e.g. `pixels_wrist` + `pixels_exo`), the views are composed **side-by-side into a single labeled video per episode**.

| Option | Description |
| --- | --- |
| `-n, --num` | Number of episodes to sample (default `4`). |
| `--episodes` | Comma-separated episode indices to render exactly (overrides random sampling), e.g. `--episodes 0,3,17`. |
| `--seed` | Seed for the random sampling (default `0`), so a preview is reproducible. |
| `-o, --output` | Output directory. Defaults to `<cache>/previews/<name>`. |
| `--key` | Comma-separated image column(s) to render. Defaults to all detected image columns. |
| `--fps` | Frame rate of the written MP4s (default `15`). |
| `--open` | Open the output folder when done. |

Rendering MP4s needs the optional video stack; install it with `pip install "stable-worldmodel[format]"`.

## `swm merge <sources...>`

Concatenate several datasets into a single one. Episodes from each source are appended in the order given and renumbered into one contiguous episode range. Sources must share the same columns; a mismatch fails before anything is written.

```bash
swm merge shard0 shard1 shard2 -o combined
swm merge pusht_a pusht_b -o pusht_all -f lance --overwrite
```

```
Merging shard0, shard1, shard2 → combined (lance, mode=error)
Done. Output: ~/.stable_worldmodel/combined.lance
```

| Option | Description |
| --- | --- |
| `-o, --output` | Output dataset name (required). |
| `-f, --dest-format` | Output format. Defaults to the first source's detected format. |
| `--overwrite` | Replace the output dataset if it already exists. |
| `--mode` | Write mode: `error` \| `overwrite` \| `append`. Overrides `--overwrite`. |

## `swm envs`

List all environments registered by `stable-worldmodel`, grouped by action type.

```bash
swm envs
```

## `swm fovs <env>`

Display the factors of variation (FoV) for a given environment — the properties you can randomize at reset to study generalization and robustness.

```bash
swm fovs PushT-v1
# or with the full id:
swm fovs swm/PushT-v1
```

```
       Factors of Variation — swm/PushT-v1
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Factor                ┃ Type     ┃ Range           ┃ Default ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ agent.color           │ RGBBox   │ [0,255]^3       │ -       │
│ agent.scale           │ Box      │ [20.0, 60.0]    │ -       │
│ agent.shape           │ Discrete │ [0, 2]          │ -       │
│ block.color           │ RGBBox   │ [0,255]^3       │ -       │
│ background.color      │ RGBBox   │ [0,255]^3       │ -       │
└───────────────────────┴──────────┴─────────────────┴─────────┘
```

## `swm checkpoints`

List model checkpoints saved in your cache directory. Accepts an optional filter string (regex) to narrow results.

```bash
swm checkpoints
swm checkpoints pusht
```

## `swm --version`

Print the installed version.

```bash
swm --version
```
