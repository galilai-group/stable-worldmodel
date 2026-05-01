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
