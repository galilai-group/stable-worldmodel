---
title: Buffer
summary: Online history buffer for policies that need past observations
---

`HistoryBuffer` is a per-env ring buffer over batched info dicts. It is used internally by [`WorldModelPolicy`](policy.md) to feed strided history into the planner, but it can also be used standalone for any code path that consumes `(n_envs, ...)`-shaped step data and needs a sliding window over time.

## **[ How it works ]**

Each call to [`append`][stable_worldmodel.buffer.HistoryBuffer.append] takes a dict whose values have a leading env dim of size `n_envs` (e.g. `EnvPool`'s stacked infos with shape `(n_envs, 1, ...)`). The buffer slices the env dim and pushes one entry per env onto its corresponding deque.

[`get(n)`][stable_worldmodel.buffer.HistoryBuffer.get] returns the last `n` entries per env, strided by `action_block` env steps so history is surfaced at planning cadence regardless of frameskip. Output is in chronological order (oldest → newest). Pass `env_ids` to build history only for a subset of envs (e.g. the ones being re-planned).

### Warm-up

Each env is padded independently, so the output shape is always the same and envs never constrain each other (a freshly-reset env does not shrink its neighbors' history):

- Strided (non-block) keys are left-padded by repeating the env's **oldest available entry**.
- Block keys are left-padded with **zero blocks**.

!!! warning "Warm-up output is partially synthetic"
    Padded entries are **fake**: until an env has lived `(n - 1) * action_block + 1` steps, the consumer — in planning, the world model — receives duplicated copies of the episode's first frame with zero actions between them, as if the env had been stationary before the episode began. This is a deliberate trade-off: training clips are never padded, so *any* warm-up scheme is off-distribution, and repeating the first frame (the repo's forward-fill precedent, cf. `PreJEPA._encode_video`) with zero actions is the self-consistent choice that keeps histories stackable across desynchronized envs.

`get` returns `None` only when some selected env is empty (or `n <= 0`).

### Output shapes

| Per-step value shape | Output shape |
|---|---|
| `(n_envs,)` | `(n_envs, n)` |
| `(n_envs, T, ...)` | `(n_envs, n * T, ...)` |
| `(n_envs, T, ...)` and key in `block_keys` | `(n_envs, n - 1, action_block * D)` where `D` is the flattened per-env per-step size |

`block_keys` is intended for actions when the planner operates at macro-action cadence. Block `i` concatenates (flat, chronological) the `action_block` raw entries recorded **between** strided frames `i` and `i + 1` — "the block leaving frame `i`" — matching the training convention that pairs `action[t]` with `frame[t]` (an env's `info['action']` at step `j` echoes the action executed *entering* step `j`, so the entries strictly after frame `i` up to and including frame `i + 1` are exactly the actions executed between the two frames). There are therefore `n - 1` blocks for `n` frames; block keys are omitted entirely when `n == 1`. NaN entries (the reset frame's action echo) are zeroed.

## **[ Example ]**

```python
import numpy as np
from stable_worldmodel.buffer import HistoryBuffer

# 2 envs, hold up to 5 env steps, return history at frameskip-2 cadence
buf = HistoryBuffer(n_envs=2, max_len=5, action_block=2, block_keys=('action',))

for t in range(5):
    buf.append({
        'pixels': np.full((2, 1, 64, 64, 3), t, dtype=np.uint8),  # (n_envs, T, H, W, C)
        'action': np.full((2, 1, 2), t, dtype=np.float32),        # (n_envs, T, D)
    })

out = buf.get(3)
# out['pixels'].shape == (2, 3, 64, 64, 3)   # strided frames: t=0, t=2, t=4
# out['action'].shape == (2, 2, 4)           # blocks between them:
#                                            #   [a_1,a_2] (leaving t=0) and
#                                            #   [a_3,a_4] (leaving t=2)

buf.reset(env_ids=[0])  # clear env 0 only (e.g. on episode reset)
```

## **[ Reference ]**

::: stable_worldmodel.buffer.HistoryBuffer
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.buffer.HistoryBuffer.append

::: stable_worldmodel.buffer.HistoryBuffer.get

::: stable_worldmodel.buffer.HistoryBuffer.reset
