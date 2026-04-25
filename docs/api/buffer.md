---
title: Buffer
summary: Online history buffer for policies that need past observations
---

`HistoryBuffer` is a per-env ring buffer over batched info dicts. It is used internally by [`WorldModelPolicy`](policy.md) to feed strided history into the planner, but it can also be used standalone for any code path that consumes `(n_envs, ...)`-shaped step data and needs a sliding window over time.

## **[ How it works ]**

Each call to [`append`][stable_worldmodel.buffer.HistoryBuffer.append] takes a dict whose values have a leading env dim of size `n_envs` (e.g. `EnvPool`'s stacked infos with shape `(n_envs, 1, ...)`). The buffer slices the env dim and pushes one entry per env onto its corresponding deque.

[`get(n)`][stable_worldmodel.buffer.HistoryBuffer.get] returns up to the last `n` entries per env, strided by `action_block` env steps so history is surfaced at planning cadence regardless of frameskip. Output is in chronological order (oldest → newest).

### Warm-up

`get` returns the largest `k ≤ n` that fits in every env's buffer. The shape grows from 1 up to `n` over the first warm-up window:

- Strided (non-block) keys reach `k = n` after `(n - 1) * action_block + 1` env steps.
- Block keys reach `k = n` after `n * action_block` env steps (full windows are required, so `k = min(n, min_len // action_block)`).

`get` returns `None` only when some env is empty (or `n <= 0`).

### Output shapes

| Per-step value shape | Output shape |
|---|---|
| `(n_envs,)` | `(n_envs, k)` |
| `(n_envs, T, ...)` | `(n_envs, k * T, ...)` |
| `(n_envs, T, ...)` and key in `block_keys` (with `action_block > 1`) | `(n_envs, k, action_block * D)` where `D` is the flattened per-env per-step size |

`block_keys` is intended for actions when the planner operates at macro-action cadence: instead of striding (one action per stride point), it concatenates the `action_block` raw entries within each window into a single flat block. With `block_keys` and `action_block > 1`, `k = min(n, min_len // action_block)` so only **full** blocks are returned.

## **[ Example ]**

```python
import numpy as np
from stable_worldmodel.buffer import HistoryBuffer

# 2 envs, hold up to 5 env steps, return history at frameskip-2 cadence
buf = HistoryBuffer(n_envs=2, max_len=5, action_block=2, block_keys=('action',))

for t in range(4):
    buf.append({
        'pixels': np.full((2, 1, 64, 64, 3), t, dtype=np.uint8),  # (n_envs, T, H, W, C)
        'action': np.full((2, 1, 2), t, dtype=np.float32),        # (n_envs, T, D)
    })

out = buf.get(2)
# out['pixels'].shape == (2, 2, 64, 64, 3)   # strided: t=1 and t=3
# out['action'].shape == (2, 2, 4)           # blocked:   [a_0,a_1] and [a_2,a_3]

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
