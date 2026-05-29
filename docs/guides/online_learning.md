---
title: Online Learning with ReplayBuffer
summary: Fill a replay buffer from rollouts, train an agent on it, and dump it to disk — all without leaving the Dataset/Writer abstractions used by the rest of the library.
---

`ReplayBuffer` is an in-memory ring-storage buffer that **doubles as a `Dataset` and a `Writer`**. That dual identity is the whole point: the same object can be filled by a rollout (`Writer` side) and iterated by a `DataLoader` for training (`Dataset` side), so you can interleave collection and learning without copying data or maintaining a parallel structure.

## **[ Mental Model ]**

```text
        ┌─────────────────────────┐         ┌──────────────────────┐
        │      rollout / env      │         │     training loop    │
        │  (acts as a *producer*) │         │  (acts as *consumer*)│
        └────────────┬────────────┘         └──────────▲───────────┘
                     │ writer.write_episode(ep)        │ buffer.sample(B,H,step=k)
                     ▼                                 │
              ┌─────────────────────────────────────────────┐
              │              ReplayBuffer                   │
              │  per-column ring arrays  (max_steps cap)    │
              │  whole-episode FIFO eviction                │
              │  clips never cross episode boundaries       │
              └────────────────────┬────────────────────────┘
                                   │ buffer.dump(path, format=...)
                                   ▼
                       ┌──────────────────────┐
                       │   on-disk dataset    │
                       │ (any registered      │
                       │  Format: hdf5/folder │
                       │  /lance/video/...)   │
                       └──────────────────────┘
```

Every operation on the left arrow uses the standard `Writer` protocol (`write_episode`, `__enter__`, `__exit__`). Every operation on the right arrow uses the standard `Dataset` protocol (`__len__`, `__getitem__`) plus an explicit `sample()` for batched access with a custom sampler. Persistence reuses the registered format writers — there's nothing replay-buffer-specific in the on-disk artifact, and a dumped buffer is loadable with `swm.data.load_dataset`.

## **[ Quick Tour ]**

```python
from torch.utils.data import DataLoader

import stable_worldmodel as swm
from stable_worldmodel.data import ReplayBuffer

# 1) Create a buffer. max_steps caps total in-RAM transitions.
buf = ReplayBuffer(max_steps=100_000, history_len=4)

# 2) Fill it. World.collect accepts the buffer directly as a writer.
world = swm.World('swm/PushT-v1', num_envs=4, image_shape=(64, 64))
world.set_policy(swm.policy.RandomPolicy(seed=0))
world.collect(writer=buf, episodes=20, seed=0)

# 3) Train on it via a DataLoader (Dataset path).
loader = DataLoader(buf, batch_size=64, shuffle=True)
for batch in loader:
    train_step(batch)              # batch[col] shape: (64, history_len, ...)

# 4) Or sample directly (custom-sampler path).
batch = buf.sample(batch_size=64)  # {col: (64, history_len, ...)} numpy arrays

# 5) Dump to disk at any point — picks any registered Format.
buf.dump('runs/replay.h5', format='hdf5')
```

## **[ Filling the Buffer ]**

The buffer implements the same `Writer` protocol as `HDF5Writer`, `FolderWriter`, etc. — `write_episode(ep_dict)` plus the `with`-statement entry/exit hooks. So anywhere a `Writer` fits, the buffer fits.

### Via `World.collect`

`World.collect` accepts the buffer directly as its `writer=` argument — the same call pattern as collecting to disk, but the destination is your in-memory buffer instead of an `.h5`/`.lance`/folder dataset. This is the recommended path because `World` already handles batched env stepping, episode boundaries, and the per-step → per-episode buffering for you.

```python
import stable_worldmodel as swm
from stable_worldmodel.data import ReplayBuffer

world = swm.World('swm/PushT-v1', num_envs=4, image_shape=(64, 64))
world.set_policy(my_policy)

buf = ReplayBuffer(max_steps=200_000, history_len=4)
world.collect(writer=buf, episodes=20, seed=0)
```

`world.collect(...)` is mutually exclusive in its destination: pass **either** `path=...` (with optional `format=`) **or** `writer=...`, not both. Every column the env emits in `infos` becomes a column in the buffer, schema-inferred from the first completed episode.

### Manual rollout loop

If you're rolling out outside `World` (e.g. a custom env stepper, a multi-agent setup, or experiments with non-standard episode-termination logic), call `write_episode` yourself when an episode finishes:

```python
buf = ReplayBuffer(max_steps=200_000, history_len=4)

obs, info = env.reset()
ep = {'pixels': [], 'action': [], 'reward': []}
while training:
    a = policy.act(obs)
    next_obs, r, terminated, truncated, _ = env.step(a)
    ep['pixels'].append(obs['pixels'])
    ep['action'].append(a)
    ep['reward'].append(np.float32(r))
    obs = next_obs
    if terminated or truncated:
        buf.write_episode(ep)         # episode complete → into the buffer
        ep = {k: [] for k in ep}
        obs, info = env.reset()
```

`ep_dict` may use either lists of per-step arrays (what `World.collect` produces) or bulk arrays of shape `(ep_len, ...)`. The buffer accepts both; lengths must agree across columns.

### Schema is locked after the first episode

The first `write_episode` call defines the column layout (which keys, what shapes, what dtypes). Subsequent writes are validated against it; a missing column, an extra column, a wrong per-step shape, or a length mismatch raise `ValueError` before any data is touched.

```python
buf.write_episode({'pixels': frames, 'action': actions})  # establishes schema
buf.write_episode({'pixels': frames})                     # ValueError: missing 'action'
```

### Capacity & eviction

`max_steps` is a hard cap on total stored steps. When a new episode wouldn't fit, the buffer evicts whole oldest episodes until it does. Clips are guaranteed never to span across episode boundaries, so eviction is always at episode granularity — no torn samples.

```python
buf = ReplayBuffer(max_steps=50, history_len=2)
buf.write_episode(_ep(30))   # buf: [30 steps]
buf.write_episode(_ep(15))   # buf: [30, 15] — 45 steps used
buf.write_episode(_ep(20))   # buf: [15, 20] — oldest episode evicted to fit
```

Episodes longer than `max_steps` are rejected with a clear error rather than silently truncated.

## **[ Sampling ]**

Two sampling paths are available:

### Path A — DataLoader (Dataset interface)

`ReplayBuffer` subclasses `Dataset`. `__getitem__(idx)` returns one clip of `history_len` consecutive steps; `__len__` returns the total number of valid clips currently stored. PyTorch's `DataLoader` does the rest:

```python
loader = DataLoader(buf, batch_size=64, shuffle=True, num_workers=2)
for batch in loader:
    train_step(batch)
```

`__getitem__` is **O(log N)** in the number of stored episodes thanks to the cached cumulative-clip-starts array, so DataLoader iteration scales to large buffers.

### Path B — `buffer.sample()` (custom sampler)

For more control — schedules, curricula, prioritized replay — call `sample(batch_size, history_len, step=...)` directly. The sampler logic is *injected at construction*; the buffer just plumbs the current step through.

```python
def my_sampler(step, buffer, batch_size, history_len):
    """fn(step, buffer, batch_size, history_len) -> indices of valid clips."""
    n = buffer.num_valid_ends(history_len)
    return np.random.randint(0, n, batch_size)

buf = ReplayBuffer(max_steps=100_000, history_len=4, sampler=my_sampler)
batch = buf.sample(batch_size=64)
```

The sampler returns flat indices into `[0, buffer.num_valid_ends(history_len))`. The buffer maps those to the corresponding clips and stacks them into `(batch_size, history_len, ...)` arrays. All sampling logic the user might want — uniform, recency-biased, prioritized, scheduled — lives in this one function.

#### Step-conditioned sampling

The first argument to the sampler is "the current step." By default, the buffer auto-increments an internal counter every time `sample()` is called; pass `step=...` to override per call:

```python
buf.sample(batch_size=64)                      # step = 0, then 1, then 2, ...
buf.sample(batch_size=64, step=global_step)    # bypass the counter
```

This is what makes **schedules** clean. For example, focus on recent data early in training (when the policy is still bad) and drift toward uniform later (when older data is still informative):

```python
def warmup_then_uniform(step, buffer, batch_size, history_len):
    n = buffer.num_valid_ends(history_len)
    if step < 10_000:
        # First 10k samples: draw from the most-recent 1k clips only.
        return np.random.randint(max(0, n - 1000), n, batch_size)
    return np.random.randint(0, n, batch_size)

buf = ReplayBuffer(max_steps=200_000, history_len=4, sampler=warmup_then_uniform)
```

The sampler can read live buffer state (`num_valid_ends`, `num_episodes`, `num_steps_stored`, `lengths`) — useful when the rule depends on how filled the buffer is, not just on the step count.

#### Returning weights via importance sampling

Prioritized replay can be expressed by drawing indices proportional to a weight vector you maintain externally:

```python
class PrioritizedSampler:
    def __init__(self, alpha=0.6):
        self.priorities = np.zeros(0, dtype=np.float32)
        self.alpha = alpha

    def __call__(self, step, buffer, batch_size, history_len):
        n = buffer.num_valid_ends(history_len)
        # In a real implementation you'd update self.priorities to length n.
        p = self.priorities[:n] ** self.alpha
        p = p / p.sum()
        return np.random.choice(n, size=batch_size, p=p, replace=True)

buf = ReplayBuffer(max_steps=200_000, history_len=4,
                   sampler=PrioritizedSampler(alpha=0.6))
```

## **[ Frameskip Semantics ]**

`frameskip > 1` strides observations and **keeps actions dense** — matching the convention used by `FolderDataset` and `Dataset.__getitem__`. With `history_len=H` and `frameskip=K`:

- Observation columns (`pixels`, `proprio`, …) → shape `(H, ...)`, sampled at positions `0, K, 2K, …, (H-1)K` within the clip.
- The `action` column → shape `(H, K * action_dim)`, *all* `H * K` raw actions reshaped so each "step" carries the action chunk taken to advance to the next observation.

```python
buf = ReplayBuffer(max_steps=10_000, history_len=4, frameskip=2)
# clip = buf[0]:
#   clip['pixels'].shape   == (4, ...)        # 4 strided observations
#   clip['action'].shape   == (4, 2 * A_dim)  # action chunks, one per obs step
```

If you don't use frameskipped rollouts, leave it at the default `1` — strided/dense behavior coincide and there's no overhead.

## **[ Persistence ]**

`buffer.dump(path, format=...)` walks the current episodes through any registered `Format`'s writer:

```python
buf.dump('checkpoints/replay.h5', format='hdf5')                 # single file
buf.dump('checkpoints/replay_folder', format='folder')           # JPEGs + .npz
buf.dump('checkpoints/replay.lance', format='lance')             # column store
```

The output is **just a regular dataset** — re-loadable via `swm.data.load_dataset(...)` like anything else, with no buffer-specific metadata involved. This is also how you snapshot a buffer mid-run for offline replays, share an experience set with a teammate, or convert in-memory experience to a permanent training corpus.

`mode='overwrite'` is the default for `dump` (the typical use is checkpointing the latest snapshot); pass `mode='append'` to extend an existing on-disk dataset, or `mode='error'` to refuse to clobber.

## **[ Online RL: Putting It Together ]**

A typical online-learning loop interleaves "collect a few episodes" and "train a few steps":

```python
import torch
import stable_worldmodel as swm
from stable_worldmodel.data import ReplayBuffer

world = swm.World('swm/PushT-v1', num_envs=4, image_shape=(64, 64))
world.set_policy(policy)

buf = ReplayBuffer(max_steps=200_000, history_len=4)

# Warm-start from a prior dataset, if you have one.
warm_start = swm.data.load_dataset('runs/prior.h5', num_steps=4)
for ep_idx in range(len(warm_start.lengths)):
    buf.write_episode(warm_start.load_episode(ep_idx))

global_step = 0
while global_step < TOTAL_STEPS:
    # 1) Collect: roll out the current policy for K episodes into the buffer.
    world.collect(writer=buf, episodes=K, seed=global_step)

    # 2) Train: M gradient updates on freshly mixed data.
    for _ in range(M):
        batch = buf.sample(batch_size=256, step=global_step)
        policy.update(batch)
        global_step += 1

    # 3) (Optional) checkpoint replay state alongside model weights.
    if global_step % CHECKPOINT_EVERY == 0:
        buf.dump(f'runs/step_{global_step:06d}.h5', format='hdf5')
        torch.save(policy.state_dict(), f'runs/step_{global_step:06d}.pt')
```

`world.collect(writer=buf, ...)` shares the same policy you're training, so each collect phase produces on-policy episodes for the next training phase. Switching to off-policy data (a fixed expert, a different policy snapshot) is just a different `world.set_policy(...)` call before `collect`.

A few practical notes:

- **`max_steps` is a memory budget**, not a hyperparameter. Pick it from the size of one transition × how many you can hold in RAM. For `(224, 224, 3)` uint8 pixels, 200k steps is ~30 GB — adjust accordingly.
- **Pass `step=global_step` explicitly** if you also use the buffer through a `DataLoader`. The internal counter only tracks `sample()` calls; sharing with the loader path can desync it.
- **Snapshot before risky changes.** Replay buffer state is hard to reproduce — `dump()` is fast and the result is a normal dataset, so checkpoint generously.
- **`buf.clear()` reuses the column allocations** — no reallocation, no GC pressure. Use it when starting a new phase of training rather than constructing a new buffer.

## **[ When Not to Use ReplayBuffer ]**

This buffer is in-memory only. If your experience volume exceeds RAM, you have two reasonable options:

1. **Periodic flush + reload.** Periodically call `dump(...)` and clear the buffer, then load the on-disk dataset alongside the buffer for training. The buffer holds recent experience; older experience lives on disk.
2. **Use a `Format` writer directly.** If you don't need fast random access into the live experience and want crash-safety, write directly through `HDF5Writer` / `LanceWriter` / `FolderWriter` and load the result as a regular dataset.

The buffer also doesn't support intra-episode visibility: only complete episodes (terminated or truncated) become sample-able. For very long episodes where you want to sample mid-flight, either chunk them into shorter logical "sub-episodes" before writing, or maintain a parallel structure for the in-progress episode.

## **[ API ]**

::: stable_worldmodel.data.ReplayBuffer
    options:
        heading_level: 3
        members: false
        show_source: false

::: stable_worldmodel.data.buffer.ReplayBuffer.write_episode
::: stable_worldmodel.data.buffer.ReplayBuffer.write_episodes
::: stable_worldmodel.data.buffer.ReplayBuffer.sample
::: stable_worldmodel.data.buffer.ReplayBuffer.num_valid_ends
::: stable_worldmodel.data.buffer.ReplayBuffer.dump
::: stable_worldmodel.data.buffer.ReplayBuffer.episodes
::: stable_worldmodel.data.buffer.ReplayBuffer.clear
