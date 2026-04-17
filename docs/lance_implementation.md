# LanceDB Dataset Support — Implementation Guide

This document walks through the design decisions and code paths added in the `lance` branch. It is meant to help you read and reason about the code, not as end-user documentation.

---

## 1. Why Lance at all?

HDF5 stores the entire dataset in one compressed file. Random access requires seeking inside that file and decompressing the relevant chunk. When training on a remote store (S3, GCS) this is slow because the file must be partially fetched over HTTP with per-request overhead for every sample.

Lance stores each column in separate fragment files. Reading `action` never touches the `pixels` fragments. Row-level random access is provided by the `Permutation` API, which maps logical row indices to physical file locations without reading unneeded data.

---

## 2. Data model assumptions

### What the converter (`lance_conversion.py`) produces

Every row in the Lance table represents one timestep. The schema is:

| Column | Arrow type | Notes |
|---|---|---|
| `episode_idx` | `int32` | Monotonically non-decreasing; uniquely identifies an episode |
| `step_idx` | `int32` | Per-episode step counter starting at 0 |
| `pixels` | `binary` | JPEG-encoded frame blob (single camera) |
| `pixels_top` | `binary` | JPEG-encoded frame blob (multi-camera example) |
| `action` | `list<float32>[dim]` | Fixed-size list, dim = flattened action dim |
| `proprio` | `list<float32>[dim]` | Fixed-size list |
| `state` | `list<float32>[dim]` | Fixed-size list (dataset-dependent) |

Image columns are auto-detected from the naming convention: `pixels` (single camera) or `pixels_<view>` (multi-camera, e.g. `pixels_top`, `pixels_wrist_left`). Pass `image_keys` explicitly to `convert_hdf5_to_lance` to override. Every non-image column is stored as a `pa.list_(pa.float32(), dim)` fixed-size list.

### Column naming constraint: no dots in Lance field names

**Lance uses `.` as a struct-field path separator.** This is enforced in
`lance-core/src/datatypes/schema.rs` (line 300 as of lance-core 4.0.0) — any
top-level field whose name contains a dot is rejected at table-creation time
because the engine cannot distinguish it from a nested struct path:

```
LanceError(Schema): Top level field pixels.top cannot contain `.`.
Maybe you meant to create a struct field?
```

This is a **Lance restriction** (not LanceDB, not Arrow). Dots are valid inside
actual struct hierarchies, but not in flat top-level column names.

**The converter handles this transparently.** `convert_hdf5_to_lance` calls
`_to_lance_name()` on every column before writing, replacing `.` with `_`.
Your HDF5 file can use whatever naming it likes:

```
HDF5 column name   →   Lance field name
pixels.top         →   pixels_top
pixels.wrist_left  →   pixels_wrist_left
observation.state  →   observation_state
action             →   action              (unchanged)
```

A `logging.info` message is emitted for any column that gets renamed so you
can see exactly what happened. After conversion, refer to columns by their
Lance names (underscores) when constructing a `LanceDataset`.

### What the loader (`LanceDataset`) handles

The loader is more general than the converter:

- `image_columns` is a **set** — multiple image columns are fully supported.
- Image columns are auto-detected by name (`pixels` or `pixels_*`) when `image_columns` is not passed explicitly.
- `_extract_column` handles five Arrow types:
  - `pa.binary` / `pa.large_binary` → `list[bytes]` (image blobs)
  - `pa.fixed_size_list` → `numpy.ndarray` of shape `(N, dim)` via zero-copy flatten + reshape
  - `pa.list_` (variable-length) → `list` via `to_pylist()`, then stacked by `_pylist_to_numpy`
  - `pa.string` / `pa.large_string` → `list[str]`
  - scalar numeric (`int`, `float`) → `numpy.ndarray`

For a multi-camera dataset with `pixels_top` and `pixels_wrist`, both are decoded automatically without any extra configuration — the `pixels_*` naming convention is enough.

---

## 3. Episode structure

Rows are laid out episode-contiguously: all steps of episode 0, then all steps of episode 1, etc. `_compute_episode_structure` exploits this by scanning only the `episode_idx` column and vectorising boundary detection with numpy:

```python
boundary_mask = np.diff(all_ep_ids) != 0
change_positions = np.flatnonzero(boundary_mask) + 1
offsets = np.concatenate([[0], change_positions])      # start row of each episode
lengths = np.diff(np.concatenate([offsets, [N]]))      # length of each episode
```

This avoids a Python loop over all rows. The resulting `lengths` and `offsets` arrays are passed to the base `Dataset` class which builds `clip_indices` — a flat list mapping sample index → `(ep_idx, local_start)`.

**Row ordering after compaction**: `lancedb.compact_files()` merges fragments in fragment-ID order, so logical row ordering is preserved. What changes is the physical row address (fragment + offset), not the position in the table. The episode-contiguous assumption is therefore safe across compaction. The `ValueError` guard in `_compute_episode_structure` catches tables where data was inserted out of order in the first place, not compaction.

---

## 4. Batch-level fetch: `__getitems__`

PyTorch DataLoader (≥ 2.0) checks whether the dataset defines `__getitems__`. If it does, the DataLoader collects all sample indices for a batch and calls `dataset.__getitems__(indices)` once instead of `dataset.__getitem__(idx)` N times.

Without this, each sample triggers a separate `Permutation.__getitems__` call — for S3 that is one HTTP round-trip per sample. With batch size 64 that is 64 sequential round-trips per batch.

With `__getitems__` the implementation:

1. Collects all global row IDs across the batch into a single flat list.
2. Deduplicates and sorts (Lance `Permutation` requires sorted unique IDs).
3. Calls `self._perm.__getitems__(unique_rows)` once — one round-trip.
4. Uses zero-copy `RecordBatch.slice(offset, length)` to extract each sample's window from the returned batch.

For overlapping windows (when `frameskip=1` and consecutive samples share rows) deduplication actually reduces the number of rows fetched below `batch_size × span`.

---

## 5. Why `torch.tensor` not `torch.from_numpy`

`torch.from_numpy` shares the numpy buffer and returns a **non-resizable** tensor. When `__getitems__` is defined, DataLoader runs collation inside the worker process. Collation uses `storage.resize_()` to preallocate the output batch tensor. Non-resizable tensors crash here:

```
RuntimeError: Trying to resize storage that is not resizable
```

`torch.tensor(data)` always copies into PyTorch-owned resizable storage. The copy cost is negligible compared to I/O.

The one deliberate exception is `_decode_image`, which uses `torch.from_numpy(arr)`. This is safe because the resulting per-frame tensor is immediately consumed by `torch.stack(frames)`, which allocates a fresh resizable tensor. The non-resizable frame tensors never reach the collation step.

---

## 6. Column caching (`keys_to_cache`)

HDF5Dataset caches full columns in RAM because HDF5 random access is expensive (seeks + decompression). For Lance, `__getitems__` batches all row IDs into one `take()` call, so the marginal cost of reading an additional column is small. Benchmark measurements showed <1% throughput gain from caching `action` and `proprio`, with the downside of holding entire columns in RAM (OOM risk on large datasets).

`keys_to_cache` is still accepted (with a warning at runtime) for API parity and for unusual use cases where a column is accessed many times outside the DataLoader hot path (e.g. in eval code that iterates the dataset serially). The behaviour is identical to `HDF5Dataset`: the full column is loaded into `self._cache` at init, and `_fetch_columns` is recomputed to exclude cached columns from Lance reads.

---

## 7. Data flow for one batch

```
DataLoader.__next__()
    └─ LanceDataset.__getitems__(indices=[i0, i1, ..., i63])
           │
           ├─ 1. for each idx → clip_indices[idx] → (ep_idx, local_start)
           │       g_start = offsets[ep_idx] + local_start
           │       rows = [g_start, g_start+1, ..., g_start+span-1]
           │
           ├─ 2. unique_rows = sorted(set(all_rows))
           │    big_batch = Permutation.__getitems__(unique_rows)   ← one Lance take
           │    (reindex if dedup changed order)
           │
           └─ 3. for each sample i:
                   sub_batch = big_batch.slice(row_offsets[i], span)  ← zero-copy
                   _process_batch(ep_idx, g_start, sub_batch)
                       │
                       ├─ cached columns  → self._cache[col][g_start:g_end]
                       ├─ image columns   → _extract_column → _decode_image × T → torch.stack
                       └─ numeric columns → _extract_column → _prepare_numeric_tensor
                                            (fixed-size list fast path:
                                             col.flatten().to_numpy().reshape(N, dim))
```

---

## 8. What the model actually needs: end-to-end data flow

Understanding this section helps reason about why the dataset is structured the
way it is — what the shapes mean, why `action` is treated differently from
every other column, and what "correctness" means for the Lance implementation.

### What the model is doing

This is a **JEPA-style world model** (Joint Embedding Predictive Architecture).
Given a window of past observations and the actions taken between them, it
predicts what the *embedding* of the next observation will be — not the raw
pixels, but a learned compact representation of them.

It never reconstructs images. It learns in embedding space and predicts there.

### The temporal window

Every training sample is a short clip of one episode. Two parameters shape it:

```
frameskip = 5    # skip 5 raw frames between each macro-step
num_steps = 4    # load 4 macro-steps
span     = 20    # raw frames fetched per sample (num_steps × frameskip)
```

If raw frames are `f0 … f19`, the dataset picks every 5th frame:

```
f0  f1  f2  f3  f4 | f5  f6  f7  f8  f9 | f10 … f14 | f15 … f19
      macro-step 0  |    macro-step 1     | step 2     | step 3
```

- **`pixels`, `proprio`, `state`** — downsampled with `[::frameskip]`,
  yielding `num_steps` rows: `(f0, f5, f10, f15)`.
- **`action`** — *not* downsampled. All 20 raw actions are kept and reshaped
  to `(num_steps, frameskip × action_dim)` so the model sees the complete
  sequence of actions executed *within* each macro-step, not just the one at
  the boundary. This is why the dataset treats `action` differently from every
  other column.

### Batch shape at the DataLoader output

```
pixels    : (B, 4, 3, 224, 224)   # T=4 RGB frames, CHW, resized
action    : (B, 4, frameskip × A) # 4 macro-steps × 5 raw actions each
proprio   : (B, 4, P)             # proprioceptive vector at each macro-step
state     : (B, 4, S)             # full state vector (dataset-dependent)
```

B = 128 (default). All non-pixel columns are z-score normalised (mean/std
computed once over the full dataset at startup and applied as a transform).

### What the model does with the batch

```
encode()
  pixels (B, 4, 3, 224, 224)
    → flatten time  →  (B×4, 3, 224, 224)
    → ViT encoder → CLS token
    → projector MLP  →  (B, 4, D)     ← one embedding per macro-step
  action
    → Embedder MLP   →  (B, 4, D)     ← one action embedding per macro-step

predict()
  context = embeddings[:, :3, :]      ← history (first 3 steps)
  context_actions = action_emb[:, :3, :]
    → ARPredictor Transformer
    → pred_proj MLP  →  (B, 3, D)     ← predicted embedding for steps 1, 2, 3

loss
  target  = embeddings[:, 1:, :]      ← actual embeddings for steps 1, 2, 3
  pred_loss   = MSE(predicted, target)
  sigreg_loss = variance regulariser (prevents embedding collapse)
  total_loss  = pred_loss + λ · sigreg_loss
```

### Full pipeline

```
HDF5 / Lance table
  │
  │  Dataset.__getitems__(batch_indices)
  │    clip_indices[idx] → (episode, local_start)
  │    fetch span=20 raw rows per sample
  │    pixels/proprio/state  [::frameskip]  → 4 steps
  │    action                keeps 20 rows  → reshape (4, 5×A)
  │
  ▼
Raw batch  {pixels: (B,4,H,W,C), action: (B,4,5A), ...}
  │
  │  Transform (applied per-sample before collation)
  │    ImagePreprocessor   → resize 224×224, HWC→CHW, float32, ImageNet normalise
  │    ColumnNormalizer    → (x − mean) / std  for action, proprio, state
  │
  ▼
Normalised batch  {pixels: (B,4,3,224,224), action: (B,4,5A), ...}
  │
  │  model.encode()
  │    ViT + projector  →  (B, 4, D=192)
  │    Embedder         →  (B, 4, D=192)
  │
  ▼
Embedding space  (B, 4, D)
  │
  │  model.predict()
  │    context [:, :3]  →  ARPredictor Transformer  →  (B, 3, D)
  │
  ▼
Loss
  │    pred_loss  = MSE(predicted, actual[:, 1:])
  │    sigreg     = variance regulariser over time axis
  │
  ▼
Backward / optimiser step
```

### Why this matters for the dataset layer

The dataset's contract with the model is:

1. Deliver `(B, num_steps, ...)` tensors with correct temporal ordering.
2. Action has shape `(B, num_steps, frameskip × A)` — the raw action sequence
   packed per macro-step, not just one action per step.
3. All columns normalised before the model sees them.
4. Clips are episode-contiguous — rows come from one unbroken episode window,
   never crossing an episode boundary.

The Lance implementation preserves all four invariants, which is what
`test_lance_dataset_output_parity` and `test_lance_dataset_getitems_matches_getitem`
verify.

---

## 9. Factory function (`create_dataset`)

`stable_worldmodel.data.create_dataset(cfg)` accepts a Hydra config node or plain dict and dispatches to the right backend:

- If `uri` and `table_name` are present → `LanceDataset`
- Otherwise → `HDF5Dataset`

This keeps all existing YAML configs working unchanged. Switching a run to Lance only requires adding `+data.dataset.uri=...` and `+data.dataset.table_name=...` as CLI overrides.
