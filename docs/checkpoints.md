---
title: Checkpoints
summary: How to save and load pretrained model checkpoints
---

Model checkpoints in `stable_worldmodel` use a simple two-file format: a `.pt` weights file and a `config.json` that describes the model architecture. The `load_pretrained()` function uses the config to reconstruct the model and then loads the weights — no manual instantiation required.

## Checkpoint format

A valid checkpoint is a directory containing:

```
my_run/
├── weights.pt      # model.state_dict() saved with torch.save()
└── config.json     # Hydra-compatible model config (JSON)
```

The `config.json` must be a valid [Hydra](https://hydra.cc/) instantiation config with a `_target_` key pointing to the model class. It is generated automatically when you use `save_pretrained()`.

## Cache location

By default, all checkpoints are stored under:

```
~/.stable_worldmodel/checkpoints/
```

You can override this with the `STABLEWM_HOME` environment variable:

```bash
export STABLEWM_HOME=/path/to/custom/dir
```

Or by passing `cache_dir` directly to `save_pretrained()` / `load_pretrained()`.

---

## Saving a checkpoint

```python
from stable_worldmodel.wm.utils import save_pretrained

save_pretrained(
    model=my_model,         # any torch.nn.Module
    run_name='my_run',      # name for the checkpoint folder
    config=cfg,             # OmegaConf DictConfig (from Hydra)
    config_key='model',     # optional: extract a sub-key from the config
    filename='weights.pt',  # optional: defaults to 'weights.pt'
)
# Saves to: ~/.stable_worldmodel/checkpoints/my_run/weights.pt
#                                             my_run/config.json
```

!!! warning "Config is required for automatic loading"
    If you omit `config`, only the weights are saved. You will have to instantiate the model manually before calling `load_state_dict()`.

---

## Loading a checkpoint

`load_pretrained()` supports three input formats, all resolved relative to `~/.stable_worldmodel/checkpoints/`.

### 1. Explicit `.pt` file

```python
from stable_worldmodel.wm.utils import load_pretrained

model = load_pretrained('my_run/weights.pt')
```

A `config.json` must exist in the same directory as the `.pt` file.

### 2. Folder

```python
model = load_pretrained('my_run/')
```

The folder must contain **exactly one** `.pt` file and a `config.json`. If multiple `.pt` files are present, specify the file directly (format 1).

### 3. HuggingFace repository

```python
model = load_pretrained('nice-user/my-worldmodel')
```

If the repo is not already cached locally, `load_pretrained()` downloads `weights.pt` and `config.json` from HuggingFace and caches them at:

```
~/.stable_worldmodel/checkpoints/models--nice-user--my-worldmodel/
```

Subsequent calls load from the local cache without re-downloading.

---

## Listing available checkpoints

Use the CLI to inspect what is available in your cache:

```bash
swm checkpoints           # list all checkpoints
swm checkpoints pusht     # filter by name (regex)
```

---

## Full example: train → save → load

```python
import stable_worldmodel as swm
from stable_worldmodel.wm.utils import save_pretrained, load_pretrained

# --- Training ---
model = MyWorldModel(...)
train(model, ...)

# --- Saving ---
save_pretrained(
    model=model,
    run_name='pusht_wm_v1',
    config=cfg,             # your Hydra config
    config_key='model',     # only save the model sub-config
)

# --- Loading later ---
model = load_pretrained('pusht_wm_v1')
model.eval()
```

---

## Using a loaded model as a policy

Once loaded, pass the model to `AutoCostModel` or `AutoActionableModel` to use it with the `World` API:

```python
from stable_worldmodel.policy import AutoCostModel, WorldModelPolicy, PlanConfig
from stable_worldmodel.solver import CEMSolver

cost_model = AutoCostModel('pusht_wm_v1')

policy = WorldModelPolicy(
    solver=CEMSolver(model=cost_model, num_samples=300),
    config=PlanConfig(horizon=10, receding_horizon=5),
)

world = swm.World('swm/PushT-v1', num_envs=4)
world.set_policy(policy)
results = world.evaluate(episodes=50, seed=0)
```

See [Policy](api/policy.md) for details on `AutoCostModel`, `AutoActionableModel`, and `WorldModelPolicy`.
