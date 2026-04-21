---
title: Saving & Loading Checkpoints
summary: How to save and load pretrained model checkpoints
---

Model checkpoints in `stable_worldmodel` use a simple two-file format: a `.pt` weights file and a `config.json` that contains all information needed to re-instantiate the model, following [Hydra's instantiation syntax](https://hydra.cc/docs/advanced/instantiate_objects/overview/) but stored as plain JSON.

## Checkpoint format

A valid checkpoint is a directory containing:

```
my_run/
├── weights.pt      # model.state_dict() saved with torch.save()
└── config.json     # Hydra instantiation config (JSON)
```

`load_pretrained()` reads `config.json` to reconstruct the model architecture, then loads the weights from the `.pt` file — no manual instantiation needed.

### The `config.json` format

`config.json` must follow [Hydra's `_target_` instantiation convention](https://hydra.cc/docs/advanced/instantiate_objects/overview/). The `_target_` key is the fully-qualified class path; all other keys are passed as constructor arguments.

```json
{
  "_target_": "my_package.models.MyWorldModel",
  "hidden_dim": 256,
  "num_layers": 4,
  "action_dim": 2
}
```

`load_pretrained()` calls `hydra.utils.instantiate(config)` internally, which is equivalent to:

```python
from my_package.models import MyWorldModel
model = MyWorldModel(hidden_dim=256, num_layers=4, action_dim=2)
```

Nested modules follow the same pattern:

```json
{
  "_target_": "my_package.models.MyWorldModel",
  "encoder": {
    "_target_": "my_package.models.ResNetEncoder",
    "channels": 64
  },
  "hidden_dim": 256
}
```

---

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

`save_pretrained()` saves the model weights and serializes the config to `config.json`. The `config` argument must be a dictionary (plain `dict` or an OmegaConf `DictConfig`) that follows the Hydra instantiation syntax shown above.

```python
from stable_worldmodel.wm.utils import save_pretrained

# Option A: build the config manually as a plain dict
config = {
    "_target_": "my_package.models.MyWorldModel",
    "hidden_dim": 256,
    "num_layers": 4,
}

# Option B: use the DictConfig produced by Hydra in your training script
# config = cfg.model  (already a DictConfig)

save_pretrained(
    model=my_model,         # any torch.nn.Module
    run_name='my_run',      # name for the checkpoint folder
    config=config,          # dict or OmegaConf DictConfig
    config_key='model',     # optional: extract a sub-key from the config
    filename='weights.pt',  # optional: defaults to 'weights.pt'
)
# Saves to: ~/.stable_worldmodel/checkpoints/my_run/weights.pt
#                                             my_run/config.json
```

`config_key` is useful when you pass a full Hydra config and only want to persist the model sub-config (e.g., `cfg` contains `cfg.model`, `cfg.training`, … and you only need `cfg.model`).

!!! warning "Config is required for automatic loading"
    If you omit `config`, only the weights are saved. You will have to instantiate the model manually and call `load_state_dict()` yourself.

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
model = MyWorldModel(hidden_dim=256, num_layers=4)
train(model, ...)

# --- Saving ---
config = {
    "_target_": "my_package.models.MyWorldModel",
    "hidden_dim": 256,
    "num_layers": 4,
}

save_pretrained(
    model=model,
    run_name='pusht_wm_v1',
    config=config,
)

# --- Loading later ---
model = load_pretrained('pusht_wm_v1')
model.eval()
```

---

## Using a loaded model as a policy

Once loaded, wrap the model with `AutoCostModel` or `AutoActionableModel` to use it with the `World` API:

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

See [Policy](../api/policy.md) for details on `AutoCostModel`, `AutoActionableModel`, and `WorldModelPolicy`.
