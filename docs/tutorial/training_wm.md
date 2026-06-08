title: Training A World Model
summary: Train or plug in a model that implements the planning cost interface, then evaluate it with model-predictive control.
sidebar_title: Train World-Model
---

This tutorial covers the second and third stages of a world-model experiment:
training a model from collected trajectories and using it for planning.

The important integration point is small: a model used by
`WorldModelPolicy` must provide `get_cost(info_dict, action_candidates)`.
Planning solvers such as CEM and MPPI search over action sequences and choose
the sequence with the lowest predicted cost.

## Prerequisites

Install the full package and collect the dataset from
[Collect Dataset](collect_data.md):

```bash
pip install 'stable-worldmodel[all]'
export STABLEWM_HOME=$PWD/.stablewm
```

This tutorial assumes the dataset exists at:

```text
$STABLEWM_HOME/datasets/tutorial_pusht.lance
```

## The cost-model contract

A cost model receives the current `World` info dict and a batch of candidate
action sequences. It returns one scalar cost per environment and candidate.
Lower is better.

```python
import torch


class MyWorldModel(torch.nn.Module):
    def get_cost(
        self,
        info_dict: dict,
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """
        info_dict:
            Dict produced by World. Tensor values usually have shape
            (num_envs, history, ...).

        action_candidates:
            Tensor with shape
            (num_envs, num_samples, horizon, action_dim).

        returns:
            Cost tensor with shape (num_envs, num_samples).
        """
        costs = ...
        return costs
```

For goal-conditioned planning, `info_dict` usually contains `pixels`,
`goal`, `state`, and `goal_state` or related goal columns. Image models often
compare predicted future embeddings against a goal embedding; state models
often compare predicted state against `goal_state`.

## Load trajectory clips

Training starts from the same dataset API used everywhere else:

```python
import stable_worldmodel as swm

dataset = swm.data.load_dataset(
    'tutorial_pusht.lance',
    num_steps=8,
    frameskip=1,
    keys_to_load=['pixels', 'action', 'state'],
)
```

With PyTorch:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

batch = next(iter(loader))
print(batch['pixels'].shape)  # (B, T, C, H, W)
print(batch['action'].shape)  # (B, T, action_dim)
```

The first action in an episode may be `NaN` because there is no previous
environment action at reset. Training code should replace it before computing
losses:

```python
batch['action'] = torch.nan_to_num(batch['action'], 0.0)
```

## Use a reference training script

The repository ships reference training scripts under `scripts/train/`. For
example, `scripts/train/lewm.py` trains the LeWM baseline from image clips and
actions.

For a short GPU smoke run:

```bash
python scripts/train/lewm.py \
  data.dataset.name=tutorial_pusht.lance \
  trainer.max_epochs=2 \
  loader.batch_size=32 \
  num_workers=2 \
  output_model_name=tutorial_lewm
```

For a CPU-only smoke run, override the trainer settings:

```bash
python scripts/train/lewm.py \
  data.dataset.name=tutorial_pusht.lance \
  trainer.max_epochs=1 \
  trainer.accelerator=cpu \
  trainer.devices=1 \
  trainer.precision=32 \
  loader.batch_size=8 \
  num_workers=0 \
  output_model_name=tutorial_lewm_cpu
```

These commands are useful for checking that the dataset and training
dependencies are wired correctly. For real experiments, increase epochs,
batch size, image resolution, and model size according to your hardware.

## Save a custom model

If you train your own model, save it with the checkpoint format used by the
library:

```python
from stable_worldmodel.wm.utils import save_pretrained

model_config = {
    '_target_': 'my_project.models.MyWorldModel',
    'hidden_dim': 256,
    'action_dim': 2,
}

save_pretrained(
    model=model,
    run_name='tutorial_state_wm',
    config=model_config,
    filename='weights.pt',
)
```

`config` must be the Hydra instantiation config for the model itself. If your
training script has a larger config with keys such as `data`, `loader`, and
`trainer`, pass only the model sub-config or use `config_key='model'`.

## Evaluate with MPC

Load the model and wrap it with a solver-backed policy:

```python
import stable_worldmodel as swm
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy
from stable_worldmodel.solver import CEMSolver
from stable_worldmodel.wm.utils import load_pretrained


device = 'cuda'
model = load_pretrained('tutorial_state_wm/weights.pt').to(device).eval()
model.requires_grad_(False)

world = swm.World(
    'swm/PushT-v1',
    num_envs=8,
    image_shape=(96, 96),
    max_episode_steps=100,
)

solver = CEMSolver(
    model=model,
    num_samples=300,
    n_steps=5,
    device=device,
)

policy = WorldModelPolicy(
    solver=solver,
    config=PlanConfig(
        horizon=10,
        receding_horizon=5,
        action_block=1,
        warm_start=True,
    ),
)

world.set_policy(policy)
results = world.evaluate(episodes=32, seed=0, video='videos/tutorial_eval')
world.close()

print(results)
```

Use the same preprocessing at evaluation time that you used during training.
If you normalized actions or state columns, pass those reversible processors
to `WorldModelPolicy(process=...)`. If you normalized images, pass the image
transforms with `WorldModelPolicy(transform=...)`.

## Evaluate from dataset starts and goals

For goal-reaching experiments, it is common to initialize the simulator from
states stored in a dataset and ask the policy to reach a future dataset goal.
In this mode, `num_envs` must equal the number of requested episodes.

```python
dataset = swm.data.load_dataset(
    'tutorial_pusht.lance',
    num_steps=32,
    keys_to_load=['pixels', 'action', 'state'],
)

world = swm.World(
    'swm/PushT-v1',
    num_envs=4,
    image_shape=(96, 96),
    max_episode_steps=100,
)
world.set_policy(policy)

results = world.evaluate(
    dataset=dataset,
    episodes_idx=[0, 1, 2, 3],
    start_steps=[0, 5, 10, 15],
    goal_offset=25,
    eval_budget=50,
    callables=[
        {'method': '_set_state', 'args': {'state': {'value': 'state'}}},
        {
            'method': '_set_goal_state',
            'args': {'goal_state': {'value': 'goal_state'}},
        },
    ],
    video='videos/tutorial_dataset_eval',
)
```

Some environments need callables to restore internal simulator state from the
dataset. PushT supports `_set_state` and `_set_goal_state`; the planned
evaluation script in `scripts/plan/eval_wm.py` shows the full Hydra version
of that workflow.

## Debug checklist

- `get_cost()` must return finite costs with shape `(num_envs, num_samples)`.
- The solver's `device` must match the model device.
- `PlanConfig.horizon * action_block` should fit inside the evaluation
  budget.
- Use `warm_start=True` for faster receding-horizon planning once the model
  is stable.
- Start with small `num_samples` and short horizons while debugging, then
  increase them for final evaluation.
