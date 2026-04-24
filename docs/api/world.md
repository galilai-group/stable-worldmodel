title: World
summary: Runs a policy against a pool of vectorized environments, with HDF5 data collection and dataset-driven evaluation.
---

`World` is the main entry point for rolling out policies in `stable_worldmodel`. It bundles:

1. A batched simulator (`EnvPool`) that steps `num_envs` envs in parallel and can skip terminated envs via a mask.
2. A preprocessing pipeline (`MegaWrapper`) that resizes pixels, lifts everything into the info dict, and applies optional transforms.
3. A rollout loop that drives `policy.get_action(infos)` and handles resets, per-env termination, and episode accounting.

/// tab | Basic Usage
```python
import stable_worldmodel as swm
from stable_worldmodel.policy import RandomPolicy

world = swm.World(
    env_name="swm/PushT-v1",
    num_envs=4,
    image_shape=(64, 64),
)
world.set_policy(RandomPolicy())

# All stacked tensors in world.infos have shape (num_envs, 1, ...).
world.reset(seed=0)
# world.infos["pixels"]  -> (4, 1, 64, 64, 3)
```
///

/// tab | Collect a dataset
```python
import stable_worldmodel as swm

world = swm.World("swm/PushT-v1", num_envs=8, image_shape=(64, 64))
world.set_policy(expert_policy)

# Roll out 500 episodes in parallel and dump them to an HDF5 file.
world.collect("data/pusht_expert.h5", episodes=500, seed=0)
```
///

/// tab | Episodic evaluation
```python
results = world.evaluate(
    episodes=100,
    seed=42,
    video="videos/",          # optional: mp4 per episode
)

print(f"Success rate: {results['success_rate']:.1f}%")
```
///

/// tab | Dataset-driven evaluation
```python
# One env per target episode. Each env starts at the chosen step and aims
# for the state `goal_offset` timesteps later. Run capped at `eval_budget`.
results = world.evaluate(
    dataset=dataset,
    episodes_idx=[0, 1, 2, 3],
    start_steps=[0, 10, 20, 30],
    goal_offset=30,
    eval_budget=50,
    video="videos/",
)
```
///

/// tab | Per-environment reset options
`reset(options=...)` accepts a list of per-env dicts to seed domain randomization or variations:

```python
per_env = [
    {"variation": ["agent.color"], "variation_values": {"agent.color": [255, 0, 0]}},
    {"variation": ["agent.color"], "variation_values": {"agent.color": [0, 255, 0]}},
    {"variation": ["agent.color"], "variation_values": {"agent.color": [0, 0, 255]}},
]
world.reset(options=per_env)
```
///

## Info convention

Every tensor / array value in `world.infos` carries a leading time dim of 1 after the env dim:

```
world.infos["pixels"].shape  # (num_envs, 1, H, W, C)
world.infos["state"].shape   # (num_envs, 1, state_dim)
```

Non-array values (strings, nested objects) stay as a Python list of length `num_envs`. `rewards`, `terminateds`, and `truncateds` are returned from the last `step()` separately and are shape `(num_envs,)` — they do not carry the time dim.

## Reset modes

`evaluate` (and internally `_run`) support two termination policies:

- `reset_mode='auto'` — terminated envs are reset immediately. The run continues until `episodes` episodes have finished (or `max_steps` is reached). This is the default for episodic eval.
- `reset_mode='wait'` — terminated envs are frozen (step is skipped for them via the env-pool mask). The run stops when all envs are done. This is the default for dataset eval, so every env gets to complete its specific start→goal task.

---

::: stable_worldmodel.world.World
    options:
        heading_level: 2
        members: false
        show_source: false

## **[ Rollouts ]**

::: stable_worldmodel.world.World.collect
::: stable_worldmodel.world.World.evaluate

## **[ Environment ]**

::: stable_worldmodel.world.World.reset
::: stable_worldmodel.world.World.set_policy
::: stable_worldmodel.world.World.close

## **[ Properties ]**

::: stable_worldmodel.world.World.num_envs

## EnvPool

The underlying batched simulator. You rarely touch it directly — `World` builds one for you — but its action and observation spaces are what the policy sees.

::: stable_worldmodel.world.EnvPool
    options:
        heading_level: 3
        members:
          - num_envs
          - action_space
          - single_action_space
          - observation_space
          - single_observation_space
          - variation_space
          - single_variation_space
          - reset
          - step
          - close
        show_source: false
