---
title: PettingZoo
summary: Multi-agent environments through PettingZoo's Parallel and AEC APIs
---

# PettingZoo

Stable World-Model can run PettingZoo Parallel and AEC environments through
`World.from_pettingzoo`. Each PettingZoo environment instance is treated as one
SWM environment, and agents are represented as per-agent columns in
`world.infos`.

PettingZoo's per-agent action spaces are exposed to SWM as one flat ndarray
action per environment. The wrapper reconstructs the PettingZoo agent-action
mapping internally before stepping the underlying env.

## Parallel API

```python
import stable_worldmodel as swm
from pettingzoo.butterfly import pistonball_v6

world = swm.World.from_pettingzoo(
    lambda: pistonball_v6.parallel_env(render_mode="rgb_array"),
    num_envs=4,
)
world.set_policy(swm.MultiAgentRandomPolicy(seed=0))
world.collect("pistonball.lance", episodes=16, seed=0)
```

## AEC API

AEC environments are turn-based: one SWM step advances the currently selected
PettingZoo agent.

```python
import stable_worldmodel as swm
from pettingzoo.classic import rps_v2

world = swm.World.from_pettingzoo(
    lambda: rps_v2.env(),
    num_envs=4,
    api="aec",
)
world.set_policy(swm.MultiAgentRandomPolicy(seed=0))
world.collect("rps.lance", episodes=16, seed=0)
```

## Info Columns

Agent keys are flattened into stable column names:

| Column | Meaning |
| --- | --- |
| `agent_mask` | Boolean mask over `possible_agents` |
| `current_agent_idx` | Current AEC agent index (`-1` for completed envs) |
| `observation.<agent>` | Agent observation |
| `action.<agent>` | Agent action for the transition |
| `reward.<agent>` | Agent reward |
| `terminated.<agent>` | Agent termination flag |
| `truncated.<agent>` | Agent truncation flag |

If an observation or action space is a `gymnasium.spaces.Dict`, nested fields
are flattened one level deeper, for example
`observation.player_0.action_mask`.

For AEC envs, `action.<agent>` records the selected action on that agent's turn
and carries the previous action otherwise.
