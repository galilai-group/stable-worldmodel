"""Tests for TwoRoom environment expert policy."""

import gymnasium as gym
import numpy as np

from stable_worldmodel.envs.two_room.expert_policy import ExpertPolicy
from stable_worldmodel.world.env_pool import EnvPool
from stable_worldmodel.world.world import _slice_policy_info


NUM_ENVS = 4
MASK = np.array([False, True, False, True])


################################
## ExpertPolicy Tests         ##
################################


def _pool():
    return EnvPool(
        [lambda: gym.make('swm/TwoRoom-v1') for _ in range(NUM_ENVS)]
    )


def test_expert_policy_get_action_with_env_mask_returns_selected_rows():
    """An env's action must not depend on which other envs were ready."""
    pool = _pool()
    # Noise off, so a difference means indexing rather than RNG draw count.
    policy = ExpertPolicy(action_noise=0.0, action_repeat_prob=0.0, seed=0)
    policy.set_env(pool)
    _, infos = pool.reset(seed=0)
    indices = np.flatnonzero(MASK)

    full = np.asarray(policy.get_action(infos))
    ready = _slice_policy_info(policy, infos, indices, NUM_ENVS)
    narrowed = np.asarray(policy.get_action(ready, env_mask=MASK))

    assert narrowed.shape[0] == len(indices)
    for row, env_idx in enumerate(indices):
        np.testing.assert_allclose(
            narrowed[row], full[env_idx], rtol=1e-6, atol=1e-6
        )
    pool.close()
