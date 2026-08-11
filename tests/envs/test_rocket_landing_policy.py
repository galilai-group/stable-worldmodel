"""Tests for rocket landing environment expert policy."""

import numpy as np
import pytest


# Skips where cvxpy/pybullet/PyFlyt are absent; none are declared in pyproject.
expert_policy = pytest.importorskip(
    'stable_worldmodel.envs.rocket_landing.expert_policy'
)


NUM_ENVS = 4
MASK = np.array([False, True, False, True])
OBS_DIM = 13


################################
## ExpertPolicy Tests         ##
################################


def test_expert_policy_keeps_every_controller_on_a_masked_call():
    """A masked call must not resize the per-env controller list."""
    policy = expert_policy.ExpertPolicy()
    policy._ensure_controller_count(NUM_ENVS)
    controllers = list(policy.controllers)

    obs = np.zeros((NUM_ENVS, OBS_DIM), dtype=np.float32)
    obs[:, 6] = 1.0  # unit quaternion
    policy.get_action({'state': obs[np.flatnonzero(MASK)]}, env_mask=MASK)

    assert len(policy.controllers) == NUM_ENVS
    assert policy.controllers == controllers
