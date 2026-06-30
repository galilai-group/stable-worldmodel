"""Tests for the WireHarness env's variation-space wiring.

The env trains ONE target configuration per instance (one SAC per stage), so the
goal is FIXED to ``config.MOVER_TARGETS[stage]`` every episode. Only the start
pose is sampled — jointly, over the discrete configuration set
``{MOVER_STARTS} ∪ {targets[j != stage]}``. Both start and target poses are
written into ``variation_space`` and read back from it to drive the physics, so
the variation space is the single source of truth for an episode.
"""

import os
import sys

import numpy as np
import pytest


pytest.importorskip("mujoco")

# WireHarness env.py resolves ``import config`` / ``from model.mover import
# Mover`` off sys.path (the package is not import-relative), so the env's own
# directory must be importable — mirrors how data collection is launched.
import stable_worldmodel  # noqa: E402

_WH_DIR = os.path.join(
    os.path.dirname(stable_worldmodel.__file__), "envs", "wire_harness"
)
if _WH_DIR not in sys.path:
    sys.path.insert(0, _WH_DIR)

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym  
import stable_worldmodel.envs  

from stable_worldmodel.envs.wire_harness import config as C  


ENV_ID = "swm/WireHarness-v0"
STAGE = 0
N = len(C.MOVER_BODY_NAMES)
OBS_DIM = 2 * N + N * (N - 1)  
ACT_DIM = 2 * N                

def _goal_config(stage=STAGE):
    return np.array(C.MOVER_TARGETS[stage], dtype=np.float64)


def _start_candidates(stage=STAGE):
    """{MOVER_STARTS} ∪ {targets[j != stage]} — what _sample_starts draws from."""
    cands = [np.array(C.MOVER_STARTS, dtype=np.float64)]
    cands += [
        np.array(C.MOVER_TARGETS[j], dtype=np.float64)
        for j in range(C.N_STAGES)
        if j != stage
    ]
    return cands


def _read_variation(vs):
    """(starts, targets) as (N, 2) arrays read from the variation space."""
    starts, targets = [], []
    for i in range(N):
        mv = vs[f"mover_{i + 1}"]
        starts.append(np.asarray(mv["start_position"].value, dtype=np.float64))
        targets.append(np.asarray(mv["target_position"].value, dtype=np.float64))
    return np.array(starts), np.array(targets)


@pytest.fixture
def env():
    e = gym.make(ENV_ID, stage=STAGE)
    yield e
    e.close()


def test_spaces_and_info(env):
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.action_space.shape == (ACT_DIM,)

    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    for key in ("state", "goal_state"):
        assert key in info, f"missing info key '{key}'"
    assert info["state"].shape == (2 * N,)
    assert info["goal_state"].shape == (2 * N,)


def test_goal_is_fixed_stage_target(env):
    """Goal must NOT vary: it is config.MOVER_TARGETS[stage] every reset."""
    goal = _goal_config()
    for seed in range(6):
        obs, info = env.reset(seed=seed)
        vs = env.get_wrapper_attr("variation_space")
        _, targets = _read_variation(vs)
        np.testing.assert_allclose(targets, goal, atol=1e-3)
        np.testing.assert_allclose(
            info["goal_state"].reshape(N, 2), goal, atol=1e-3
        )


def test_start_drawn_from_config_set_and_realized(env):
    """Start is one of the discrete candidate configs (joint), it differs from
    the goal, and the physics is actually placed there (state == variation start).
    """
    candidates = _start_candidates()
    goal = _goal_config()
    seen = set()
    for seed in range(8):
        obs, info = env.reset(seed=seed)
        vs = env.get_wrapper_attr("variation_space")
        starts, _ = _read_variation(vs)

        # joint membership in the discrete candidate set
        assert any(np.allclose(starts, c, atol=1e-3) for c in candidates), (
            f"start config not in candidate set:\n{starts}"
        )
        # start != goal (non-trivial episode)
        assert not np.allclose(starts, goal, atol=1e-3)
        # variation space is the source of truth the physics follows
        np.testing.assert_allclose(
            info["state"].reshape(N, 2), starts, atol=1e-2
        )
        seen.add(tuple(np.round(starts.reshape(-1), 2)))

    # sampling actually explores more than one start across seeds
    assert len(seen) >= 2


def test_start_initial_option_forces_mover_starts(env):
    obs, info = env.reset(seed=3, options={"start": "initial"})
    vs = env.get_wrapper_attr("variation_space")
    starts, _ = _read_variation(vs)
    np.testing.assert_allclose(
        starts, np.array(C.MOVER_STARTS, dtype=np.float64), atol=1e-3
    )


def test_seeded_reset_is_reproducible(env):
    env.reset(seed=11)
    a, _ = _read_variation(env.get_wrapper_attr("variation_space"))
    env.reset(seed=11)
    b, _ = _read_variation(env.get_wrapper_attr("variation_space"))
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_explicit_variation_values_respected(env):
    """Dataset-eval path: an explicit start_position override lands in the
    variation space and is realized by the physics (no resampling on top)."""
    target_start = np.array([3.0, 2.0], dtype=np.float32)
    obs, info = env.reset(
        seed=0,
        options={"variation_values": {"mover_1.start_position": target_start}},
    )
    vs = env.get_wrapper_attr("variation_space")
    np.testing.assert_allclose(
        vs["mover_1"]["start_position"].value, target_start, atol=1e-6
    )
    # mover_1 (first mover) is actually placed at the override
    np.testing.assert_allclose(
        info["state"].reshape(N, 2)[0], target_start, atol=1e-2
    )


def test_step_runs_and_is_finite(env):
    obs, info = env.reset(seed=0)
    rewards = []
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        assert obs.shape == (OBS_DIM,)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        rewards.append(reward)
        for key in ("state", "goal_state"):
            assert key in info
        if terminated or truncated:
            break
    assert len(rewards) > 0
