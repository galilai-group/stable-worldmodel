"""Tests for WireHarnessBasicEnv's variation-space and world-model wiring.

One mover visits a fixed sequence of targets. The start position and the whole
target sequence live in ``variation_space``; ``reset()`` reads them back and
drives the physics from them, so the variation space is the single source of
truth for an episode. ``_set_state`` / ``_set_goal_state`` are the
dataset-driven evaluation hooks used by ``World.evaluate``.
"""

import os

import numpy as np
import pytest


pytest.importorskip('mujoco')

os.environ.setdefault('MUJOCO_GL', 'egl')

import gymnasium as gym  # noqa: E402
import stable_worldmodel.envs  # noqa: E402, F401  (registers swm/WireHarness-v0)

from stable_worldmodel.envs.wire_harness import env_basic as C  # noqa: E402


ENV_ID = 'swm/WireHarness-v0'
OBS_DIM = 4
ACT_DIM = 2
N_TARGETS = len(C.DEFAULT_TARGETS)


def _mover_var(env):
    return env.get_wrapper_attr('variation_space').spaces['mover']


def _read_variation(env):
    """(start, targets) as arrays read back from the variation space."""
    mv = _mover_var(env)
    start = np.asarray(mv.spaces['start_position'].value, dtype=np.float64)
    targets = np.asarray(mv.spaces['target_positions'].value, dtype=np.float64)
    return start, targets


@pytest.fixture
def env():
    e = gym.make(ENV_ID)
    yield e
    e.close()


def test_spaces_and_info(env):
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.action_space.shape == (ACT_DIM,)

    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    for key in ('state', 'goal_state'):
        assert key in info, f"missing info key '{key}'"
    assert info['state'].shape == (2,)
    assert info['goal_state'].shape == (2,)


def test_variation_space_drives_the_physics(env):
    """The realized mover position matches the sampled start every reset."""
    seen = set()
    for seed in range(6):
        _, info = env.reset(seed=seed)
        start, targets = _read_variation(env)

        assert targets.shape == (N_TARGETS, 2)
        np.testing.assert_allclose(info['state'], start, atol=1e-2)
        # goal_state tracks the first target of the sampled sequence
        np.testing.assert_allclose(info['goal_state'], targets[0], atol=1e-3)

        seen.add(tuple(np.round(start, 3)))

    # sampling actually explores more than one start across seeds
    assert len(seen) >= 2


def test_sampled_layout_stays_in_bounds(env):
    for seed in range(6):
        env.reset(seed=seed)
        start, targets = _read_variation(env)
        for pos in np.vstack([start[None, :], targets]):
            assert (
                C.WireHarnessBasicEnv.X_MIN
                <= pos[0]
                <= C.WireHarnessBasicEnv.X_MAX
            )
            assert (
                C.WireHarnessBasicEnv.Y_MIN
                <= pos[1]
                <= C.WireHarnessBasicEnv.Y_MAX
            )


def test_seeded_reset_is_reproducible(env):
    env.reset(seed=11)
    a_start, a_targets = _read_variation(env)
    env.reset(seed=11)
    b_start, b_targets = _read_variation(env)
    np.testing.assert_allclose(a_start, b_start, atol=1e-9)
    np.testing.assert_allclose(a_targets, b_targets, atol=1e-9)


def test_explicit_variation_values_respected(env):
    """Dataset-eval path: an explicit override lands in the variation space and
    is realized by the physics (no resampling on top)."""
    start = np.array([3.0, 2.0], dtype=np.float32)
    targets = np.tile(np.array([1.5, 1.0], dtype=np.float32), (N_TARGETS, 1))

    _, info = env.reset(
        seed=0,
        options={
            'variation_values': {
                'mover.start_position': start,
                'mover.target_positions': targets,
            }
        },
    )
    mv = _mover_var(env)
    np.testing.assert_allclose(
        mv.spaces['start_position'].value, start, atol=1e-6
    )
    np.testing.assert_allclose(
        mv.spaces['target_positions'].value, targets, atol=1e-6
    )
    np.testing.assert_allclose(info['state'], start, atol=1e-2)


def test_set_state_and_goal_state_hooks(env):
    """The World.evaluate callables must move the mover and retarget success."""
    env.reset(seed=0)

    state = np.array([2.0, 1.0], dtype=np.float32)
    goal = np.array([2.2, 1.0], dtype=np.float32)
    env.get_wrapper_attr('_set_state')(state)
    env.get_wrapper_attr('_set_goal_state')(goal)

    obs, reward, terminated, truncated, info = env.step(
        np.zeros(ACT_DIM, dtype=np.float32)
    )
    np.testing.assert_allclose(info['state'], state, atol=1e-2)
    np.testing.assert_allclose(info['goal_state'], goal, atol=1e-6)
    assert not truncated  # _set_state resets the step counter


def test_eval_goal_terminates_on_reach(env):
    """With an eval goal set, reaching it ends the episode (dataset success)."""
    env.reset(seed=0)
    state = np.array([2.0, 1.0], dtype=np.float32)
    env.get_wrapper_attr('_set_state')(state)
    # goal well inside goal_radius of the current position
    env.get_wrapper_attr('_set_goal_state')(state.copy())

    _, _, terminated, _, _ = env.step(np.zeros(ACT_DIM, dtype=np.float32))
    assert terminated


def test_targets_advance_in_sequence(env):
    """Without an eval goal, reaching a target advances to the next one."""
    env.reset(seed=0)
    _, targets = _read_variation(env)

    env.get_wrapper_attr('_set_state')(targets[0].astype(np.float32))
    _, _, terminated, _, info = env.step(np.zeros(ACT_DIM, dtype=np.float32))

    assert not terminated, 'first target should not end the episode'
    assert info['current_target_idx'] == 1
    np.testing.assert_allclose(info['goal_state'], targets[1], atol=1e-3)


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
        for key in ('state', 'goal_state'):
            assert key in info
        if terminated or truncated:
            break
    assert len(rewards) > 0
