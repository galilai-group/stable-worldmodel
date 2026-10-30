import numpy as np
import pytest

from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper


@pytest.fixture
def qpos_match_env():
    env = ReacherDMControlWrapper(task='qpos_match', seed=0)
    try:
        yield env
    finally:
        env.close()


def test_qpos_match_reset_with_options_sets_target(qpos_match_env):
    target = np.array([0.3, -0.3])
    qpos_match_env.reset(options={'target_qpos': target})
    np.testing.assert_allclose(qpos_match_env.env.task.target_qpos, target)


def test_qpos_match_plain_reset_clears_stale_target(qpos_match_env):
    target = np.array([0.3, -0.3])
    _obs, _ = qpos_match_env.reset(options={'target_qpos': target})
    assert qpos_match_env.env.task.target_qpos is not None

    # A plain reset (no options) must not keep the previous episode's goal,
    # otherwise get_termination() can fire on the stale target.
    qpos_match_env.reset()
    assert qpos_match_env.env.task.target_qpos is None


def test_qpos_match_termination_tolerance(qpos_match_env):
    qpos_match_env.reset(options={'target_qpos': np.zeros(2)})
    task = qpos_match_env.env.task
    physics = qpos_match_env.env.physics

    task.target_qpos = physics.data.qpos + 0.04
    assert task.get_termination(physics) is not None

    task.target_qpos = physics.data.qpos + 0.06
    assert task.get_termination(physics) is None
