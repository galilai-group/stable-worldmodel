import numpy as np
import pytest

from stable_worldmodel.envs.pusht.env import PushT


@pytest.mark.parametrize(
    'shape,angle,expected_success,expected_distance',
    [
        ('L', np.pi, False, np.pi),
        ('T', np.pi, False, np.pi),
        ('Z', np.pi, True, np.pi),
        ('square', np.pi / 2, True, np.pi / 2),
        ('I', np.pi, True, np.pi),
        ('small_tee', np.pi, False, np.pi),
        ('+', np.pi / 2, True, np.pi / 2),
    ],
)
def test_pusht_eval_state_respects_block_shape_rotation_symmetry(
    shape,
    angle,
    expected_success,
    expected_distance,
):
    env = PushT()
    env.variation_space.set_value({'block.shape': env.shapes.index(shape)})

    goal_state = np.array([100.0, 100.0, 256.0, 256.0, 0.0, 0.0, 0.0])
    cur_state = goal_state.copy()
    cur_state[4] = angle

    success, distance = env.eval_state(goal_state, cur_state)

    assert bool(success) is expected_success
    assert distance == pytest.approx(expected_distance)
