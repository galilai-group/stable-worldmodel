import gymnasium as gym
import numpy as np

import stable_worldmodel  # noqa: F401


def test_floor_rotation_keeps_the_floor_under_the_walker():
    env = gym.make('swm/WalkerDMControl-v0')
    try:
        floor = env.unwrapped._mjcf_model.find('geom', 'floor')
        original = np.asarray(floor.pos, dtype=np.float64).copy()
        for degrees in (0.0, 0.125, -0.25, 10.0):
            env.reset(
                options={'variation_values': {'floor.rotation_y': [degrees]}}
            )
            physics = env.unwrapped.dmc_env.physics
            position = np.asarray(
                physics.named.data.geom_xpos['floor'], dtype=np.float64
            )
            rotation = np.asarray(
                physics.named.data.geom_xmat['floor'], dtype=np.float64
            ).reshape(3, 3)
            angle = np.deg2rad(degrees)
            expected = np.array(
                [
                    np.cos(angle) * original[0] + np.sin(angle) * original[2],
                    original[1],
                    -np.sin(angle) * original[0] + np.cos(angle) * original[2],
                ]
            )
            np.testing.assert_allclose(position, expected, atol=1e-5)
            np.testing.assert_allclose(position @ rotation[:, 2], 0, atol=1e-5)
    finally:
        env.close()
