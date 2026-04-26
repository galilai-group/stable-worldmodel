import numpy as np

from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.gymnasium_control.base import GymControlWrapper


class PendulumWrapper(GymControlWrapper):
    DEFAULT_VARIATIONS = (
        'physics.g',
        'physics.m',
        'physics.l',
        'physics.max_torque',
        'physics.max_speed',
        'physics.dt',
        'visuals.bg',
        'visuals.rod',
        'visuals.hub',
    )

    def __init__(self, env_id='Pendulum-v1', render_mode=None, **kwargs):
        super().__init__(env_id, render_mode=render_mode, **kwargs)
        self.variation_space = swm_spaces.Dict(
            {
                'physics': swm_spaces.Dict(
                    {
                        'g': swm_spaces.Box(
                            low=5.0,
                            high=15.0,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([10.0], dtype=np.float32),
                        ),
                        'm': swm_spaces.Box(
                            low=0.5,
                            high=1.5,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([1.0], dtype=np.float32),
                        ),
                        'l': swm_spaces.Box(
                            low=0.5,
                            high=1.5,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([1.0], dtype=np.float32),
                        ),
                        'max_torque': swm_spaces.Box(
                            low=1.0,
                            high=4.0,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([2.0], dtype=np.float32),
                        ),
                        'max_speed': swm_spaces.Box(
                            low=4.0,
                            high=12.0,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([8.0], dtype=np.float32),
                        ),
                        'dt': swm_spaces.Box(
                            low=0.025,
                            high=0.1,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([0.05], dtype=np.float32),
                        ),
                    }
                ),
                'visuals': swm_spaces.Dict(
                    {
                        'bg': swm_spaces.Box(
                            low=0.6,
                            high=1.0,
                            shape=(3,),
                            dtype=np.float32,
                            init_value=np.array(
                                [1.0, 1.0, 1.0], dtype=np.float32
                            ),
                        ),
                        'rod': swm_spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(3,),
                            dtype=np.float32,
                            init_value=np.array(
                                [204 / 255, 77 / 255, 77 / 255],
                                dtype=np.float32,
                            ),
                        ),
                        'hub': swm_spaces.Box(
                            low=0.0,
                            high=0.4,
                            shape=(3,),
                            dtype=np.float32,
                            init_value=np.array(
                                [0.0, 0.0, 0.0], dtype=np.float32
                            ),
                        ),
                    }
                ),
            }
        )

    def _apply_physical_variations(self, active):
        env = self.env.unwrapped
        phys = self.variation_space['physics']
        env.g = float(phys['g'].value[0])
        env.m = float(phys['m'].value[0])
        env.l = float(phys['l'].value[0])
        env.max_torque = float(phys['max_torque'].value[0])
        env.max_speed = float(phys['max_speed'].value[0])
        env.dt = float(phys['dt'].value[0])

    def _color_swaps(self):
        v = self.variation_space['visuals']
        return [
            ((255, 255, 255), v['bg'].value),
            ((204, 77, 77), v['rod'].value),
            ((0, 0, 0), v['hub'].value),
        ]
