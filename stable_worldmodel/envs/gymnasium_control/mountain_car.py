import numpy as np

from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.gymnasium_control.base import GymControlWrapper


def _mc_visuals_space():
    return swm_spaces.Dict(
        {
            'bg': swm_spaces.Box(
                low=0.6,
                high=1.0,
                shape=(3,),
                dtype=np.float32,
                init_value=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            ),
            'fg': swm_spaces.Box(
                low=0.0,
                high=0.4,
                shape=(3,),
                dtype=np.float32,
                init_value=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            ),
        }
    )


class MountainCarWrapper(GymControlWrapper):
    DEFAULT_VARIATIONS = (
        'physics.gravity',
        'physics.force',
        'physics.max_speed',
        'visuals.bg',
        'visuals.fg',
    )

    def __init__(self, env_id='MountainCar-v0', render_mode=None, **kwargs):
        super().__init__(env_id, render_mode=render_mode, **kwargs)
        self.variation_space = swm_spaces.Dict(
            {
                'physics': swm_spaces.Dict(
                    {
                        'gravity': swm_spaces.Box(
                            low=0.00125,
                            high=0.00375,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([0.0025], dtype=np.float32),
                        ),
                        'force': swm_spaces.Box(
                            low=0.0005,
                            high=0.0015,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([0.001], dtype=np.float32),
                        ),
                        'max_speed': swm_spaces.Box(
                            low=0.035,
                            high=0.105,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([0.07], dtype=np.float32),
                        ),
                    }
                ),
                'visuals': _mc_visuals_space(),
            }
        )

    def _apply_physical_variations(self, active):
        env = self.env.unwrapped
        phys = self.variation_space['physics']
        env.gravity = float(phys['gravity'].value[0])
        env.force = float(phys['force'].value[0])
        env.max_speed = float(phys['max_speed'].value[0])

    def _color_swaps(self):
        v = self.variation_space['visuals']
        return [
            ((255, 255, 255), v['bg'].value),
            ((0, 0, 0), v['fg'].value),
        ]


class MountainCarContinuousWrapper(GymControlWrapper):
    """Continuous MountainCar.

    Note: gravity is hardcoded (0.0025) inside the env's ``step`` and not exposed
    as an attribute, so it cannot be varied without monkey-patching the step fn.
    """

    DEFAULT_VARIATIONS = (
        'physics.power',
        'physics.max_speed',
        'visuals.bg',
        'visuals.fg',
    )

    def __init__(
        self, env_id='MountainCarContinuous-v0', render_mode=None, **kwargs
    ):
        super().__init__(env_id, render_mode=render_mode, **kwargs)
        self.variation_space = swm_spaces.Dict(
            {
                'physics': swm_spaces.Dict(
                    {
                        'power': swm_spaces.Box(
                            low=0.00075,
                            high=0.00225,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([0.0015], dtype=np.float32),
                        ),
                        'max_speed': swm_spaces.Box(
                            low=0.035,
                            high=0.105,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([0.07], dtype=np.float32),
                        ),
                    }
                ),
                'visuals': _mc_visuals_space(),
            }
        )

    def _apply_physical_variations(self, active):
        env = self.env.unwrapped
        phys = self.variation_space['physics']
        env.power = float(phys['power'].value[0])
        env.max_speed = float(phys['max_speed'].value[0])

    def _color_swaps(self):
        v = self.variation_space['visuals']
        return [
            ((255, 255, 255), v['bg'].value),
            ((0, 0, 0), v['fg'].value),
        ]
