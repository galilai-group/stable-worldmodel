import numpy as np

from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.gymnasium_control.base import GymControlWrapper


class CartPoleWrapper(GymControlWrapper):
    DEFAULT_VARIATIONS = ()

    def __init__(self, env_id='CartPole-v1', render_mode=None, **kwargs):
        super().__init__(env_id, render_mode=render_mode, **kwargs)
        self.variation_space = swm_spaces.Dict(
            {
                'physics': swm_spaces.Dict(
                    {
                        'gravity': swm_spaces.Box(
                            low=4.9,
                            high=14.7,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([9.8], dtype=np.float32),
                        ),
                        'masscart': swm_spaces.Box(
                            low=0.5,
                            high=1.5,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([1.0], dtype=np.float32),
                        ),
                        'masspole': swm_spaces.Box(
                            low=0.05,
                            high=0.5,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([0.1], dtype=np.float32),
                        ),
                        'length': swm_spaces.Box(
                            low=0.25,
                            high=0.75,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([0.5], dtype=np.float32),
                        ),
                        'force_mag': swm_spaces.Box(
                            low=5.0,
                            high=15.0,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([10.0], dtype=np.float32),
                        ),
                        'tau': swm_spaces.Box(
                            low=0.01,
                            high=0.04,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([0.02], dtype=np.float32),
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
                        'cart': swm_spaces.Box(
                            low=0.0,
                            high=0.4,
                            shape=(3,),
                            dtype=np.float32,
                            init_value=np.array(
                                [0.0, 0.0, 0.0], dtype=np.float32
                            ),
                        ),
                        'pole': swm_spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(3,),
                            dtype=np.float32,
                            init_value=np.array(
                                [202 / 255, 152 / 255, 101 / 255],
                                dtype=np.float32,
                            ),
                        ),
                        'axle': swm_spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(3,),
                            dtype=np.float32,
                            init_value=np.array(
                                [129 / 255, 132 / 255, 203 / 255],
                                dtype=np.float32,
                            ),
                        ),
                    }
                ),
            }
        )

    def _apply_physical_variations(self, active):
        env = self.env.unwrapped
        phys = self.variation_space['physics']
        env.gravity = float(phys['gravity'].value[0])
        env.masscart = float(phys['masscart'].value[0])
        env.masspole = float(phys['masspole'].value[0])
        env.length = float(phys['length'].value[0])
        env.force_mag = float(phys['force_mag'].value[0])
        env.tau = float(phys['tau'].value[0])
        env.total_mass = env.masspole + env.masscart
        env.polemass_length = env.masspole * env.length

    def _color_swaps(self):
        v = self.variation_space['visuals']
        return [
            ((255, 255, 255), v['bg'].value),
            ((0, 0, 0), v['cart'].value),
            ((202, 152, 101), v['pole'].value),
            ((129, 132, 203), v['axle'].value),
        ]


if __name__ == '__main__':
    env = CartPoleWrapper()
    obs, info = env.reset(seed=0)
    print('obs:', obs, 'info keys:', list(info))
    obs, info = env.reset(
        seed=1, options={'variation': ['physics.gravity', 'physics.length']}
    )
    print(
        'gravity =',
        env.env.unwrapped.gravity,
        'length =',
        env.env.unwrapped.length,
    )
    env.close()
