import numpy as np

from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.gymnasium_control.base import GymControlWrapper


class AcrobotWrapper(GymControlWrapper):
    DEFAULT_VARIATIONS = ()

    def __init__(self, env_id='Acrobot-v1', render_mode=None, **kwargs):
        super().__init__(env_id, render_mode=render_mode, **kwargs)
        self.variation_space = swm_spaces.Dict(
            {
                'physics': swm_spaces.Dict(
                    {
                        'link_length_1': swm_spaces.Box(
                            low=0.5,
                            high=1.5,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([1.0], dtype=np.float32),
                        ),
                        'link_length_2': swm_spaces.Box(
                            low=0.5,
                            high=1.5,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([1.0], dtype=np.float32),
                        ),
                        'link_mass_1': swm_spaces.Box(
                            low=0.5,
                            high=1.5,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([1.0], dtype=np.float32),
                        ),
                        'link_mass_2': swm_spaces.Box(
                            low=0.5,
                            high=1.5,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([1.0], dtype=np.float32),
                        ),
                        'link_com_pos_1': swm_spaces.Box(
                            low=0.25,
                            high=0.75,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([0.5], dtype=np.float32),
                        ),
                        'link_com_pos_2': swm_spaces.Box(
                            low=0.25,
                            high=0.75,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([0.5], dtype=np.float32),
                        ),
                        'link_moi': swm_spaces.Box(
                            low=0.5,
                            high=1.5,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([1.0], dtype=np.float32),
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
                        'line': swm_spaces.Box(
                            low=0.0,
                            high=0.4,
                            shape=(3,),
                            dtype=np.float32,
                            init_value=np.array(
                                [0.0, 0.0, 0.0], dtype=np.float32
                            ),
                        ),
                        'link': swm_spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(3,),
                            dtype=np.float32,
                            init_value=np.array(
                                [0.0, 204 / 255, 204 / 255],
                                dtype=np.float32,
                            ),
                        ),
                        'joint': swm_spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(3,),
                            dtype=np.float32,
                            init_value=np.array(
                                [204 / 255, 204 / 255, 0.0],
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
        env.LINK_LENGTH_1 = float(phys['link_length_1'].value[0])
        env.LINK_LENGTH_2 = float(phys['link_length_2'].value[0])
        env.LINK_MASS_1 = float(phys['link_mass_1'].value[0])
        env.LINK_MASS_2 = float(phys['link_mass_2'].value[0])
        env.LINK_COM_POS_1 = float(phys['link_com_pos_1'].value[0])
        env.LINK_COM_POS_2 = float(phys['link_com_pos_2'].value[0])
        env.LINK_MOI = float(phys['link_moi'].value[0])

    def _color_swaps(self):
        v = self.variation_space['visuals']
        return [
            ((255, 255, 255), v['bg'].value),
            ((0, 204, 204), v['link'].value),
            ((204, 204, 0), v['joint'].value),
            ((0, 0, 0), v['line'].value),
        ]
