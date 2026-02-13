import os
import tempfile

import numpy as np
from dm_control import mjcf
from dm_control.rl import control
from dm_control.suite import quadruped
from dm_control.suite.wrappers import action_scale

from stable_worldmodel import spaces as swm_space
from stable_worldmodel.envs.dmcontrol.dmcontrol import DMControlWrapper


_DEFAULT_TIME_LIMIT = 20
_CONTROL_TIMESTEP = 0.02

_WALK_SPEED = 0.5
_RUN_SPEED = 5

_TASKS = ('walk', 'run')

_TASK_SPEEDS = {
    'walk': _WALK_SPEED,
    'run': _RUN_SPEED,
}


class QuadrupedDMControlWrapper(DMControlWrapper):
    def __init__(self, task='walk', seed=None, environment_kwargs=None):
        if task not in _TASKS:
            raise ValueError(
                f"Unknown task '{task}'. Must be one of {list(_TASKS)}"
            )
        self._desired_speed = _TASK_SPEEDS[task]
        xml = quadruped.make_model(
            floor_size=_DEFAULT_TIME_LIMIT * self._desired_speed
        )
        assets = quadruped.common.ASSETS
        xml = xml.replace(b'file="./common/', b'file="common/')
        suite_dir = os.path.dirname(quadruped.__file__)  # .../dm_control/suite
        self._mjcf_model = mjcf.from_xml_string(
            xml,
            model_dir=suite_dir,
            assets=assets or {},
        )
        self.compile_model(seed=seed, environment_kwargs=environment_kwargs)
        super().__init__(self.env, 'quadruped')
        self.variation_space = swm_space.Dict(
            {
                'agent': swm_space.Dict(
                    {
                        'color': swm_space.Box(
                            low=0.0,
                            high=1.0,
                            shape=(3,),
                            dtype=np.float64,
                            init_value=np.array(
                                [0.6, 0.3, 0.3], dtype=np.float64
                            ),
                        ),
                        'torso_density': swm_space.Box(
                            low=500,
                            high=1500,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([1000], dtype=np.float32),
                        ),
                        'foot_back_left_density': swm_space.Box(
                            low=500,
                            high=1500,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([1000], dtype=np.float32),
                        ),
                        # TODO lock back left knee joint (by default it is unlocked 1)
                        'knee_back_left_locked': swm_space.Discrete(
                            2, init_value=1
                        ),
                    }
                ),
                'floor': swm_space.Dict(
                    {
                        'friction': swm_space.Box(
                            low=0.0,
                            high=1.0,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([1.0], dtype=np.float32),
                        ),
                        'color': swm_space.Box(
                            low=0.0,
                            high=1.0,
                            shape=(2, 3),
                            dtype=np.float64,
                            init_value=np.array(
                                [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
                                dtype=np.float64,
                            ),
                        ),
                    }
                ),
                'light': swm_space.Dict(
                    {
                        'intensity': swm_space.Box(
                            low=0.0,
                            high=1.0,
                            shape=(1,),
                            dtype=np.float64,
                            init_value=np.array([0.7], dtype=np.float64),
                        ),
                    }
                ),
            }
        )

    @property
    def info(self):
        info = super().info
        info['torso_velocity'] = self.env.physics.torso_velocity().copy()
        return info

    def compile_model(self, seed=None, environment_kwargs=None):
        """Compile the MJCF model into DMControl env."""
        assert self._mjcf_model is not None, 'No MJCF model to compile!'
        self._mjcf_tempdir = tempfile.TemporaryDirectory()
        mjcf.export_with_assets(
            self._mjcf_model,
            self._mjcf_tempdir.name,
            out_file_name='quadruped.xml',
        )
        xml_path = os.path.join(self._mjcf_tempdir.name, 'quadruped.xml')
        physics = quadruped.Physics.from_xml_path(xml_path)
        task = quadruped.Move(random=seed, desired_speed=self._desired_speed)
        environment_kwargs = environment_kwargs or {}
        env = control.Environment(
            physics,
            task,
            time_limit=_DEFAULT_TIME_LIMIT,
            control_timestep=_CONTROL_TIMESTEP,
            **environment_kwargs,
        )
        env = action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)
        self.env = env
        # Mark the environment as clean.
        self._dirty = False

    def modify_mjcf_model(self, mjcf_model):
        """Apply visual variations to the MuJoCo model based on variation space.

        Modifies the MJCF model XML to apply sampled variations including floor colors,
        and lighting. Only variations that are enabled in the variation_options are applied.

        Args:
            mjcf_model (mjcf.RootElement): The MuJoCo XML model to modify.

        Returns:
            mjcf.RootElement: The modified model with variations applied.

        Note:
            - Variations are only applied if specified in variation_options during reset
            - Some variations call self.mark_dirty() to trigger recompilation
        """
        # Modify floor color
        grid_texture = mjcf_model.find('texture', 'grid')
        texture_changed = grid_texture.rgb1 is None or not np.allclose(
            grid_texture.rgb1, self.variation_space['floor']['color'].value[0]
        )
        texture_changed = texture_changed or (
            grid_texture.rgb2 is None
            or not np.allclose(
                grid_texture.rgb2,
                self.variation_space['floor']['color'].value[1],
            )
        )
        grid_texture.rgb1 = self.variation_space['floor']['color'].value[0]
        grid_texture.rgb2 = self.variation_space['floor']['color'].value[1]

        # Modify agent (quadruped) color via material
        agent_color_changed = False

        desired_rgb = np.asarray(
            self.variation_space['agent']['color'].value, dtype=np.float32
        ).reshape(3)
        desired_rgba = np.concatenate(
            [desired_rgb, np.array([1.0], dtype=np.float32)], axis=0
        )

        self_mat = mjcf_model.find('material', 'self')
        assert self_mat is not None, "Expected material named 'self'"

        if self_mat.rgba is None or not np.allclose(
            np.asarray(self_mat.rgba, dtype=np.float32), desired_rgba
        ):
            agent_color_changed = True
        self_mat.rgba = desired_rgba

        mass_changed = False

        # Modify floor friction
        floor_geom = mjcf_model.find('geom', 'floor')
        desired_friction = float(
            np.asarray(
                self.variation_space['floor']['friction'].value
            ).reshape(-1)[0]
        )

        # MJCF may store geom.friction as None if not specified in XML.
        # MuJoCo default is: [1, 0.005, 0.0001]
        if floor_geom.friction is None:
            current_friction = np.array([1.0, 0.005, 0.0001], dtype=np.float32)
        else:
            current_friction = np.asarray(
                floor_geom.friction, dtype=np.float32
            ).copy()

        new_friction = current_friction.copy()
        new_friction[0] = desired_friction

        friction_changed = not np.allclose(current_friction, new_friction)
        floor_geom.friction = new_friction

        mass_changed = False
        # Modify torso density
        torso_geom = mjcf_model.find('geom', 'torso')
        base = torso_geom.density if torso_geom.density is not None else 1000.0
        desired_density = float(
            np.asarray(
                self.variation_space['agent']['torso_density'].value
            ).reshape(-1)[0]
        )
        if not np.allclose(base, desired_density):
            mass_changed = True
        torso_geom.density = desired_density

        # Modify left back foot density
        bfoot_geom = mjcf_model.find('geom', 'foot_back_left')
        base = bfoot_geom.density if bfoot_geom.density is not None else 1000.0
        desired_density = float(
            np.asarray(
                self.variation_space['agent']['foot_back_left_density'].value
            ).reshape(-1)[0]
        )
        if not np.allclose(base, desired_density):
            mass_changed = True
        bfoot_geom.density = desired_density

        # Modify light intensity if a global light exists.
        light_changed = False
        light = mjcf_model.find('light', 'light')
        desired_diffuse = self.variation_space['light']['intensity'].value[
            0
        ] * np.ones((3), dtype=np.float32)
        light_changed = light.diffuse is None or not np.allclose(
            light.diffuse, desired_diffuse
        )
        light.diffuse = desired_diffuse

        # If any properties changed, mark the model as dirty.
        if (
            light_changed
            or texture_changed
            or friction_changed
            or mass_changed
            or agent_color_changed
        ):
            self.mark_dirty()
        return mjcf_model


if __name__ == '__main__':
    env = QuadrupedDMControlWrapper(seed=0)
    obs, info = env.reset()
    print('obs shape:', obs.shape)
    print('info:', info)
    env.close()
