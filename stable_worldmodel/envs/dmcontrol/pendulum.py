import os
import tempfile

import numpy as np
from dm_control import mjcf
from dm_control.rl import control
from dm_control.suite import pendulum
from dm_control.suite.wrappers import action_scale

from stable_worldmodel import spaces as swm_space
from stable_worldmodel.envs.dmcontrol.dmcontrol import DMControlWrapper


_DEFAULT_TIME_LIMIT = 20


class PendulumDMControlWrapper(DMControlWrapper):
    def __init__(self, seed=None, environment_kwargs=None):
        xml, assets = pendulum.get_model_and_assets()
        xml = xml.replace(b'file="./common/', b'file="common/')
        suite_dir = os.path.dirname(pendulum.__file__)  # .../dm_control/suite
        self._mjcf_model = mjcf.from_xml_string(
            xml,
            model_dir=suite_dir,
            assets=assets or {},
        )
        self.compile_model(seed=seed, environment_kwargs=environment_kwargs)
        super().__init__(self.env, "pendulum")
        self.variation_space = swm_space.Dict(
            {  # TODO check default values to match original cheetah env
                "agent": swm_space.Dict(
                    {
                        "color": swm_space.Box(
                            low=0.0,
                            high=1.0,
                            shape=(3,),
                            dtype=np.float64,
                            init_value=np.array([0.7, 0.5, 0.3], dtype=np.float64),
                        ),
                        "pole_density": swm_space.Box(
                            low=500,
                            high=1500,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([1000], dtype=np.float32),
                        ),
                        "mass_density": swm_space.Box(
                            low=500,
                            high=1500,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([1000], dtype=np.float32),
                        ),
                        # TODO implement mass shape variation
                        "mass_shape": swm_space.Discrete(2, init_value=1),  # 0: box, 1: sphere
                    }
                ),
                "floor": swm_space.Dict(
                    {
                        "color": swm_space.Box(
                            low=0.0,
                            high=1.0,
                            shape=(2, 3),
                            dtype=np.float64,
                            init_value=np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], dtype=np.float64),
                        ),
                    }
                ),
                "light": swm_space.Dict(
                    {
                        "intensity": swm_space.Box(
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

    def compile_model(self, seed=None, environment_kwargs=None):
        """Compile the MJCF model into DMControl env."""
        assert self._mjcf_model is not None, "No MJCF model to compile!"
        self._mjcf_tempdir = tempfile.TemporaryDirectory()
        mjcf.export_with_assets(
            self._mjcf_model,
            self._mjcf_tempdir.name,
            out_file_name="pendulum.xml",
        )
        xml_path = os.path.join(self._mjcf_tempdir.name, "pendulum.xml")
        physics = pendulum.Physics.from_xml_path(xml_path)
        task = pendulum.SwingUp(random=seed)
        environment_kwargs = environment_kwargs or {}
        env = control.Environment(physics, task, time_limit=_DEFAULT_TIME_LIMIT, **environment_kwargs)
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
        grid_texture = mjcf_model.find("texture", "grid")
        texture_changed = grid_texture.rgb1 is None or not np.allclose(
            grid_texture.rgb1, self.variation_space["floor"]["color"].value[0]
        )
        texture_changed = texture_changed or (
            grid_texture.rgb2 is None
            or not np.allclose(grid_texture.rgb2, self.variation_space["floor"]["color"].value[1])
        )
        grid_texture.rgb1 = self.variation_space["floor"]["color"].value[0]
        grid_texture.rgb2 = self.variation_space["floor"]["color"].value[1]

        # Modify agent (acrobot) color via material
        agent_color_changed = False

        desired_rgb = np.asarray(self.variation_space["agent"]["color"].value, dtype=np.float32).reshape(3)
        desired_rgba = np.concatenate([desired_rgb, np.array([1.0], dtype=np.float32)], axis=0)

        self_mat = mjcf_model.find("material", "self")
        assert self_mat is not None, "Expected material named 'self'"

        if self_mat.rgba is None or not np.allclose(np.asarray(self_mat.rgba, dtype=np.float32), desired_rgba):
            agent_color_changed = True
        self_mat.rgba = desired_rgba

        mass_changed = False

        # Modify pole density
        pole_geom = mjcf_model.find("geom", "pole")
        base = pole_geom.density if pole_geom.density is not None else 1000.0
        desired_density = float(np.asarray(self.variation_space["agent"]["pole_density"].value).reshape(-1)[0])
        if not np.allclose(base, desired_density):
            mass_changed = True
        pole_geom.density = desired_density

        # Modify mass density
        mass_geom = mjcf_model.find("geom", "mass")
        base = mass_geom.density if mass_geom.density is not None else 1000.0
        desired_density = float(np.asarray(self.variation_space["agent"]["mass_density"].value).reshape(-1)[0])
        if not np.allclose(base, desired_density):
            mass_changed = True
        mass_geom.density = desired_density

        # Modify light intensity if a global light exists.
        light_changed = False
        light = mjcf_model.find("light", "light")
        desired_diffuse = self.variation_space["light"]["intensity"].value[0] * np.ones((3), dtype=np.float32)
        light_changed = light.diffuse is None or not np.allclose(light.diffuse, desired_diffuse)
        light.diffuse = desired_diffuse

        # If any properties changed, mark the model as dirty.
        if light_changed or texture_changed or mass_changed or agent_color_changed:
            self.mark_dirty()
        return mjcf_model


if __name__ == "__main__":
    env = PendulumDMControlWrapper(seed=0)
    obs, info = env.reset()
    print("obs shape:", obs.shape)
    print("info:", info)
    env.close()
