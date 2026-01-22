import os
import tempfile

import numpy as np
from dm_control import mjcf
from dm_control.rl import control
from dm_control.suite import manipulator
from dm_control.suite.wrappers import action_scale

from stable_worldmodel import spaces as swm_space
from stable_worldmodel.envs.dmcontrol.dmcontrol import DMControlWrapper


_CONTROL_TIMESTEP = 0.01  # (Seconds)
_TIME_LIMIT = 10  # (Seconds)

_SMALL_TARGET = 0.015


class ManipulatorDMControlWrapper(DMControlWrapper):
    def __init__(self, seed=None, environment_kwargs=None):
        xml, assets = manipulator.make_model(use_peg=False, insert=False)
        xml = xml.replace(b'file="./common/', b'file="common/')
        suite_dir = os.path.dirname(manipulator.__file__)  # .../dm_control/suite
        self._mjcf_model = mjcf.from_xml_string(
            xml,
            model_dir=suite_dir,
            assets=assets or {},
        )
        self.compile_model(seed=seed, environment_kwargs=environment_kwargs)
        super().__init__(self.env, "manipulator")
        self.variation_space = swm_space.Dict(
            {
                "agent": swm_space.Dict(
                    {
                        "color": swm_space.Box(
                            low=0.0,
                            high=1.0,
                            shape=(3,),
                            dtype=np.float64,
                            init_value=np.array([0.7, 0.5, 0.3], dtype=np.float64),
                        ),
                        "upper_arm_density": swm_space.Box(
                            low=500,
                            high=1500,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([1000], dtype=np.float32),
                        ),
                        "hand_density": swm_space.Box(
                            low=500,
                            high=1500,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([1000], dtype=np.float32),
                        ),
                        # TODO should modify length of arm
                        "upper_arm_length": swm_space.Box(
                            low=500,
                            high=1500,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([1000], dtype=np.float32),
                        ),
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
                "target": swm_space.Dict(
                    {
                        "color": swm_space.Box(
                            low=0.0,
                            high=1.0,
                            shape=(3,),
                            dtype=np.float64,
                            init_value=np.array([0.6, 0.3, 0.3], dtype=np.float64),
                        ),
                        "shape": swm_space.Discrete(2, init_value=1),  # 0: box, 1: sphere
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
            out_file_name="manipulator.xml",
        )
        xml_path = os.path.join(self._mjcf_tempdir.name, "manipulator.xml")
        physics = manipulator.Physics.from_xml_path(xml_path)
        task = manipulator.Bring(use_peg=False, insert=False, fully_observable=True, random=seed)
        environment_kwargs = environment_kwargs or {}
        env = control.Environment(
            physics, task, time_limit=_TIME_LIMIT, control_timestep=_CONTROL_TIMESTEP, **environment_kwargs
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

        # Modify agent (manipulator) color via material
        agent_color_changed = False

        desired_rgb = np.asarray(self.variation_space["agent"]["color"].value, dtype=np.float32).reshape(3)
        desired_rgba = np.concatenate([desired_rgb, np.array([1.0], dtype=np.float32)], axis=0)

        self_mat = mjcf_model.find("material", "self")
        assert self_mat is not None, "Expected material named 'self'"

        if self_mat.rgba is None or not np.allclose(np.asarray(self_mat.rgba, dtype=np.float32), desired_rgba):
            agent_color_changed = True
        self_mat.rgba = desired_rgba

        mass_changed = False

        # Modify arm density
        arm_geom = mjcf_model.find("geom", "upper_arm")
        base = arm_geom.density if arm_geom.density is not None else 1000.0
        desired_density = float(np.asarray(self.variation_space["agent"]["upper_arm_density"].value).reshape(-1)[0])
        if not np.allclose(base, desired_density):
            mass_changed = True
        arm_geom.density = desired_density

        # Modify hand density
        hand_geom = mjcf_model.find("geom", "hand")
        base = hand_geom.density if hand_geom.density is not None else 1000.0
        desired_density = float(np.asarray(self.variation_space["agent"]["hand_density"].value).reshape(-1)[0])
        if not np.allclose(base, desired_density):
            mass_changed = True
        hand_geom.density = desired_density

        # Modify light intensity if a global light exists.
        light_changed = False
        light = mjcf_model.find("light", "light")
        desired_diffuse = self.variation_space["light"]["intensity"].value[0] * np.ones((3), dtype=np.float32)
        light_changed = light.diffuse is None or not np.allclose(light.diffuse, desired_diffuse)
        light.diffuse = desired_diffuse

        # Modify target appearance (color, shape)
        target_changed = False

        target_geom = mjcf_model.find("geom", "target_ball")
        target_mat = mjcf_model.find("material", "target")

        assert target_geom is not None, "Expected geom named 'target_ball'"
        assert target_mat is not None, "Expected material named 'target'"

        # ----- Color -----
        desired_rgb = np.asarray(self.variation_space["target"]["color"].value, dtype=np.float32).reshape(3)
        desired_rgba = np.concatenate([desired_rgb, [1.0]], axis=0)

        if target_mat.rgba is None or not np.allclose(target_mat.rgba, desired_rgba):
            target_changed = True
        target_mat.rgba = desired_rgba

        # ----- Shape -----
        # 0 = box, 1 = sphere
        shape_id = int(self.variation_space["target"]["shape"].value)

        if shape_id == 0:
            desired_type = "box"
            desired_size = np.array([_SMALL_TARGET, _SMALL_TARGET, _SMALL_TARGET], dtype=np.float32)
        else:
            desired_type = "sphere"
            desired_size = np.array([_SMALL_TARGET], dtype=np.float32)

        if target_geom.type != desired_type:
            target_changed = True
        target_geom.type = desired_type
        target_geom.size = desired_size

        # If any properties changed, mark the model as dirty.
        if light_changed or texture_changed or mass_changed or target_changed or agent_color_changed:
            self.mark_dirty()
        return mjcf_model


if __name__ == "__main__":
    env = ManipulatorDMControlWrapper(seed=0)
    obs, info = env.reset()
    print("obs shape:", obs.shape)
    print("info:", info)
    env.close()
