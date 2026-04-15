
import os
import tempfile

import numpy as np

from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.maniskill.maniskill import ManiSkillWrapper


_CUBEMAP_CACHE: dict[tuple[int, int, int], object] = {}


def _solid_cubemap(color):
    import sapien.render as sr
    from PIL import Image

    key = tuple(int(round(c * 255)) for c in color[:3])
    if key in _CUBEMAP_CACHE:
        return _CUBEMAP_CACHE[key]

    tmpdir = tempfile.mkdtemp(prefix='sapien_bg_')
    pixel = np.array([[list(key)]], dtype=np.uint8)
    paths = []
    for face in ('px', 'nx', 'py', 'ny', 'pz', 'nz'):
        path = os.path.join(tmpdir, f'{face}.png')
        Image.fromarray(pixel).save(path)
        paths.append(path)

    cubemap = sr.RenderCubemap(*paths)
    _CUBEMAP_CACHE[key] = cubemap
    return cubemap


def _first_attr(obj, *names):
    if obj is None:
        return None
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _iter_entities(actor):
    if actor is None:
        return
    direct = getattr(actor, '_objs', None) or getattr(actor, '_bodies', None)
    if direct:
        for item in direct:
            yield from _iter_entities(item)
        return
    entity = getattr(actor, 'entity', None)
    if entity is not None and entity is not actor:
        yield entity
        return
    if hasattr(actor, 'get_components'):
        yield actor


def _iter_rigid_components(actor):
    for entity in _iter_entities(actor):
        for component in entity.get_components():
            if hasattr(component, 'mass') and hasattr(component, 'get_collision_shapes'):
                yield component


def _iter_render_shapes(actor):
    for entity in _iter_entities(actor):
        for component in entity.get_components():
            for shape in getattr(component, 'render_shapes', None) or []:
                yield shape


def _set_render_color(actor, rgba):
    for shape in _iter_render_shapes(actor):
        try:
            mat = getattr(shape, 'material', None)
            if mat is None:
                continue
            if hasattr(mat, 'set_base_color'):
                mat.set_base_color(rgba)
            else:
                mat.base_color = rgba
        except RuntimeError:
            continue


def _set_rigid_body_physics(
    actor,
    *,
    mass=None,
    friction=None,
    restitution=None,
    linear_damping=None,
    angular_damping=None,
):
    for component in _iter_rigid_components(actor):
        if mass is not None:
            try:
                component.mass = float(mass)
            except Exception:
                if hasattr(component, 'set_mass'):
                    component.set_mass(float(mass))
        if linear_damping is not None and hasattr(component, 'set_linear_damping'):
            component.set_linear_damping(float(linear_damping))
        if angular_damping is not None and hasattr(component, 'set_angular_damping'):
            component.set_angular_damping(float(angular_damping))

        if friction is None and restitution is None:
            continue
        for shape in component.get_collision_shapes():
            mat = shape.get_physical_material()
            if mat is None:
                continue
            if friction is not None:
                if hasattr(mat, 'set_static_friction'):
                    mat.set_static_friction(float(friction))
                if hasattr(mat, 'set_dynamic_friction'):
                    mat.set_dynamic_friction(float(friction))
            if restitution is not None and hasattr(mat, 'set_restitution'):
                mat.set_restitution(float(restitution))
            if hasattr(shape, 'set_physical_material'):
                shape.set_physical_material(mat)


def _set_scene_gravity(scene, gravity):
    px = _first_attr(scene, 'px', 'physx_system')
    if px is None:
        return
    if hasattr(px, 'set_scene_config'):
        try:
            cfg = px.get_scene_config()
            cfg.gravity = gravity
            px.set_scene_config(cfg)
            return
        except Exception:
            pass
    if hasattr(px, 'gravity'):
        px.gravity = gravity
    elif hasattr(scene, 'set_gravity'):
        scene.set_gravity(gravity)


def _color_box(r, g, b):
    return swm_spaces.Box(
        low=0.0, high=1.0, shape=(3,), dtype=np.float32,
        init_value=np.array([r, g, b], dtype=np.float32),
    )


def _scalar_box(low, high, init):
    return swm_spaces.Box(
        low=low, high=high, shape=(1,), dtype=np.float32,
        init_value=np.array([init], dtype=np.float32),
    )


def _as_float(value, default=0.0):
    try:
        return float(np.asarray(value).reshape(-1)[0])
    except Exception:
        return default


def _iter_robots(agent):
    if agent is None:
        return
    robot = getattr(agent, 'robot', None)
    if robot is not None:
        yield robot
        return
    for sub in getattr(agent, 'agents', []) or []:
        sub_robot = getattr(sub, 'robot', None)
        if sub_robot is not None:
            yield sub_robot


def _build_object_physics_space(mass=(0.02, 0.5, 0.1)):
    return swm_spaces.Dict({
        'mass':            _scalar_box(*mass),
        'friction':        _scalar_box(0.1, 2.0, 0.5),
        'restitution':     _scalar_box(0.0, 0.8, 0.1),
        'linear_damping':  _scalar_box(0.0, 2.0, 0.0),
        'angular_damping': _scalar_box(0.0, 2.0, 0.05),
    })


class TableTopManiSkillWrapper(ManiSkillWrapper):

    task_id: str = ''
    default_control_mode: str = 'pd_ee_delta_pose'
    actors: dict = {}

    def __init__(self, control_mode: str | None = None, seed: int | None = None, **make_kwargs):
        if not self.task_id:
            raise ValueError(f'{type(self).__name__}.task_id must be set')
        super().__init__(
            task_id=self.task_id,
            control_mode=control_mode or self.default_control_mode,
            seed=seed,
            **make_kwargs,
        )
        self.variation_space = self._build_variation_space()


    def _build_variation_space(self):
        space = swm_spaces.Dict({
            'robot':      swm_spaces.Dict({'color': _color_box(0.8, 0.8, 0.8)}),
            'ground':     swm_spaces.Dict({'color': _color_box(0.5, 0.5, 0.5)}),
            'ambient':    swm_spaces.Dict({'color': _color_box(0.3, 0.3, 0.3)}),
            'background': swm_spaces.Dict({'color': _color_box(0.0, 0.0, 0.0)}),
            'light':      swm_spaces.Dict({'intensity': _scalar_box(0.3, 1.5, 1.0)}),
            'physics': swm_spaces.Dict({
                'gravity_x': _scalar_box(-3.0, 3.0, 0.0),
                'gravity_y': _scalar_box(-3.0, 3.0, 0.0),
                'gravity_z': _scalar_box(-15.0, -1.0, -9.81),
            }),
            'ground_physics': swm_spaces.Dict({
                'friction':    _scalar_box(0.1, 2.0, 1.0),
                'restitution': _scalar_box(0.0, 0.8, 0.0),
            }),
            'robot_physics': swm_spaces.Dict({
                'joint_friction': _scalar_box(0.0, 1.0, 0.0),
                'joint_damping':  _scalar_box(0.0, 10.0, 0.0),
            }),
        })
        for key, cfg in self.actors.items():
            space[key] = swm_spaces.Dict({'color': _color_box(*cfg['rgb'])})
            if cfg.get('physics', False):
                space[f'{key}_physics'] = _build_object_physics_space(
                    mass=cfg.get('mass', (0.02, 0.5, 0.1)),
                )
        return space


    def reset(self, seed=None, options=None):
        options = options or {}
        swm_spaces.reset_variation_space(self.variation_space, seed, options)
        obs, info = super().reset(seed=seed, options=options)
        self.apply_variations()
        return obs, info

    def apply_variations(self):
        base = self.env.unwrapped
        self._apply_ground(base)
        self._apply_robot(base)
        self._apply_scene(base)
        for key, cfg in self.actors.items():
            self._apply_actor(base, key, cfg)

    def _apply_actor(self, base, key, cfg):
        actor = _first_attr(base, *cfg['attrs'])
        _set_render_color(actor, self._rgba(key))
        if not cfg.get('physics', False):
            return
        _set_rigid_body_physics(
            actor,
            mass=self._scalar(f'{key}_physics', 'mass'),
            friction=self._scalar(f'{key}_physics', 'friction'),
            restitution=self._scalar(f'{key}_physics', 'restitution'),
            linear_damping=self._scalar(f'{key}_physics', 'linear_damping'),
            angular_damping=self._scalar(f'{key}_physics', 'angular_damping'),
        )


    def _rgba(self, key):
        c = np.asarray(self.variation_space[key]['color'].value, dtype=np.float32)
        return [float(c[0]), float(c[1]), float(c[2]), 1.0]

    def _rgb(self, key):
        c = np.asarray(self.variation_space[key]['color'].value, dtype=np.float32)
        return [float(c[0]), float(c[1]), float(c[2])]

    def _scalar(self, group, key):
        return _as_float(self.variation_space[group][key].value)


    def _apply_ground(self, base):
        table_scene = _first_attr(base, 'table_scene', 'scene_builder')
        ground = (
            getattr(table_scene, 'ground', None)
            or _first_attr(base, 'ground', '_ground')
        )
        _set_render_color(ground, self._rgba('ground'))
        _set_rigid_body_physics(
            ground,
            friction=self._scalar('ground_physics', 'friction'),
            restitution=self._scalar('ground_physics', 'restitution'),
        )

    def _apply_robot(self, base):
        agent = getattr(base, 'agent', None)
        if agent is None:
            return
        rgba = self._rgba('robot')
        joint_friction = self._scalar('robot_physics', 'joint_friction')
        joint_damping = self._scalar('robot_physics', 'joint_damping')

        for robot in _iter_robots(agent):
            for link in getattr(robot, 'links', []) or []:
                _set_render_color(link, rgba)
            for joint in getattr(robot, 'active_joints', []) or []:
                if hasattr(joint, 'set_friction'):
                    joint.set_friction(joint_friction)
                if hasattr(joint, 'set_drive_properties'):
                    stiffness = _as_float(getattr(joint, 'stiffness', 0.0))
                    joint.set_drive_properties(stiffness, joint_damping)

    def _apply_scene(self, base):
        scene = _first_attr(base, 'scene', '_scene')
        sub_scenes = getattr(scene, 'sub_scenes', None) or (
            [scene] if scene is not None else []
        )

        intensity = self._scalar('light', 'intensity')
        light_color = [intensity, intensity, intensity]
        ambient = self._rgb('ambient')
        background = np.asarray(
            self.variation_space['background']['color'].value, dtype=np.float32,
        )

        for sub in sub_scenes:
            if sub is None:
                continue
            for light in getattr(sub, 'lights', []) or []:
                if hasattr(light, 'set_color'):
                    light.set_color(light_color)
            if hasattr(sub, 'set_ambient_light'):
                sub.set_ambient_light(ambient)
            elif hasattr(sub, 'ambient_light'):
                sub.ambient_light = ambient

            render_sys = getattr(sub, 'render_system', None)
            if render_sys is not None and hasattr(render_sys, 'set_cubemap'):
                try:
                    render_sys.set_cubemap(_solid_cubemap(background))
                except Exception:
                    pass

        gravity = [
            self._scalar('physics', 'gravity_x'),
            self._scalar('physics', 'gravity_y'),
            self._scalar('physics', 'gravity_z'),
        ]
        _set_scene_gravity(scene, gravity)
