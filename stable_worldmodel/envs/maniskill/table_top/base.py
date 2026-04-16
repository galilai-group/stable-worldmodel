
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
        spaces_dict = {
            'robot':      swm_spaces.Dict({'color': _color_box(0.8, 0.8, 0.8)}),
            'ground':     swm_spaces.Dict({'color': _color_box(0.5, 0.5, 0.5)}),
            'ambient':    swm_spaces.Dict({'color': _color_box(0.3, 0.3, 0.3)}),
            'background': swm_spaces.Dict({'color': _color_box(0.0, 0.0, 0.0)}),
            'light':      swm_spaces.Dict({'intensity': _scalar_box(0.3, 1.5, 1.0)}),
        }
        for key, cfg in self.actors.items():
            spaces_dict[key] = swm_spaces.Dict({'color': _color_box(*cfg['rgb'])})
        return swm_spaces.Dict(spaces_dict)


    def reset(self, seed=None, options=None):
        options = options or {}
        swm_spaces.reset_variation_space(self.variation_space, seed, options)
        obs, info = super().reset(seed=seed, options=options)
        self.apply_variations(self._compute_active_keys(options))
        return obs, info

    def _compute_active_keys(self, options):
        active = set()
        var_keys = options.get('variation', ()) or ()
        if 'all' in var_keys:
            active.update(self.variation_space.spaces.keys())
        else:
            for k in var_keys:
                active.add(k.split('.', 1)[0])
        for k in (options.get('variation_values') or {}).keys():
            active.add(k.split('.', 1)[0])
        return active

    def apply_variations(self, active):
        if not active:
            return
        base = self.env.unwrapped
        if 'ground' in active:
            self._apply_ground(base)
        if 'robot' in active:
            self._apply_robot(base)
        if {'ambient', 'background', 'light'} & active:
            self._apply_scene(base, active)
        for key, cfg in self.actors.items():
            if key in active:
                self._apply_actor(base, key, cfg)

    def _apply_actor(self, base, key, cfg):
        actor = _first_attr(base, *cfg['attrs'])
        _set_render_color(actor, self._rgba(key))


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

    def _apply_robot(self, base):
        agent = getattr(base, 'agent', None)
        if agent is None:
            return
        rgba = self._rgba('robot')
        for robot in _iter_robots(agent):
            for link in getattr(robot, 'links', []) or []:
                _set_render_color(link, rgba)

    def _apply_scene(self, base, active):
        scene = _first_attr(base, 'scene', '_scene')
        sub_scenes = getattr(scene, 'sub_scenes', None) or (
            [scene] if scene is not None else []
        )

        do_light = 'light' in active
        do_ambient = 'ambient' in active
        do_background = 'background' in active

        light_color = None
        if do_light:
            intensity = self._scalar('light', 'intensity')
            light_color = [intensity, intensity, intensity]
        ambient = self._rgb('ambient') if do_ambient else None
        background = (
            np.asarray(self.variation_space['background']['color'].value, dtype=np.float32)
            if do_background else None
        )

        for sub in sub_scenes:
            if sub is None:
                continue
            if do_light:
                for light in getattr(sub, 'lights', []) or []:
                    if hasattr(light, 'set_color'):
                        light.set_color(light_color)
            if do_ambient:
                if hasattr(sub, 'set_ambient_light'):
                    sub.set_ambient_light(ambient)
                elif hasattr(sub, 'ambient_light'):
                    sub.ambient_light = ambient
            if do_background:
                render_sys = getattr(sub, 'render_system', None)
                if render_sys is not None and hasattr(render_sys, 'set_cubemap'):
                    try:
                        render_sys.set_cubemap(_solid_cubemap(background))
                    except Exception:
                        pass
