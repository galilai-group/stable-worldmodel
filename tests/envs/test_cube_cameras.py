"""Tests for the configurable cameras of the OGBench Cube env."""

import numpy as np
import pytest


pytest.importorskip('mujoco')
pytest.importorskip('ogbench')

import mujoco  # noqa: E402

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.envs.ogbench.cube_env import (  # noqa: E402
    DEFAULT_CAMERAS,
    CubeEnv,
)


BASELINE = 0.06


def _stereo_cameras(baseline=BASELINE):
    """A camera at the default view and one shifted along its x axis."""
    front = DEFAULT_CAMERAS['front_pixels']
    x_axis = np.array(front['xyaxes'][:3])
    right_pos = tuple(np.array(front['pos']) + baseline * x_axis)
    return [
        {'name': 'left', **front},
        {'name': 'right', 'pos': right_pos, 'xyaxes': front['xyaxes']},
    ]


def _copy_state(src, dst):
    dst.set_state(src._data.qpos.copy(), src._data.qvel.copy())
    dst._data.mocap_pos[:] = src._data.mocap_pos
    dst._data.mocap_quat[:] = src._data.mocap_quat
    mujoco.mj_forward(dst._model, dst._data)


def _camera_names(env):
    return [
        mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        for i in range(env._model.ncam)
    ]


@pytest.fixture(scope='module')
def default_env():
    env = CubeEnv(env_type='single')
    env.reset(seed=0)
    yield env
    env.close()


@pytest.fixture(scope='module')
def stereo_env():
    env = CubeEnv(env_type='single', cameras=_stereo_cameras())
    env.reset(seed=0)
    yield env
    env.close()


def test_default_cameras_unchanged(default_env):
    assert list(default_env.cameras) == list(DEFAULT_CAMERAS)
    assert set(_camera_names(default_env)) >= set(DEFAULT_CAMERAS)
    assert default_env._multiview is False
    space = default_env.variation_space['camera']['angle_delta']
    assert space.shape == (1, 2)
    # single view: plain image from `front_pixels`
    image = default_env.render_multiview()
    assert isinstance(image, np.ndarray)
    assert image.shape == (224, 224, 3)
    np.testing.assert_allclose(
        image.astype(int),
        default_env.render(camera='front_pixels').astype(int),
        atol=1,
    )


def test_multiview_flag_unchanged():
    env = CubeEnv(env_type='single', multiview=True)
    env.reset(seed=0)
    views = env.render_multiview()
    assert list(views) == list(DEFAULT_CAMERAS)
    assert env.variation_space['camera']['angle_delta'].shape == (2, 2)
    env.close()


def test_configured_cameras_replace_builtins(stereo_env):
    assert list(stereo_env.cameras) == ['left', 'right']
    names = _camera_names(stereo_env)
    assert 'left' in names and 'right' in names
    assert not set(names) & set(DEFAULT_CAMERAS)


def test_configured_views_in_order(stereo_env):
    views = stereo_env.render_multiview()
    assert list(views) == ['left', 'right']
    for image in views.values():
        assert image.shape == (224, 224, 3)
        assert image.dtype == np.uint8
    # render() defaults to the first configured camera
    np.testing.assert_allclose(
        stereo_env.render().astype(int), views['left'].astype(int), atol=1
    )
    assert stereo_env.variation_space['camera']['angle_delta'].shape == (
        2,
        2,
    )


def test_configured_camera_pose(default_env, stereo_env):
    _copy_state(default_env, stereo_env)
    views = stereo_env.render_multiview()
    reference = default_env.render(camera='front_pixels')
    # a camera configured at the default pose reproduces the default view
    np.testing.assert_allclose(
        views['left'].astype(int), reference.astype(int), atol=1
    )
    assert np.abs(views['right'].astype(int) - reference).mean() > 1.0

    model, data = stereo_env._model, stereo_env._data
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'left')
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'right')
    offset = data.cam_xpos[right] - data.cam_xpos[left]
    x_axis = data.cam_xmat[left].reshape(3, 3)[:, 0]
    np.testing.assert_allclose(offset, BASELINE * x_axis, atol=1e-9)
    np.testing.assert_allclose(
        data.cam_xmat[right], data.cam_xmat[left], atol=1e-9
    )
    assert model.cam_fovy[right] == model.cam_fovy[left]


def test_extra_camera_fields_are_passed_through():
    cameras = _stereo_cameras()
    cameras[1]['fovy'] = 60.0
    env = CubeEnv(env_type='single', cameras=cameras)
    env.reset(seed=0)
    right = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_CAMERA, 'right')
    assert env._model.cam_fovy[right] == 60.0
    env.close()


def test_single_configured_camera_is_plain_view():
    env = CubeEnv(env_type='single', cameras=_stereo_cameras()[:1])
    env.reset(seed=0)
    assert env._multiview is False
    assert isinstance(env.render_multiview(), np.ndarray)
    env.close()


@pytest.mark.parametrize(
    'cameras, kwargs, match',
    [
        ([], {}, 'at least one'),
        ([{'name': 'a', 'pos': (0, 0, 1)}], {}, 'missing'),
        ([{'pos': (0, 0, 1), 'xyaxes': (1, 0, 0, 0, 1, 0)}], {}, 'missing'),
        (_stereo_cameras()[:1] * 2, {}, 'duplicate'),
        (_stereo_cameras(), {'multiview': False}, 'multiview=False'),
    ],
)
def test_invalid_cameras(cameras, kwargs, match):
    with pytest.raises(ValueError, match=match):
        CubeEnv(env_type='single', cameras=cameras, **kwargs)


def test_world_emits_one_pixels_key_per_camera():
    world = swm.World(
        'swm/OGBCube-v0',
        num_envs=2,
        image_shape=(64, 64),
        max_episode_steps=5,
        env_type='single',
        cameras=_stereo_cameras(),
    )
    world.reset(seed=0)
    keys = sorted(k for k in world.infos if k.startswith('pixels'))
    assert keys == ['pixels.left', 'pixels.right']
    assert world.infos['pixels.left'].shape == (2, 1, 64, 64, 3)
    assert not np.array_equal(
        world.infos['pixels.left'], world.infos['pixels.right']
    )
    world.close()
