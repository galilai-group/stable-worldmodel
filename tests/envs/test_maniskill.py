import gymnasium as gym
import numpy as np
import pytest

import stable_worldmodel as swm
from stable_worldmodel.envs.maniskill import TASK_SPECS
from stable_worldmodel.envs.maniskill.env import (
    DEFAULT_VARIATIONS,
    build_variation_space,
)


# --- CPU-safe: registry sanity (no mani_skill / GPU needed) ---------------


def test_maniskill_specs_are_registered():
    ids = [spec['id'] for spec in TASK_SPECS]
    assert len(ids) == len(set(ids)), 'TASK_SPECS ids must be unique'
    for env_id in ids:
        assert env_id in swm.envs.WORLDS, f'{env_id} not registered'


def test_maniskill_specs_have_task_id():
    for spec in TASK_SPECS:
        assert 'id' in spec and 'task_id' in spec, (
            f'spec missing id/task_id: {spec}'
        )


def test_maniskill_variation_space_factors():
    """The env exposes real (verified-applied) Factors of Variation."""
    vs = build_variation_space()
    top_level = set(vs.spaces.keys())
    assert {'light', 'camera', 'object', 'rendering'} <= top_level, top_level
    names = set(vs.names())
    assert 'rendering.transparent_arm' in names
    for key in DEFAULT_VARIATIONS:
        assert key in names, f'{key} missing from variation_space ({names})'


# --- GPU + Vulkan required: skipped cleanly without mani_skill installed ---


@pytest.mark.parametrize(
    'env_id',
    ['swm/MSPickCube-v0', 'swm/SimplerCarrotOnPlate-v0'],
)
def test_maniskill_environment_rollout(env_id):
    pytest.importorskip('mani_skill')

    env = gym.make(env_id)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape[0] > 0

    obs, info = env.reset(seed=0)
    assert 'env_name' in info, 'env_name must be present'
    assert 'proprio' in info, 'proprio state must be exposed'
    assert 'state' in info, 'flattened state must be exposed'

    img = env.render()
    assert img.ndim == 3 and img.shape[-1] == 3, 'render must be (H, W, 3)'
    assert img.dtype == np.uint8

    obs, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert 'success' in info

    env.close()
