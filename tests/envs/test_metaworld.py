import gymnasium as gym
import numpy as np
import pytest


pytest.importorskip('metaworld')

import stable_worldmodel  # noqa: E402, F401  (registers the swm/* env ids)


MT10 = [
    'swm/MetaWorldReach-v0',
    'swm/MetaWorldPush-v0',
    'swm/MetaWorldPickPlace-v0',
    'swm/MetaWorldDoorOpen-v0',
    'swm/MetaWorldDrawerOpen-v0',
    'swm/MetaWorldDrawerClose-v0',
    'swm/MetaWorldButtonPressTopdown-v0',
    'swm/MetaWorldPegInsertSide-v0',
    'swm/MetaWorldWindowOpen-v0',
    'swm/MetaWorldWindowClose-v0',
]

# Tasks with a free manipulable object (mass / friction / scale / position).
OBJECT_TASKS = [
    'swm/MetaWorldReach-v0',
    'swm/MetaWorldPush-v0',
    'swm/MetaWorldPickPlace-v0',
]


@pytest.mark.parametrize('env_id', MT10)
def test_metaworld_initialization(env_id):
    env = gym.make(env_id)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (39,)
    assert env.action_space.shape == (4,)

    obs, info = env.reset(seed=0)
    assert obs.shape == (39,)
    for key in ('env_name', 'state', 'proprio', 'goal_state'):
        assert key in info, f"missing info key '{key}'"
    assert info['env_name'] == gym.spec(env_id).kwargs['env_name']
    env.close()


@pytest.mark.parametrize('env_id', MT10)
def test_metaworld_step_output(env_id):
    env = gym.make(env_id)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )
    assert obs.shape == (39,)
    assert info['state'].shape == (39,)
    assert 'success' in info and 'env_name' in info
    assert np.isfinite(reward)
    env.close()


def test_metaworld_visual_randomization():
    env = gym.make('swm/MetaWorldPush-v0')
    color = np.array([0.5, 0.1, 0.9])
    env.reset(
        seed=0,
        options={
            'variation_values': {
                'object.color': color,
                'table.color': color,
            }
        },
    )

    vs = env.get_wrapper_attr('variation_space')
    np.testing.assert_allclose(vs['object']['color'].value, color)

    # The override must reach the MuJoCo model, not just the variation space.
    base = env.get_wrapper_attr('_base')
    objgeom = env.get_wrapper_attr('_objgeom_id')
    np.testing.assert_allclose(base.model.geom_rgba[objgeom][:3], color)
    env.close()


@pytest.mark.parametrize('env_id', OBJECT_TASKS)
def test_metaworld_object_physics_applied(env_id):
    env = gym.make(env_id)
    env.reset(
        seed=0,
        options={
            'variation_values': {
                'object.mass': np.array([3.0]),
                'object.friction': np.array([1.5]),
            }
        },
    )
    base = env.get_wrapper_attr('_base')
    obj_body = env.get_wrapper_attr('_obj_body_id')
    objgeom = env.get_wrapper_attr('_objgeom_id')
    np.testing.assert_allclose(base.model.body_mass[obj_body], 3.0)
    np.testing.assert_allclose(base.model.geom_friction[objgeom][0], 1.5)
    env.close()


def test_metaworld_goal_override_reaches_base():
    env = gym.make('swm/MetaWorldReach-v0')
    goal = np.array([0.1, 0.8, 0.2])
    env.reset(seed=0, options={'variation_values': {'goal.position': goal}})

    base = env.get_wrapper_attr('_base')
    np.testing.assert_allclose(base._target_pos, goal)
    env.close()


def test_metaworld_mass_init_value_persists():
    """init_value sets a fixed mass that survives resets without options."""
    env = gym.make(
        'swm/MetaWorldPush-v0', init_value={'object.mass': np.array([2.5])}
    )
    base = env.get_wrapper_attr('_base')
    obj_body = env.get_wrapper_attr('_obj_body_id')

    env.reset(seed=0)
    np.testing.assert_allclose(base.model.body_mass[obj_body], 2.5)
    env.reset(seed=1)
    np.testing.assert_allclose(base.model.body_mass[obj_body], 2.5)
    env.close()


def test_metaworld_variation_space_is_per_task():
    """Each task only advertises the factors its MuJoCo model can honor."""
    reach = gym.make('swm/MetaWorldReach-v0')
    reach.reset(seed=0)
    reach_keys = set(reach.get_wrapper_attr('variation_space').sampling_order)
    assert {'object.color', 'object.mass', 'goal.position'} <= reach_keys
    reach.close()

    # Window has no recolorable task object and no free object body.
    window = gym.make('swm/MetaWorldWindowOpen-v0')
    window.reset(seed=0)
    window_keys = set(
        window.get_wrapper_attr('variation_space').sampling_order
    )
    assert 'object.color' not in window_keys
    assert 'object.mass' not in window_keys
    assert 'goal.position' in window_keys
    window.close()

    # Top-down button press exposes no goal site.
    button = gym.make('swm/MetaWorldButtonPressTopdown-v0')
    button.reset(seed=0)
    button_keys = set(
        button.get_wrapper_attr('variation_space').sampling_order
    )
    assert 'goal.position' not in button_keys
    button.close()


@pytest.mark.parametrize(
    'env_id', ['swm/MetaWorldPickPlace-v0', 'swm/MetaWorldDrawerOpen-v0']
)
def test_metaworld_randomize_all(env_id):
    """Sampling every factor at once must not raise on any task type."""
    env = gym.make(env_id)
    env.reset(seed=0, options={'variation': ['all']})
    env.step(env.action_space.sample())
    env.close()


def _reach_until_success(env, max_steps=200):
    """Drive the hand to the target and return (info, terminated) at success."""
    obs, info = env.reset(seed=0)
    target = np.asarray(env.unwrapped._target_pos, dtype=np.float32)
    terminated = False
    for _ in range(max_steps):
        delta = np.clip((target - obs[:3]) * 12.5, -1.0, 1.0)
        action = np.array([delta[0], delta[1], delta[2], 0.0], np.float32)
        obs, _, terminated, _, info = env.step(action)
        if info['success'] >= 1.0 or terminated:
            break
    return info, terminated


def test_metaworld_terminates_on_success():
    """By default, solving the task ends the episode as terminated, so World's
    success-rate eval (which keys off `terminated`) registers the success."""
    env = gym.make('swm/MetaWorldReach-v0')
    info, terminated = _reach_until_success(env)
    assert info['success'] >= 1.0
    assert terminated
    env.close()


def test_metaworld_terminate_on_success_can_be_disabled():
    """With terminate_on_success=False, success is reported but not terminal."""
    env = gym.make('swm/MetaWorldReach-v0', terminate_on_success=False)
    info, terminated = _reach_until_success(env)
    assert info['success'] >= 1.0
    assert not terminated
    env.close()
