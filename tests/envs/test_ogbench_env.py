import numpy as np
import pytest


pytest.importorskip('ogbench')

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.envs.ogbench.cube_env import CubeEnv  # noqa: E402
from stable_worldmodel.envs.ogbench.maze_env import MazeEnv  # noqa: E402
from stable_worldmodel.policy import RandomPolicy  # noqa: E402


class TestCubeGoalInfo:
    def test_reset_and_step_info_contain_rendered_goal(self):
        env = CubeEnv(env_type='single', height=64, width=64)
        try:
            _, info = env.reset(seed=0)
            assert 'goal' in info
            goal = info['goal']
            assert isinstance(goal, np.ndarray)
            assert goal.shape == (64, 64, 3)
            assert goal.dtype == np.uint8

            _, _, _, _, info = env.step(env.action_space.sample())
            assert 'goal' in info
            np.testing.assert_array_equal(info['goal'], goal)
        finally:
            env.close()

    def test_goal_differs_from_initial_render(self):
        env = CubeEnv(env_type='single', height=64, width=64)
        try:
            _, info = env.reset(seed=0)
            # The goal image shows the cube at the target position, not
            # the initial one.
            assert not np.array_equal(info['goal'], env.render())
        finally:
            env.close()

    def test_render_goal_opt_out(self):
        env = CubeEnv(env_type='single', height=64, width=64)
        try:
            _, info = env.reset(seed=0, options={'render_goal': False})
            assert 'goal' not in info
        finally:
            env.close()

    def test_data_collection_mode_has_no_goal(self):
        env = CubeEnv(
            env_type='single',
            mode='data_collection',
            terminate_at_goal=False,
            height=64,
            width=64,
        )
        try:
            _, info = env.reset(seed=0)
            assert 'goal' not in info

            _, _, _, _, info = env.step(env.action_space.sample())
            assert 'goal' not in info
        finally:
            env.close()


class TestMazeGoalInfo:
    def test_reset_and_step_info_contain_rendered_goal(self):
        env = MazeEnv(
            maze_type='arena',
            height=64,
            width=64,
            render_mode='rgb_array',
        )
        try:
            _, info = env.reset(seed=0)
            assert 'goal' in info
            goal = info['goal']
            assert isinstance(goal, np.ndarray)
            assert goal.shape == (64, 64, 3)
            assert goal.dtype == np.uint8

            _, _, _, _, info = env.step(env.action_space.sample())
            assert 'goal' in info
            np.testing.assert_array_equal(info['goal'], goal)
        finally:
            env.close()


class GoalCheckingPolicy(RandomPolicy):
    """Random policy that records the goal seen at every planning call."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.goal_shapes = []

    def get_action(self, info_dict, **kwargs):
        assert 'goal' in info_dict, "'goal' missing from infos"
        self.goal_shapes.append(info_dict['goal'].shape)
        return super().get_action(info_dict, **kwargs)


class TestEvaluateWithoutDataset:
    def test_cube_episodic_evaluation_provides_goal(self):
        world = swm.World(
            'swm/OGBCube-v0',
            num_envs=1,
            image_shape=(32, 32),
            max_episode_steps=2,
            env_type='single',
            height=64,
            width=64,
            visualize_info=False,
        )
        try:
            policy = GoalCheckingPolicy(seed=0)
            world.set_policy(policy)
            results = world.evaluate(episodes=1, seed=0)

            assert 'success_rate' in results
            assert len(results['episode_successes']) == 1
            assert policy.goal_shapes
            assert all(
                shape == (1, 1, 32, 32, 3) for shape in policy.goal_shapes
            )
        finally:
            world.close()
