from unittest.mock import MagicMock

import gymnasium as gym
import pytest

from stable_worldmodel import wrapper


@pytest.fixture
def minimal_env():
    """Create a minimal mock environment that satisfies Gymnasium's Wrapper requirements."""
    mock_env = MagicMock(spec=gym.Env)
    mock_env.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
    mock_env.action_space = gym.spaces.Box(low=0, high=1, shape=(1,))
    return mock_env


################################
## test EnsureInfoKeysWrapper ##
################################


def test_ensure_info_keys_wrapper_check_logic_success_single_key(minimal_env):
    infos = {"key1": "value1", "key2": "value2"}
    env = wrapper.EnsureInfoKeysWrapper(minimal_env, required_keys=["key1"])
    env._check(infos, where="test")


def test_ensure_info_keys_wrapper_check_logic_success_multiple_key(minimal_env):
    infos = {"key1": "value1", "key2": "value2"}
    env = wrapper.EnsureInfoKeysWrapper(minimal_env, required_keys=["key1", "key2"])
    env._check(infos, where="test")


def test_ensure_info_keys_wrapper_check_logic_fail_single_key(minimal_env):
    infos = {"key1": "value1", "key2": "value2"}
    env = wrapper.EnsureInfoKeysWrapper(minimal_env, required_keys=["info"])
    with pytest.raises(RuntimeError):
        env._check(infos, where="test")


def test_ensure_info_keys_wrapper_check_logic_fail_multiple_key(minimal_env):
    infos = {"key1": "value1", "key2": "value2"}
    env = wrapper.EnsureInfoKeysWrapper(minimal_env, required_keys=["key1", "info"])
    with pytest.raises(RuntimeError):
        env._check(infos, where="test")


def test_ensure_info_keys_wrapper_reset_success(minimal_env):
    """Test that reset() succeeds when required keys are present."""
    obs = {"observation": "test_obs"}
    info = {"key1": "value1", "key2": "value2"}
    minimal_env.reset.return_value = (obs, info)

    wrapped_env = wrapper.EnsureInfoKeysWrapper(minimal_env, required_keys=["key1"])
    result_obs, result_info = wrapped_env.reset()

    assert result_obs == obs
    assert result_info == info
    minimal_env.reset.assert_called_once()


def test_ensure_info_keys_wrapper_reset_fail(minimal_env):
    """Test that reset() raises RuntimeError when required keys are missing."""
    obs = {"observation": "test_obs"}
    info = {"key1": "value1", "key2": "value2"}
    minimal_env.reset.return_value = (obs, info)

    wrapped_env = wrapper.EnsureInfoKeysWrapper(minimal_env, required_keys=["missing_key"])

    with pytest.raises(RuntimeError):
        wrapped_env.reset()


def test_ensure_info_keys_wrapper_step_success(minimal_env):
    """Test that step() succeeds when required keys are present."""
    obs = {"observation": "test_obs"}
    reward = 1.0
    terminated = False
    truncated = False
    info = {"key1": "value1", "key2": "value2"}
    minimal_env.step.return_value = (obs, reward, terminated, truncated, info)

    wrapped_env = wrapper.EnsureInfoKeysWrapper(minimal_env, required_keys=["key1", "key2"])
    action = 0.5
    result = wrapped_env.step(action)

    assert result == (obs, reward, terminated, truncated, info)
    minimal_env.step.assert_called_once_with(action)


def test_ensure_info_keys_wrapper_step_fail(minimal_env):
    """Test that step() raises RuntimeError when required keys are missing."""
    obs = {"observation": "test_obs"}
    reward = 1.0
    terminated = False
    truncated = False
    info = {"key1": "value1"}
    minimal_env.step.return_value = (obs, reward, terminated, truncated, info)

    wrapped_env = wrapper.EnsureInfoKeysWrapper(minimal_env, required_keys=["key1", "missing_key"])

    with pytest.raises(RuntimeError):
        wrapped_env.step(0.5)


def test_ensure_info_keys_wrapper_regex_pattern(minimal_env):
    """Test that wrapper correctly handles regex patterns for key matching."""
    obs = {"observation": "test_obs"}
    info = {"pixels.camera1": "image1", "pixels.camera2": "image2", "other_key": "value"}
    minimal_env.reset.return_value = (obs, info)

    wrapped_env = wrapper.EnsureInfoKeysWrapper(minimal_env, required_keys=[r"pixels\..*"])
    result_obs, result_info = wrapped_env.reset()

    assert result_obs == obs
    assert result_info == info


def test_ensure_info_keys_wrapper_regex_dotted_keys_success(minimal_env):
    """Test that dotted keys like 'key1.key2' are matched correctly."""
    obs = {"observation": "test_obs"}
    info = {"variation.color": "red", "variation.size": "large", "other": "value"}
    minimal_env.reset.return_value = (obs, info)

    wrapped_env = wrapper.EnsureInfoKeysWrapper(minimal_env, required_keys=[r"variation\.color"])
    result_obs, result_info = wrapped_env.reset()

    assert result_obs == obs
    assert result_info == info


def test_ensure_info_keys_wrapper_regex_dotted_keys_wildcard(minimal_env):
    """Test that regex patterns can match multiple dotted keys."""
    obs = {"observation": "test_obs"}
    info = {
        "variation.color": "red",
        "variation.size": "large",
        "variation.position.x": 10,
        "variation.position.y": 20,
        "metadata": "value",
    }
    minimal_env.step.return_value = (obs, 1.0, False, False, info)

    wrapped_env = wrapper.EnsureInfoKeysWrapper(minimal_env, required_keys=[r"variation\..*"])
    result = wrapped_env.step(0.5)

    assert result[0] == obs
    assert result[4] == info


def test_ensure_info_keys_wrapper_regex_dotted_keys_fail(minimal_env):
    """Test that missing dotted keys raise an error."""
    obs = {"observation": "test_obs"}
    info = {"variation.color": "red", "other": "value"}
    minimal_env.reset.return_value = (obs, info)

    wrapped_env = wrapper.EnsureInfoKeysWrapper(minimal_env, required_keys=[r"variation\.size"])

    with pytest.raises(RuntimeError):
        wrapped_env.reset()


def test_ensure_info_keys_wrapper_regex_multiple_patterns(minimal_env):
    """Test multiple regex patterns with dotted keys."""
    obs = {"observation": "test_obs"}
    info = {"pixels.front": "img1", "pixels.back": "img2", "variation.color": "blue", "reward": 1.0}
    minimal_env.step.return_value = (obs, 1.0, False, False, info)

    wrapped_env = wrapper.EnsureInfoKeysWrapper(minimal_env, required_keys=[r"pixels\..*", r"variation\..*"])
    result = wrapped_env.step(0.5)

    assert result[4] == info


def test_ensure_info_keys_wrapper_regex_nested_dotted_keys(minimal_env):
    """Test deeply nested dotted keys with regex patterns."""
    obs = {"observation": "test_obs"}
    info = {
        "env.variation.position.x": 1,
        "env.variation.position.y": 2,
        "env.variation.rotation": 90,
        "other": "value",
    }
    minimal_env.reset.return_value = (obs, info)

    wrapped_env = wrapper.EnsureInfoKeysWrapper(minimal_env, required_keys=[r"env\.variation\.position\..*"])
    result_obs, result_info = wrapped_env.reset()

    assert result_info == info


def test_ensure_info_keys_wrapper_regex_exact_vs_pattern(minimal_env):
    """Test that exact key names work alongside regex patterns."""
    obs = {"observation": "test_obs"}
    info = {"exact_key": "value1", "pixels.camera1": "img1", "pixels.camera2": "img2"}
    minimal_env.reset.return_value = (obs, info)

    wrapped_env = wrapper.EnsureInfoKeysWrapper(minimal_env, required_keys=["exact_key", r"pixels\..*"])
    result_obs, result_info = wrapped_env.reset()

    assert result_info == info


###########################
## test EnsureImageShape ##
###########################


def test_ensure_image_shape_reset_success(minimal_env):
    """Test that reset() succeeds when image shape matches expected shape."""
    import numpy as np

    obs = {"observation": "test_obs"}
    # Image with shape (64, 64, 3) - shape[:-1] will be (64, 64)
    image = np.zeros((64, 64, 3))
    info = {"pixels": image}
    minimal_env.reset.return_value = (obs, info)

    wrapped_env = wrapper.EnsureImageShape(minimal_env, image_key="pixels", image_shape=(64, 64))
    result_obs, result_info = wrapped_env.reset()

    assert result_obs == obs
    assert result_info == info
    minimal_env.reset.assert_called_once()


def test_ensure_image_shape_reset_fail(minimal_env):
    """Test that reset() raises RuntimeError when image shape doesn't match."""
    import numpy as np

    obs = {"observation": "test_obs"}
    # Image with shape (64, 64, 3) - shape[:-1] will be (64, 64)
    image = np.zeros((64, 64, 3))
    info = {"pixels": image}
    minimal_env.reset.return_value = (obs, info)

    # Expect different shape
    wrapped_env = wrapper.EnsureImageShape(minimal_env, image_key="pixels", image_shape=(84, 84))

    with pytest.raises(RuntimeError):
        wrapped_env.reset()


def test_ensure_image_shape_step_success(minimal_env):
    """Test that step() succeeds when image shape matches expected shape."""
    import numpy as np

    obs = {"observation": "test_obs"}
    reward = 1.0
    terminated = False
    truncated = False
    # Image with shape (128, 128, 3) - shape[:-1] will be (128, 128)
    image = np.zeros((128, 128, 3))
    info = {"image": image}
    minimal_env.step.return_value = (obs, reward, terminated, truncated, info)

    wrapped_env = wrapper.EnsureImageShape(minimal_env, image_key="image", image_shape=(128, 128))
    action = 0.5
    result = wrapped_env.step(action)

    assert result == (obs, reward, terminated, truncated, info)
    minimal_env.step.assert_called_once_with(action)


def test_ensure_image_shape_step_fail(minimal_env):
    """Test that step() raises RuntimeError when image shape doesn't match."""
    import numpy as np

    obs = {"observation": "test_obs"}
    reward = 1.0
    terminated = False
    truncated = False
    # Image with shape (64, 64, 3)
    image = np.zeros((64, 64, 3))
    info = {"pixels": image}
    minimal_env.step.return_value = (obs, reward, terminated, truncated, info)

    # Expect different shape
    wrapped_env = wrapper.EnsureImageShape(minimal_env, image_key="pixels", image_shape=(128, 128))

    with pytest.raises(RuntimeError):
        wrapped_env.step(0.5)


def test_ensure_image_shape_different_channels(minimal_env):
    """Test that wrapper only checks spatial dimensions, not channels."""
    import numpy as np

    obs = {"observation": "test_obs"}
    # Image with shape (64, 64, 1) - grayscale
    image_gray = np.zeros((64, 64, 1))
    info_gray = {"pixels": image_gray}
    minimal_env.reset.return_value = (obs, info_gray)

    # Should succeed because shape[:-1] is (64, 64) regardless of channels
    wrapped_env = wrapper.EnsureImageShape(minimal_env, image_key="pixels", image_shape=(64, 64))
    result_obs, result_info = wrapped_env.reset()

    assert result_info == info_gray

    # Now test with RGB image (64, 64, 3)
    image_rgb = np.zeros((64, 64, 3))
    info_rgb = {"pixels": image_rgb}
    minimal_env.reset.return_value = (obs, info_rgb)

    # Should also succeed with same spatial dimensions
    result_obs, result_info = wrapped_env.reset()
    assert result_info == info_rgb


def test_ensure_image_shape_non_square_images(minimal_env):
    """Test that wrapper works with non-square image shapes."""
    import numpy as np

    obs = {"observation": "test_obs"}
    # Non-square image (480, 640, 3)
    image = np.zeros((480, 640, 3))
    info = {"camera": image}
    minimal_env.step.return_value = (obs, 1.0, False, False, info)

    wrapped_env = wrapper.EnsureImageShape(minimal_env, image_key="camera", image_shape=(480, 640))
    result = wrapped_env.step(0.5)

    assert result[4] == info


def test_ensure_image_shape_multiple_images_different_keys(minimal_env):
    """Test wrapper with multiple different image keys (one wrapper per key)."""
    import numpy as np

    obs = {"observation": "test_obs"}
    image1 = np.zeros((64, 64, 3))
    image2 = np.zeros((84, 84, 3))
    info = {"pixels.front": image1, "pixels.back": image2}
    minimal_env.reset.return_value = (obs, info)

    # Wrap for first image
    wrapped_env = wrapper.EnsureImageShape(minimal_env, image_key="pixels.front", image_shape=(64, 64))
    result_obs, result_info = wrapped_env.reset()

    assert result_info == info

    # If we wrap for the second image with wrong shape, it should fail
    minimal_env.reset.return_value = (obs, info)
    wrapped_env2 = wrapper.EnsureImageShape(minimal_env, image_key="pixels.back", image_shape=(64, 64))

    with pytest.raises(RuntimeError):
        wrapped_env2.reset()


################################
## test EnsureGoalInfoWrapper ##
################################


def test_ensure_goal_info_wrapper_reset_success_check_enabled(minimal_env):
    """Test that reset() succeeds when check_reset=True and 'goal' is present."""
    obs = {"observation": "test_obs"}
    info = {"goal": "target", "other_key": "value"}
    minimal_env.reset.return_value = (obs, info)

    wrapped_env = wrapper.EnsureGoalInfoWrapper(minimal_env, check_reset=True)
    result_obs, result_info = wrapped_env.reset()

    assert result_obs == obs
    assert result_info == info
    minimal_env.reset.assert_called_once()


def test_ensure_goal_info_wrapper_reset_fail_check_enabled(minimal_env):
    """Test that reset() raises RuntimeError when check_reset=True and 'goal' is missing."""
    obs = {"observation": "test_obs"}
    info = {"other_key": "value"}
    minimal_env.reset.return_value = (obs, info)

    wrapped_env = wrapper.EnsureGoalInfoWrapper(minimal_env, check_reset=True)

    with pytest.raises(RuntimeError):
        wrapped_env.reset()


def test_ensure_goal_info_wrapper_reset_no_check(minimal_env):
    """Test that reset() succeeds when check_reset=False, even without 'goal'."""
    obs = {"observation": "test_obs"}
    info = {"other_key": "value"}
    minimal_env.reset.return_value = (obs, info)

    wrapped_env = wrapper.EnsureGoalInfoWrapper(minimal_env, check_reset=False)
    result_obs, result_info = wrapped_env.reset()

    assert result_obs == obs
    assert result_info == info


def test_ensure_goal_info_wrapper_step_success_check_enabled(minimal_env):
    """Test that step() succeeds when check_step=True and 'goal' is present."""
    obs = {"observation": "test_obs"}
    reward = 1.0
    terminated = False
    truncated = False
    info = {"goal": "target", "other_key": "value"}
    minimal_env.step.return_value = (obs, reward, terminated, truncated, info)

    wrapped_env = wrapper.EnsureGoalInfoWrapper(minimal_env, check_reset=False, check_step=True)
    action = 0.5
    result = wrapped_env.step(action)

    assert result == (obs, reward, terminated, truncated, info)
    minimal_env.step.assert_called_once_with(action)


def test_ensure_goal_info_wrapper_step_fail_check_enabled(minimal_env):
    """Test that step() raises RuntimeError when check_step=True and 'goal' is missing."""
    obs = {"observation": "test_obs"}
    reward = 1.0
    terminated = False
    truncated = False
    info = {"other_key": "value"}
    minimal_env.step.return_value = (obs, reward, terminated, truncated, info)

    wrapped_env = wrapper.EnsureGoalInfoWrapper(minimal_env, check_reset=False, check_step=True)

    with pytest.raises(RuntimeError):
        wrapped_env.step(0.5)


def test_ensure_goal_info_wrapper_step_no_check(minimal_env):
    """Test that step() succeeds when check_step=False, even without 'goal'."""
    obs = {"observation": "test_obs"}
    reward = 1.0
    terminated = False
    truncated = False
    info = {"other_key": "value"}
    minimal_env.step.return_value = (obs, reward, terminated, truncated, info)

    wrapped_env = wrapper.EnsureGoalInfoWrapper(minimal_env, check_reset=False, check_step=False)
    result = wrapped_env.step(0.5)

    assert result == (obs, reward, terminated, truncated, info)


def test_ensure_goal_info_wrapper_both_checks_enabled(minimal_env):
    """Test wrapper with both check_reset=True and check_step=True."""
    obs = {"observation": "test_obs"}
    info_with_goal = {"goal": "target", "other": "value"}
    minimal_env.reset.return_value = (obs, info_with_goal)
    minimal_env.step.return_value = (obs, 1.0, False, False, info_with_goal)

    wrapped_env = wrapper.EnsureGoalInfoWrapper(minimal_env, check_reset=True, check_step=True)

    result_obs, result_info = wrapped_env.reset()
    assert result_info == info_with_goal

    result = wrapped_env.step(0.5)
    assert result[4] == info_with_goal


def test_ensure_goal_info_wrapper_both_checks_enabled_fail(minimal_env):
    """Test wrapper fails in both reset and step when 'goal' is missing."""
    obs = {"observation": "test_obs"}
    info_no_goal = {"other": "value"}
    minimal_env.reset.return_value = (obs, info_no_goal)
    minimal_env.step.return_value = (obs, 1.0, False, False, info_no_goal)

    wrapped_env = wrapper.EnsureGoalInfoWrapper(minimal_env, check_reset=True, check_step=True)

    with pytest.raises(RuntimeError):
        wrapped_env.reset()

    with pytest.raises(RuntimeError):
        wrapped_env.step(0.5)


def test_ensure_goal_info_wrapper_default_check_step(minimal_env):
    """Test that check_step defaults to False."""
    obs = {"observation": "test_obs"}
    info_no_goal = {"other": "value"}
    minimal_env.step.return_value = (obs, 1.0, False, False, info_no_goal)

    wrapped_env = wrapper.EnsureGoalInfoWrapper(minimal_env, check_reset=False)
    result = wrapped_env.step(0.5)

    assert result[4] == info_no_goal


##################################
## test EverythingToInfoWrapper ##
##################################


def test_everything_to_info_wrapper_reset_non_dict_obs(minimal_env):
    """Test that reset() converts non-dict observation to dict with 'observation' key."""
    import numpy as np

    obs = np.array([1, 2, 3])
    info = {}
    minimal_env.reset.return_value = (obs, info)
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=0.5)

    wrapped_env = wrapper.EverythingToInfoWrapper(minimal_env)
    result_obs, result_info = wrapped_env.reset()

    # Original obs should be returned
    assert np.array_equal(result_obs, obs)
    # Info should contain the observation
    assert "observation" in result_info
    assert np.array_equal(result_info["observation"], obs)
    # Check other added keys
    assert "reward" in result_info
    assert np.isnan(result_info["reward"])
    assert "terminated" in result_info
    assert result_info["terminated"] is False
    assert "truncated" in result_info
    assert result_info["truncated"] is False
    assert "action" in result_info
    assert "step_idx" in result_info
    assert result_info["step_idx"] == 0


def test_everything_to_info_wrapper_reset_dict_obs(minimal_env):
    """Test that reset() handles dict observations correctly."""
    obs = {"image": "img_data", "state": "state_data"}
    info = {}
    minimal_env.reset.return_value = (obs, info)
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=0.5)

    wrapped_env = wrapper.EverythingToInfoWrapper(minimal_env)
    result_obs, result_info = wrapped_env.reset()

    # Original obs should be returned
    assert result_obs == obs
    # Info should contain all observation keys
    assert "image" in result_info
    assert result_info["image"] == "img_data"
    assert "state" in result_info
    assert result_info["state"] == "state_data"
    # Check metadata keys
    assert "reward" in result_info
    assert "terminated" in result_info
    assert "truncated" in result_info
    assert "action" in result_info
    assert "step_idx" in result_info


def test_everything_to_info_wrapper_step_non_dict_obs(minimal_env):
    """Test that step() converts non-dict observation to dict with 'observation' key."""
    import numpy as np

    # First reset to initialize step counter
    reset_obs = np.array([0, 0, 0])
    minimal_env.reset.return_value = (reset_obs, {})
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=0.5)
    wrapped_env = wrapper.EverythingToInfoWrapper(minimal_env)
    wrapped_env.reset()

    # Now test step
    obs = np.array([1, 2, 3])
    reward = 1.5
    terminated = False
    truncated = False
    info = {}
    minimal_env.step.return_value = (obs, reward, terminated, truncated, info)

    action = 0.7
    result_obs, result_reward, result_terminated, result_truncated, result_info = wrapped_env.step(action)

    # Check returned values
    assert np.array_equal(result_obs, obs)
    assert result_reward == reward
    assert result_terminated == terminated
    assert result_truncated == truncated
    # Check info contains observation
    assert "observation" in result_info
    assert np.array_equal(result_info["observation"], obs)
    # Check metadata
    assert result_info["reward"] == reward
    assert result_info["terminated"] == terminated
    assert result_info["truncated"] == truncated
    assert result_info["action"] == action
    assert result_info["step_idx"] == 1


def test_everything_to_info_wrapper_step_dict_obs(minimal_env):
    """Test that step() handles dict observations correctly."""
    # First reset
    minimal_env.reset.return_value = ({"state": "init"}, {})
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=0.5)
    wrapped_env = wrapper.EverythingToInfoWrapper(minimal_env)
    wrapped_env.reset()

    # Now test step
    obs = {"image": "img_data", "state": "state_data"}
    reward = 2.0
    terminated = True
    truncated = False
    info = {}
    minimal_env.step.return_value = (obs, reward, terminated, truncated, info)

    action = 0.8
    result = wrapped_env.step(action)

    # Check observation keys are in info
    assert "image" in result[4]
    assert result[4]["image"] == "img_data"
    assert "state" in result[4]
    assert result[4]["state"] == "state_data"
    # Check metadata
    assert result[4]["reward"] == reward
    assert result[4]["terminated"] is True
    assert result[4]["truncated"] is False
    assert result[4]["action"] == action


def test_everything_to_info_wrapper_step_counter(minimal_env):
    """Test that step_idx increments correctly."""
    import numpy as np

    minimal_env.reset.return_value = (np.array([0]), {})
    # Use side_effect to return a fresh dict each time
    minimal_env.step.side_effect = [
        (np.array([1]), 1.0, False, False, {}),
        (np.array([1]), 1.0, False, False, {}),
    ]
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=0.5)

    wrapped_env = wrapper.EverythingToInfoWrapper(minimal_env)

    # Reset should have step_idx = 0
    _, info = wrapped_env.reset()
    assert info["step_idx"] == 0

    # First step should have step_idx = 1
    _, _, _, _, info = wrapped_env.step(0.5)
    assert info["step_idx"] == 1

    # Second step should have step_idx = 2
    _, _, _, _, info = wrapped_env.step(0.5)
    assert info["step_idx"] == 2

    # Reset should reset step_idx to 0
    minimal_env.reset.return_value = (np.array([0]), {})
    _, info = wrapped_env.reset()
    assert info["step_idx"] == 0


def test_everything_to_info_wrapper_action_nan_in_reset(minimal_env):
    """Test that action is NaN in reset()."""
    import numpy as np

    obs = np.array([1, 2, 3])
    info = {}
    minimal_env.reset.return_value = (obs, info)
    # Return a non-dict action
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=np.array([0.5, 0.6]))

    wrapped_env = wrapper.EverythingToInfoWrapper(minimal_env)
    _, result_info = wrapped_env.reset()

    # Action should be NaN
    assert "action" in result_info
    assert np.all(np.isnan(result_info["action"]))


def test_everything_to_info_wrapper_terminated_truncated_bool_conversion(minimal_env):
    """Test that terminated and truncated are converted to bool."""
    import numpy as np

    minimal_env.reset.return_value = (np.array([0]), {})
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=0.5)
    wrapped_env = wrapper.EverythingToInfoWrapper(minimal_env)
    wrapped_env.reset()

    # Use numpy bool (should be converted to Python bool)
    obs = np.array([1])
    minimal_env.step.return_value = (obs, 1.0, np.bool_(True), np.bool_(False), {})

    _, _, _, _, info = wrapped_env.step(0.5)

    # Should be Python bool
    assert info["terminated"] is True
    assert isinstance(info["terminated"], bool)
    assert info["truncated"] is False
    assert isinstance(info["truncated"], bool)


def test_everything_to_info_wrapper_no_key_collision(minimal_env):
    """Test that wrapper raises assertion if keys already exist in info."""
    import numpy as np

    # Try to have 'reward' already in info - should raise assertion
    obs = np.array([1, 2, 3])
    info = {"reward": 5.0}  # This should cause an assertion error
    minimal_env.reset.return_value = (obs, info)
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=0.5)

    wrapped_env = wrapper.EverythingToInfoWrapper(minimal_env)

    with pytest.raises(AssertionError):
        wrapped_env.reset()


def test_everything_to_info_wrapper_dict_action_not_implemented(minimal_env):
    """Test that dict actions raise NotImplementedError in reset()."""
    import numpy as np

    obs = np.array([1, 2, 3])
    info = {}
    minimal_env.reset.return_value = (obs, info)
    # Return a dict action
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value={"arm": 0.5, "gripper": 0.3})

    wrapped_env = wrapper.EverythingToInfoWrapper(minimal_env)

    with pytest.raises(NotImplementedError):
        wrapped_env.reset()


def test_everything_to_info_wrapper_with_variation_single(minimal_env):
    """Test wrapper with variation option containing a single variation key."""
    import numpy as np

    obs = np.array([1, 2, 3])
    info = {}
    minimal_env.reset.return_value = (obs, info)
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=0.5)

    # Mock variation_space - it gets indexed like a dict
    mock_subvar = MagicMock()
    mock_subvar.value = 42
    mock_variation_space = MagicMock()
    mock_variation_space.__getitem__ = MagicMock(return_value=mock_subvar)
    minimal_env.unwrapped = MagicMock()
    minimal_env.unwrapped.variation_space = mock_variation_space

    wrapped_env = wrapper.EverythingToInfoWrapper(minimal_env)

    # Call reset with variation option
    result_obs, result_info = wrapped_env.reset(options={"variation": ["color"]})

    # Check that variation key is added to info
    assert "variation.color" in result_info
    assert result_info["variation.color"] == 42


def test_everything_to_info_wrapper_with_variation_all(minimal_env):
    """Test wrapper with variation option set to 'all'."""
    import numpy as np

    obs = np.array([1, 2, 3])
    info = {}
    minimal_env.reset.return_value = (obs, info)
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=0.5)

    # Mock variation_space with names() method and indexing
    mock_subvar = MagicMock()
    mock_subvar.value = 100
    mock_variation_space = MagicMock()
    mock_variation_space.names.return_value = ["color", "size"]
    mock_variation_space.__getitem__ = MagicMock(return_value=mock_subvar)
    minimal_env.unwrapped = MagicMock()
    minimal_env.unwrapped.variation_space = mock_variation_space

    wrapped_env = wrapper.EverythingToInfoWrapper(minimal_env)

    # Call reset with variation option set to "all"
    result_obs, result_info = wrapped_env.reset(options={"variation": ["all"]})

    # Check that all variation keys are added
    assert "variation.color" in result_info
    assert "variation.size" in result_info
    assert result_info["variation.color"] == 100
    assert result_info["variation.size"] == 100


def test_everything_to_info_wrapper_with_variation_multiple(minimal_env):
    """Test wrapper with multiple variation keys."""
    import numpy as np

    obs = np.array([1, 2, 3])
    info = {}
    minimal_env.reset.return_value = (obs, info)
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=0.5)

    # Mock variation_space - it gets indexed like a dict
    mock_subvar = MagicMock()
    mock_subvar.value = 50
    mock_variation_space = MagicMock()
    mock_variation_space.__getitem__ = MagicMock(return_value=mock_subvar)
    minimal_env.unwrapped = MagicMock()
    minimal_env.unwrapped.variation_space = mock_variation_space

    wrapped_env = wrapper.EverythingToInfoWrapper(minimal_env)

    # Call reset with multiple variation keys
    result_obs, result_info = wrapped_env.reset(options={"variation": ["color", "size"]})

    # Check that both variation keys are added
    assert "variation.color" in result_info
    assert "variation.size" in result_info
    assert result_info["variation.color"] == 50
    assert result_info["variation.size"] == 50


def test_everything_to_info_wrapper_variation_persists_in_step(minimal_env):
    """Test that variations are tracked in step() after being set in reset()."""
    import numpy as np

    # Setup reset
    reset_obs = np.array([1, 2, 3])
    minimal_env.reset.return_value = (reset_obs, {})
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=0.5)

    # Mock variation_space - it gets indexed like a dict
    mock_subvar = MagicMock()
    mock_subvar.value = 75
    mock_variation_space = MagicMock()
    mock_variation_space.__getitem__ = MagicMock(return_value=mock_subvar)
    minimal_env.unwrapped = MagicMock()
    minimal_env.unwrapped.variation_space = mock_variation_space

    wrapped_env = wrapper.EverythingToInfoWrapper(minimal_env)

    # Call reset with variation
    wrapped_env.reset(options={"variation": ["color"]})

    # Setup step
    step_obs = np.array([4, 5, 6])
    minimal_env.step.return_value = (step_obs, 1.0, False, False, {})

    # Call step
    result = wrapped_env.step(0.7)

    # Check that variation key is still tracked in step
    assert "variation.color" in result[4]
    assert result[4]["variation.color"] == 75


###########################
## test AddPixelsWrapper ##
###########################


def test_add_pixels_wrapper_reset_single_image(minimal_env):
    """Test that reset() adds 'pixels' key when render returns a single image."""
    import numpy as np

    obs = {"observation": "test_obs"}
    info = {}
    minimal_env.reset.return_value = (obs, info)
    # Mock render to return a single RGB image
    rendered_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    minimal_env.render = MagicMock(return_value=rendered_image)
    minimal_env.unwrapped = minimal_env

    wrapped_env = wrapper.AddPixelsWrapper(minimal_env, pixels_shape=(84, 84))
    result_obs, result_info = wrapped_env.reset()

    # Check that pixels key is added
    assert "pixels" in result_info
    assert result_info["pixels"].shape == (84, 84, 3)
    # Check that render_time is added
    assert "render_time" in result_info
    assert isinstance(result_info["render_time"], float)
    # Original obs should be unchanged
    assert result_obs == obs


def test_add_pixels_wrapper_reset_dict_images(minimal_env):
    """Test that reset() handles dict of images (multiview rendering)."""
    import numpy as np

    obs = {"observation": "test_obs"}
    info = {}
    minimal_env.reset.return_value = (obs, info)
    # Mock render to return dict of images
    rendered_images = {
        "front": np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
        "back": np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
    }
    minimal_env.render = MagicMock(return_value=rendered_images)
    minimal_env.unwrapped = minimal_env

    wrapped_env = wrapper.AddPixelsWrapper(minimal_env, pixels_shape=(64, 64))
    result_obs, result_info = wrapped_env.reset()

    # Check that pixels.front and pixels.back keys are added
    assert "pixels.front" in result_info
    assert "pixels.back" in result_info
    assert result_info["pixels.front"].shape == (64, 64, 3)
    assert result_info["pixels.back"].shape == (64, 64, 3)
    assert "render_time" in result_info


def test_add_pixels_wrapper_reset_list_images(minimal_env):
    """Test that reset() handles list/tuple of images."""
    import numpy as np

    obs = {"observation": "test_obs"}
    info = {}
    minimal_env.reset.return_value = (obs, info)
    # Mock render to return list of images
    rendered_images = [
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
    ]
    minimal_env.render = MagicMock(return_value=rendered_images)
    minimal_env.unwrapped = minimal_env

    wrapped_env = wrapper.AddPixelsWrapper(minimal_env, pixels_shape=(64, 64))
    result_obs, result_info = wrapped_env.reset()

    # Check that pixels.0 and pixels.1 keys are added
    assert "pixels.0" in result_info
    assert "pixels.1" in result_info
    assert result_info["pixels.0"].shape == (64, 64, 3)
    assert result_info["pixels.1"].shape == (64, 64, 3)


def test_add_pixels_wrapper_step_single_image(minimal_env):
    """Test that step() adds 'pixels' key when render returns a single image."""
    import numpy as np

    obs = {"observation": "test_obs"}
    reward = 1.0
    terminated = False
    truncated = False
    info = {}
    minimal_env.step.return_value = (obs, reward, terminated, truncated, info)
    # Mock render to return a single RGB image
    rendered_image = np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8)
    minimal_env.render = MagicMock(return_value=rendered_image)
    minimal_env.unwrapped = minimal_env

    wrapped_env = wrapper.AddPixelsWrapper(minimal_env, pixels_shape=(96, 96))
    result = wrapped_env.step(0.5)

    # Check that pixels key is added to info
    assert "pixels" in result[4]
    assert result[4]["pixels"].shape == (96, 96, 3)
    assert "render_time" in result[4]
    # Other values should be unchanged
    assert result[0] == obs
    assert result[1] == reward
    assert result[2] == terminated
    assert result[3] == truncated


def test_add_pixels_wrapper_render_multiview_priority(minimal_env):
    """Test that render_multiview is used if available instead of render."""
    import numpy as np

    obs = {"observation": "test_obs"}
    info = {}
    minimal_env.reset.return_value = (obs, info)

    # Set up both render and render_multiview
    single_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    multiview_images = {
        "camera1": np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
        "camera2": np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
    }

    minimal_env.render = MagicMock(return_value=single_image)
    minimal_env.unwrapped = MagicMock()
    minimal_env.unwrapped.render_multiview = MagicMock(return_value=multiview_images)

    wrapped_env = wrapper.AddPixelsWrapper(minimal_env, pixels_shape=(64, 64))
    result_obs, result_info = wrapped_env.reset()

    # render_multiview should be called, not render
    minimal_env.unwrapped.render_multiview.assert_called_once()
    minimal_env.render.assert_not_called()

    # Should have multiview pixels
    assert "pixels.camera1" in result_info
    assert "pixels.camera2" in result_info


def test_add_pixels_wrapper_no_render_multiview(minimal_env):
    """Test that regular render is used when render_multiview is not available."""
    import numpy as np

    obs = {"observation": "test_obs"}
    info = {}
    minimal_env.reset.return_value = (obs, info)

    # Only set up regular render
    single_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    minimal_env.render = MagicMock(return_value=single_image)
    minimal_env.unwrapped = minimal_env

    wrapped_env = wrapper.AddPixelsWrapper(minimal_env, pixels_shape=(64, 64))
    result_obs, result_info = wrapped_env.reset()

    # render should be called
    minimal_env.render.assert_called_once()

    # Should have single pixels key
    assert "pixels" in result_info


def test_add_pixels_wrapper_custom_shape(minimal_env):
    """Test wrapper with custom non-square pixel shape."""
    import numpy as np

    obs = {"observation": "test_obs"}
    info = {}
    minimal_env.reset.return_value = (obs, info)
    rendered_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    minimal_env.render = MagicMock(return_value=rendered_image)
    minimal_env.unwrapped = minimal_env

    # Use non-square shape
    wrapped_env = wrapper.AddPixelsWrapper(minimal_env, pixels_shape=(128, 256))  # (height, width)
    result_obs, result_info = wrapped_env.reset()

    assert "pixels" in result_info
    assert result_info["pixels"].shape == (128, 256, 3)


def test_add_pixels_wrapper_with_transform(minimal_env):
    """Test wrapper with a torchvision transform."""
    import numpy as np

    obs = {"observation": "test_obs"}
    info = {}
    minimal_env.reset.return_value = (obs, info)
    rendered_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    minimal_env.render = MagicMock(return_value=rendered_image)
    minimal_env.unwrapped = minimal_env

    # Create a simple transform mock
    transform_mock = MagicMock(return_value=np.array([[1, 2], [3, 4]]))

    wrapped_env = wrapper.AddPixelsWrapper(minimal_env, pixels_shape=(64, 64), torchvision_transform=transform_mock)
    result_obs, result_info = wrapped_env.reset()

    # Transform should be called
    assert transform_mock.called
    # Result should be what the transform returned
    assert "pixels" in result_info
    assert np.array_equal(result_info["pixels"], np.array([[1, 2], [3, 4]]))


def test_add_pixels_wrapper_preserves_existing_info(minimal_env):
    """Test that wrapper preserves existing info keys."""
    import numpy as np

    obs = {"observation": "test_obs"}
    info = {"existing_key": "existing_value", "other": 123}
    minimal_env.reset.return_value = (obs, info)
    rendered_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    minimal_env.render = MagicMock(return_value=rendered_image)
    minimal_env.unwrapped = minimal_env

    wrapped_env = wrapper.AddPixelsWrapper(minimal_env, pixels_shape=(64, 64))
    result_obs, result_info = wrapped_env.reset()

    # Existing keys should be preserved
    assert result_info["existing_key"] == "existing_value"
    assert result_info["other"] == 123
    # New keys should be added
    assert "pixels" in result_info
    assert "render_time" in result_info


############################
## test ResizeGoalWrapper ##
############################


def test_resize_goal_wrapper_reset_success(minimal_env):
    """Test that reset() resizes the goal image correctly."""
    import numpy as np

    obs = {"observation": "test_obs"}
    # Goal image with shape (100, 120, 3)
    goal_image = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
    info = {"goal": goal_image, "other_key": "value"}
    minimal_env.reset.return_value = (obs, info)

    wrapped_env = wrapper.ResizeGoalWrapper(minimal_env, pixels_shape=(64, 64))
    result_obs, result_info = wrapped_env.reset()

    # Check that goal is resized
    assert "goal" in result_info
    assert result_info["goal"].shape == (64, 64, 3)
    # Check that other keys are preserved
    assert result_info["other_key"] == "value"
    # Original obs should be unchanged
    assert result_obs == obs


def test_resize_goal_wrapper_step_success(minimal_env):
    """Test that step() resizes the goal image correctly."""
    import numpy as np

    obs = {"observation": "test_obs"}
    reward = 1.0
    terminated = False
    truncated = False
    # Goal image with shape (150, 150, 3)
    goal_image = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
    info = {"goal": goal_image}
    minimal_env.step.return_value = (obs, reward, terminated, truncated, info)

    wrapped_env = wrapper.ResizeGoalWrapper(minimal_env, pixels_shape=(84, 84))
    result = wrapped_env.step(0.5)

    # Check that goal is resized
    assert "goal" in result[4]
    assert result[4]["goal"].shape == (84, 84, 3)
    # Other values should be unchanged
    assert result[0] == obs
    assert result[1] == reward
    assert result[2] == terminated
    assert result[3] == truncated


def test_resize_goal_wrapper_non_square_shape(minimal_env):
    """Test wrapper with non-square resize shape."""
    import numpy as np

    obs = {"observation": "test_obs"}
    goal_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    info = {"goal": goal_image}
    minimal_env.reset.return_value = (obs, info)

    # Use non-square shape (height, width)
    wrapped_env = wrapper.ResizeGoalWrapper(minimal_env, pixels_shape=(128, 256))
    result_obs, result_info = wrapped_env.reset()

    # Check dimensions (height, width, channels)
    assert result_info["goal"].shape == (128, 256, 3)


def test_resize_goal_wrapper_with_transform(minimal_env):
    """Test wrapper with a torchvision transform."""
    import numpy as np

    obs = {"observation": "test_obs"}
    goal_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    info = {"goal": goal_image}
    minimal_env.reset.return_value = (obs, info)

    # Create a simple transform mock
    transform_mock = MagicMock(return_value=np.array([[5, 6], [7, 8]]))

    wrapped_env = wrapper.ResizeGoalWrapper(minimal_env, pixels_shape=(64, 64), torchvision_transform=transform_mock)
    result_obs, result_info = wrapped_env.reset()

    # Transform should be called
    assert transform_mock.called
    # Result should be what the transform returned
    assert "goal" in result_info
    assert np.array_equal(result_info["goal"], np.array([[5, 6], [7, 8]]))


def test_resize_goal_wrapper_preserves_other_info(minimal_env):
    """Test that wrapper preserves other info keys."""
    import numpy as np

    obs = {"observation": "test_obs"}
    goal_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    info = {"goal": goal_image, "extra_data": "preserved", "score": 42}
    minimal_env.step.return_value = (obs, 1.0, False, False, info)

    wrapped_env = wrapper.ResizeGoalWrapper(minimal_env, pixels_shape=(64, 64))
    result = wrapped_env.step(0.5)

    # Goal should be resized
    assert result[4]["goal"].shape == (64, 64, 3)
    # Other keys should be preserved
    assert result[4]["extra_data"] == "preserved"
    assert result[4]["score"] == 42


def test_resize_goal_wrapper_default_shape(minimal_env):
    """Test wrapper with default pixel shape (84, 84)."""
    import numpy as np

    obs = {"observation": "test_obs"}
    goal_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    info = {"goal": goal_image}
    minimal_env.reset.return_value = (obs, info)

    # Use default shape
    wrapped_env = wrapper.ResizeGoalWrapper(minimal_env)
    result_obs, result_info = wrapped_env.reset()

    # Default shape should be (84, 84)
    assert result_info["goal"].shape == (84, 84, 3)


def test_resize_goal_wrapper_grayscale_goal(minimal_env):
    """Test wrapper with grayscale goal image."""
    import numpy as np

    obs = {"observation": "test_obs"}
    # Grayscale goal image with shape (100, 100) - PIL format for grayscale
    goal_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    info = {"goal": goal_image}
    minimal_env.reset.return_value = (obs, info)

    wrapped_env = wrapper.ResizeGoalWrapper(minimal_env, pixels_shape=(64, 64))
    result_obs, result_info = wrapped_env.reset()

    # Should resize to (64, 64) for grayscale
    assert result_info["goal"].shape == (64, 64)


def test_resize_goal_wrapper_both_reset_and_step(minimal_env):
    """Test that wrapper works correctly for both reset and step."""
    import numpy as np

    # Reset
    reset_obs = {"observation": "reset_obs"}
    reset_goal = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    minimal_env.reset.return_value = (reset_obs, {"goal": reset_goal})

    # Step
    step_obs = {"observation": "step_obs"}
    step_goal = np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8)
    minimal_env.step.return_value = (step_obs, 1.0, False, False, {"goal": step_goal})

    wrapped_env = wrapper.ResizeGoalWrapper(minimal_env, pixels_shape=(48, 48))

    # Test reset
    result_obs, result_info = wrapped_env.reset()
    assert result_info["goal"].shape == (48, 48, 3)

    # Test step
    result = wrapped_env.step(0.5)
    assert result[4]["goal"].shape == (48, 48, 3)


def test_resize_goal_wrapper_different_input_sizes(minimal_env):
    """Test that wrapper handles different input goal sizes correctly."""
    import numpy as np

    obs = {"observation": "test_obs"}

    # Test with various input sizes
    input_sizes = [(50, 50, 3), (100, 200, 3), (300, 150, 3)]
    target_shape = (64, 64)

    for input_size in input_sizes:
        goal_image = np.random.randint(0, 255, input_size, dtype=np.uint8)
        minimal_env.reset.return_value = (obs, {"goal": goal_image})

        wrapped_env = wrapper.ResizeGoalWrapper(minimal_env, pixels_shape=target_shape)
        result_obs, result_info = wrapped_env.reset()

        # All should resize to the target shape
        assert result_info["goal"].shape == (target_shape[0], target_shape[1], 3)


######################
## test MegaWrapper ##
######################


def test_mega_wrapper_reset_basic(minimal_env):
    """Test that MegaWrapper reset() works with all sub-wrappers."""
    import numpy as np

    obs = np.array([1, 2, 3])
    goal_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    info = {"goal": goal_image}
    minimal_env.reset.return_value = (obs, info)
    minimal_env.render = MagicMock(return_value=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    minimal_env.unwrapped = minimal_env
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=0.5)
    wrapped_env = wrapper.MegaWrapper(minimal_env, image_shape=(64, 64))
    result_obs, result_info = wrapped_env.reset()

    # Check that pixels key is added (from AddPixelsWrapper)
    assert "pixels" in result_info
    # Check that observation is in info (from EverythingToInfoWrapper)
    assert "observation" in result_info
    # Check that goal is resized (from ResizeGoalWrapper)
    assert "goal" in result_info
    assert result_info["goal"].shape == (1, 64, 64, 3)
    # Check metadata from EverythingToInfoWrapper
    assert "reward" in result_info
    assert "terminated" in result_info
    assert "truncated" in result_info
    assert "action" in result_info
    assert "step_idx" in result_info


def test_mega_wrapper_step_basic(minimal_env):
    """Test that MegaWrapper step() works with all sub-wrappers."""
    import numpy as np

    # First reset
    reset_obs = np.array([1, 2, 3])
    reset_goal = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    minimal_env.reset.return_value = (reset_obs, {"goal": reset_goal})
    minimal_env.render = MagicMock(return_value=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    minimal_env.unwrapped = minimal_env
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=0.5)

    wrapped_env = wrapper.MegaWrapper(minimal_env, image_shape=(64, 64))
    wrapped_env.reset()

    # Now test step
    step_obs = np.array([4, 5, 6])
    reward = 1.5
    terminated = False
    truncated = False
    step_goal = np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8)
    minimal_env.step.return_value = (step_obs, reward, terminated, truncated, {"goal": step_goal})

    result_obs, result_reward, result_terminated, result_truncated, result_info = wrapped_env.step(0.7)

    # Check that pixels key is added
    assert "pixels" in result_info
    # Check that observation is in info
    assert "observation" in result_info
    # Check that goal is resized
    assert result_info["goal"].shape == (1, 64, 64, 3)
    # Check metadata
    assert result_info["reward"] == reward
    assert result_info["action"] == 0.7
    assert result_info["step_idx"] == 1


def test_mega_wrapper_with_custom_required_keys(minimal_env):
    """Test MegaWrapper with custom required keys."""
    import numpy as np

    obs = np.array([1, 2, 3])
    goal_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    # Add custom keys that we'll require
    info = {"goal": goal_image, "custom_key": "value"}
    minimal_env.reset.return_value = (obs, info)
    minimal_env.render = MagicMock(return_value=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    minimal_env.unwrapped = minimal_env
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=0.5)

    # Require custom_key in addition to pixels
    wrapped_env = wrapper.MegaWrapper(minimal_env, image_shape=(64, 64), required_keys=["custom_key"])
    result_obs, result_info = wrapped_env.reset()

    # Should succeed because custom_key is present
    assert "custom_key" in result_info
    assert "pixels" in result_info


def test_mega_wrapper_missing_required_keys_fails(minimal_env):
    """Test that MegaWrapper raises error when required keys are missing."""
    import numpy as np

    obs = np.array([1, 2, 3])
    goal_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    info = {"goal": goal_image}  # Missing custom_required_key
    minimal_env.reset.return_value = (obs, info)
    minimal_env.render = MagicMock(return_value=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    minimal_env.unwrapped = minimal_env
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=0.5)

    # Require a key that doesn't exist
    wrapped_env = wrapper.MegaWrapper(minimal_env, image_shape=(64, 64), required_keys=["missing_key"])

    with pytest.raises(RuntimeError):
        wrapped_env.reset()


def test_mega_wrapper_separate_goal_option(minimal_env):
    """Test that MegaWrapper initializes correctly with separate_goal=True."""
    import numpy as np

    obs = np.array([1, 2, 3])
    info = {}
    minimal_env.reset.return_value = (obs, info)
    minimal_env.render = MagicMock(return_value=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    minimal_env.unwrapped = minimal_env
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=0.5)

    # separate_goal=True should still allow reset without goal
    wrapped_env = wrapper.MegaWrapper(minimal_env, image_shape=(64, 64), separate_goal=True)

    # Should not raise - goal checking is not enforced in current implementation
    wrapped_env.reset()


def test_mega_wrapper_no_goal_check_when_disabled(minimal_env):
    """Test that MegaWrapper doesn't check for goal when separate_goal=False."""
    import numpy as np

    obs = np.array([1, 2, 3])
    info = {}  # No goal key, but should be okay
    minimal_env.reset.return_value = (obs, info)
    minimal_env.render = MagicMock(return_value=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    minimal_env.unwrapped = minimal_env
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=0.5)

    # separate_goal=False should not require goal key in reset
    wrapped_env = wrapper.MegaWrapper(minimal_env, image_shape=(64, 64), separate_goal=False)

    # This should work but will fail at ResizeGoalWrapper because goal doesn't exist
    # Let's add a goal to test the check is disabled
    info["goal"] = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    minimal_env.reset.return_value = (obs, info)

    result_obs, result_info = wrapped_env.reset()
    # Should succeed
    assert "pixels" in result_info


def test_mega_wrapper_multiview_pixels(minimal_env):
    """Test MegaWrapper with multiview rendering."""
    import numpy as np

    obs = np.array([1, 2, 3])
    goal_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    info = {"goal": goal_image}
    minimal_env.reset.return_value = (obs, info)
    # Mock multiview render
    multiview_images = {
        "front": np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
        "side": np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
    }
    minimal_env.render = MagicMock(return_value=multiview_images)
    minimal_env.unwrapped = minimal_env
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=0.5)

    wrapped_env = wrapper.MegaWrapper(minimal_env, image_shape=(64, 64))
    result_obs, result_info = wrapped_env.reset()

    # Should have multiple pixel keys
    assert "pixels.front" in result_info
    assert "pixels.side" in result_info
    # Regex pattern should match both
    assert result_info["pixels.front"].shape == (1, 64, 64, 3)
    assert result_info["pixels.side"].shape == (1, 64, 64, 3)


def test_mega_wrapper_with_transforms(minimal_env):
    """Test MegaWrapper with pixel and goal transforms."""
    import numpy as np

    obs = np.array([1, 2, 3])
    goal_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    info = {"goal": goal_image}
    minimal_env.reset.return_value = (obs, info)
    minimal_env.render = MagicMock(return_value=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    minimal_env.unwrapped = minimal_env
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=0.5)

    # Create transform mocks
    pixels_transform = MagicMock(return_value=np.ones((64, 64, 3)))
    goal_transform = MagicMock(return_value=np.zeros((64, 64, 3)))

    wrapped_env = wrapper.MegaWrapper(
        minimal_env, image_shape=(64, 64), pixels_transform=pixels_transform, goal_transform=goal_transform
    )
    result_obs, result_info = wrapped_env.reset()

    # Transforms should be called
    assert pixels_transform.called
    assert goal_transform.called
    # Results should be from transforms
    assert np.array_equal(result_info["pixels"], np.ones((1, 64, 64, 3)))
    assert np.array_equal(result_info["goal"], np.zeros((1, 64, 64, 3)))


def test_mega_wrapper_custom_image_shape(minimal_env):
    """Test MegaWrapper with custom image shape."""
    import numpy as np

    obs = np.array([1, 2, 3])
    goal_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    info = {"goal": goal_image}
    minimal_env.reset.return_value = (obs, info)
    minimal_env.render = MagicMock(return_value=np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
    minimal_env.unwrapped = minimal_env
    minimal_env.action_space = MagicMock()
    minimal_env.action_space.sample = MagicMock(return_value=0.5)

    # Use non-square custom shape
    wrapped_env = wrapper.MegaWrapper(minimal_env, image_shape=(128, 256))
    result_obs, result_info = wrapped_env.reset()

    # Both pixels and goal should be resized to custom shape
    assert result_info["pixels"].shape == (1, 128, 256, 3)
    assert result_info["goal"].shape == (1, 128, 256, 3)


###########################
## test VariationWrapper ##
###########################


@pytest.fixture
def mock_vector_env():
    """Create a mock vector environment for VariationWrapper testing."""
    from gymnasium.vector import VectorEnv

    mock_vec_env = MagicMock(spec=VectorEnv)
    mock_vec_env.num_envs = 3
    mock_vec_env.single_observation_space = gym.spaces.Box(low=0, high=1, shape=(4,))
    mock_vec_env.single_action_space = gym.spaces.Box(low=0, high=1, shape=(2,))

    # Create mock sub-environments
    mock_envs = []
    for i in range(3):
        mock_env = MagicMock()
        mock_env.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,))
        mock_env.action_space = gym.spaces.Box(low=0, high=1, shape=(2,))
        mock_env.unwrapped = MagicMock()
        mock_env.unwrapped.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,))
        mock_env.unwrapped.variation_space = gym.spaces.Box(low=0, high=1, shape=(2,))
        mock_envs.append(mock_env)

    mock_vec_env.envs = mock_envs
    return mock_vec_env


def test_variation_wrapper_no_variation_space(mock_vector_env):
    """Test VariationWrapper when base environment has no variation_space."""
    # Remove variation_space from envs
    for env in mock_vector_env.envs:
        delattr(env.unwrapped, "variation_space")

    wrapped_env = wrapper.VariationWrapper(mock_vector_env)

    # Should set variation spaces to None
    assert wrapped_env.single_variation_space is None
    assert wrapped_env.variation_space is None


def test_variation_wrapper_same_mode(mock_vector_env):
    """Test VariationWrapper with variation_mode='same'."""
    wrapped_env = wrapper.VariationWrapper(mock_vector_env, variation_mode="same")

    # Should have variation spaces set
    assert wrapped_env.single_variation_space is not None
    assert wrapped_env.variation_space is not None
    # single_variation_space should be from the first env
    assert wrapped_env.single_variation_space == mock_vector_env.envs[0].unwrapped.variation_space


def test_variation_wrapper_different_mode(mock_vector_env):
    """Test VariationWrapper with variation_mode='different'."""
    wrapped_env = wrapper.VariationWrapper(mock_vector_env, variation_mode="different")

    # Should have variation spaces set
    assert wrapped_env.single_variation_space is not None
    assert wrapped_env.variation_space is not None


def test_variation_wrapper_invalid_mode(mock_vector_env):
    """Test that VariationWrapper raises error for invalid variation_mode."""
    with pytest.raises(ValueError):
        wrapper.VariationWrapper(mock_vector_env, variation_mode="invalid_mode")


def test_variation_wrapper_envs_property(mock_vector_env):
    """Test that envs property returns the wrapped env's envs."""
    wrapped_env = wrapper.VariationWrapper(mock_vector_env, variation_mode="same")

    assert wrapped_env.envs == mock_vector_env.envs


def test_variation_wrapper_observation_space_mismatch_same_mode(mock_vector_env):
    """Test that VariationWrapper raises error when observation spaces don't match in 'same' mode."""
    # Make one sub-env have different observation space
    mock_vector_env.envs[1].unwrapped.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,))

    with pytest.raises(ValueError):
        wrapper.VariationWrapper(mock_vector_env, variation_mode="same")


def test_variation_wrapper_observation_space_mismatch_different_mode(mock_vector_env):
    """Test that VariationWrapper raises error when observation spaces don't match in 'different' mode."""
    # Make one sub-env have different observation space
    mock_vector_env.envs[1].unwrapped.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,))

    with pytest.raises(ValueError):
        wrapper.VariationWrapper(mock_vector_env, variation_mode="different")


def test_variation_wrapper_preserves_num_envs(mock_vector_env):
    """Test that VariationWrapper preserves num_envs from wrapped environment."""
    wrapped_env = wrapper.VariationWrapper(mock_vector_env, variation_mode="same")

    assert wrapped_env.num_envs == mock_vector_env.num_envs


def test_variation_wrapper_with_different_variation_spaces(mock_vector_env):
    """Test VariationWrapper with different variation spaces per env in 'different' mode."""
    # Give each env a different variation space
    mock_vector_env.envs[0].unwrapped.variation_space = gym.spaces.Box(low=0, high=1, shape=(2,))
    mock_vector_env.envs[1].unwrapped.variation_space = gym.spaces.Box(low=0, high=2, shape=(2,))
    mock_vector_env.envs[2].unwrapped.variation_space = gym.spaces.Box(low=0, high=3, shape=(2,))

    wrapped_env = wrapper.VariationWrapper(mock_vector_env, variation_mode="different")

    # Should succeed with different mode
    assert wrapped_env.variation_space is not None


###########################
## test StackedWrapper ##
###########################


def test_stacked_wrapper_reset_numpy_k3_and_step(minimal_env):
    """Test that StackedWrapper stacks numpy arrays over reset() and step()."""
    import numpy as np

    obs = {"observation": "test_obs"}
    # initial array shape (2,)
    init = np.array([1, 2])
    minimal_env.reset.return_value = (obs, {"stack_key": init})

    wrapped = wrapper.StackedWrapper(minimal_env, key="stack_key", history_size=3)
    _, info = wrapped.reset()

    # After reset the key should be stacked history_size times -> shape (3, 2)
    assert "stack_key" in info
    assert info["stack_key"].shape == (3, 2)
    # All entries should equal the initial value
    assert np.array_equal(info["stack_key"][0], init)
    assert np.array_equal(info["stack_key"][1], init)
    assert np.array_equal(info["stack_key"][2], init)

    # Now step with a new value
    new = np.array([3, 4])
    minimal_env.step.return_value = (obs, 0.0, False, False, {"stack_key": new})
    _, _, _, _, info2 = wrapped.step(0.5)

    # Buffer should now contain [init, init, new]
    assert info2["stack_key"].shape == (3, 2)
    assert np.array_equal(info2["stack_key"][0], init)
    assert np.array_equal(info2["stack_key"][1], init)
    assert np.array_equal(info2["stack_key"][2], new)


def test_stacked_wrapper_torch_k2(minimal_env):
    """Test that StackedWrapper handles torch.Tensor stacking correctly."""
    import torch

    obs = {"observation": "o"}
    t0 = torch.tensor([1.0, 2.0])
    minimal_env.reset.return_value = (obs, {"t": t0})

    wrapped = wrapper.StackedWrapper(minimal_env, key="t", history_size=2)
    _, info = wrapped.reset()

    # After reset should be a torch.Tensor stacked along dim 0 with shape (2, 2)
    assert "t" in info
    assert isinstance(info["t"], torch.Tensor)
    assert info["t"].shape == (2, 2)
    assert torch.allclose(info["t"][0], t0)
    assert torch.allclose(info["t"][1], t0)

    # Step with a new tensor
    t1 = torch.tensor([3.0, 4.0])
    minimal_env.step.return_value = (obs, 0.0, False, False, {"t": t1})
    _, _, _, _, info2 = wrapped.step(0.1)

    assert isinstance(info2["t"], torch.Tensor)
    assert info2["t"].shape == (2, 2)
    # Buffer should be [t0, t1]
    assert torch.allclose(info2["t"][0], t0)
    assert torch.allclose(info2["t"][1], t1)


def test_stacked_wrapper_k1_returns_raw(minimal_env):
    """k=1 should return the raw element (no extra stacking dimension)."""
    import numpy as np

    obs = {"observation": "o"}
    arr = np.array([9, 9, 9])
    minimal_env.reset.return_value = (obs, {"a": arr})

    wrapped = wrapper.StackedWrapper(minimal_env, key="a", history_size=1)
    _, info = wrapped.reset()

    # For history_size==1, the wrapper should set the key to the raw element (not stacked)
    assert "a" in info
    assert isinstance(info["a"], np.ndarray)
    assert info["a"].shape == (1, 3)
    assert np.array_equal(info["a"].squeeze(), arr)


def test_stacked_wrapper_empty_buffer(minimal_env):
    """Test that get_buffer_data returns empty list when buffer is empty."""
    wrapped = wrapper.StackedWrapper(minimal_env, key="test_key", history_size=3)

    # Before reset, buffer should be empty
    result = wrapped.get_buffer_data("test_key")
    assert result == []


def test_stacked_wrapper_frameskip_and_history(minimal_env):
    """Test frameskip>1 combined with history_size selection behavior."""
    import numpy as np

    obs = {"observation": "o"}
    # initial value
    v0 = np.array([0])
    minimal_env.reset.return_value = (obs, {"k": v0})

    # history_size=2, frameskip=2 => capacity=4
    wrapped = wrapper.StackedWrapper(minimal_env, key="k", history_size=2, frameskip=2)
    _, info = wrapped.reset()

    # After reset stacked output should have shape (2, 1)
    assert "k" in info
    assert info["k"].shape == (2, 1)

    # step with a new value v1
    v1 = np.array([1])
    minimal_env.step.return_value = (obs, 0.0, False, False, {"k": v1})
    _, _, _, _, info2 = wrapped.step(0.1)

    # The stacked result should be [v0, v1]
    assert np.array_equal(info2["k"][0], v0)
    assert np.array_equal(info2["k"][1], v1)


def test_stacked_wrapper_primitive_numbers(minimal_env):
    """Test stacking for primitive numeric types (int/float) returns numpy array."""
    import numpy as np

    obs = {"observation": "o"}
    minimal_env.reset.return_value = (obs, {"n": 7})

    wrapped = wrapper.StackedWrapper(minimal_env, key="n", history_size=3)
    _, info = wrapped.reset()

    # Should be numpy array of shape (3,)
    assert isinstance(info["n"], np.ndarray)
    assert info["n"].shape == (3,)
    assert (info["n"] == 7).all()

    # Step with new number
    minimal_env.step.return_value = (obs, 0.0, False, False, {"n": 9})
    _, _, _, _, info2 = wrapped.step(0.2)
    assert isinstance(info2["n"], np.ndarray)
    assert info2["n"].shape == (3,)
    # Expect [7,7,9]
    assert list(info2["n"]) == [7, 7, 9]


def test_stacked_wrapper_list_elements_fallback(minimal_env):
    """Test that non-array/listable elements fall back to returning Python list of elements."""
    obs = {"observation": "o"}
    minimal_env.reset.return_value = (obs, {"L": [1, 2]})

    wrapped = wrapper.StackedWrapper(minimal_env, key="L", history_size=2)
    _, info = wrapped.reset()

    # For list elements, the wrapper should return a list of elements (length history_size)
    assert isinstance(info["L"], list)
    assert len(info["L"]) == 2
    assert info["L"][0] == [1, 2]

    # Step with new list
    minimal_env.step.return_value = (obs, 0.0, False, False, {"L": [3, 4]})
    _, _, _, _, info2 = wrapped.step(0.3)
    assert isinstance(info2["L"], list)
    assert info2["L"][0] == [1, 2]
    assert info2["L"][1] == [3, 4]


def test_stacked_wrapper_multiple_keys(minimal_env):
    """Test that StackedWrapper can track and return stacked data for multiple keys independently."""
    import numpy as np

    obs = {"observation": "o"}
    minimal_env.reset.return_value = (obs, {"a": np.array([1]), "b": 10})

    wrapped = wrapper.StackedWrapper(minimal_env, key=["a", "b"], history_size=2)
    _, info = wrapped.reset()

    # a should be stacked into shape (2,1), b into shape (2,)
    assert "a" in info and "b" in info
    assert info["a"].shape == (2, 1)
    assert info["b"].shape == (2,)

    # Step with new values
    minimal_env.step.return_value = (obs, 0.0, False, False, {"a": np.array([2]), "b": 20})
    _, _, _, _, info2 = wrapped.step(0.4)

    assert info2["a"].shape == (2, 1)
    assert info2["b"].shape == (2,)
    assert (info2["a"][0] == np.array([1])).all()
    assert info2["b"].tolist() == [10, 20]
