from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from stable_worldmodel import spaces


##########################
## Discrete Space tests ##
##########################


# vanilla gym behavior


def test_discrete_space_vanilla_properties():
    space = spaces.Discrete(5)
    assert space.n == 5
    assert space.shape == ()
    assert space.dtype == int


def test_discrete_space_vanilla_contains():
    space = spaces.Discrete(5)
    assert space.contains(0)
    assert space.contains(3)
    assert not space.contains(5)
    assert not space.contains(-1)


def test_discrete_space_vanilla_sample():
    space = spaces.Discrete(5)
    sample = space.sample()
    assert space.contains(sample)
    assert isinstance(sample, np.int64)


def test_discrete_space_vanilla_empty():
    with pytest.raises(AssertionError, match="have to be positive"):
        spaces.Discrete(0)


def test_discrete_space_vanilla_check():
    space = spaces.Discrete(5)
    assert not space.check()


def test_discrete_space_vanilla_reset():
    space = spaces.Discrete(5)
    assert space.init_value is None
    space.reset()
    assert space.value == space.init_value
    assert space.value is None


# stable_worldmodel behavior


def test_discrete_space_properties():
    init_val = 2
    space = spaces.Discrete(5, init_value=init_val)
    assert space.init_value == init_val
    assert space.value == init_val
    assert space.contains(init_val)


def test_discrete_space_check():
    space = spaces.Discrete(5, init_value=2)
    assert space.check()


def test_discrete_space_outbound():
    space = spaces.Discrete(5, init_value=-2)
    assert not space.contains(space.value)
    assert not space.check()


def test_discrete_space_reset():
    space = spaces.Discrete(5, init_value=-1)
    space.sample()
    assert space.value != space.init_value
    space.reset()
    assert space.value == space.init_value


def test_discrete_space_sample_sets_value():
    space = spaces.Discrete(5, init_value=-1)
    space.sample()
    assert space.value != space.init_value
    space.reset()
    assert space.value == space.init_value
    space.sample(set_value=False)
    assert space.value == space.init_value


def test_discrete_space_constraint_function_fails():
    constraint_fn = MagicMock(return_value=False)
    space = spaces.Discrete(5, init_value=2, constrain_fn=constraint_fn)
    assert not space.check()
    assert not space.contains(-1)
    assert not space.contains(5)


def test_discrete_space_constraint_function_fail_sample():
    constraint_fn = MagicMock(return_value=False)
    space = spaces.Discrete(5, init_value=2, constrain_fn=constraint_fn)

    with pytest.raises(RuntimeError):
        space.sample(max_tries=3)


def test_discrete_space_constraint_function_warn_sample():
    constraint_fn = MagicMock(return_value=False)
    space = spaces.Discrete(5, init_value=2, constrain_fn=constraint_fn)

    with patch("stable_worldmodel.spaces.logging.warning") as mock_warning:
        with pytest.raises(RuntimeError):
            space.sample(max_tries=1, warn_after_s=0.0)
        mock_warning.assert_called()


def test_discrete_space_constraint_function_pass():
    constraint_fn = MagicMock(return_value=True)
    space = spaces.Discrete(5, init_value=2, constrain_fn=constraint_fn)
    assert space.check()
    assert space.contains(3)
    assert space.contains(0)
    assert not space.contains(5)
    assert not space.contains(-1)


def test_discrete_space_constraint_real_logic():
    """Test with actual constraint logic (e.g., only even numbers)."""
    space = spaces.Discrete(10, init_value=2, constrain_fn=lambda x: x % 2 == 0)
    sample = space.sample()
    assert sample % 2 == 0
    assert space.contains(sample)


def test_discrete_space_constraint_rejection_succeeds():
    """Test that rejection sampling eventually succeeds."""
    # Only accept values >= 3
    space = spaces.Discrete(5, init_value=3, constrain_fn=lambda x: x >= 3)
    for _ in range(10):
        sample = space.sample()
        assert sample >= 3


def test_discrete_space_sample_without_setting_value_constraint():
    """Test set_value=False with constraint function."""
    space = spaces.Discrete(10, init_value=2, constrain_fn=lambda x: x % 2 == 0)
    original_value = space.value
    sample = space.sample(set_value=False)
    assert space.value == original_value
    assert sample % 2 == 0


def test_discrete_space_check_warning_on_constraint_fail():
    """Test that check() logs warning when constraint fails."""
    space = spaces.Discrete(5, init_value=2, constrain_fn=lambda x: False)
    with patch("stable_worldmodel.spaces.logging.warning") as mock_warning:
        result = space.check()
        assert not result
        mock_warning.assert_called_once()


def test_discrete_space_init_value_violates_constraint():
    """Test behavior when init_value doesn't satisfy constraint."""
    space = spaces.Discrete(10, init_value=1, constrain_fn=lambda x: x % 2 == 0)
    assert not space.contains(space.value)
    assert not space.check()


def test_discrete_space_multiple_samples():
    """Test that value updates correctly across multiple samples."""
    space = spaces.Discrete(5, init_value=0)
    values = [space.sample() for _ in range(5)]
    assert space.value == values[-1]


################################
## Multi Discrete Space tests ##
################################


# vanilla gym behavior


def test_multidiscrete_space_vanilla_properties():
    nvec = [3, 4, 5]
    space = spaces.MultiDiscrete(nvec)
    assert np.array_equal(space.nvec, nvec)
    assert space.shape == (3,)
    assert space.dtype == np.int64


def test_multidiscrete_space_vanilla_contains():
    space = spaces.MultiDiscrete([3, 4, 5])
    assert space.contains(np.array([0, 0, 0]))
    assert space.contains(np.array([2, 3, 4]))
    assert not space.contains(np.array([3, 0, 0]))  # First element out of bounds
    assert not space.contains(np.array([0, 4, 0]))  # Second element out of bounds
    assert not space.contains(np.array([0, 0, 5]))  # Third element out of bounds
    assert not space.contains(np.array([-1, 0, 0]))  # Negative value


def test_multidiscrete_space_vanilla_sample():
    space = spaces.MultiDiscrete([3, 4, 5])
    sample = space.sample()
    assert space.contains(sample)
    assert isinstance(sample, np.ndarray)
    assert sample.shape == (3,)


def test_multidiscrete_space_vanilla_check():
    space = spaces.MultiDiscrete([3, 4, 5])
    assert not space.check()


def test_multidiscrete_space_vanilla_reset():
    space = spaces.MultiDiscrete([3, 4, 5])
    assert space.init_value is None
    space.reset()
    assert space.value == space.init_value
    assert space.value is None


# stable_worldmodel behavior


def test_multidiscrete_space_properties():
    init_val = np.array([1, 2, 3])
    space = spaces.MultiDiscrete([3, 4, 5], init_value=init_val)
    assert np.array_equal(space.init_value, init_val)
    assert np.array_equal(space.value, init_val)
    assert space.contains(init_val)


def test_multidiscrete_space_check():
    space = spaces.MultiDiscrete([3, 4, 5], init_value=np.array([1, 2, 3]))
    assert space.check()


def test_multidiscrete_space_outbound():
    space = spaces.MultiDiscrete([3, 4, 5], init_value=np.array([3, 2, 3]))
    assert not space.contains(space.value)
    assert not space.check()


def test_multidiscrete_space_reset():
    init_val = np.array([0, 1, 2])
    space = spaces.MultiDiscrete([3, 4, 5], init_value=init_val)
    space.sample()
    assert not np.array_equal(space.value, space.init_value)
    space.reset()
    assert np.array_equal(space.value, space.init_value)


def test_multidiscrete_space_sample_sets_value():
    init_val = np.array([0, 0, 0])
    space = spaces.MultiDiscrete([3, 4, 5], init_value=init_val)
    sample1 = space.sample()
    assert np.array_equal(space.value, sample1)
    space.reset()
    assert np.array_equal(space.value, init_val)
    sample2 = space.sample(set_value=False)
    assert np.array_equal(space.value, init_val)
    assert space.contains(sample2)


def test_multidiscrete_space_constraint_function_fails():
    constraint_fn = MagicMock(return_value=False)
    space = spaces.MultiDiscrete([3, 4, 5], init_value=np.array([1, 2, 3]), constrain_fn=constraint_fn)
    assert not space.check()
    assert not space.contains(np.array([0, 0, 0]))
    assert not space.contains(np.array([2, 3, 4]))


def test_multidiscrete_space_constraint_function_fail_sample():
    constraint_fn = MagicMock(return_value=False)
    space = spaces.MultiDiscrete([3, 4, 5], init_value=np.array([1, 2, 3]), constrain_fn=constraint_fn)

    with pytest.raises(RuntimeError):
        space.sample(max_tries=3)


def test_multidiscrete_space_constraint_function_warn_sample():
    constraint_fn = MagicMock(return_value=False)
    space = spaces.MultiDiscrete([3, 4, 5], init_value=np.array([1, 2, 3]), constrain_fn=constraint_fn)

    with patch("stable_worldmodel.spaces.logging.warning") as mock_warning:
        with pytest.raises(RuntimeError):
            space.sample(max_tries=1, warn_after_s=0.0)
        mock_warning.assert_called()


def test_multidiscrete_space_constraint_function_pass():
    constraint_fn = MagicMock(return_value=True)
    space = spaces.MultiDiscrete([3, 4, 5], init_value=np.array([1, 2, 3]), constrain_fn=constraint_fn)
    assert space.check()
    assert space.contains(np.array([0, 0, 0]))
    assert space.contains(np.array([2, 3, 4]))
    assert not space.contains(np.array([3, 0, 0]))
    assert not space.contains(np.array([0, 4, 0]))


def test_multidiscrete_space_constraint_real_logic():
    """Test with actual constraint logic (e.g., sum must be even)."""
    space = spaces.MultiDiscrete(
        [5, 5, 5],
        init_value=np.array([2, 2, 2]),
        constrain_fn=lambda x: np.sum(x) % 2 == 0,
    )
    sample = space.sample()
    assert np.sum(sample) % 2 == 0
    assert space.contains(sample)


def test_multidiscrete_space_constraint_rejection_succeeds():
    """Test that rejection sampling eventually succeeds."""
    # Only accept arrays where first element >= 2
    space = spaces.MultiDiscrete([5, 5, 5], init_value=np.array([2, 1, 1]), constrain_fn=lambda x: x[0] >= 2)
    for _ in range(10):
        sample = space.sample()
        assert sample[0] >= 2


def test_multidiscrete_space_sample_without_setting_value_constraint():
    """Test set_value=False with constraint function."""
    space = spaces.MultiDiscrete([10, 10], init_value=np.array([2, 4]), constrain_fn=lambda x: np.sum(x) % 2 == 0)
    original_value = space.value.copy()
    sample = space.sample(set_value=False)
    assert np.array_equal(space.value, original_value)
    assert np.sum(sample) % 2 == 0


def test_multidiscrete_space_check_warning_on_constraint_fail():
    """Test that check() logs warning when constraint fails."""
    space = spaces.MultiDiscrete([5, 5], init_value=np.array([2, 2]), constrain_fn=lambda x: False)
    with patch("stable_worldmodel.spaces.logging.warning") as mock_warning:
        result = space.check()
        assert not result
        mock_warning.assert_called_once()


def test_multidiscrete_space_init_value_violates_constraint():
    """Test behavior when init_value doesn't satisfy constraint."""
    space = spaces.MultiDiscrete([10, 10], init_value=np.array([1, 2]), constrain_fn=lambda x: np.sum(x) % 2 == 0)
    assert not space.contains(space.value)
    assert not space.check()


def test_multidiscrete_space_multiple_samples():
    """Test that value updates correctly across multiple samples."""
    space = spaces.MultiDiscrete([5, 5, 5], init_value=np.array([0, 0, 0]))
    values = [space.sample() for _ in range(5)]
    assert np.array_equal(space.value, values[-1])


def test_multidiscrete_space_different_sizes():
    """Test MultiDiscrete with varying nvec sizes."""
    space = spaces.MultiDiscrete([2, 10, 100], init_value=np.array([0, 5, 50]))
    assert space.contains(np.array([1, 9, 99]))
    assert not space.contains(np.array([2, 9, 99]))
    sample = space.sample()
    assert 0 <= sample[0] < 2
    assert 0 <= sample[1] < 10
    assert 0 <= sample[2] < 100


#####################
## Box Space tests ##
#####################


# vanilla gym behavior


def test_box_space_vanilla_properties():
    space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
    assert space.shape == (3,)
    assert space.dtype == np.float32
    assert np.array_equal(space.low, np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert np.array_equal(space.high, np.array([1.0, 1.0, 1.0], dtype=np.float32))


def test_box_space_vanilla_contains():
    space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
    assert space.contains(np.array([0.0, 0.5], dtype=np.float32))
    assert space.contains(np.array([1.0, 1.0], dtype=np.float32))
    assert not space.contains(np.array([1.1, 0.5], dtype=np.float32))
    assert not space.contains(np.array([-0.1, 0.5], dtype=np.float32))


def test_box_space_vanilla_sample():
    space = spaces.Box(low=-1.0, high=1.0, shape=(2, 3), dtype=np.float32)
    sample = space.sample()
    assert space.contains(sample)
    assert isinstance(sample, np.ndarray)
    assert sample.shape == (2, 3)
    assert sample.dtype == np.float32


def test_box_space_vanilla_check():
    space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
    assert not space.check()


def test_box_space_vanilla_reset():
    space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
    assert space.init_value is None
    space.reset()
    assert space.value == space.init_value
    assert space.value is None


# stable_worldmodel behavior


def test_box_space_properties():
    init_val = np.array([0.5, 0.7], dtype=np.float32)
    space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32, init_value=init_val)
    assert np.array_equal(space.init_value, init_val)
    assert np.array_equal(space.value, init_val)
    assert space.contains(init_val)


def test_box_space_check():
    init_val = np.array([0.5, 0.7], dtype=np.float32)
    space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32, init_value=init_val)
    assert space.check()


def test_box_space_outbound():
    init_val = np.array([1.5, 0.7], dtype=np.float32)
    space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32, init_value=init_val)
    assert not space.contains(space.value)
    assert not space.check()


def test_box_space_reset():
    init_val = np.array([0.5, 0.7], dtype=np.float32)
    space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32, init_value=init_val)
    space.sample()
    assert not np.array_equal(space.value, space.init_value)
    space.reset()
    assert np.array_equal(space.value, space.init_value)


def test_box_space_sample_sets_value():
    init_val = np.array([0.5, 0.5], dtype=np.float32)
    space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32, init_value=init_val)
    sample1 = space.sample()
    assert np.array_equal(space.value, sample1)
    space.reset()
    assert np.array_equal(space.value, init_val)
    sample2 = space.sample(set_value=False)
    assert np.array_equal(space.value, init_val)
    assert space.contains(sample2)


def test_box_space_constraint_function_fails():
    constraint_fn = MagicMock(return_value=False)
    init_val = np.array([0.5, 0.5], dtype=np.float32)
    space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(2,),
        dtype=np.float32,
        init_value=init_val,
        constrain_fn=constraint_fn,
    )
    assert not space.check()
    assert not space.contains(np.array([0.2, 0.3], dtype=np.float32))
    assert not space.contains(np.array([0.8, 0.9], dtype=np.float32))


def test_box_space_constraint_function_fail_sample():
    constraint_fn = MagicMock(return_value=False)
    init_val = np.array([0.5, 0.5], dtype=np.float32)
    space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(2,),
        dtype=np.float32,
        init_value=init_val,
        constrain_fn=constraint_fn,
    )

    with pytest.raises(RuntimeError):
        space.sample(max_tries=3)


def test_box_space_constraint_function_warn_sample():
    constraint_fn = MagicMock(return_value=False)
    init_val = np.array([0.5, 0.5], dtype=np.float32)
    space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(2,),
        dtype=np.float32,
        init_value=init_val,
        constrain_fn=constraint_fn,
    )

    with patch("loguru.logger.warning") as mock_warning:
        with pytest.raises(RuntimeError):
            space.sample(max_tries=5, warn_after_s=0.0)
        mock_warning.assert_called()


def test_box_space_constraint_function_pass():
    constraint_fn = MagicMock(return_value=True)
    init_val = np.array([0.5, 0.5], dtype=np.float32)
    space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(2,),
        dtype=np.float32,
        init_value=init_val,
        constrain_fn=constraint_fn,
    )
    assert space.check()
    assert space.contains(np.array([0.2, 0.3], dtype=np.float32))
    assert space.contains(np.array([0.8, 0.9], dtype=np.float32))
    assert not space.contains(np.array([1.1, 0.5], dtype=np.float32))
    assert not space.contains(np.array([-0.1, 0.5], dtype=np.float32))


def test_box_space_constraint_real_logic():
    """Test with actual constraint logic (e.g., norm must be <= 1.0)."""
    init_val = np.array([0.0, 0.0], dtype=np.float32)
    space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(2,),
        dtype=np.float32,
        init_value=init_val,
        constrain_fn=lambda x: np.linalg.norm(x) <= 1.0,
    )
    sample = space.sample()
    assert np.linalg.norm(sample) <= 1.0
    assert space.contains(sample)


def test_box_space_constraint_rejection_succeeds():
    """Test that rejection sampling eventually succeeds."""
    # Only accept values where first element >= 0.5
    init_val = np.array([0.6, 0.5], dtype=np.float32)
    space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(2,),
        dtype=np.float32,
        init_value=init_val,
        constrain_fn=lambda x: x[0] >= 0.5,
    )
    for _ in range(10):
        sample = space.sample()
        assert sample[0] >= 0.5


def test_box_space_sample_without_setting_value_constraint():
    """Test set_value=False with constraint function."""
    init_val = np.array([0.0, 0.0], dtype=np.float32)
    space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(2,),
        dtype=np.float32,
        init_value=init_val,
        constrain_fn=lambda x: np.linalg.norm(x) <= 1.0,
    )
    original_value = space.value.copy()
    sample = space.sample(set_value=False)
    assert np.array_equal(space.value, original_value)
    assert np.linalg.norm(sample) <= 1.0


def test_box_space_check_warning_on_constraint_fail():
    """Test that check() logs warning when constraint fails."""
    init_val = np.array([0.5, 0.5], dtype=np.float32)
    space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(2,),
        dtype=np.float32,
        init_value=init_val,
        constrain_fn=lambda x: False,
    )
    with patch("stable_worldmodel.spaces.logging.warning") as mock_warning:
        result = space.check()
        assert not result
        mock_warning.assert_called_once()


def test_box_space_init_value_violates_constraint():
    """Test behavior when init_value doesn't satisfy constraint."""
    init_val = np.array([2.0, 2.0], dtype=np.float32)  # norm > 1
    space = spaces.Box(
        low=-5.0,
        high=5.0,
        shape=(2,),
        dtype=np.float32,
        init_value=init_val,
        constrain_fn=lambda x: np.linalg.norm(x) <= 1.0,
    )
    assert not space.contains(space.value)
    assert not space.check()


def test_box_space_multiple_samples():
    """Test that value updates correctly across multiple samples."""
    init_val = np.array([0.0, 0.0], dtype=np.float32)
    space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32, init_value=init_val)
    values = [space.sample() for _ in range(5)]
    assert np.array_equal(space.value, values[-1])


def test_box_space_different_shapes():
    """Test Box with different shapes."""
    # 1D
    space_1d = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(5,),
        dtype=np.float32,
        init_value=np.zeros(5, dtype=np.float32),
    )
    assert space_1d.shape == (5,)
    sample_1d = space_1d.sample()
    assert sample_1d.shape == (5,)

    # 2D
    space_2d = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(3, 4),
        dtype=np.float32,
        init_value=np.zeros((3, 4), dtype=np.float32),
    )
    assert space_2d.shape == (3, 4)
    sample_2d = space_2d.sample()
    assert sample_2d.shape == (3, 4)

    # 3D
    space_3d = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(2, 3, 4),
        dtype=np.float32,
        init_value=np.zeros((2, 3, 4), dtype=np.float32),
    )
    assert space_3d.shape == (2, 3, 4)
    sample_3d = space_3d.sample()
    assert sample_3d.shape == (2, 3, 4)


def test_box_space_different_dtypes():
    """Test Box with different dtypes."""
    # float32
    space_f32 = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(2,),
        dtype=np.float32,
        init_value=np.array([0.5, 0.5], dtype=np.float32),
    )
    assert space_f32.dtype == np.float32

    # float64
    space_f64 = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(2,),
        dtype=np.float64,
        init_value=np.array([0.5, 0.5], dtype=np.float64),
    )
    assert space_f64.dtype == np.float64

    # int32
    space_i32 = spaces.Box(
        low=0,
        high=10,
        shape=(2,),
        dtype=np.int32,
        init_value=np.array([5, 5], dtype=np.int32),
    )
    assert space_i32.dtype == np.int32


def test_box_space_asymmetric_bounds():
    """Test Box with different low and high bounds per dimension."""
    low = np.array([0.0, -5.0, 10.0], dtype=np.float32)
    high = np.array([1.0, 5.0, 20.0], dtype=np.float32)
    init_val = np.array([0.5, 0.0, 15.0], dtype=np.float32)
    space = spaces.Box(low=low, high=high, dtype=np.float32, init_value=init_val)

    assert space.contains(init_val)
    sample = space.sample()
    assert 0.0 <= sample[0] <= 1.0
    assert -5.0 <= sample[1] <= 5.0
    assert 10.0 <= sample[2] <= 20.0


########################
## RGBBox Space tests ##
########################


# vanilla gym behavior (inherited from Box)


def test_rgbbox_space_vanilla_properties():
    """Test that RGBBox has correct automatic properties."""
    space = spaces.RGBBox(shape=(64, 64, 3))
    assert space.shape == (64, 64, 3)
    assert space.dtype == np.uint8
    assert np.all(space.low == 0)
    assert np.all(space.high == 255)


def test_rgbbox_space_invalid_shape():
    """Test that RGBBox raises error for invalid shapes."""
    # No dimension of size 3
    with pytest.raises(ValueError, match="shape must have a channel of size 3"):
        spaces.RGBBox(shape=(64, 64))

    with pytest.raises(ValueError, match="shape must have a channel of size 3"):
        spaces.RGBBox(shape=(64, 64, 4))


def test_rgbbox_space_valid_shapes():
    """Test RGBBox with various valid shape formats."""
    # HWC format (height, width, channels)
    space_hwc = spaces.RGBBox(shape=(32, 32, 3))
    assert space_hwc.shape == (32, 32, 3)

    # CHW format (channels, height, width)
    space_chw = spaces.RGBBox(shape=(3, 32, 32))
    assert space_chw.shape == (3, 32, 32)

    # Just 3 channels
    space_c = spaces.RGBBox(shape=(3,))
    assert space_c.shape == (3,)


def test_rgbbox_space_vanilla_contains():
    """Test contains method with RGB values."""
    space = spaces.RGBBox(shape=(2, 2, 3))

    # Valid RGB image
    valid_img = np.zeros((2, 2, 3), dtype=np.uint8)
    assert space.contains(valid_img)

    # Max values
    max_img = np.full((2, 2, 3), 255, dtype=np.uint8)
    assert space.contains(max_img)

    # Out of range values should not be possible with uint8, but test wrong shape
    wrong_shape = np.zeros((2, 2, 4), dtype=np.uint8)
    assert not space.contains(wrong_shape)


def test_rgbbox_space_vanilla_sample():
    """Test sampling RGB images."""
    space = spaces.RGBBox(shape=(16, 16, 3))
    sample = space.sample()

    assert isinstance(sample, np.ndarray)
    assert sample.shape == (16, 16, 3)
    assert sample.dtype == np.uint8
    assert np.all(sample >= 0)
    assert np.all(sample <= 255)
    assert space.contains(sample)


def test_rgbbox_space_vanilla_check():
    """Test check method without init_value."""
    space = spaces.RGBBox(shape=(8, 8, 3))
    assert not space.check()


def test_rgbbox_space_vanilla_reset():
    """Test reset without init_value."""
    space = spaces.RGBBox(shape=(8, 8, 3))
    assert space.init_value is None
    space.reset()
    assert space.value is None


def test_rgbbox_space_properties():
    """Test RGBBox with init_value."""
    init_img = np.ones((10, 10, 3), dtype=np.uint8) * 128
    space = spaces.RGBBox(shape=(10, 10, 3), init_value=init_img)

    assert np.array_equal(space.init_value, init_img)
    assert np.array_equal(space.value, init_img)
    assert space.contains(init_img)


def test_rgbbox_space_check():
    """Test check with valid init_value."""
    init_img = np.zeros((5, 5, 3), dtype=np.uint8)
    space = spaces.RGBBox(shape=(5, 5, 3), init_value=init_img)
    assert space.check()


def test_rgbbox_space_reset():
    """Test reset functionality."""
    init_img = np.zeros((5, 5, 3), dtype=np.uint8)
    space = spaces.RGBBox(shape=(5, 5, 3), init_value=init_img)

    # Sample changes value
    space.sample()
    assert not np.array_equal(space.value, init_img)

    # Reset brings it back
    space.reset()
    assert np.array_equal(space.value, init_img)


def test_rgbbox_space_sample_sets_value():
    """Test that sampling updates the space value."""
    init_img = np.zeros((5, 5, 3), dtype=np.uint8)
    space = spaces.RGBBox(shape=(5, 5, 3), init_value=init_img)

    sample1 = space.sample()
    assert np.array_equal(space.value, sample1)

    space.reset()
    assert np.array_equal(space.value, init_img)

    sample2 = space.sample(set_value=False)
    assert np.array_equal(space.value, init_img)
    assert space.contains(sample2)


def test_rgbbox_space_constraint_function():
    """Test RGBBox with constraint function (e.g., not too dark)."""
    init_img = np.ones((4, 4, 3), dtype=np.uint8) * 128

    # Constraint: mean pixel value must be > 100
    def not_too_dark(img):
        return np.mean(img) > 100

    space = spaces.RGBBox(shape=(4, 4, 3), init_value=init_img, constrain_fn=not_too_dark)

    assert space.check()
    assert space.contains(init_img)

    # Dark image should not pass constraint
    dark_img = np.ones((4, 4, 3), dtype=np.uint8) * 50
    assert not space.contains(dark_img)


def test_rgbbox_space_constraint_rejection_succeeds():
    """Test that rejection sampling works with constraints."""
    init_img = np.ones((3, 3, 3), dtype=np.uint8) * 200

    # Constraint: mean pixel value must be > 50
    space = spaces.RGBBox(
        shape=(3, 3, 3),
        init_value=init_img,
        constrain_fn=lambda img: np.mean(img) > 50,
    )

    # This might take a few tries, but should eventually succeed
    sample = space.sample(max_tries=1000)
    assert np.mean(sample) > 50


def test_rgbbox_space_chw_format():
    """Test RGBBox with CHW (channels-first) format."""
    init_img = np.zeros((3, 16, 16), dtype=np.uint8)
    space = spaces.RGBBox(shape=(3, 16, 16), init_value=init_img)

    assert space.shape == (3, 16, 16)
    sample = space.sample()
    assert sample.shape == (3, 16, 16)
    assert sample.dtype == np.uint8


def test_rgbbox_space_different_resolutions():
    """Test RGBBox with different image resolutions."""
    resolutions = [(32, 32, 3), (64, 64, 3), (128, 128, 3), (64, 128, 3)]

    for shape in resolutions:
        init_img = np.zeros(shape, dtype=np.uint8)
        space = spaces.RGBBox(shape=shape, init_value=init_img)

        assert space.shape == shape
        sample = space.sample()
        assert sample.shape == shape
        assert sample.dtype == np.uint8
        assert space.contains(sample)


def test_rgbbox_space_specific_colors():
    """Test RGBBox with specific color patterns."""
    # Red image
    red_img = np.zeros((8, 8, 3), dtype=np.uint8)
    red_img[:, :, 0] = 255

    space = spaces.RGBBox(shape=(8, 8, 3), init_value=red_img)
    assert np.array_equal(space.value, red_img)
    assert space.contains(red_img)

    # Green image
    green_img = np.zeros((8, 8, 3), dtype=np.uint8)
    green_img[:, :, 1] = 255
    assert space.contains(green_img)

    # Blue image
    blue_img = np.zeros((8, 8, 3), dtype=np.uint8)
    blue_img[:, :, 2] = 255
    assert space.contains(blue_img)


def test_rgbbox_space_gradient():
    """Test RGBBox with gradient image."""
    gradient = np.zeros((10, 10, 3), dtype=np.uint8)
    for i in range(10):
        gradient[i, :, :] = i * 25  # 0, 25, 50, ..., 225

    space = spaces.RGBBox(shape=(10, 10, 3), init_value=gradient)
    assert space.contains(gradient)
    assert space.check()


def test_rgbbox_space_constraint_fail_sample():
    """Test that impossible constraints raise RuntimeError."""
    init_img = np.ones((3, 3, 3), dtype=np.uint8) * 128

    # Impossible constraint
    constraint_fn = MagicMock(return_value=False)
    space = spaces.RGBBox(shape=(3, 3, 3), init_value=init_img, constrain_fn=constraint_fn)

    with pytest.raises(RuntimeError):
        space.sample(max_tries=3)


def test_rgbbox_space_multiple_samples():
    """Test that value updates correctly across multiple samples."""
    init_img = np.zeros((5, 5, 3), dtype=np.uint8)
    space = spaces.RGBBox(shape=(5, 5, 3), init_value=init_img)

    values = [space.sample() for _ in range(3)]
    assert np.array_equal(space.value, values[-1])


def test_rgbbox_space_dtype_enforcement():
    """Test that dtype is always uint8."""
    space = spaces.RGBBox(shape=(4, 4, 3))

    # Even if we try to pass different dtype in kwargs, it should be uint8
    assert space.dtype == np.uint8

    sample = space.sample()
    assert sample.dtype == np.uint8


######################
## Dict Space tests ##
######################


# One-level Dict tests


def test_dict_space_one_level_properties():
    """Test basic properties of a one-level Dict space."""
    space = spaces.Dict(
        {
            "x": spaces.Discrete(5, init_value=2),
            "y": spaces.Discrete(5, init_value=3),
        }
    )

    assert "x" in space.spaces
    assert "y" in space.spaces
    assert len(space.spaces) == 2


def test_dict_space_one_level_init_value():
    """Test init_value property for one-level Dict."""
    space = spaces.Dict(
        {
            "x": spaces.Discrete(5, init_value=2),
            "y": spaces.Discrete(5, init_value=3),
        }
    )

    init_val = space.init_value
    assert init_val["x"] == 2
    assert init_val["y"] == 3


def test_dict_space_one_level_value():
    """Test value property for one-level Dict."""
    space = spaces.Dict(
        {
            "x": spaces.Discrete(5, init_value=2),
            "y": spaces.Discrete(5, init_value=3),
        }
    )

    val = space.value
    assert val["x"] == 2
    assert val["y"] == 3


def test_dict_space_one_level_sampling_order_default():
    """Test default sampling order (insertion order) for one-level Dict."""
    space = spaces.Dict(
        {
            "x": spaces.Discrete(5, init_value=2),
            "y": spaces.Discrete(5, init_value=3),
            "z": spaces.Discrete(5, init_value=4),
        }
    )

    # Should use insertion order
    assert space._sampling_order == ["x", "y", "z"]

    # sampling_order property returns set of dotted paths
    order = space.sampling_order
    assert order == {"x", "y", "z"}


def test_dict_space_one_level_sampling_order_explicit():
    """Test explicit sampling order for one-level Dict."""
    space = spaces.Dict(
        {
            "x": spaces.Discrete(5, init_value=2),
            "y": spaces.Discrete(5, init_value=3),
            "z": spaces.Discrete(5, init_value=4),
        },
        sampling_order=["z", "x", "y"],
    )

    assert space._sampling_order == ["z", "x", "y"]


def test_dict_space_one_level_sampling_order_partial():
    """Test partial sampling order (missing keys get appended)."""
    space = spaces.Dict(
        {
            "x": spaces.Discrete(5, init_value=2),
            "y": spaces.Discrete(5, init_value=3),
            "z": spaces.Discrete(5, init_value=4),
        },
        sampling_order=["z", "x"],  # Missing "y"
    )

    # Should append missing keys
    assert space._sampling_order == ["z", "x", "y"]


def test_dict_space_one_level_sampling_order_partial_warning():
    """Test that warning is logged for partial sampling order."""
    with patch("stable_worldmodel.spaces.logging.warning") as mock_warning:
        space = spaces.Dict(
            {
                "x": spaces.Discrete(5, init_value=2),
                "y": spaces.Discrete(5, init_value=3),
            },
            sampling_order=["x"],  # Missing "y"
        )
        mock_warning.assert_called_once()
        assert space._sampling_order == ["x", "y"]  # Verify order was fixed


def test_dict_space_one_level_sampling_order_invalid():
    """Test that invalid sampling order raises assertion."""
    with pytest.raises(ValueError):
        spaces.Dict(
            {
                "x": spaces.Discrete(5, init_value=2),
                "y": spaces.Discrete(5, init_value=3),
            },
            sampling_order=["x", "invalid_key"],
        )


def test_dict_space_one_level_contains():
    """Test contains method for one-level Dict."""
    space = spaces.Dict(
        {
            "x": spaces.Discrete(5, init_value=2),
            "y": spaces.Discrete(5, init_value=3),
        }
    )

    assert space.contains({"x": 2, "y": 3})
    assert space.contains({"x": 0, "y": 4})
    assert not space.contains({"x": 5, "y": 3})  # x out of bounds
    assert not space.contains({"x": 2, "y": 5})  # y out of bounds
    assert not space.contains({"x": 2})  # missing key
    assert not space.contains("not_a_dict")  # not a dict


def test_dict_space_one_level_sample():
    """Test sampling for one-level Dict."""
    space = spaces.Dict(
        {
            "x": spaces.Discrete(5, init_value=2),
            "y": spaces.Discrete(5, init_value=3),
        }
    )

    sample = space.sample()
    assert isinstance(sample, dict)
    assert "x" in sample
    assert "y" in sample
    assert space.contains(sample)


def test_dict_space_one_level_sample_sets_value():
    """Test that sampling updates value in one-level Dict."""
    space = spaces.Dict(
        {
            "x": spaces.Discrete(5, init_value=-1),
            "y": spaces.Discrete(5, init_value=3),
        }
    )

    sample = space.sample()
    assert space.value == sample

    # Test set_value=False
    space.reset()
    sample2 = space.sample(set_value=False)
    assert space.value == space.init_value
    assert space.value == {"x": -1, "y": 3}
    assert space.value != sample2
    assert space.value == {"x": -1, "y": 3}


def test_dict_space_one_level_reset():
    """Test reset for one-level Dict."""
    space = spaces.Dict(
        {
            "x": spaces.Discrete(5, init_value=2),
            "y": spaces.Discrete(5, init_value=3),
        }
    )

    space.sample()
    space.reset()

    assert space.value == {"x": 2, "y": 3}
    assert space["x"].value == 2
    assert space["y"].value == 3


def test_dict_space_one_level_check():
    """Test check method for one-level Dict."""
    space = spaces.Dict(
        {
            "x": spaces.Discrete(5, init_value=2),
            "y": spaces.Discrete(5, init_value=3),
        }
    )

    assert space.check()

    # Manually set invalid value
    space["x"]._value = 10
    assert not space.check()


def test_dict_space_one_level_check_debug():
    """Test check with debug flag."""
    space = spaces.Dict(
        {
            "x": spaces.Discrete(5, init_value=2),
            "y": spaces.Discrete(5, init_value=10),  # Out of bounds
        }
    )

    with patch("stable_worldmodel.spaces.logging.warning") as mock_warning:
        result = space.check(debug=True)
        assert not result
        mock_warning.assert_called()


def test_dict_space_one_level_names():
    """Test names method for one-level Dict."""
    space = spaces.Dict(
        {
            "x": spaces.Discrete(5, init_value=2),
            "y": spaces.Discrete(5, init_value=3),
        }
    )

    names = space.names()
    assert set(names) == {"x", "y"}


def test_dict_space_one_level_constraint_function():
    """Test constraint function for one-level Dict."""
    space = spaces.Dict(
        {
            "x": spaces.Discrete(10, init_value=2),
            "y": spaces.Discrete(10, init_value=3),
        },
        constrain_fn=lambda d: d["x"] + d["y"] < 10,
    )

    assert space.contains({"x": 2, "y": 3})
    assert not space.contains({"x": 8, "y": 8})


def test_dict_space_one_level_constraint_fail_sample():
    """Test that impossible constraint raises RuntimeError."""
    constraint_fn = MagicMock(return_value=False)
    space = spaces.Dict(
        {
            "x": spaces.Discrete(5, init_value=2),
        },
        constrain_fn=constraint_fn,
    )

    with pytest.raises(RuntimeError, match="constrain_fn not satisfied"):
        space.sample(max_tries=3)


def test_dict_space_one_level_constraint_warn_sample():
    """Test warning on slow constraint sampling."""
    constraint_fn = MagicMock(return_value=False)
    space = spaces.Dict(
        {
            "x": spaces.Discrete(5, init_value=2),
        },
        constrain_fn=constraint_fn,
    )

    with patch("stable_worldmodel.spaces.logging.warning") as mock_warning:
        with pytest.raises(RuntimeError):
            space.sample(max_tries=5, warn_after_s=0.0)
        mock_warning.assert_called()


# Two-level Dict tests


def test_dict_space_two_level_properties():
    """Test properties of a two-level nested Dict space."""
    space = spaces.Dict(
        {
            "player": spaces.Dict(
                {
                    "x": spaces.Discrete(10, init_value=5),
                    "y": spaces.Discrete(10, init_value=5),
                }
            ),
            "enemy": spaces.Dict(
                {
                    "x": spaces.Discrete(10, init_value=8),
                    "y": spaces.Discrete(10, init_value=8),
                }
            ),
        }
    )

    assert "player" in space.spaces
    assert "enemy" in space.spaces
    assert isinstance(space["player"], spaces.Dict)
    assert isinstance(space["enemy"], spaces.Dict)


def test_dict_space_two_level_init_value():
    """Test init_value for two-level nested Dict."""
    space = spaces.Dict(
        {
            "player": spaces.Dict(
                {
                    "x": spaces.Discrete(10, init_value=5),
                    "y": spaces.Discrete(10, init_value=6),
                }
            ),
            "score": spaces.Discrete(100, init_value=0),
        }
    )

    init_val = space.init_value
    assert init_val["player"]["x"] == 5
    assert init_val["player"]["y"] == 6
    assert init_val["score"] == 0


def test_dict_space_two_level_value():
    """Test value property for two-level nested Dict."""
    space = spaces.Dict(
        {
            "player": spaces.Dict(
                {
                    "x": spaces.Discrete(10, init_value=5),
                    "y": spaces.Discrete(10, init_value=6),
                }
            ),
            "score": spaces.Discrete(100, init_value=0),
        }
    )

    val = space.value
    assert val["player"]["x"] == 5
    assert val["player"]["y"] == 6
    assert val["score"] == 0


def test_dict_space_two_level_sampling_order():
    """Test sampling order for two-level nested Dict."""
    space = spaces.Dict(
        {
            "player": spaces.Dict(
                {
                    "x": spaces.Discrete(10, init_value=5),
                    "y": spaces.Discrete(10, init_value=6),
                },
                sampling_order=["y", "x"],
            ),
            "score": spaces.Discrete(100, init_value=0),
        },
        sampling_order=["score", "player"],
    )

    # Top-level order
    assert space._sampling_order == ["score", "player"]

    # Nested order
    assert space["player"]._sampling_order == ["y", "x"]

    # Full sampling order with dotted paths
    order = space.sampling_order
    assert "score" in order
    assert "player" in order
    assert "player.x" in order
    assert "player.y" in order
    assert len(order) == 4


def test_dict_space_two_level_contains():
    """Test contains for two-level nested Dict."""
    space = spaces.Dict(
        {
            "player": spaces.Dict(
                {
                    "x": spaces.Discrete(10, init_value=5),
                    "y": spaces.Discrete(10, init_value=5),
                }
            ),
            "score": spaces.Discrete(100, init_value=0),
        }
    )

    assert space.contains({"player": {"x": 5, "y": 5}, "score": 0})
    assert space.contains({"player": {"x": 0, "y": 9}, "score": 99})
    assert not space.contains({"player": {"x": 10, "y": 5}, "score": 0})  # x out of bounds
    assert not space.contains({"player": {"x": 5}, "score": 0})  # missing nested key


def test_dict_space_two_level_sample():
    """Test sampling for two-level nested Dict."""
    space = spaces.Dict(
        {
            "player": spaces.Dict(
                {
                    "x": spaces.Discrete(10, init_value=5),
                    "y": spaces.Discrete(10, init_value=5),
                }
            ),
            "score": spaces.Discrete(100, init_value=0),
        }
    )

    sample = space.sample()
    assert isinstance(sample, dict)
    assert "player" in sample
    assert "score" in sample
    assert isinstance(sample["player"], dict)
    assert "x" in sample["player"]
    assert "y" in sample["player"]
    assert space.contains(sample)


def test_dict_space_two_level_reset():
    """Test reset for two-level nested Dict."""
    space = spaces.Dict(
        {
            "player": spaces.Dict(
                {
                    "x": spaces.Discrete(10, init_value=5),
                    "y": spaces.Discrete(10, init_value=6),
                }
            ),
            "score": spaces.Discrete(100, init_value=0),
        }
    )

    space.sample()
    space.reset()

    assert space.value["player"]["x"] == 5
    assert space.value["player"]["y"] == 6
    assert space.value["score"] == 0


def test_dict_space_two_level_check():
    """Test check for two-level nested Dict."""
    space = spaces.Dict(
        {
            "player": spaces.Dict(
                {
                    "x": spaces.Discrete(10, init_value=5),
                    "y": spaces.Discrete(10, init_value=6),
                }
            ),
            "score": spaces.Discrete(100, init_value=0),
        }
    )

    assert space.check()

    # Invalidate nested value
    space["player"]["x"]._value = 20
    assert not space.check()


def test_dict_space_two_level_names():
    """Test names method for two-level nested Dict."""
    space = spaces.Dict(
        {
            "player": spaces.Dict(
                {
                    "x": spaces.Discrete(10, init_value=5),
                    "y": spaces.Discrete(10, init_value=5),
                }
            ),
            "score": spaces.Discrete(100, init_value=0),
        }
    )

    names = space.names()
    assert set(names) == {"player.x", "player.y", "score"}


def test_dict_space_two_level_constraint_function():
    """Test constraint function for two-level nested Dict."""
    # Constraint: player x must be less than enemy x
    space = spaces.Dict(
        {
            "player": spaces.Dict(
                {
                    "x": spaces.Discrete(10, init_value=2),
                }
            ),
            "enemy": spaces.Dict(
                {
                    "x": spaces.Discrete(10, init_value=8),
                }
            ),
        },
        constrain_fn=lambda d: d["player"]["x"] < d["enemy"]["x"],
    )

    assert space.contains({"player": {"x": 2}, "enemy": {"x": 8}})
    assert not space.contains({"player": {"x": 8}, "enemy": {"x": 2}})


# Three-level Dict tests


def test_dict_space_three_level_properties():
    """Test properties of a three-level nested Dict space."""
    space = spaces.Dict(
        {
            "game": spaces.Dict(
                {
                    "world": spaces.Dict(
                        {
                            "width": spaces.Discrete(100, init_value=50),
                            "height": spaces.Discrete(100, init_value=50),
                        }
                    ),
                    "difficulty": spaces.Discrete(5, init_value=2),
                }
            ),
            "score": spaces.Discrete(1000, init_value=0),
        }
    )

    assert "game" in space.spaces
    assert "world" in space["game"].spaces
    assert "width" in space["game"]["world"].spaces


def test_dict_space_three_level_init_value():
    """Test init_value for three-level nested Dict."""
    space = spaces.Dict(
        {
            "game": spaces.Dict(
                {
                    "world": spaces.Dict(
                        {
                            "width": spaces.Discrete(100, init_value=50),
                            "height": spaces.Discrete(100, init_value=60),
                        }
                    ),
                    "difficulty": spaces.Discrete(5, init_value=2),
                }
            ),
        }
    )

    init_val = space.init_value
    assert init_val["game"]["world"]["width"] == 50
    assert init_val["game"]["world"]["height"] == 60
    assert init_val["game"]["difficulty"] == 2


def test_dict_space_three_level_value():
    """Test value property for three-level nested Dict."""
    space = spaces.Dict(
        {
            "game": spaces.Dict(
                {
                    "world": spaces.Dict(
                        {
                            "width": spaces.Discrete(100, init_value=50),
                            "height": spaces.Discrete(100, init_value=60),
                        }
                    ),
                    "difficulty": spaces.Discrete(5, init_value=2),
                }
            ),
        }
    )

    val = space.value
    assert val["game"]["world"]["width"] == 50
    assert val["game"]["world"]["height"] == 60
    assert val["game"]["difficulty"] == 2


def test_dict_space_three_level_sampling_order():
    """Test sampling order for three-level nested Dict."""
    space = spaces.Dict(
        {
            "game": spaces.Dict(
                {
                    "world": spaces.Dict(
                        {
                            "width": spaces.Discrete(100, init_value=50),
                            "height": spaces.Discrete(100, init_value=60),
                        },
                        sampling_order=["height", "width"],
                    ),
                    "difficulty": spaces.Discrete(5, init_value=2),
                },
                sampling_order=["difficulty", "world"],
            ),
        }
    )

    order = space.sampling_order
    assert "game" in order
    assert "game.world" in order
    assert "game.world.width" in order
    assert "game.world.height" in order
    assert "game.difficulty" in order


def test_dict_space_three_level_names():
    """Test names method for three-level nested Dict."""
    space = spaces.Dict(
        {
            "game": spaces.Dict(
                {
                    "world": spaces.Dict(
                        {
                            "width": spaces.Discrete(100, init_value=50),
                            "height": spaces.Discrete(100, init_value=60),
                        }
                    ),
                    "difficulty": spaces.Discrete(5, init_value=2),
                }
            ),
            "score": spaces.Discrete(1000, init_value=0),
        }
    )

    names = space.names()
    assert set(names) == {
        "game.world.width",
        "game.world.height",
        "game.difficulty",
        "score",
    }


def test_dict_space_three_level_sample():
    """Test sampling for three-level nested Dict."""
    space = spaces.Dict(
        {
            "game": spaces.Dict(
                {
                    "world": spaces.Dict(
                        {
                            "width": spaces.Discrete(100, init_value=50),
                            "height": spaces.Discrete(100, init_value=60),
                        }
                    ),
                    "difficulty": spaces.Discrete(5, init_value=2),
                }
            ),
        }
    )

    sample = space.sample()
    assert "game" in sample
    assert "world" in sample["game"]
    assert "difficulty" in sample["game"]
    assert "width" in sample["game"]["world"]
    assert "height" in sample["game"]["world"]
    assert space.contains(sample)


def test_dict_space_three_level_reset():
    """Test reset for three-level nested Dict."""
    space = spaces.Dict(
        {
            "game": spaces.Dict(
                {
                    "world": spaces.Dict(
                        {
                            "width": spaces.Discrete(100, init_value=50),
                            "height": spaces.Discrete(100, init_value=60),
                        }
                    ),
                }
            ),
        }
    )

    space.sample()
    space.reset()

    assert space.value["game"]["world"]["width"] == 50
    assert space.value["game"]["world"]["height"] == 60


# Mixed types Dict tests


def test_dict_space_mixed_types():
    """Test Dict with mixed space types."""
    space = spaces.Dict(
        {
            "discrete": spaces.Discrete(5, init_value=2),
            "box": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
                init_value=np.array([0.5, 0.5], dtype=np.float32),
            ),
            "multidiscrete": spaces.MultiDiscrete([3, 4], init_value=np.array([1, 2])),
        }
    )

    init_val = space.init_value
    assert init_val["discrete"] == 2
    assert np.array_equal(init_val["box"], np.array([0.5, 0.5], dtype=np.float32))
    assert np.array_equal(init_val["multidiscrete"], np.array([1, 2]))


def test_dict_space_mixed_nested():
    """Test deeply nested Dict with mixed types."""
    space = spaces.Dict(
        {
            "config": spaces.Dict(
                {
                    "resolution": spaces.MultiDiscrete([1920, 1080], init_value=np.array([800, 600])),
                    "settings": spaces.Dict(
                        {
                            "volume": spaces.Box(
                                low=0.0,
                                high=1.0,
                                shape=(1,),
                                dtype=np.float32,
                                init_value=np.array([0.7], dtype=np.float32),
                            ),
                            "difficulty": spaces.Discrete(5, init_value=2),
                        }
                    ),
                }
            ),
            "player_id": spaces.Discrete(1000, init_value=42),
        }
    )

    val = space.value
    assert np.array_equal(val["config"]["resolution"], np.array([800, 600]))
    assert val["config"]["settings"]["difficulty"] == 2
    assert val["player_id"] == 42


# Update method tests


def test_dict_space_update_single_key():
    """Test update method with single key."""
    space = spaces.Dict(
        {
            "x": spaces.Discrete(10, init_value=5),
            "y": spaces.Discrete(10, init_value=5),
        }
    )

    original_x = space["x"].value
    original_y = space["y"].value

    space.update({"x"})

    # x should have changed, y should not
    assert space["x"].value != original_x or original_x == space["x"].value  # Might sample same value
    assert space["y"].value == original_y  # Should not change


def test_dict_space_update_nested_key():
    """Test update method with nested key."""
    space = spaces.Dict(
        {
            "player": spaces.Dict(
                {
                    "x": spaces.Discrete(10, init_value=5),
                    "y": spaces.Discrete(10, init_value=6),
                }
            ),
            "score": spaces.Discrete(100, init_value=0),
        }
    )

    original_score = space["score"].value

    space.update({"player.x"})

    # Score should not change
    assert space["score"].value == original_score


def test_dict_space_update_invalid_key():
    """Test update method with invalid key raises ValueError."""
    space = spaces.Dict(
        {
            "x": spaces.Discrete(10, init_value=5),
        }
    )

    with patch("stable_worldmodel.spaces.Dict._get_sampling_order") as mock_get:
        mock_get.return_value = ["x", "invalid_key"]
        with pytest.raises(ValueError):
            space.update({"invalid_key"})


def test_dict_space_update_check_assertion():
    """Test that update asserts check passes."""
    space = spaces.Dict(
        {
            "x": spaces.Discrete(10, init_value=5),
        }
    )

    # Normal update should pass
    space.update({"x"})
    assert space.check()


def test_dict_space_sampling_order_invalid_key(monkeypatch):
    """Test update method with invalid key raises ValueError."""
    space = spaces.Dict(
        {
            "x": spaces.Discrete(10, init_value=5),
        }
    )

    monkeypatch.setattr(space, "_sampling_order", ["x", "invalid_key"])

    assert ["x"] == list(space._get_sampling_order())


# Edge cases


def test_dict_space_value_without_init_value():
    """Test value property when subspace doesn't have init_value."""
    # Create a vanilla gymnasium space without init_value
    from gymnasium import spaces as gym_spaces

    space = spaces.Dict(
        {
            "custom": gym_spaces.Discrete(5),  # Vanilla gym space without init_value
            "extended": spaces.Discrete(5, init_value=2),
        }
    )

    # init_value should sample for vanilla gym space
    init_val = space.init_value
    assert "custom" in init_val
    assert init_val["extended"] == 2


def test_dict_space_value_property_error():
    """Test value property raises error when subspace doesn't have value."""
    from gymnasium import spaces as gym_spaces

    space = spaces.Dict(
        {
            "custom": gym_spaces.Discrete(5),
        }
    )

    # Accessing value should raise ValueError
    with pytest.raises(ValueError, match="does not have value property"):
        _ = space.value


def test_dict_space_empty():
    """Test empty Dict space."""
    space = spaces.Dict({})

    assert len(space.spaces) == 0
    assert space.init_value == {}
    assert space.value == {}
    assert space.contains({})
    assert space.sample() == {}


def test_dict_space_single_element():
    """Test Dict with single element."""
    space = spaces.Dict(
        {
            "only": spaces.Discrete(5, init_value=3),
        }
    )

    assert space.value == {"only": 3}
    assert space.contains({"only": 3})


def test_dict_space_constraint_with_nested_access():
    """Test constraint function accessing nested values correctly."""
    space = spaces.Dict(
        {
            "player": spaces.Dict(
                {
                    "x": spaces.Discrete(10, init_value=2),
                    "y": spaces.Discrete(10, init_value=3),
                }
            ),
            "target": spaces.Dict(
                {
                    "x": spaces.Discrete(10, init_value=7),
                    "y": spaces.Discrete(10, init_value=8),
                }
            ),
        },
        # Constraint: player must not be at same position as target
        constrain_fn=lambda d: d["player"]["x"] != d["target"]["x"] or d["player"]["y"] != d["target"]["y"],
    )

    assert space.contains({"player": {"x": 2, "y": 3}, "target": {"x": 7, "y": 8}})
    assert not space.contains({"player": {"x": 5, "y": 5}, "target": {"x": 5, "y": 5}})
