import re
import time
from collections import deque
from collections.abc import Callable, Iterable

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.utils import is_space_dtype_shape_equiv
from gymnasium.vector import VectorWrapper
from gymnasium.vector.utils import (
    batch_differing_spaces,
    batch_space,
)

from stable_worldmodel.utils import get_in


class EnsureInfoKeysWrapper(gym.Wrapper):
    """Validates that required keys are present in the info dict after reset and step.

    Supports regex patterns for flexible key matching. Raises RuntimeError if any
    required pattern has no matching key.

    Args:
        env: The Gymnasium environment to wrap.
        required_keys: Iterable of regex patterns as strings. Each pattern must match
            at least one key in the info dict.

    Raises:
        RuntimeError: If any required pattern has no matching key in info dict.
    """

    def __init__(self, env, required_keys: Iterable[str]):
        super().__init__(env)
        self._patterns: list[re.Pattern] = []
        for k in required_keys:
            self._patterns.append(re.compile(k))
        # else:
        #     # exact match
        #     self._patterns.append(re.compile(rf"^{re.escape(k)}$"))

    def _check(self, info: dict, where: str):
        keys = list(info.keys())
        missing = [p.pattern for p in self._patterns if not any(p.fullmatch(k) for k in keys)]
        if missing:
            raise RuntimeError(
                f"{where}: required info keys missing (patterns with no match): {missing}. Present keys: {keys}"
            )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._check(info, "step()")
        return obs, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self._check(info, "reset()")
        return obs, info


class EnsureImageShape(gym.Wrapper):
    """Validates that an image in the info dict has the expected spatial dimensions.

    Args:
        env: The Gymnasium environment to wrap.
        image_key: Key in info dict containing the image to validate.
        image_shape: Expected (height, width) tuple for the image.

    Raises:
        RuntimeError: If the image shape doesn't match the expected dimensions.
    """

    def __init__(self, env, image_key, image_shape):
        super().__init__(env)
        self.image_key = image_key
        self.image_shape = image_shape  # (height, width)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info[self.image_key].shape[:-1] != self.image_shape:
            raise RuntimeError(f"Image shape {info[self.image_key].shape} should be {self.image_shape}")
        return obs, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        if info[self.image_key].shape[:-1] != self.image_shape:
            raise RuntimeError(f"Image shape {info[self.image_key].shape} should be {self.image_shape}")
        return obs, info


class EnsureGoalInfoWrapper(gym.Wrapper):
    """Validates that 'goal' key is present in info dict during reset and/or step.

    Useful for goal-conditioned environments to ensure goal information is provided.

    Args:
        env: The Gymnasium environment to wrap.
        check_reset: If True, validates 'goal' key is in info after reset().
        check_step: If True, validates 'goal' key is in info after step().

    Raises:
        RuntimeError: If 'goal' key is missing when validation is enabled.
    """

    def __init__(self, env, check_reset, check_step: bool = False):
        super().__init__(env)
        self.check_reset = check_reset
        self.check_step = check_step

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        if self.check_reset and "goal" not in info:
            raise RuntimeError("The info dict returned by reset() must contain the key 'goal'.")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.check_step and "goal" not in info:
            raise RuntimeError("The info dict returned by step() must contain the key 'goal'.")
        return obs, reward, terminated, truncated, info


class EverythingToInfoWrapper(gym.Wrapper):
    """Moves all transition information into the info dict for unified data access.

    Adds observation, reward, terminated, truncated, action, and step_idx to info.
    Optionally tracks environment variations when specified in reset options.

    Args:
        env: The Gymnasium environment to wrap.

    Info Keys Added:
        - observation (or dict keys if obs is dict): Current observation.
        - reward: Reward value (NaN after reset).
        - terminated: Episode termination flag.
        - truncated: Episode truncation flag.
        - action: Action taken (NaN sample after reset).
        - step_idx: Current step counter.
        - variation.{key}: Variation values if requested via reset options.

    Note:
        Pass options={"variation": ["key1", "key2"]} or ["all"] to reset() to track variations.
    """

    def __init__(self, env):
        super().__init__(env)
        self._variations_watch = []

    def reset(self, *args, **kwargs):
        self._step_counter = 0
        obs, info = self.env.reset(*args, **kwargs)
        if type(obs) is not dict:
            _obs = {"observation": obs}
        else:
            _obs = obs
        for key in _obs:
            assert key not in info
            info[key] = _obs[key]

        assert "reward" not in info
        info["reward"] = np.nan
        assert "terminated" not in info
        info["terminated"] = False
        assert "truncated" not in info
        info["truncated"] = False
        assert "action" not in info
        info["action"] = self.env.action_space.sample()
        assert "step_idx" not in info
        info["step_idx"] = self._step_counter

        # add all variations to info if needed
        options = kwargs.get("options") or {}

        if "variation" in options:
            var_opt = options["variation"]
            assert isinstance(options["variation"], list | tuple), (
                "variation option must be a list or tuple containing variation names to sample"
            )
            if len(var_opt) == 1 and var_opt[0] == "all":
                self._variations_watch = self.env.unwrapped.variation_space.names()
            else:
                self._variations_watch = var_opt

        for key in self._variations_watch:
            var_key = f"variation.{key}"
            assert var_key not in info
            subvar_space = get_in(self.env.unwrapped.variation_space, key.split("."))
            info[var_key] = subvar_space.value

        if type(info["action"]) is dict:
            raise NotImplementedError
        else:
            info["action"] *= np.nan
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_counter += 1
        if type(obs) is not dict:
            _obs = {"observation": obs}
        else:
            _obs = obs
        for key in _obs:
            assert key not in info
            info[key] = _obs[key]
        assert "reward" not in info
        info["reward"] = reward
        assert "terminated" not in info
        info["terminated"] = bool(terminated)
        assert "truncated" not in info
        info["truncated"] = bool(truncated)
        assert "action" not in info
        info["action"] = action
        assert "step_idx" not in info
        info["step_idx"] = self._step_counter

        for key in self._variations_watch:
            var_key = f"variation.{key}"
            assert var_key not in info
            subvar_space = get_in(self.env.unwrapped.variation_space, key.split("."))
            info[var_key] = subvar_space.value

        return obs, reward, terminated, truncated, info


class AddPixelsWrapper(gym.Wrapper):
    """Adds rendered environment pixels to info dict with optional resizing and transforms.

    Supports single images, dictionaries of images (multiview), or lists of images.
    Uses PIL for resizing and optional torchvision transforms.

    Args:
        env: The Gymnasium environment to wrap.
        pixels_shape: Target (height, width) for resized images. Defaults to (84, 84).
        torchvision_transform: Optional transform to apply to PIL images.

    Info Keys Added:
        - pixels: Rendered image (single view).
        - pixels.{key}: Individual images (multiview dict).
        - pixels.{idx}: Individual images (multiview list).
        - render_time: Time taken to render in seconds.
    """

    def __init__(
        self,
        env,
        pixels_shape: tuple[int, int] = (84, 84),  # (height, width)
        torchvision_transform: Callable | None = None,
    ):
        super().__init__(env)
        self.pixels_shape = pixels_shape
        self.torchvision_transform = torchvision_transform
        # For resizing, use PIL (required for torchvision transforms)
        from PIL import Image

        self.Image = Image

    def _get_pixels(self):
        # Render the environment as an RGB array
        render = getattr(self.env.unwrapped, "render_multiview", None)
        render = render if callable(render) else self.env.render

        t0 = time.time()
        img = render()
        t1 = time.time()

        def _process_img(img_array):
            # Convert to PIL Image for resizing
            pil_img = self.Image.fromarray(img_array)
            height, width = self.pixels_shape
            pil_img = pil_img.resize((width, height), self.Image.BILINEAR)
            # Optionally apply torchvision transform
            if self.torchvision_transform is not None:
                pixels = self.torchvision_transform(pil_img)
            else:
                pixels = np.array(pil_img)
            return pixels

        if isinstance(img, dict):
            pixels = {f"pixels.{k}": _process_img(v) for k, v in img.items()}
        elif isinstance(img, (list | tuple)):
            pixels = {f"pixels.{i}": _process_img(v) for i, v in enumerate(img)}
        else:
            pixels = {"pixels": _process_img(img)}

        return pixels, t1 - t0

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        pixels, info["render_time"] = self._get_pixels()
        info.update(pixels)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pixels, info["render_time"] = self._get_pixels()
        info.update(pixels)
        return obs, reward, terminated, truncated, info


class ResizeGoalWrapper(gym.Wrapper):
    """Resizes goal images in info dict with optional transforms.

    Applies PIL-based resizing and optional torchvision transforms to the 'goal'
    image in info dict during both reset and step.

    Args:
        env: The Gymnasium environment to wrap.
        pixels_shape: Target (height, width) for resized goal images. Defaults to (84, 84).
        torchvision_transform: Optional transform to apply to PIL goal images.
    """

    def __init__(
        self,
        env,
        pixels_shape: tuple[int, int] = (84, 84),  # (height, width)
        torchvision_transform: Callable | None = None,
    ):
        super().__init__(env)
        self.pixels_shape = pixels_shape
        self.torchvision_transform = torchvision_transform
        # For resizing, use PIL (required for torchvision transforms)
        from PIL import Image

        self.Image = Image

    def _format(self, img):
        # Convert to PIL Image for resizing
        pil_img = self.Image.fromarray(img)
        height, width = self.pixels_shape
        pil_img = pil_img.resize((width, height), self.Image.BILINEAR)
        # Optionally apply torchvision transform
        if self.torchvision_transform is not None:
            pixels = self.torchvision_transform(pil_img)
        else:
            pixels = np.array(pil_img)
        return pixels

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        info["goal"] = self._format(info["goal"])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["goal"] = self._format(info["goal"])
        return obs, reward, terminated, truncated, info


class StackedWrapper(gym.Wrapper):
    """Stacks specified key(s) in the info dict over the last k steps.

    The initial reset will fill the stack(s) with the initial value(s).

    Note:
        Stacked values are combined into a tensor/array along a new first
        dimension if the data type is a torch.Tensor or np.ndarray.

    Args:
        env: The Gymnasium environment to wrap.
        key: The key or list of keys in the info dict to stack.
        n_stacks: The number of steps to stack.
    """

    def __init__(
        self,
        env: gym.Env,
        key: str | list[str],
        n_stacks: int,
    ):
        super().__init__(env)
        self.keys = [key] if isinstance(key, str) else key
        self.n_stacks = n_stacks
        self.buffers: dict[str, deque] = {k: deque([], maxlen=n_stacks) for k in self.keys}

    def get_buffer_data(self, key: str):
        buffer = self.buffers[key]
        if not buffer:
            return []

        if self.n_stacks == 1:
            return buffer[0]

        new_info = list(buffer)
        first_elem = new_info[0]

        if torch.is_tensor(first_elem):
            return torch.stack(new_info, dim=0)
        elif isinstance(first_elem, np.ndarray):
            return np.stack(new_info, axis=0)
        else:
            return new_info

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        for k in self.keys:
            assert k in info, f"Key {k} not found in info dict during reset."
            data = info[k]
            buffer = self.buffers[k]
            buffer.clear()
            buffer.extend([data] * self.n_stacks)
            info[k] = self.get_buffer_data(k)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for k in self.keys:
            assert k in info, f"Key {k} not found in info dict during step."
            self.buffers[k].append(info[k])
            info[k] = self.get_buffer_data(k)
        return obs, reward, terminated, truncated, info


class MegaWrapper(gym.Wrapper):
    """Combines multiple wrappers for comprehensive environment preprocessing.

    Applies in sequence:
        AddPixelsWrapper → EverythingToInfoWrapper → EnsureInfoKeysWrapper →
        EnsureGoalInfoWrapper → ResizeGoalWrapper → StackedWrapper

    This provides a complete preprocessing pipeline with rendered pixels, unified
    info dict, key validation, goal checking, goal resizing, and temporal stacking.

    Args:
        env: The Gymnasium environment to wrap.
        image_shape: Target (height, width) for pixels and goal. Defaults to (84, 84).
        pixels_transform: Optional torchvision transform for rendered pixels.
        goal_transform: Optional torchvision transform for goal images.
        required_keys: Additional regex patterns for keys that must be in info.
            Pattern ``^pixels(?:\\..*)?$`` is always added.
        separate_goal: If True, validates 'goal' is present in info. Defaults to True.
        n_stacks: Number of steps to stack (passed to StackedWrapper).
    """

    def __init__(
        self,
        env,
        image_shape: tuple[int, int] = (84, 84),
        pixels_transform: Callable | None = None,
        goal_transform: Callable | None = None,
        required_keys: Iterable | None = None,
        separate_goal: Iterable = True,
        n_stacks: int = 1,
    ):
        super().__init__(env)

        if required_keys is None:
            required_keys = []
        required_keys.append(r"^pixels(?:\..*)?$")

        # Build pipeline
        # this adds `pixels` key to info with optional transform
        env = AddPixelsWrapper(env, image_shape, pixels_transform)
        # this removes the info output, everything is in observation!
        env = EverythingToInfoWrapper(env)
        # check that necessary keys are in the observation
        env = EnsureInfoKeysWrapper(env, required_keys)
        # check goal is provided
        env = EnsureGoalInfoWrapper(env, check_reset=separate_goal, check_step=separate_goal)
        env = ResizeGoalWrapper(env, image_shape, goal_transform)

        # We will wrap with StackedWrapper dynamically after we know the keys
        self.env = env
        self._n_stacks = n_stacks
        self._stack_initialized = False

    def _init_stack(self, info):
        """Attach a StackedWrapper around self.env dynamically."""
        keys = list(info.keys())
        self.env = StackedWrapper(self.env, keys, self._n_stacks)
        self._stack_initialized = True

        # Initialize buffers manually from the current info
        for k in self.env.keys:
            buf = self.env.buffers[k]
            buf.clear()
            buf.extend([info[k]] * self.env.n_stacks)
            info[k] = self.env.get_buffer_data(k)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)

        if not self._stack_initialized:
            self._init_stack(info)

        return obs, info

    def step(self, action):
        if not self._stack_initialized:
            raise RuntimeError("StackedWrapper not yet initialized — call reset() first.")
        return self.env.step(action)


class VariationWrapper(VectorWrapper):
    """Manages variation spaces for vectorized environments.

    Handles batching of variation spaces across multiple environments, supporting
    either shared variations (same) or independent variations (different).

    Args:
        env: The vectorized Gymnasium environment to wrap.
        variation_mode: Mode for handling variations across environments:
            - "same": All environments share the same variation space (batched).
            - "different": Each environment has independent variation spaces.

    Raises:
        ValueError: If variation_mode is invalid or sub-environment spaces don't match.

    Note:
        Base environment must have a ``variation_space`` attribute. If missing,
        variation spaces are set to None.
    """

    def __init__(
        self,
        env,
        variation_mode: str | gym.Space = "same",
    ):
        super().__init__(env)

        base_env = env.envs[0].unwrapped

        if not hasattr(base_env, "variation_space"):
            self.single_variation_space = None
            self.variation_space = None
            return

        if variation_mode == "same":
            self.single_variation_space = base_env.variation_space
            self.variation_space = batch_space(self.single_variation_space, self.num_envs)

        elif variation_mode == "different":
            self.single_variation_space = base_env.variation_space
            self.variation_space = batch_differing_spaces([sub_env.unwrapped.variation_space for sub_env in env.envs])

        else:
            raise ValueError(
                f"Invalid `variation_mode`, expected: 'same' or 'different' or tuple of single and batch variation space, actual got {variation_mode}"
            )

        # check sub-environment obs and action spaces
        for sub_env in env.envs:
            if variation_mode == "same":
                if not is_space_dtype_shape_equiv(sub_env.unwrapped.observation_space, self.single_observation_space):
                    raise ValueError(
                        f"VariationWrapper(..., variation_mode='same') however the sub-environments observation spaces do not share a common shape and dtype, single_observation_space={self.single_observation_space}, sub-environment observation_space={sub_env.observation_space}"
                    )
            else:
                if not is_space_dtype_shape_equiv(sub_env.unwrapped.observation_space, self.single_observation_space):
                    raise ValueError(
                        f"VariationWrapper(..., variation_mode='different' or custom space) however the sub-environments observation spaces do not share a common shape and dtype, single_observation_space={self.single_observation_space}, sub-environment observation_space={sub_env.observation_space}"
                    )

    @property
    def envs(self):
        return getattr(self.env, "envs", None)
