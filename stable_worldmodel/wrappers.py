import time
from collections.abc import Callable, Iterable

import gymnasium as gym
import numpy as np
from gymnasium.spaces.utils import is_space_dtype_shape_equiv
from gymnasium.vector import VectorWrapper
from gymnasium.vector.utils import (
    batch_differing_spaces,
    batch_space,
)

from stable_worldmodel.utils import get_in


class EnsureInfoKeysWrapper(gym.Wrapper):
    """Gymnasium wrapper to ensure certain keys are present in the info dict.
    If a key is missing, it is added with a default value.
    """

    def __init__(self, env, required_keys):
        super().__init__(env)
        self.required_keys = required_keys

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for key in self.required_keys:
            if key not in info:
                raise RuntimeError(f"Key {key} is not present in the env output")
        return obs, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        for key in self.required_keys:
            if key not in info:
                raise RuntimeError(f"Key {key} is not present in the env output")
        return obs, info


class EnsureImageShape(gym.Wrapper):
    """Gymnasium wrapper to ensure certain keys are present in the info dict.
    If a key is missing, it is added with a default value.
    """

    def __init__(self, env, image_key, image_shape):
        super().__init__(env)
        self.image_key = image_key
        self.image_shape = image_shape

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
    """Gymnasium wrapper to ensure the observation is included in the info dict
    under a specified key after reset and step.
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
    """Gymnasium wrapper that adds a 'pixels' key to the info dict,
    containing a rendered and resized image of the environment.
    Optionally applies a torchvision transform to the image.
    Optionally applies another user-supplied wrapper to the environment.
    """

    def __init__(
        self,
        env,
        pixels_shape: tuple[int, int] = (84, 84),
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
        t0 = time.time()
        img = self.env.render()
        t1 = time.time()
        # Convert to PIL Image for resizing
        pil_img = self.Image.fromarray(img)
        pil_img = pil_img.resize(self.pixels_shape, self.Image.BILINEAR)
        # Optionally apply torchvision transform
        if self.torchvision_transform is not None:
            pixels = self.torchvision_transform(pil_img)
        else:
            pixels = np.array(pil_img)
        return pixels, t1 - t0

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        info["pixels"], info["render_time"] = self._get_pixels()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["pixels"], info["render_time"] = self._get_pixels()
        return obs, reward, terminated, truncated, info


class ResizeGoalWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        pixels_shape: tuple[int, int] = (84, 84),
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
        pil_img = pil_img.resize(self.pixels_shape, self.Image.BILINEAR)
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


class MegaWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        image_shape: tuple[int, int] = (84, 84),
        pixels_transform: Callable | None = None,
        goal_transform: Callable | None = None,
        required_keys: Iterable | None = None,
        separate_goal: Iterable | None = True,
    ):
        super().__init__(env)
        if required_keys is None:
            required_keys = []
        required_keys.append("pixels")

        # this adds `pixels` key to info with optional transform
        env = AddPixelsWrapper(env, image_shape, pixels_transform)
        # this removes the info output, everything is in observation!
        env = EverythingToInfoWrapper(env)
        # check that necessary keys are in the observation
        env = EnsureInfoKeysWrapper(env, required_keys)
        # check goal is provided
        env = EnsureGoalInfoWrapper(env, check_reset=separate_goal, check_step=separate_goal)
        self.env = ResizeGoalWrapper(env, image_shape, goal_transform)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        return self.env.step(action)


class VariationWrapper(VectorWrapper):
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
