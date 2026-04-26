import re
import time
from collections.abc import Callable, Iterable
from typing import Any
from collections.abc import Sequence

import gymnasium as gym
import numpy as np

from stable_worldmodel.utils import get_in


class EnsureInfoKeysWrapper(gym.Wrapper):
    """Validates that required keys are present in the info dict."""

    def __init__(self, env: gym.Env, required_keys: Iterable[str]):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            required_keys: Iterable of regex patterns that must match keys in info.
        """
        super().__init__(env)
        self._patterns: list[re.Pattern] = []
        for k in required_keys:
            self._patterns.append(re.compile(k))

    def _check(self, info: dict, where: str) -> None:
        """Check if all required patterns have at least one match in info.

        Args:
            info: The info dictionary to check.
            where: String indicating where the check is performed (e.g., "reset").

        Raises:
            RuntimeError: If any required pattern is missing from info.
        """
        keys = list(info.keys())
        missing = [
            p.pattern
            for p in self._patterns
            if not any(p.fullmatch(k) for k in keys)
        ]
        if missing:
            raise RuntimeError(
                f'{where}: required info keys missing (patterns with no match): {missing}. Present keys: {keys}'
            )

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform environment step and validate info keys.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._check(info, 'step()')
        return obs, reward, terminated, truncated, info

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset environment and validate info keys.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.
        """
        obs, info = self.env.reset(*args, **kwargs)
        self._check(info, 'reset()')
        return obs, info


class EnsureImageShape(gym.Wrapper):
    """Validates that an image in the info dict has the expected spatial dimensions."""

    def __init__(
        self, env: gym.Env, image_key: str, image_shape: tuple[int, int]
    ):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            image_key: Key in info dict containing the image.
            image_shape: Expected (height, width) of the image.
        """
        super().__init__(env)
        self.image_key = image_key
        self.image_shape = image_shape  # (height, width)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform step and validate image shape.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.

        Raises:
            RuntimeError: If image shape does not match expected shape.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info[self.image_key].shape[:-1] != self.image_shape:
            raise RuntimeError(
                f'Image shape {info[self.image_key].shape} should be {self.image_shape}'
            )
        return obs, reward, terminated, truncated, info

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset and validate image shape.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.

        Raises:
            RuntimeError: If image shape does not match expected shape.
        """
        obs, info = self.env.reset(*args, **kwargs)
        if info[self.image_key].shape[:-1] != self.image_shape:
            raise RuntimeError(
                f'Image shape {info[self.image_key].shape} should be {self.image_shape}'
            )
        return obs, info


class EnsureGoalInfoWrapper(gym.Wrapper):
    """Validates that 'goal' key is present in info dict."""

    def __init__(
        self, env: gym.Env, check_reset: bool, check_step: bool = False
    ):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            check_reset: Whether to check 'goal' presence on reset.
            check_step: Whether to check 'goal' presence on each step.
        """
        super().__init__(env)
        self.check_reset = check_reset
        self.check_step = check_step

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset and validate goal presence.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.

        Raises:
            RuntimeError: If 'goal' is missing and check_reset is True.
        """
        obs, info = self.env.reset(*args, **kwargs)
        if self.check_reset and 'goal' not in info:
            raise RuntimeError(
                "The info dict returned by reset() must contain the key 'goal'."
            )
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform step and validate goal presence.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.

        Raises:
            RuntimeError: If 'goal' is missing and check_step is True.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.check_step and 'goal' not in info:
            raise RuntimeError(
                "The info dict returned by step() must contain the key 'goal'."
            )
        return obs, reward, terminated, truncated, info


class EverythingToInfoWrapper(gym.Wrapper):
    """Moves all transition information into the info dict."""

    def __init__(self, env: gym.Env):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)
        self._variations_watch: Sequence[str] = []
        self._step_counter = 0
        self._id = 0

    def _gen_id(self) -> int:
        """Generate a random unique identifier for the current episode.

        Returns:
            A random 64-bit integer.
        """
        max_int = np.iinfo(np.int64).max
        rng = self.env.unwrapped.np_random
        return int(
            rng.integers(0, max_int)
            if hasattr(rng, 'integers')
            else rng.randint(0, max_int)
        )

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset environment and move all data to info.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.
        """
        self._step_counter = 0
        obs, info = self.env.reset(*args, **kwargs)
        if not isinstance(obs, dict):
            _obs = {'observation': obs}
        else:
            _obs = obs

        for key, val in _obs.items():
            assert key not in info
            info[key] = val

        assert 'reward' not in info
        info['reward'] = np.nan
        assert 'terminated' not in info
        info['terminated'] = False
        assert 'truncated' not in info
        info['truncated'] = False
        assert 'action' not in info
        info['action'] = self.env.action_space.sample()
        assert 'step_idx' not in info
        info['step_idx'] = self._step_counter
        assert 'id' not in info
        self._id = self._gen_id()
        info['id'] = self._id

        # add all variations to info if needed
        options = kwargs.get('options') or {}

        if 'variation' in options:
            var_opt = options['variation']
            assert isinstance(var_opt, list | tuple), (
                'variation option must be a list or tuple containing variation names to sample, found: '
                f'{type(var_opt)}'
            )
            if len(var_opt) == 1 and var_opt[0] == 'all':
                self._variations_watch = (
                    self.env.unwrapped.variation_space.names()
                )
            else:
                self._variations_watch = var_opt

        for key in self._variations_watch:
            var_key = f'variation.{key}'
            assert var_key not in info
            subvar_space = get_in(
                self.env.unwrapped.variation_space, key.split('.')
            )
            info[var_key] = subvar_space.value

        if isinstance(info['action'], dict):
            raise NotImplementedError
        else:
            info['action'] = np.full_like(info['action'], np.nan)
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform step and move all data to info.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_counter += 1
        if not isinstance(obs, dict):
            _obs = {'observation': obs}
        else:
            _obs = obs
        for key, val in _obs.items():
            assert key not in info
            info[key] = val
        assert 'reward' not in info
        info['reward'] = reward
        assert 'terminated' not in info
        info['terminated'] = bool(terminated)
        assert 'truncated' not in info
        info['truncated'] = bool(truncated)
        assert 'action' not in info
        info['action'] = action
        assert 'step_idx' not in info
        info['step_idx'] = self._step_counter
        assert 'id' not in info
        info['id'] = self._id

        for key in self._variations_watch:
            var_key = f'variation.{key}'
            assert var_key not in info
            subvar_space = get_in(
                self.env.unwrapped.variation_space, key.split('.')
            )
            info[var_key] = subvar_space.value

        return obs, reward, terminated, truncated, info


class AddPixelsWrapper(gym.Wrapper):
    """Adds rendered environment pixels to info dict."""

    def __init__(
        self,
        env: gym.Env,
        pixels_shape: tuple[int, int] = (84, 84),  # (height, width)
        torchvision_transform: Callable[[Any], Any] | None = None,
        resample: int | None = None,
    ):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            pixels_shape: Target (height, width) for rendered pixels.
            torchvision_transform: Optional transform to apply to the pixels.
            resample: PIL resample filter (e.g. ``Image.BILINEAR``,
                ``Image.NEAREST``). Defaults to BILINEAR.
        """
        super().__init__(env)
        self.pixels_shape = pixels_shape
        self.torchvision_transform = torchvision_transform
        # For resizing, use PIL (required for torchvision transforms)
        from PIL import Image

        self.Image = Image
        self.resample = resample if resample is not None else Image.BILINEAR

    def _get_pixels(self) -> tuple[dict[str, np.ndarray], float]:
        """Render environment and process pixels.

        Returns:
            A tuple of (pixels dictionary, render time).
        """
        # Render the environment as an RGB array
        render = getattr(self.env.unwrapped, 'render_multiview', None)
        render_fn = render if callable(render) else self.env.render

        t0 = time.time()
        img = render_fn()
        t1 = time.time()

        def _process_img(img_array: np.ndarray) -> np.ndarray:
            # Convert to PIL Image for resizing
            pil_img = self.Image.fromarray(img_array)
            height, width = self.pixels_shape
            pil_img = pil_img.resize((width, height), self.resample)
            # Optionally apply torchvision transform
            if self.torchvision_transform is not None:
                pixels = self.torchvision_transform(pil_img)
            else:
                pixels = np.array(pil_img)
            return pixels

        if isinstance(img, dict):
            pixels = {f'pixels.{k}': _process_img(v) for k, v in img.items()}
        elif isinstance(img, (list | tuple)):
            pixels = {
                f'pixels.{i}': _process_img(v) for i, v in enumerate(img)
            }
        else:
            pixels = {'pixels': _process_img(img)}

        return pixels, t1 - t0

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset environment and add pixels to info.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.
        """
        obs, info = self.env.reset(*args, **kwargs)
        pixels, info['render_time'] = self._get_pixels()
        info.update(pixels)
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform step and add pixels to info.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        pixels, info['render_time'] = self._get_pixels()
        info.update(pixels)
        return obs, reward, terminated, truncated, info


class ResizeGoalWrapper(gym.Wrapper):
    """Resizes goal images in info dict."""

    def __init__(
        self,
        env: gym.Env,
        pixels_shape: tuple[int, int] = (84, 84),  # (height, width)
        torchvision_transform: Callable[[Any], Any] | None = None,
        resample: int | None = None,
    ):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            pixels_shape: Target (height, width) for resizing goal images.
            torchvision_transform: Optional transform to apply to goal images.
            resample: PIL resample filter (e.g. ``Image.BILINEAR``,
                ``Image.NEAREST``). Defaults to BILINEAR.
        """
        super().__init__(env)
        self.pixels_shape = pixels_shape
        self.torchvision_transform = torchvision_transform
        # For resizing, use PIL (required for torchvision transforms)
        from PIL import Image

        self.Image = Image
        self.resample = resample if resample is not None else Image.BILINEAR

    def _format(self, img: np.ndarray) -> np.ndarray:
        """Resize and transform a goal image.

        Args:
            img: The original goal image as a numpy array.

        Returns:
            The processed goal image.
        """
        # Convert to PIL Image for resizing
        pil_img = self.Image.fromarray(img)
        height, width = self.pixels_shape
        pil_img = pil_img.resize((width, height), self.resample)
        # Optionally apply torchvision transform
        if self.torchvision_transform is not None:
            pixels = self.torchvision_transform(pil_img)
        else:
            pixels = np.array(pil_img)
        return pixels

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset environment and format goal image.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.
        """
        obs, info = self.env.reset(*args, **kwargs)
        if 'goal' in info:
            info['goal'] = self._format(info['goal'])
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform step and format goal image.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        if 'goal' in info:
            info['goal'] = self._format(info['goal'])
        return obs, reward, terminated, truncated, info


_RESAMPLE_ALIASES = {
    'nearest': 'NEAREST',
    'bilinear': 'BILINEAR',
    'bicubic': 'BICUBIC',
    'lanczos': 'LANCZOS',
    'box': 'BOX',
    'hamming': 'HAMMING',
}


def _resolve_resample(resample: str | int | None) -> int | None:
    if resample is None or isinstance(resample, int):
        return resample
    from PIL import Image

    key = resample.lower()
    if key not in _RESAMPLE_ALIASES:
        raise ValueError(
            f'Unknown resample mode {resample!r}; '
            f'choose from {sorted(_RESAMPLE_ALIASES)}.'
        )
    return getattr(Image, _RESAMPLE_ALIASES[key])


class MegaWrapper(gym.Wrapper):
    """Combines multiple wrappers for comprehensive environment preprocessing."""

    def __init__(
        self,
        env: gym.Env,
        image_shape: tuple[int, int] = (84, 84),
        pixels_transform: Callable[[Any], Any] | None = None,
        goal_transform: Callable[[Any], Any] | None = None,
        required_keys: Iterable[str] | None = None,
        separate_goal: bool = True,
        image_resample: str | int | None = None,
    ):
        """Initialize the mega wrapper pipeline.

        Args:
            env: The environment to wrap.
            image_shape: Target (height, width) for all image processing.
            pixels_transform: Optional transform for rendered pixels.
            goal_transform: Optional transform for goal images.
            required_keys: Keys that must be present in info dict.
            separate_goal: Whether to handle goal separately.
            image_resample: PIL resample mode used when resizing pixels and
                goal images. Accepts a PIL constant or a string in
                ``{'nearest','bilinear','bicubic','lanczos','box','hamming'}``.
                Defaults to bilinear. Use ``'nearest'`` for crisp pixel art.
        """
        super().__init__(env)

        req_keys = list(required_keys) if required_keys is not None else []
        req_keys.append(r'^pixels(?:\..*)?$')

        resample = _resolve_resample(image_resample)

        # this adds `pixels` key to info with optional transform
        env = AddPixelsWrapper(env, image_shape, pixels_transform, resample)
        # this removes the info output, everything is in observation!
        env = EverythingToInfoWrapper(env)
        # check that necessary keys are in the observation
        env = EnsureInfoKeysWrapper(env, req_keys)
        env = ResizeGoalWrapper(env, image_shape, goal_transform, resample)

        self.env = env
