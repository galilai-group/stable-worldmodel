import gymnasium as gym
from gymnasium.wrappers import (
    RecordVideo,
    AddRenderObservation,
    ResizeObservation,
    TransformObservation,
    NumpyToTorch,
    TimeLimit,
)
import torchvision.transforms.v2 as transforms
import torch
import numpy as np
from typing import Optional, Callable, Tuple, Iterable
import time


class EnsureObservationKeys(gym.Wrapper):
    """
    Gymnasium wrapper to ensure certain keys are present in the info dict.
    If a key is missing, it is added with a default value.
    """

    def __init__(self, env, required_keys):
        super().__init__(env)
        self.required_keys = required_keys

    def step(self, action):
        obs, reward, terminated, truncated = self.env.step(action)
        for key in self.required_keys:
            if key not in obs:
                raise RuntimeError(f"Key {key} is not present in the env output")
        return obs, reward, terminated, truncated

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs = result
            for key in self.required_keys:
                if key not in obs:
                    raise RuntimeError(f"Key {key} is not present in the env output")
            return obs
        else:
            raise RuntimeError("The output of the env should be a 2-element tuple")


class EnsureImageShape(gym.Wrapper):
    """
    Gymnasium wrapper to ensure certain keys are present in the info dict.
    If a key is missing, it is added with a default value.
    """

    def __init__(self, env, image_key, image_shape):
        super().__init__(env)
        self.image_key = image_key
        self.image_shape = image_shape

    def step(self, action):
        obs, reward, terminated, truncated = self.env.step(action)
        if obs[self.image_key].shape != self.image_shape:
            raise RuntimeError(
                f"Image shape {obs[self.image_key].shape} should be {self.image_shape}"
            )
        return obs, reward, terminated, truncated

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs = result

            if obs[self.image_key].shape != self.image_shape:
                raise RuntimeError(
                    f"Image shape {obs[self.image_key].shape} should be {self.image_shape}"
                )
            return obs
        else:
            raise RuntimeError("The output of the env should be a 2-element tuple")


class InfoInObservation(gym.Wrapper):
    """
    Gymnasium wrapper to ensure the observation is included in the info dict
    under a specified key after reset and step.
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for key in info:
            assert key not in obs
            obs[key] = info[key]
        return info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for key in info:
            assert key not in obs
            obs[key] = info[key]
        return obs, reward, terminated, truncated


class AddPixelsWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that adds a 'pixels' key to the info dict,
    containing a rendered and resized image of the environment.
    Optionally applies a torchvision transform to the image.
    Optionally applies another user-supplied wrapper to the environment.
    """

    def __init__(
        self,
        env,
        pixels_shape: Tuple[int, int] = (84, 84),
        torchvision_transform: Optional[Callable] = None,
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

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["pixels"], info["render_time"] = self._get_pixels()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["pixels"], info["render_time"] = self._get_pixels()
        return obs, reward, terminated, truncated, info


class MegaWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        pixels_shape: Tuple[int, int] = (84, 84),
        torchvision_transform: Optional[Callable] = None,
        required_keys: Optional[Iterable] = None,
    ):
        if required_keys is None:
            required_keys = []
        required_keys.append("pixels")

        # this adds `pixels` key to info with optional transform
        env = AddPixelsWrapper(env, pixels_shape, torchvision_transform)
        # this removes the info output, everything is in observation!
        env = InfoInObservation(env)
        # check that necessary keys are in the observation
        env = EnsureObservationKeys(env, required_keys)
        # sanity check image shape
        self.env = EnsureImageShape(env, "pixels", pixels_shape)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


class TransformObservation(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        transform=None,
        image_shape=(3, 224, 224),
        mean=None,
        std=None,
        source_key="pixels",
        target_key="pixels",
    ):
        super(TransformObservation, self).__init__(env)
        self.source_key = source_key
        self.target_key = target_key

        assert len(image_shape) == 3
        if transform:
            self.transform = transform
        else:
            if image_shape[0] == 3:
                t = [transforms.RGB()]
            elif image_shape[0] == 1:
                t = [transforms.Grayscale()]
            t.extend(
                [
                    transforms.Resize((224, 224)),  # Resize the image
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float, scale=True),
                ]
            )
            if mean is not None and std is not None:
                t.append(transforms.Normalize(mean=mean, std=std))
            t = transforms.Compose(t)
            self.transform = t
        # Update the observation space to include the new transformed observation
        original_space = self.observation_space
        transformed_space = gym.spaces.Box(
            low=0, high=1, shape=image_shape, dtype=np.float32
        )
        original_space[self.source_key] = transformed_space
        self.observation_space = original_space

    def observation(self, observation):
        pixels = torch.from_numpy(observation[self.source_key].copy())
        pixels = pixels.permute(2, 0, 1)
        pixels = self.transform(pixels)
        observation[self.target_key] = pixels
        return observation
