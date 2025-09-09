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


class EnsureInfoKeys(gym.Wrapper):
    """
    Gymnasium wrapper to ensure certain keys are present in the info dict.
    If a key is missing, it is added with a default value.
    """

    def __init__(self, env, required_keys, default_value=None):
        super().__init__(env)
        self.required_keys = required_keys
        self.default_value = default_value

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for key in self.required_keys:
            if key not in info:
                info[key] = self.default_value
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            for key in self.required_keys:
                if key not in info:
                    info[key] = self.default_value
            return obs, info
        else:
            # For older Gymnasium versions or custom envs
            return result


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
