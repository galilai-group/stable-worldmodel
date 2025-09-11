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
import gymnasium as gym
from gymnasium.vector import VectorWrapper


class EnsureInfoKeysWrapper(gym.Wrapper):
    """
    Gymnasium wrapper to ensure certain keys are present in the info dict.
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

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for key in self.required_keys:
            if key not in info:
                raise RuntimeError(f"Key {key} is not present in the env output")
        return obs, info


# class VecEnsureInfoKeysWrapper(VectorWrapper):
#     """
#     Vectorized Gymnasium wrapper that ensures specific keys are present in each observation dict.
#     If a key is missing, it is added with a default value.
#     """

#     def __init__(self, env, required_keys):
#         super().__init__(env)
#         self.required_keys = required_keys

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         self._ensure_keys(info)
#         return obs, info

#     def step(self, actions):
#         obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
#         self._ensure_keys(infos)
#         return obs, rewards, terminateds, truncateds, infos

#     def _ensure_keys(self, infos):
#         # obs: dict of arrays (standard for Dict spaces)
#         for info in infos:
#             for key in self.required_keys:
#                 if key not in info:
#                     raise RuntimeError(f"Key {key} is not present in the env output")
#         else:
#             raise RuntimeError("Obs should be a dict")


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
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info[self.image_key].shape[:-1] != self.image_shape:
            raise RuntimeError(
                f"Image shape {info[self.image_key].shape} should be {self.image_shape}"
            )
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if info[self.image_key].shape[:-1] != self.image_shape:
            raise RuntimeError(
                f"Image shape {info[self.image_key].shape} should be {self.image_shape}"
            )
        return obs, info


class EnsureGoalInfoWrapper(gym.Wrapper):
    def __init__(self, env, check_reset, check_step: bool = False):
        super().__init__(env)
        self.check_reset = check_reset
        self.check_step = check_step

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.check_reset and "goal" not in info:
            raise RuntimeError(
                "The info dict returned by reset() must contain the key 'goal'."
            )
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.check_step and "goal" not in info:
            raise RuntimeError(
                "The info dict returned by step() must contain the key 'goal'."
            )
        return obs, reward, terminated, truncated, info


class EverythingToInfoWrapper(gym.Wrapper):
    """
    Gymnasium wrapper to ensure the observation is included in the info dict
    under a specified key after reset and step.
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if type(obs) is not dict:
            _obs = {"observation": obs}
        else:
            _obs = obs
        for key in _obs:
            assert key not in info
            info[key] = _obs[key]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
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
        info["terminated"] = reward
        assert "truncated" not in info
        info["truncated"] = reward
        assert "action" not in info
        info["action"] = reward
        return obs, reward, terminated, truncated, info


# class VecEverythingToInfoWrapper(VectorWrapper):
#     """
#     Vectorized Gymnasium wrapper that merges each info dict into its corresponding observation dict.
#     Assumes observations are dicts.
#     """

#     def reset(self, **kwargs):
#         obs, infos = self.env.reset(**kwargs)
#         self._merge_info(obs, infos)
#         return obs, infos

#     def step(self, actions):
#         obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
#         self._merge_info(obs, infos)
#         return obs, rewards, terminateds, truncateds, infos

#     def _merge_info(self, obs, infos):

#         # We'll assume obs is a dict of arrays (standard for Dict spaces)
#         if isinstance(obs, dict):
#             # Convert dict of arrays to list of dicts
#             obs_list = [
#                 dict((k, v[i]) for k, v in obs.items()) for i in range(len(infos))
#             ]
#         else:
#             # Already a list of dicts
#             if type(obs[0]) is not dict:
#                 obs = [{"observation": o} for o in obs]
#             obs_list = list(obs)
#         # Merge info into obs for each env
#         for i, info in enumerate(infos):
#             info.update(obs_list[i])


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
        pixels_shape: Tuple[int, int] = (84, 84),
        torchvision_transform: Optional[Callable] = None,
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

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["goal"] = self._format(info["goal"])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["goal"] = self._format(info["goal"])
        return obs, reward, terminated, truncated, info


# class VecAddPixelsWrapper(gym.vector.VectorWrapper):
#     """
#     Vectorized wrapper that adds a 'pixels' key to each info dict,
#     containing a rendered and resized image of each environment.
#     """

#     def __init__(
#         self,
#         env,
#         pixels_shape: Tuple[int, int] = (84, 84),
#         torchvision_transform: Optional[Callable] = None,
#     ):
#         super().__init__(env)
#         self.pixels_shape = pixels_shape
#         self.torchvision_transform = torchvision_transform
#         from PIL import Image

#         self.Image = Image

#     def _get_pixels(self):
#         # env.envs is a list of the underlying environments
#         pixels_list = []
#         for e in self.env.envs:
#             img = e.render()
#             pil_img = self.Image.fromarray(img)
#             pil_img = pil_img.resize(self.pixels_shape, self.Image.BILINEAR)
#             if self.torchvision_transform is not None:
#                 pixels = self.torchvision_transform(pil_img)
#             else:
#                 pixels = np.array(pil_img)
#             pixels_list.append(pixels)
#         return pixels_list

#     def reset(self, **kwargs):
#         obs, infos = self.env.reset(**kwargs)
#         pixels_list = self._get_pixels()
#         for info, pixels in zip(infos, pixels_list):
#             info["pixels"] = pixels
#         return obs, infos

#     def step(self, actions):
#         obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
#         pixels_list = self._get_pixels()
#         for info, pixels in zip(infos, pixels_list):
#             info["pixels"] = pixels
#         return obs, rewards, terminateds, truncateds, infos


class MegaWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        image_shape: Tuple[int, int] = (84, 84),
        pixels_transform: Optional[Callable] = None,
        goal_transform: Optional[Callable] = None,
        required_keys: Optional[Iterable] = None,
        separate_goal: Optional[Iterable] = True,
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
        env = EnsureGoalInfoWrapper(
            env, check_reset=separate_goal, check_step=separate_goal
        )
        self.env = ResizeGoalWrapper(env, image_shape, goal_transform)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


# class VecMegaWrapper(gym.Wrapper):
#     def __init__(
#         self,
#         env,
#         pixels_shape: Tuple[int, int] = (84, 84),
#         torchvision_transform: Optional[Callable] = None,
#         required_keys: Optional[Iterable] = None,
#     ):
#         super().__init__(env)
#         if required_keys is None:
#             required_keys = []
#         required_keys.append("pixels")

#         # this adds `pixels` key to info with optional transform
#         env = VecAddPixelsWrapper(env, pixels_shape, torchvision_transform)
#         # this removes the info output, everything is in observation!
#         env = VecEverythingToInfoWrapper(env)
#         # check that necessary keys are in the observation
#         env = VecEnsureInfoKeysWrapper(env, required_keys)

#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)

#     def step(self, action):
#         return self.env.step(action)


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
