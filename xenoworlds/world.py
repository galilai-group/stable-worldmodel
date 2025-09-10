import gymnasium as gym
import numpy as np
import torch
from loguru import logger as logging
from torchvision import transforms
from .wrappers import MegaWrapper
from pathlib import Path


class World:
    def __init__(
        self,
        env_name,
        num_envs,
        image_shape,
        image_transform=None,
        goal_wrappers: list = None,
        seed: int = 2349867,
        max_episode_steps: int = 100,
        sample_goal_every_k_steps: int = -1,
        **kwargs,
    ):
        self.envs = gym.make_vec(
            env_name,
            num_envs=num_envs,
            vectorization_mode="sync",
            wrappers=[lambda x: MegaWrapper(x, image_shape, image_transform)],
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

        self.goal_envs = gym.make_vec(
            env_name,
            num_envs=num_envs,
            vectorization_mode="sync",
            wrappers=[lambda x: MegaWrapper(x, image_shape, image_transform)],
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

        logging.info("WORLD INITIALIZED")
        logging.info(f"ACTION SPACE: {self.envs.action_space}")
        logging.info(f"OBSERVATION SPACE: {self.envs.observation_space}")
        self.num_envs = num_envs
        self.seed = seed

        rng = torch.Generator()
        rng.manual_seed(seed)
        self.goal_seed = torch.randint(0, 2**32 - 1, (1,), generator=rng).item()

        # note if sample_goal_every_k_steps is set to -1, will sample goal once per episode
        # TODO implement sample_goal_every_k_steps

    @property
    def observation_space(self):
        return self.envs.observation_space

    @property
    def action_space(self):
        return self.envs.action_space

    @property
    def single_action_space(self):
        return self.envs.single_action_space

    @property
    def single_observation_space(self):
        return self.envs.single_observation_space

    def close(self, **kwargs):
        return self.envs.close(**kwargs)

    # TEMOPORARY, need to delete!!!
    def denormalize(self, x):
        # x is (B,C,H,W) in [-1,1]
        return (x * 0.5) + 0.5

    def __iter__(self):
        self.terminations = np.array([False] * self.num_envs)
        self.truncations = np.array([False] * self.num_envs)
        self.rewards = None
        logging.info(f"Resetting the ({self.num_envs}) world(s)!")
        self.states, self.infos = self.envs.reset(seed=self.seed)
        self.cur_goals = self.infos["goal"]
        self.cur_goal_images = self.infos["goal_image"]
        return self

    def __next__(self):
        if not all(self.terminations) and not all(self.truncations):
            return self.states, self.infos
        else:
            raise StopIteration

    def step(self):
        actions = self.policy.get_action(self.infos)
        (self.states, self.rewards, self.terminateds, self.truncateds, self.infos) = (
            self.envs.step(actions)
        )

    def reset(self, seed=None, options=None):
        self.states, self.infos = self.envs.reset(seed=seed, options=options)

    def set_policy(self, policy):
        self.policy = policy
        self.policy.set_env(self.envs)

    def record_video(self, video_path, max_steps=500, fps=30, seed=None, options=None):
        """
        Records a video of a random policy running in the first environment of a Gymnasium VecEnv.
        Args:
            video_path: Output path for the video file.
            max_steps: Maximum number of steps to record.
            fps: Frames per second for the video.
        """
        import imageio

        out = [
            imageio.get_writer(
                Path(video_path) / f"env_{i}.mp4",
                "output.mp4",
                fps=fps,
                codec="libx264",
            )
            for i in range(self.envs.num_envs)
        ]

        self.reset(seed, options)
        for i, o in enumerate(out):
            o.append_data(self.infos["pixels"][i])
        for _ in range(max_steps):
            self.step()
            for i, o in enumerate(out):
                o.append_data(self.infos["pixels"][i])
            if np.any(self.terminateds) or np.any(self.truncateds):
                break
        [o.close() for o in out]
        print(f"Video saved to {video_path}")
