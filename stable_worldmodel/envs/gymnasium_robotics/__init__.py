import gymnasium as gym
import gymnasium_robotics

from .fetch import FetchWrapper


gym.register_envs(gymnasium_robotics)

__all__ = ['FetchWrapper']
