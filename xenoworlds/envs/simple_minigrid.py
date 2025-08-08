from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from minigrid.wrappers import (
    RGBImgObsWrapper,
    FullyObsWrapper,
    ImgObsWrapper,
    StochasticActionWrapper,
)
from stable_baselines3 import PPO
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
import torch
from torch import nn
import gymnasium as gym
from gymnasium.wrappers import (
    RecordEpisodeStatistics,
    RecordVideo,
    FrameStackObservation,
    AddRenderObservation,
    ResizeObservation,
)

from stable_baselines3.common.evaluation import evaluate_policy
import optimalssl as ossl
import numpy as np
from collections import deque
import multiprocessing
import torchvision

# Each tile is encoded as a 3 dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE)

# OBJECT_TO_IDX = {
#     "unseen": 0,
#     "empty": 1,
#     "wall": 2,
#     "floor": 3,
#     "door": 4,
#     "key": 5,
#     "ball": 6,
#     "box": 7,
#     "goal": 8,
#     "lava": 9,
#     "agent": 10,
# }
# STATE_TO_IDX = {
#     "open": 0,
#     "closed": 1,
#     "locked": 2,
# }


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)
        print(observation_space.spaces.items())
        n_input_channels = observation_space.shape[0]
        self.cnn = ossl.module.Resnet9(512, num_channels=n_input_channels)
        self.linear = nn.Sequential(nn.Linear(512, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class BFSPlanner(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)
        print(observation_space.spaces.items())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        assert "image" in observations
        # we need to go from 8 to 10 avoid walls (2)
        maze = observations["image"][:, :, 0]
        start = tuple(np.concatenate(np.nonzero(maze == 10)).tolist())
        end = tuple(np.concatenate(np.nonzero(maze == 8)).tolist())
        maze = maze == 2
        rows, cols = len(maze), len(maze[0])

        queue = deque([(start, [])])  # Store (current cell, path to reach it)
        visited = {tuple(start)}
        ij_directions = np.stack([(1, 0), (0, 1), (-1, 0), (0, -1)])

        while queue:
            (row, col), actions = queue.popleft()

            if (row, col) == end:
                optimum = actions[0]
                # 0 = North, 1 = East, 2 = South, 3 = West
                direction = observations["direction"]

                # action: 0=left, 1=right, 2=forward
                if np.equal(optimum, ij_directions[direction]).all():
                    return 2
                elif np.equal(optimum, ij_directions[(direction - 1) % 4]).all():
                    return 0
                elif np.equal(optimum, ij_directions[(direction + 1) % 4]).all():
                    return 1
                else:
                    return np.random.randint(3)

            # Explore adjacent cells (up, down, left, right)
            for dr, dc in ij_directions:
                new_row, new_col = row + dr, col + dc

                # Check boundaries and obstacles
                if (
                    0 <= new_row < rows
                    and 0 <= new_col < cols
                    and maze[new_row][new_col] == 0
                    and (new_row, new_col) not in visited
                ):
                    visited.add((new_row, new_col))
                    queue.append(((new_row, new_col), actions + [(dr, dc)]))

        return None  # No path found


def generate_trajectory(args):
    seed, transform = args
    obs, _ = env.reset(seed=seed)
    terminated = False
    images = []
    while not terminated:
        # 5. Execute the Action and Receive Feedback
        action = agent(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        images.append(transform(obs["pixels"]))
    return images


policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)
num_eval_episodes = 5
random_action = 0.9
env = gym.make("MiniGrid-FourRooms-v0", render_mode="rgb_array", max_episode_steps=256)
env = FullyObsWrapper(env)
env = AddRenderObservation(env, render_only=False)
env = StochasticActionWrapper(env, prob=1 - random_action)
# env = RecordVideo(
#     env,
#     video_folder="test",
#     name_prefix="eval",
#     episode_trigger=lambda x: True,
# )
# env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

agent = BFSPlanner(env.observation_space)
# print(np.unique(obs["image"][:, :, 0]))
pool = multiprocessing.Pool(processes=10)
results = pool.imap(
    generate_trajectory,
    [(i, torchvision.transforms.v2.Resize((64, 64))) for i in range(10)],
)
print(len(list(results)))

env.close()

print(f"Episode time taken: {env.time_queue}")
print(f"Episode total rewards: {env.return_queue}")
print(f"Episode lengths: {env.length_queue}")

asdf
env = FullyObsWrapper(env)
obs, _ = env.reset()
print(obs["image"])
print(obs["image"].shape)
gatherer = OffPolicyAlgorithm(
    "MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0
)
print(gatherer.policy)
asdf

env = RGBImgObsWrapper(env)
obs, _ = env.reset()
print(obs["image"].shape)
env = ImgObsWrapper(env)
obs, _ = env.reset()
print(obs.shape)
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
print(
    evaluate_policy(
        model,
        env,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        callback=None,
        reward_threshold=None,
        return_episode_rewards=True,
        warn=True,
    )
)

model.learn(total_timesteps=5000, log_interval=10, progress_bar=True)
print(
    evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        callback=None,
        reward_threshold=None,
        return_episode_rewards=True,
        warn=True,
    )
)
