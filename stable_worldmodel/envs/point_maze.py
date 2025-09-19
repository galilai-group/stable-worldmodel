import gymnasium as gym
import gymnasium_robotics
import random

from stable_baselines3 import PPO, SAC
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from pathlib import Path
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback

from gymnasium.wrappers import (
    NormalizeReward,
    NormalizeObservation,
    FlattenObservation,
    FrameStackObservation,
)
import pandas as pd
from PIL import Image
import hydra
from tqdm.rich import trange
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
import lightning
import wandb
from omegaconf import OmegaConf
from uuid import uuid4
from dataclasses import field


class FourierEmbedding(torch.nn.Module):
    def __init__(self, input_dim, embedding_size, scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_size = embedding_size
        self.scale = scale

        # Initialize the embedding matrix with random frequencies
        self.B = torch.nn.Parameter(torch.randn(input_dim, embedding_size // 2) * scale)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x_proj = x @ self.B  # (batch_size, embedding_size)
        embedding = torch.cat(
            [torch.sin(x_proj), torch.cos(x_proj)], dim=-1
        )  # (batch_size, 2 * embedding_size)
        return embedding


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)
        # n_input_channels = observation_space.shape[0]
        # self.cnn = ossl.module.Resnet9(512, num_channels=n_input_channels)
        self.embedding = FourierEmbedding(8, 256)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(256 * 4, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.embedding(observations).flatten(1, 2))


def generate_maze(width, height):
    """Generates a maze using recursive backtracking."""

    maze = [[1] * width for _ in range(height)]  # Initialize maze with walls

    def carve_path(x, y):
        maze[y][x] = 0  # Mark current cell as path

        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]  # Possible directions to move
        random.shuffle(directions)  # Randomize direction order

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < width - 1 and 0 < ny < height - 1 and maze[ny][nx] == 1:
                maze[ny - dy // 2][nx - dx // 2] = 0  # Carve path to next cell
                carve_path(nx, ny)  # Recursively carve from the next cell

    carve_path(1, 1)  # Start carving from (1, 1)
    return maze


@dataclass
class PPOConfig:
    learning_rate: float = 0.0003
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 50
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class Config:
    max_episode_steps: int = 400
    maze_width: int = 5
    maze_height: int = 5
    save_episodes: int = 5
    num_train_steps: int = 1000000
    seed: int = 43
    dump_folder: Path = Path("point_maze_dataset")
    PPO: PPOConfig = field(default_factory=PPOConfig)


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=Config)


def make_env(cfg):
    example_map = generate_maze(cfg["maze_width"], cfg["maze_height"])
    env = gym.make(
        "PointMaze_UMaze-v3",
        maze_map=example_map,
        render_mode="rgb_array_list",
        max_episode_steps=cfg["max_episode_steps"],
        camera_name="tracking",
        width=224,
        height=224,
    )
    env = FlattenObservation(env)
    env = FrameStackObservation(env, stack_size=4)
    # env = NormalizeObservation(env)
    # env = NormalizeReward(env)
    # env = AddRenderObservation(env, render_only=False)
    return env


@hydra.main(version_base=None, config_name="config")
def main(cfg):
    cfg["dump_folder"] = Path(cfg["dump_folder"])
    run = wandb.init(
        project="sb3_v2",
        config=OmegaConf.to_container(cfg),
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )
    lightning.seed_everything(cfg["seed"])
    gym.register_envs(gymnasium_robotics)
    env = make_env(cfg)
    d = max(cfg["maze_width"], cfg["maze_height"])
    env.unwrapped.point_env.mujoco_renderer.default_cam_config["distance"] = d
    env.unwrapped.point_env.mujoco_renderer.default_cam_config["azimuth"] = 0
    env.unwrapped.point_env.mujoco_renderer.default_cam_config["elevation"] = -90

    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        ortho_init=False,
        activation_fn=torch.nn.ReLU,
    )
    model = PPO(
        "MultiInputPolicy",
        env,
        # n_timesteps=100000,
        policy_kwargs=policy_kwargs,
        verbose=2,
        **cfg["PPO"],
        tensorboard_log=f"runs/{run.id}",
    )
    # print(
    #     evaluate_policy(
    #         model,
    #         env,
    #         n_eval_episodes=1,
    #         deterministic=True,
    #         render=False,
    #         callback=None,
    #         reward_threshold=None,
    #         return_episode_rewards=True,
    #         warn=True,
    #     )
    # )
    eval_callback = EvalCallback(
        env,
        log_path=f"runs/{run.id}",
        eval_freq=500,
        deterministic=True,
        render=True,
    )
    model.learn(
        total_timesteps=cfg["num_train_steps"],
        log_interval=1,
        progress_bar=True,
        callback=[WandbCallback(verbose=2), eval_callback],
    )

    for seed in trange(cfg["save_episodes"], desc="episode"):
        frame_idx = 0
        storage = cfg["dump_folder"] / str(uuid4())
        storage.mkdir(parents=True, exist_ok=True)
        metadata = pd.DataFrame(
            columns=["file_name", "action", "reward", "state"],
            index=range(cfg["max_episode_steps"]),
        )
        obs, _ = env.reset(seed=seed)
        terminated, truncated = False, False
        while not terminated and not truncated:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            # we take -1 since they are stacked
            metadata.loc[frame_idx, "state"] = obs[-1]
            metadata.loc[frame_idx, "reward"] = reward
            metadata.loc[frame_idx, "action"] = action.tolist()
            frame_idx += 1
        metadata = metadata.iloc[:frame_idx]
        frames = env.render()[:-1]
        for frame_idx, frame in enumerate(frames):
            file_name = storage / f"frame_{frame_idx}.jpg"
            img = Image.fromarray(frame)
            img.save(str(file_name))
        metadata["file_name"] = [f"frame_{i}.jpg" for i in range(len(frames))]
        metadata["episode"] = seed
        metadata["width"] = cfg["maze_width"]
        metadata["height"] = cfg["maze_height"]
        metadata.to_csv(storage / "metadata.csv")
    env.close()


if __name__ == "__main__":
    """
    MUJOCO_GL=egl python point_maze.py \
        hydra/launcher=submitit_slurm hydra.launcher.timeout_min=60 \
        hydra.launcher.partition=scavenge \
        hydra.launcher.gpus_per_node=1 \
        hydra.launcher.tasks_per_node=1 \
        seed=0 \
        maze_width=5 \
        maze_height=5

    MUJOCO_GL=egl python point_maze.py  --multirun \
        hydra/launcher=submitit_slurm hydra.launcher.timeout_min=60 \
        hydra.launcher.partition=scavenge \
        hydra.launcher.gpus_per_node=1 \
        hydra.launcher.tasks_per_node=1 \
        seed=0 \
        maze_width=5 \
        maze_height=5 \
        PPO.learning_rate=1e-2,1e-4,1e-6 \
        PPO.ent_coef=0,0.01,0.1 \
        PPO.vf_coef=0.1,0.5,0.9 \
        PPO.gamma=0.5,0.9,0.99 \
        PPO.gae_lambda=0.5,0.9,0.99

    """
    main()
