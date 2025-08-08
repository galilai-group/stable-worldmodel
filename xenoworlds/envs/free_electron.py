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
import matplotlib.pyplot as plt
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
import numpy as np
from tqdm.rich import trange


def single_trajectory(length, device="cpu"):
    actions = torch.randn(length, 2, device=device) / np.sqrt(length)
    for i in range(1, length):
        actions[i] = actions[i] + actions[i - 1]
    return actions


class FourierEmbedding(torch.nn.Module):
    def __init__(self, input_dim, embedding_size, scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_size = embedding_size
        self.scale = scale

        # Initialize the embedding matrix with random frequencies
        self.register_parameter(
            "B", torch.nn.Parameter(torch.randn(input_dim, embedding_size // 2) * scale)
        )

    def forward(self, x):
        # x: (batch_size, input_dim)
        print(x, self.B)
        x_proj = x @ self.B  # (batch_size, embedding_size)
        embedding = torch.cat(
            [torch.sin(x_proj), torch.cos(x_proj)], dim=-1
        )  # (batch_size, 2 * embedding_size)
        return embedding


class Encoder(torch.nn.Module):
    def __init__(self, output_dim) -> None:
        super().__init__()
        # self.embedding = FourierEmbedding(2, 32)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(observations)


def generate_graph(length):
    G = torch.zeros(length, length)
    r = torch.exp(-torch.linspace(-3, 3, 11).square())
    for i in range(r.size(0)):
        v = G.diagonal(offset=-r.size(0) // 2 + i)
        v.add_(r[i])
    return G


@dataclass
class Config:
    length: int = 128
    max_episode_steps: int = 400
    maze_width: int = 5
    maze_height: int = 5
    save_episodes: int = 5
    num_train_steps: int = 1000000
    seed: int = 43
    dump_folder: Path = Path("point_maze_dataset")


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=Config)


class Buffer:
    def __init__(self, length):
        self.length = length
        self.items = []

    def add(self, item):
        self.items.append(item)
        if len(self.items) > self.length:
            self.items.pop(0)

    def get(self):
        return self.items


def vicreg(x, G):
    return (x @ x.T - G).square().mean()
    with torch.no_grad():
        G_norm = G.diagonal().sum()
        S, U = torch.linalg.eigh(G)
        Gsqrt = (U * S.clip(0).sqrt()) @ U.T
        Y = U * S.clip(0).sqrt()
        Y = Y[:, -x.size(1) :]
    return torch.linalg.svdvals(x.T @ G @ x).sqrt().mean()
    return (x - Y).square().mean()
    S = torch.linalg.svdvals(Gsqrt @ x).sum()
    return (x.square().sum() - 2 * S + G_norm) / x.size(0)


@hydra.main(version_base=None, config_name="config")
def main(cfg):

    # best: 0.008
    # worst: 0.56

    # X = torch.randn(10, 10)
    # Y = torch.randn(10, 10)

    # U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    # Uy, Sy, Vhy = torch.linalg.svd(Y, full_matrices=False)
    # print(torch.linalg.svdvals(X.T @ Y).sum())
    # print(torch.linalg.svdvals((U * S).T @ (Uy * Sy)).sum())
    # S, U = torch.linalg.eigh(X @ X.T)
    # Sy, Uy = torch.linalg.eigh(Y @ Y.T)
    # print(torch.linalg.svdvals((U * S.abs().sqrt()).T @ (Uy * Sy.abs().sqrt())).sum())
    # print(torch.trace((U * S.abs().sqrt()) @ U.T @ (Uy * Sy.abs().sqrt()) @ Uy.T))
    # print(torch.sum(S.abs().sqrt() * Sy.abs().sqrt()))

    # asdf
    G = generate_graph(cfg["length"])
    plt.imshow(G, aspect="auto")
    plt.savefig("graph.png")
    plt.close()
    traj = single_trajectory(cfg["length"]).numpy()
    plt.plot(traj[:, 0], traj[:, 1])
    plt.savefig("traj.png")
    plt.close()
    G = generate_graph(cfg["length"])
    plt.imshow(traj @ traj.T, aspect="auto")
    plt.savefig("tgraph.png")
    plt.close()

    import optimalssl

    ll = optimalssl.losses.RelativeMSE(bias=False, weight_decay=0.1)

    model = Encoder(2).cuda()
    probe = torch.nn.Linear(2, 2, bias=False).cuda()
    optim = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()),
        lr=0.001,
        weight_decay=0,
    )
    losses = []
    repeats = 4
    for step in trange(10000 // repeats):
        for repeat in range(repeats):
            torch.manual_seed(step)
            trajectory = single_trajectory(cfg["length"], device="cuda")
            preds = model(trajectory)
            #
            Wstar = torch.linalg.solve(preds.T @ preds, preds.T @ trajectory)
            terror = (preds @ Wstar - trajectory).square().mean(0).sum()
            probing_loss = (probe(preds.detach()) - trajectory).square().mean(0).sum()
            ssl_loss = ll(preds, trajectory @ trajectory.T)["loss"]
            loss = probing_loss + ssl_loss * 10
            loss.backward()
            losses.append(probing_loss.item())
            optim.step()
            optim.zero_grad()
        # print(probing_loss.item(), loss.item())
        if step % (500 // repeats) == 0:
            losses = []
            torch.manual_seed(999999999)
            model.eval()
            probe.eval()
            with torch.inference_mode():
                for _ in range(100):
                    trajectory = single_trajectory(cfg["length"], device="cuda")
                    preds = model(trajectory)
                    probing_loss = (probe(preds) - trajectory).square().mean(0).sum()
                    losses.append(probing_loss.item())
            model.train()
            probe.train()
            print(
                "Probing:",
                probing_loss.item(),
                "SSL:",
                ssl_loss.item(),
                "TRUE",
                terror.item(),
                "avg:",
                np.mean(losses),
            )
    print(np.mean(losses[-10:]))
    sadf
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
