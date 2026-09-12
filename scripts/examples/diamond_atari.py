"""Example: Train Diamond diffusion world model on Atari data.

This script demonstrates the end-to-end pipeline:
  1. Collect Atari trajectories with a random policy
  2. Train the EDM-based diffusion world model

Usage:
    python scripts/examples/diamond_atari.py

    # Custom config
    python scripts/examples/diamond_atari.py env_name=ALE/Pong-v5 episodes=1000
"""

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from scripts.data.collect_atari import collect_atari
from scripts.train.diamond import run as train_diamond


@hydra.main(
    version_base=None,
    config_path='../train/config',
    config_name='diamond_atari',
)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    print(f'Collecting {cfg.episodes} episodes from {cfg.env_name}...')
    dataset_path = collect_atari(
        env_name=cfg.env_name,
        num_envs=cfg.num_envs,
        episodes=cfg.episodes,
        seed=cfg.seed,
    )

    print(f'Training Diamond on {dataset_path}...')
    os.environ['LOCAL_DATASET_DIR'] = str(dataset_path.parent)
    train_diamond(
        config_path=str(
            Path(__file__).parents[2]
            / 'train'
            / 'config'
            / 'diamond_atari.yaml'
        )
    )


if __name__ == '__main__':
    main()
