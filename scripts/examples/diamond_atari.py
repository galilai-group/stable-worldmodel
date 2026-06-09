"""Example: Train Diamond diffusion world model on Atari data.

This script demonstrates the end-to-end pipeline:
  1. Collect Atari trajectories with a random policy
  2. Train the EDM-based diffusion world model

Usage:
    # Collect data and train (default: Breakout, 500 episodes)
    python scripts/examples/diamond_atari.py

    # Custom environment
    python scripts/examples/diamond_atari.py --env ALE/Pong-v5 --episodes 1000
"""

import argparse
import os
from pathlib import Path

from scripts.data.collect_atari import collect_atari
from scripts.train.diamond import run as train_diamond


def main():
    parser = argparse.ArgumentParser(description='Train Diamond on Atari')
    parser.add_argument(
        '--env',
        type=str,
        default='ALE/Breakout-v5',
        help='Atari environment name',
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=500,
        help='Episodes to collect for training',
    )
    parser.add_argument(
        '--num-envs',
        type=int,
        default=4,
        help='Parallel environments for collection',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to Diamond training config YAML',
    )
    args = parser.parse_args()

    # Step 1: Collect trajectories
    game_name = args.env.replace('/', '_').lower()
    print(f'Collecting {args.episodes} episodes from {args.env}...')
    dataset_path = collect_atari(
        env_name=args.env,
        num_envs=args.num_envs,
        episodes=args.episodes,
        seed=args.seed,
    )

    # Step 2: Train the diffusion world model
    config_path = args.config or str(
        Path(__file__).parents[2] / 'train' / 'config' / 'diamond_atari.yaml'
    )

    print(f'Training Diamond on {dataset_path}...')
    os.environ['LOCAL_DATASET_DIR'] = str(dataset_path.parent)
    train_diamond(config_path=config_path)


if __name__ == '__main__':
    main()
