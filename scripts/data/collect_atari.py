"""Collect Atari trajectories for world model training.

Collects episodes from Atari environments using a random policy with
no-ops, storing observations, actions, rewards, and terminal signals
in Lance format compatible with the Diamond training pipeline.

Example:
    python scripts/data/collect_atari.py --env ALE/Breakout-v5 --episodes 500
"""

from pathlib import Path

import numpy as np
from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.policy import RandomPolicy


def collect_atari(
    env_name: str = 'ALE/Breakout-v5',
    num_envs: int = 4,
    episodes: int = 500,
    max_episode_steps: int = 27000,
    seed: int = 42,
    cache_dir: str | None = None,
):
    rng = np.random.default_rng(seed)

    world = swm.World(
        env_name,
        num_envs=num_envs,
        image_shape=(64, 64),
        max_episode_steps=max_episode_steps,
        goal_conditioned=False,
        render_mode='rgb_array',
    )

    policy = RandomPolicy(seed=rng.integers(0, 1_000_000).item())
    world.set_policy(policy)

    game_name = env_name.replace('/', '_').lower()
    output_path = (
        Path(cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / f'{game_name}.lance'
    )

    world.collect(
        str(output_path),
        episodes=episodes,
        seed=rng.integers(0, 1_000_000).item(),
    )

    logging.success(f'Collected {episodes} episodes to {output_path}')
    return output_path


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(
        version_base=None,
        config_path='config',
        config_name='collect_atari',
    )
    def main(cfg: DictConfig):
        OmegaConf.resolve(cfg)
        collect_atari(
            env_name=cfg.env_name,
            num_envs=cfg.num_envs,
            episodes=cfg.episodes,
            max_episode_steps=cfg.max_episode_steps,
            seed=cfg.seed,
            cache_dir=cfg.get('cache_dir', None),
        )

    main()
