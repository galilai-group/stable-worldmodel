from pathlib import Path

import hydra
import numpy as np
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm


@hydra.main(version_base=None, config_path='./config', config_name='maniskill')
def run(cfg: DictConfig):
    """Collect a random-policy dataset on a ManiSkill task (mirrors collect_cube.py).

    ManiSkill has no expert oracle, so we use RandomPolicy. The dataset records
    pixels / action / proprio / state (flat sim state) — enough for LeWM training
    and for the goal-conditioned MPC eval (which sets start/goal via the flat state).
    """
    world = swm.World(
        cfg.env_name,
        **cfg.world,
        goal_conditioned=False,
    )

    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None
    rng = np.random.default_rng(cfg.seed)
    world.set_policy(swm.policy.RandomPolicy(cfg.seed))

    out = (
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / 'maniskill/mspickcube_random.lance'
    )
    world.collect(
        out,
        episodes=cfg.num_traj,
        seed=rng.integers(0, 1_000_000).item(),
        options=options,
    )

    logging.success('🎉 Completed data collection for ManiSkill 🎉')


if __name__ == '__main__':
    run()
