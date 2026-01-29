import hydra
import numpy as np
from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.envs.two_room import ExpertPolicy


@hydra.main(version_base=None, config_path='./', config_name='config')
def run(cfg):
    """Run data collection script"""

    world = swm.World('swm/TwoRoom-v1', **cfg.world, render_mode='rgb_array')
    world.set_policy(ExpertPolicy())

    options = cfg.get('options')
    rng = np.random.default_rng(cfg.seed)

    world.record_dataset(
        'tworoom',
        episodes=cfg.num_traj,
        seed=rng.integers(0, 1_000_000).item(),
        cache_dir=cfg.cache_dir,
        options=options,
    )

    logging.success(' 🎉🎉🎉 Completed data collection for tworoom 🎉🎉🎉')


if __name__ == '__main__':
    run()
