import hydra
import numpy as np
from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.envs.pusht import WeakPolicy


@hydra.main(version_base=None, config_path='./config', config_name='default')
def run(cfg):
    """Run data collection script"""

    world = swm.World('swm/PushT-v1', **cfg.world, render_mode='rgb_array')
    world.set_policy(WeakPolicy(dist_constraint=100))

    rng = np.random.default_rng(cfg.seed)

    for i in range(10):
        world.record_dataset(
            f'pusht_toy/shard_{i}',
            episodes=500,
            seed=rng.integers(0, 1_000_000).item(),
            cache_dir=cfg.cache_dir,
            mode=cfg.ds_type,
        )

    logging.success(' 🎉🎉🎉 Completed data collection for pusht_toy 🎉🎉🎉')


if __name__ == '__main__':
    run()
