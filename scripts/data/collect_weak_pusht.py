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

    options = cfg.get('options')
    traj_per_shard = cfg.num_traj // cfg.num_shards

    rng = np.random.default_rng(cfg.seed)

    for i in range(cfg.num_shards):
        world.record_dataset(
            f'pusht_weak_100/shard_{i}',
            episodes=traj_per_shard,
            seed=rng.integers(0, 1_000_000).item(),
            cache_dir=cfg.cache_dir,
            mode=cfg.ds_type,
            options=options,
        )

    logging.success(
        ' 🎉🎉🎉 Completed data collection for pusht_weak_100 🎉🎉🎉'
    )


if __name__ == '__main__':
    run()
