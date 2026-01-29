import hydra
import numpy as np
from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.envs.pusht import WeakPolicy


@hydra.main(version_base=None, config_path='./', config_name='config')
def run(cfg):
    """Run data collection script"""

    world = swm.World('swm/PushT-v1', **cfg.world, render_mode='rgb_array')
    world.set_policy(WeakPolicy(dist_constraint=100))

    variation_list = list(world.single_variation_space.names())
    variation_default = {
        'agent.start_position',
        'block.start_position',
        'block.angle',
    }

    # exclude default variations
    variation_list = set(variation_list)
    rng = np.random.default_rng(cfg.seed)

    for var in variation_list:
        var = var.replace('variation.', '')
        if var in variation_default:
            continue
        world = swm.World('swm/PushT-v1', **cfg.world, render_mode='rgb_array')
        world.set_policy(WeakPolicy(dist_constraint=100))
        print(f'Collecting data for variable: {var}')
        var_name = var.replace('.', '/')
        world.record_dataset(
            f'pusht_fov/{var_name}',
            episodes=1000,
            seed=rng.integers(0, 1_000_000).item(),
            cache_dir=cfg.cache_dir,
            options={'variation': tuple([var] + list(variation_default))},
        )

        logging.success(
            f' 🎉🎉🎉 Completed data collection for pusht {var_name} 🎉🎉🎉'
        )


if __name__ == '__main__':
    run()
