import os

os.environ['MUJOCO_GL'] = 'egl'

import hydra
import numpy as np
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench_manip import ExpertPolicy


@hydra.main(version_base=None, config_path='./config', config_name='ogb')
def run(cfg: DictConfig):
    """Run parallel data collection script"""

    world = swm.World(
        'swm/OGBScene-v0',
        **cfg.world,
        env_type='single',
        ob_type='pixels',
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=False,
        mode='data_collection',
    )

    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None

    rng = np.random.default_rng(cfg.seed)
    world.set_policy(ExpertPolicy(policy_type='plan_oracle'))

    world.record_dataset(
        'ogb_scene_single_expert',
        episodes=cfg.num_traj,
        seed=rng.integers(0, 1_000_000).item(),
        cache_dir=cfg.cache_dir,
        options=options,
    )

    logging.success(
        '🎉🎉🎉 Completed data collection for ogbench scene  🎉🎉🎉'
    )


if __name__ == '__main__':
    run()
