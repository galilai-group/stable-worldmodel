import hydra
from loguru import logger as logging
import numpy as np

import stable_worldmodel as swm


@hydra.main(version_base=None, config_path='./', config_name='config')
def run(cfg):
    """Run data collection script"""

    world = swm.World('swm/PinPad-Discrete-v0', **cfg.world)
    world.set_policy(swm.envs.pinpad.ExpertPolicyDiscrete())
    logging.info("Set world's policy to expert policy")

    logging.info(f'Collecting data for {cfg.num_traj} trajectories')
    dataset_name = 'pinpad_discrete'
    world.record_dataset(
        dataset_name,
        episodes=cfg.num_traj,
        seed=np.random.default_rng(cfg.seed).integers(0, int(2**20)).item(),
        cache_dir=cfg.cache_dir,
        options=cfg.get('options'),
    )
    logging.success(
        f' 🎉🎉🎉 Completed data collection for {dataset_name} 🎉🎉🎉'
    )

    dataset = swm.data.HDF5Dataset(
        name=dataset_name,
        keys_to_load=['pixels', 'action', 'observation'],
    )
    logging.info(f'Loaded dataset from {dataset.h5_path}')
    swm.utils.record_video_from_dataset(
        video_path='./videos/pinpad_discrete',
        dataset=dataset,
        episode_idx=[0, 1, 2, 3],
        max_steps=cfg.world.max_episode_steps,
        fps=30,
        viewname='pixels',
    )
    logging.success(
        f' 🎉🎉🎉 Completed video recording from dataset for {dataset_name} 🎉🎉🎉'
    )


if __name__ == '__main__':
    run()
