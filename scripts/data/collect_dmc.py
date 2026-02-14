import os
from pathlib import Path

from omegaconf import OmegaConf

os.environ['MUJOCO_GL'] = 'egl'

import hydra
import numpy as np
from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.envs.dmcontrol import ExpertPolicy

ENVS = {
    'swm/CartpoleDMControl-v0': ('cartpole',),
    'swm/WalkerDMControl-v0': ('walker',),
    'swm/QuadrupedDMControl-v0': ('quadruped',),
    'swm/BallInCupDMControl-v0': ('ballincup',),
    # 'swm/AcrobotDMControl-v0': ('acrobot',),
    'swm/FingerDMControl-v0': ('finger',),
    'swm/HopperDMControl-v0': ('hopper',),
    # 'swm/HumanoidDMControl-v0': ('humanoid',),
    # 'swm/ManipulatorDMControl-v0': ('manipulator',),
    'swm/CheetahDMControl-v0': ('cheetah',),
    'swm/ReacherDMControl-v0': ('reacher',),
    'swm/PendulumDMControl-v0': ('pendulum',),
}


@hydra.main(version_base=None, config_path='./config', config_name='dmc')
def run(cfg):
    """Run data collection script"""

    world = swm.World(cfg.env_name, **cfg.world)

    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None

    rng = np.random.default_rng(cfg.seed)
    ckpt_path = Path(cfg.expert_ckpt_path)
    name = ENVS[cfg.env_name][0]

    world.set_policy(
        ExpertPolicy(
            ckpt_path=ckpt_path / f'{name}/expert_policy.zip',
            vec_normalize_path=ckpt_path / f'{name}/vec_normalize.pkl',
            noise_std=cfg.noise_std,
            device=cfg.device,
        )
    )

    world.record_dataset(
        f'dmc/{name}_expert',
        episodes=cfg.num_traj,
        seed=rng.integers(0, 1_000_000).item(),
        cache_dir=cfg.cache_dir,
        options=options,
    )

    logging.success(' 🎉🎉🎉 Completed data collection for tworoom 🎉🎉🎉')


if __name__ == '__main__':
    run()
