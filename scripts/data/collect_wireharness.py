import os


# Rendering must be ON for data collection; set the GL backend before mujoco /
# the env are imported. EnvPool is in-process, so one setting covers every env.
os.environ.setdefault('MUJOCO_GL', 'egl')

from pathlib import Path

import hydra
import numpy as np
from loguru import logger as logging
from omegaconf import OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.envs.wire_harness import ExpertPolicy


@hydra.main(
    version_base=None, config_path='./config', config_name='wireharness'
)
def run(cfg):
    """Collect WireHarness expert rollouts."""

    world = swm.World(cfg.env_name, **cfg.world)

    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None

    rng = np.random.default_rng(cfg.seed)

    ckpt_path = Path(os.path.expanduser(str(cfg.expert_ckpt_path)))
    world.set_policy(
        ExpertPolicy(
            ckpt_path=ckpt_path / 'best_one_mover_sac/best_model.zip',
            vec_normalize_path=ckpt_path
            / 'best_one_mover_sac/vec_normalize.pkl',
            noise_std=cfg.noise_std,
            device=cfg.device,
        )
    )

    out = (
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / 'wireharness_expert.lance'
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    world.collect(
        path=out,
        episodes=cfg.num_traj,
        seed=rng.integers(0, 1_000_000).item(),
        options=options,
    )

    logging.success(' 🎉🎉🎉 Completed data collection for wireharness 🎉🎉🎉')


if __name__ == '__main__':
    run()
