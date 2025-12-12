import os


os.environ["MUJOCO_GL"] = "egl"

import hydra
import numpy as np
from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench_manip import ExpertPolicy


@hydra.main(version_base=None, config_path="./", config_name="config")
def run(cfg):
    """Run data collection script"""

    world = swm.World(
        "swm/OGBScene-v0",
        **cfg.world,
        env_type="single",
        ob_type="pixels",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=False,
        mode="data_collection",
    )

    world.set_policy(ExpertPolicy(policy_type="plan_oracle"))

    options = cfg.get("options")
    traj_per_shard = cfg.num_traj // cfg.num_shards

    rng = np.random.default_rng(cfg.seed)

    for i in range(cfg.num_shards):
        world.record_dataset(
            f"ogb_scene_oracle/shard_{i}",
            episodes=traj_per_shard,
            seed=rng.integers(0, 1_000_000).item(),
            cache_dir=cfg.cache_dir,
            mode=cfg.ds_type,
            options=options,
        )

    logging.success(" 🎉🎉🎉 Completed data collection for ogb_scene_oracle 🎉🎉🎉")


if __name__ == "__main__":
    run()
