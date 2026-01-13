import hydra
import numpy as np
from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.envs.two_room import ExpertPolicy


@hydra.main(version_base=None, config_path="./", config_name="config")
def run(cfg):
    """Run data collection script"""

    world = swm.World("swm/TwoRoom-v0", **cfg.world, render_mode="rgb_array")
    world.set_policy(ExpertPolicy())

    options = cfg.get("options")
    traj_per_shard = cfg.num_traj // cfg.num_shards

    rng = np.random.default_rng(cfg.seed)

    for i in range(cfg.num_shards):
        world.record_dataset(
            f"tworoom_noisy/shard_{i}",
            episodes=traj_per_shard,
            seed=rng.integers(0, 1_000_000).item(),
            cache_dir=cfg.cache_dir,
            mode=cfg.ds_type,
            options=options,
        )

    logging.success(" 🎉🎉🎉 Completed data collection for tworoom_noisy 🎉🎉🎉")


if __name__ == "__main__":
    run()
