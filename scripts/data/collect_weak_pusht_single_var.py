import hydra
import numpy as np
from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.envs.pusht import WeakPolicy


@hydra.main(version_base=None, config_path="./", config_name="config")
def run(cfg):
    """Run data collection script"""

    world = swm.World("swm/PushT-v1", **cfg.world, render_mode="rgb_array")
    world.set_policy(WeakPolicy(dist_constraint=100))

    variation_list = list(world.single_variation_space.names())
    traj_per_var = cfg.num_traj // len(variation_list)
    shard_per_var = traj_per_var // cfg.num_shards

    print("Available variations: ", variation_list)
    print("Trajectories per variable: ", traj_per_var)

    rng = np.random.default_rng(cfg.seed)

    default = ["agent.start_position", "block.start_position", "block.angle"]

    for var in variation_list:
        if var in default:
            continue

        world = swm.World("swm/PushT-v1", **cfg.world, render_mode="rgb_array")
        world.set_policy(WeakPolicy(dist_constraint=100))
        print(f"Collecting data for variable: {var}")
        var_name = var.replace(".", "/")
        for idx in range(cfg.num_shards):
            world.record_dataset(
                f"pusht_single_var_weak_100/{var_name}/shard_{idx}",
                episodes=shard_per_var,
                seed=rng.integers(0, 1_000_000).item(),
                cache_dir=cfg.cache_dir,
                mode=cfg.ds_type,
                options={"variation": tuple([var] + default)},
            )

    logging.success(" 🎉🎉🎉 Completed data collection for pusht_single_var_weak_100 🎉🎉🎉")


if __name__ == "__main__":
    run()
