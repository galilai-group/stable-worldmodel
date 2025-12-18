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

    print("Available variations: ", world.single_variation_space.names())

    var_types = list(world.single_variation_space.names())
    traj_per_var = cfg.num_traj // len(var_types)

    rng = np.random.default_rng(cfg.seed)

    default = ["agent.position", "goal.position", "door.number", "door.size", "door.position"]

    for var in var_types:
        var_name = var.replace(".", "/")
        world.record_dataset(
            f"tworoom_noisy_single_var/{var_name}",
            episodes=traj_per_var,
            seed=rng.integers(0, 1_000_000).item(),
            cache_dir=cfg.cache_dir,
            mode=cfg.ds_type,
            options={"variation": tuple([var] + default)},
        )

    logging.success(" 🎉🎉🎉 Completed data collection for tworoom_noisy_single_var 🎉🎉🎉")


if __name__ == "__main__":
    run()
