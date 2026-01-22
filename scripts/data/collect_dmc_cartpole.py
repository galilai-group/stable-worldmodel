import os


os.environ["MUJOCO_GL"] = "egl"


import concurrent.futures

import hydra
import numpy as np
from loguru import logger as logging
from omegaconf import DictConfig

import stable_worldmodel as swm


def collect_shard(shard_id, seed, cfg):
    """
    Worker function to collect a single shard.
    Instantiates its own World and Policy to avoid pickling issues.
    """
    try:
        world = swm.World(
            "swm/CartpoleDMControl-v0",
            **cfg.world,
            goal_conditioned=False,
        )
        world.set_policy(swm.policy.RandomPolicy(seed=seed))

        options = cfg.get("options")
        traj_per_shard = cfg.num_traj // cfg.num_shards

        logging.info(f"Process {shard_id}: Started collecting {traj_per_shard} trajectories...")

        world.record_dataset(
            f"dmc_cartpole/shard_{shard_id}",
            episodes=traj_per_shard,
            seed=seed,
            cache_dir=cfg.cache_dir,
            options=options,
        )

        world.close()
        return f"Shard {shard_id} success"

    except Exception as e:
        logging.error(f"Shard {shard_id} failed: {e}")
        raise e


@hydra.main(version_base=None, config_path="./", config_name="config")
def run(cfg: DictConfig):
    """Run parallel data collection script"""
    max_workers = cfg.get("num_workers", 4)

    logging.info(f"🚀 Starting data collection with {max_workers} parallel workers")

    rng = np.random.default_rng(cfg.seed)
    seeds = [rng.integers(0, 1_000_000).item() for _ in range(cfg.num_shards)]

    collect_shard(0, seeds[0], cfg)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for i in range(cfg.num_shards):
            futures.append(executor.submit(collect_shard, i, seeds[i], cfg))

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                logging.success(f"✅ {result}")
            except Exception as e:
                logging.error(f"❌ Worker failed with error: {e}")

    logging.success("🎉🎉🎉 Completed data collection for all shards 🎉🎉🎉")


if __name__ == "__main__":
    run()
