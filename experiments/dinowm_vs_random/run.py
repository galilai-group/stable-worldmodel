from pathlib import Path

import datasets
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm


def img_transform():
    transform = transforms.Compose(
        [
            transforms.Resize(size=224),
            transforms.CenterCrop(size=224),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # TODO replace with ImageNet stats
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    episode_idx = dataset["episode_idx"][:]
    step_idx = dataset["step_idx"][:]
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)  # convert to lengths
    return np.array(lengths)


@hydra.main(version_base=None, config_path=".", config_name="config")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    # create world environment
    world = swm.World(**cfg.world, image_shape=(224, 224), render_mode="rgb_array")

    # create the transform
    transform = {
        "pixels": img_transform(),
        "goal": img_transform(),
    }
    dataset_path = Path(cfg.cache_dir or swm.data.get_cache_dir(), cfg.preprocess_dataset)
    dataset = datasets.load_from_disk(dataset_path).with_format("numpy")
    ep_indices = np.unique(dataset["episode_idx"][:], return_index=True)[1]

    # create the processing
    action_process = preprocessing.StandardScaler()
    action_process.fit(dataset["action"][:])
    proprio_process = preprocessing.StandardScaler()
    proprio_process.fit(dataset["proprio"][:])

    process = {
        "action": action_process,
        "proprio": proprio_process,
        "goal_proprio": proprio_process,
    }

    # -- run evaluation

    # determine policy
    policy = swm.policy.RandomPolicy(cfg.seed)
    if cfg.policy != "random":
        model = swm.policy.AutoCostModel(cfg.policy).to("cuda")
        config = swm.PlanConfig(**cfg.plan_config)
        solver = swm.solver.CEMSolver(model, **cfg.solver)
        policy = swm.policy.WorldModelPolicy(solver=solver, config=config, process=process, transform=transform)

    # sample the episodes
    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(len(ep_indices) - 1, size=cfg.eval.num_eval, replace=False)
    eval_episodes = dataset[random_episode_indices]["episode_idx"]
    episode_len = get_episodes_length(dataset, eval_episodes)

    # drop episodes that are too short
    max_start_idx = episode_len - cfg.eval.num_steps - 1
    eval_episodes = eval_episodes[max_start_idx >= 0]
    episode_len = episode_len[max_start_idx >= 0]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    rnd_start_idx = g.integers(0, max_start_idx + 1, size=len(eval_episodes))

    world.set_policy(policy)

    metrics = world.evaluate_from_dataset(
        cfg.eval.dataset_name,
        num_steps=cfg.eval.num_steps,
        start_steps=rnd_start_idx.tolist(),
        episodes_idx=eval_episodes.tolist(),
        cache_dir=cfg.get("cache_dir", None),
        callables={
            "_set_state": "state",
            "_set_goal_state": "goal_state",
        },
    )

    # dump results
    print(metrics)


if __name__ == "__main__":
    run()
