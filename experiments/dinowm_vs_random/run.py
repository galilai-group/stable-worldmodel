from pathlib import Path

import datasets
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm
import wandb


def img_transform():
    transform = transforms.Compose(
        [
            transforms.Resize(size=224),
            transforms.CenterCrop(size=224),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    episode_idx = dataset["episode_idx"][:]
    step_idx = dataset["step_idx"][:]
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


@hydra.main(version_base=None, config_path=".", config_name="config")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget, (
        "Planning horizon must be smaller than or equal to eval_budget"
    )
    if cfg.wandb.use_wandb:
        # Initialize wandb
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=dict(cfg))

    # create world environment
    world = swm.World(**cfg.world, image_shape=(224, 224), render_mode="rgb_array")

    # create the transform
    transform = {
        "pixels": img_transform(),
        "goal": img_transform(),
    }

    dataset_path = Path(cfg.cache_dir or swm.data.get_cache_dir(), cfg.eval.dataset_name)
    dataset = datasets.load_from_disk(dataset_path).with_format("numpy")
    ep_indices, _ = np.unique(dataset["episode_idx"][:], return_index=True)

    # create the processing
    ACTION_MEAN = np.array([-0.0087, 0.0068])
    ACTION_STD = np.array([0.2019, 0.2002])
    PROPRIO_MEAN = np.array([236.6155, 264.5674, -2.93032027, 2.54307914])
    PROPRIO_STD = np.array([101.1202, 87.0112, 74.84556075, 74.14009094])

    action_process = preprocessing.StandardScaler()
    action_process.mean_ = ACTION_MEAN
    action_process.scale_ = ACTION_STD

    proprio_process = preprocessing.StandardScaler()
    proprio_process.mean_ = PROPRIO_MEAN
    proprio_process.scale_ = PROPRIO_STD

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

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in dataset["episode_idx"]])

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset["step_idx"] <= max_start_per_row
    dataset_start = dataset.select(np.nonzero(valid_mask)[0])

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(len(dataset_start) - 1, size=cfg.eval.num_eval, replace=False)
    eval_episodes = dataset_start[random_episode_indices]["episode_idx"]
    eval_start_idx = dataset_start[random_episode_indices]["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    metrics = world.evaluate_from_dataset(
        cfg.eval.dataset_name,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        cache_dir=cfg.get("cache_dir", None),
        callables={
            "_set_state": "state",
            "_set_goal_state": "goal_state",
        },
    )

    if cfg.wandb.use_wandb:
        # Log metrics to wandb
        wandb.log(metrics)
        # Finish wandb run
        wandb.finish()

    # dump results
    print(metrics)
    # ---- dump results to a txt file ----
    results_path = Path(__file__).parent / "results.txt"
    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs
        f.write(f"policy: {cfg.policy}\n")
        f.write(f"dataset_name: {cfg.eval.dataset_name}\n")
        f.write(f"goal_offset_steps: {cfg.eval.goal_offset_steps}\n")
        f.write(f"eval_budget: {cfg.eval.eval_budget}\n")
        f.write(f"horizon: {cfg.plan_config.horizon}\n")
        f.write(f"receding_horizon: {cfg.plan_config.receding_horizon}\n")
        f.write(f"metrics: {metrics}\n")


if __name__ == "__main__":
    run()
