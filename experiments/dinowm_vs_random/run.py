import time
import types
from pathlib import Path

import datasets
import hydra
import numpy as np
import stable_pretraining as spt
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm
import wandb


def img_transform():
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=196),
            transforms.CenterCrop(size=196),
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
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224), render_mode="rgb_array")

    # create the transform
    transform = {
        "pixels": img_transform(),
        "goal": img_transform(),
    }

    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir(), cfg.eval.dataset_name)
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
        model = model.eval()

        def criterion(self, info_dict: dict, action_candidates: torch.Tensor):
            """
            Compute the cost for planning.
            Supports: 'current_distance' vs 'final_distance'
            Supports: 'mse' vs 'cosine'
            Assumed Shape: (B, N, T, P, D) -> Batch, Samples, Time, Patches, Features
            """
            emb_keys = [k for k in self.extra_encoders.keys() if k != "action"]
            cost = 0.0

            # Retrieve config settings
            criterion_type = cfg.criterion.type
            loss_type = getattr(cfg.criterion, "loss_type", "mse")  # Default to mse if not set

            for key in emb_keys + ["pixels"]:
                preds = info_dict[f"predicted_{key}_embed"]  # (B, N, T, P, D)
                goal = info_dict[f"{key}_goal_embed"]  # (B, N, P, D)

                # Align Dimensions
                # We need goal to be (B, N, 1, P, D) to broadcast against Time (Dim 2)
                if goal.ndim == preds.ndim - 1:
                    goal_expanded = goal.unsqueeze(2)
                else:
                    goal_expanded = goal

                # Select Targets based on Criterion Strategy
                if criterion_type == "final_distance":
                    # Select only the last time step
                    # preds slice: (B, N, 1, P, D)
                    preds_target = preds[:, :, -1:]
                    # goal stays (B, N, 1, P, D) via broadcasting
                    target_goal = goal_expanded
                else:
                    # Use full horizon
                    preds_target = preds
                    target_goal = goal_expanded.expand_as(preds)

                # Compute Loss (Result should be (B, N, T) per sample/time)
                if loss_type == "cosine":
                    # Cosine Sim reduces the last dim (Features) automatically
                    # Input: (B, N, T, P, D) -> Sim: (B, N, T, P)
                    sim = F.cosine_similarity(preds_target, target_goal, dim=-1)
                    raw_loss = 1.0 - sim

                    # Average over Patches (Dim 3) -> Result (B, N, T)
                    step_loss = raw_loss.mean(dim=3)

                else:  # mse
                    # MSE keeps all dims: (B, N, T, P, D)
                    raw_loss = F.mse_loss(preds_target, target_goal, reduction="none")

                    # Average over Patches(3) and Features(4) -> Result (B, N, T)
                    step_loss = raw_loss.mean(dim=(3, 4))

                # Aggregate Costs
                if criterion_type == "final_distance":
                    cost += step_loss.mean(dim=2)

                elif criterion_type == "current_distance":
                    # step_loss is (B, N, T)
                    T = preds.shape[2]

                    # Create discounts (1, 1, T) to broadcast against (B, N, T)
                    discounts = torch.tensor([cfg.criterion.discount**i for i in range(T)], device=preds.device).view(
                        1, 1, -1
                    )

                    # Weighted sum over time
                    cost += (step_loss * discounts).mean(dim=2)

            # Action Regularization
            if cfg.criterion.action_reg > 0:
                # action_candidates: (B, N, T, A)
                # Mean over Time(2) and Actions(3)
                cost += cfg.criterion.action_reg * (action_candidates**2).mean(dim=(2, 3))

            return cost

        # Assign to model
        model.criterion = types.MethodType(criterion, model)

        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
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

    dataset = swm.data.FrameDataset(cfg.eval.dataset_name)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables={
            "_set_state": "state",
            "_set_goal_state": "goal_state",
        },
    )
    end_time = time.time()

    if cfg.wandb.use_wandb:
        # Log metrics to wandb
        wandb.log(metrics)
        # Finish wandb run
        wandb.finish()

    # dump results
    print(metrics)
    # ---- dump results to a txt file ----
    results_path = Path(__file__).parent / cfg.output.filename
    if torch.cuda.is_available():
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
    else:
        gpu_name = "No GPU available"
    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs
        f.write(f"policy: {cfg.policy}\n")
        f.write(f"dataset_name: {cfg.eval.dataset_name}\n")
        f.write(f"goal_offset_steps: {cfg.eval.goal_offset_steps}\n")
        f.write(f"eval_budget: {cfg.eval.eval_budget}\n")
        f.write(f"horizon: {cfg.plan_config.horizon}\n")
        f.write(f"receding_horizon: {cfg.plan_config.receding_horizon}\n")
        f.write(f"seed: {cfg.seed}\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"number of samples: {cfg.sollver.num_samples}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")
        f.write(f"gpu: {gpu_name}\n")


if __name__ == "__main__":
    run()
