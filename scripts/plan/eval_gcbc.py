import os


os.environ['MUJOCO_GL'] = 'egl'


import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
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
            transforms.Resize(size=224),
            transforms.CenterCrop(size=224),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(dataset_name, cache_dir=dataset_path)
    return dataset


@hydra.main(version_base=None, config_path='.', config_name='config')
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), 'Planning horizon must be smaller than or equal to eval_budget'

    if cfg.wandb.enable:
        # Initialize wandb
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=dict(cfg),
        )

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(
        **cfg.world, image_shape=(224, 224), render_mode='rgb_array'
    )

    # create the transform
    transform = {
        'pixels': img_transform(),
        'goal': img_transform(),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)

    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )
    ep_indices, _ = np.unique(
        dataset.get_col_data(col_name), return_index=True
    )

    # create the processing
    action_process = preprocessing.StandardScaler()
    action_process.fit(dataset.get_col_data('action'))

    proprio_process = preprocessing.StandardScaler()
    proprio_process.fit(dataset.get_col_data('proprio'))

    process = {
        'action': action_process,
        'proprio': proprio_process,
        'goal_proprio': proprio_process,
    }

    # -- run evaluation
    policy = cfg.get('policy', 'random')

    if policy != 'random':
        model = swm.policy.AutoActionableModel(cfg.policy)
        model = model.to('cuda')
        model = model.eval()
        model.requires_grad_(False)

        policy = swm.policy.FeedForwardPolicy(
            model=model, process=process, transform=transform
        )
    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != 'random'
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data('step_idx') <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), 'valid starting points found for evaluation.')

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)['step_idx']

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError(
            'Not enough episodes with sufficient length for evaluation.'
        )

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(
            cfg.eval.get('callables'), resolve=True
        ),
        video_path=results_path,
    )
    end_time = time.time()

    if cfg.wandb.enable:
        # Log metrics to wandb
        wandb.log(metrics)
        # Finish wandb run
        wandb.finish()

    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open('a') as f:
        f.write('\n')  # separate from previous runs

        f.write('==== CONFIG ====\n')
        f.write(OmegaConf.to_yaml(cfg))
        f.write('\n')

        f.write('==== RESULTS ====\n')
        f.write(f'metrics: {metrics}\n')
        f.write(f'evaluation_time: {end_time - start_time} seconds\n')


if __name__ == '__main__':
    run()
