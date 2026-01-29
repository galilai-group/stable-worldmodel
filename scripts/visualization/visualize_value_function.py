import os
from pathlib import Path


os.environ['MUJOCO_GL'] = 'egl'

import gymnasium as gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger as logging
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

import stable_worldmodel as swm
from stable_worldmodel.wrapper import MegaWrapper, VariationWrapper

from utils import get_state_grid


# ============================================================================
# Dataset Loading
# ============================================================================


def get_dataset(cfg, dataset_name):
    """Load a dataset for preprocessing statistics."""
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(dataset_name, cache_dir=dataset_path)
    return dataset


# ============================================================================
# Setting up Environment, transform and processing
# ============================================================================


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


def get_env(cfg):
    """Setup dataset with image transforms and normalization."""

    env = gym.make_vec(
        cfg.env.env_name,
        num_envs=1,
        vectorization_mode='sync',
        wrappers=[
            lambda x: MegaWrapper(
                x,
                image_shape=(cfg.image_size, cfg.image_size),
                pixels_transform=None,
                goal_transform=None,
                history_size=cfg.env.history_size,
                frame_skip=cfg.env.frame_skip,
            )
        ]
        + ([]),
        max_episode_steps=50,
        render_mode='rgb_array',
    )

    env = VariationWrapper(env)
    env.unwrapped.autoreset_mode = gym.vector.AutoresetMode.DISABLED

    # create the transform
    transform = {
        'pixels': img_transform(),
        'goal': img_transform(),
    }

    # create the processing by fitting scalers on the training dataset
    training_dataset_name = cfg.get('training_dataset', None)
    if training_dataset_name is None:
        raise ValueError(
            'training_dataset must be specified in config to compute preprocessing stats'
        )

    dataset = get_dataset(cfg, training_dataset_name)
    logging.info(
        f'Fitting preprocessing scalers on dataset: {training_dataset_name}'
    )

    action_process = preprocessing.StandardScaler()
    action_process.fit(dataset.get_col_data('action'))

    proprio_process = preprocessing.StandardScaler()
    proprio_process.fit(dataset.get_col_data('proprio'))

    process = {
        'action': action_process,
        'proprio': proprio_process,
        'goal_proprio': proprio_process,
    }

    return env, process, transform


def prepare_info(info_dict, process, transform):
    """Pre-process and transform observations."""
    for k, v in info_dict.items():
        is_numpy = isinstance(v, (np.ndarray | np.generic))

        if k in process:
            if not is_numpy:
                raise ValueError(
                    f"Expected numpy array for key '{k}' in process, got {type(v)}"
                )

            # flatten extra dimensions if needed
            shape = v.shape
            if len(shape) > 2:
                v = v.reshape(-1, *shape[2:])

            # process and reshape back
            v = process[k].transform(v)
            v = v.reshape(shape)

        # collapse env and time dimensions for transform (e, t, ...) -> (e * t, ...)
        # then restore after transform
        if k in transform:
            shape = None
            if is_numpy or torch.is_tensor(v):
                if v.ndim > 2:
                    shape = v.shape
                    v = v.reshape(-1, *shape[2:])

            v = torch.stack([transform[k](x) for x in v])
            is_numpy = isinstance(v, (np.ndarray | np.generic))

            if shape is not None:
                v = v.reshape(*shape[:2], *v.shape[1:])

        if is_numpy and v.dtype.kind not in 'USO':
            v = torch.from_numpy(v)

        info_dict[k] = v

    return info_dict


# ============================================================================
# Model Loading
# ============================================================================


def get_iql_model(cfg):
    """Load IQL model with value function.

    The model must implement:
    - encode(info, pixels_key, emb_keys, prefix, target): encode observations
    - predict_values(embedding, embedding_goal): predict values
    """

    model = swm.policy.AutoActionableModel(cfg.world_model.model_name)
    model = model.to(cfg.get('device', 'cpu'))
    model = model.eval()

    # Verify the model has the required methods
    if not hasattr(model, 'predict_values'):
        raise ValueError(
            f"Model {cfg.world_model.model_name} does not have a 'predict_values' method. "
            'This script requires an IQL model with a value function.'
        )

    return model


# ============================================================================
# Computing Value Function Grid
# ============================================================================


def collect_embeddings_and_values(
    model, env, process, transform, cfg, ref_indices
):
    """Collect embeddings and compute value function for state grid.

    Args:
        model: IQL model with encode and predict_values methods
        env: Gymnasium environment
        process: Data preprocessing dictionary
        transform: Image transform dictionary
        cfg: Configuration object
        ref_indices: List of (row, col) tuples for reference states

    Returns:
        grid: (N, 2) array of grid coordinates
        embeddings: List of embeddings for each variation
        pixels: List of pixel observations for each variation
        values_per_ref: Dict mapping ref_idx to value arrays for each variation
    """
    grid, state_grid = get_state_grid(
        env.unwrapped.envs[0].unwrapped, cfg.env.grid_size
    )

    embeddings = []
    pixels = []

    # Determine which keys to use for encoding based on model's extra_encoders
    emb_keys = (
        list(model.extra_encoders.keys())
        if hasattr(model, 'extra_encoders')
        else []
    )

    # First pass: collect all embeddings and pixels
    for variation_cfg in cfg.env.variations:
        variation_embeddings = []
        variation_pixels = []

        for i, state in tqdm(
            enumerate(state_grid),
            desc='Collecting embeddings',
            total=len(state_grid),
        ):
            options = {'state': state}
            default_variation = cfg.env.get('default_variation', None)
            if default_variation is not None:
                options['variation'] = list(default_variation)

            if variation_cfg.variation['fields'] is not None:
                assert variation_cfg.variation['values'] is not None and len(
                    variation_cfg.variation['fields']
                ) == len(variation_cfg.variation['values']), (
                    'Both fields and values must be provided for variation.'
                )
                options['variation_values'] = dict(
                    zip(
                        variation_cfg.variation['fields'],
                        variation_cfg.variation['values'],
                    )
                )

            _, infos = env.reset(options=options)
            infos = prepare_info(infos, process, transform)

            for key in infos:
                if isinstance(infos[key], torch.Tensor):
                    infos[key] = infos[key].to(cfg.get('device', 'cpu'))

            # Replace NaN values with 0 (occurs at sequence boundaries)
            for key in emb_keys:
                if key in infos and torch.is_tensor(infos[key]):
                    infos[key] = torch.nan_to_num(infos[key], 0.0)

            # Encode the observation
            with torch.no_grad():
                infos = model.encode(
                    infos,
                    pixels_key='pixels',
                    emb_keys=emb_keys,
                    target='embed',
                )

            variation_embeddings.append(infos['embed'].cpu().detach())
            variation_pixels.append(infos['pixels'][0].cpu().detach())

        embeddings.append(variation_embeddings)
        pixels.append(variation_pixels)

    # Stack embeddings for value computation
    grid_size = cfg.env.grid_size
    values_per_ref = {}

    for var_idx in range(len(cfg.env.variations)):
        var_embeddings = torch.cat(embeddings[var_idx], dim=0)  # (N, T, P, D)

        for ref_idx in ref_indices:
            r_idx, c_idx = ref_idx
            flat_ref_idx = r_idx * grid_size + c_idx

            # Get reference embedding as goal - use last timestep only (B, 1, P, D)
            ref_embedding = var_embeddings[
                flat_ref_idx : flat_ref_idx + 1, -1:, :, :
            ]

            # Compute values for all states relative to this reference
            values = []
            batch_size = 32

            for batch_start in range(0, len(var_embeddings), batch_size):
                batch_end = min(batch_start + batch_size, len(var_embeddings))
                batch_embeddings = var_embeddings[batch_start:batch_end].to(
                    cfg.get('device', 'cpu')
                )
                # Goal embedding should be (B, 1, P, D)
                batch_ref = ref_embedding.expand(
                    batch_end - batch_start, -1, -1, -1
                ).to(cfg.get('device', 'cpu'))

                with torch.no_grad():
                    batch_values = model.predict_values(
                        batch_embeddings, batch_ref
                    )
                    values.append(batch_values.cpu())

            values = torch.cat(values, dim=0)  # (N, T, 1)

            # Take value from last timestep
            if values.dim() == 3:
                values = values[:, -1, 0]  # (N,)
            else:
                values = values.squeeze(-1)  # (N,)

            if ref_idx not in values_per_ref:
                values_per_ref[ref_idx] = []
            values_per_ref[ref_idx].append(values.numpy())

    return grid, embeddings, pixels, values_per_ref


# ============================================================================
# Visualization
# ============================================================================


def plot_value_maps(
    grid, pixels, values_per_ref, grid_size, save_path='value_maps.pdf'
):
    """Plot value function maps for reference states.

    Args:
        grid: (N, 2) array of physical state coordinates
        pixels: List of pixel observations
        values_per_ref: Dict mapping ref_idx to value arrays
        grid_size: Size of the grid
        save_path: Output file path
    """
    height, width = grid_size, grid_size
    grid_2d = grid.reshape(height, width, -1)
    X = grid_2d[:, :, 0]
    Y = grid_2d[:, :, 1]

    ref_indices = list(values_per_ref.keys())

    # Create subplots: rows for references, 2 columns (Image, Value Map)
    fig, axes = plt.subplots(
        len(ref_indices), 2, figsize=(10, 4 * len(ref_indices))
    )

    if len(ref_indices) == 1:
        axes = axes.reshape(1, -1)

    for i, ref_idx in enumerate(ref_indices):
        r_idx, c_idx = ref_idx
        flat_ref_idx = r_idx * width + c_idx

        ax_img = axes[i, 0]
        ax_map = axes[i, 1]

        # --- A. Plot Reference Image ---
        ref_img = pixels[flat_ref_idx]

        # Handle format: Un-normalize and Channels First/Last
        if ref_img.min() < 0:
            ref_img = (ref_img * 0.5) + 0.5

        if ref_img.shape[0] in [1, 3]:
            ref_img = np.moveaxis(ref_img, 0, -1)

        ax_img.imshow(ref_img.clip(0, 1))
        ax_img.set_title('Goal Observation')
        ax_img.axis('off')

        # --- B. Plot Value Map ---
        # Use first variation's values
        values = values_per_ref[ref_idx][0]
        values_2d = values.reshape(height, width)

        # print statistics of values for this reference
        logging.info(
            f'Reference {ref_idx}: Value stats - min: {values.min():.4f}, max: {values.max():.4f}, mean: {values.mean():.4f}, std: {values.std():.4f}'
        )

        # Negate values since higher value = closer to goal (less "distance")
        # For visualization, we want lower values (darker) = closer to goal
        display_values = -values_2d

        contour = ax_map.contourf(
            X, Y, display_values, levels=50, cmap='viridis'
        )
        ax_map.invert_yaxis()  # Match image coordinate system

        # Mark reference on the map
        ref_x = X[r_idx, c_idx]
        ref_y = Y[r_idx, c_idx]
        ax_map.scatter(
            ref_x, ref_y, c='red', marker='X', s=100, edgecolors='white'
        )

        ax_map.set_title(f'goal at ({ref_x:.2f}, {ref_y:.2f})')
        ax_map.axis('off')

        # Add colorbar
        cbar = fig.colorbar(contour, ax=ax_map, fraction=0.046, pad=0.04)
        cbar.set_label('Negative Value (lower = better)')

    plt.suptitle('Value Function Visualization', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, format='pdf')
    logging.info(f'Value maps saved to {save_path}')
    plt.close(fig)


def make_runtime_cfg(global_cfg, dataset_cfg):
    return OmegaConf.merge(
        {
            'device': global_cfg.device,
            'seed': global_cfg.seed,
            'image_size': global_cfg.image_size,
            'patch_size': global_cfg.patch_size,
            'cache_dir': global_cfg.cache_dir,
        },
        dataset_cfg,
    )


# ===========================================================================
# Main function
# ===========================================================================


@hydra.main(
    version_base=None, config_path='./configs', config_name='config_value'
)
def run(cfg):
    """Run value function visualization script."""
    cache_dir = swm.data.utils.get_cache_dir()
    cfg.cache_dir = cache_dir

    for dataset_name, dataset_cfg in cfg.datasets.items():
        logging.info('==============================')
        logging.info(f'Processing dataset: {dataset_name}')
        logging.info('==============================')

        local_cfg = make_runtime_cfg(cfg, dataset_cfg)
        wm_cfg = local_cfg.world_model
        env_cfg = local_cfg.env

        # --- Setup env and model ---
        env, process, transform = get_env(local_cfg)
        model = get_iql_model(local_cfg)

        model_name = wm_cfg.model_name

        # Define reference indices for value map visualization
        grid_size = env_cfg.grid_size
        ref_indices = [
            (0, 0),
            (0, grid_size - 1),
            (grid_size // 2, grid_size // 2),
            (grid_size - 1, grid_size // 2),
        ]

        # --- Collect embeddings and compute values ---
        logging.info('Computing embeddings and value function...')
        grid, embeddings, pixels_variations, values_per_ref = (
            collect_embeddings_and_values(
                model, env, process, transform, local_cfg, ref_indices
            )
        )

        # --- Plot value maps for each variation ---
        for var_idx, var_pixels_list in enumerate(pixels_variations):
            pixels = torch.cat(var_pixels_list, dim=0).cpu().numpy()

            var_suffix = (
                f'var_{env_cfg.variations[var_idx]["variation"]["fields"][0]}'
                if env_cfg.variations[var_idx]['variation']['fields']
                is not None
                else 'var_original'
            )

            valuemap_save_path = (
                f'{dataset_name}_{model_name}_{var_suffix}_valuemap.pdf'
            )

            plot_value_maps(
                grid,
                pixels,
                values_per_ref,
                grid_size,
                save_path=valuemap_save_path,
            )

        logging.info(f'Finished dataset: {dataset_name}')

    logging.info('All datasets processed.')


if __name__ == '__main__':
    run()
