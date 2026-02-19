import os


os.environ['MUJOCO_GL'] = 'egl'

from collections import OrderedDict

import gymnasium as gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import torch
from einops import rearrange
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.transforms import v2 as transforms
from tqdm import tqdm
from transformers import AutoModel, AutoModelForImageClassification

import stable_worldmodel as swm
from stable_worldmodel.wrapper import MegaWrapper, VariationWrapper

from utils import get_state_grid


DINO_PATCH_SIZE = 14  # DINO encoder uses 14x14 patches

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

    # create the processing
    ACTION_MEAN = np.array([-0.0087, 0.0068])
    ACTION_STD = np.array([0.2019, 0.2002])
    PROPRIO_MEAN = np.array([236.6155, 264.5674, -2.93032027, 2.54307914])
    PROPRIO_STD = np.array([101.1202, 87.0112, 74.84556075, 74.14009094])

    obs_space = env.unwrapped.observation_space
    action_space = env.unwrapped.action_space

    def _space_dim(space):
        if space is None or not hasattr(space, 'shape') or not space.shape:
            return None
        return space.shape[-1]

    def _obs_dim(key):
        if hasattr(obs_space, 'spaces') and key in obs_space.spaces:
            return _space_dim(obs_space.spaces[key])
        return None

    def _make_scaler(key, target_dim, mean, std):
        if target_dim is None:
            logging.warning(
                f"Missing target dim for '{key}', skipping standardization."
            )
            return None

        mean = np.asarray(mean) if mean is not None else None
        std = np.asarray(std) if std is not None else None

        if (
            mean is None
            or std is None
            or mean.shape[0] != target_dim
            or std.shape[0] != target_dim
        ):
            logging.warning(
                f"Stats for '{key}' do not match dim {target_dim}; using identity standardization."
            )
            mean = np.zeros(target_dim, dtype=np.float32)
            std = np.ones(target_dim, dtype=np.float32)

        scaler = preprocessing.StandardScaler()
        scaler.mean_ = mean
        scaler.scale_ = std
        scaler.var_ = std**2
        scaler.n_features_in_ = target_dim
        return scaler

    action_dim = _space_dim(action_space)
    proprio_dim = _obs_dim('proprio')
    goal_proprio_dim = _obs_dim('goal_proprio') or proprio_dim

    action_process = _make_scaler(
        'action', action_dim, ACTION_MEAN, ACTION_STD
    )
    proprio_process = _make_scaler(
        'proprio', proprio_dim, PROPRIO_MEAN, PROPRIO_STD
    )
    goal_proprio_process = _make_scaler(
        'goal_proprio', goal_proprio_dim, PROPRIO_MEAN, PROPRIO_STD
    )

    process = {}
    if action_process is not None:
        process['action'] = action_process
    if proprio_process is not None:
        process['proprio'] = proprio_process
    if goal_proprio_process is not None:
        process['goal_proprio'] = goal_proprio_process

    with open_dict(cfg) as cfg:
        cfg.extra_dims = {}
        for key in cfg.world_model.get('encoding', {}):
            if key == 'action':
                inpt_dim = env.unwrapped.action_space.shape[1]
            elif hasattr(obs_space, 'spaces') and key in obs_space.spaces:
                inpt_dim = obs_space.spaces[key].shape[1]
            elif hasattr(obs_space, 'spaces') and key not in obs_space.spaces:
                raise ValueError(
                    f"Encoding key '{key}' not found in dataset columns."
                )
            else:
                inpt_dim = obs_space.shape[0]
            cfg.extra_dims[key] = inpt_dim

    return env, process, transform


def prepare_info(info_dict, process, transform):
    # pre-process and transform observations
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
# Model Architecture
# ============================================================================


def get_encoder(cfg):
    """Factory function to create encoder based on backbone type."""

    # Define encoder configurations
    ENCODER_CONFIGS = {
        'resnet': {
            'prefix': 'microsoft/resnet-',
            'model_class': AutoModelForImageClassification,
            'embedding_attr': lambda model: model.config.hidden_sizes[-1],
            'post_init': lambda model: setattr(
                model.classifier, '1', torch.nn.Identity()
            ),
            'interpolate_pos_encoding': False,
        },
        'vit': {'prefix': 'google/vit-'},
        'dino': {'prefix': 'facebook/dino-'},
        'dinov2': {'prefix': 'facebook/dinov2-'},
        'dinov3': {
            'prefix': 'facebook/dinov3-'
        },  # TODO handle resnet base in dinov3
        'mae': {'prefix': 'facebook/vit-mae-'},
        'ijepa': {'prefix': 'facebook/ijepa'},
        'vjepa2': {'prefix': 'facebook/vjepa2-vit'},
        'siglip2': {'prefix': 'google/siglip2-'},
    }

    # Find matching encoder
    encoder_type = None
    for name, config in ENCODER_CONFIGS.items():
        if cfg.world_model.backbone.name.startswith(config['prefix']):
            encoder_type = name
            break

    if encoder_type is None:
        raise ValueError(
            f'Unsupported backbone: {cfg.world_model.backbone.name}'
        )

    config = ENCODER_CONFIGS[encoder_type]

    # Load model
    backbone = config.get('model_class', AutoModel).from_pretrained(
        cfg.world_model.backbone.name
    )

    # CLIP style model
    if hasattr(backbone, 'vision_model'):
        backbone = backbone.vision_model

    # Post-initialization if needed (e.g., ResNet)
    if 'post_init' in config:
        config['post_init'](backbone)

    # Get embedding dimension
    embedding_dim = config.get(
        'embedding_attr', lambda model: model.config.hidden_size
    )(backbone)

    # Determine number of patches
    is_cnn = encoder_type == 'resnet'
    num_patches = 1 if is_cnn else (cfg.image_size // cfg.patch_size) ** 2

    interp_pos_enc = config.get('interpolate_pos_encoding', True)

    return backbone, embedding_dim, num_patches, interp_pos_enc


def get_world_model(cfg):
    """Load and setup world model.
    For visualization, we only need the model to implement the `encode` method."""

    if cfg.world_model.model_name is not None:
        model = swm.policy.AutoCostModel(cfg.world_model.model_name).to(
            cfg.get('device', 'cpu')
        )
        model = model.to(cfg.get('device', 'cpu'))
        model = model.eval()
    else:  # no checkpoint found, build model from scratch
        encoder, embedding_dim, num_patches, interp_pos_enc = get_encoder(cfg)
        embedding_dim += sum(
            emb_dim for emb_dim in cfg.world_model.get('encoding', {}).values()
        )  # add all extra dims

        logging.info(f'Patches: {num_patches}, Embedding dim: {embedding_dim}')

        # Build causal predictor (transformer that predicts next latent states)

        print('>>>> DIM PREDICTOR:', embedding_dim)

        predictor = swm.wm.prejepa.CausalPredictor(
            num_patches=num_patches,
            num_frames=cfg.world_model.history_size,
            dim=embedding_dim,
            **cfg.world_model.predictor,
        )

        # Build action and proprioception encoders
        extra_encoders = OrderedDict()
        for key, emb_dim in cfg.world_model.get('encoding', {}).items():
            inpt_dim = cfg.extra_dims[key]
            extra_encoders[key] = swm.wm.prejepa.Embedder(
                in_chans=inpt_dim, emb_dim=emb_dim
            )
            print(
                f'Build encoder for {key} with input dim {inpt_dim} and emb dim {emb_dim}'
            )

        extra_encoders = torch.nn.ModuleDict(extra_encoders)

        # Assemble world model
        model = swm.wm.prejepa.PreJEPA(
            encoder=spt.backbone.EvalOnly(encoder),
            predictor=predictor,
            extra_encoders=extra_encoders,
            history_size=cfg.world_model.history_size,
            num_pred=cfg.world_model.num_preds,
            interpolate_pos_encoding=interp_pos_enc,
        )
        model.to(cfg.get('device', 'cpu'))
        model = model.eval()
    return model


# ============================================================================
# Computing Embeddings
# ============================================================================


def collect_embeddings(world_model, env, process, transform, cfg):
    """Go through the environment and collect embeddings using the world model."""

    grid, state_grid = get_state_grid(
        env.unwrapped.envs[0].unwrapped, cfg.env.grid_size
    )
    embeddings = []
    pixels = []
    for variation_cfg in cfg.env.variations:
        variation_embeddings = []
        variation_pixels = []
        for i, state in tqdm(
            enumerate(state_grid), desc='Collecting embeddings'
        ):
            options = {'state': state}
            default_variation = cfg.env.get('default_variation', None)
            if default_variation is not None:
                options['variation'] = list(default_variation)
            # for the first state of each variation, add variation options
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
            infos = world_model.encode(infos, target='embed')
            if cfg.world_model.get(
                'backbone_only', False
            ):  # use only vision backbone embeddings
                variation_embeddings.append(
                    infos['pixels_embed'].cpu().detach()
                )
            else:  # use full model embeddings (proropio + action + vision)
                variation_embeddings.append(infos['embed'].cpu().detach())
            variation_pixels.append(infos['pixels'][0].cpu().detach())
        embeddings.append(variation_embeddings)
        pixels.append(variation_pixels)

    return grid, embeddings, pixels


# ============================================================================
# Dimensionality Reduction
# ============================================================================


def compute_dimensionality_reduction(embeddings, cfg):
    """
    Computes t-SNE projection on the collected embeddings.
    """
    logging.info(
        f'Computing dimensionality reduction with {cfg.dimensionality_reduction}'
    )
    # Flatten if embeddings are spatial (e.g. from patch tokens)
    # Shape: (N_samples, Embedding_Dim)
    embeddings = rearrange(embeddings, 'b ... -> b (...)')

    n_samples = embeddings.shape[0]

    # Perplexity must be < n_samples. Default is 30, which breaks for small grids.
    perplexity = min(30, n_samples - 1) if n_samples > 1 else 1

    # Initialize and fit t-SNE
    if cfg.dimensionality_reduction == 'tsne':
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=cfg.get('seed', 42),
        )
        embeddings_2d = tsne.fit_transform(embeddings)
    elif cfg.dimensionality_reduction == 'pca':
        pca = PCA(n_components=2, random_state=cfg.seed)
        embeddings_2d = pca.fit_transform(embeddings)

    return embeddings_2d


# ============================================================================
# Visualization
# ============================================================================


def plot_distance_maps(
    grid, embeddings, pixels, save_path='distance_maps.pdf'
):
    """
    Plots pairs of (Reference Image, Distance Heatmap).

    Args:
        grid: (H, W, 2) array of physical state coordinates.
        embeddings: (H, W, D) array of high-dim embeddings.
        pixels: (H, W, C, H_img, W_img) or (H, W, H_img, W_img, C) array of images.
    """
    height, width = grid.shape[0], grid.shape[1]
    X = grid[:, :, 0]
    Y = grid[:, :, 1]

    # Define Reference Indices (Top-Left, Top-Right, Center, Bottom-Center)
    ref_indices = [
        (0, 0),
        (0, width - 1),
        (height // 2, width // 2),
        (height - 1, width // 2),
    ]

    # Create subplots: 4 rows (one per reference), 2 columns (Image, Heatmap)
    # You can also do 1 row with 8 cols, but 4x2 is usually clearer for pairs.
    fig, axes = plt.subplots(
        len(ref_indices), 2, figsize=(10, 4 * len(ref_indices))
    )

    for i, (r_idx, c_idx) in enumerate(ref_indices):
        ax_img = axes[i, 0]
        ax_map = axes[i, 1]

        # --- A. Plot Reference Image ---
        ref_img = pixels[r_idx, c_idx]

        # Handle format: Un-normalize and Channels First/Last
        if ref_img.min() < 0:
            ref_img = (ref_img * 0.5) + 0.5

        if ref_img.shape[0] in [1, 3]:
            ref_img = np.moveaxis(ref_img, 0, -1)

        ax_img.imshow(ref_img.clip(0, 1))
        ax_img.set_title('Reference Observation')
        ax_img.axis('off')

        # --- B. Plot Distance Heatmap ---
        ref_emb = embeddings[r_idx, c_idx]
        dists = np.linalg.norm(embeddings - ref_emb, axis=-1)

        contour = ax_map.contourf(X, Y, dists, levels=50, cmap='viridis')
        ax_map.invert_yaxis()  # Match image coordinate system (y axis going downwards)

        # Mark reference on the map too
        ref_x = X[r_idx, c_idx]
        ref_y = Y[r_idx, c_idx]
        ax_map.scatter(
            ref_x, ref_y, c='red', marker='X', s=100, edgecolors='white'
        )

        ax_map.set_title(f'Distance from ({ref_x:.2f}, {ref_y:.2f})')
        ax_map.axis('off')

        # Add colorbar for this row
        cbar = fig.colorbar(contour, ax=ax_map, fraction=0.046, pad=0.04)
        cbar.set_label('L2 Distance')

    plt.suptitle('Latent Distances vs. Visual Input', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save as PDF
    plt.savefig(save_path, format='pdf')
    logging.info(f'Distance maps saved to {save_path}')
    plt.close(fig)


def plot_representations(
    grid,
    representations_2d,
    variations_cfg,
    samples_per_variation,
    title_suffix='Latent Space',
    save_path='latent_vis.pdf',
):
    """
    Plots the ground truth grid and the 2D representations side-by-side.
    Colors are generated based on the grid position to visualize topology.
    Different variations are plotted with different markers.
    """
    # Create colors based on grid position (Normalized x, y -> R, G, 0)
    # This color map applies to ONE grid. We will reuse it for every variation.
    grid_norm = (grid - grid.min(axis=0)) / (
        grid.max(axis=0) - grid.min(axis=0) + 1e-6
    )

    colors = np.zeros((len(grid), 4))
    colors[:, 0] = grid_norm[:, 0]  # Red varies with dimension 0
    colors[:, 1] = grid_norm[:, 1]  # Green varies with dimension 1
    colors[:, 2] = 0.5  # Constant Blue
    colors[:, 3] = 1.0  # Alpha

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Plot 1: Ground Truth Grid (Physical State) ---
    # We only plot the grid once as it is the same for all variations
    axes[0].scatter(
        grid[:, 0],
        grid[:, 1],
        c=colors,
        s=50,
        edgecolor='k',
        alpha=0.8,
        marker='o',
    )
    axes[0].set_title('Physical State Grid')
    axes[0].set_xlabel('State Dim 0')
    axes[0].set_ylabel('State Dim 1')
    axes[0].grid(True, linestyle='--', alpha=0.3)

    # --- Plot 2: 2D Representation (All Variations) ---
    # Define a list of markers to distinguish variations
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']

    for var_idx in range(len(variations_cfg)):
        # Calculate indices for this variation
        start_idx = var_idx * samples_per_variation
        end_idx = start_idx + samples_per_variation

        # Extract subset of t-SNE points for this variation
        subset_2d = representations_2d[start_idx:end_idx]

        # Cycle through markers if variations > len(markers)
        marker = markers[var_idx % len(markers)]

        var_label = (
            f'Var {variations_cfg[var_idx]["variation"]["fields"][0]}'
            if variations_cfg[var_idx]['variation']['fields'] is not None
            else 'Original'
        )
        axes[1].scatter(
            subset_2d[:, 0],
            subset_2d[:, 1],
            c=colors,  # Reuse the same topology colors
            s=50,
            edgecolor='k',
            alpha=0.8,
            marker=marker,
            label=var_label,
        )

    axes[1].set_title(f'2D Projection ({title_suffix})')
    axes[1].set_xlabel('Projected Dim 1')
    axes[1].set_ylabel('Projected Dim 2')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    axes[1].legend(loc='best', title='Variations')

    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    logging.info(f'Visualization saved to {save_path}')
    plt.close(fig)


def make_runtime_cfg(global_cfg, dataset_cfg):
    return OmegaConf.merge(
        {
            'device': global_cfg.device,
            'seed': global_cfg.seed,
            'image_size': global_cfg.image_size,
            'patch_size': global_cfg.patch_size,
            'dimensionality_reduction': global_cfg.dimensionality_reduction,
            'cache_dir': global_cfg.cache_dir,
        },
        dataset_cfg,
    )


# ===========================================================================
# Main function
# ===========================================================================


@hydra.main(
    version_base=None, config_path='./configs', config_name='config_envs'
)
def run(cfg):
    """Run visualization script for all datasets/envs."""
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
        world_model = get_world_model(local_cfg)

        model_name = (
            wm_cfg.model_name
            if wm_cfg.model_name is not None
            else wm_cfg.backbone.type
        )

        # --- Collect embeddings ---
        logging.info('Computing embeddings from environment...')
        grid, embeddings_variations, pixels_variations = collect_embeddings(
            world_model, env, process, transform, local_cfg
        )

        # --- Prepare embeddings for dimensionality reduction ---
        all_embeddings_list = []
        for var_list in embeddings_variations:
            var_emb = torch.cat(var_list, dim=0).cpu().numpy()
            var_emb = rearrange(var_emb, 'b ... -> b (...)')
            all_embeddings_list.append(var_emb)

        all_embeddings_global = np.concatenate(all_embeddings_list, axis=0)

        num_variations = len(all_embeddings_list)
        samples_per_variation = all_embeddings_list[0].shape[0]

        # --- Dimensionality reduction ---
        logging.info(
            f'Computing global {local_cfg.dimensionality_reduction} '
            f'for {num_variations} variations '
            f'({all_embeddings_global.shape[0]} samples)...'
        )
        representations_2d = compute_dimensionality_reduction(
            all_embeddings_global, local_cfg
        )

        # --- Plot latent space ---
        dr_save_path = f'{dataset_name}_{model_name}_{local_cfg.dimensionality_reduction}.pdf'

        plot_representations(
            grid,
            representations_2d,
            variations_cfg=env_cfg.variations,
            samples_per_variation=samples_per_variation,
            title_suffix=f'{dataset_name} Latent Space',
            save_path=dr_save_path,
        )

        # --- Distance maps per variation ---
        grid_size = env_cfg.grid_size

        for var_idx, var_pixels_list in enumerate(pixels_variations):
            embeddings = all_embeddings_list[var_idx]
            pixels = torch.cat(var_pixels_list, dim=0).cpu().numpy()

            var_suffix = (
                f'var_{env_cfg.variations[var_idx]["variation"]["fields"][0]}'
                if env_cfg.variations[var_idx]['variation']['fields']
                is not None
                else 'var_original'
            )

            distmap_save_path = (
                f'{dataset_name}_{model_name}_{var_suffix}_distmap.pdf'
            )

            plot_distance_maps(
                grid.reshape(grid_size, grid_size, -1),
                embeddings.reshape(grid_size, grid_size, -1),
                pixels.reshape(grid_size, grid_size, *pixels.shape[1:]),
                save_path=distmap_save_path,
            )

        logging.info(f'Finished dataset: {dataset_name}')

    logging.info('All datasets processed.')


if __name__ == '__main__':
    run()
