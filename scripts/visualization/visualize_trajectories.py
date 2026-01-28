from collections import OrderedDict
from pathlib import Path

import datasets
import hydra
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import torch
from einops import rearrange
from loguru import logger as logging
from omegaconf import open_dict
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import AutoModel, AutoModelForImageClassification

import stable_worldmodel as swm


DINO_PATCH_SIZE = 14  # DINO encoder uses 14x14 patches

# ============================================================================
# Data Loading
# ============================================================================


def get_episodes_length(dataset, episodes):
    episode_idx = dataset['episode_idx'][:]
    step_idx = dataset['step_idx'][:]
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_data(cfg, dataset_cfg, model_cfg):
    """Setup dataset with image transforms and normalization."""

    # Load dataset
    dataset_path = Path(
        cfg.cache_dir or swm.data.utils.get_cache_dir(),
        dataset_cfg.dataset_name,
    )
    dataset = datasets.load_from_disk(dataset_path).with_format('numpy')

    # Get unique episode indices
    ep_indices = np.unique(dataset['episode_idx'][:])

    # Compute episode lengths
    episode_len = get_episodes_length(dataset, ep_indices)

    # Randomly sample episodes
    g = np.random.default_rng(cfg.seed)
    assert dataset_cfg.n_trajectories <= len(ep_indices), (
        f'Requested number of trajectories ({dataset_cfg.n_trajectories}) '
        f'exceeds available episodes ({len(ep_indices)}) in dataset {dataset_cfg.dataset_name}.'
    )
    num_trajectories = dataset_cfg.n_trajectories
    sampled_ep_indices = g.choice(
        ep_indices, size=num_trajectories, replace=False
    )

    # Build start and end steps for full trajectories

    ep_len_dict = {ep_id: episode_len[i] for i, ep_id in enumerate(ep_indices)}

    start_steps = np.zeros(num_trajectories, dtype=int)
    end_steps = []
    for ep_id in sampled_ep_indices:
        # we make sure that the end step allows for full frameskip intervals
        end_steps.append(
            ep_len_dict[ep_id] - (ep_len_dict[ep_id] % cfg.frameskip)
        )

    end_steps = np.array(end_steps, dtype=int)

    # Load dataset and keep only needed trajectories

    def get_img_pipeline(key, target, img_size=224):
        return spt.data.transforms.Compose(
            spt.data.transforms.ToImage(
                **spt.data.dataset_stats.ImageNet,
                source=key,
                target=target,
            ),
            spt.data.transforms.Resize(img_size, source=key, target=target),
            spt.data.transforms.CenterCrop(
                img_size, source=key, target=target
            ),
        )

    def norm_col_transform(dataset, col='pixels'):
        """Normalize column to zero mean, unit variance."""
        data = dataset[col][:]
        mean = data.mean(0).unsqueeze(0)
        std = data.std(0).unsqueeze(0)
        return lambda x: (x - mean) / std

    # Use dataset_cfg for specific dataset parameters
    dataset = swm.data.FrameDataset(
        dataset_cfg.dataset_name,
        num_steps=dataset_cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        cache_dir=cfg.get('cache_dir', None),
    )

    all_norm_transforms = []
    # Use global cfg for encoding keys to ensure consistency
    for key in model_cfg.get('encoding', {}):
        trans_fn = norm_col_transform(dataset.dataset, key)
        trans_fn = spt.data.transforms.WrapTorchTransform(
            trans_fn, source=key, target=key
        )
        all_norm_transforms.append(trans_fn)

    # Image size must be multiple of DINO patch size (14)
    img_size = (cfg.image_size // cfg.patch_size) * DINO_PATCH_SIZE

    # Apply transforms to all steps
    traj_list = []
    for j in range(num_trajectories):
        transform = spt.data.transforms.Compose(
            *[
                get_img_pipeline(f'{col}.{i}', f'{col}.{i}', img_size)
                for col in ['pixels']
                for i in range(end_steps[j] // cfg.frameskip)
            ],
            *all_norm_transforms,
        )

        dataset.transform = transform
        data = dataset.load_chunk(
            episode=sampled_ep_indices[j],
            start=start_steps[j],
            end=end_steps[j],
        )[0]
        data['id'] = (
            torch.ones(data['pixels'].shape[0], 1) * sampled_ep_indices[j]
        )
        traj_list.append(data)

    with open_dict(model_cfg) as model_cfg:
        model_cfg.extra_dims = {}
        for key in model_cfg.get('encoding', {}):
            if key not in dataset.dataset.column_names:
                raise ValueError(
                    f"Encoding key '{key}' not found in dataset columns."
                )
            inpt_dim = dataset.dataset[0][key].numel()
            model_cfg.extra_dims[key] = (
                inpt_dim if key != 'action' else inpt_dim * cfg.frameskip
            )

    return traj_list


# ============================================================================
# Model Architecture
# ============================================================================


def get_encoder(cfg, model_cfg):
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
        if model_cfg.backbone.name.startswith(config['prefix']):
            encoder_type = name
            break

    if encoder_type is None:
        raise ValueError(f'Unsupported backbone: {model_cfg.backbone.name}')

    config = ENCODER_CONFIGS[encoder_type]

    # Load model
    backbone = config.get('model_class', AutoModel).from_pretrained(
        model_cfg.backbone.name
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


def get_world_model(cfg, model_cfg):
    """Load and setup world model.
    For visualization, we only need the model to implement the `encode` method."""

    if model_cfg.model_name is not None:  # load checkpointed model
        model = swm.policy.AutoCostModel(model_cfg.model_name).to(
            cfg.get('device', 'cpu')
        )
        model = model.to(cfg.get('device', 'cpu'))
        model = model.eval()
    else:  # no checkpoint found, build model from scratch
        encoder, embedding_dim, num_patches, interp_pos_enc = get_encoder(
            cfg, model_cfg
        )
        embedding_dim += sum(
            emb_dim for emb_dim in model_cfg.get('encoding', {}).values()
        )  # add all extra dims

        logging.info(f'Patches: {num_patches}, Embedding dim: {embedding_dim}')

        # Build causal predictor (transformer that predicts next latent states)

        print('>>>> DIM PREDICTOR:', embedding_dim)

        predictor = swm.wm.prejepa.CausalPredictor(
            num_patches=num_patches,
            num_frames=model_cfg.history_size,
            dim=embedding_dim,
            **model_cfg.predictor,
        )

        # Build action and proprioception encoders
        extra_encoders = OrderedDict()
        for key, emb_dim in model_cfg.get('encoding', {}).items():
            inpt_dim = model_cfg.extra_dims[key]
            extra_encoders[key] = swm.wm.prejepa.Embedder(
                in_chans=inpt_dim, emb_dim=emb_dim
            )
            print(
                f'Build encoder for {key} with input dim {inpt_dim} and emb dim {emb_dim}'
            )

        extra_encoders = torch.nn.ModuleDict(extra_encoders)

        # Assemble world model
        model = swm.wm.prejepa(
            encoder=spt.backbone.EvalOnly(encoder),
            predictor=predictor,
            extra_encoders=extra_encoders,
            history_size=model_cfg.history_size,
            num_pred=model_cfg.num_preds,
            interpolate_pos_encoding=interp_pos_enc,
        )
        model.to(cfg.get('device', 'cpu'))
        model = model.eval()
    return model


# ============================================================================
# Embedding Collection and Visualization
# ============================================================================


def collect_embeddings(cfg, exp_cfg):
    """
    Loads a specific dataset and its corresponding world model,
    encodes the data, and returns the flattened embeddings.
    """
    # load trajectories of the dataset
    trajs = get_data(cfg, exp_cfg.dataset, exp_cfg.world_model)
    world_model = get_world_model(cfg, exp_cfg.world_model)

    logging.info(
        f'Encoding dataset: {exp_cfg.dataset.dataset_name} using model: {exp_cfg.world_model.model_name}...'
    )
    trajs_embeddings = []  # list to store the embeddings of the trajectories (T x D)
    predicted_embeddings = []  # list to store the predicted embeddings of the trajectories (T x D)
    trajs_pixels = []  # list to store the pixels of the trajectories (T x C x H x W)

    # Process batches and collect embeddings
    for traj in tqdm(trajs, desc=f'Processing {exp_cfg.dataset.dataset_name}'):
        init_state = {}
        for key in traj:
            if isinstance(traj[key], torch.Tensor):
                traj[key] = traj[key].unsqueeze(0).to(cfg.get('device', 'cpu'))
                init_state[key] = traj[key][:, :1].unsqueeze(0)

        # store pixels for visualization
        trajs_pixels.append(traj['pixels'].squeeze(0).cpu().detach())

        # Encode trajectory
        traj = world_model.encode(traj, target='embed')
        if exp_cfg.world_model.get(
            'backbone_only', False
        ):  # use only vision backbone embeddings
            flat_embed = rearrange(traj['pixels_embed'][0], 't p d ->t (p d)')
            trajs_embeddings.append(flat_embed.cpu().detach())
        else:  # use full model embeddings (proropio + action + vision)
            flat_embed = rearrange(traj['embed'][0], 't p d ->t (p d)')
            trajs_embeddings.append(flat_embed.cpu().detach())

        # predict trajectory
        traj = world_model.rollout(
            init_state, traj['action'][:, :-1].unsqueeze(0)
        )
        if exp_cfg.world_model.get(
            'backbone_only', False
        ):  # use only vision backbone embeddings
            flat_predicted = rearrange(
                traj['predicted_pixels_embed'][0, 0], 't p d ->t (p d)'
            )
            predicted_embeddings.append(flat_predicted.cpu().detach())
        else:  # use full model embeddings (proropio + action + vision)
            flat_predicted = rearrange(
                traj['predicted_embedding'][0, 0], 't p d ->t (p d)'
            )
            predicted_embeddings.append(flat_predicted.cpu().detach())

    return trajs_embeddings, predicted_embeddings, trajs_pixels


# ============================================================================
# Visualization Functions
# ============================================================================


def reconstruct_trajectories(flat_embeddings_2d, trajectory_lengths):
    """
    Splits the flattened t-SNE results back into a list of trajectory arrays.
    """
    trajs_2d = []
    start_idx = 0
    for length in trajectory_lengths:
        end_idx = start_idx + length
        trajs_2d.append(flat_embeddings_2d[start_idx:end_idx])
        start_idx = end_idx
    return trajs_2d


def plot_static_trajectories(
    trajs_2d,
    preds_2d,
    labels,
    output_file='trajectories_tsne.pdf',
    interp_factor=10,
):
    """
    Plots truth and predicted trajectories.
    Truth = Solid lines.
    Prediction = Dashed lines.
    Args:
        trajs_2d: List of (N, 2) numpy arrays.
        preds_2d: List of (N, 2) numpy arrays corresponding to predicted trajectories.
        labels: List of labels corresponding to trajs_2d.
        output_file: Filename for the saved plot.
        interp_factor: How many times to increase point density.
                       1 = original points, 10 = 10x denser.
    """
    plt.figure(figsize=(12, 10))

    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(unique_labels)))
    label_color_map = dict(zip(unique_labels, colors))

    added_labels = set()

    # Zip truth and prediction together
    for traj, pred, label in zip(trajs_2d, preds_2d, labels):
        # Skip trajectories that are too short
        if len(traj) < 2 or len(pred) < 2:
            continue

        base_color = label_color_map[label]
        rgb = mcolors.to_rgb(base_color)

        # --- Helper for interpolation ---
        def get_dense_traj(arr_2d):
            x_ind = np.arange(len(arr_2d))
            new_ind = np.linspace(
                0, len(arr_2d) - 1, num=len(arr_2d) * interp_factor
            )
            f_x = interp1d(x_ind, arr_2d[:, 0], kind='linear')
            f_y = interp1d(x_ind, arr_2d[:, 1], kind='linear')
            return f_x(new_ind), f_y(new_ind)

        # 1. Plot Truth (Solid)
        t_x, t_y = get_dense_traj(traj)
        num_points = len(t_x)

        # Color gradient for truth
        t_colors = np.zeros((num_points, 4))
        t_colors[:, :3] = rgb
        t_colors[:, 3] = np.linspace(0.1, 1.0, num_points)  # Alpha gradient

        plt.scatter(t_x, t_y, c=t_colors, s=5, edgecolors='none')
        # Markers for Truth
        plt.scatter(
            traj[0, 0],
            traj[0, 1],
            color=base_color,
            marker='o',
            s=40,
            zorder=10,
            label=label if label not in added_labels else '',
        )
        plt.scatter(
            traj[-1, 0],
            traj[-1, 1],
            color=base_color,
            marker='x',
            s=40,
            zorder=10,
        )

        # 2. Plot Prediction (Dashed line effect via scatter with spacing or simple plot)
        # Using simple plot for dashed effect is cleaner for predictions
        p_x, p_y = get_dense_traj(pred)

        # We plot the prediction as a faint dashed line to distinguish it
        plt.plot(
            p_x,
            p_y,
            color=base_color,
            linestyle='--',
            linewidth=1.5,
            alpha=0.7,
        )

        # End marker for prediction (Distinct from truth)
        plt.scatter(
            pred[-1, 0],
            pred[-1, 1],
            color=base_color,
            marker='*',
            s=50,
            zorder=10,
            edgecolors='white',
            linewidth=0.5,
        )

        added_labels.add(label)

    plt.title('Trajectory Embeddings: Truth (Solid) vs Prediction (Dashed)')
    plt.xlabel('t-SNE dim 1')
    plt.ylabel('t-SNE dim 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    logging.info(f'Static trajectory plot saved to {output_file}')


def create_video_visualization(
    trajs_2d,
    preds_2d,
    trajs_pixels,
    labels,
    output_file='trajectories_video.mp4',
    max_trajs=5,
):
    """
    Creates MP4 with:
    Left: Video pixels
    Right: Latent space with TWO trails (Truth = Circle, Pred = Star/Cross)
    """
    indices = np.linspace(
        0, len(trajs_2d) - 1, min(len(trajs_2d), max_trajs), dtype=int
    )

    subset_trajs = [trajs_2d[i] for i in indices]
    subset_preds = [preds_2d[i] for i in indices]
    subset_pixels = [trajs_pixels[i] for i in indices]
    subset_labels = [labels[i] for i in indices]

    n_rows = len(subset_trajs)
    max_len = max([t.shape[0] for t in subset_trajs])

    # Color Setup
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(unique_labels)))
    label_color_map = dict(zip(unique_labels, colors))

    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Calculate bounds based on both truth and preds
    all_points = np.vstack(subset_trajs + subset_preds)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    margin = (x_max - x_min) * 0.1

    img_plots = []
    scat_truth_plots = []
    scat_pred_plots = []

    for i in range(n_rows):
        ax_vid = axes[i, 0]
        ax_lat = axes[i, 1]
        label = subset_labels[i]
        base_color = label_color_map[label]

        # --- Left: Video ---
        frame0 = subset_pixels[i][0].permute(1, 2, 0).numpy()
        frame0 = (frame0 - frame0.min()) / (frame0.max() - frame0.min() + 1e-6)
        img_plots.append(ax_vid.imshow(frame0))
        ax_vid.set_title(f'{label}')
        ax_vid.axis('off')

        # --- Right: Latent ---
        # Background faint traces
        ax_lat.plot(
            subset_trajs[i][:, 0],
            subset_trajs[i][:, 1],
            color=base_color,
            alpha=0.15,
            linewidth=1,
        )
        ax_lat.plot(
            subset_preds[i][:, 0],
            subset_preds[i][:, 1],
            color=base_color,
            alpha=0.15,
            linewidth=1,
            linestyle='--',
        )

        # Dynamic scatter trails
        # Truth trail
        scat_t = ax_lat.scatter(
            [], [], color=base_color, s=20, marker='o', zorder=2
        )
        # Prediction trail
        scat_p = ax_lat.scatter(
            [],
            [],
            color=base_color,
            s=30,
            marker='*',
            zorder=2,
            edgecolors='black',
            linewidth=0.2,
        )

        scat_truth_plots.append(scat_t)
        scat_pred_plots.append(scat_p)

        ax_lat.set_xlim(x_min - margin, x_max + margin)
        ax_lat.set_ylim(y_min - margin, y_max + margin)
        ax_lat.set_title('Latent: Truth (o) vs Pred (*)')

    def update(frame_idx):
        artists = []
        for i in range(n_rows):
            traj_len = len(subset_trajs[i])
            idx = min(frame_idx, traj_len - 1)

            # Update Video
            frame = subset_pixels[i][idx].permute(1, 2, 0).numpy()
            frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-6)
            img_plots[i].set_data(frame)
            artists.append(img_plots[i])

            # Helper for trail colors
            def get_trail_colors(n_pts):
                c = np.zeros((n_pts, 4))
                c[:, :3] = mcolors.to_rgb(label_color_map[subset_labels[i]])
                c[:, 3] = np.linspace(0.1, 1.0, n_pts)
                return c

            # Update Truth Trail
            path_t = subset_trajs[i][: idx + 1]
            scat_truth_plots[i].set_offsets(path_t)
            scat_truth_plots[i].set_color(get_trail_colors(len(path_t)))
            artists.append(scat_truth_plots[i])

            # Update Pred Trail
            path_p = subset_preds[i][: idx + 1]
            scat_pred_plots[i].set_offsets(path_p)
            scat_pred_plots[i].set_color(get_trail_colors(len(path_p)))
            artists.append(scat_pred_plots[i])

        return artists

    logging.info(f'Generating animation with {n_rows} trajectories...')
    ani = animation.FuncAnimation(
        fig, update, frames=max_len, interval=100, blit=True
    )
    ani.save(output_file, fps=10, extra_args=['-vcodec', 'libx264'])
    plt.close()
    logging.info(f'Animation saved to {output_file}')


# ============================================================================
# Main Run Loop
# ============================================================================


@hydra.main(
    version_base=None,
    config_path='./configs',
    config_name='config_trajectories',
)
def run(cfg):
    """Run visualization script for multiple datasets and compute joint t-SNE."""

    all_embeddings_list = []
    all_predicted_embeddings_list = []
    all_pixels_list = []
    all_labels_list = []
    trajectory_lengths = []

    # Iterate over all defined datasets and collect embeddings
    if cfg.datasets:
        for key in cfg.datasets:
            print(f'Processing dataset key: {key}')
            exp_cfg = cfg.datasets[key]

            embeddings, predicted_embeddings, trajs_pixels = (
                collect_embeddings(cfg, exp_cfg)
            )

            if embeddings is not None:
                all_embeddings_list.extend(embeddings)
                all_predicted_embeddings_list.extend(predicted_embeddings)
                all_pixels_list.extend(trajs_pixels)

                dataset_name = exp_cfg.dataset.dataset_name
                num_traj = len(embeddings)

                all_labels_list.extend([dataset_name] * num_traj)

                for emb in embeddings:
                    trajectory_lengths.append(emb.shape[0])

    if not all_embeddings_list:
        logging.warning('No embeddings generated.')
        return

    # Check dims
    ref_dim = all_embeddings_list[0].shape[1]
    for i, emb in enumerate(all_embeddings_list):
        if emb.shape[1] != ref_dim:
            raise ValueError(
                f'Dimension mismatch. Expected {ref_dim}, got {emb.shape[1]}'
            )
        if all_predicted_embeddings_list[i].shape[1] != ref_dim:
            raise ValueError(
                f'Dimension mismatch in predicted embeddings. Expected {ref_dim}, got {all_predicted_embeddings_list[i].shape[1]}'
            )

    # Concatenate Truth embeddings
    truth_tensor = torch.cat(all_embeddings_list, dim=0)
    # Concatenate Predicted embeddings
    pred_tensor = torch.cat(all_predicted_embeddings_list, dim=0)

    # Create Joint tensor for dimensionality reduction
    full_embeddings = torch.cat([truth_tensor, pred_tensor], dim=0).numpy()

    # Run dimensionality reduction:
    if cfg.dimensionality_reduction == 'tsne':
        logging.info(
            f'Computing t-SNE on {full_embeddings.shape[0]} points (Truth + Pred)...'
        )
        tsne = TSNE(n_components=2, random_state=cfg.seed)
        embeddings_2d_flat = tsne.fit_transform(full_embeddings)
    elif cfg.dimensionality_reduction == 'pca':
        logging.info(
            f'Computing PCA on {full_embeddings.shape[0]} points (Truth + Pred)...'
        )
        pca = PCA(n_components=2, random_state=cfg.seed)
        embeddings_2d_flat = pca.fit_transform(full_embeddings)
    else:
        raise ValueError(
            f'Unsupported dimensionality reduction method: {cfg.dimensionality_reduction}'
        )

    logging.info(
        f'Dimensionality reduction completed. Output shape: {embeddings_2d_flat.shape}'
    )

    # Split back into Truth and Pred components
    total_points = embeddings_2d_flat.shape[0]
    split_idx = (
        total_points // 2
    )  # Exact split since truth and pred have same lengths

    truth_2d_flat = embeddings_2d_flat[:split_idx]
    pred_2d_flat = embeddings_2d_flat[split_idx:]

    # Reconstruct list of (T, 2) arrays for both
    trajs_2d = reconstruct_trajectories(truth_2d_flat, trajectory_lengths)
    preds_2d = reconstruct_trajectories(pred_2d_flat, trajectory_lengths)

    # Plot 1: Static 2D Latent Space (Truth + Pred)
    plot_static_trajectories(
        trajs_2d,
        preds_2d,
        all_labels_list,
        output_file='trajectories_tsne.pdf',
    )

    # Plot 2: Video Visualization (Truth + Pred)
    create_video_visualization(
        trajs_2d,
        preds_2d,
        all_pixels_list,
        all_labels_list,
        output_file='trajectories_video.mp4',
        max_trajs=4,
    )

    return


if __name__ == '__main__':
    run()
