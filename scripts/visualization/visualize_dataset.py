from collections import OrderedDict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import torch
from einops import rearrange
from loguru import logger as logging
from omegaconf import open_dict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoModelForImageClassification

import stable_worldmodel as swm


DINO_PATCH_SIZE = 14  # DINO encoder uses 14x14 patches

# ============================================================================
# Data Loading
# ============================================================================


def get_data(cfg, dataset_cfg, model_cfg):
    """Setup dataset with image transforms and normalization."""

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
    if dataset_cfg.data_format == 'frame':
        dataset = swm.data.FrameDataset(
            dataset_cfg.dataset_name,
            num_steps=dataset_cfg.n_steps,
            frameskip=cfg.frameskip,
            transform=None,
            cache_dir=cfg.get('cache_dir', None),
        )
    elif dataset_cfg.data_format == 'video':
        dataset = swm.data.VideoDataset(
            dataset_cfg.dataset_name,
            num_steps=dataset_cfg.n_steps,
            frameskip=cfg.frameskip,
            transform=None,
            cache_dir=cfg.get('cache_dir', None),
        )
    else:
        raise NotImplementedError(
            f"Data format '{dataset_cfg.data_format}' not supported."
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
    transform = spt.data.transforms.Compose(
        *[
            get_img_pipeline(f'{col}.{i}', f'{col}.{i}', img_size)
            for col in ['pixels']
            for i in range(dataset_cfg.n_steps)
        ],
        *all_norm_transforms,
    )

    dataset.transform = transform
    rnd_gen = torch.Generator().manual_seed(cfg.seed)

    # Use dataset_cfg for split and loader settings
    visual_set, _ = spt.data.random_split(
        dataset,
        lengths=[dataset_cfg.visual_split, 1 - dataset_cfg.visual_split],
        generator=rnd_gen,
    )
    logging.info(f'Visual ({dataset_cfg.dataset_name}): {len(visual_set)}')

    visual = DataLoader(
        visual_set,
        batch_size=dataset_cfg.batch_size,
        num_workers=dataset_cfg.num_workers,
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )
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

    return visual


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

    if model_cfg.model_name is not None:
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
    data = get_data(cfg, exp_cfg.dataset, exp_cfg.world_model)
    world_model = get_world_model(cfg, exp_cfg.world_model)

    logging.info(
        f'Encoding dataset: {exp_cfg.dataset.dataset_name} using model: {exp_cfg.world_model.model_name}...'
    )
    dataset_embeddings = []

    # Process batches and collect embeddings
    for batch in tqdm(data, desc=f'Processing {exp_cfg.dataset.dataset_name}'):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(cfg.get('device', 'cpu'))

        # Encode
        batch = world_model.encode(batch, target='embed')
        if exp_cfg.world_model.get(
            'backbone_only', False
        ):  # use only vision backbone embeddings
            dataset_embeddings.append(batch['pixels_embed'].cpu().detach())
        else:  # use full model embeddings (proropio + action + vision)
            dataset_embeddings.append(batch['embed'].cpu().detach())

    # Consolidate and flatten embeddings for this dataset
    if len(dataset_embeddings) > 0:
        dataset_embeddings = torch.cat(dataset_embeddings, dim=0)
        # Flatten: b ... -> b (...)
        dataset_embeddings = rearrange(dataset_embeddings, 'b ... -> b (...)')
        return dataset_embeddings

    return None


# ============================================================================
# Main Visualization Script
# ============================================================================


def plot_joint_dimensionality_reduction(
    embeddings_2d, labels, output_file='joint_tsne.pdf'
):
    """
    Plots the 2D embeddings from any dimensionality reduction method, coloring points by their dataset label.

    Args:
        embeddings_2d (np.ndarray): Shape (N, 2) containing 2D coordinates.
        labels (list or np.ndarray): Shape (N,) containing dataset names for each point.
        output_file (str): Path to save the resulting plot.
    """
    plt.figure(figsize=(12, 10))

    # Get unique datasets to iterate over for the legend
    unique_datasets = np.unique(labels)

    # Use a colormap suitable for categorical data
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_datasets)))

    for dataset_name, color in zip(unique_datasets, colors):
        # Select indices belonging to this dataset
        indices = np.where(labels == dataset_name)

        plt.scatter(
            embeddings_2d[indices, 0],
            embeddings_2d[indices, 1],
            label=dataset_name,
            c=[color],
            alpha=0.6,
            s=10,  # Marker size
        )

    plt.title('Joint t-SNE of Dataset Embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(title='Datasets', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    logging.info(f'Saving plot to {output_file}')
    # Matplotlib automatically handles the output format based on the file extension
    plt.savefig(output_file, dpi=300)
    plt.close()


@hydra.main(
    version_base=None, config_path='./configs', config_name='config_datasets'
)
def run(cfg):
    """Run visualization script for multiple datasets and compute joint t-SNE."""

    all_embeddings_list = []
    all_labels_list = []  # Added to track source datasets

    # Iterate over all defined datasets and collect embeddings
    for exp_cfg in cfg.datasets.values():
        embeddings = collect_embeddings(cfg, exp_cfg)

        if embeddings is not None:
            all_embeddings_list.append(embeddings)

            # Create a label entry for every point in this batch
            dataset_name = exp_cfg.dataset.dataset_name
            num_points = embeddings.shape[0]
            all_labels_list.extend([dataset_name] * num_points)

    # Concatenate all datasets for joint dimensionality reduction
    if not all_embeddings_list:
        logging.warning('No embeddings generated from any experiment.')
        return

    # Ensure dimensions match before concatenating
    ref_dim = all_embeddings_list[0].shape[1]
    for i, emb in enumerate(all_embeddings_list):
        if emb.shape[1] != ref_dim:
            raise ValueError(
                f'Dimension mismatch in dataset {i}: expected {ref_dim}, got {emb.shape[1]}. '
                'Ensure image_size, patch_size, and model architecture are consistent across all datasets.'
            )

    logging.info('Computing Joint Dimensionality Reduction...')
    # Concatenate all tensors then convert to numpy for dimensionality reduction
    full_embeddings = torch.cat(all_embeddings_list, dim=0).numpy()
    all_labels = np.array(all_labels_list)  # Convert labels to numpy array

    if cfg.dimensionality_reduction == 'tsne':
        # now we compute t-SNE on the embeddings
        tsne = TSNE(n_components=2, random_state=cfg.seed)
        embeddings_2d = tsne.fit_transform(full_embeddings)
    elif cfg.dimensionality_reduction == 'pca':
        pca = PCA(n_components=2, random_state=cfg.seed)
        embeddings_2d = pca.fit_transform(full_embeddings)

    logging.info(
        f'Dimensionality reduction ({cfg.dimensionality_reduction}) completed. Output shape: {embeddings_2d.shape}'
    )

    # Plot the results
    plot_joint_dimensionality_reduction(
        embeddings_2d,
        all_labels,
        output_file=f'joint_{cfg.dimensionality_reduction}.pdf',
    )

    return


if __name__ == '__main__':
    run()
