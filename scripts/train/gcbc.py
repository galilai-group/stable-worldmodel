import os
from collections import OrderedDict
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

import stable_worldmodel as swm


# ============================================================================
# Data Setup
# ============================================================================
def get_data(cfg):
    """Setup dataset with image transforms and normalization."""

    def get_img_pipeline(key, target, img_size=224):
        return spt.data.transforms.Compose(
            spt.data.transforms.ToImage(
                **spt.data.dataset_stats.ImageNet,
                source=key,
                target=target,
            ),
            spt.data.transforms.Resize(img_size, source=key, target=target),
        )

    def get_column_normalizer(dataset, source: str, target: str):
        """Get normalizer for a specific column in the dataset."""
        data = torch.from_numpy(dataset.get_col_data(source)[:])
        data = data[~torch.isnan(data).any(dim=1)]
        mean = data.mean(0, keepdim=True).clone()
        std = data.std(0, keepdim=True).clone()

        def norm_fn(x):
            return ((x - mean) / std).float()

        normalizer = spt.data.transforms.WrapTorchTransform(
            norm_fn, source=source, target=target
        )

        return normalizer

    cache_dir = None
    if not hasattr(cfg, 'local_cache_dir'):
        cache_dir = os.environ.get('SLURM_TMPDIR', None)

    dataset = swm.data.HDF5Dataset(
        cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        cache_dir=cache_dir,
        keys_to_load=['pixels', 'action', 'proprio'],
        keys_to_cache=['action', 'proprio'],
    )

    norm_action_transform = get_column_normalizer(dataset, 'action', 'action')
    norm_proprio_transform = get_column_normalizer(
        dataset, 'proprio', 'proprio'
    )

    # Apply transforms to all steps and goal observations
    transform = spt.data.transforms.Compose(
        get_img_pipeline('pixels', 'pixels', cfg.image_size),
        norm_action_transform,
        norm_proprio_transform,
    )

    dataset.transform = transform

    dataset = swm.data.GoalDataset(
        dataset=dataset,
        goal_probabilities=(0.0, 0.0, 1.0, 0.0),
        current_goal_offset=cfg.dinowm.history_size,
        goal_keys={'pixels': 'goal_pixels', 'proprio': 'goal_proprio'},
        seed=cfg.seed,
    )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    logging.info(f'Train: {len(train_set)}, Val: {len(val_set)}')

    train = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )
    val = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return spt.data.DataModule(train=train, val=val)


# ============================================================================
# Model Architecture
# ============================================================================
def get_gcbc_policy(cfg):
    """Build goal-conditioned behavioral cloning policy: frozen encoder (e.g. DINO) + trainable action predictor."""

    def forward(self, batch, stage):
        """Forward: encode observations and goals, predict actions, compute losses."""

        proprio_key = 'proprio' if 'proprio' in batch else None

        # Replace NaN values with 0 (occurs at sequence boundaries)
        if proprio_key is not None:
            batch[proprio_key] = torch.nan_to_num(batch[proprio_key], 0.0)
        batch['action'] = torch.nan_to_num(batch['action'], 0.0)

        # Encode all timesteps into latent embeddings
        batch = self.model.encode(
            batch,
            target='embed',
            pixels_key='pixels',
        )

        # Encode goal into latent embedding
        batch = self.model.encode(
            batch,
            target='goal_embed',
            pixels_key='goal_pixels',
            prefix='goal_',
        )

        # Use history to predict next actions
        embedding = batch['embed'][
            :, : cfg.dinowm.history_size, :, :
        ]  # (B, T-1, patches, dim)
        goal_embedding = batch['goal_embed']  # (B, 1, patches, dim)
        action_pred, _ = self.model.predict_actions(
            embedding, goal_embedding
        )  # (B, num_preds, action_dim)
        action_target = batch['action'][
            :, : cfg.dinowm.history_size, :
        ]  # (B, num_preds, action_dim)

        # Compute action MSE
        action_loss = F.mse_loss(action_pred, action_target)
        if torch.isnan(action_loss):
            logging.error(
                f'NaN loss! action_pred has nan: {torch.isnan(action_pred).any()}, action_target has nan: {torch.isnan(action_target).any()}'
            )
        batch['loss'] = action_loss

        # Log all losses
        prefix = 'train/' if self.training else 'val/'
        losses_dict = {
            f'{prefix}{k}': v.detach()
            for k, v in batch.items()
            if '_loss' in k
        }
        losses_dict[f'{prefix}loss'] = batch['loss'].detach()
        self.log_dict(
            losses_dict, on_step=True, sync_dist=True
        )  # , on_epoch=True, sync_dist=True)

        return batch

    # Load encoder based on config
    encoder_type = cfg.get('encoder_type', 'dino')
    if encoder_type == 'dino':
        # Load frozen DINO encoder
        encoder = AutoModel.from_pretrained('facebook/dinov2-small')
        embedding_dim = encoder.config.hidden_size
        encoder_trainable = False
        logging.info('Using pretrained frozen DINO encoder')
    elif encoder_type == 'vit_tiny':
        # Load trainable ViT tiny from scratch
        encoder = spt.backbone.utils.vit_hf(
            'tiny',
            patch_size=cfg.patch_size,
            image_size=cfg.image_size,
            pretrained=False,
            use_mask_token=False,
        )
        embedding_dim = encoder.config.hidden_size
        encoder_trainable = True
        logging.info('Using trainable ViT tiny encoder (from scratch)')
    else:
        raise ValueError(f'Unknown encoder_type: {encoder_type}')

    # Calculate actual number of patches based on the actual image size used by DINO
    assert cfg.image_size % cfg.patch_size == 0, (
        'Image size must be multiple of patch size'
    )
    num_patches = (cfg.image_size // cfg.patch_size) ** 2
    if cfg.dinowm.get('use_proprio_encoder', True):
        embedding_dim += cfg.dinowm.proprio_embed_dim  # Total embedding size

    logging.info(f'Patches: {num_patches}, Embedding dim: {embedding_dim}')

    # Build causal predictor (transformer that predicts next actions)
    effective_act_dim = (
        cfg.frameskip * cfg.dinowm.action_dim
    )  # NOTE: 'frameskip' > 1 is used to predict action chunks
    predictor = swm.wm.gcrl.Predictor(
        num_patches=num_patches,
        num_frames=cfg.dinowm.history_size,
        dim=embedding_dim,
        out_dim=effective_act_dim,
        **cfg.predictor,
    )

    # Build proprioception encoder (optional)
    extra_encoders = None
    if cfg.dinowm.get('use_proprio_encoder', True):
        extra_encoders = OrderedDict()
        extra_encoders['proprio'] = swm.wm.gcrl.Embedder(
            in_chans=cfg.dinowm.proprio_dim,
            emb_dim=cfg.dinowm.proprio_embed_dim,
        )
        extra_encoders = torch.nn.ModuleDict(extra_encoders)

    logging.info(
        f'Action dim: {effective_act_dim}, Proprio encoder: {extra_encoders is not None}'
    )

    # Assemble policy
    # Wrap encoder in EvalOnly if frozen (DINO), otherwise keep trainable (ViT tiny)
    wrapped_encoder = (
        spt.backbone.EvalOnly(encoder) if not encoder_trainable else encoder
    )
    gcbc_policy = swm.wm.gcrl.GCRL(
        encoder=wrapped_encoder,
        action_predictor=predictor,
        extra_encoders=extra_encoders,
        history_size=cfg.dinowm.history_size,
    )

    # Wrap in stable_spt Module with separate optimizers for each component
    def add_opt(module_name, lr):
        return {
            'modules': str(module_name),
            'optimizer': {'type': 'AdamW', 'lr': lr},
            'scheduler': {'type': 'LinearWarmupCosineAnnealingLR'},
        }

    optim_config = {
        'action_predictor_opt': add_opt(
            'model.action_predictor', cfg.predictor_lr
        ),
    }

    # Add proprio encoder optimizer if enabled
    if extra_encoders is not None:
        optim_config['proprio_opt'] = add_opt(
            'model.extra_encoders.proprio', cfg.proprio_encoder_lr
        )

    # Add encoder optimizer if trainable (ViT tiny)
    if encoder_trainable:
        optim_config['encoder_opt'] = add_opt(
            'model.encoder', cfg.get('encoder_lr', 1e-4)
        )

    gcbc_policy = spt.Module(
        model=gcbc_policy,
        forward=forward,
        optim=optim_config,
    )
    return gcbc_policy


# ============================================================================
# Training Setup
# ============================================================================
def setup_pl_logger(cfg):
    if not cfg.wandb.enable:
        return None

    wandb_run_id = cfg.wandb.get('run_id', None)
    wandb_logger = WandbLogger(
        name='dino_gcbc',
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        resume='allow' if wandb_run_id else None,
        id=wandb_run_id,
        log_model=False,
    )

    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))
    return wandb_logger


class ModelObjectCallBack(Callback):
    """Callback to pickle model after each epoch."""

    def __init__(
        self, dirpath, filename='model_object', epoch_interval: int = 1
    ):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ) -> None:
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                output_path = Path(
                    self.dirpath,
                    f'{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt',
                )
                torch.save(pl_module, output_path)
                logging.info(f'Saved world model object to {output_path}')
            # Additionally, save at final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                final_path = self.dirpath / f'{self.filename}_object.ckpt'
                torch.save(pl_module, final_path)
                logging.info(f'Saved final world model object to {final_path}')


# ============================================================================
# Main Entry Point
# ============================================================================
@hydra.main(version_base=None, config_path='./config', config_name='gcbc')
def run(cfg):
    """Run training of predictor"""

    wandb_logger = setup_pl_logger(cfg)
    data = get_data(cfg)
    gcbc_policy = get_gcbc_policy(cfg)

    cache_dir = swm.data.utils.get_cache_dir()
    dump_object_callback = ModelObjectCallBack(
        dirpath=cache_dir,
        filename=cfg.output_model_name,
        epoch_interval=3,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[dump_object_callback],
        num_sanity_val_steps=1,
        logger=wandb_logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=gcbc_policy,
        data=data,
        ckpt_path=f'{cache_dir}/{cfg.output_model_name}_weights.ckpt',
    )
    manager()


if __name__ == '__main__':
    run()
