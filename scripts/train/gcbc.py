import os

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.callbacks import Callback
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.utils import save_pretrained
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from torch.nn import functional as F
from torch.utils.data import DataLoader

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

    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)

    # Dataset params come from the shared data node (config/data/pusht.yaml),
    # like scripts/train/lewm.py.
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    print(
        f'Loading dataset "{dataset_name}" from {"local cache: " + cache_dir if cache_dir else "default location"}'
    )

    dataset = swm.data.load_dataset(
        dataset_name,
        transform=None,
        cache_dir=cache_dir,
        **dataset_cfg,
    )

    # Normalize every non-pixel column (as in scripts/train/lewm.py) and mirror
    # pixels/proprio into goal observations for the goal-conditioned policy.
    keys_to_load = list(cfg.data.dataset.keys_to_load)
    transforms = [get_img_pipeline('pixels', 'pixels', cfg.image_size)]
    for col in keys_to_load:
        if col == 'pixels':
            continue
        transforms.append(get_column_normalizer(dataset, col, col))

    goal_keys = {'pixels': 'goal_pixels'}
    if 'proprio' in keys_to_load:
        goal_keys['proprio'] = 'goal_proprio'

    dataset.transform = spt.data.transforms.Compose(*transforms)

    dataset = swm.data.GoalDataset(
        dataset=dataset,
        goal_probabilities=(0.0, 0.0, 1.0, 0.0),
        current_goal_offset=cfg.wm.history_size,
        goal_keys=goal_keys,
        seed=cfg.seed,
    )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train_subset_fraction = cfg.get('train_subset_fraction', 1.0)
    if train_subset_fraction < 1.0:
        train_set, _ = spt.data.random_split(
            train_set,
            lengths=[train_subset_fraction, 1 - train_subset_fraction],
            generator=rnd_gen,
        )
        logging.info(
            f'Using {train_subset_fraction:.1%} of training data: {len(train_set)} samples'
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
            :, : cfg.wm.history_size, :, :
        ]  # (B, T-1, patches, dim)
        goal_embedding = batch['goal_embed']  # (B, 1, patches, dim)
        action_pred, _ = self.model.predict_actions(
            embedding, goal_embedding
        )  # (B, num_preds, action_dim)
        action_target = batch['action'][
            :, : cfg.wm.history_size, :
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

    # Instantiate the frozen pretrained encoder from the model config, then
    # freeze it (mirrors scripts/train/prejepa.py — no EvalOnly wrapper needed).
    encoder = hydra.utils.instantiate(cfg.model.encoder)
    encoder.eval()
    encoder.requires_grad_(False)

    # Fill the runtime-dependent predictor dims left as ??? in the config.
    assert cfg.image_size % cfg.patch_size == 0, (
        'Image size must be multiple of patch size'
    )
    num_patches = (cfg.image_size // cfg.patch_size) ** 2
    embedding_dim = encoder.config.hidden_size
    # NOTE: 'frameskip' > 1 is used to predict action chunks.
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    use_proprio = cfg.wm.get('use_proprio_encoder', True)
    if use_proprio:
        embedding_dim += cfg.wm.proprio_embed_dim  # concatenated to patches

    logging.info(
        f'Patches: {num_patches}, Embedding dim: {embedding_dim}, '
        f'Action dim: {effective_act_dim}, Proprio encoder: {use_proprio}'
    )

    with open_dict(cfg):
        cfg.model.action_predictor.num_patches = num_patches
        cfg.model.action_predictor.dim = embedding_dim
        cfg.model.action_predictor.out_dim = effective_act_dim
        # Optional proprioception encoder, built into the (saved) model config as
        # a ModuleDict so load_pretrained can rebuild it (mirrors prejepa.py).
        if use_proprio:
            cfg.model.extra_encoders = {
                '_target_': 'torch.nn.ModuleDict',
                'modules': {
                    'proprio': {
                        '_target_': 'stable_worldmodel.wm.gcrl.module.Embedder',
                        'in_chans': cfg.wm.proprio_dim,
                        'emb_dim': cfg.wm.proprio_embed_dim,
                    }
                },
            }

    # Build the policy from the (now fully-specified) model config, passing the
    # pre-built frozen encoder. The cfg.model.encoder node is kept in the saved
    # config so load_pretrained can rebuild the encoder from scratch.
    gcbc_policy = hydra.utils.instantiate(cfg.model, encoder=encoder)

    def add_opt(module_name, lr):
        return {
            'modules': str(module_name),
            'optimizer': {'type': 'AdamW', 'lr': lr},
            'scheduler': {'type': 'LinearWarmupCosineAnnealingLR'},
        }

    # Encoder is frozen; train the action predictor (+ proprio encoder if used).
    optim_config = {
        'action_predictor_opt': add_opt(
            'model.action_predictor', cfg.predictor_lr
        ),
    }
    if use_proprio:
        optim_config['proprio_opt'] = add_opt(
            'model.extra_encoders.proprio', cfg.proprio_encoder_lr
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


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


# ============================================================================
# Main Entry Point
# ============================================================================
@hydra.main(version_base=None, config_path='./config', config_name='gcbc')
def run(cfg):
    """Run training of predictor"""

    wandb_logger = setup_pl_logger(cfg)
    data = get_data(cfg)
    gcbc_policy = get_gcbc_policy(cfg)

    cache_dir = swm.data.utils.get_cache_dir(sub_folder='checkpoints')
    # Save `cfg.model` (the instantiable GCRL sub-config), not the full training
    # cfg — `load_pretrained` does `instantiate(config).load_state_dict(...)`, so
    # config.json must be the model config itself.
    save_ckpt_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=3,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[save_ckpt_callback],
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
