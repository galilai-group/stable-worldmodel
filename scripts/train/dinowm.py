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


DINO_PATCH_SIZE = 14  # DINO encoder uses 14x14 patches


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
            spt.data.transforms.CenterCrop(img_size, source=key, target=target),
        )

    def norm_col_transform(dataset, col="pixels"):
        """Normalize column to zero mean, unit variance."""
        data = dataset[col][:]
        mean = data.mean(0).unsqueeze(0)
        std = data.std(0).unsqueeze(0)
        return lambda x: (x - mean) / std

    dataset = swm.data.StepsDataset(
        cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        cache_dir=cfg.get("cache_dir", None),
    )

    # Image size must be multiple of DINO patch size (14)
    img_size = (cfg.image_size // cfg.patch_size) * DINO_PATCH_SIZE

    norm_action_transform = norm_col_transform(dataset.dataset, "action")
    norm_proprio_transform = norm_col_transform(dataset.dataset, "proprio")

    # Apply transforms to all steps
    transform = spt.data.transforms.Compose(
        *[get_img_pipeline(f"{col}.{i}", f"{col}.{i}", img_size) for col in ["pixels"] for i in range(cfg.n_steps)],
        spt.data.transforms.WrapTorchTransform(
            norm_action_transform,
            source="action",
            target="action",
        ),
        spt.data.transforms.WrapTorchTransform(
            norm_proprio_transform,
            source="proprio",
            target="proprio",
        ),
    )

    dataset.transform = transform
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    logging.info(f"Train: {len(train_set)}, Val: {len(val_set)}")

    train = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )
    val = DataLoader(val_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)

    return spt.data.DataModule(train=train, val=val)


# ============================================================================
# Model Architecture
# ============================================================================
def forward(self, batch, stage):
    """Forward: encode observations, predict next states, compute losses."""

    proprio_key = "proprio" if "proprio" in batch else None

    # Replace NaN values with 0 (occurs at sequence boundaries)
    if proprio_key is not None:
        batch[proprio_key] = torch.nan_to_num(batch[proprio_key], 0.0)
    if "action" in batch:
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    # Encode all timesteps into latent embeddings
    batch = self.model.encode(
        batch,
        target="embed",
        pixels_key="pixels",
        proprio_key=proprio_key,
        action_key="action",
    )

    # Use history to predict next states
    embedding = batch["embed"][:, :-1, :, :]  # (B, T-1, patches, dim)
    pred_embedding = self.model.predict(embedding)
    target_embedding = batch["embed"][:, 1:, :, :]  # (B, T-1, patches, dim)

    # Compute pixel reconstruction loss
    pixels_dim = batch["pixels_embed"].shape[-1]
    pixels_loss = F.mse_loss(pred_embedding[..., :pixels_dim], target_embedding[..., :pixels_dim].detach())
    loss = pixels_loss
    batch["pixels_loss"] = pixels_loss

    # Add proprioception loss if available
    if proprio_key is not None:
        proprio_dim = batch["proprio_embed"].shape[-1]
        proprio_loss = F.mse_loss(
            pred_embedding[..., pixels_dim : pixels_dim + proprio_dim],
            target_embedding[..., pixels_dim : pixels_dim + proprio_dim].detach(),
        )
        loss = loss + proprio_loss
        batch["proprio_loss"] = proprio_loss

    batch["loss"] = loss

    # Log all losses
    prefix = "train/" if self.training else "val/"
    losses_dict = {f"{prefix}{k}": v.detach() for k, v in batch.items() if "_loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)  # , on_epoch=True, sync_dist=True)

    return batch


def get_world_model(cfg):
    """Build world model: frozen DINO encoder + trainable causal predictor."""

    # Load frozen DINO encoder
    encoder = AutoModel.from_pretrained("facebook/dinov2-small")
    embedding_dim = encoder.config.hidden_size

    num_patches = (cfg.image_size // cfg.patch_size) ** 2
    embedding_dim += cfg.dinowm.proprio_embed_dim + cfg.dinowm.action_embed_dim  # Total embedding size

    logging.info(f"Patches: {num_patches}, Embedding dim: {embedding_dim}")

    # Build causal predictor (transformer that predicts next latent states)
    predictor = swm.wm.dinowm.CausalPredictor(
        num_patches=num_patches,
        num_frames=cfg.dinowm.history_size,
        dim=embedding_dim,
        **cfg.predictor,
    )

    # Build action and proprioception encoders
    effective_act_dim = cfg.frameskip * cfg.dinowm.action_dim
    action_encoder = swm.wm.dinowm.Embedder(in_chans=effective_act_dim, emb_dim=cfg.dinowm.action_embed_dim)
    proprio_encoder = swm.wm.dinowm.Embedder(in_chans=cfg.dinowm.proprio_dim, emb_dim=cfg.dinowm.proprio_embed_dim)

    logging.info(f"Action dim: {effective_act_dim}, Proprio dim: {cfg.dinowm.proprio_dim}")

    # Assemble world model
    world_model = swm.wm.DINOWM(
        encoder=spt.backbone.EvalOnly(encoder),
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        history_size=cfg.dinowm.history_size,
        num_pred=cfg.dinowm.num_preds,
        device="cuda",
    )

    # Wrap in stable_spt Module with separate optimizers for each component
    def add_opt(module_name, lr):
        return {"modules": str(module_name), "optimizer": {"type": "AdamW", "lr": lr}}

    world_model = spt.Module(
        model=world_model,
        forward=forward,
        optim={
            "predictor_opt": add_opt("model.predictor", cfg.predictor_lr),
            "proprio_opt": add_opt("model.proprio_encoder", cfg.proprio_encoder_lr),
            "action_opt": add_opt("model.action_encoder", cfg.action_encoder_lr),
        },
    )
    return world_model


# ============================================================================
# Training Setup
# ============================================================================
def setup_pl_logger(cfg):
    if not cfg.wandb.enable:
        return None

    wandb_run_id = cfg.wandb.get("run_id", None)
    wandb_logger = WandbLogger(
        name="dino_wm",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        resume="allow" if wandb_run_id else None,
        id=wandb_run_id,
        log_model=False,
    )

    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))
    return wandb_logger


class ModelObjectCallBack(Callback):
    """Callback to pickle model after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                output_path = Path(
                    self.dirpath,
                    f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt",
                )
                torch.save(pl_module, output_path)
                logging.info(f"Saved world model object to {output_path}")
            # Additionally, save at final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                final_path = self.dirpath / f"{self.filename}.ckpt"
                torch.save(pl_module, final_path)
                logging.info(f"Saved final world model object to {final_path}")


# ============================================================================
# Main Entry Point
# ============================================================================
@hydra.main(version_base=None, config_path="./", config_name="config")
def run(cfg):
    """Run training of predictor"""

    wandb_logger = setup_pl_logger(cfg)
    data = get_data(cfg)
    world_model = get_world_model(cfg)

    cache_dir = swm.data.get_cache_dir()
    dump_object_callback = ModelObjectCallBack(dirpath=cache_dir, filename=cfg.output_model_name, epoch_interval=1)
    # checkpoint_callback = ModelCheckpoint(dirpath=cache_dir, filename=f"{cfg.output_model_name}_weights")

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[dump_object_callback],
        num_sanity_val_steps=1,
        logger=wandb_logger,
        enable_checkpointing=True,
        ckpt_path=f"{cache_dir}/{cfg.output_model_name}_weights.ckpt",
    )

    manager = spt.Manager(trainer=trainer, module=world_model, data=data)
    manager()


if __name__ == "__main__":
    run()
