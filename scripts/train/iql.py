import os
from collections import OrderedDict
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from einops import rearrange, repeat
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

    def norm_col_transform(dataset, col="pixels"):
        """Normalize column to zero mean, unit variance."""
        data = torch.from_numpy(dataset.get_col_data(col)[:])
        mean = data.mean(0).unsqueeze(0).clone()
        std = data.std(0).unsqueeze(0).clone()
        return lambda x: ((x - mean) / std).float()

    cache_dir = None
    if not hasattr(cfg, "local_cache_dir"):
        cache_dir = os.environ.get("SLURM_TMPDIR", None)

    dataset = swm.data.HDF5Dataset(
        cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        cache_dir=cache_dir,
        keys_to_load=["pixels", "action", "proprio"],
        keys_to_cache=["action", "proprio"],
    )

    norm_action_transform = norm_col_transform(dataset, "action")
    norm_proprio_transform = norm_col_transform(dataset, "proprio")

    # Apply transforms to all steps and goal observations
    transform = spt.data.transforms.Compose(
        get_img_pipeline("pixels", "pixels", cfg.image_size),
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

    dataset = swm.data.GoalDataset(
        dataset=dataset,
        goal_probabilities=(0.3, 0.5, 0.2),  # random, future, current
        goal_keys={"pixels": "goal_pixels", "proprio": "goal_proprio"},
        seed=cfg.seed,
    )

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
        prefetch_factor=2,
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )
    val = DataLoader(val_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)

    return spt.data.DataModule(train=train, val=val)


# ============================================================================
# Model Architecture
# ============================================================================
def get_gciql_value_model(cfg):
    """Build goal-conditioned behavvioral cloning policy: frozen encoder (e.g. DINO) + trainable action predictor."""

    expectile_loss = swm.wm.iql.ExpectileLoss(tau=0.9)

    def forward_value(self, batch, stage):
        """Forward: encode observations and goals, predict actions, compute losses."""

        proprio_key = "proprio" if "proprio" in batch else None

        # Replace NaN values with 0 (occurs at sequence boundaries)
        if proprio_key is not None:
            batch[proprio_key] = torch.nan_to_num(batch[proprio_key], 0.0)
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        # Encode all timesteps into latent embeddings
        batch = self.model.encode(
            batch,
            target="embed",
            pixels_key="pixels",
        )

        # Encode goal into latent embedding
        batch = self.model.encode(
            batch, target="goal_embed", pixels_key="goal_pixels", emb_keys=["proprio"], prefix="goal_"
        )

        # Use history to predict values
        embedding = batch["embed"][:, : cfg.dinowm.history_size, :, :]  # (B, T, patches, dim)
        target_embedding = batch["embed"][:, cfg.dinowm.num_preds :, :, :]  # (B, T, patches, dim)
        goal_embedding = batch["goal_embed"]  # (B, 1, patches, dim)

        # Reshape to (B, T*P, dim) for the predictor
        embedding_flat = rearrange(embedding, "b t p d -> b (t p) d")
        goal_embedding_flat = rearrange(goal_embedding, "b t p d -> b (t p) d")
        target_embedding_flat = rearrange(target_embedding, "b t p d -> b (t p) d")

        value_pred = self.model.value_predictor.forward_student(embedding_flat, goal_embedding_flat)
        with torch.no_grad():
            gamma = 0.99
            value_target = gamma * self.model.value_predictor.forward_teacher(
                target_embedding_flat, goal_embedding_flat
            )
            goal_embedding_repeated = repeat(goal_embedding, "b 1 p d -> b t p d", t=embedding.shape[1])
            eq_mask = torch.isclose(embedding, goal_embedding_repeated, atol=1e-6, rtol=1e-5).all(dim=(-1, -2))
            reward = -(~eq_mask).float().unsqueeze(-1)
            value_target += reward

        # Compute action MSE
        value_loss = expectile_loss(value_pred, value_target.detach())
        batch["value_loss"] = value_loss
        batch["loss"] = value_loss

        # Log all losses
        prefix = "train/" if self.training else "val/"
        losses_dict = {f"{prefix}{k}": v.detach() for k, v in batch.items() if "_loss" in k}
        losses_dict[f"{prefix}loss"] = batch["loss"].detach()
        self.log_dict(losses_dict, on_step=True, sync_dist=True)  # , on_epoch=True, sync_dist=True)

        return batch

    # Load frozen DINO encoder
    encoder = AutoModel.from_pretrained("facebook/dinov2-small")
    embedding_dim = encoder.config.hidden_size

    # Calculate actual number of patches based on the actual image size used by DINO
    assert cfg.image_size % cfg.patch_size == 0, "Image size must be multiple of patch size"
    num_patches = (cfg.image_size // cfg.patch_size) ** 2
    embedding_dim += cfg.dinowm.proprio_embed_dim  # Total embedding size

    logging.info(f"Patches: {num_patches}, Embedding dim: {embedding_dim}")

    # Build causal predictor (transformer that predicts next actions)
    effective_act_dim = cfg.frameskip * cfg.dinowm.action_dim  # NOTE: 'frameskip' > 1 is used to predict action chunks
    action_predictor = swm.wm.iql.Predictor(
        num_patches=num_patches,
        num_frames=cfg.dinowm.history_size,
        dim=embedding_dim,
        out_dim=effective_act_dim,
        **cfg.predictor,
    )

    value_predictor = swm.wm.iql.Predictor(
        num_patches=num_patches,
        num_frames=cfg.dinowm.history_size,
        dim=embedding_dim,
        out_dim=1,
        **cfg.predictor,
    )
    wrapped_value_predictor = spt.TeacherStudentWrapper(
        value_predictor,
        warm_init=True,
        base_ema_coefficient=0.995,
        final_ema_coefficient=0.995,
    )

    # Build proprioception encoder
    extra_encoders = OrderedDict()
    extra_encoders["proprio"] = swm.wm.iql.Embedder(
        in_chans=cfg.dinowm.proprio_dim, emb_dim=cfg.dinowm.proprio_embed_dim
    )
    extra_encoders = torch.nn.ModuleDict(extra_encoders)

    logging.info(f"Action dim: {effective_act_dim}, Proprio dim: {cfg.dinowm.proprio_dim}")

    # Assemble policy
    gciql_model = swm.wm.iql.GCIQL(
        encoder=spt.backbone.EvalOnly(encoder),
        action_predictor=action_predictor,
        value_predictor=wrapped_value_predictor,
        extra_encoders=extra_encoders,
        history_size=cfg.dinowm.history_size,
        num_pred=cfg.dinowm.num_preds,
    )

    # Wrap in stable_spt Module with separate optimizers for each component
    def add_opt(module_name, lr):
        return {"modules": str(module_name), "optimizer": {"type": "AdamW", "lr": lr}}

    gciql_value_model = spt.Module(
        model=gciql_model,
        forward=forward_value,
        optim={
            "value_predictor_opt": add_opt("model.value_predictor", cfg.predictor_lr),
            "proprio_opt": add_opt("model.extra_encoders.proprio", cfg.proprio_encoder_lr),
        },
    )
    return gciql_value_model


def get_gciql_action_model(cfg, trained_value_model):
    """Build goal-conditioned behavvioral cloning policy: frozen encoder (e.g. DINO) + trainable action predictor."""

    def forward_action(self, batch, stage):
        """Forward: encode observations and goals, predict actions, compute losses."""

        proprio_key = "proprio" if "proprio" in batch else None

        # Replace NaN values with 0 (occurs at sequence boundaries)
        if proprio_key is not None:
            batch[proprio_key] = torch.nan_to_num(batch[proprio_key], 0.0)
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        with torch.no_grad():
            # Encode all timesteps into latent embeddings
            batch = self.model.encode(
                batch,
                target="embed",
                pixels_key="pixels",
            )

            # Encode goal into latent embedding
            batch = self.model.encode(
                batch, target="goal_embed", pixels_key="goal_pixels", emb_keys=["proprio"], prefix="goal_"
            )

            # Use history to predict next actions
            embedding = batch["embed"][:, : cfg.dinowm.history_size, :, :]  # (B, T, patches, dim)
            target_embedding = batch["embed"][:, cfg.dinowm.num_preds :, :, :]  # (B, T, patches, dim)
            goal_embedding = batch["goal_embed"]  # (B, 1, patches, dim)

            # Reshape to (B, T*P, dim) for the predictor
            embedding_flat = rearrange(embedding, "b t p d -> b (t p) d")
            goal_embedding_flat = rearrange(goal_embedding, "b t p d -> b (t p) d")
            target_embedding_flat = rearrange(target_embedding, "b t p d -> b (t p) d")
            gamma = 0.99
            value = self.model.value_predictor(embedding_flat, goal_embedding_flat)
            value_target = self.model.value_predictor(target_embedding_flat, goal_embedding_flat)
            goal_embedding_repeated = repeat(goal_embedding, "b 1 p d -> b t p d", t=embedding.shape[1])
            eq_mask = torch.isclose(embedding, goal_embedding_repeated, atol=1e-6, rtol=1e-5).all(dim=(-1, -2))
            reward = -(~eq_mask).float().unsqueeze(-1)
            advantage = reward + gamma * value_target - value  # (B, T, 1)

        action_pred = self.model.action_predictor(embedding_flat.detach(), goal_embedding_flat.detach())

        # policy is extracted via AWR
        beta = 3.0
        # TODO how to compute std?
        action_loss = torch.exp(advantage.detach() * beta) * F.mse_loss(
            action_pred, batch["action"][:, : cfg.dinowm.history_size], reduction="none"
        )
        action_loss = action_loss.mean()
        batch["actor_loss"] = action_loss
        batch["loss"] = action_loss

        # Log all losses
        prefix = "train/" if self.training else "val/"
        losses_dict = {f"{prefix}{k}": v.detach() for k, v in batch.items() if "_loss" in k}
        losses_dict[f"{prefix}loss"] = batch["loss"].detach()
        self.log_dict(losses_dict, on_step=True, sync_dist=True)  # , on_epoch=True, sync_dist=True)

        return batch

    # Assemble policy
    trained_value_model.model.extra_encoders.eval()
    gciql_model = swm.wm.iql.GCIQL(
        encoder=spt.backbone.EvalOnly(trained_value_model.model.encoder.backbone),
        action_predictor=trained_value_model.model.action_predictor,
        value_predictor=spt.backbone.EvalOnly(trained_value_model.model.value_predictor.student),
        extra_encoders=trained_value_model.model.extra_encoders,
        history_size=cfg.dinowm.history_size,
        num_pred=cfg.dinowm.num_preds,
    )

    # Wrap in stable_spt Module with separate optimizers for each component
    def add_opt(module_name, lr):
        return {"modules": str(module_name), "optimizer": {"type": "AdamW", "lr": lr}}

    gciql_action_model = spt.Module(
        model=gciql_model,
        forward=forward_action,
        optim={
            "action_predictor_opt": add_opt("model.action_predictor", cfg.predictor_lr),
        },
    )
    return gciql_action_model


# ============================================================================
# Training Setup
# ============================================================================
def setup_pl_logger(cfg):
    if not cfg.wandb.enable:
        return None

    wandb_run_id = cfg.wandb.get("run_id", None)
    wandb_logger = WandbLogger(
        name="dino_gciql",
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
                final_path = self.dirpath / f"{self.filename}_object.ckpt"
                torch.save(pl_module, final_path)
                logging.info(f"Saved final world model object to {final_path}")


# ============================================================================
# Main Entry Point
# ============================================================================
@hydra.main(version_base=None, config_path="./", config_name="config")
def run(cfg):
    """Run training of IQL goal-conditioned policy."""

    wandb_logger = setup_pl_logger(cfg)
    data = get_data(cfg)

    # First train value function
    gciql_model = get_gciql_value_model(cfg)

    cache_dir = swm.data.utils.get_cache_dir()
    dump_object_callback = ModelObjectCallBack(
        dirpath=cache_dir,
        filename=f"{cfg.output_model_name}_value",
        epoch_interval=3,
    )
    # checkpoint_callback = ModelCheckpoint(dirpath=cache_dir, filename=f"{cfg.output_model_name}_weights")

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[dump_object_callback],
        num_sanity_val_steps=1,
        logger=wandb_logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=gciql_model,
        data=data,
        ckpt_path=f"{cache_dir}/{cfg.output_model_name}_value_weights.ckpt",
    )
    manager()

    # Extract policy from trained value function
    gciql_action_model = get_gciql_action_model(cfg, gciql_model)

    dump_object_callback = ModelObjectCallBack(
        dirpath=cache_dir,
        filename=f"{cfg.output_model_name}_policy",
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
        module=gciql_action_model,
        data=data,
        ckpt_path=f"{cache_dir}/{cfg.output_model_name}_policy_weights.ckpt",
    )
    manager()


if __name__ == "__main__":
    run()
