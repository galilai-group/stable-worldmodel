from collections import OrderedDict
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForImageClassification

import stable_worldmodel as swm


# def resnet_encoder(cfg):
#     assert cfg.backbone.startswith("microsoft/resnet-"), f"ResNet encoder selected but backbone {cfg.backbone} is not."
#     backbone = AutoModelForImageClassification.from_pretrained(cfg.backbone)
#     backbone.classifier[1] = torch.nn.Identity()
#     return backbone, backbone.config.hidden_sizes[-1], 1


# def vit_encoder(cfg):
#     assert cfg.backbone.startswith("google/vit-"), f"ViT encoder selected but backbone {cfg.backbone} is not ViT."
#     backbone = AutoModel.from_pretrained(cfg.backbone)
#     embedding_dim = backbone.config.hidden_size
#     num_patches = (cfg.image_size // cfg.patch_size) ** 2
#     return backbone, embedding_dim, num_patches


# def dinov1_encoder(cfg):
#     # https://huggingface.co/collections/JulianAssmann/dino-v1-models (dinov1)
#     assert cfg.backbone.startswith("facebook/dino-"), f"DINO encoder selected but backbone {cfg.backbone} is not."
#     backbone = AutoModel.from_pretrained(cfg.backbone)
#     embedding_dim = backbone.config.hidden_size
#     num_patches = (cfg.image_size // cfg.patch_size) ** 2
#     return backbone, embedding_dim, num_patches


# def dinov2_encoder(cfg):
#     # https://huggingface.co/collections/facebook/dinov2
#     assert cfg.backbone.startswith("facebook/dinov2-"), f"DINOv2 encoder selected but backbone {cfg.backbone} is not."
#     backbone = AutoModel.from_pretrained(cfg.backbone)
#     embedding_dim = backbone.config.hidden_size
#     num_patches = (cfg.image_size // cfg.patch_size) ** 2
#     return backbone, embedding_dim, num_patches


# def dinov3_encoder(cfg):
#     # https://huggingface.co/collections/facebook/dinov3
#     assert cfg.backbone.startswith("facebook/dinov3-"), f"DINOv3 encoder selected but backbone {cfg.backbone} is not."
#     backbone = AutoModel.from_pretrained(cfg.backbone)
#     embedding_dim = backbone.config.hidden_size
#     num_patches = (cfg.image_size // cfg.patch_size) ** 2
#     return backbone, embedding_dim, num_patches


# def mae_encoder(cfg):
#     # facebook/vit-mae-base, vit-mae-large
#     assert cfg.backbone.startswith("facebook/vit-mae-"), f"MAE encoder selected but backbone {cfg.backbone} is not."
#     backbone = AutoModel.from_pretrained(cfg.backbone)
#     embedding_dim = backbone.config.hidden_size
#     num_patches = (cfg.image_size // cfg.patch_size) ** 2
#     return backbone, embedding_dim, num_patches


# def ijepa_encoder(cfg):
#     # facebook/ijepa_vith14_1k, ijepa_vith14_22k (impact of pre-training dataset size)
#     # facebook/ijepa_vith14_1k, facebook/ijepa_vith16_1k (impact of resolution 224 vs 448)
#     assert cfg.backbone.startswith("facebook/ijepa-"), f"IJEPA encoder selected but backbone {cfg.backbone} is not."
#     backbone = AutoModel.from_pretrained(cfg.backbone)
#     embedding_dim = backbone.config.hidden_size
#     num_patches = (cfg.image_size // cfg.patch_size) ** 2
#     return backbone, embedding_dim, num_patches


# def clip_encoder(cfg):
#     # timm/vit_base_patch32_clip_224.metaclip_400m
#     assert cfg.backbone.startswith("timm/vit_base_patch32_clip_"), (
#         f"CLIP encoder selected but backbone {cfg.backbone} is not."
#     )
#     backbone = AutoModel.from_pretrained(cfg.backbone)
#     embedding_dim = backbone.config.num_features
#     num_patches = (cfg.image_size // cfg.patch_size) ** 2
#     return backbone, embedding_dim, num_patches


# def vjepa2_encoder(cfg):
#     # https://huggingface.co/collections/facebook/v-jepa-2
#     assert cfg.backbone.startswith("facebook/vjepa2-vit"), (
#         f"V-JEPA 2 encoder selected but backbone {cfg.backbone} is not."
#     )
#     backbone = AutoModel.from_pretrained(cfg.backbone)
#     embedding_dim = backbone.config.hidden_size
#     num_patches = (cfg.image_size // cfg.patch_size) ** 2
#     return backbone, embedding_dim, num_patches


def get_encoder(cfg):
    """Factory function to create encoder based on backbone type."""

    # Define encoder configurations
    ENCODER_CONFIGS = {
        "resnet": {
            "prefix": "microsoft/resnet-",
            "model_class": AutoModelForImageClassification,
            "embedding_attr": lambda model: model.config.hidden_sizes[-1],
            "post_init": lambda model: setattr(model.classifier, "1", torch.nn.Identity()),
            "interpolate_pos_encoding": False,
        },
        "vit": {"prefix": "google/vit-"},
        "dino": {"prefix": "facebook/dino-"},
        "dinov2": {"prefix": "facebook/dinov2-"},
        "dinov3": {"prefix": "facebook/dinov3-"},  # TODO handle resnet base in dinov3
        "mae": {"prefix": "facebook/vit-mae-"},
        "ijepa": {"prefix": "facebook/ijepa"},
        "vjepa2": {"prefix": "facebook/vjepa2-vit"},
        "siglip2": {"prefix": "google/siglip2-"},
    }

    # Find matching encoder
    encoder_type = None
    for name, config in ENCODER_CONFIGS.items():
        if cfg.backbone.startswith(config["prefix"]):
            encoder_type = name
            break

    if encoder_type is None:
        raise ValueError(f"Unsupported backbone: {cfg.backbone}")

    config = ENCODER_CONFIGS[encoder_type]

    # Load model
    backbone = config.get("model_class", AutoModel).from_pretrained(cfg.backbone)

    # CLIP style model
    if hasattr(backbone, "vision_model"):
        backbone = backbone.vision_model

    # Post-initialization if needed (e.g., ResNet)
    if "post_init" in config:
        config["post_init"](backbone)

    # Get embedding dimension
    embedding_dim = config.get("embedding_attr", lambda model: model.config.hidden_size)(backbone)

    # Determine number of patches
    is_cnn = encoder_type == "resnet"
    num_patches = 1 if is_cnn else (cfg.image_size // cfg.patch_size) ** 2

    interp_pos_enc = config.get("interpolate_pos_encoding", True)

    return backbone, embedding_dim, num_patches, interp_pos_enc


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

    all_norm_transforms = []
    for key in cfg.pyro.get("encoding", {}):
        trans_fn = norm_col_transform(dataset.dataset, key)
        trans_fn = spt.data.transforms.WrapTorchTransform(trans_fn, source=key, target=key)
        all_norm_transforms.append(trans_fn)

    # Image size must be multiple of DINO patch size (14)
    img_size = (cfg.image_size // cfg.patch_size) * DINO_PATCH_SIZE

    # Apply transforms to all steps
    transform = spt.data.transforms.Compose(
        *[get_img_pipeline(f"{col}.{i}", f"{col}.{i}", img_size) for col in ["pixels"] for i in range(cfg.n_steps)],
        *all_norm_transforms,
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

    with open_dict(cfg) as cfg:
        cfg.extra_dims = {}
        for key in cfg.pyro.get("encoding", {}):
            if key not in dataset.dataset.column_names:
                raise ValueError(f"Encoding key '{key}' not found in dataset columns.")
            inpt_dim = dataset.dataset[0][key].numel()
            cfg.extra_dims[key] = inpt_dim if key != "action" else inpt_dim * cfg.frameskip

    return spt.data.DataModule(train=train, val=val)


# ============================================================================
# Model Architecture
# ============================================================================


def get_world_model(cfg):
    """Build world model: frozen DINO encoder + trainable causal predictor."""

    def forward(self, batch, stage):
        """Forward: encode observations, predict next states, compute losses."""

        # Replace NaN values with 0 (occurs at sequence boundaries)
        for key in self.model.extra_encoders.keys():
            batch[key] = torch.nan_to_num(batch[key], 0.0)

        # Encode all timesteps into latent embeddings
        batch = self.model.encode(batch, target="embed")

        # Use history to predict next states
        embedding = batch["embed"][:, : cfg.pyro.history_size, :, :]  # (B, T-1, patches, dim)
        pred_embedding = self.model.predict(embedding)
        target_embedding = batch["embed"][:, cfg.pyro.num_preds :, :, :]  # (B, T-1, patches, dim)

        # Compute pixel reconstruction loss
        pixels_dim = batch["pixels_embed"].shape[-1]
        pixels_loss = F.mse_loss(pred_embedding[..., :pixels_dim], target_embedding[..., :pixels_dim].detach())
        batch["pixels_loss"] = pixels_loss

        start = pixels_dim
        action_dim_range = [0, 0]
        for key in self.model.extra_encoders.keys():
            emb_dim = batch[f"{key}_embed"].shape[-1]
            if key == "action":
                action_dim_range = [start, start + emb_dim]
                continue  # skip action encoding loss

            emb_dim = batch[f"{key}_embed"].shape[-1]
            pred_embedding[..., start : start + emb_dim]
            target_embedding[..., start : start + emb_dim]

            if key in self.model.extra_encoders:
                extra_loss = F.mse_loss(
                    pred_embedding[..., start : start + emb_dim],
                    target_embedding[..., start : start + emb_dim].detach(),
                )
                # loss = loss + extra_loss
                batch[f"{key}_loss"] = extra_loss

            start += emb_dim

        # Total loss
        actionless_pred = torch.cat(
            [pred_embedding[..., : action_dim_range[0]], pred_embedding[..., action_dim_range[1] :]], dim=-1
        )

        actionless_target = torch.cat(
            [target_embedding[..., : action_dim_range[0]], target_embedding[..., action_dim_range[1] :]], dim=-1
        )

        batch["loss"] = F.mse_loss(actionless_pred, actionless_target.detach())

        # Log all losses
        prefix = "train/" if self.training else "val/"
        losses_dict = {f"{prefix}{k}": v.detach() for k, v in batch.items() if "_loss" in k}
        self.log_dict(losses_dict, on_step=True, sync_dist=True)  # , on_epoch=True, sync_dist=True)

        return batch

    encoder, embedding_dim, num_patches, interp_pos_enc = get_encoder(cfg)
    embedding_dim += sum(emb_dim for emb_dim in cfg.pyro.get("encoding", {}).values())  # add all extra dims

    logging.info(f"Patches: {num_patches}, Embedding dim: {embedding_dim}")

    # Build causal predictor (transformer that predicts next latent states)

    print(">>>> DIM PREDICTOR:", embedding_dim)

    predictor = swm.wm.dinowm.CausalPredictor(
        num_patches=num_patches,
        num_frames=cfg.pyro.history_size,
        dim=embedding_dim,
        **cfg.predictor,
    )

    # Build action and proprioception encoders
    extra_encoders = OrderedDict()
    for key, emb_dim in cfg.pyro.get("encoding", {}).items():
        inpt_dim = cfg.extra_dims[key]
        extra_encoders[key] = swm.wm.pyro.Embedder(in_chans=inpt_dim, emb_dim=emb_dim)
        print(f"Build encoder for {key} with input dim {inpt_dim} and emb dim {emb_dim}")

    extra_encoders = torch.nn.ModuleDict(extra_encoders)

    # Assemble world model
    world_model = swm.wm.PYRO(
        encoder=spt.backbone.EvalOnly(encoder),
        predictor=predictor,
        extra_encoders=extra_encoders,
        history_size=cfg.pyro.history_size,
        num_pred=cfg.pyro.num_preds,
        interpolate_pos_encoding=interp_pos_enc,
    )

    # Wrap in stable_spt Module with separate optimizers for each component
    def add_opt(module_name, lr):
        return {"modules": str(module_name), "optimizer": {"type": "AdamW", "lr": lr}}

    optim_dict = {"predictor_opt": add_opt("model.predictor", cfg.predictor_lr)}
    optim_dict.update(
        {f"{key}_opt": add_opt(f"model.extra_encoders.{key}", cfg.encoding_lr) for key in extra_encoders}
    )

    world_model = spt.Module(model=world_model, forward=forward, optim=optim_dict)
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
                final_path = self.dirpath / f"{self.filename}_object.ckpt"
                torch.save(pl_module, final_path)
                logging.info(f"Saved final world model object to {final_path}")


# ============================================================================
# Main Entry Point
# ============================================================================
@hydra.main(version_base=None, config_path="./config", config_name="pusht")
def run(cfg):
    """Run training of predictor"""

    wandb_logger = setup_pl_logger(cfg)
    data = get_data(cfg)
    world_model = get_world_model(cfg)

    cache_dir = swm.data.get_cache_dir()
    dump_object_callback = ModelObjectCallBack(
        dirpath=cache_dir,
        filename=cfg.output_model_name,
        epoch_interval=10,
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
        module=world_model,
        data=data,
        ckpt_path=f"{cache_dir}/{cfg.output_model_name}_weights.ckpt",
    )
    manager()


if __name__ == "__main__":
    run()
