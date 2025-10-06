from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from loguru import logger as logging
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

import stable_worldmodel as swm


def get_data(dataset_name):
    """Return data and action space dim for training predictor."""
    # -- make transform operations
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = spt.data.transforms.Compose(
        spt.data.transforms.ToImage(
            mean=mean,
            std=std,
            source="pixels",
            target="pixels",
        ),
        spt.data.transforms.ToImage(
            mean=mean,
            std=std,
            source="goal",
            target="goal",
        ),
    )

    # -- load dataset
    data_dir = swm.data.get_cache_dir()
    dataset = swm.data.StepsDataset(
        "parquet",
        data_files=str(Path(data_dir, dataset_name, "*.parquet")),
        split="train",
        num_steps=2,
        frameskip=5,
        transform=transform,
    )

    train_set, val_set = spt.data.random_split(dataset, lengths=[0.9, 0.1])

    print(f"Train set size: {len(train_set)}, Val set size: {len(val_set)}")

    train = DataLoader(train_set, batch_size=32, num_workers=4, drop_last=True)
    val = DataLoader(val_set, batch_size=32, num_workers=4)
    data_module = spt.data.DataModule(train=train, val=val)

    # -- determine action space dimension
    action = dataset[0]["action"]
    action_dim = action[0].shape[-1]
    return data_module, action_dim


def forward(self, batch, stage):
    """Forward pass for predictor training"""

    proprio_key = "proprio" if "proprio" in batch else None
    batch = self.model.encode(
        batch,
        target="embedding",
        pixels_key="pixels",
        proprio_key=proprio_key,
        action_key="action",
    )

    # predictions
    embedding = batch["embedding"][:, : self.model.history_size, :, :]  # (B, history_size, P, d)
    pred_embedding = self.model.predict(embedding)

    # targets values
    target_embedding = batch["embedding"][:, self.model.num_pred :, :, :]  # (B, T-history_size, P, d)

    # == pixels loss
    pixels_dim = batch["pixels_embed"].shape[-1]
    pixels_loss = F.mse_loss(pred_embedding[..., :pixels_dim], target_embedding[..., :pixels_dim].detach())

    batch["pixels_loss"] = pixels_loss
    loss = pixels_loss

    # == proprio loss
    if proprio_key is not None:
        proprio_dim = batch["proprio_embed"].shape[-1]
        proprio_loss = F.mse_loss(
            pred_embedding[..., pixels_dim : pixels_dim + proprio_dim],
            target_embedding[..., pixels_dim : pixels_dim + proprio_dim].detach(),
        )
        batch["proprio_loss"] = proprio_loss
        loss = loss + proprio_loss

    batch["loss"] = loss

    # == logging
    losses_dict = {k: v.item() for k, v in batch.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, on_epoch=True, sync_dist=True)

    return batch


def get_world_model(action_dim):
    """Return stable_spt module with world model"""

    # encoder = swm.wm.dinowm.DinoV2Encoder("dinov2_vits14", feature_key="x_norm_patchtokens")
    encoder = AutoModel.from_pretrained("facebook/dinov2-small")
    emb_dim = encoder.config.hidden_size  # encoder.emb_dim  # 384 for vits14

    HISTORY_SIZE = 3
    PREDICTION_HORIZON = 1
    proprio_dim = 4
    proprio_emb_dim = 10
    action_emb_dim = 10
    image_size = 224  # 224 for dinov2_vits14
    patch_size = 16  # 16 size for create 14 patches

    num_patches = (image_size // patch_size) ** 2  # 256 for 224×224

    logging.info(f"Encoder: {encoder}, emb_dim: {emb_dim}, num_patches: {num_patches}")

    encoder.train(False)
    encoder.requires_grad_(False)

    encoder.eval()

    # -- create predictor
    predictor = swm.wm.dinowm.CausalPredictor(
        num_patches=num_patches,
        num_frames=HISTORY_SIZE,
        dim=emb_dim + proprio_emb_dim + action_emb_dim,
        depth=6,
        heads=16,
        mlp_dim=2048,
        pool="mean",
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.0,
    )

    logging.info(f"Predictor: {predictor}")

    # -- create action encoder
    action_encoder = swm.wm.dinowm.Embedder(in_chans=action_dim, emb_dim=action_emb_dim)

    logging.info(f"Action dim: {action_dim}, action emb dim: {action_emb_dim}")

    # -- create proprioceptive encoder
    proprio_encoder = swm.wm.dinowm.Embedder(in_chans=proprio_dim, emb_dim=proprio_emb_dim)
    logging.info(f"Proprio dim: {proprio_dim}, proprio emb dim: {proprio_emb_dim}")

    world_model = swm.wm.DINOWM(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        history_size=HISTORY_SIZE,
        num_pred=PREDICTION_HORIZON,
        device="cuda",
    )

    # -- world model as a stable_spt module
    world_model = spt.Module(
        model=world_model,
        forward=forward,
    )
    return world_model


@hydra.main(version_base=None, config_path="./", config_name="config")
def run(cfg):
    """Run training of predictor"""
    data, action_dim = get_data(cfg.dataset_name)
    world_model = get_world_model(action_dim)

    cache_dir = swm.data.get_cache_dir()
    checkpoint_callback = ModelCheckpoint(
        dirpath=cache_dir, filename=f"{cfg.output_model_name}_weights"
    )  # , save_last=True)

    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=1,
        precision="16-mixed",
        logger=False,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer, module=world_model, data=data
    )  # , ckpt_path=f"{cfg.output_model_name}_ckpt")
    manager()

    if hasattr(cfg, "dump_object") and cfg.dump_object:
        # -- save the world model object
        output_path = Path(cache_dir, f"{cfg.output_model_name}_object.ckpt")
        torch.save(world_model.to("cpu"), output_path)
        print(f"Saved world model object to {output_path}")


if __name__ == "__main__":
    run()
