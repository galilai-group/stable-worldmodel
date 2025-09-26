import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm

import torch
import torchvision

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForImageClassification
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path


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
    data_dir = swm.utils.get_cache_dir()
    dataset = swm.data.StepsDataset(
        "parquet",
        data_files=str(Path(data_dir, dataset_name, "*.parquet")),
        split="train",
        num_steps=2,
        frameskip=1,
        transform=transform,
    )

    train_set, val_set = spt.data.random_split(dataset, lengths=[0.9, 0.1])

    print(f"Train set size: {len(train_set)}, Val set size: {len(val_set)}")

    train = DataLoader(train_set, batch_size=128, num_workers=1, drop_last=True)
    val = DataLoader(val_set, batch_size=128, num_workers=1)
    data_module = spt.data.DataModule(train=train, val=val)

    # -- determine action space dimension
    action = dataset[0]["action"]
    action_dim = action[0].shape[-1]
    return data_module, action_dim


def forward(self, batch, stage):
    """Forward pass for predictor training"""

    actions = batch["action"]

    # -- process actions
    actions = actions.flatten(0, 1).float()  # (B,T,A) -> (B*T,A)

    # -- compute current state
    B, T = batch["pixels"].shape[:2]
    batch["pixels"] = batch["pixels"].flatten(0, 1)  # (B,T,C,H,W) -> (B*T,C,H,W)
    batch = self.model.encode(batch)
    D = batch["embedding"].shape[-1]

    # -- predict next state
    preds = self.model.predict(batch, actions)

    # -- compute prediction error
    if self.training:
        state = batch["embedding"]
        loss_fn = torch.nn.MSELoss()
        next_states = state.reshape(B, T, D)[:, 1:]  # drop s_0
        preds = preds.reshape(B, T, D)[:, :-1]  # drop s_t+1
        batch["loss"] = loss_fn(preds, next_states.detach())

    # NOTE: can add decoder reconstruction here if needed

    return batch


def get_world_model(action_dim):
    """Return stable_spt module with world model"""
    world_model = swm.wm.DummyWorldModel((224, 224, 3), action_dim)
    # NOTE: can add a decoder here if needed

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

    cache_dir = swm.utils.get_cache_dir()
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
