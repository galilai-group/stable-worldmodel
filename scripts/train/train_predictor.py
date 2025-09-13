import lightning as pl
import stable_pretraining as spt
import torch
import torchvision

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForImageClassification

from pathlib import Path

import xenoworlds as xen

def get_data():
    """Return data and action space dim for training predictor"""

    # -- number of rollout steps to include in the dataset
    num_steps = 5

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
    #minari_dataset = minari.load_dataset("dinowm/pusht_noise-v0", download=True)
    #print(minari_dataset[0])

    # dataset = spt.data.MinariStepsDataset(
    #     minari_dataset, num_steps=num_steps, transform=transform
    # )

    dataset = xen.data.StepsDataset("parquet", data_files=str(Path('./dataset', "*.parquet")), split="train",
                                    num_steps=2, frameskip=1, transform=transform
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

    # -- process observation
    B, T, C, H, W = batch["pixels"].shape
    obs = batch["pixels"].float()
    obs = obs.flatten(0, 1)  # (B,T,C,H,W) -> (B*T,C,H,W)

    # -- process actions
    actions = actions.flatten(0, 1).float()  # (B,T,A) -> (B*T,A)

    # -- compute current state
    state = batch["embedding"] = self.backbone(obs)["logits"]
    D = state.shape[-1]

    # -- predict next state
    state_action_pair = torch.cat([state, actions], 1)
    preds = batch["prediction"] = self.predictor(state_action_pair)

    # -- compute prediction error
    if self.training:
        loss_fn = torch.nn.MSELoss()
        next_states = state.reshape(B, T, D)[:, 1:]  # drop s_0
        preds = preds.reshape(B, T, D)[:, :-1]  # drop s_t+1
        batch["loss"] = loss_fn(preds, next_states)

    # NOTE: can add decoder reconstruction here if needed

    return batch


def get_world_model(action_dim):
    """Return stable_spt module with world model"""
    config = AutoConfig.from_pretrained("microsoft/resnet-18")
    encoder = AutoModelForImageClassification.from_config(config)
    hidden_dim = encoder.classifier[1].in_features
    encoder.classifier[1] = torch.nn.Identity()

    # -- create predictor
    predictor = torchvision.ops.MLP(
        hidden_dim + action_dim,
        [1024, 1024, hidden_dim],
        norm_layer=torch.nn.BatchNorm1d,
    )

    # NOTE: can add a decoder here if needed

    # -- world model as a stable_spt module
    world_model = spt.Module(
        backbone=spt.backbone.EvalOnly(encoder),  # frozen encoder
        predictor=predictor,
        forward=forward,
    )

    return world_model


def run():
    """Run training of predictor"""
    data, action_dim = get_data()
    world_model = get_world_model(action_dim)

    trainer = pl.Trainer(
        max_epochs=1,
        num_sanity_val_steps=1,
        precision="16-mixed",
        logger=False,
        enable_checkpointing=False,
    )
    manager = spt.Manager(trainer=trainer, module=world_model, data=data)
    manager()


if __name__ == "__main__":
    run()
