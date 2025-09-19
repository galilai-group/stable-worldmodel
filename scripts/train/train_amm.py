import lightning as pl
import minari
import stable_ssl as ssl
import torch
import torchvision

from stable_ssl.data import transforms
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForImageClassification


def get_data(num_steps=2):
    """Return data and action space dim for training predictor"""

    # -- make transform operations
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        transforms.ToImage(
            mean=mean,
            std=std,
            source="observations.pixels",
            target="observations.pixels",
        ),
    )

    # -- load dataset
    minari_dataset = minari.load_dataset(
        "swm/ImagePositioning-v1", download=True
    )
    dataset = ssl.data.MinariStepsDataset(
        minari_dataset, num_steps=num_steps, transform=transform
    )
    train_set, val_set = ssl.data.random_split(dataset, lengths=[0.5, 0.5])
    train = DataLoader(train_set, batch_size=2, num_workers=20, drop_last=True)
    val = DataLoader(val_set, batch_size=2, num_workers=10)
    data_module = ssl.data.DataModule(train=train, val=val)

    # -- determine action space dimension
    action = dataset[0]["actions"]
    action_dim = (
        sum(a.size for a in action.values())
        if isinstance(action, dict)
        else action.size
    )
    action_dim //= num_steps

    return data_module, action_dim


def forward(self, batch, stage):
    """Forward pass for predictor training"""

    actions = batch["actions"]

    if type(actions) is dict:
        actions = [a.flatten(2) for a in actions.values()]
        actions = torch.cat(actions, -1)
    else:
        actions.flatten(2)

    # -- process actions
    actions = actions.flatten(0, 1)  # (B,T,A) -> (B*T,A)

    # -- process observation
    B, T, C, H, W = batch["observations"]["pixels"].shape
    obs = batch["observations"]["pixels"]
    obs = obs.flatten(0, 1)  # (B,T,C,H,W) -> (B*T,C,H,W)
    obs = self.backbone.preprocess(obs)

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

    return batch


def get_world_model(action_dim):
    """Return stable_ssl module with world model"""
    config = AutoConfig.from_pretrained("microsoft/resnet-18")
    encoder = AutoModelForImageClassification.from_config(config)
    encoder.classifier[1] = torch.nn.Identity()

    # hidden dim
    hidden_dim = encoder.classifier[0].in_features

    # -- create predictor
    predictor = torchvision.ops.MLP(
        hidden_dim + action_dim,
        [1024, 1024, hidden_dim],
        norm_layer=torch.nn.BatchNorm1d,
    )

    # -- world model as a stable_ssl module
    world_model = ssl.Module(
        backbone=ssl.backbone.EvalOnly(encoder),  # frozen encoder
        predictor=predictor,
        forward=forward,
    )

    return world_model


def run(num_steps=2):
    """Run training of predictor"""
    data, action_dim = get_data(num_steps=num_steps)
    world_model = get_world_model(action_dim)

    trainer = pl.Trainer(
        max_epochs=1,
        num_sanity_val_steps=1,
        precision="16-mixed",
        logger=False,
        enable_checkpointing=False,
    )
    manager = ssl.Manager(trainer=trainer, module=world_model, data=data)
    manager()


if __name__ == "__main__":
    TRAIN_STEPS = 2
    run(num_steps=TRAIN_STEPS)
