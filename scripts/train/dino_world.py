import lightning as pl
import minari
import stable_ssl as ssl
import torch
import torchvision


from stable_ssl.data import transforms
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    ViTConfig,
    ViTModel,
)


class Config:
    """Configuration for the training script"""

    # encoder
    encoder_lr: float = 1e-6
    img_size: int = 224
    num_hist: int = 3
    num_pred: int = 1
    frameskip: int = 5
    concat_dim: int = 1
    train_encoder: bool = False

    # decoder
    decoder_lr: float = 3e-4
    has_decoder: bool = False

    # action encoder
    action_encoder_lr: float = 5e-4
    action_embed_dim: int = 10
    normalize_action: True

    # predictor
    predictor_lr: float = 5e-4
    has_predictor: bool = True


class ProprioEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_frames=1,
        tubelet_size=1,
        in_chans=8,
        emb_dim=10,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.emb_dim = emb_dim

        self.patch_embed = torch.nn.Conv1d(
            in_chans, emb_dim, kernel_size=tubelet_size, stride=tubelet_size
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, B) -> (B, D, T)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)
        return x


def get_data():
    """Return data and action space dim for training predictor"""

    # -- number of rollout steps to include in the dataset
    num_steps = 25

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
    minari_dataset = minari.load_dataset("dinowm/pusht_noise-v0", download=True)
    dataset = ssl.data.MinariStepsDataset(
        minari_dataset, num_steps=num_steps, transform=transform
    )  # TODO add frameskip stuff
    train_set, val_set = ssl.data.random_split(dataset, lengths=[0.9, 0.1])
    train = DataLoader(train_set, batch_size=32, num_workers=16, drop_last=True)
    val = DataLoader(val_set, batch_size=32, num_workers=16)
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

    # NOTE: can add decoder reconstruction here if needed

    return batch


def get_world_model(action_dim):
    """Return stable_ssl module with world model"""

    config = AutoConfig.from_pretrained("facebook/dinov2-small")
    encoder = AutoModelForImageClassification.from_config(config)
    hidden_dim = encoder.classifier.in_features
    encoder.classifier = torch.nn.Identity()

    if not Config.train_encoder:
        encoder = ssl.backbone.EvalOnly(encoder)

    # -- create predictor

    # create vit predictor
    # config from: https://github.com/gaoyuezhou/dino_wm/blob/main/conf/predictor/vit.yaml
    pred_cfg = ViTConfig(
        hidden_size=2048,
        num_hidden_layers=6,
        num_attention_heads=16,
        intermediate_size=hidden_dim * 4,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    predictor = ViTModel(pred_cfg)
    predictor.embeddings.dropout = torch.nn.Identity()  # embed dropout

    # -- create action encoder
    action_encoder = ProprioEmbedding()

    # predictor = torchvision.ops.MLP(
    #     hidden_dim + action_dim,
    #     [1024, 1024, hidden_dim],
    #     norm_layer=torch.nn.BatchNorm1d,
    # )

    # NOTE: can add a decoder here if needed

    # -- world model as a stable_ssl module
    world_model = ssl.Module(
        backbone=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        forward=forward,
    )

    return world_model


def run():
    """Run training of predictor"""
    data, action_dim = get_data()
    world_model = get_world_model(action_dim)

    trainer = pl.Trainer(
        max_epochs=100,
        num_sanity_val_steps=1,
        precision="16-mixed",
        logger=False,
        enable_checkpointing=False,
    )
    manager = ssl.Manager(trainer=trainer, module=world_model, data=data)
    manager()


if __name__ == "__main__":
    run()
