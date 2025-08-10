from xml.parsers.expat import model
import hydra
import lightning as pl
import minari
from omegaconf import OmegaConf
import stable_ssl as ssl
import torch
from einops import rearrange, repeat
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from stable_ssl.data import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel
from lightning.pytorch.callbacks import ModelCheckpoint
from xenoworlds.predictor import CausalPredictor


class Config:
    """Configuration for the training script"""

    num_workers: int = 4  # 16
    batch_size: int = 4  # 32 reduce for 8 gpu

    # encoder
    encoder_lr: float = 1e-6
    img_size: int = 224
    num_hist: int = 3
    num_pred: int = 1
    frameskip: int = 5
    num_patches: int = 1
    train_encoder: bool = False

    # action encoder
    action_encoder_lr: float = 5e-4
    action_emb_dim: int = 10
    normalize_action: True

    # proprio encoder
    proprio_encoder_lr: float = 5e-4
    proprio_emb_dim: int = 10

    # predictor
    predictor_lr: float = 5e-4
    has_predictor: bool = True

    # decoder
    decoder_lr: float = 3e-4
    has_decoder: bool = False


class Embedder(torch.nn.Module):
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
        x = x.float()
        x = x.permute(0, 2, 1)  # (B, T, B) -> (B, D, T)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)
        return x


def get_data():
    """Return data and action space dim for training predictor"""

    # -- number of rollout steps to include in the dataset
    num_steps = Config.num_hist + Config.num_pred

    # -- make transform operations
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # transform = transforms.Compose(
    #     transforms.ToImage(
    #         mean=mean,
    #         std=std,
    #         source="observations.pixels",
    #         target="observations.pixels",
    #     ),
    # )

    # from https://github.com/gaoyuezhou/dino_wm/blob/main/datasets/img_transforms.py
    trans_key = "observations.pixels"
    transform = transforms.Compose(
        transforms.Resize(Config.img_size, source=trans_key, target=trans_key),
        transforms.CenterCrop(Config.img_size, source=trans_key, target=trans_key),
        transforms.ToImage(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            source=trans_key,
            target=trans_key,
        ),
    )

    # -- load dataset
    minari_dataset = minari.load_dataset(
        "dinowm/pusht_noise-v0", download=True
    )  # xenoworlds/PushT-v1
    dataset = ssl.data.MinariStepsDataset(
        minari_dataset, num_steps=num_steps, transform=transform
    )
    train_set, val_set = ssl.data.random_split(dataset, lengths=[0.9, 0.1])
    train = DataLoader(
        train_set,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        drop_last=True,
    )
    val = DataLoader(
        val_set, batch_size=Config.batch_size, num_workers=Config.num_workers
    )
    data_module = ssl.data.DataModule(train=train, val=val)

    # -- determine action space dimension
    action = dataset[0]["actions"]
    action_dim = (
        sum(a.size for a in action.values())
        if isinstance(action, dict)
        else action.size
    )

    action_dim //= num_steps
    proprio_dim = dataset[0]["observations"]["proprio"].shape[-1]

    return data_module, action_dim, proprio_dim


def forward(self, batch, stage):
    """Forward pass for predictor training"""

    def encode(obs, actions, proprio):
        with torch.amp.autocast(enabled=False, device_type="cuda"):
            # -- encode actions
            actions = self.action_encoder(actions)  # (B,T,A) -> (B,T,A_emb)
            batch["action_emb"] = actions

            # -- encode proprioceptive
            proprio = self.proprio_encoder(proprio)  # (B,T,P) -> (B,T,P_emb)
            batch["proprio_emb"] = proprio

        # -- encode observations to get states
        B, T, C, H, W = obs.shape
        obs = rearrange(obs, "b t ... -> (b t) ...")

        # get the state
        state = self.backbone(obs).last_hidden_state  # (B*T, n_patches, D)
        state = state[:, 1:, :]  # drop cls token
        batch["state_emb"] = state
        state = z = rearrange(state, "(b t) p d -> b t p d", b=B)

        # -- merge state, action, proprio
        n_patches = state.shape[2]
        # share action/proprio embedding accros patches for each time step
        proprio_repeat = repeat(proprio.unsqueeze(2), "b t 1 d -> b t p d", p=n_patches)
        actions_repeat = repeat(actions.unsqueeze(2), "b t 1 d -> b t p d", p=n_patches)
        # z (B, T, P, dim+A_emb+P_emb)
        z = torch.cat([state, actions_repeat, proprio_repeat], dim=3)
        batch["embeddings"] = z
        return z

    # -- predict the next state
    def predict(z):
        T = z.shape[1]
        z = rearrange(z, "b t p d -> b (t p) d")
        preds = self.predictor(z)
        preds = rearrange(preds, "b (t p) d -> b t p d", t=T)
        batch["preds"] = preds
        return preds

    # -- preprocess inputs
    actions = batch["actions"]

    if type(actions) is dict:
        actions = [a.flatten(2) for a in actions.values()]
        actions = torch.cat(actions, -1)
    else:
        actions.flatten(2)

    proprio = batch["observations"]["proprio"]
    obs = batch["observations"]["pixels"]

    # -- compute prediction error
    z = encode(obs, actions, proprio)
    z_src = z[:, : Config.num_hist, :, :]
    z_tgt = z[:, Config.num_pred :, :, :]
    z_preds = predict(z_src)

    action_dim = Config.action_emb_dim
    proprio_dim = Config.proprio_emb_dim

    # remove action predicted by the predictor
    z_preds_actionless = z_preds[:, :, :, :-action_dim]
    z_tgt_actionless = z_tgt[:, :, :, :-action_dim].detach()

    # visual part of the latent
    z_pred_visual = z_preds_actionless[:, :, :, :-proprio_dim]
    z_tgt_visual = z_tgt_actionless[:, :, :, :-proprio_dim].detach()

    # proprio part of the latent
    z_pred_proprio = z_preds_actionless[:, :, :, -proprio_dim:]
    z_tgt_proprio = z_tgt_actionless[:, :, :, -proprio_dim:].detach()

    z_visual_loss = F.mse_loss(z_pred_visual, z_tgt_visual)
    z_proprio_loss = F.mse_loss(z_pred_proprio, z_tgt_proprio)
    z_loss = F.mse_loss(z_preds_actionless, z_tgt_actionless)

    # NOTE: can add decoder reconstruction here if needed

    prefix = "" if self.training else "val_"
    batch[f"{prefix}loss"] = z_loss
    batch[f"{prefix}visual_loss"] = z_visual_loss
    batch[f"{prefix}proprio_loss"] = z_proprio_loss

    # -- log losses
    losses_dict = {k: v.item() for k, v in batch.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, on_epoch=True)
    return batch


def get_world_model(action_dim, proprio_dim):
    """Return stable_ssl module with world model"""

    config = AutoConfig.from_pretrained("facebook/dinov2-small")
    encoder = AutoModel.from_config(config)
    emb_dim = config.hidden_size
    num_patches = (Config.img_size // config.patch_size) ** 2

    logging.info(f"Encoder: {encoder}, emb_dim: {emb_dim}, num_patches: {num_patches}")

    if not Config.train_encoder:
        encoder = ssl.backbone.EvalOnly(encoder)

    # -- create predictor
    predictor = CausalPredictor(
        num_patches=num_patches,
        num_frames=Config.num_hist,
        dim=emb_dim + Config.proprio_emb_dim + Config.action_emb_dim,
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
    action_encoder = Embedder(in_chans=action_dim, emb_dim=Config.action_emb_dim)
    logging.info(f"Action dim: {action_dim}, action emb dim: {Config.action_emb_dim}")

    # -- create proprioceptive encoder
    proprio_encoder = Embedder(in_chans=proprio_dim, emb_dim=Config.proprio_emb_dim)
    logging.info(
        f"Proprio dim: {proprio_dim}, proprio emb dim: {Config.proprio_emb_dim}"
    )

    # NOTE: can add a decoder here if needed

    # -- world model as a stable_ssl module
    world_model = ssl.Module(
        backbone=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        forward=forward,
    )

    return world_model


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
    )

    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))
    return wandb_logger


@hydra.main(version_base=None, config_path="./", config_name="slurm")
def run(cfg):
    """Run training of predictor"""

    wandb_logger = setup_pl_logger(cfg)
    data, action_dim, proprio_dim = get_data()
    world_model = get_world_model(action_dim, proprio_dim)

    # compute limit_train_size, one epoch 2h so to get 15min we need 1/8 of the epoch
    num_epochs = 100
    total_batches = len(data.train_dataloader()) * num_epochs
    limit_train_size = int(len(data.train_dataloader()) / 8)
    new_epoch_size = total_batches // limit_train_size

    logging.info(
        f"Total batches: {total_batches}, limit_train_size: {limit_train_size}, new_epoch_size: {new_epoch_size}"
    )

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/", save_top_k=1, monitor="val_loss", mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="dino_wm",
    )

    trainer = pl.Trainer(
        max_epochs=new_epoch_size,
        num_sanity_val_steps=1,
        # precision="16-mixed", causes NaN loss
        logger=wandb_logger,
        limit_train_batches=limit_train_size,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
    )

    manager = ssl.Manager(
        trainer=trainer,
        module=world_model,
        data=data,
    )

    manager()

    # TODO add multiple optimizer


if __name__ == "__main__":
    run()
