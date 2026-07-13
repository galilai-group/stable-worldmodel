import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from stable_pretraining import data as dt

import stable_worldmodel as swm
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.utils import save_pretrained


def maskgit_mask(tokens, mask_token_id, mask_floor=0.5, mask_ceiling=1.0):
    """Per-example p ~ U(floor, ceiling), Bernoulli-mask frames [1:]. Frame 0 is kept clean."""
    B, T, S = tokens.shape
    labels = tokens
    input_ids = tokens.clone()
    p = mask_floor + (mask_ceiling - mask_floor) * torch.rand(B, device=tokens.device)
    mask = torch.rand(B, T - 1, S, device=tokens.device) < p[:, None, None]
    input_ids[:, 1:][mask] = mask_token_id
    return input_ids, labels


def get_img_preprocessor(source: str, target: str, img_size: int = 64):
    """Match LeWM's transform stack but at Genie's native 64x64."""
    imagenet_stats = dt.dataset_stats.ImageNet
    return dt.transforms.Compose(
        dt.transforms.ToImage(**imagenet_stats, source=source, target=target),
        dt.transforms.Resize(img_size, source=source, target=target),
    )


def _video_minus_one_to_one(pixels: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalised (B, T, C, H, W) -> Genie's (B, T, H, W, C) in [-1, 1].

    The dataset transform leaves pixels imagenet-normalised; we undo that
    cheaply and rescale so ST_ViViT and LAM see the conventional [-1, 1] range.
    """
    mean = pixels.new_tensor(dt.dataset_stats.ImageNet["mean"]).view(1, 1, 3, 1, 1)
    std = pixels.new_tensor(dt.dataset_stats.ImageNet["std"]).view(1, 1, 3, 1, 1)
    pixels = pixels * std + mean                # back to [0, 1]
    pixels = pixels.mul(2).sub(1).clamp(-1, 1)
    return pixels.permute(0, 1, 3, 4, 2).contiguous()


#########################
##      tokenizer      ##
#########################

def tokenizer_forward(self, batch, stage, cfg):
    video = _video_minus_one_to_one(batch["pixels"])
    tok = self.model.tokenizer

    video_hat, _, vq_loss = tok(video)
    recon_loss = F.mse_loss(video_hat, video)
    loss = recon_loss + vq_loss

    out = {
        "loss": loss,
        f"{stage}/recon_loss": recon_loss.detach(),
        f"{stage}/vq_loss": vq_loss.detach(),
    }
    self.log_dict({k: v for k, v in out.items() if k != "loss"},
                  on_step=True, sync_dist=True)
    return out




############################################
##   lam + fwd dynamics + action encoder  ##
############################################

def joint_forward(self, batch, stage, cfg):
    video = _video_minus_one_to_one(batch["pixels"])     # (B, T, H, W, C)
    raw_action = batch["action"].float()                 # (B, T-1, A) or (B, T, A)

    tok = self.model.tokenizer
    lam = self.model.lam
    dyn = self.model.dynamics
    act_enc = self.model.action_encoder

    # Frozen tokenizer (no grad, eval mode).
    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)

    # LAM: inferred action codes + own reconstruction loss.
    _, action_q_TE, lam_vq_loss = lam.extract_actions(video)
    next_pred = lam.predict_next_frames(video[:, :-1], action_q_TE)
    lam_recon = F.mse_loss(next_pred, video[:, 1:])
    lam_loss = lam_recon + lam_vq_loss

    # Tokens come from the frozen tokenizer.
    with torch.no_grad():
        tokens, _, _ = tok.encode(video)

    input_ids, labels = maskgit_mask(
        tokens, dyn.mask_token_id,
        cfg.get("mask_floor", 0.5), cfg.get("mask_ceiling", 1.0),
    )
    dyn_out = dyn(input_ids, labels, actions_T=action_q_TE.detach())

    # Action encoder: align raw actions with LAM's embedding space.
    # raw_action may come in as (B, T, A) or (B, T-1, A). LAM yields T-1 actions.
    if raw_action.size(1) == action_q_TE.size(1) + 1:
        raw_action = raw_action[:, :-1]
    pred_action_embed = act_enc(raw_action)
    act_enc_loss = F.mse_loss(pred_action_embed, action_q_TE.detach())

    loss = lam_loss + dyn_out["loss"] + cfg.get("act_enc_weight", 1.0) * act_enc_loss

    out = {
        "loss": loss,
        f"{stage}/lam_recon": lam_recon.detach(),
        f"{stage}/lam_vq": lam_vq_loss.detach(),
        f"{stage}/dyn_loss": dyn_out["loss"].detach(),
        f"{stage}/dyn_acc": dyn_out["acc"].detach(),
        f"{stage}/act_enc_loss": act_enc_loss.detach(),
    }
    self.log_dict({k: v for k, v in out.items() if k != "loss"},
                  on_step=True, sync_dist=True)
    return out


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

@hydra.main(version_base=None, config_path="./config", config_name="genie")
def run(cfg):
    assert cfg.phase in ("tokenizer", "joint"), f"unknown phase: {cfg.phase}"

    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    print(f"Loading dataset '{dataset_name}' from "
          f"{'local cache: ' + cache_dir if cache_dir else 'default location'}")

    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg,
    )

    transforms = [get_img_preprocessor("pixels", "pixels", img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(dataset, col, col))

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim("action")
        )

    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, generator=rnd_gen)
    val_cfg = {**cfg.loader, "shuffle": False, "drop_last": False}
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    # In joint phase, load the trained tokenizer.
    if cfg.phase == "joint":
        ckpt_path = cfg.get("tokenizer_ckpt", None)
        if ckpt_path is None:
            raise SystemExit(
                "joint phase requires +tokenizer_ckpt=<path/to/tokenizer.pt>"
            )
        state = torch.load(ckpt_path, map_location="cpu")
        # save_pretrained writes nested dicts; train.py's save_ckpt writes {"module": ...}.
        # accept either.
        if "module" in state:
            tok_state = state["module"]
        elif "model" in state and "tokenizer" in state["model"]:
            tok_state = state["model"]["tokenizer"]
        else:
            tok_state = state
        world_model.tokenizer.load_state_dict(tok_state)
        print(f"[init] loaded tokenizer from {ckpt_path}")

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": max(1, int(0.02 * total_steps)),
                "max_steps": total_steps,
            },
            "interval": "epoch",
        },
    }

    fwd = tokenizer_forward if cfg.phase == "tokenizer" else joint_forward
    data_module = spt.data.DataModule(train=train, val=val)
    module = spt.Module(
        model=world_model,
        forward=partial(fwd, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##  logging / callbacks ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    save_cb = SaveCkptCallback(
        run_name=f"{cfg.output_model_name}_{cfg.phase}",
        cfg=cfg,
        epoch_interval=1,
    )

    ##########################
    ##       training       ##
    ##########################

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[save_cb],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_resume = run_dir / f"{cfg.output_model_name}_{cfg.phase}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        ckpt_path=ckpt_resume if ckpt_resume.exists() else None,
    )
    manager()


if __name__ == "__main__":
    run()
