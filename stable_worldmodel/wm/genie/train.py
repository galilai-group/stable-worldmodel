"""
train.py — Genie training in two phases, with PushT data.

If `cfg.data_path` doesn't exist, this collects a PushT dataset using swm's
built-in WeakPolicy. Then trains in two phases:

Phase 1: train the video tokenizer (ST-ViViT) alone.
Phase 2: train the LAM and dynamics model jointly, tokenizer frozen.

Usage:
    python train.py --phase tokenizer
    python train.py --phase joint --tokenizer-ckpt checkpoints/tokenizer_final.pt
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

import stable_worldmodel as swm
from stable_worldmodel.envs.pusht import WeakPolicy

# Adjust import paths to match your layout
from stable_worldmodel.wm.genie.st_vivit import ST_ViViT
from stable_worldmodel.wm.genie.lam import LAM
from stable_worldmodel.wm.genie.st_maskgit import ST_MaskGIT, ST_Transformer
from stable_worldmodel.wm.genie.vq import VectorQuantizer
from stable_worldmodel.wm.genie.genie import Genie


# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Data — PushT
    data_path: str = "data/pusht_demo.lance"
    num_episodes: int = 2_000
    num_envs: int = 8
    collect_seed: int = 0
    dist_constraint: int = 100

    height: int = 128
    width: int = 128
    channels: int = 3
    temporal_dim: int = 16
    batch_size: int = 16
    num_workers: int = 4

    # Architecture
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 12
    tokenizer_patch_size: int = 4
    lam_patch_size: int = 16
    tokenizer_codes: int = 1024
    lam_codes: int = 8
    vq_embed_dim: int = 32

    # Optimization
    lr_tokenizer: float = 3e-4
    lr_joint: float = 3e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 5_000

    # Step budgets
    tokenizer_steps: int = 300_000
    joint_steps: int = 500_000

    # Bookkeeping
    dead_code_reinit_every: int = 500
    log_every: int = 100
    save_every: int = 5_000
    val_every: int = 1_000
    val_samples: int = 4
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    ckpt_dir: str = "checkpoints"
    device: str = "cuda"
    dtype: str = "bfloat16"


# ────────────────────────────────────────────────────────────────────────────
# Data: collect-if-missing, then load
# ────────────────────────────────────────────────────────────────────────────

def ensure_pusht_dataset(cfg: TrainConfig) -> None:
    """Collect a PushT dataset to cfg.data_path if it doesn't already exist."""
    path = Path(cfg.data_path)
    if path.exists():
        print(f"[data] using existing dataset at {path}")
        return

    print(f"[data] {path} not found — collecting {cfg.num_episodes} episodes...")
    path.parent.mkdir(parents=True, exist_ok=True)

    world = swm.World(
        "swm/PushT-v1",
        num_envs=cfg.num_envs,
        image_shape=(cfg.height, cfg.width),
        render_mode="rgb_array",
    )
    world.set_policy(WeakPolicy(dist_constraint=cfg.dist_constraint))
    world.collect(str(path), episodes=cfg.num_episodes, seed=cfg.collect_seed)
    print(f"[data] collection complete → {path}")


class PushTVideoDataset(Dataset):
    """
    Wraps swm.data.load_dataset to yield (T, H, W, c) float32 tensors in [-1, 1].
    The LAM learns latent actions from video; PushT's ground-truth actions are
    not used here.
    """
    # swm's per-sample image key — adjust if your version differs.
    IMAGE_KEY = "observation.image"

    def __init__(self, path: str, num_steps: int):
        self.inner = swm.data.load_dataset(path, num_steps=num_steps)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> Tensor:
        sample = self.inner[idx]
        video = sample[self.IMAGE_KEY]                           # (T, H, W, c)

        if not isinstance(video, torch.Tensor):
            video = torch.as_tensor(video)
        if video.dtype == torch.uint8:
            video = video.float() / 127.5 - 1.0                  # [0, 255] → [-1, 1]
        elif video.dtype.is_floating_point and video.max() > 1.5:
            video = video / 127.5 - 1.0
        else:
            video = video.float() * 2.0 - 1.0                    # [0, 1] → [-1, 1]
        return video.contiguous()


# ────────────────────────────────────────────────────────────────────────────
# MaskGIT masking
# ────────────────────────────────────────────────────────────────────────────

def maskgit_mask(tokens: Tensor, mask_token_id: int) -> tuple[Tensor, Tensor]:
    """
    Per-example p ~ Uniform(0.5, 1.0), Bernoulli mask the last T-1 frames.
    Frame 0 is never masked.
    """
    B, T, S = tokens.shape
    labels = tokens
    input_ids = tokens.clone()

    p = 0.5 + 0.5 * torch.rand(B, device=tokens.device)
    mask = torch.rand(B, T - 1, S, device=tokens.device) < p[:, None, None]
    input_ids[:, 1:][mask] = mask_token_id
    return input_ids, labels


# ────────────────────────────────────────────────────────────────────────────
# Model construction
# ────────────────────────────────────────────────────────────────────────────

def build_genie(cfg: TrainConfig) -> Genie:
    def make_st() -> ST_Transformer:
        return ST_Transformer(
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            d_model=cfg.d_model,
            temporal_dim=cfg.temporal_dim,
        )

    tokenizer = ST_ViViT(
        encoder=make_st(),
        decoder=make_st(),
        vq=VectorQuantizer(num_codes=cfg.tokenizer_codes, embed_dim=cfg.vq_embed_dim),
        patch_size=cfg.tokenizer_patch_size,
        height=cfg.height, width=cfg.width, channels=cfg.channels,
        temporal_dim=cfg.temporal_dim, d_model=cfg.d_model,
    )

    lam = LAM(
        encoder=make_st(),
        decoder=make_st(),
        vq=VectorQuantizer(num_codes=cfg.lam_codes, embed_dim=cfg.vq_embed_dim),
        patch_size=cfg.lam_patch_size,
        height=cfg.height, width=cfg.width, channels=cfg.channels,
        temporal_dim=cfg.temporal_dim, d_model=cfg.d_model,
    )

    spatial_tokens = (cfg.height // cfg.tokenizer_patch_size) * (cfg.width // cfg.tokenizer_patch_size)
    dynamics = ST_MaskGIT(
        decoder=make_st(),
        spatial_dim=spatial_tokens,
        temporal_dim=cfg.temporal_dim,
        d_model=cfg.d_model,
        image_vocab_size=cfg.tokenizer_codes,
        action_embed_dim=cfg.vq_embed_dim,
    )

    return Genie(tokenizer=tokenizer, lam=lam, dynamics=dynamics)


# ────────────────────────────────────────────────────────────────────────────
# Schedules and checkpointing
# ────────────────────────────────────────────────────────────────────────────

def cosine_with_warmup(step: int, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def save_ckpt(module: nn.Module, optimizer, step: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"module": module.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "step": step}, path)


def load_ckpt(module: nn.Module, optimizer, path: Path) -> int:
    state = torch.load(path, map_location="cpu")
    module.load_state_dict(state["module"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state.get("step", 0)


@torch.no_grad()
def _tokenizer_pre_vq(tok: ST_ViViT, video: Tensor) -> Tensor:
    B, T = video.size(0), video.size(1)
    patches = tok.patchify(rearrange(video, "B T H W c -> (B T) c H W"))
    patches = rearrange(patches, "(B T) C h w -> B T (h w) C", B=B, T=T)
    return tok.pre_vq_proj(tok.encoder(patches + tok.enc_pos_embed_TSC))


@torch.no_grad()
def _lam_pre_vq(lam: LAM, video: Tensor) -> Tensor:
    B, T = video.size(0), video.size(1)
    patches = lam.enc_patchify(rearrange(video, "B T H W c -> (B T) c H W"))
    patches = rearrange(patches, "(B T) C h w -> B T (h w) C", B=B, T=T)
    z = lam.encoder(patches + lam.enc_pos_embed_TSC).mean(dim=2)[:, 1:, :]
    return lam.pre_vq_proj(z)


# ────────────────────────────────────────────────────────────────────────────
# Validation + wandb helpers
# ────────────────────────────────────────────────────────────────────────────

def _video_to_uint8_grid(video: Tensor) -> np.ndarray:
    """(B, T, H, W, c) float in [-1, 1] -> (B*H, T*W, 3) uint8 image grid."""
    v = video.detach().clamp(-1, 1).add(1).mul(127.5).byte().cpu().numpy()
    B, T, H, W, c = v.shape
    return v.transpose(0, 2, 1, 3, 4).reshape(B * H, T * W, c)


def _stack_recon_grid(gt: Tensor, recon: Tensor) -> np.ndarray:
    g = _video_to_uint8_grid(gt)
    r = _video_to_uint8_grid(recon)
    sep = np.full((4, g.shape[1], 3), 255, dtype=np.uint8)
    return np.concatenate([g, sep, r], axis=0)


@torch.no_grad()
def validate_tokenizer(tok: nn.Module, val_loader: DataLoader, cfg: TrainConfig) -> dict:
    tok.eval()
    device = torch.device(cfg.device)
    total_recon, total_vq, n = 0.0, 0.0, 0
    grid = None
    for video in val_loader:
        video = video.to(device)
        video_hat, _, vq_loss = tok(video)
        recon_loss = F.mse_loss(video_hat, video)
        total_recon += recon_loss.item() * video.size(0)
        total_vq += vq_loss.item() * video.size(0)
        n += video.size(0)
        if grid is None:
            k = min(cfg.val_samples, video.size(0))
            grid = _stack_recon_grid(video[:k], video_hat[:k])
    tok.train()
    return {"val/recon": total_recon / max(1, n), "val/vq": total_vq / max(1, n), "_grid": grid}


@torch.no_grad()
def validate_joint(genie: Genie, val_loader: DataLoader, cfg: TrainConfig) -> dict:
    tok, lam, dyn = genie.tokenizer, genie.lam, genie.dynamics
    tok.eval(); lam.eval(); dyn.eval()
    device = torch.device(cfg.device)
    total_lam_recon, total_lam_vq, total_dyn, total_acc, n = 0.0, 0.0, 0.0, 0.0, 0
    grid = None
    for video in val_loader:
        video = video.to(device)
        _, action_q_TE, lam_vq_loss = lam.extract_actions(video)
        next_pred = lam.predict_next_frames(video[:, :-1], action_q_TE)
        lam_recon = F.mse_loss(next_pred, video[:, 1:])
        tokens, _, _ = tok.encode(video)
        input_ids, labels = maskgit_mask(tokens, dyn.mask_token_id)
        dyn_out = dyn(input_ids, labels, actions_T=action_q_TE)
        total_lam_recon += lam_recon.item() * video.size(0)
        total_lam_vq += lam_vq_loss.item() * video.size(0)
        total_dyn += dyn_out["loss"].item() * video.size(0)
        total_acc += dyn_out["acc"].item() * video.size(0)
        n += video.size(0)
        if grid is None:
            k = min(cfg.val_samples, video[:, :-1].size(0))
            grid = _stack_recon_grid(video[:k, 1:], next_pred[:k])
    lam.train(); dyn.train()
    return {
        "val/lam_recon": total_lam_recon / max(1, n),
        "val/lam_vq": total_lam_vq / max(1, n),
        "val/dyn_loss": total_dyn / max(1, n),
        "val/dyn_acc": total_acc / max(1, n),
        "_grid": grid,
    }


def _log_wandb(run, payload: dict, step: int) -> None:
    if run is None:
        return
    import wandb
    log = {k: v for k, v in payload.items() if not k.startswith("_")}
    if "_grid" in payload and payload["_grid"] is not None:
        log["val/reconstruction"] = wandb.Image(payload["_grid"], caption=f"step {step} (top: gt, bottom: recon)")
    run.log(log, step=step)


# ────────────────────────────────────────────────────────────────────────────
# Phase 1: tokenizer
# ────────────────────────────────────────────────────────────────────────────

def train_tokenizer(
    genie: Genie,
    loader: DataLoader,
    cfg: TrainConfig,
    start_step: int = 0,
    val_loader: DataLoader | None = None,
    wandb_run=None,
):
    tok = genie.tokenizer
    tok.train()
    device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype)
    ckpt_dir = Path(cfg.ckpt_dir)

    opt = torch.optim.AdamW(tok.parameters(), lr=cfg.lr_tokenizer, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_with_warmup(s, cfg.warmup_steps, cfg.tokenizer_steps))

    step = start_step
    while step < cfg.tokenizer_steps:
        for video in loader:
            video = video.to(device, non_blocking=(device.type != "mps"))

            with torch.autocast(device_type=device.type, dtype=dtype):
                video_hat, _, vq_loss = tok(video)
                recon_loss = F.mse_loss(video_hat, video)
                loss = recon_loss + vq_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tok.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()

            if step % cfg.log_every == 0:
                print(f"[tokenizer] step {step:>7d} | recon {recon_loss.item():.4f} "
                      f"| vq {vq_loss.item():.4f} | lr {sched.get_last_lr()[0]:.2e}")
                _log_wandb(wandb_run, {
                    "train/recon": recon_loss.item(),
                    "train/vq": vq_loss.item(),
                    "train/lr": sched.get_last_lr()[0],
                }, step)

            if step and step % cfg.dead_code_reinit_every == 0:
                n = tok.vq.reinit_dead_codes(_tokenizer_pre_vq(tok, video))
                if n:
                    print(f"[tokenizer] step {step:>7d} | reinit'd {n} dead codes")
                    _log_wandb(wandb_run, {"train/dead_codes_reinit": n}, step)

            if val_loader is not None and step and step % cfg.val_every == 0:
                val = validate_tokenizer(tok, val_loader, cfg)
                print(f"[tokenizer] step {step:>7d} | "
                      f"val_recon {val['val/recon']:.4f} | val_vq {val['val/vq']:.4f}")
                _log_wandb(wandb_run, val, step)

            if step and step % cfg.save_every == 0:
                save_ckpt(tok, opt, step, ckpt_dir / f"tokenizer_step{step}.pt")

            step += 1
            if step >= cfg.tokenizer_steps:
                break

    if val_loader is not None:
        val = validate_tokenizer(tok, val_loader, cfg)
        print(f"[tokenizer] step {step:>7d} | final val_recon {val['val/recon']:.4f} | val_vq {val['val/vq']:.4f}")
        _log_wandb(wandb_run, val, step)
    save_ckpt(tok, opt, step, ckpt_dir / "tokenizer_final.pt")


# ────────────────────────────────────────────────────────────────────────────
# Phase 2: LAM + dynamics, tokenizer frozen
# ────────────────────────────────────────────────────────────────────────────

def train_joint(
    genie: Genie,
    loader: DataLoader,
    cfg: TrainConfig,
    start_step: int = 0,
    val_loader: DataLoader | None = None,
    wandb_run=None,
):
    tok, lam, dyn = genie.tokenizer, genie.lam, genie.dynamics

    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    lam.train()
    dyn.train()

    device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype)
    ckpt_dir = Path(cfg.ckpt_dir)

    params = list(lam.parameters()) + list(dyn.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr_joint, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_with_warmup(s, cfg.warmup_steps, cfg.joint_steps))

    step = start_step
    while step < cfg.joint_steps:
        for video in loader:
            video = video.to(device, non_blocking=(device.type != "mps"))

            with torch.autocast(device_type=device.type, dtype=dtype):
                _, action_q_TE, lam_vq_loss = lam.extract_actions(video)
                next_pred = lam.predict_next_frames(video[:, :-1], action_q_TE)
                lam_recon = F.mse_loss(next_pred, video[:, 1:])
                lam_loss = lam_recon + lam_vq_loss

                with torch.no_grad():
                    tokens, _, _ = tok.encode(video)

                input_ids, labels = maskgit_mask(tokens, dyn.mask_token_id)

                dyn_out = dyn(input_ids, labels, actions_T=action_q_TE.detach())

                loss = lam_loss + dyn_out["loss"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            sched.step()

            if step % cfg.log_every == 0:
                print(f"[joint] step {step:>7d} | "
                      f"lam_recon {lam_recon.item():.4f} | lam_vq {lam_vq_loss.item():.4f} | "
                      f"dyn {dyn_out['loss'].item():.4f} | acc {dyn_out['acc'].item():.3f} | "
                      f"lr {sched.get_last_lr()[0]:.2e}")
                _log_wandb(wandb_run, {
                    "train/lam_recon": lam_recon.item(),
                    "train/lam_vq": lam_vq_loss.item(),
                    "train/dyn_loss": dyn_out["loss"].item(),
                    "train/dyn_acc": dyn_out["acc"].item(),
                    "train/lr": sched.get_last_lr()[0],
                }, step)

            if step and step % cfg.dead_code_reinit_every == 0:
                n = lam.vq.reinit_dead_codes(_lam_pre_vq(lam, video))
                if n:
                    print(f"[joint] step {step:>7d} | reinit'd {n} LAM codes")
                    _log_wandb(wandb_run, {"train/lam_dead_codes_reinit": n}, step)

            if val_loader is not None and step and step % cfg.val_every == 0:
                val = validate_joint(genie, val_loader, cfg)
                print(f"[joint] step {step:>7d} | "
                      f"val_lam_recon {val['val/lam_recon']:.4f} | val_dyn {val['val/dyn_loss']:.4f} "
                      f"| val_acc {val['val/dyn_acc']:.3f}")
                _log_wandb(wandb_run, val, step)

            if step and step % cfg.save_every == 0:
                save_ckpt(genie, opt, step, ckpt_dir / f"joint_step{step}.pt")

            step += 1
            if step >= cfg.joint_steps:
                break

    if val_loader is not None:
        val = validate_joint(genie, val_loader, cfg)
        print(f"[joint] step {step:>7d} | final val_lam_recon {val['val/lam_recon']:.4f} "
              f"| val_dyn {val['val/dyn_loss']:.4f} | val_acc {val['val/dyn_acc']:.3f}")
        _log_wandb(wandb_run, val, step)
    save_ckpt(genie, opt, step, ckpt_dir / "joint_final.pt")


# ────────────────────────────────────────────────────────────────────────────
# Entry
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["tokenizer", "joint"], default="tokenizer")
    parser.add_argument("--tokenizer-ckpt", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.config:
        for k, v in json.loads(Path(args.config).read_text()).items():
            setattr(cfg, k, v)

    # 1. Make sure PushT data exists, collecting it if not
    ensure_pusht_dataset(cfg)

    # 2. Build dataset / loader
    dataset = PushTVideoDataset(str(Path(cfg.data_path).absolute()), num_steps=cfg.temporal_dim)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    # 3. Build model
    genie = build_genie(cfg).to(cfg.device)

    # 4. Train
    start_step = 0
    if args.phase == "tokenizer":
        if args.resume:
            start_step = load_ckpt(genie.tokenizer, None, Path(args.resume))
        train_tokenizer(genie, loader, cfg, start_step=start_step)

    elif args.phase == "joint":
        if args.tokenizer_ckpt is None:
            raise SystemExit("--tokenizer-ckpt is required for phase 'joint'")
        load_ckpt(genie.tokenizer, None, Path(args.tokenizer_ckpt))
        if args.resume:
            start_step = load_ckpt(genie, None, Path(args.resume))
        train_joint(genie, loader, cfg, start_step=start_step)


if __name__ == "__main__":
    main()
