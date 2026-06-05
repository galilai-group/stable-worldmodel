"""Training script for DreamerV4-style world model.

Two-phase:
  1. (optional) tokenizer fine-tuning - skipped here, we use frozen pre-trained ViT
  2. dynamics training with flow matching + shortcut bootstrap (dynamics_loss)

Usage:
    uv run python scripts/train/dreamer4.py --config-name dreamer4 \
        data.dataset.name=<dataset> \
        wandb.enabled=true \
        trainer.max_epochs=50
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from torch import nn
from torch.utils.data import DataLoader

from stable_pretraining import data as dt
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.dreamer4.module import dynamics_loss
from stable_worldmodel.wm.utils import save_pretrained
from lightning.pytorch.callbacks import Callback


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize   = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class SaveCkptCallback(Callback):
    def __init__(self, run_name, cfg, epoch_interval: int = 5):
        super().__init__()
        self.run_name       = run_name
        self.cfg            = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        if (trainer.current_epoch + 1) % self.epoch_interval == 0:
            self._save(pl_module.model, trainer.current_epoch + 1)
        if (trainer.current_epoch + 1) == trainer.max_epochs:
            self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


def dreamer4_forward(self, batch, stage, cfg):
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    # encode pixels -> packed_z (B, T, 1, d_spatial)
    output   = self.model.encode(batch)
    packed_z = output['packed_z']                  # (B, T, 1, d_spatial)
    actions  = batch.get('action', None)

    # dynamics loss (flow matching + shortcut bootstrap)
    loss, aux = dynamics_loss(
        self.model.dynamics,
        z1=packed_z.detach(),                      # encoder is frozen
        actions=actions,
        k_max=cfg.wm.k_max,
        self_fraction=cfg.wm.self_fraction,
        bootstrap_start=cfg.wm.bootstrap_start,
        global_step=self.global_step,
    )

    output['loss'] = loss
    losses_dict = {f'{stage}/{k}': v for k, v in aux.items()}
    losses_dict[f'{stage}/loss'] = loss.detach()
    self.log_dict(losses_dict, on_step=True, sync_dist=True)

    return output


@hydra.main(version_base=None, config_path='./config', config_name='dreamer4')
def run(cfg):
    dataset_cfg  = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir    = os.environ.get('LOCAL_DATASET_DIR', None)
    logging.info(f'Loading "{dataset_name}" from {"local: " + cache_dir if cache_dir else "default"}')

    dataset       = swm.data.load_dataset(dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg)
    img_processor = get_img_preprocessor('pixels', 'pixels', cfg.img_size)

    extra_transforms = []
    for col in cfg.data.dataset.keys_to_load:
        if col == 'pixels':
            continue
        extra_transforms.append(get_column_normalizer(dataset, col, col))

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col == 'pixels':
                continue
            setattr(cfg.wm, f'{col}_dim', dataset.get_dim(col))
        cfg.model.dynamics.action_dim = cfg.wm.action_dim

    transform = spt.data.transforms.Compose(img_processor, *extra_transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    train = DataLoader(train_set, **cfg.loader, generator=rnd_gen)
    val_cfg = {**cfg.loader, 'shuffle': False, 'drop_last': False}
    val = DataLoader(val_set, **val_cfg)

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * cfg.trainer.max_epochs * len(train))),
                'max_steps': cfg.trainer.max_epochs * len(train),
            },
            'interval': 'epoch',
        }
    }

    data_module  = spt.data.DataModule(train=train, val=val)
    pl_module    = spt.Module(
        model=world_model,
        forward=partial(dreamer4_forward, cfg=cfg),
        optim=optimizers,
    )

    run_id  = cfg.get('subdir') or ''
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[SaveCkptCallback(run_name=cfg.output_model_name, cfg=cfg)],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager   = spt.Manager(
        trainer=trainer,
        module=pl_module,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )
    manager()


if __name__ == '__main__':
    run()
