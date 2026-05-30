import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from functools import partial
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.utils import save_pretrained
from stable_worldmodel.wm.diamond.edm_train import edm_loss_step


def get_img_preprocessor(source: str, target: str, img_size: int = 64):
    to_image = dt.transforms.ToImage(source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class SaveCkptCallback(pl.callbacks.Callback):
    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
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


def diamond_forward(self, batch, stage, cfg):
    """Forward used by the training harness to compute EDM loss + optional reward/term."""
    # batch['pixels']: (B, T, C, H, W) or (B, T, H, W, C) depending on dataset
    pixels = batch['pixels']
    # ensure channel-first
    if pixels.ndim == 5 and pixels.shape[-1] == 3:
        pixels = pixels.permute(0, 1, 4, 2, 3).contiguous()

    B, T, C, H, W = pixels.shape
    L = cfg.wm.get('history_size', 4)

    # construct next_frame and history
    next_frame = pixels[:, L].to(next(self.model.parameters()).device)
    history = (
        pixels[:, :L]
        .reshape(B, C * L, H, W)
        .to(next(self.model.parameters()).device)
    )

    cond_vec = None
    if 'action' in batch:
        # simple action encoding: flatten first L actions
        actions = (
            batch['action'][:, :L]
            .reshape(B, -1)
            .to(next(self.model.parameters()).device)
        )
        cond_vec = actions

    train_batch = {
        'next_frame': next_frame.float(),
        'history': history.float(),
        'cond_vec': cond_vec,
    }

    device = next(self.model.parameters()).device
    loss = edm_loss_step(self.model, train_batch, device)

    out = {'loss': loss}
    self.log('train/loss', loss.detach(), on_step=True, sync_dist=True)
    return out


@hydra.main(version_base=None, config_path='./config', config_name='diamond')
def run(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )

    transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels'):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, generator=rnd_gen
    )
    val_cfg = {**cfg.loader}
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'interval': 'epoch',
        }
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_module = spt.Module(
        model=world_model,
        forward=partial(diamond_forward, cfg=cfg),
        optim=optimizers,
    )

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg, epoch_interval=1
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer, module=world_module, data=data_module
    )
    manager()


if __name__ == '__main__':
    run()
