import os
from pathlib import Path

import importlib
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
    device = next(self.model.parameters()).device
    pixels = batch['pixels']
    # ensure channel-first (B, T, C, H, W)
    if pixels.ndim == 5 and pixels.shape[-1] == 3:
        pixels = pixels.permute(0, 1, 4, 2, 3).contiguous()

    B, T, C, H, W = pixels.shape
    L = cfg.wm.get('history_size', 4)

    next_frame = pixels[:, L].to(device)
    history = pixels[:, :L].reshape(B, C * L, H, W).to(device)

    cond_vec = None
    if 'action' in batch:
        actions = batch['action'][:, :L].to(device)
        if (
            hasattr(self.model, 'action_encoder')
            and self.model.action_encoder is not None
        ):
            act_emb = self.model.action_encoder(actions)
            cond_vec = act_emb.mean(dim=1)
        else:
            cond_vec = actions.reshape(B, -1).float()

    train_batch = {
        'next_frame': next_frame.float(),
        'history': history.float(),
        'cond_vec': cond_vec,
    }

    if 'reward' in batch:
        train_batch['reward'] = (
            batch['reward'][:, L].to(device).float().unsqueeze(-1)
        )
    if 'terminal' in batch:
        train_batch['terminal'] = (
            batch['terminal'][:, L].to(device).float().unsqueeze(-1)
        )

    loss = edm_loss_step(self.model, train_batch, device)

    out = {'loss': loss}
    self.log('train/loss', loss.detach(), on_step=True, sync_dist=True)
    return out


def instantiate_from_config(conf):
    """Instantiate object from a small OmegaConf/dict spec with _target_."""
    if conf is None:
        return None
    if isinstance(conf, dict) or OmegaConf.is_config(conf):
        conf = OmegaConf.to_container(conf, resolve=True)
    assert '_target_' in conf, 'config must contain _target_'
    target = conf['_target_']
    module_name, class_name = target.rsplit('.', 1)
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    # build kwargs excluding _target_
    kwargs = {k: v for k, v in conf.items() if not k.startswith('_')}
    return cls(**kwargs)


def run(config_path: str = None):
    # load config yaml (default path under scripts/train/config/diamond.yaml)
    if config_path is None:
        here = Path(__file__).parent / 'config' / 'diamond.yaml'
        config_path = str(here)
    cfg = OmegaConf.load(config_path)

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

    world_model = instantiate_from_config(cfg.model)

    # instantiate and attach action_encoder from config if present
    action_encoder = None
    if 'action_encoder' in cfg and cfg.action_encoder is not None:
        try:
            action_encoder = instantiate_from_config(cfg.action_encoder)
            world_model.action_encoder = action_encoder
        except Exception:
            action_encoder = None

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
