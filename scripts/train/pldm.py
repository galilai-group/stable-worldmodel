from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from stable_worldmodel.wm.pldm.module import (
    MLP,
    Embedder,
    Predictor,
)
from stable_worldmodel.wm.pldm import PLDM
from stable_worldmodel.wm.loss import PLDMLoss, TemporalStraighteningLoss
from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(
        norm_fn, source=source, target=target
    )
    return normalizer


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(
        self, run_name, cfg, epoch_interval: int = 1
    ):
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
        save_pretrained(model, run_name=self.run_name, config=self.cfg, filename=f'weights_epoch_{epoch}.pt')


def pldm_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""
    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode(batch)

    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']

    inpt_emb = emb[:, : cfg.wm.history_size]  # (B, T-1, D)
    inpt_act = act_emb[:, : cfg.wm.history_size]
    tgt_emb = emb[:, cfg.wm.num_preds :]  # (B, T-1, patches, dim)
    pred_emb = self.model.predict(inpt_emb, inpt_act)

    output['idm_emb'] = torch.cat([emb[:, 1:], emb[:, :-1]], dim=-1)
    output['act_label'] = batch['action'][:, :-1].detach()
    output['act_pred'] = self.idm(output['idm_emb'])
    output['pred_loss'] = (pred_emb - tgt_emb).square().mean()
    output['temp_straight_loss'] = self.path_straight(emb)
    output.update(self.pldm(emb, output['act_pred'], output['act_label']))

    output['loss'] = output['pred_loss']
    for k, v in cfg.loss.items():
        loss_key = f'{k}_loss'
        if not v.enabled or (loss_key not in output):
            continue
        output['loss'] = output['loss'] + v.weight * output[loss_key]

    # log all losses
    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)

    return output


@hydra.main(version_base=None, config_path='./config', config_name='pldm')
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    img_processor = get_img_preprocessor('pixels', 'pixels', cfg.img_size)

    extra_transforms = []
    for col in cfg.data.dataset.keys_to_load:
        if col in ['pixels']:
            continue
        normalizer = get_column_normalizer(dataset, col, col)
        extra_transforms.append(normalizer)

    if hasattr(cfg.data.dataset, 'keys_to_merge'):
        for col in cfg.data.dataset.keys_to_merge:
            normalizer = get_column_normalizer(dataset, col, col)
            extra_transforms.append(normalizer)

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col in ['pixels']:
                continue
            setattr(cfg.wm, f'{col}_dim', dataset.get_dim(col))

    transform = spt.data.transforms.Compose(img_processor, *extra_transforms)

    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train = DataLoader(train_set, **cfg.loader, generator=rnd_gen)
    val_cfg = {**cfg.loader}
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val = DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get('embed_dim', hidden_dim)

    predictor = Predictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=nn.BatchNorm1d,
    )

    idm = MLP(
        input_dim=2 * embed_dim, hidden_dim=512, output_dim=effective_act_dim
    )

    world_model = PLDM(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    models = {
        'model': world_model,
        'idm': idm,
    }

    losses = {
        'pldm': PLDMLoss(),
        'path_straight': TemporalStraighteningLoss(),
    }

    optimizers = {}
    for model_name in models.keys():
        optimizers[f'{model_name}_opt'] = {
            'modules': str(model_name),
            'optimizer': dict(cfg.optimizer),
            'scheduler': {'type': 'LinearWarmupCosineAnnealingLR'},
            'interval': 'epoch',
        }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        **models,
        **losses,
        forward=partial(pldm_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )
    logging.info(f'🫆🫆🫆 Run ID: {run_id} 🫆🫆🫆')

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg, epoch_interval=5
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f'{cfg.output_model_name}_weights.ckpt',
    )

    manager()
    return


if __name__ == '__main__':
    run()
