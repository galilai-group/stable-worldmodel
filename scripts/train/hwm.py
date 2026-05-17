from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from torch import nn
from torch.utils.data import DataLoader

from stable_worldmodel.wm.utils import save_pretrained
from prejepa import (
    VideoPipeline,
    get_column_normalizer,
    get_encoder,
    get_img_preprocessor,
)


class SaveCkptCallback(Callback):
    """Save HWM checkpoints with stable-worldmodel's loadable format."""

    def __init__(self, run_name, cfg, epoch_interval=1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_interval == 0:
            self._save(pl_module.model, epoch)
        if epoch == trainer.max_epochs:
            self._save(pl_module.model, epoch)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


@hydra.main(version_base=None, config_path='./config', config_name='hwm')
def run(cfg):
    encoding_keys = [
        key for key in cfg.wm.get('encoding', {}) if key != cfg.loss.action_key
    ]
    keys_to_load = ['pixels', 'action'] + encoding_keys

    dataset = swm.data.HDF5Dataset(
        cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.low_level.frameskip,
        transform=None,
        cache_dir=cfg.get('cache_dir', None),
        keys_to_load=keys_to_load,
        keys_to_cache=['action'] + encoding_keys,
    )
    sample_stride = cfg.macro_action.get('sample_stride', 1)
    if sample_stride > 1:
        dataset.clip_indices = [
            (ep, start)
            for ep, start in dataset.clip_indices
            if start % sample_stride == 0
        ]
        logging.info(
            f'Using macro-action sample_stride={sample_stride}; '
            f'dataset has {len(dataset)} samples.'
        )

    normalizers = [
        get_column_normalizer(dataset, col, col)
        for col in encoding_keys
    ]

    if cfg.backbone.get('is_video_encoder', False):
        from transformers import AutoVideoProcessor

        processor = AutoVideoProcessor.from_pretrained(cfg.backbone.name)
        transform = spt.data.transforms.Compose(
            VideoPipeline(processor, source='pixels', target='pixels'),
            spt.data.transforms.Resize(
                cfg.image_size, source='pixels', target='pixels'
            ),
            *normalizers,
        )
    else:
        transform = spt.data.transforms.Compose(
            get_img_preprocessor('pixels', 'pixels', cfg.image_size),
            *normalizers,
        )
    dataset.transform = transform

    with open_dict(cfg) as cfg:
        cfg.extra_dims = {}
        for key in cfg.wm.get('encoding', {}):
            if key == cfg.loss.action_key:
                cfg.extra_dims[key] = cfg.macro_action.latent_dim
                continue
            if key not in dataset.column_names:
                raise ValueError(
                    f"Encoding key '{key}' not found in dataset columns."
                )
            dim = dataset.get_dim(key)
            cfg.extra_dims[key] = dim

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, [cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True,
    )

    encoder, embed_dim, num_patches, interp_pos_enc = get_encoder(cfg)
    embed_dim += sum(cfg.wm.get('encoding', {}).values())

    if cfg.backbone.get('is_video_encoder', False):
        num_patches += num_patches * (cfg.n_steps // 4)

    predictor_kwargs = {k: v for k, v in cfg.predictor.items() if k != 'size'}
    predictor = swm.wm.prejepa.CausalPredictor(
        num_patches=num_patches,
        num_frames=cfg.wm.history_size,
        dim=embed_dim,
        **predictor_kwargs,
    )

    extra_encoders = nn.ModuleDict()
    for key, emb_dim in cfg.wm.get('encoding', {}).items():
        if key == cfg.loss.action_key:
            extra_encoders[key] = swm.wm.prejepa.Embedder(
                in_chans=cfg.macro_action.latent_dim, emb_dim=emb_dim
            )
        else:
            extra_encoders[key] = swm.wm.prejepa.Embedder(
                in_chans=cfg.extra_dims[key], emb_dim=emb_dim
            )

    world_model = swm.wm.PreJEPA(
        encoder=spt.backbone.EvalOnly(encoder),
        predictor=predictor,
        extra_encoders=extra_encoders,
        history_size=cfg.wm.history_size,
        num_pred=cfg.wm.num_preds,
        interpolate_pos_encoding=interp_pos_enc,
    )
    action_encoder = swm.wm.hwm.SequenceEncoder(
        output_dim=cfg.macro_action.latent_dim,
        input_dim=cfg.macro_action.low_level_action_dim,
        d_model=cfg.action_encoder.d_model,
        step_mlp=cfg.action_encoder.step_mlp,
        nhead=cfg.action_encoder.nhead,
        num_layers=cfg.action_encoder.num_layers,
        ff_mult=cfg.action_encoder.ff_mult,
        dropout=cfg.action_encoder.dropout,
        use_cls=cfg.action_encoder.use_cls,
        stochastic=cfg.action_encoder.stochastic,
        max_chunk=cfg.macro_action.max_chunk,
        norm_first=cfg.action_encoder.norm_first,
        uniform_input=cfg.action_encoder.uniform_input,
    )

    world_model = spt.Module(
        model=world_model,
        action_encoder=action_encoder,
        forward=partial(swm.wm.hwm.hwm_forward, cfg=cfg),
        optim={
            'model_opt': {
                'modules': '^(model|action_encoder)',
                'optimizer': dict(cfg.optimizer),
            }
        },
    )

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f'Run ID: {run_id}')

    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enable:
        logger = WandbLogger(
            name='hwm_dino_wm',
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            resume='allow' if run_id else None,
            id=run_id or None,
            log_model=False,
        )
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            SaveCkptCallback(
                run_name=cfg.output_model_name,
                cfg=cfg,
                epoch_interval=cfg.save_epoch_interval,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval='step'),
        ],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
        ckpt_path=run_dir / f'{cfg.output_model_name}_weights.ckpt',
    )
    manager()


if __name__ == '__main__':
    run()
