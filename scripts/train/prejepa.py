from collections import OrderedDict
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
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoModelForImageClassification,
    AutoVideoProcessor,
)

# fmt: off
ENCODER_CONFIGS = {
    'resnet': {
        'prefix': 'microsoft/resnet-',
        'model_class': AutoModelForImageClassification,
        'embedding_attr': lambda m: m.config.hidden_sizes[-1],
        'post_init': lambda m: setattr(m.classifier, '1', nn.LayerNorm(m.config.hidden_sizes[-1])),
        'interpolate_pos_encoding': False,
    },
    'vit':    {'prefix': 'google/vit-'},
    'dino':   {'prefix': 'facebook/dino-'},
    'dinov2':  {'prefix': 'facebook/dinov2-'},
    'dinov3':  {'prefix': 'facebook/dinov3-'},
    'webssl':  {'prefix': 'facebook/webssl-'},
    'mae':    {'prefix': 'facebook/vit-mae-'},
    'ijepa':  {'prefix': 'facebook/ijepa'},
    'vjepa2':  {'prefix': 'facebook/vjepa2-vit'},
    'siglip2': {'prefix': 'google/siglip2-'},
}
# fmt: on


def get_encoder(cfg):
    """Load a pretrained vision encoder and return (backbone, embed_dim, num_patches, interp_pos_enc)."""
    encoder_cfg = next(
        (
            c
            for c in ENCODER_CONFIGS.values()
            if cfg.backbone.name.startswith(c['prefix'])
        ),
        None,
    )
    if encoder_cfg is None:
        raise ValueError(f'Unsupported backbone: {cfg.backbone.name}')

    backbone = encoder_cfg.get('model_class', AutoModel).from_pretrained(
        cfg.backbone.name
    )
    if hasattr(backbone, 'vision_model'):  # CLIP-style
        backbone = backbone.vision_model
    if 'post_init' in encoder_cfg:
        encoder_cfg['post_init'](backbone)

    embed_dim = encoder_cfg.get(
        'embedding_attr', lambda m: m.config.hidden_size
    )(backbone)
    is_cnn = cfg.backbone.name.startswith('microsoft/resnet-')
    num_patches = 1 if is_cnn else (cfg.image_size // cfg.patch_size) ** 2
    interp_pos_enc = encoder_cfg.get('interpolate_pos_encoding', True)

    return backbone, embed_dim, num_patches, interp_pos_enc


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def get_img_preprocessor(source, target, img_size=224):
    stats = spt.data.dataset_stats.ImageNet
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(**stats, source=source, target=target),
        spt.data.transforms.Resize(img_size, source=source, target=target),
    )


def get_column_normalizer(dataset, source, target):
    data = torch.from_numpy(dataset.get_col_data(source)[:])
    data = data[~torch.isnan(data).any(dim=1)]
    mean, std = (
        data.mean(0, keepdim=True).clone(),
        data.std(0, keepdim=True).clone(),
    )
    return spt.data.transforms.WrapTorchTransform(
        lambda x: ((x - mean) / std).float(),
        source=source,
        target=target,
    )


class VideoPipeline(spt.data.transforms.Transform):
    def __init__(self, processor, source='image', target='image'):
        super().__init__()
        self.processor, self.source, self.target = processor, source, target

    def __call__(self, x):
        frames = self.nested_get(x, self.source)
        self.nested_set(
            x,
            self.processor(frames, return_tensors='pt')[
                'pixel_values_videos'
            ].squeeze(0),
            self.target,
        )
        return x


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class ModelObjectCallBack(Callback):
    """Save the model object periodically and at the final epoch."""

    def __init__(self, dirpath, filename='model_object', epoch_interval=1):
        super().__init__()
        self.dirpath, self.filename, self.epoch_interval = (
            Path(dirpath),
            filename,
            epoch_interval,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_interval == 0:
            path = self.dirpath / f'{self.filename}_epoch_{epoch}_object.ckpt'
            torch.save(pl_module.model, path)
            logging.info(f'Saved world model to {path}')
        if epoch == trainer.max_epochs:
            path = self.dirpath / f'{self.filename}_object.ckpt'
            torch.save(pl_module.model, path)
            logging.info(f'Saved final world model to {path}')


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def _strip_action_dims(tensor, action_range):
    """Remove the action dimensions from the last axis."""
    return torch.cat(
        [tensor[..., : action_range[0]], tensor[..., action_range[1] :]],
        dim=-1,
    )


def dinowm_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute losses."""
    for key in self.model.extra_encoders:
        batch[key] = torch.nan_to_num(batch[key], 0.0).squeeze()

    batch = self.model.encode(
        batch,
        target='embed',
        is_video=cfg.backbone.get('is_video_encoder', False),
    )

    embedding = batch['embed'][:, : cfg.wm.history_size, ...]
    pred_embedding = self.model.predict(embedding)
    target_embedding = batch['embed'][:, cfg.wm.num_preds :, ...].detach()

    # Per-modality losses
    pixels_dim = batch['pixels_embed'].size(-1)
    batch['pixels_loss'] = F.mse_loss(
        pred_embedding[..., :pixels_dim], target_embedding[..., :pixels_dim]
    )

    start, action_range = pixels_dim, [0, 0]
    for key in self.model.extra_encoders:
        dim = batch[f'{key}_embed'].size(-1)
        lo, hi = start, start + dim
        if key == 'action':
            action_range = [lo, hi]
        else:
            batch[f'{key}_loss'] = F.mse_loss(
                pred_embedding[..., lo:hi],
                target_embedding[..., lo:hi].detach(),
            )
        start = hi

    # Actionless embeddings (for probes and total loss)
    batch['actionless_embed'] = _strip_action_dims(
        batch['embed'], action_range
    )
    batch['actionless_prev_embed'] = _strip_action_dims(
        embedding, action_range
    )
    batch['actionless_pred_embed'] = _strip_action_dims(
        pred_embedding, action_range
    )
    batch['actionless_target_embed'] = _strip_action_dims(
        target_embedding, action_range
    )

    batch['loss'] = F.mse_loss(
        batch['actionless_pred_embed'],
        batch['actionless_target_embed'].detach(),
    )

    if batch['loss'].isnan():
        raise ValueError('NaN loss encountered!')

    self.log_dict(
        {f'{stage}/{k}': v.detach() for k, v in batch.items() if '_loss' in k},
        on_step=True,
        sync_dist=True,
    )
    return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path='./config', config_name='prejepa')
def run(cfg):
    # --- Dataset ---
    encoding_keys = list(cfg.wm.get('encoding', {}).keys())
    keys_to_load = ['pixels'] + encoding_keys

    dataset = swm.data.HDF5Dataset(
        cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        cache_dir=cfg.get('cache_dir', None),
        keys_to_load=keys_to_load,
        keys_to_cache=encoding_keys,
    )

    normalizers = [
        get_column_normalizer(dataset, col, col)
        for col in cfg.wm.get('encoding', {})
    ]

    if cfg.backbone.get('is_video_encoder', False):
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
            if key not in dataset.column_names:
                raise ValueError(
                    f"Encoding key '{key}' not found in dataset columns."
                )
            dim = dataset.get_dim(key)
            cfg.extra_dims[key] = (
                dim if key != 'action' else dim * cfg.frameskip
            )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, [cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # --- Model ---
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

    extra_encoders = nn.ModuleDict(
        OrderedDict(
            (
                key,
                swm.wm.prejepa.Embedder(
                    in_chans=cfg.extra_dims[key], emb_dim=emb_dim
                ),
            )
            for key, emb_dim in cfg.wm.get('encoding', {}).items()
        )
    )

    world_model = swm.wm.PreJEPA(
        encoder=spt.backbone.EvalOnly(encoder),
        predictor=predictor,
        extra_encoders=extra_encoders,
        history_size=cfg.wm.history_size,
        num_pred=cfg.wm.num_preds,
        interpolate_pos_encoding=interp_pos_enc,
    )

    world_model = spt.Module(
        model=world_model,
        forward=partial(dinowm_forward, cfg=cfg),
        optim={
            'model_opt': {'modules': 'model', 'optimizer': dict(cfg.optimizer)}
        },
    )

    # --- Training ---
    run_id = cfg.get('subdir') or ''
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f'Run ID: {run_id}')

    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enable:
        logger = WandbLogger(
            name='dino_wm',
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
            spt.callbacks.CPUOffloadCallback(),
            ModelObjectCallBack(
                dirpath=run_dir,
                filename=cfg.output_model_name,
                epoch_interval=5,
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
