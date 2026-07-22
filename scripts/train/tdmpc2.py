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
from torch.utils.data import DataLoader
import numpy as np

from stable_worldmodel.wm.tdmpc2 import tdmpc2_forward
from stable_worldmodel.wm.utils import save_pretrained


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained.

    Writes a state-dict ``weights_epoch_<n>.pt`` plus a ``config.json`` under
    ``$STABLEWM_HOME/checkpoints/<run_name>/`` (the format consumed by
    ``swm.wm.utils.load_pretrained``), replacing the old pickled
    ``*_object.ckpt`` dump. ``cfg`` is the instantiable model config
    (``cfg.model``), so ``config.json`` alone can rebuild the model.
    """

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


def get_column_normalizer(dataset, source, target):
    """Z-score normalization transform computed from the full dataset column."""
    data = torch.from_numpy(dataset.get_col_data(source)[:])
    data = data[~torch.isnan(data).any(dim=1)]
    mean, std = (
        data.mean(0, keepdim=True).clone(),
        data.std(0, keepdim=True).clone(),
    )
    mean, std = mean.squeeze(), std.squeeze() + 1e-2

    def norm_fn(x):
        return ((x - mean.to(x.device)) / std.to(x.device)).float()

    return spt.data.transforms.WrapTorchTransform(
        norm_fn, source=source, target=target
    )


def get_img_preprocessor(source, target, img_size=64):
    """ImageNet-normalized + resized image preprocessing pipeline."""
    stats = spt.data.dataset_stats.ImageNet
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(**stats, source=source, target=target),
        spt.data.transforms.Resize(img_size, source=source, target=target),
    )


@hydra.main(version_base=None, config_path='./config', config_name='tdmpc2')
def run(cfg):
    """
    Main training entry point for the TD-MPC2 model.

    Uses dataset rewards directly.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    torch.set_float32_matmul_precision('high')

    model_cfg = cfg.model.cfg  # the config TDMPC2 is instantiated from
    encoding_keys = list(model_cfg.wm.get('encoding', {}).keys())
    if not encoding_keys:
        raise ValueError(
            'No encoding modalities defined in cfg.model.cfg.wm.encoding!'
        )

    use_pixels = 'pixels' in encoding_keys
    goal_obs_key = cfg.get(
        'goal_obs_key'
    )  # if set, concatenate episode goal into this key
    extra_keys = [k for k in encoding_keys if k != 'pixels']

    keys_to_load = list(encoding_keys) + ['action', 'reward']

    base_dataset = swm.data.load_dataset(
        cfg.dataset_name,
        cache_dir=cfg.get('cache_dir'),
        num_steps=model_cfg.wm.horizon + 1,
        keys_to_load=keys_to_load,
        keys_to_cache=keys_to_load if cfg.get('cache_dataset', True) else [],
    )

    if goal_obs_key is not None:
        if goal_obs_key not in encoding_keys:
            raise ValueError(
                f'cfg.goal_obs_key="{goal_obs_key}" must be one of the encoding keys {encoding_keys}.'
            )
        _raw_obs = base_dataset.get_col_data(goal_obs_key)[:]
        _ep_off = (
            base_dataset.get_col_data('ep_offset')[:].flatten().astype(int)
        )
        _ep_len = base_dataset.get_col_data('ep_len')[:].flatten().astype(int)
        _goal_idx = np.clip(_ep_off + _ep_len - 1, 0, len(_raw_obs) - 1)
        goals_by_step = np.empty_like(_raw_obs)
        for _ep, (_off, _len) in enumerate(
            zip(_ep_off.tolist(), _ep_len.tolist())
        ):
            goals_by_step[_off : _off + _len] = _raw_obs[_goal_idx[_ep]]
        base_dataset._cache[goal_obs_key] = np.concatenate(
            [_raw_obs, goals_by_step], axis=-1
        )
        logging.info(
            f'Goal augmentation: appended last obs of each episode to "{goal_obs_key}" '
            f'(dim {_raw_obs.shape[-1]} → {base_dataset._cache[goal_obs_key].shape[-1]})'
        )

    raw_actions = base_dataset.get_col_data('action')[:]
    valid_actions = raw_actions[~np.isnan(raw_actions).any(axis=1)]
    act_max = valid_actions.max()
    act_min = valid_actions.min()

    if act_max > 1.01 or act_min < -1.01:
        logging.error(
            f'Dataset actions fall outside the [-1, 1] range! (Min: {act_min:.2f}, Max: {act_max:.2f}).\n'
            'TD-MPC2 uses a Tanh actor and strictly requires actions to be bounded between [-1, 1].\n'
            'Please normalize your dataset actions.'
        )
        raise ValueError(
            'Unnormalized actions detected in the dataset. Training aborted.'
        )

    with open_dict(cfg):
        model_cfg.action_dim = base_dataset.get_dim('action')
        model_cfg.extra_dims = {'action': model_cfg.action_dim}

        for key in extra_keys:
            if goal_obs_key is not None and key == goal_obs_key:
                model_cfg.extra_dims[key] = base_dataset._cache[key].shape[-1]
            else:
                model_cfg.extra_dims[key] = base_dataset.get_dim(key)

    transforms = []
    if use_pixels:
        transforms.append(
            get_img_preprocessor('pixels', 'pixels', model_cfg.image_size)
        )

    for key in extra_keys:
        if goal_obs_key is not None and key == goal_obs_key:
            aug_data = torch.from_numpy(base_dataset._cache[key]).float()
            aug_clean = aug_data[~torch.isnan(aug_data).any(dim=1)]
            _mean = aug_clean.mean(0).clone()
            _std = aug_clean.std(0).clone() + 1e-2
            transforms.append(
                spt.data.transforms.WrapTorchTransform(
                    lambda x, m=_mean, s=_std: (
                        (x - m.to(x.device)) / s.to(x.device)
                    ).float(),
                    source=key,
                    target=key,
                )
            )
        else:
            transforms.append(get_column_normalizer(base_dataset, key, key))

    base_dataset.transform = spt.data.transforms.Compose(*transforms)

    train_set, val_set = spt.data.random_split(
        base_dataset, [cfg.train_split, 1 - cfg.train_split]
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    model = hydra.utils.instantiate(cfg.model)

    def add_opt(module_regex, lr, eps=1e-8):
        opt_cfg = dict(cfg.optimizer)
        opt_cfg['lr'] = lr
        opt_cfg['eps'] = eps
        return {'modules': module_regex, 'optimizer': opt_cfg}

    module = spt.Module(
        model=model,
        forward=partial(tdmpc2_forward, cfg=model_cfg),
        hparams=OmegaConf.to_container(cfg, resolve=True),
        optim={
            'enc_opt': add_opt(
                r'model\.(cnn|pixel_encoder|extra_encoders|sim_norm).*',
                cfg.optimizer.lr * cfg.get('enc_lr_scale', 0.3),
            ),
            'wm_opt': add_opt(
                r'model\.(dynamics|reward|qs).*',
                cfg.optimizer.lr,
            ),
            'pi_opt': add_opt(
                r'model\.pi.*', cfg.optimizer.lr * 0.1, eps=1e-5
            ),
        },
    )
    subdir = cfg.subdir
    run_dir = Path(swm.data.utils.get_cache_dir(), subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enable:
        logger = WandbLogger(
            name=f'{model_cfg.wm.name}_{cfg.dataset_name}_{subdir}',
            project=cfg.wandb.project,
            resume='allow' if subdir else None,
            id=subdir or None,
            log_model=False,
        )
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=[
            SaveCkptCallback(run_name=cfg.output_model_name, cfg=cfg.model)
        ],
    )
    spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
    )()


if __name__ == '__main__':
    run()
