import os
from functools import partial
from pathlib import Path
import random

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from torch.nn import functional as F
from torch.utils.data import DataLoader

os.environ['STABLEWM_HOME'] = '/sonic_home/luizfacury/stable-worldmodel/eval'
os.environ['MUJOCO_GL'] = 'egl'

from stable_worldmodel.wm.tdmpc2 import (
    TDMPC2,
    two_hot,
    two_hot_inv,
    log_std,
    gaussian_logprob,
    squash,
)


class ModelObjectCallBack(Callback):
    """
    PyTorch Lightning callback to periodically save the entire model object to disk.
    """

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
        if epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs:
            path = self.dirpath / f'{self.filename}_epoch_{epoch}_object.ckpt'
            torch.save(pl_module.model, path)
            logging.info(f'Saved world model to {path}')


def tdmpc2_forward(self, batch, stage, cfg):
    """
    Executes the forward pass and loss computation for the TD-MPC2 world model and policy.

    This function dynamically extracts encoded modalities, unrolls the latent dynamics over a
    fixed horizon, and computes the consistency, reward, and Q-value losses. It also performs
    the actor policy update using a conservative Q-value estimate to prevent out-of-distribution actions.

    Args:
        batch (dict): Dictionary containing environment observations, actions, and rewards.
        stage (str): The current execution stage (e.g., 'train', 'validate').
        cfg (DictConfig): Configuration object containing model and training hyperparameters.

    Returns:
        dict: The batch dictionary updated with the computed total loss under the 'loss' key.
    """
    encoding_keys = list(cfg.wm.get('encoding', {}).keys())
    B, T_plus_1 = batch['action'].shape[:2]

    flat_obs_dict = {}
    flat_goal_dict = {}

    for key in encoding_keys:
        obs = batch[key]
        goal = batch[f'goal_{key}']  # GoalDataset maps 'state' to 'goal_state'

        # Align Goal sequence length with Obs sequence
        if goal.ndim == obs.ndim and goal.shape[1] == 1:
            goal = goal.expand(-1, T_plus_1, *([-1] * (goal.ndim - 2)))
        elif goal.ndim == obs.ndim - 1:
            goal = goal.unsqueeze(1).expand(
                -1, T_plus_1, *([-1] * (goal.ndim - 1))
            )

        flat_obs_dict[key] = obs.reshape(-1, *obs.shape[2:])
        flat_goal_dict[key] = goal.reshape(-1, *goal.shape[2:])

    all_z = self.model.encode(flat_obs_dict, flat_goal_dict).reshape(
        B, T_plus_1, -1
    )

    z = all_z[:, 0]
    target_zs = all_z[:, 1:]

    loss_consistency, loss_reward, loss_value, loss_pi = 0, 0, 0, 0
    discount = cfg.wm.get('discount', 0.99)
    entropy_coef = cfg.wm.get('entropy_coef', 1e-4)

    for t in range(cfg.wm.horizon):
        action = batch['action'][:, t]
        reward = batch['reward'][:, t]

        next_z_pred, reward_pred = self.model.forward(z, action)

        loss_consistency += F.mse_loss(
            next_z_pred, target_zs[:, t].detach()
        ) * (cfg.wm.rho**t)
        target_reward = two_hot(reward, cfg)
        loss_reward += -(
            target_reward * F.log_softmax(reward_pred, dim=-1)
        ).sum(-1).mean() * (cfg.wm.rho**t)

        with torch.no_grad():
            next_z_for_q = next_z_pred.detach()
            mean_raw, log_std_raw = self.model.pi(next_z_for_q).chunk(
                2, dim=-1
            )
            log_std_bounded = log_std(log_std_raw, low=-10, dif=12)
            eps = torch.randn_like(mean_raw)
            next_action_pred = torch.tanh(
                mean_raw + eps * log_std_bounded.exp()
            )

            next_z_a = torch.cat([next_z_for_q, next_action_pred], dim=-1)
            q_indices = random.sample(range(cfg.wm.num_q), 2)
            next_qs = [
                two_hot_inv(self.model.target_qs[i](next_z_a), cfg)
                for i in q_indices
            ]
            next_q_min = torch.min(next_qs[0], next_qs[1])
            target_q = reward.unsqueeze(1) + discount * next_q_min
            target_q_two_hot = two_hot(target_q, cfg)

        z_a = torch.cat([z, action], dim=-1)
        for q in self.model.qs:
            loss_value += -(
                target_q_two_hot * F.log_softmax(q(z_a), dim=-1)
            ).sum(-1).mean() * (cfg.wm.rho**t)

        # Actor Update
        z_detached = z.detach()
        mean_raw, log_std_raw = self.model.pi(z_detached).chunk(2, dim=-1)
        log_std_bounded = log_std(log_std_raw, low=-10, dif=12)
        eps = torch.randn_like(mean_raw)
        log_prob = gaussian_logprob(eps, log_std_bounded)
        action_pi_raw = mean_raw + eps * log_std_bounded.exp()
        mu, action_pi, log_prob = squash(mean_raw, action_pi_raw, log_prob)

        scaled_log_prob = log_prob * cfg.action_dim
        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        scaled_entropy = -log_prob * entropy_scale

        z_pi = torch.cat([z_detached, action_pi], dim=-1)
        self.model.qs.requires_grad_(False)
        qs_pi = torch.stack(
            [two_hot_inv(q(z_pi), cfg) for q in self.model.qs], dim=0
        )

        q_indices = random.sample(range(cfg.wm.num_q), 2)
        q_pi_min = torch.min(qs_pi[q_indices[0]], qs_pi[q_indices[1]])
        self.model.qs.requires_grad_(True)

        if t == 0:
            self.model.scale.update(q_pi_min)
        q_pi_normalized = self.model.scale(q_pi_min)

        step_pi_loss = -(entropy_coef * scaled_entropy + q_pi_normalized)
        loss_pi += step_pi_loss.mean() * (cfg.wm.rho**t)

        z = next_z_pred

    loss_consistency /= cfg.wm.horizon
    loss_reward /= cfg.wm.horizon
    loss_value /= cfg.wm.horizon * cfg.wm.num_q
    loss_pi /= cfg.wm.horizon

    total_loss = (
        cfg.wm.consistency_coef * loss_consistency
        + cfg.wm.reward_coef * loss_reward
        + cfg.wm.value_coef * loss_value
        + loss_pi
    )

    self.log_dict(
        {
            f'{stage}/loss': total_loss,
            f'{stage}/consist': loss_consistency,
            f'{stage}/reward': loss_reward,
            f'{stage}/value': loss_value,
            f'{stage}/policy': loss_pi,
        },
        on_step=True,
        sync_dist=True,
        prog_bar=True,
    )

    if stage == 'train':
        for q, t_q in zip(self.model.qs, self.model.target_qs):
            for p, p_t in zip(q.parameters(), t_q.parameters()):
                p_t.data.lerp_(p.data, cfg.wm.tau)

    batch['loss'] = total_loss
    return batch


def get_column_normalizer(dataset, source, target):
    """
    Creates a Z-score normalization transform for a specific dataset column.

    Args:
        dataset: The target dataset used to compute the global mean and standard deviation.
        source (str): The key of the input column to read.
        target (str): The key where the normalized output will be stored.

    Returns:
        Transform: A callable transform that applies the computed normalization to the data.
    """
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
    """
    Creates a standardized image preprocessing pipeline.
    Applies ImageNet statistical normalization and resizes the images to the requested dimensions.

    Args:
        source (str): The dictionary key of the raw image input.
        target (str): The dictionary key to store the processed image output.
        img_size (int): The target height and width for the resizing operation.

    Returns:
        Transform: A composed image transformation pipeline.
    """
    stats = spt.data.dataset_stats.ImageNet
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(**stats, source=source, target=target),
        spt.data.transforms.Resize(img_size, source=source, target=target),
    )


class ClampActionTransform(spt.data.transforms.Transform):
    """
    Data transform that hard-clips all continuous action arrays to the bounds [-1.0, 1.0].
    Ensures safe bounded outputs for the actor's Tanh layers.
    """

    def __call__(self, batch):
        if 'action' in batch:
            batch['action'] = torch.clamp(batch['action'], -1.0, 1.0)
        return batch


@hydra.main(version_base=None, config_path='./config', config_name='tdmpc2')
def run(cfg):
    """
    Main training entry point for the TD-MPC2 model.

    Dynamically structures the dataset and applies transforms/normalizers based on the available
    modalities defined in the configuration (e.g., pixels, state arrays). Initializes the TD-MPC2
    architecture and orchestrates the PyTorch Lightning training loop.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    torch.set_float32_matmul_precision('high')

    encoding_keys = list(cfg.wm.get('encoding', {}).keys())
    if not encoding_keys:
        raise ValueError('No encoding modalities defined in cfg.wm.encoding!')

    use_pixels = 'pixels' in encoding_keys
    extra_keys = [k for k in encoding_keys if k != 'pixels']

    keys_to_load = encoding_keys + ['action', 'reward']

    base_dataset = swm.data.HDF5Dataset(
        cfg.dataset_name,
        num_steps=cfg.wm.horizon + 1,
        keys_to_load=keys_to_load,
        cache_dir=os.environ.get('STABLEWM_HOME', None),
    )

    with open_dict(cfg):
        cfg.action_dim = base_dataset.get_dim('action')
        cfg.extra_dims = {'action': cfg.action_dim}

        for key in extra_keys:
            cfg.extra_dims[key] = base_dataset.get_dim(key)

    transforms = [ClampActionTransform()]

    if use_pixels:
        transforms.append(
            get_img_preprocessor('pixels', 'pixels', cfg.image_size)
        )

    for key in extra_keys:
        transforms.append(get_column_normalizer(base_dataset, key, key))

    base_dataset.transform = spt.data.transforms.Compose(*transforms)

    goal_keys = {key: f'goal_{key}' for key in encoding_keys}
    if use_pixels:
        goal_keys['pixels'] = 'goal_pixels'

    dataset = swm.data.GoalDataset(
        dataset=base_dataset,
        goal_probabilities=(0.0, 1.0, 0.0, 0.0),
        current_goal_offset=cfg.wm.horizon + 1,
        goal_keys=goal_keys,
        seed=cfg.seed,
    )

    train_set, val_set = spt.data.random_split(
        dataset, [cfg.train_split, 1 - cfg.train_split]
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )

    model = TDMPC2(cfg)

    def add_opt(module_regex, lr, eps=1e-8):
        opt_cfg = dict(cfg.optimizer)
        opt_cfg['lr'] = lr
        opt_cfg['eps'] = eps
        return {'modules': module_regex, 'optimizer': opt_cfg}

    module = spt.Module(
        model=model,
        forward=partial(tdmpc2_forward, cfg=cfg),
        hparams=OmegaConf.to_container(cfg, resolve=True),
        optim={
            'wm_opt': add_opt(
                r'model\.(cnn|pixel_encoder|extra_encoders|dynamics|reward|qs).*',
                cfg.optimizer.lr,
            ),
            'pi_opt': add_opt(r'model\.pi.*', cfg.optimizer.lr, eps=1e-5),
        },
    )

    subdir = cfg.subdir if cfg.subdir is not None else 'tdmpc2_run'
    run_dir = Path(swm.data.utils.get_cache_dir(), subdir)
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            ModelObjectCallBack(
                dirpath=run_dir, filename=cfg.output_model_name
            )
        ],
    )
    spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
    )()


if __name__ == '__main__':
    run()
