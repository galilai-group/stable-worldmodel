import os
from collections import OrderedDict
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from einops import rearrange, repeat
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModel

import stable_worldmodel as swm


# ============================================================================
# Data Setup
# ============================================================================
def get_data(cfg, goal_probabilities):
    """Setup dataset with image transforms and normalization."""

    def get_img_pipeline(key, target, img_size=224):
        return spt.data.transforms.Compose(
            spt.data.transforms.ToImage(
                **spt.data.dataset_stats.ImageNet,
                source=key,
                target=target,
            ),
            spt.data.transforms.Resize(img_size, source=key, target=target),
        )

    def get_column_normalizer(dataset, source: str, target: str):
        """Get normalizer for a specific column in the dataset."""
        data = torch.from_numpy(dataset.get_col_data(source)[:])
        data = data[~torch.isnan(data).any(dim=1)]
        mean = data.mean(0, keepdim=True).clone()
        std = data.std(0, keepdim=True).clone()

        def norm_fn(x):
            return ((x - mean) / std).float()

        normalizer = spt.data.transforms.WrapTorchTransform(
            norm_fn, source=source, target=target
        )

        return normalizer

    cache_dir = None
    if not hasattr(cfg, 'local_cache_dir'):
        cache_dir = os.environ.get('SLURM_TMPDIR', None)

    dataset = swm.data.HDF5Dataset(
        cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        cache_dir=cache_dir,
        keys_to_load=['pixels', 'action', 'proprio'],
        keys_to_cache=['action', 'proprio'],
    )

    norm_action_transform = get_column_normalizer(dataset, 'action', 'action')
    norm_proprio_transform = get_column_normalizer(
        dataset, 'proprio', 'proprio'
    )

    # Apply transforms to all steps and goal observations
    transform = spt.data.transforms.Compose(
        get_img_pipeline('pixels', 'pixels', cfg.image_size),
        norm_action_transform,
        norm_proprio_transform,
    )

    dataset.transform = transform

    dataset = swm.data.GoalDataset(
        dataset=dataset,
        goal_probabilities=goal_probabilities,
        gamma=cfg.goal_gamma,
        current_goal_offset=cfg.dinowm.history_size,
        goal_keys={'pixels': 'goal_pixels', 'proprio': 'goal_proprio'},
        seed=cfg.seed,
    )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    logging.info(f'Train: {len(train_set)}, Val: {len(val_set)}')

    train = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )
    val = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return spt.data.DataModule(train=train, val=val)


# ============================================================================
# Model Architecture
# ============================================================================
def get_gciql_critics_model(cfg):
    """Build goal-conditioned IQL model for training value and Q functions."""

    expectile_loss = swm.wm.gcrl.ExpectileLoss(tau=cfg.get('expectile', 0.9))
    gamma = cfg.get('discount', 0.99)

    def forward(self, batch, stage):
        """Forward: encode observations and goals, predict actions, compute losses."""

        proprio_key = 'proprio' if 'proprio' in batch else None

        # Replace NaN values with 0 (occurs at sequence boundaries)
        if proprio_key is not None:
            batch[proprio_key] = torch.nan_to_num(batch[proprio_key], 0.0)
        batch['action'] = torch.nan_to_num(batch['action'], 0.0)
        # Encode all timesteps into latent embeddings
        batch = self.model.encode(
            batch,
            target='embed',
            pixels_key='pixels',
        )

        # Encode goal into latent embedding
        batch = self.model.encode(
            batch,
            target='goal_embed',
            pixels_key='goal_pixels',
            prefix='goal_',
        )

        # Detach encoder outputs if encoder is frozen
        if not encoder_trainable:
            batch['embed'] = batch['embed'].detach()
            batch['goal_embed'] = batch['goal_embed'].detach()
            batch['pixels_embed'] = batch['pixels_embed'].detach()
            batch['pixels_goal_embed'] = batch['pixels_goal_embed'].detach()

        # NaN detection after encoding
        nan_checks = {
            'proprio': batch.get('proprio'),
            'goal_proprio': batch.get('goal_proprio'),
            'proprio_embed': batch.get('proprio_embed'),
            'proprio_goal_embed': batch.get('proprio_goal_embed'),
            'pixels_embed': batch.get('pixels_embed'),
            'pixels_goal_embed': batch.get('pixels_goal_embed'),
            'embed': batch.get('embed'),
            'goal_embed': batch.get('goal_embed'),
        }
        for name, tensor in nan_checks.items():
            if tensor is not None and torch.isnan(tensor).any():
                logging.warning(
                    f'NaN detected in {name}! '
                    f'count={torch.isnan(tensor).sum().item()}, '
                    f'shape={tensor.shape}'
                )

        # Use history to predict values
        embedding = batch['embed'][
            :, : cfg.dinowm.history_size, :, :
        ]  # (B, T, patches, dim)
        target_embedding = batch['embed'][
            :, cfg.dinowm.td_offset :, :, :
        ]  # (B, T, patches, dim)
        goal_embedding = batch['goal_embed']  # (B, 1, patches, dim)

        # Reshape to (B, T*P, dim) for the predictor
        embedding_flat = rearrange(embedding, 'b t p d -> b (t p) d')
        goal_embedding_flat = rearrange(goal_embedding, 'b t p d -> b (t p) d')
        target_embedding_flat = rearrange(
            target_embedding, 'b t p d -> b (t p) d'
        )

        # Compute masks and rewards
        with torch.no_grad():
            # Compare raw data instead of embeddings (embeddings differ due to GPU non-determinism)
            obs_pixels = batch['pixels'][
                :, : cfg.dinowm.history_size
            ]  # (B, T, C, H, W)
            goal_pixels_repeated = repeat(
                batch['goal_pixels'],
                'b 1 c h w -> b t c h w',
                t=obs_pixels.shape[1],
            )
            obs_proprio = batch['proprio'][
                :, : cfg.dinowm.history_size
            ]  # (B, T, D)
            goal_proprio_repeated = repeat(
                batch['goal_proprio'], 'b 1 d -> b t d', t=obs_proprio.shape[1]
            )
            pixels_match = (obs_pixels == goal_pixels_repeated).all(
                dim=(2, 3, 4)
            )  # (B, T)
            proprio_match = (obs_proprio == goal_proprio_repeated).all(
                dim=2
            )  # (B, T)
            eq_mask = pixels_match & proprio_match  # (B, T)
            # masks are 1 if non-terminal, 0 if terminal (at goal)
            masks = (~eq_mask).float().unsqueeze(-1)
            # rewards are -1 if non-terminal, 0 if terminal
            reward = -masks

        # Get actions for Q function input
        actions = batch['action'][
            :, : cfg.dinowm.history_size
        ]  # (B, T, action_dim)

        # ===== Value Loss (IQL style) =====
        # Get Q(s, a, g) from target critic (teacher network)
        with torch.no_grad():
            (q1, q2) = self.model.critic_predictor.forward_teacher(
                embedding_flat, actions, goal_embedding_flat
            )
            q = torch.minimum(q1, q2)

        # Get V(s, g) from value network (student, trainable)
        v = self.model.value_predictor.forward_student(
            embedding_flat, goal_embedding_flat
        )

        # Compute IQL value loss with expectile regression
        # expectile_loss(preds, targets, adv) computes: weight(adv) * (targets - preds)^2
        # For IQL: we want weight(q - v) * (q - v)^2, so v as preds, q as targets
        value_loss = expectile_loss(v, q.detach())

        # ===== Critic Loss (IQL style) =====
        # Get V(s', g) for next observations from target value network
        with torch.no_grad():
            next_v = self.model.value_predictor.forward_teacher(
                target_embedding_flat, goal_embedding_flat
            )
            # Compute Q target using V for bootstrapping: Q(s,a,g) = r + γ * V(s',g)
            q_target = reward + gamma * masks * next_v

        # Get Q(s, a, g) predictions from critic (student network)
        (q1_pred, q2_pred) = self.model.critic_predictor.forward_student(
            embedding_flat, actions, goal_embedding_flat
        )
        critic_loss = (
            (q1_pred - q_target) ** 2 + (q2_pred - q_target) ** 2
        ).mean()

        # ===== Total Loss =====
        total_loss = value_loss + critic_loss

        batch['value_loss'] = value_loss
        batch['critic_loss'] = critic_loss
        batch['loss'] = total_loss

        # ====== Debug Logging =====

        # For logging compatibility
        value_pred = v
        value_target = q
        adv = q - v

        # NaN detection after value prediction
        if torch.isnan(v).any():
            logging.warning(
                f'NaN in value_pred! count={torch.isnan(v).sum().item()}'
            )
        if torch.isnan(q).any():
            logging.warning(
                f'NaN in q target! count={torch.isnan(q).sum().item()}'
            )

        # NaN detection after loss computation
        if torch.isnan(value_loss):
            logging.warning(
                f'NaN in value_loss! '
                f'value_pred range: [{value_pred.min().item():.4f}, {value_pred.max().item():.4f}], '
                f'value_target range: [{value_target.min().item():.4f}, {value_target.max().item():.4f}]'
            )

        # Log all losses
        prefix = 'train/' if self.training else 'val/'
        losses_dict = {
            f'{prefix}{k}': v.detach()
            for k, v in batch.items()
            if '_loss' in k
        }
        losses_dict[f'{prefix}loss'] = batch['loss'].detach()
        self.log_dict(losses_dict, on_step=True, sync_dist=True)

        # Log diagnostics for collapse detection
        with torch.no_grad():
            # Keep for logging
            goal_embedding_repeated = repeat(
                goal_embedding, 'b 1 p d -> b t p d', t=embedding.shape[1]
            )
            td_error = value_target - value_pred
            embed_dist = (
                (embedding - goal_embedding_repeated)
                .pow(2)
                .sum(dim=(-1, -2))
                .sqrt()
            )

            # Check per-sample pixel AND proprio match
            # Compare last frame of history window (what the value network sees)
            last_hist_idx = cfg.dinowm.history_size - 1
            last_frame_pixels = batch['pixels'][
                :, last_hist_idx
            ]  # (B, C, H, W)
            goal_pixels_squeezed = batch['goal_pixels'][:, 0]  # (B, C, H, W)
            last_frame_proprio = batch['proprio'][:, last_hist_idx]  # (B, D)
            goal_proprio_squeezed = batch['goal_proprio'][:, 0]  # (B, D)

            # Compare last frame of target (last frame of clip) with goal
            # This is what value_target is computed from
            last_target_pixels = batch['pixels'][:, -1]  # (B, C, H, W)
            last_target_proprio = batch['proprio'][:, -1]  # (B, D)

            # Per-sample checks (last frame of history)
            pixels_match_per_sample = (
                last_frame_pixels == goal_pixels_squeezed
            ).all(dim=(1, 2, 3))  # (B,)
            proprio_match_per_sample = (
                last_frame_proprio == goal_proprio_squeezed
            ).all(dim=1)  # (B,)
            both_match = pixels_match_per_sample & proprio_match_per_sample

            # Per-sample checks (last frame of target/clip)
            target_pixels_match = (
                last_target_pixels == goal_pixels_squeezed
            ).all(dim=(1, 2, 3))  # (B,)
            target_proprio_match = (
                last_target_proprio == goal_proprio_squeezed
            ).all(dim=1)  # (B,)
            target_both_match = target_pixels_match & target_proprio_match

            # Check embedding components separately
            # batch['pixels_embed'] and batch['pixels_goal_embed'] are pixel-only embeddings
            # Compare last frame of history window (what the value network sees)
            pixels_embed_obs = batch['pixels_embed'][
                :, last_hist_idx
            ]  # (B, P, D_pixels)
            pixels_embed_goal = batch['pixels_goal_embed'][
                :, 0
            ]  # (B, P, D_pixels)
            has_proprio_embed = 'proprio_embed' in batch
            if has_proprio_embed:
                proprio_embed_obs = batch['proprio_embed'][
                    :, last_hist_idx
                ]  # (B, D_proprio)
                proprio_embed_goal = batch['proprio_goal_embed'][
                    :, 0
                ]  # (B, D_proprio)

            if both_match.any():
                pixel_embed_diff = (
                    (
                        pixels_embed_obs[both_match]
                        - pixels_embed_goal[both_match]
                    )
                    .abs()
                    .max()
                )
                if has_proprio_embed:
                    proprio_embed_diff = (
                        (
                            proprio_embed_obs[both_match]
                            - proprio_embed_goal[both_match]
                        )
                        .abs()
                        .max()
                    )
                else:
                    proprio_embed_diff = torch.tensor(
                        -1.0, device=embedding.device
                    )
                # Log value prediction specifically when goal matches current state
                # value_pred has shape (B, T, 1), take last timestep
                value_pred_at_goal = value_pred[:, -1, 0][both_match].mean()
            else:
                pixel_embed_diff = torch.tensor(-1.0, device=embedding.device)
                proprio_embed_diff = torch.tensor(
                    -1.0, device=embedding.device
                )
                value_pred_at_goal = torch.tensor(
                    float('nan'), device=embedding.device
                )

            # Log value target when goal matches last frame of target/clip
            # When goal matches target, value_target should be close to 0
            if target_both_match.any():
                # value_target has shape (B, T, 1), take last timestep
                value_target_at_goal = value_target[:, -1, 0][
                    target_both_match
                ].mean()
            else:
                value_target_at_goal = torch.tensor(
                    float('nan'), device=embedding.device
                )

            self.log_dict(
                {
                    f'{prefix}debug_pixels_match_rate': pixels_match_per_sample.float().mean(),
                    f'{prefix}debug_proprio_match_rate': proprio_match_per_sample.float().mean(),
                    f'{prefix}debug_both_match_rate': both_match.float().mean(),
                    f'{prefix}debug_target_match_rate': target_both_match.float().mean(),
                    f'{prefix}debug_pixel_embed_diff': pixel_embed_diff,
                    f'{prefix}debug_proprio_embed_diff': proprio_embed_diff,
                    f'{prefix}debug_value_pred_at_goal': value_pred_at_goal,
                    f'{prefix}debug_value_target_at_goal': value_target_at_goal,
                },
                on_step=True,
                sync_dist=True,
            )

            collapse_diagnostics = {
                # Value prediction stats - std ≈ 0 indicates collapse
                f'{prefix}value_pred_mean': value_pred.mean(),
                f'{prefix}value_pred_std': value_pred.std(),
                f'{prefix}value_pred_min': value_pred.min(),
                f'{prefix}value_pred_max': value_pred.max(),
                # Value target stats
                f'{prefix}value_target_mean': value_target.mean(),
                f'{prefix}value_target_std': value_target.std(),
                # Advantage stats
                f'{prefix}adv_mean': adv.mean(),
                f'{prefix}adv_std': adv.std(),
                f'{prefix}adv_min': adv.min(),
                f'{prefix}adv_max': adv.max(),
                f'{prefix}accept_prob': (adv >= 0).float().mean(),
                # Reward stats - mean ≈ -1 means reward too sparse
                f'{prefix}reward_mean': reward.mean(),
                f'{prefix}goal_match_rate': eq_mask.float().mean(),
                # TD error stats - std ≈ 0 indicates no learning signal
                f'{prefix}td_error_mean': td_error.mean(),
                f'{prefix}td_error_std': td_error.std(),
                # Embedding distance to goal
                f'{prefix}embed_goal_dist_mean': embed_dist.mean(),
                f'{prefix}embed_goal_dist_std': embed_dist.std(),
            }
            self.log_dict(collapse_diagnostics, on_step=True, sync_dist=True)

        return batch

    # Load encoder based on config
    encoder_type = cfg.get('encoder_type', 'dino')
    if encoder_type == 'dino':
        # Load frozen DINO encoder
        encoder = AutoModel.from_pretrained('facebook/dinov2-small')
        embedding_dim = encoder.config.hidden_size
        encoder_trainable = False
        logging.info('Using pretrained frozen DINO encoder')
    elif encoder_type == 'vit_tiny':
        # Load trainable ViT tiny from scratch
        encoder = spt.backbone.utils.vit_hf(
            'tiny',
            patch_size=cfg.patch_size,
            image_size=cfg.image_size,
            pretrained=False,
            use_mask_token=False,
        )
        embedding_dim = encoder.config.hidden_size
        encoder_trainable = True
        logging.info('Using trainable ViT tiny encoder (from scratch)')
    else:
        raise ValueError(f'Unknown encoder_type: {encoder_type}')

    # Calculate actual number of patches based on the actual image size used by DINO
    assert cfg.image_size % cfg.patch_size == 0, (
        'Image size must be multiple of patch size'
    )
    num_patches = (cfg.image_size // cfg.patch_size) ** 2
    if cfg.dinowm.get('use_proprio_encoder', True):
        embedding_dim += cfg.dinowm.proprio_embed_dim  # Total embedding size

    logging.info(f'Patches: {num_patches}, Embedding dim: {embedding_dim}')

    # Build causal predictor (transformer that predicts next actions)
    effective_act_dim = (
        cfg.frameskip * cfg.dinowm.action_dim
    )  # NOTE: 'frameskip' > 1 is used to predict action chunks
    action_predictor = swm.wm.gcrl.Predictor(
        num_patches=num_patches,
        num_frames=cfg.dinowm.history_size,
        dim=embedding_dim,
        out_dim=effective_act_dim,
        **cfg.predictor,
    )

    # Single value predictor for V(s, g) - expectile regression target
    value_predictor = swm.wm.gcrl.Predictor(
        num_patches=num_patches,
        num_frames=cfg.dinowm.history_size,
        dim=embedding_dim,
        out_dim=1,
        non_positive_output=False,
        **cfg.predictor,
    )

    wrapped_value_predictor = spt.TeacherStudentWrapper(
        value_predictor,
        warm_init=True,
        base_ema_coefficient=cfg.get('value_ema_tau', 0.995),
        final_ema_coefficient=cfg.get('value_ema_tau', 0.995),
    )

    # Double Q predictor for Q(s, a, g) - TD learning with action input
    critic_predictor = swm.wm.gcrl.DoublePredictorWrapper(
        swm.wm.gcrl.QPredictor,
        num_patches=num_patches,
        num_frames=cfg.dinowm.history_size,
        dim=embedding_dim,
        action_dim=effective_act_dim,
        non_positive_output=False,
        **cfg.predictor,
    )

    wrapped_critic_predictor = spt.TeacherStudentWrapper(
        critic_predictor,
        warm_init=True,
        base_ema_coefficient=cfg.get('value_ema_tau', 0.995),
        final_ema_coefficient=cfg.get('value_ema_tau', 0.995),
    )

    # Build proprioception encoder (optional)
    extra_encoders = None
    if cfg.dinowm.get('use_proprio_encoder', True):
        extra_encoders = OrderedDict()
        extra_encoders['proprio'] = swm.wm.gcrl.Embedder(
            in_chans=cfg.dinowm.proprio_dim,
            emb_dim=cfg.dinowm.proprio_embed_dim,
        )
        extra_encoders = torch.nn.ModuleDict(extra_encoders)

    logging.info(
        f'Action dim: {effective_act_dim}, Proprio encoder: {extra_encoders is not None}'
    )

    # Assemble policy
    # Wrap encoder in EvalOnly if frozen (DINO), otherwise keep trainable (ViT tiny)
    wrapped_encoder = (
        spt.backbone.EvalOnly(encoder) if not encoder_trainable else encoder
    )
    gciql_model = swm.wm.gcrl.GCRL(
        encoder=wrapped_encoder,
        action_predictor=action_predictor,
        value_predictor=wrapped_value_predictor,
        critic_predictor=wrapped_critic_predictor,
        extra_encoders=extra_encoders,
        history_size=cfg.dinowm.history_size,
    )

    # Wrap in stable_spt Module with separate optimizers for each component
    def add_opt(module_name, lr):
        return {
            'modules': str(module_name),
            'optimizer': {'type': 'AdamW', 'lr': lr},
        }

    optim_config = {
        'value_predictor_opt': add_opt(
            'model.value_predictor', cfg.predictor_lr
        ),
        'critic_predictor_opt': add_opt(
            'model.critic_predictor', cfg.predictor_lr
        ),
    }

    # Add proprio encoder optimizer if enabled
    if extra_encoders is not None:
        optim_config['proprio_opt'] = add_opt(
            'model.extra_encoders.proprio', cfg.proprio_encoder_lr
        )

    # Add encoder optimizer if trainable (ViT tiny)
    if encoder_trainable:
        optim_config['encoder_opt'] = add_opt(
            'model.encoder', cfg.get('encoder_lr', 3e-4)
        )

    gciql_critics_model = spt.Module(
        model=gciql_model,
        forward=forward,
        optim=optim_config,
    )
    return gciql_critics_model


def get_gciql_actor_model(cfg, trained_critics_model):
    """Build goal-conditioned IQL model for extracting policy via AWR from trained value and Q functions."""

    def forward(self, batch, stage):
        """Forward: encode observations and goals, predict actions, compute losses."""

        proprio_key = 'proprio' if 'proprio' in batch else None

        # Replace NaN values with 0 (occurs at sequence boundaries)
        if proprio_key is not None:
            batch[proprio_key] = torch.nan_to_num(batch[proprio_key], 0.0)
        batch['action'] = torch.nan_to_num(batch['action'], 0.0)

        with torch.no_grad():
            # Encode all timesteps into latent embeddings
            batch = self.model.encode(
                batch,
                target='embed',
                pixels_key='pixels',
            )

            # Encode goal into latent embedding
            batch = self.model.encode(
                batch,
                target='goal_embed',
                pixels_key='goal_pixels',
                emb_keys=['proprio']
                if 'proprio' in self.model.extra_encoders
                else [],
                prefix='goal_',
            )

            # Use history to predict next actions
            embedding = batch['embed'][
                :, : cfg.dinowm.history_size, :, :
            ]  # (B, T, patches, dim)
            target_embedding = batch['embed'][
                :, cfg.dinowm.td_offset :, :, :
            ]  # (B, T, patches, dim)
            goal_embedding = batch['goal_embed']  # (B, 1, patches, dim)

            # Reshape to (B, T*P, dim) for the predictor
            embedding_flat = rearrange(embedding, 'b t p d -> b (t p) d')
            goal_embedding_flat = rearrange(
                goal_embedding, 'b t p d -> b (t p) d'
            )
            target_embedding_flat = rearrange(
                target_embedding, 'b t p d -> b (t p) d'
            )
            # Single value predictor returns V(s, g)
            value = self.model.value_predictor(
                embedding_flat, goal_embedding_flat
            )

            next_value = self.model.value_predictor(
                target_embedding_flat, goal_embedding_flat
            )

            # Advantage is next_value - current_value
            advantage = next_value - value  # (B, T, 1)

        action_pred, action_stds = self.model.predict_actions(
            embedding.detach(), goal_embedding.detach()
        )

        # Policy is extracted via AWR (Advantage Weighted Regression)
        alpha = cfg.get('awr_alpha', 3.0)
        exp_adv = torch.exp(advantage.detach() * alpha)
        exp_adv = torch.minimum(
            exp_adv, torch.tensor(100.0, device=exp_adv.device)
        )

        # Compute negative log-likelihood loss for Gaussian policy
        # -log N(a | μ, σ) = log(σ) + 0.5 * ((a - μ) / σ)²
        target_actions = batch['action'][:, : cfg.dinowm.history_size]
        log_stds = torch.clamp(
            self.model.log_stds, self.model.log_std_min, self.model.log_std_max
        )
        var = torch.exp(2 * log_stds)
        raw_nll_loss = (
            log_stds + 0.5 * ((target_actions - action_pred) ** 2) / var
        )

        action_loss = exp_adv * raw_nll_loss
        action_loss = action_loss.mean()
        batch['awr_loss'] = action_loss
        batch['loss'] = action_loss

        # Log all losses
        prefix = 'train/' if self.training else 'val/'
        losses_dict = {
            f'{prefix}{k}': v.detach()
            for k, v in batch.items()
            if '_loss' in k
        }
        losses_dict[f'{prefix}actor_loss'] = batch['loss'].detach()

        # Debug logging for exploding loss investigation
        losses_dict[f'{prefix}debug/value_mean'] = value.mean().detach()
        losses_dict[f'{prefix}debug/value_min'] = value.min().detach()
        losses_dict[f'{prefix}debug/value_max'] = value.max().detach()
        losses_dict[f'{prefix}debug/next_value_mean'] = (
            next_value.mean().detach()
        )
        losses_dict[f'{prefix}debug/next_value_min'] = (
            next_value.min().detach()
        )
        losses_dict[f'{prefix}debug/next_value_max'] = (
            next_value.max().detach()
        )
        losses_dict[f'{prefix}debug/advantage_mean'] = (
            advantage.mean().detach()
        )
        losses_dict[f'{prefix}debug/advantage_min'] = advantage.min().detach()
        losses_dict[f'{prefix}debug/advantage_max'] = advantage.max().detach()
        losses_dict[f'{prefix}debug/exp_adv_mean'] = exp_adv.mean().detach()
        losses_dict[f'{prefix}debug/exp_adv_min'] = exp_adv.min().detach()
        losses_dict[f'{prefix}debug/exp_adv_max'] = exp_adv.max().detach()
        losses_dict[f'{prefix}debug/raw_nll_mean'] = (
            raw_nll_loss.mean().detach()
        )
        losses_dict[f'{prefix}debug/raw_nll_max'] = raw_nll_loss.max().detach()
        losses_dict[f'{prefix}debug/action_pred_mean'] = (
            action_pred.mean().detach()
        )
        losses_dict[f'{prefix}debug/action_pred_std'] = (
            action_pred.std().detach()
        )
        losses_dict[f'{prefix}debug/action_log_stds'] = (
            log_stds.mean().detach()
        )
        losses_dict[f'{prefix}debug/action_stds'] = action_stds.mean().detach()

        self.log_dict(
            losses_dict, on_step=True, sync_dist=True
        )  # , on_epoch=True, sync_dist=True)

        return batch

    # Assemble policy
    for encoder in trained_critics_model.model.extra_encoders.values():
        encoder.eval()
    # Get underlying encoder (handle both EvalOnly wrapper and raw encoder cases)
    encoder_backbone = (
        trained_critics_model.model.encoder.backbone
        if hasattr(trained_critics_model.model.encoder, 'backbone')
        else trained_critics_model.model.encoder
    )
    gciql_model = swm.wm.gcrl.GCRL(
        encoder=spt.backbone.EvalOnly(encoder_backbone),
        action_predictor=trained_critics_model.model.action_predictor,
        value_predictor=spt.backbone.EvalOnly(
            trained_critics_model.model.value_predictor.student
        ),
        extra_encoders=trained_critics_model.model.extra_encoders,
        history_size=cfg.dinowm.history_size,
    )

    # Wrap in stable_spt Module with separate optimizers for each component
    def add_opt(module_name, lr):
        return {
            'modules': str(module_name),
            'optimizer': {'type': 'AdamW', 'lr': lr},
        }

    gciql_actor_model = spt.Module(
        model=gciql_model,
        forward=forward,
        optim={
            'action_predictor_opt': add_opt(
                'model.action_predictor', cfg.predictor_lr
            ),
            'log_stds_opt': add_opt('model.log_stds', cfg.predictor_lr),
        },
    )
    return gciql_actor_model


# ============================================================================
# Training Setup
# ============================================================================
def setup_pl_logger(cfg, postfix=''):
    if not cfg.wandb.enable:
        return None

    wandb_run_id = cfg.wandb.get('run_id', None)
    wandb_logger = WandbLogger(
        name=f'dino_gciql{postfix}',
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        resume='allow' if wandb_run_id else None,
        id=wandb_run_id,
        log_model=False,
    )

    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))
    return wandb_logger


class ModelObjectCallBack(Callback):
    """Callback to pickle model after each epoch."""

    def __init__(
        self, dirpath, filename='model_object', epoch_interval: int = 1
    ):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ) -> None:
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                output_path = Path(
                    self.dirpath,
                    f'{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt',
                )
                torch.save(pl_module, output_path)
                logging.info(f'Saved world model object to {output_path}')
            # Additionally, save at final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                final_path = self.dirpath / f'{self.filename}_object.ckpt'
                torch.save(pl_module, final_path)
                logging.info(f'Saved final world model object to {final_path}')


# ============================================================================
# Main Entry Point
# ============================================================================
@hydra.main(version_base=None, config_path='./config', config_name='gciql')
def run(cfg):
    """Run training of IQL goal-conditioned policy."""

    wandb_logger_value = setup_pl_logger(cfg, postfix='_value')
    goal_probs = (
        cfg.goal_probabilities.random,
        cfg.goal_probabilities.geometric_future,
        cfg.goal_probabilities.uniform_future,
        cfg.goal_probabilities.current,
    )
    data = get_data(cfg, goal_probabilities=goal_probs)

    # First train value function
    gciql_critics_model = get_gciql_critics_model(cfg)

    cache_dir = swm.data.utils.get_cache_dir()

    if cfg.get('train_value', True):
        dump_object_callback = ModelObjectCallBack(
            dirpath=cache_dir,
            filename=f'{cfg.output_model_name}_value',
            epoch_interval=3,
        )
        # checkpoint_callback = ModelCheckpoint(dirpath=cache_dir, filename=f"{cfg.output_model_name}_weights")

        trainer = pl.Trainer(
            **cfg.trainer,
            callbacks=[dump_object_callback],
            num_sanity_val_steps=1,
            logger=wandb_logger_value,
            enable_checkpointing=True,
        )

        manager = spt.Manager(
            trainer=trainer,
            module=gciql_critics_model,
            data=data,
            ckpt_path=f'{cache_dir}/{cfg.output_model_name}_value_weights.ckpt',
        )
        manager()

    # Extract policy from trained value function
    wandb_logger_policy = setup_pl_logger(cfg, postfix='_policy')
    goal_probs = (
        cfg.actor_goal_probabilities.random,
        cfg.actor_goal_probabilities.geometric_future,
        cfg.actor_goal_probabilities.uniform_future,
        cfg.actor_goal_probabilities.current,
    )
    data = get_data(cfg, goal_probabilities=goal_probs)

    # load value function weights
    checkpoint = torch.load(
        f'{cache_dir}/{cfg.output_model_name}_value_weights.ckpt'
    )
    gciql_critics_model.load_state_dict(checkpoint['state_dict'])

    gciql_actor_model = get_gciql_actor_model(cfg, gciql_critics_model)

    dump_object_callback = ModelObjectCallBack(
        dirpath=cache_dir,
        filename=f'{cfg.output_model_name}_policy',
        epoch_interval=3,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[dump_object_callback],
        num_sanity_val_steps=1,
        logger=wandb_logger_policy,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=gciql_actor_model,
        data=data,
        ckpt_path=f'{cache_dir}/{cfg.output_model_name}_policy_weights.ckpt',
    )
    manager()


if __name__ == '__main__':
    run()
