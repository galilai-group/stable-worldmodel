import torch
from torch.nn import functional as F


def latent_loss(pred, target, loss_type: str):
    if loss_type == 'l1':
        return F.l1_loss(pred, target)
    if loss_type == 'mse':
        return F.mse_loss(pred, target)
    raise ValueError(f'Unsupported loss type: {loss_type}')


def strip_action_dims(tensor, action_range):
    return torch.cat(
        [tensor[..., : action_range[0]], tensor[..., action_range[1] :]],
        dim=-1,
    )


def sample_waypoint_batch(batch, cfg):
    """Randomly choose high-level source/target waypoints inside a span."""
    pixels = batch['pixels']
    batch_size, num_waypoints = pixels.shape[:2]
    device = pixels.device

    source_idx = torch.randint(
        0, num_waypoints - 1, (batch_size,), device=device
    )
    target_idx = torch.empty_like(source_idx)
    for row, src in enumerate(source_idx.tolist()):
        target_idx[row] = torch.randint(
            src + 1, num_waypoints, (1,), device=device
        )

    rows = torch.arange(batch_size, device=device)
    pair_idx = torch.stack([source_idx, target_idx], dim=1)
    pair_rows = rows[:, None]
    selected = {
        key: value[pair_rows, pair_idx]
        if torch.is_tensor(value) and value.shape[:2] == pixels.shape[:2]
        else value
        for key, value in batch.items()
    }

    low_level_actions = batch['action']
    max_chunk = int(cfg.macro_action.max_chunk)
    action_dim = int(cfg.macro_action.low_level_action_dim)
    chunks = low_level_actions.new_zeros(
        batch_size, 1, max_chunk, action_dim
    )
    masks = torch.ones(
        batch_size, 1, max_chunk, dtype=torch.bool, device=device
    )

    for row, (src, dst) in enumerate(
        zip(source_idx.tolist(), target_idx.tolist())
    ):
        chunk = low_level_actions[row, src:dst]
        length = min(chunk.shape[0], max_chunk)
        chunks[row, 0, :length] = chunk[:length]
        masks[row, 0, :length] = False

    selected['action_chunk'] = chunks
    selected['action_mask'] = masks
    selected['waypoint_gap'] = (target_idx - source_idx).float()
    return selected


def hwm_forward(self, batch, stage, cfg):
    """Forward/loss for a long-timescale HWM predictor.

    This stage shares the DINO-WM latent space and predicts between randomly
    sampled waypoints inside a fixed maximum span. The intervening low-level
    action blocks are compressed into one latent macro-action.
    """
    batch = sample_waypoint_batch(batch, cfg)
    action_key = cfg.loss.action_key
    action_chunks = torch.nan_to_num(batch['action_chunk'], 0.0)
    action_masks = batch['action_mask']
    latent_actions = self.action_encoder(action_chunks, action_masks)[:, 0]
    batch[action_key] = torch.stack(
        [latent_actions, torch.zeros_like(latent_actions)], dim=1
    )

    for key in self.model.extra_encoders:
        batch[key] = torch.nan_to_num(batch[key], 0.0)
        if batch[key].ndim == 2:
            batch[key] = batch[key][:, None, :]

    batch = self.model.encode(
        batch,
        target='emb',
        is_video=cfg.backbone.get('is_video_encoder', False),
    )

    prev_embedding = batch['emb'][:, : cfg.wm.history_size, ...]
    pred_embedding = self.model.predict(prev_embedding)
    target_embedding = batch['emb'][:, cfg.wm.history_size :, ...].detach()

    loss_type = cfg.loss.type
    pixels_dim = batch['pixels_emb'].size(-1)
    batch['pixels_loss'] = latent_loss(
        pred_embedding[..., :pixels_dim],
        target_embedding[..., :pixels_dim],
        loss_type,
    )

    start, action_range = pixels_dim, [0, 0]
    for key in self.model.extra_encoders:
        dim = batch[f'{key}_emb'].size(-1)
        lo, hi = start, start + dim
        if key == cfg.loss.action_key:
            action_range = [lo, hi]
        else:
            batch[f'{key}_loss'] = latent_loss(
                pred_embedding[..., lo:hi],
                target_embedding[..., lo:hi].detach(),
                loss_type,
            )
        start = hi

    batch['actionless_emb'] = strip_action_dims(batch['emb'], action_range)
    batch['actionless_prev_emb'] = strip_action_dims(
        prev_embedding, action_range
    )
    batch['actionless_pred_emb'] = strip_action_dims(
        pred_embedding, action_range
    )
    batch['actionless_target_emb'] = strip_action_dims(
        target_embedding, action_range
    )

    batch['loss'] = latent_loss(
        batch['actionless_pred_emb'],
        batch['actionless_target_emb'].detach(),
        loss_type,
    )

    if batch['loss'].isnan():
        raise ValueError('NaN loss encountered!')

    self.log_dict(
        {f'{stage}/{k}': v.detach() for k, v in batch.items() if '_loss' in k},
        on_step=True,
        sync_dist=True,
    )
    self.log(
        f'{stage}/waypoint_gap',
        batch['waypoint_gap'].mean(),
        on_step=True,
        sync_dist=True,
    )
    return batch
