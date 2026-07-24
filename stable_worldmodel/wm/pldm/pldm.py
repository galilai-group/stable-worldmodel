"""JEPA Implementation"""

import torch
from einops import rearrange
from torch import nn

from .module import detach_clone


class PLDM(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """
        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(
            pixels, 'b t ... -> (b t) ...'
        )  # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info['emb'] = rearrange(emb, '(b t) d -> b t d', b=b)

        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, 'b t d -> (b t) d'))
        preds = rearrange(preds, '(b t) d -> b t d', b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = None):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, H, C, h, w) — H context frames (block timesteps)
        action_sequence: (B, S, T, action_dim) — strictly-future candidates
        info['action_history']: (B, S, H - 1, action_dim) — executed action
            blocks between the context frames (required when H > 1)
         - S is the number of action plan samples
         - T is the planning horizon
        Returns ``info`` with ``predicted_emb`` of shape (B, S, H + T, D);
        the first H entries are the encoded context frames.
        """
        if history_size is None:
            history_size = getattr(self.predictor, 'num_frames', 3)

        assert 'pixels' in info, 'pixels not in info_dict'
        H = info['pixels'].size(2)
        B, S, T = action_sequence.shape[:3]
        act_past = info.get('action_history')
        if act_past is None:
            act_past = action_sequence.new_zeros(
                B, S, 0, action_sequence.size(-1)
            )
        assert act_past.size(2) == H - 1, (
            f'action_history must hold H-1={H - 1} executed blocks, '
            f'got {act_past.size(2)}'
        )
        # action paired with context frame k is the block leaving it; the
        # current frame (k = H-1) pairs with the first candidate
        info['action'] = torch.cat(
            [act_past, action_sequence[:, :, :1]], dim=2
        )
        n_steps = T - 1

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info['emb'] = _init['emb'].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, 'b s ... -> (b s) ...').clone()
        act = rearrange(info['action'], 'b s ... -> (b s) ...')
        act_future = rearrange(
            action_sequence[:, :, 1:], 'b s ... -> (b s) ...'
        )

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, '(b s) ... -> b s ...', b=B, s=S)
        info['predicted_emb'] = pred_rollout

        return info


__all__ = ['PLDM']
