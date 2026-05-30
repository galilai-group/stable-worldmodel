import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class Diamond(nn.Module):
    """Minimal Diamond (diffusion) world-model wrapper.

    This class provides a compatible interface with other world-models in the
    repository (LeWM / PLDM). It is intentionally lightweight: it expects an
    encoder, a predictor (or diffusion model proxy), and an action_encoder to be
    passed in at construction. The public API mirrors LeWM to ease integration
    with existing tooling in this codebase.
    """

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        **kwargs,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        # optional reward/termination head
        self.rhead = kwargs.get('rhead', None)

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with `pixels` and optional `action` keys
        """
        pixels = info['pixels'].to(next(self.encoder.parameters()).dtype)
        b = pixels.size(0)
        pixels = rearrange(pixels, 'b t ... -> (b t) ...')
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        emb = self.projector(pixels_emb)
        info['emb'] = rearrange(emb, '(b t) d -> b t d', b=b)

        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])

        return info

    def predict(self, emb, act_emb):
        """Predict next-state embedding. Kept compatible with LeWM/PLDM."""
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, 'b t d -> (b t) d'))
        preds = rearrange(preds, '(b t) d -> b t d', b=emb.size(0))
        return preds

    def predict_reward_term(self, emb):
        """Optional helper to call reward/termination head if present."""
        if self.rhead is None:
            raise RuntimeError('No reward/termination head attached')
        # returns reward (B,T,1) and terminal logits (B,T,1)
        return self.rhead(emb)

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and an action sequence.

        The API mirrors LeWM.rollout so downstream code can use Diamond
        interchangeably for planning/rollouts.
        """
        assert 'pixels' in info, 'pixels not in info_dict'
        H = info['pixels'].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info['action'] = act_0
        n_steps = T - H

        # encode initial state if not already present
        if 'emb' not in info:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            _init = self.encode(_init)
            info['emb'] = (
                _init['emb'].detach().unsqueeze(1).expand(B, S, -1, -1)
            )

        emb_init = rearrange(info['emb'], 'b s ... -> (b s) ...')
        act_flat = rearrange(act_0, 'b s ... -> (b s) ...')
        act_future_flat = rearrange(act_future, 'b s ... -> (b s) ...')
        all_act_emb = self.action_encoder(
            torch.cat([act_flat, act_future_flat], dim=1)
        )

        # autoregressive rollout
        HS = history_size
        emb_list = list(emb_init.unbind(dim=1))
        for t in range(n_steps + 1):
            lo = max(0, H + t - HS)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)
            act_trunc = all_act_emb[:, lo : H + t]
            emb_list.append(self.predict(emb_trunc, act_trunc)[:, -1])

        emb = torch.stack(emb_list, dim=1)
        pred_rollout = rearrange(emb, '(b s) ... -> b s ...', b=B, s=S)
        info['predicted_emb'] = pred_rollout
        return info

    def criterion(self, info_dict: dict):
        pred_emb = info_dict['predicted_emb']
        goal_emb = info_dict['goal_emb']
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction='none',
        ).sum(dim=tuple(range(2, pred_emb.ndim)))
        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        assert 'goal' in info_dict, 'goal not in info_dict'

        if 'goal_emb' not in info_dict:
            goal = {
                k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)
            }
            goal['pixels'] = goal['goal']
            for k in list(info_dict.keys()):
                if k.startswith('goal_'):
                    goal[k[len('goal_') :]] = goal.pop(k)
            goal.pop('action', None)
            goal = self.encode(goal)
            info_dict['goal_emb'] = goal['emb']

        info_dict = self.rollout(info_dict, action_candidates)
        cost = self.criterion(info_dict)
        return cost


__all__ = ['Diamond']
