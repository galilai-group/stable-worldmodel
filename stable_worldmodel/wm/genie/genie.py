import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from stable_worldmodel.wm.genie.module import (
    ActionEncoder,
    LAM,
    ST_MaskGIT,
    ST_ViViT,
)


class Genie(nn.Module):
    def __init__(
        self,
        tokenizer: ST_ViViT,
        dynamics: ST_MaskGIT,
        lam: LAM | None = None,
        action_encoder: ActionEncoder | None = None,
        history_size: int = 1,
        num_unmask_steps: int = 8,
        temperature: float = 0.0,
        cost_mode: str = 'embed',
        action_mode: str = 'raw',
        action_block: int = 1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.dynamics = dynamics
        self.lam = lam
        self.action_encoder = action_encoder

        self.history_size = history_size
        self.num_unmask_steps = num_unmask_steps
        self.temperature = temperature

        assert cost_mode in ('pixel', 'token', 'embed'), (
            f'unknown cost_mode {cost_mode!r}'
        )
        self.cost_mode = cost_mode
        assert action_mode in ('raw', 'code'), (
            f'unknown action_mode {action_mode!r}'
        )
        self.action_mode = action_mode
        assert action_block >= 1
        self.action_block = action_block

    @property
    def temporal_dim(self) -> int:
        return self.dynamics.pos_embed_TSC.size(1)

    @property
    def num_actions(self) -> int:
        if self.lam is None:
            raise AttributeError('num_actions requires a LAM')
        return self.lam.vq.num_codes

    @staticmethod
    def _pixels_to_tokenizer(pixels):
        return (
            pixels.mul(2)
            .sub(1)
            .clamp(-1, 1)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

    @staticmethod
    def _tokenizer_to_pixels(video):
        return (
            video.add(1).div(2).clamp(0, 1).permute(0, 1, 4, 2, 3).contiguous()
        )

    def _embed_actions(self, ac):
        if self.action_mode == 'raw':
            assert self.action_encoder is not None, (
                "action_mode='raw' requires an ActionEncoder"
            )
            return self.action_encoder(ac.float())
        assert self.lam is not None, "action_mode='code' requires a LAM"
        return self.lam.vq.codebook(ac.long())

    @torch.no_grad()
    def encode(self, info):
        """Tokenize the prompt frames with the frozen tokenizer.
        info: dict with pixels of shape (B, C, H, W) or (B, T_hist, C, H, W).
        Writes info['tokens'] of shape (B, T_hist, S).
        """
        if 'tokens' in info:
            return info
        pixels = info['pixels']
        T_model = self.temporal_dim
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(1)
        T_hist = pixels.size(1)
        if T_hist < T_model:
            pad = pixels[:, -1:].expand(-1, T_model - T_hist, -1, -1, -1)
            pixels_full = torch.cat([pixels, pad], dim=1)
        else:
            pixels_full = pixels[:, :T_model]
        video = self._pixels_to_tokenizer(pixels_full)
        tokens_full, _, _ = self.tokenizer.encode(video)
        info['tokens'] = tokens_full[:, :T_hist]
        return info

    @torch.no_grad()
    def encode_goal(self, info):
        """Tokenize the goal image once for later cost eval.
        info: dict with goal of shape (B, C, H, W).
        Writes info['goal_tokens'] of shape (B, S).
        """
        if 'goal_tokens' in info:
            return info
        goal = info['goal']
        if goal.ndim == 4:
            goal = goal.unsqueeze(1)
        goal_full = goal.expand(-1, self.temporal_dim, -1, -1, -1)
        goal_video = self._pixels_to_tokenizer(goal_full)
        tokens, _, _ = self.tokenizer.encode(goal_video)
        info['goal_tokens'] = tokens[:, 0]
        return info

    @torch.no_grad()
    def rollout(self, info, action_candidates):
        """Roll the dynamics forward under S action candidates via MaskGIT unmasking.
        info:              dict with pixels; tokens will be produced by self.encode.
        action_candidates: (B, S, T, A_raw) if action_mode='raw' else (B, S, T).
        Writes info['predicted_tokens'] of shape (B, S, T_model, S_spatial),
        and info['predicted_pixels'] (B, S, T_model, C, H, W) when cost_mode='pixel'.
        """
        info = self.encode(info)
        prompt = info['tokens']
        B, H, Sp = prompt.shape
        S = action_candidates.size(1)
        T_model = self.temporal_dim

        prompt_BS = (
            prompt.unsqueeze(1).expand(B, S, H, Sp).reshape(B * S, H, Sp)
        )

        if self.action_mode == 'raw':
            ac = action_candidates.reshape(
                B * S, action_candidates.size(2), -1
            )
            if self.action_block > 1:
                raw_dim = ac.size(-1) // self.action_block
                ac = ac.reshape(
                    B * S, ac.size(1), self.action_block, raw_dim
                ).mean(dim=2)
        else:
            ac = action_candidates.reshape(B * S, action_candidates.size(2))

        # actions align with frame transitions, so the first action drives the H-th
        # frame; prepend H-1 zeros so ac[H-1:] carries the candidate at the right slot.
        target_T = T_model - 1
        prefix_len = H - 1
        if prefix_len > 0:
            if self.action_mode == 'raw':
                prefix = ac.new_zeros(B * S, prefix_len, ac.size(-1))
            else:
                prefix = ac.new_zeros(B * S, prefix_len, dtype=ac.dtype)
            ac = torch.cat([prefix, ac], dim=1)
        if ac.size(1) >= target_T:
            ac = ac[:, :target_T]
        else:
            pad_len = target_T - ac.size(1)
            if self.action_mode == 'raw':
                pad = ac.new_zeros(B * S, pad_len, ac.size(-1))
            else:
                pad = ac.new_zeros(B * S, pad_len, dtype=ac.dtype)
            ac = torch.cat([ac, pad], dim=1)

        action_embeds = self._embed_actions(ac)

        full_tokens = self.dynamics.rollout(
            prompt_TS=prompt_BS,
            num_new_frames=T_model - H,
            num_steps=self.num_unmask_steps,
            actions_T=action_embeds,
            temperature=self.temperature,
        )
        info['predicted_tokens'] = rearrange(
            full_tokens, '(b s) t k -> b s t k', b=B, s=S
        )

        if self.cost_mode == 'pixel':
            decoded = self.tokenizer.decode_from_indices(full_tokens)
            decoded = self._tokenizer_to_pixels(decoded)
            info['predicted_pixels'] = rearrange(
                decoded, '(b s) t c h w -> b s t c h w', b=B, s=S
            )

        return info

    @torch.no_grad()
    def criterion(self, info):
        """Cost between the last predicted frame and the goal, per action candidate.
        Returns a (B, S) tensor; lower is better.
        """
        if self.cost_mode == 'pixel':
            assert 'predicted_pixels' in info
            pred = info['predicted_pixels']
            goal = info['goal']
            while goal.ndim > 4 and goal.size(1) == 1:
                goal = goal.squeeze(1)
            last = pred[..., -1, :, :, :]
            goal_b = goal.unsqueeze(1).expand_as(last)
            return F.mse_loss(last, goal_b, reduction='none').sum(
                dim=(2, 3, 4)
            )

        assert 'predicted_tokens' in info
        assert 'goal_tokens' in info
        pred_last = info['predicted_tokens'][..., -1, :]
        goal = info['goal_tokens']

        if self.cost_mode == 'token':
            return (
                (pred_last != goal.unsqueeze(1).expand_as(pred_last))
                .float()
                .sum(-1)
            )

        codebook = self.tokenizer.vq.codebook
        pred_e = codebook(pred_last)
        goal_e = codebook(goal).unsqueeze(1).expand_as(pred_e)
        return F.mse_loss(pred_e, goal_e, reduction='none').sum(dim=(-2, -1))

    @torch.no_grad()
    def get_cost(self, info, action_candidates):
        """Compute the cost of action candidates given an info dict with goal and initial state."""
        # the solver may broadcast tensors along a candidate axis of size S; drop it
        # so encode/rollout see the raw (B, ...) prompt and candidates flow separately.
        S = action_candidates.size(1)
        collapsed = {}
        for k, v in info.items():
            if torch.is_tensor(v) and v.ndim >= 2 and v.size(1) == S:
                collapsed[k] = v[:, 0]
            else:
                collapsed[k] = v
        if 'goal' in collapsed:
            self.encode_goal(collapsed)
        collapsed = self.rollout(collapsed, action_candidates)
        return self.criterion(collapsed)


__all__ = ['Genie']
