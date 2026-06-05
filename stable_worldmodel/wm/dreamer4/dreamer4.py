import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .module import Dynamics, make_schedule, sample_next_frame


class DreamerV4WM(nn.Module):
    """DreamerV4-style world model for stable-worldmodel.

    Architecture:
      - encoder: frozen ViT (from stable_pretraining) maps frames -> CLS tokens
      - projector: linear D_emb -> d_spatial (bottleneck)
      - dynamics: block-causal transformer with flow matching + shortcut forcing

    Training uses dynamics_loss() from module.py (flow matching + bootstrap consistency).
    Inference uses sample_next_frame() with K=1 shortcut steps (fast) or K=k_max (precise).

    Implements the same encode/predict/rollout/get_cost interface as LeWM/PLDM.
    """

    def __init__(
        self,
        encoder,
        dynamics: Dynamics,
        projector: nn.Module | None = None,
        k_max: int = 8,
        eval_K: int = 1,
    ):
        """
        Args:
            encoder:    ViT from stable_pretraining (output.last_hidden_state[:, 0] used)
            dynamics:   Dynamics module from module.py
            projector:  Linear D_emb -> d_spatial; identity if None
            k_max:      max flow integration steps (must be power of 2)
            eval_K:     steps used at rollout/MPC time (1=fastest, k_max=most precise)
        """
        super().__init__()
        self.encoder = encoder
        self.dynamics = dynamics
        self.projector = projector or nn.Identity()
        self.k_max = k_max
        self.eval_K = eval_K

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, info: dict) -> dict:
        """Encode pixels -> latent embeddings and packed spatial tokens.

        info['pixels']: (B, T, C, H, W)
        Sets:
          info['emb']:      (B, T, d_spatial)  - flat CLS embedding (for MPC cost)
          info['packed_z']: (B, T, 1, d_spatial) - spatial tokens for dynamics
          info['act_emb']:  (B, T, action_dim)   - passthrough (actions not embedded here)
        """
        pixels = info['pixels'].to(next(self.encoder.parameters()).dtype)
        b = pixels.size(0)
        pixels = rearrange(pixels, 'b t ... -> (b t) ...')
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        cls = output.last_hidden_state[:, 0]  # (B*T, D_emb)
        emb = self.projector(cls)  # (B*T, d_spatial)
        emb = rearrange(emb, '(b t) d -> b t d', b=b)  # (B, T, d_spatial)

        info['emb'] = emb
        info['packed_z'] = emb.unsqueeze(2)  # (B, T, 1, d_spatial)

        return info

    # ------------------------------------------------------------------
    # Predict (single step, used during training)
    # ------------------------------------------------------------------

    def predict(
        self,
        packed_z: torch.Tensor,
        actions: torch.Tensor,
        step_idxs: torch.Tensor,
        signal_idxs: torch.Tensor,
    ) -> torch.Tensor:
        """One dynamics forward pass (used in training loss computation).

        packed_z: (B, T, n_spatial, d_spatial)  noisy input
        Returns:  (B, T, n_spatial, d_spatial)  x1_hat (predicted clean latent)
        """
        return self.dynamics(actions, step_idxs, signal_idxs, packed_z)

    # ------------------------------------------------------------------
    # Rollout (autoregressive, used for MPC)
    # ------------------------------------------------------------------

    def rollout(
        self, info: dict, action_sequence: torch.Tensor, history_size: int = 8
    ) -> dict:
        """Autoregressively roll out the world model.

        action_sequence: (B, S, T, action_dim)
          S = number of MPC action candidates
          T = history_len + horizon

        Sets info['predicted_emb']: (B, S, T_pred, d_spatial)
        """
        assert 'pixels' in info, 'pixels not in info'
        H = info['pixels'].size(1)  # history length
        B, S, T = action_sequence.shape[:3]
        n_future = T - H

        act_hist = action_sequence[:, :, :H]
        act_future = action_sequence[:, :, H:]

        # encode history (reuse cached if present)
        # info['pixels'] is (B, H, C, h, w) — encode the full history sequence
        if 'packed_z' not in info:
            _init = self.encode(
                {k: v for k, v in info.items() if torch.is_tensor(v)}
            )
            # _init['packed_z']: (B, H, 1, d_spatial) — expand over S
            info['packed_z'] = (
                _init['packed_z'].unsqueeze(1).expand(B, S, -1, -1, -1)
            )
            info['emb'] = _init['emb'].unsqueeze(1).expand(B, S, -1, -1)

        sched = make_schedule(self.k_max, self.eval_K)

        # flatten B and S for rollout
        pz = rearrange(
            info['packed_z'], 'b s t n d -> (b s) t n d'
        ).detach()  # (BS, H, 1, d_s)
        acts = rearrange(
            torch.cat([act_hist, act_future], dim=2),
            'b s t a -> (b s) t a',
        )  # (BS, T, action_dim)

        emb_list = list(pz.unbind(dim=1))  # H tensors of (BS, 1, d_s)

        for t in range(n_future + 1):
            lo = max(0, H + t - history_size)
            ctx = torch.stack(emb_list[lo:], dim=1)  # (BS, HS, 1, d_s)
            acts_in = acts[:, lo : H + t + 1] if acts is not None else None

            z_next = sample_next_frame(
                self.dynamics,
                past_packed=ctx,
                sched=sched,
                actions=acts_in,
                k_max=self.k_max,
            )  # (BS, n_spatial, d_s)
            emb_list.append(z_next)

        # stack and extract flat embedding (squeeze n_spatial=1)
        packed_seq = torch.stack(emb_list, dim=1)  # (BS, H+n_future+1, 1, d_s)
        pred_emb = packed_seq.squeeze(2)  # (BS, H+n_future+1, d_s)
        pred_emb = rearrange(pred_emb, '(b s) t d -> b s t d', b=B, s=S)
        info['predicted_emb'] = pred_emb

        return info

    # ------------------------------------------------------------------
    # MPC cost
    # ------------------------------------------------------------------

    def criterion(self, info_dict: dict) -> torch.Tensor:
        pred_emb = info_dict['predicted_emb']  # (B, S, T, d_spatial)
        goal_emb = info_dict['goal_emb']  # (B, ..., d_spatial)
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction='none',
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)
        return cost

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        assert 'goal' in info_dict

        if 'goal_emb' not in info_dict:
            goal = {
                k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)
            }
            goal['pixels'] = goal['goal']
            for k in list(goal.keys()):
                if k.startswith('goal_'):
                    goal[k[len('goal_') :]] = goal.pop(k)
            goal.pop('action', None)
            goal = self.encode(goal)
            info_dict['goal_emb'] = goal['emb']

        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)


__all__ = ['DreamerV4WM']
