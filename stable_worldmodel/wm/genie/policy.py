"""Costable wrapper for Genie, plug-in to the project's solver harness.

Mirrors `stable_worldmodel.wm.lewm.LeWM` / `pldm.PLDM`'s interface:

    info = model.encode(info)
    info = model.rollout(info, action_candidates)
    cost = model.criterion(info)
    cost = model.get_cost(info, action_candidates)        # encode+rollout+criterion

Action conditioning is configurable:
  * action_mode="raw"  — solver samples raw env actions; policy projects them
                         via `model.action_encoder` (a trained MLP). The
                         well-known limitation: raw_action → LAM_embed is
                         many-to-one because LAM codes depend on state, not
                         just action.
  * action_mode="code" — solver samples integer LAM codes directly (use
                         CategoricalCEMSolver). Forward path is exact; you
                         still need a separate code→action map to execute.

Shape conventions (matching LeWM):
    info["pixels"] : (B, H, C, Hi, Wi)         initial context, H frames
    info["goal"]   : (B, C, Hi, Wi)            goal image
    action_candidates / sequences :
        (B, S, T, A_raw)   when action_mode="raw"
        (B, S, T)          when action_mode="code"

Pixel ranges: this policy expects inputs in [0, 1]. The Genie tokenizer
internally rescales to [-1, 1].
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from stable_worldmodel.wm.genie.genie import Genie


class GeniePolicy(nn.Module):
    def __init__(
        self,
        model: Genie,
        history_size: int = 1,
        num_unmask_steps: int = 8,
        temperature: float = 1.0,
        cost_mode: str = "pixel",
        action_mode: str = "raw",
        record_env_idx: int = -1,
        action_block: int = 1,
    ):
        super().__init__()
        self.model = model
        self.history_size = history_size
        self.num_unmask_steps = num_unmask_steps
        self.temperature = temperature
        assert cost_mode in ("pixel", "token", "embed"), f"unknown cost_mode {cost_mode!r}"
        self.cost_mode = cost_mode
        self.cost_in_pixel_space = (cost_mode == "pixel")   # kept for external callers
        assert action_mode in ("raw", "code"), f"unknown action_mode {action_mode!r}"
        self.action_mode = action_mode
        assert action_block >= 1
        self.action_block = action_block
        self.record_env_idx = record_env_idx
        self.rollout_log: list[dict] = []
        self._step_counter = 0
        self._log_every = 1
        self.world = None
        self.stream_dir = None                  # if set, dumps per-env-step PNGs here
        self._last_obs_hash = None
        self._env_step_counter = 0

    # ────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _pixels_to_genie(pixels: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) in [0, 1] → (B, T, H, W, C) in [-1, 1]."""
        return pixels.mul(2).sub(1).clamp(-1, 1).permute(0, 1, 3, 4, 2).contiguous()

    @staticmethod
    def _genie_to_pixels(video: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W, C) in [-1, 1] → (B, T, C, H, W) in [0, 1]."""
        return video.add(1).div(2).clamp(0, 1).permute(0, 1, 4, 2, 3).contiguous()

    def _embed_actions(self, action_candidates: torch.Tensor) -> torch.Tensor:
        """(BS, T, A) raw  OR  (BS, T) ids  →  (BS, T, emb_dim) action embeds."""
        if self.action_mode == "raw":
            assert self.model.action_encoder is not None, (
                "action_mode='raw' requires model.action_encoder to be set"
            )
            return self.model.action_encoder(action_candidates.float())
        # code mode
        return self.model.lam.vq.codebook(action_candidates.long())

    # ────────────────────────────────────────────────────────────────────
    # Costable interface
    # ────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, info: dict) -> dict:
        """Tokenize the prompt video; cache tokens under info['tokens'].

        Accepts pixels with or without a leading time-history axis. The
        tokenizer is trained over T_model frames, so a short prompt is
        right-padded by repeating the last observed frame; we take only the
        real-frame slice of the output tokens.
        """
        if "tokens" in info:
            return info
        pixels = info["pixels"]
        T_model = self.model.temporal_dim
        if pixels.ndim == 4:
            # (B, C, Hi, Wi) — no time axis; treat as single-frame history.
            pixels = pixels.unsqueeze(1)                    # (B, 1, C, Hi, Wi)
        T_hist = pixels.size(1)
        if T_hist < T_model:
            pad = pixels[:, -1:].expand(-1, T_model - T_hist, -1, -1, -1)
            pixels_full = torch.cat([pixels, pad], dim=1)
        else:
            pixels_full = pixels[:, :T_model]
        video = self._pixels_to_genie(pixels_full)          # (B, T_model, Hi, Wi, C)
        tokens_full = self.model.encode(video)              # (B, T_model, S)
        info["tokens"] = tokens_full[:, :T_hist]
        return info

    @torch.no_grad()
    def encode_goal(self, info: dict) -> dict:
        """Tokenize the goal image. Replicates the single goal frame to fill the
        tokenizer's temporal window (otherwise pos_embed broadcasting maps a 1-frame
        input to T_model outputs). Stored under info['goal_tokens'].
        """
        if "goal_tokens" in info:
            return info
        T_model = self.model.temporal_dim
        goal = info["goal"]
        if goal.ndim == 4:
            goal = goal.unsqueeze(1)                       # (B, 1, C, Hi, Wi)
        goal_full = goal.expand(-1, T_model, -1, -1, -1)   # (B, T_model, C, Hi, Wi)
        goal_video = self._pixels_to_genie(goal_full)      # (B, T_model, Hi, Wi, C)
        info["goal_tokens"] = self.model.encode(goal_video)[:, 0]   # (B, Sp) — pos-0 of constant-content video
        return info

    @torch.no_grad()
    def rollout(self, info: dict, action_candidates: torch.Tensor) -> dict:
        """Run dyn iterative-unmask over action candidates.

        action_candidates: (B, S, T, A) or (B, S, T) depending on action_mode.
        After this call info contains:
            predicted_tokens : (B, S, T_model, Sp)
            predicted_pixels : (B, S, T_model, C, Hi, Wi)    (if cost_in_pixel_space)
        """
        info = self.encode(info)
        prompt = info["tokens"]                            # (B, H, Sp)
        B, H, Sp = prompt.shape
        S = action_candidates.size(1)
        T_model = self.model.temporal_dim

        # Broadcast prompt over the sample dimension and flatten.
        prompt_BS = prompt.unsqueeze(1).expand(B, S, H, Sp).reshape(B * S, H, Sp)

        if self.action_mode == "raw":
            ac = action_candidates.reshape(B * S, action_candidates.size(2), -1)
            if self.action_block > 1:
                # Solver's per-slot action is `action_block` env actions concatenated.
                # WM was trained on the mean-pool over those env-steps.
                raw_dim = ac.size(-1) // self.action_block
                ac = ac.reshape(B * S, ac.size(1), self.action_block, raw_dim).mean(dim=2)
        else:
            ac = action_candidates.reshape(B * S, action_candidates.size(2))

        target_T = T_model - 1
        prefix_len = H - 1
        if prefix_len > 0:
            if self.action_mode == "raw":
                prefix = ac.new_zeros(B * S, prefix_len, ac.size(-1))
            else:
                prefix = ac.new_zeros(B * S, prefix_len, dtype=ac.dtype)
            ac = torch.cat([prefix, ac], dim=1)

        if ac.size(1) >= target_T:
            ac = ac[:, :target_T]
        else:
            pad_len = target_T - ac.size(1)
            if self.action_mode == "raw":
                pad = ac.new_zeros(B * S, pad_len, ac.size(-1))
            else:
                pad = ac.new_zeros(B * S, pad_len, dtype=ac.dtype)
            ac = torch.cat([ac, pad], dim=1)

        action_embeds = self._embed_actions(ac)

        full_tokens = self.model.dynamics.rollout(
            prompt_TS=prompt_BS,
            num_new_frames=T_model - H,
            num_steps=self.num_unmask_steps,
            actions_T=action_embeds,
            temperature=self.temperature,
        )                                                  # (BS, T_model, Sp)

        info["predicted_tokens"] = rearrange(full_tokens, "(b s) t k -> b s t k", b=B, s=S)

        if self.cost_mode == "pixel":
            decoded = self.model.decode(full_tokens)       # (BS, T_model, Hi, Wi, C)
            decoded = self._genie_to_pixels(decoded)       # (BS, T_model, C, Hi, Wi)
            info["predicted_pixels"] = rearrange(decoded, "(b s) t c h w -> b s t c h w", b=B, s=S)

        return info

    @torch.no_grad()
    def criterion(self, info: dict) -> torch.Tensor:
        """Final-step distance from rollout to goal. Returns (B, S) cost."""
        if self.cost_mode == "pixel":
            assert "predicted_pixels" in info, "must call rollout() first"
            pred = info["predicted_pixels"]                # (B, S, T, C, H, W)
            goal = info["goal"]                            # ideally (B, C, H, W)
            while goal.ndim > 4 and goal.size(1) == 1:
                goal = goal.squeeze(1)
            last = pred[..., -1, :, :, :]                  # (B, S, C, H, W)
            goal_b = goal.unsqueeze(1).expand_as(last)
            return F.mse_loss(last, goal_b, reduction="none").sum(dim=(2, 3, 4))

        assert "predicted_tokens" in info, "must call rollout() first"
        assert "goal_tokens" in info, "must call encode_goal() first"
        pred_last = info["predicted_tokens"][..., -1, :]   # (B, S, Sp) long
        goal = info["goal_tokens"]                         # (B, Sp) long

        if self.cost_mode == "token":
            goal_b = goal.unsqueeze(1).expand_as(pred_last)
            return (pred_last != goal_b).float().sum(-1)

        # embed: codebook L2 in the pre-quantization vector space.
        codebook = self.model.tokenizer.vq.codebook
        pred_e = codebook(pred_last)                       # (B, S, Sp, E)
        goal_e = codebook(goal).unsqueeze(1).expand_as(pred_e)
        return F.mse_loss(pred_e, goal_e, reduction="none").sum(dim=(-2, -1))

    @torch.no_grad()
    def get_cost(self, info: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """One-shot encode + rollout + criterion.

        The MPC solver auto-expands every tensor in `info` to shape (B, S, ...)
        so that costs can be computed per-sample. But GeniePolicy's other
        methods (encode, encode_goal, rollout, criterion) assume (B, ...)
        without the sample axis — they handle the S broadcast internally on
        the action_candidates path. Collapse the S axis once here so all
        downstream code sees the expected layout.
        """
        S = action_candidates.size(1)
        collapsed = {}
        for k, v in info.items():
            if torch.is_tensor(v) and v.ndim >= 2 and v.size(1) == S:
                collapsed[k] = v[:, 0]                  # all samples share env state
            else:
                collapsed[k] = v
        if "goal" in collapsed:
            self.encode_goal(collapsed)
        collapsed = self.rollout(collapsed, action_candidates)
        cost = self.criterion(collapsed)

        self._step_counter += 1
        if self._step_counter % self._log_every == 0:
            n_env = cost.size(0)
            succ = int(self.world.terminateds.sum()) if self.world is not None else -1
            total = self.world.num_envs if self.world is not None else n_env
            print(f"[step {self._step_counter:>3d}] cost min={cost.min().item():7.2f} "
                  f"mean={cost.mean().item():7.2f} max={cost.max().item():7.2f} "
                  f"| success {succ}/{total}", flush=True)

        if self.stream_dir is not None and self.world is not None:
            full_obs = self.world.infos.get("pixels") if hasattr(self.world, "infos") else None
            if full_obs is not None:
                import numpy as _np
                obs_hash = hash(_np.ascontiguousarray(full_obs).tobytes())
                if obs_hash != self._last_obs_hash:
                    self._last_obs_hash = obs_hash
                    self._env_step_counter += 1
                    try:
                        from PIL import Image
                        self.stream_dir.mkdir(parents=True, exist_ok=True)
                        for i in range(full_obs.shape[0]):
                            frame = full_obs[i]
                            while frame.ndim > 3:
                                frame = frame[0]
                            if frame.dtype != _np.uint8:
                                frame = (frame.clip(0, 1) * 255).astype(_np.uint8)
                            if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
                                frame = frame.transpose(1, 2, 0)
                            Image.fromarray(frame).save(
                                self.stream_dir / f"env{i:02d}_step{self._env_step_counter:03d}.png"
                            )
                    except Exception as e:
                        print(f"[stream] save failed: {e}", flush=True)

        if 0 <= self.record_env_idx < cost.size(0):
            e = self.record_env_idx
            best_s = int(cost[e].argmin().item())
            obs = collapsed["pixels"][e] if "pixels" in collapsed else None
            goal_p = collapsed["goal"][e] if "goal" in collapsed else None
            pred = collapsed["predicted_pixels"][e, best_s] if "predicted_pixels" in collapsed else None
            self.rollout_log.append({
                "obs": obs.detach().float().cpu().numpy() if obs is not None else None,
                "goal": goal_p.detach().float().cpu().numpy() if goal_p is not None else None,
                "best_predicted": pred.detach().float().cpu().numpy() if pred is not None else None,
                "best_action": action_candidates[e, best_s].detach().float().cpu().numpy(),
                "best_cost": float(cost[e, best_s].item()),
                "cost_mean": float(cost[e].mean().item()),
                "cost_min": float(cost[e].min().item()),
            })

        return cost


__all__ = ["GeniePolicy"]
