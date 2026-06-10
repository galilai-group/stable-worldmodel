import torch
from torch import nn
from stable_worldmodel.wm.genie.st_vivit import ST_ViViT
from stable_worldmodel.wm.genie.lam import LAM
from stable_worldmodel.wm.genie.st_maskgit import ST_MaskGIT


class Genie(nn.Module):
    """
    The full Genie model: video tokenizer + latent action model + dynamics model.

    This wrapper is the inference-facing top level. Nothing here computes a loss.

    Two ways to generate:
      - `play`: one-shot. Give a prompt video and the full action sequence,
        get back the full window of pixels.
      - `step`: interactive. Extend a running context by one frame at a time.
        The caller manages the context buffer between calls.
    """
    def __init__(
        self,
        tokenizer: ST_ViViT,
        lam: LAM,
        dynamics: ST_MaskGIT,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.lam = lam
        self.dynamics = dynamics

    @property
    def temporal_dim(self) -> int:
        """Maximum window the dynamics model attends over."""
        return self.dynamics.pos_embed_TSC.size(1)

    @property
    def num_actions(self) -> int:
        """Size of the latent action codebook (e.g. 8 in the paper)."""
        return self.lam.vq.num_codes

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """(B, T_prompt, H, W, c) → (B, T_prompt, S) discrete token ids."""
        tokens, _, _ = self.tokenizer.encode(video)
        return tokens

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, T, S) discrete token ids → (B, T, H, W, c) pixels."""
        return self.tokenizer.decode_from_indices(tokens)

    @torch.no_grad()
    def play(
        self,
        prompt_video: torch.Tensor,
        action_ids: torch.Tensor,
        num_steps: int = 25,
        temperature: float = 2.0,
        unmask_mode: str = "random",
    ) -> torch.Tensor:
        """
        One-shot generation. Given a prompt video and the full action sequence
        for the window, return the full pixel video including the prompt.

        Args:
            prompt_video: (B, T_prompt, H, W, c). T_prompt must be < temporal_dim.
            action_ids:   (B, T-1) latent action ids in [0, num_actions). One per
                          frame transition in the output window. Positions inside
                          the prompt can be set to 0 if no real action is known.
            num_steps:    MaskGIT iterations per generated frame.
            temperature:  sampling temperature (paper uses 2.0).
            unmask_mode:  "random" (default) or "greedy".

        Returns:
            video: (B, T, H, W, c) where T = temporal_dim.
        """
        T = self.temporal_dim
        prompt_tokens = self.encode(prompt_video)             # (B, T_prompt, S)
        num_new_frames = T - prompt_tokens.size(1)

        action_embeds = self.lam.vq.codebook(action_ids)      # (B, T-1, E)

        full_tokens = self.dynamics.rollout(
            prompt_tokens,
            num_new_frames=num_new_frames,
            num_steps=num_steps,
            actions_T=action_embeds,
            temperature=temperature,
            unmask_mode=unmask_mode,
        )
        return self.decode(full_tokens)

    @torch.no_grad()
    def step(
        self,
        context_tokens: torch.Tensor,
        action_ids: torch.Tensor,
        num_steps: int = 25,
        temperature: float = 2.0,
        unmask_mode: str = "random",
    ) -> torch.Tensor:
        """
        Interactive generation. Extend a running token context by one frame.

        Args:
            context_tokens: (B, t, S) tokens for the first t frames (t < temporal_dim).
            action_ids:     (B, t) full action history. action_ids[:, -1] is the
                            action being taken to produce the new frame.
            num_steps:      MaskGIT iterations.
            temperature:    sampling temperature.
            unmask_mode:    "random" or "greedy".

        Returns:
            new_frame_tokens: (B, S) the freshly sampled frame in token space.
                              Caller is responsible for appending it to its context.
        """
        T = self.temporal_dim
        B, t, S = context_tokens.shape
        assert t < T, f"context already at max length {T}; cannot step further"
        assert action_ids.size(1) == t, \
            f"need {t} action ids (full history), got {action_ids.size(1)}"

        # Pad token context with fully-masked tail
        masked_tail = torch.full(
            (B, T - t, S),
            self.dynamics.mask_token_id,
            dtype=context_tokens.dtype,
            device=context_tokens.device,
        )
        full_tokens = torch.cat([context_tokens, masked_tail], dim=1)

        # Look up action embeddings, pad to T-1 with zeros (= "no action" for unknown future positions)
        action_embeds = self.lam.vq.codebook(action_ids)      # (B, t, E)
        pad_len = (T - 1) - t
        if pad_len > 0:
            zero_pad = torch.zeros(
                B, pad_len, action_embeds.size(-1),
                device=action_embeds.device, dtype=action_embeds.dtype,
            )
            action_embeds = torch.cat([action_embeds, zero_pad], dim=1)

        return self.dynamics.sample_frame(
            full_tokens,
            frame_idx=t,
            num_steps=num_steps,
            actions_T=action_embeds,
            temperature=temperature,
            unmask_mode=unmask_mode,
        )
