import torch
from einops import rearrange, repeat
from torch import nn


class GCRL(torch.nn.Module):
    def __init__(
        self,
        encoder,
        action_predictor,
        value_predictor=None,
        critic_predictor=None,
        extra_encoders=None,
        history_size=3,
        interpolate_pos_encoding=True,
        log_std_min=-5.0,
        log_std_max=2.0,
    ):
        super().__init__()

        self.encoder = encoder
        self.value_predictor = value_predictor
        self.action_predictor = action_predictor
        self.critic_predictor = critic_predictor
        self.extra_encoders = extra_encoders or {}
        self.history_size = history_size

        self.interpolate_pos_encoding = interpolate_pos_encoding

        # Learnable log_stds for action distribution (state-independent)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # Support both nn.Linear and nn.Sequential for out_proj
        out_proj = action_predictor.out_proj
        if isinstance(out_proj, nn.Sequential):
            action_dim = out_proj[-1].out_features
        else:
            action_dim = out_proj.out_features
        self.log_stds = nn.Parameter(torch.zeros(action_dim))

    def encode(
        self,
        info,
        pixels_key='pixels',
        emb_keys=None,
        prefix=None,
        target='embed',
        is_video=False,
    ):
        assert target not in info, f'{target} key already in info_dict'
        emb_keys = emb_keys or self.extra_encoders.keys()
        prefix = prefix or ''

        encode_fn = self._encode_video if is_video else self._encode_image
        pixels_embed = encode_fn(info[pixels_key].float())  # (B, T, 3, H, W)

        # == improve the embedding
        n_patches = pixels_embed.shape[2]
        embedding = pixels_embed
        info[f'pixels_{target}'] = pixels_embed

        for key in emb_keys:
            extr_enc = self.extra_encoders[key]
            extra_input = info[f'{prefix}{key}'].float()  # (B, T, dim)
            extra_embed = extr_enc(
                extra_input
            )  # (B, T, dim) -> (B, T, emb_dim)
            info[f'{key}_{target}'] = extra_embed

            # copy extra embedding across patches for each time step
            extra_tiled = repeat(
                extra_embed.unsqueeze(2), 'b t 1 d -> b t p d', p=n_patches
            )

            # concatenate along feature dimension
            embedding = torch.cat([embedding, extra_tiled], dim=3)

        info[target] = embedding  # (B, T, P, d)

        return info

    def _encode_image(self, pixels):
        # == pixels embedding
        B = pixels.shape[0]
        pixels = rearrange(pixels, 'b t ... -> (b t) ...')

        kwargs = (
            {'interpolate_pos_encoding': True}
            if self.interpolate_pos_encoding
            else {}
        )
        pixels_embed = self.encoder(pixels, **kwargs)

        if hasattr(pixels_embed, 'last_hidden_state'):
            pixels_embed = pixels_embed.last_hidden_state
            pixels_embed = pixels_embed[:, 1:, :]  # drop cls token
        else:
            pixels_embed = pixels_embed.logits.unsqueeze(
                1
            )  # (B*T, 1, emb_dim)

        pixels_embed = rearrange(pixels_embed, '(b t) p d -> b t p d', b=B)

        return pixels_embed

    def _encode_video(self, pixels):
        B, T, C, H, W = pixels.shape
        kwargs = (
            {'interpolate_pos_encoding': True}
            if self.interpolate_pos_encoding
            else {}
        )

        pixels_embeddings = []

        # roll the embedding computation over time
        for t in range(T):
            padding = max(T - (t + 1), 0)  # number of frames to pad
            past_frames = pixels[:, : t + 1, :, :, :]  # (B, t+1, C, H, W)

            # repeat last frame to pad
            pad_frames = past_frames[:, -1:, :, :, :].repeat(
                1, padding, 1, 1, 1
            )  # (B, padding, C, H, W)
            frames = torch.cat(
                [past_frames, pad_frames], dim=1
            )  # (B, T, C, H, W)

            frame_embed = self.encoder(frames, **kwargs)  # (B, 1, P, emb_dim)
            frame_embed = frame_embed.last_hidden_state
            pixels_embeddings.append(frame_embed)

        pixels_embed = torch.stack(
            pixels_embeddings, dim=1
        )  # (B, T, P, emb_dim)

        return pixels_embed

    def predict_actions(self, embedding, embedding_goal, temperature=1.0):
        """predict action distribution per frame
        Args:
            embedding: (B, T, P, d)
            embedding_goal: (B, 1, P, d)
            temperature: scaling factor for the standard deviation
        Returns:
            means: (B, T, action_dim) - action means
            stds: (action_dim,) - action standard deviations (broadcasted)
        """

        embedding = rearrange(embedding, 'b t p d -> b (t p) d')
        embedding_goal = rearrange(embedding_goal, 'b t p d -> b (t p) d')
        means = self.action_predictor(embedding, embedding_goal)

        # Clip log_stds and compute scale
        log_stds = torch.clamp(
            self.log_stds, self.log_std_min, self.log_std_max
        )
        stds = torch.exp(log_stds) * temperature

        return means, stds

    def predict_values(self, embedding, embedding_goal):
        """predict values per frame
        Args:
            embedding: (B, T, P, d)
            embedding_goal: (B, 1, P, d)
        Returns:
            preds: (B, T, 1)
        """

        embedding = rearrange(embedding, 'b t p d -> b (t p) d')
        embedding_goal = rearrange(embedding_goal, 'b t p d -> b (t p) d')
        preds = self.value_predictor(embedding, embedding_goal)

        return preds

    def get_action(self, info, sample=False, temperature=1.0):
        """Get action given observation and goal (uses last frame's prediction).

        Args:
            info: dict containing 'pixels' and 'goal' keys
            sample: if True, sample from distribution; if False, return mean
            temperature: scaling factor for std when sampling
        Returns:
            actions: (B, action_dim)
        """
        # first encode observation
        info = self.encode(info, pixels_key='pixels', target='embed')
        # encode goal
        info = self.encode(
            info,
            pixels_key='goal',
            prefix='goal_',
            target='goal_embed',
        )
        # then predict action distribution
        means, stds = self.predict_actions(
            info['embed'], info['goal_embed'], temperature=temperature
        )
        # get last frame's action prediction
        means = means[:, -1, :]

        if sample:
            # Sample from Normal distribution
            actions = means + stds * torch.randn_like(means)
        else:
            actions = means

        return actions


__all__ = ['GCRL']
