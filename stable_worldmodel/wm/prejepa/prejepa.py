import torch
import torch.nn.functional as F
from einops import rearrange, repeat


class PreJEPA(torch.nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        extra_encoders=None,
        decoder=None,
        history_size=3,
        num_pred=1,
        interpolate_pos_encoding=True,
    ):
        super().__init__()

        self.backbone = encoder
        self.predictor = predictor
        self.extra_encoders = extra_encoders or {}
        self.decoder = decoder
        self.history_size = history_size
        self.num_pred = num_pred

        self.interpolate_pos_encoding = interpolate_pos_encoding

    def encode(
        self,
        info,
        pixels_key='pixels',
        emb_keys=None,
        prefix=None,
        target='emb',
        is_video=False,
    ):
        assert target not in info, f'{target} key already in info_dict'
        emb_keys = self.extra_encoders.keys() if emb_keys is None else emb_keys
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
        pixels_embed = self.backbone(pixels, **kwargs)

        if hasattr(pixels_embed, 'last_hidden_state'):
            pixels_embed = pixels_embed.last_hidden_state
            pixels_embed = pixels_embed[:, 1:, :]  # drop cls token
        else:
            pixels_embed = pixels_embed.logits.unsqueeze(
                1
            )  # (B*T, 1, emb_dim)

        pixels_embed = rearrange(
            pixels_embed.detach(), '(b t) p d -> b t p d', b=B
        )

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

            frame_embed = self.backbone(frames, **kwargs)  # (B, 1, P, emb_dim)
            frame_embed = frame_embed.last_hidden_state
            pixels_embeddings.append(frame_embed)

        pixels_embed = torch.stack(
            pixels_embeddings, dim=1
        )  # (B, T, P, emb_dim)

        return pixels_embed

    def predict(self, embedding):
        """predict next latent state
        Args:
            embedding: (B, T, P, d)
        Returns:
            preds: (B, T, P, d)
        """

        T = embedding.shape[1]
        embedding = rearrange(embedding, 'b t p d -> b (t p) d')
        preds = self.predictor(embedding)
        preds = rearrange(preds, 'b (t p) d -> b t p d', t=T)

        return preds

    def decode(self, info):
        assert 'pixels_emb' in info, 'pixels_emb not in info_dict'
        pixels_embed = info['pixels_emb']
        num_frames = pixels_embed.shape[1]

        pixels, diff = self.decoder(
            pixels_embed
        )  # (b*num_frames, 3, 224, 224)
        pixels = rearrange(pixels, '(b t) c h w -> b t c h w', t=num_frames)

        info['reconstructed_pixels'] = pixels
        info['reconstruction_diff'] = diff

        return info

    def split_embedding(
        self, embedding, extra_dims
    ):  # action_dim, proprio_dim):
        split_embed = {}
        pixel_dim = embedding.shape[-1] - sum(extra_dims)

        # == pixels embedding
        split_embed['pixels_emb'] = embedding[..., :pixel_dim]

        # == extra embeddings
        start_dim = pixel_dim
        for i, (key, _) in enumerate(self.extra_encoders.items()):
            dim = extra_dims[i]
            extra_emb = embedding[..., start_dim : start_dim + dim]
            split_embed[f'{key}_emb'] = extra_emb[
                :, :, :, 0
            ]  # all patches are the same
            start_dim += dim

        return split_embed

    def replace_action_in_embedding(self, embedding, act):
        """Replace the action embeddings in the latent state z with the provided actions."""
        assert 'action' in self.extra_encoders, (
            'No action encoder defined in the model.'
        )
        n_patches = embedding.shape[3]
        B, N = act.shape[:2]
        act_flat = rearrange(act, 'b n ... -> (b n) ...')
        z_act = self.extra_encoders['action'](act_flat)  # (B, T, action_dim)
        action_dim = z_act.shape[-1]
        act_tiled = repeat(
            z_act.unsqueeze(2),
            '(b n) t 1 a -> b n t p a',
            b=B,
            n=N,
            p=n_patches,
        )
        # z (B, N, T, P, d) with d = dim + extra_dims
        # determine where action starts in the embedding
        extra_dim = sum(
            encoder.emb_dim for encoder in self.extra_encoders.values()
        )
        pixel_dim = embedding.shape[-1] - extra_dim

        start = pixel_dim
        for key, encoder in self.extra_encoders.items():
            if key == 'action':
                break
            start += encoder.emb_dim

        prefix = embedding[..., :start]
        suffix = embedding[..., start + action_dim :]

        new_embedding = torch.cat([prefix, act_tiled, suffix], dim=-1)

        # embedding[..., start : start + action_dim] = act_tiled
        # return embedding
        return new_embedding

    def rollout(self, info, action_sequence):
        """Rollout the world model given an initial observation and a sequence of actions.

        Params:
        obs_start: n current observations (B, n, C, H, W)
        actions: current and predicted actions (B, n+t, action_dim)

        Returns:
        z_obs: dict with latent observations (B, n+t+1, n_patches, D)
        z: predicted latent states (B, n+t+1, n_patches, D)
        """

        assert 'pixels' in info, 'pixels not in info_dict'
        n_obs = info['pixels'].shape[2]
        emb_keys = [k for k in self.extra_encoders.keys() if k != 'action']

        # == add action to info dict
        act_0 = action_sequence[:, :, :n_obs]
        info['action'] = act_0

        # check if we have already computed the initial embedding for this state
        if (
            hasattr(self, '_init_cached_info')
            and torch.equal(self._init_cached_info['id'], info['id'][:, 0])
            and torch.equal(
                self._init_cached_info['step_idx'], info['step_idx'][:, 0]
            )
        ):
            init_info_dict = {
                k: v.detach() if torch.is_tensor(v) else v
                for k, v in self._init_cached_info.items()
            }
        else:
            # prepare init_info_dict
            init_info_dict = {}
            for k, v in info.items():
                if torch.is_tensor(v):
                    # goal is the same across samples so we will only embed it once
                    init_info_dict[k] = info[k][:, 0]  # (B, 1, ...)

            init_info_dict = self.encode(
                init_info_dict,
                pixels_key='pixels',
                target='emb',
            )
            # repeat copy for each action candidate
            init_info_dict['emb'] = (
                init_info_dict['emb']
                .unsqueeze(1)
                .expand(
                    -1,
                    action_sequence.shape[1],
                    *([-1] * (init_info_dict['emb'].ndim - 1)),
                )
                .clone()
            )
            init_info_dict['pixels_emb'] = (
                init_info_dict['pixels_emb']
                .unsqueeze(1)
                .expand(
                    -1,
                    action_sequence.shape[1],
                    *([-1] * (init_info_dict['pixels_emb'].ndim - 1)),
                )
            )

            for key in emb_keys:
                init_info_dict[f'{key}_emb'] = (
                    init_info_dict[f'{key}_emb']
                    .unsqueeze(1)
                    .expand(
                        -1,
                        action_sequence.shape[1],
                        *([-1] * (init_info_dict[f'{key}_emb'].ndim - 1)),
                    )
                )

            init_info_dict = {
                k: v.detach().clone() if torch.is_tensor(v) else v
                for k, v in init_info_dict.items()
            }
            self._init_cached_info = init_info_dict

        info['emb'] = init_info_dict['emb']
        info['pixels_emb'] = init_info_dict['pixels_emb']

        for key in emb_keys:
            info[f'{key}_emb'] = init_info_dict[f'{key}_emb']

        # actually compute the embedding of action for each candidate
        info['emb'] = self.replace_action_in_embedding(
            info['emb'], action_sequence[:, :, :n_obs]
        )

        action_dim = init_info_dict['action_emb'].shape[-1]
        info['action_emb'] = info['emb'][:, :, :n_obs, 0, -action_dim:]

        # number of step to predict
        act_pred = action_sequence[:, :, n_obs:]
        n_steps = act_pred.shape[2]

        # == initial embedding
        z = info['emb']
        B, N = z.shape[:2]

        # we flatten B and N to process all candidates in a single batch in the predictor
        z_flat = rearrange(z, 'b n ... -> (b n) ...').clone()
        act_pred_flat = rearrange(act_pred, 'b n ... -> (b n) ...')

        for t in range(n_steps):
            # predict the next state
            pred_embed = self.predict(z_flat[:, -self.history_size :])[
                :, -1:
            ]  # (B*N, 1, P, D)

            # add corresponding action to new embedding
            new_action = act_pred_flat[
                None, :, t : t + 1, :
            ]  # (1, B*N, 1, action_dim)
            new_embed = self.replace_action_in_embedding(
                pred_embed.unsqueeze(0), new_action
            )[0]

            # append new embedding to the sequence
            z_flat = torch.cat([z_flat, new_embed], dim=1)

        # predict the last state (n+t+1)
        pred_embed = self.predict(z_flat[:, -self.history_size :])[
            :, -1:
        ]  # (B, 1, P, D)
        z_flat = torch.cat([z_flat, pred_embed], dim=1)
        z = rearrange(z_flat, '(b n) ... -> b n ...', b=B, n=N)
        # == update info dict with predicted embeddings
        info['predicted_embedding'] = z

        extra_dims = []
        for key in self.extra_encoders:
            if f'{key}_emb' not in info:
                raise ValueError(f'{key}_emb not in info dict')
            extra_dims.append(info[f'{key}_emb'].shape[-1])

        splitted_embed = self.split_embedding(z, extra_dims)
        info.update({f'predicted_{k}': v for k, v in splitted_embed.items()})

        return info

    def criterion(self, info_dict: dict, action_candidates: torch.Tensor):
        """Compute the cost for planning. Should be overridden for custom costs."""
        emb_keys = [k for k in self.extra_encoders.keys() if k != 'action']
        cost = 0.0

        for key in emb_keys + ['pixels']:
            preds = info_dict[f'predicted_{key}_emb']
            goal = info_dict[f'{key}_goal_emb']
            cost = cost + F.mse_loss(
                preds[:, :, -1:], goal, reduction='none'
            ).mean(dim=tuple(range(2, preds.ndim)))
        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        assert 'action' in info_dict, 'action key must be in info_dict'
        assert 'pixels' in info_dict, 'pixels key must be in info_dict'

        # == non action embeddings keys
        emb_keys = [k for k in self.extra_encoders.keys() if k != 'action']

        # == get the goal embedding

        # check if we have already computed the goal embedding for this goal
        if (
            hasattr(self, '_goal_cached_info')
            and torch.equal(
                self._goal_cached_info['id'], info_dict['id'][:, 0]
            )
            and torch.equal(
                self._goal_cached_info['step_idx'], info_dict['step_idx'][:, 0]
            )
        ):
            goal_info_dict = {
                k: v.detach() if torch.is_tensor(v) else v
                for k, v in self._goal_cached_info.items()
            }

        else:
            # prepare goal_info_dict
            goal_info_dict = {}
            for k, v in info_dict.items():
                if torch.is_tensor(v):
                    # goal is the same across samples so we will only embed it once
                    goal_info_dict[k] = info_dict[k][:, 0]  # (B, ...)
            goal_info_dict = self.encode(
                goal_info_dict,
                target='goal_emb',
                pixels_key='goal',
                prefix='goal_',
                emb_keys=emb_keys,
            )

            goal_info_dict['goal_emb'] = (
                goal_info_dict['goal_emb']
                .unsqueeze(1)
                .expand(
                    -1,
                    action_candidates.shape[1],
                    *([-1] * (goal_info_dict['goal_emb'].ndim - 1)),
                )
            )

            goal_info_dict['pixels_goal_emb'] = (
                goal_info_dict['pixels_goal_emb']
                .unsqueeze(1)
                .expand(
                    -1,
                    action_candidates.shape[1],
                    *([-1] * (goal_info_dict['pixels_goal_emb'].ndim - 1)),
                )
            )

            for key in emb_keys:
                goal_info_dict[f'{key}_goal_emb'] = (
                    goal_info_dict[f'{key}_goal_emb']
                    .unsqueeze(1)
                    .expand(
                        -1,
                        action_candidates.shape[1],
                        *([-1] * (goal_info_dict[f'{key}_goal_emb'].ndim - 1)),
                    )
                )

            goal_info_dict = {
                k: v.detach() if torch.is_tensor(v) else v
                for k, v in goal_info_dict.items()
            }
            self._goal_cached_info = goal_info_dict

        info_dict['goal_emb'] = goal_info_dict['goal_emb']
        info_dict['pixels_goal_emb'] = goal_info_dict['pixels_goal_emb']

        for key in emb_keys:
            info_dict[f'{key}_goal_emb'] = goal_info_dict[f'{key}_goal_emb']

        # == run world model
        info_dict = self.rollout(info_dict, action_candidates)

        # cost = 0.0

        # for key in emb_keys + ["pixels"]:
        #     preds = info_dict[f"predicted_{key}_embed"]
        #     goal = info_dict[f"{key}_goal_embed"]
        #     cost = cost + F.mse_loss(preds[:, :, -1:], goal, reduction="none").mean(dim=tuple(range(2, preds.ndim)))

        return self.criterion(info_dict, action_candidates)


__all_ = ['PreJEPA']
