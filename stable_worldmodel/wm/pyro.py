import torch
import torch.nn.functional as F

# from torchvision import transforms
from einops import rearrange, repeat
from torch import distributed as dist
from torch import nn


class PYRO(torch.nn.Module):
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
        pixels_key="pixels",
        emb_keys=None,
        prefix=None,
        target="embed",
    ):
        assert target not in info, f"{target} key already in info_dict"

        emb_keys = emb_keys or self.extra_encoders.keys()
        prefix = prefix or ""

        # == pixels embeddings
        pixels = info[pixels_key].float()  # (B, T, 3, H, W)
        B = pixels.shape[0]
        pixels = rearrange(pixels, "b t ... -> (b t) ...")

        kwargs = {"interpolate_pos_encoding": True} if self.interpolate_pos_encoding else {}
        pixels_embed = self.backbone(pixels, **kwargs)

        if hasattr(pixels_embed, "last_hidden_state"):
            pixels_embed = pixels_embed.last_hidden_state
            pixels_embed = pixels_embed[:, 1:, :]  # drop cls token
        else:
            pixels_embed = pixels_embed.logits.unsqueeze(1)  # (B*T, 1, emb_dim)

        pixels_embed = rearrange(pixels_embed.detach(), "(b t) p d -> b t p d", b=B)

        # == improve the embedding
        n_patches = pixels_embed.shape[2]
        embedding = pixels_embed
        info[f"pixels_{target}"] = pixels_embed

        for key in emb_keys:
            extr_enc = self.extra_encoders[key]
            extra_input = info[f"{prefix}{key}"].float()  # (B, T, dim)
            extra_embed = extr_enc(extra_input)  # (B, T, dim) -> (B, T, emb_dim)
            info[f"{key}_{target}"] = extra_embed

            # copy extra embedding across patches for each time step
            extra_tiled = repeat(extra_embed.unsqueeze(2), "b t 1 d -> b t p d", p=n_patches)

            # concatenate along feature dimension
            embedding = torch.cat([embedding, extra_tiled], dim=3)

        info[target] = embedding  # (B, T, P, d)

        return info

    def predict(self, embedding):
        """predict next latent state
        Args:
            embedding: (B, T, P, d)
        Returns:
            preds: (B, T, P, d)
        """

        T = embedding.shape[1]
        embedding = rearrange(embedding, "b t p d -> b (t p) d")
        preds = self.predictor(embedding)
        preds = rearrange(preds, "b (t p) d -> b t p d", t=T)

        return preds

    def decode(self, info):
        assert "pixels_embed" in info, "pixels_embed not in info_dict"
        pixels_embed = info["pixels_embed"]
        num_frames = pixels_embed.shape[1]

        pixels, diff = self.decoder(pixels_embed)  # (b*num_frames, 3, 224, 224)
        pixels = rearrange(pixels, "(b t) c h w -> b t c h w", t=num_frames)

        info["reconstructed_pixels"] = pixels
        info["reconstruction_diff"] = diff

        return info

    def split_embedding(self, embedding, extra_dims):  # action_dim, proprio_dim):
        split_embed = {}
        pixel_dim = embedding.shape[-1] - sum(extra_dims)

        # == pixels embedding
        split_embed["pixels_embed"] = embedding[..., :pixel_dim]

        # == extra embeddings
        start_dim = pixel_dim
        for i, (key, _) in enumerate(self.extra_encoders.items()):
            dim = extra_dims[i]
            extra_emb = embedding[..., start_dim : start_dim + dim]
            split_embed[f"{key}_embed"] = extra_emb[:, :, :, 0]  # all patches are the same
            start_dim += dim

        return split_embed

    def replace_action_in_embedding(self, embedding, act):
        """Replace the action embeddings in the latent state z with the provided actions."""
        assert "action" in self.extra_encoders, "No action encoder defined in the model."
        n_patches = embedding.shape[3]
        B, N = act.shape[:2]
        act_flat = rearrange(act, "b n ... -> (b n) ...")
        z_act = self.extra_encoders["action"](act_flat)  # (B, T, action_dim)
        action_dim = z_act.shape[-1]
        act_tiled = repeat(z_act.unsqueeze(2), "(b n) t 1 a -> b n t p a", b=B, n=N, p=n_patches)
        # z (B, N, T, P, d) with d = dim + extra_dims
        # determine where action starts in the embedding
        extra_dim = sum(encoder.emb_dim for encoder in self.extra_encoders.values())
        pixel_dim = embedding.shape[-1] - extra_dim

        start = pixel_dim
        for key, encoder in self.extra_encoders.items():
            if key == "action":
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

        assert "pixels" in info, "pixels not in info_dict"
        n_obs = info["pixels"].shape[2]
        emb_keys = [k for k in self.extra_encoders.keys() if k != "action"]

        # == add action to info dict
        act_0 = action_sequence[:, :, :n_obs]
        info["action"] = act_0

        # check if we have already computed the initial embedding for this state
        if (
            hasattr(self, "_init_cached_info")
            and torch.equal(self._init_cached_info["id"], info["id"][:, 0])
            and torch.equal(self._init_cached_info["step_idx"], info["step_idx"][:, 0])
        ):
            init_info_dict = {k: v.detach() if torch.is_tensor(v) else v for k, v in self._init_cached_info.items()}
        else:
            # prepare init_info_dict
            init_info_dict = {}
            for k, v in info.items():
                if torch.is_tensor(v):
                    # goal is the same across samples so we will only embed it once
                    init_info_dict[k] = info[k][:, 0]  # (B, 1, ...)

            init_info_dict = self.encode(
                init_info_dict,
                pixels_key="pixels",
                target="embed",
            )
            # repeat copy for each action candidate
            init_info_dict["embed"] = (
                init_info_dict["embed"]
                .unsqueeze(1)
                .expand(-1, action_sequence.shape[1], *([-1] * (init_info_dict["embed"].ndim - 1)))
                .clone()
            )
            init_info_dict["pixels_embed"] = (
                init_info_dict["pixels_embed"]
                .unsqueeze(1)
                .expand(-1, action_sequence.shape[1], *([-1] * (init_info_dict["pixels_embed"].ndim - 1)))
            )

            for key in emb_keys:
                init_info_dict[f"{key}_embed"] = (
                    init_info_dict[f"{key}_embed"]
                    .unsqueeze(1)
                    .expand(-1, action_sequence.shape[1], *([-1] * (init_info_dict[f"{key}_embed"].ndim - 1)))
                )

            init_info_dict = {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in init_info_dict.items()}
            self._init_cached_info = init_info_dict

        info["embed"] = init_info_dict["embed"]
        info["pixels_embed"] = init_info_dict["pixels_embed"]

        for key in emb_keys:
            info[f"{key}_embed"] = init_info_dict[f"{key}_embed"]

        # actually compute the embedding of action for each candidate
        info["embed"] = self.replace_action_in_embedding(info["embed"], action_sequence[:, :, :n_obs])
        # action_dim = init_info_dict["action_embed"].shape[-1]
        info["action_embed"] = action_sequence[:, :, :n_obs]  # info["embed"][:, :, :n_obs, 0, -action_dim:]

        # number of step to predict
        act_pred = action_sequence[:, :, n_obs:]
        n_steps = act_pred.shape[2]

        # == initial embedding
        z = info["embed"]
        B, N = z.shape[:2]

        # we flatten B and N to process all candidates in a single batch in the predictor
        z_flat = rearrange(z, "b n ... -> (b n) ...").clone()
        act_pred_flat = rearrange(act_pred, "b n ... -> (b n) ...")

        for t in range(n_steps):
            # predict the next state
            pred_embed = self.predict(z_flat[:, -self.history_size :])[:, -1:]  # (B*N, 1, P, D)

            # add corresponding action to new embedding
            new_action = act_pred_flat[None, :, t : t + 1, :]  # (1, B*N, 1, action_dim)
            new_embed = self.replace_action_in_embedding(pred_embed.unsqueeze(0), new_action)[0]

            # append new embedding to the sequence
            z_flat = torch.cat([z_flat, new_embed], dim=1)

        # predict the last state (n+t+1)
        pred_embed = self.predict(z_flat[:, -self.history_size :])[:, -1:]  # (B, 1, P, D)
        z_flat = torch.cat([z_flat, pred_embed], dim=1)
        z = rearrange(pred_embed, "(b n) ... -> b n ...", b=B, n=N)

        # == update info dict with predicted embeddings
        info["predicted_embedding"] = z

        extra_dims = []
        for key in self.extra_encoders:
            if f"{key}_embed" not in info:
                raise ValueError(f"{key}_embed not in info dict")
            extra_dims.append(info[f"{key}_embed"].shape[-1])

        splitted_embed = self.split_embedding(z, extra_dims)
        info.update({f"predicted_{k}": v for k, v in splitted_embed.items()})

        return info

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        assert "action" in info_dict, "action key must be in info_dict"
        assert "pixels" in info_dict, "pixels key must be in info_dict"

        # move to device and unsqueeze time
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                info_dict[k] = v.to(next(self.parameters()).device)

        # == non action embeddings keys
        emb_keys = [k for k in self.extra_encoders.keys() if k != "action"]

        # == get the goal embedding

        # check if we have already computed the goal embedding for this goal
        if (
            hasattr(self, "_goal_cached_info")
            and torch.equal(self._goal_cached_info["id"], info_dict["id"][:, 0])
            and torch.equal(self._goal_cached_info["step_idx"], info_dict["step_idx"][:, 0])
        ):
            goal_info_dict = {k: v.detach() if torch.is_tensor(v) else v for k, v in self._goal_cached_info.items()}

        else:
            # prepare goal_info_dict
            goal_info_dict = {}
            for k, v in info_dict.items():
                if torch.is_tensor(v):
                    # goal is the same across samples so we will only embed it once
                    goal_info_dict[k] = info_dict[k][:, 0]  # (B, ...)
            goal_info_dict = self.encode(
                goal_info_dict,
                target="goal_embed",
                pixels_key="goal",
                prefix="goal_",
                emb_keys=emb_keys,
            )

            goal_info_dict["goal_embed"] = (
                goal_info_dict["goal_embed"]
                .unsqueeze(1)
                .expand(-1, action_candidates.shape[1], *([-1] * (goal_info_dict["goal_embed"].ndim - 1)))
            )

            goal_info_dict["pixels_goal_embed"] = (
                goal_info_dict["pixels_goal_embed"]
                .unsqueeze(1)
                .expand(-1, action_candidates.shape[1], *([-1] * (goal_info_dict["pixels_goal_embed"].ndim - 1)))
            )

            for key in emb_keys:
                goal_info_dict[f"{key}_goal_embed"] = (
                    goal_info_dict[f"{key}_goal_embed"]
                    .unsqueeze(1)
                    .expand(-1, action_candidates.shape[1], *([-1] * (goal_info_dict[f"{key}_goal_embed"].ndim - 1)))
                )

            goal_info_dict = {k: v.detach() if torch.is_tensor(v) else v for k, v in goal_info_dict.items()}
            self._goal_cached_info = goal_info_dict

        info_dict["goal_embed"] = goal_info_dict["goal_embed"]
        info_dict["pixels_goal_embed"] = goal_info_dict["pixels_goal_embed"]

        for key in emb_keys:
            info_dict[f"{key}_goal_embed"] = goal_info_dict[f"{key}_goal_embed"]

        # == run world model
        info_dict = self.rollout(info_dict, action_candidates)

        cost = 0.0

        for key in emb_keys + ["pixels"]:
            preds = info_dict[f"predicted_{key}_embed"]
            goal = info_dict[f"{key}_goal_embed"]
            cost = cost + F.mse_loss(preds[:, :, -1:], goal, reduction="none").mean(dim=tuple(range(2, preds.ndim)))

        # TODO try with the same loss computation as the training

        return cost


class Embedder(torch.nn.Module):
    def __init__(
        self,
        num_frames=1,
        tubelet_size=1,
        in_chans=8,
        emb_dim=10,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.emb_dim = emb_dim
        self.patch_embed = torch.nn.Conv1d(in_chans, emb_dim, kernel_size=tubelet_size, stride=tubelet_size)

    def forward(self, x):
        with torch.amp.autocast(enabled=False, device_type=x.device.type):
            x = x.permute(0, 2, 1)  # (B, T, B) -> (B, D, T)
            x = self.patch_embed(x)
            x = x.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)
        return x


class CausalPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert pool in {"cls", "mean"}, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.num_patches = num_patches
        self.num_frames = num_frames

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * (num_patches), dim))  # dim for the pos encodings
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, num_frames)
        self.pool = pool

    def forward(self, x):  # x: (b, window_size * H/patch_size * W/patch_size, 384)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, num_patches=1, num_frames=1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

        self.register_buffer("bias", self.generate_mask_matrix(num_patches, num_frames))

    def forward(self, x):
        B, T, C = x.size()
        x = self.norm(x)

        # q, k, v: (B, heads, T, dim_head)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)

        attn_mask = self.bias[:, :, :T, :T] == 1  # bool mask

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False
        )

        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)

    def generate_mask_matrix(self, npatch, nwindow):
        zeros = torch.zeros(npatch, npatch)
        ones = torch.ones(npatch, npatch)
        rows = []
        for i in range(nwindow):
            row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
            rows.append(row)
        mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
        return mask


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        num_patches=1,
        num_frames=1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            num_patches=num_patches,
                            num_frames=num_frames,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def all_reduce(tensor, op=dist.ReduceOp.SUM):
    world_size = get_world_size()

    if world_size == 1:
        return tensor

    dist.all_reduce(tensor, op=op)

    return tensor


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = flatten.pow(2).sum(1, keepdim=True) - 2 * flatten @ self.embed + self.embed.pow(2).sum(0, keepdim=True)
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            all_reduce(embed_onehot_sum)
            all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
                ]
            )

        elif stride == 2:
            blocks.append(nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        emb_dim=64,
        n_embed=512,
        decay=0.99,
        quantize=True,
    ):
        super().__init__()

        self.quantize = quantize
        self.quantize_b = Quantize(emb_dim, n_embed)

        if not quantize:
            for param in self.quantize_b.parameters():
                param.requires_grad = False

        self.upsample_b = Decoder(emb_dim, emb_dim, channel, n_res_block, n_res_channel, stride=4)
        self.dec = Decoder(
            emb_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )
        self.info = f"in_channel: {in_channel}, channel: {channel}, n_res_block: {n_res_block}, n_res_channel: {n_res_channel}, emb_dim: {emb_dim}, n_embed: {n_embed}, decay: {decay}"

    def forward(self, input):
        """
        input: (b, t, num_patches, emb_dim)
        """
        num_patches = input.shape[2]
        num_side_patches = int(num_patches**0.5)
        input = rearrange(input, "b t (h w) e -> (b t) h w e", h=num_side_patches, w=num_side_patches)

        if self.quantize:
            quant_b, diff_b, id_b = self.quantize_b(input)
        else:
            quant_b, diff_b = input, torch.zeros(1).to(input.device)

        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        dec = self.decode(quant_b)
        return dec, diff_b  # diff is 0 if no quantization

    def decode(self, quant_b):
        upsample_b = self.upsample_b(quant_b)
        dec = self.dec(upsample_b)  # quant: (128, 64, 64)
        return dec

    def decode_code(self, code_b):  # not used (only used in sample.py in original repo)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        dec = self.decode(quant_b)
        return dec
