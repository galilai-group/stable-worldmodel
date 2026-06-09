import math
import torch
from torch import nn
import torch.nn.functional as F


__all__ = [
    'AdaptiveGroupNorm',
    'ResidualBlock',
    'SimpleUNet',
    'sigma_sampling',
    'c_preconditioners',
    'EDMModel',
    'make_sigma_schedule',
    'sample_euler',
    'sample_heun',
    'edm_loss_step',
    'example_train_step',
    'DiffusionPredictor',
    'DiscreteActionEncoder',
    'RewardTerminationHead',
    'ConvEncoder',
    'RewardTermModel',
    'ActorCritic',
    'example_infer',
    'example_train_step_wrapper',
]


# ── U-Net components ─────────────────────────────────────────────────────────


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, cond_dim=None):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.proj = nn.Linear(cond_dim, num_channels * 2)
        else:
            self.proj = None

    def forward(self, x, cond=None):
        h = self.gn(x)
        if self.proj is not None and cond is not None:
            ss = self.proj(cond).view(-1, 2, x.size(1), 1, 1)
            scale = 1 + ss[:, 0]
            shift = ss[:, 1]
            h = h * scale + shift
        return h


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim=None, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.norm1 = AdaptiveGroupNorm(num_groups, out_ch, cond_dim)
        self.norm2 = AdaptiveGroupNorm(num_groups, out_ch, cond_dim)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, cond=None):
        h = self.conv1(x)
        h = self.norm1(h, cond)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h, cond)
        h = self.act(h)
        return h + self.skip(x)


class SimpleUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=None,
        base_ch=96,
        cond_dim=256,
        num_groups=8,
    ):
        super().__init__()
        out_channels = (
            out_channels if out_channels is not None else in_channels
        )
        self.inc = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        self.down1 = ResidualBlock(
            base_ch, base_ch, cond_dim, num_groups=num_groups
        )
        self.down2 = ResidualBlock(
            base_ch, base_ch * 2, cond_dim, num_groups=num_groups
        )
        self.down3 = ResidualBlock(
            base_ch * 2, base_ch * 4, cond_dim, num_groups=num_groups
        )

        self.mid = ResidualBlock(
            base_ch * 4, base_ch * 4, cond_dim, num_groups=num_groups
        )

        self.up3 = ResidualBlock(
            base_ch * 8, base_ch * 2, cond_dim, num_groups=num_groups
        )
        self.up2 = ResidualBlock(
            base_ch * 4, base_ch, cond_dim, num_groups=num_groups
        )
        self.up1 = ResidualBlock(
            base_ch * 2, base_ch, cond_dim, num_groups=num_groups
        )

        self.outc = nn.Conv2d(base_ch, out_channels, 1)

        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, cond):
        x1 = self.inc(x)
        d1 = self.down1(x1, cond)
        p1 = self.pool(d1)
        d2 = self.down2(p1, cond)
        p2 = self.pool(d2)
        d3 = self.down3(p2, cond)

        m = self.mid(self.pool(d3), cond)

        u3 = self.upsample(m)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.up3(u3, cond)
        u2 = self.upsample(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up2(u2, cond)
        u1 = self.upsample(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up1(u1, cond)

        out = self.outc(u1)
        return out


# ── EDM preconditioning ──────────────────────────────────────────────────────


def sigma_sampling(batch_size, device, mean=-0.4, std=1.2):
    return torch.exp(torch.randn(batch_size, device=device) * std + mean)


def c_preconditioners(sigma, sigma_data=0.5):
    sigma1 = sigma.view(-1, 1, 1, 1)
    c_in = 1.0 / (sigma1**2 + sigma_data**2).sqrt()
    c_out = (sigma1 * sigma_data) / (sigma1**2 + sigma_data**2).sqrt()
    c_skip = (sigma_data**2) / (sigma_data**2 + sigma1**2)
    c_noise = 0.25 * torch.log(sigma1.squeeze())
    return c_in, c_out, c_skip, c_noise


class EDMModel(nn.Module):
    def __init__(
        self,
        in_ch=3,
        base_ch=64,
        cond_dim=256,
        sigma_data=0.5,
        history_frames=4,
    ):
        super().__init__()
        self.unet = SimpleUNet(
            in_channels=in_ch * (history_frames + 1),
            out_channels=in_ch,
            base_ch=base_ch,
            cond_dim=cond_dim + 1,
        )
        self.sigma_data = sigma_data

    def forward(self, x_noisy, sigma, cond):
        c_in, c_out, c_skip, c_noise = c_preconditioners(
            sigma, self.sigma_data
        )

        x_in = c_in * x_noisy
        history = cond.get('history', None)
        if history is not None:
            inp = torch.cat([x_in, history], dim=1)
        else:
            pad_channels = self.unet.inc.in_channels - x_in.size(1)
            if pad_channels > 0:
                pad = torch.zeros_like(x_in[:, :1]).expand(
                    -1, pad_channels, -1, -1
                )
                inp = torch.cat([x_in, pad], dim=1)
            else:
                inp = x_in

        cond_vec = cond.get('cond_vec', None)
        if cond_vec is not None and isinstance(cond_vec, torch.Tensor):
            c_noise_vec = c_noise.view(-1, 1).expand(cond_vec.size(0), -1)
            cond_vec = torch.cat([cond_vec, c_noise_vec], dim=-1)
        out = self.unet(inp, cond_vec)
        denoised = c_out * out + c_skip * x_noisy
        return denoised

    def predict(self, x_noisy, sigma, cond):
        return self.forward(x_noisy, sigma, cond)

    def score(self, x_noisy, sigma, cond):
        x0_hat = self.predict(x_noisy, sigma, cond)
        sigma1 = sigma.view(-1, 1, 1, 1)
        score = (x0_hat - x_noisy) / (sigma1**2)
        return score


# ── EDM sampling ─────────────────────────────────────────────────────────────


def make_sigma_schedule(sigma_max, sigma_min, n_steps):
    if n_steps == 1:
        return [sigma_min]
    return [
        sigma_max * (sigma_min / sigma_max) ** (i / (n_steps - 1))
        for i in range(n_steps)
    ]


def sample_euler(
    model, cond, shape, device, n_steps=3, sigma_max=1.0, sigma_min=1e-3
):
    schedule = make_sigma_schedule(sigma_max, sigma_min, n_steps)
    x = torch.randn(*shape, device=device) * schedule[0]

    for i in range(len(schedule) - 1):
        sigma = torch.tensor(schedule[i], device=device)
        sigma_next = torch.tensor(schedule[i + 1], device=device)
        x0 = model.predict(x, sigma, cond)
        x = x0 + (sigma_next / sigma) * (x - x0)

    x_final = model.predict(x, torch.tensor(schedule[-1], device=device), cond)
    return x_final


def sample_heun(
    model, cond, shape, device, n_steps=3, sigma_max=1.0, sigma_min=1e-3
):
    schedule = make_sigma_schedule(sigma_max, sigma_min, n_steps)
    x = torch.randn(*shape, device=device) * schedule[0]

    for i in range(len(schedule) - 1):
        s = schedule[i]
        s_next = schedule[i + 1]
        sigma = torch.tensor(s, device=device)
        sigma_next = torch.tensor(s_next, device=device)

        score1 = model.score(x, sigma, cond)
        x_euler = x - (s_next - s) * score1

        score2 = model.score(x_euler, sigma_next, cond)
        x = x - 0.5 * (s_next - s) * (score1 + score2)

    x_final = model.predict(x, torch.tensor(schedule[-1], device=device), cond)
    return x_final


# ── EDM training loss ────────────────────────────────────────────────────────


def edm_loss_step(model, batch, device):
    B = batch['next_frame'].shape[0]
    sigma = sigma_sampling(B, device)
    sigma_t = sigma.to(device)

    noise = torch.randn_like(batch['next_frame'], device=device)
    x_noisy = batch['next_frame'] + noise * sigma_t.view(-1, 1, 1, 1)

    cond = {
        'history': batch.get('history', None),
        'cond_vec': batch.get('cond_vec', None),
    }

    x0_hat = model.predict(x_noisy, sigma_t, cond)

    c_in, c_out, c_skip, c_noise = c_preconditioners(sigma_t)

    target = (
        batch['next_frame'] - c_skip.view(-1, 1, 1, 1) * x_noisy
    ) / c_out.view(-1, 1, 1, 1)

    loss = F.mse_loss(x0_hat, target)
    return loss


def example_train_step(model, optimizer, batch, device):
    model.train()
    loss = edm_loss_step(model, batch, device)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ── Diffusion / action encoders ──────────────────────────────────────────────


class DiffusionPredictor(nn.Module):
    def __init__(self, emb_dim, act_emb_dim, hidden_dim=512, depth=3):
        super().__init__()
        self.emb_dim = emb_dim
        self.act_emb_dim = act_emb_dim
        self.hidden_dim = hidden_dim

        layers = []
        in_dim = emb_dim + act_emb_dim
        for i in range(depth - 1):
            layers.append(
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim)
            )
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim if depth > 1 else in_dim, emb_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, emb, act_emb):
        x = torch.cat([emb, act_emb], dim=-1)
        B, T, _ = x.shape
        x = x.view(B * T, -1)
        out = self.net(x)
        out = out.view(B, T, self.emb_dim)
        return out


class DiscreteActionEncoder(nn.Module):
    def __init__(self, num_actions, emb_dim=64):
        super().__init__()
        self.embed = nn.Embedding(num_actions, emb_dim)

    def forward(self, x):
        return self.embed(x.squeeze(-1).long())


class RewardTerminationHead(nn.Module):
    def __init__(self, emb_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.reward_head = nn.Linear(hidden, 1)
        self.terminal_head = nn.Linear(hidden, 1)

    def forward(self, emb):
        B, T, D = emb.shape
        x = emb.view(B * T, D)
        h = self.net(x)
        reward = self.reward_head(h).view(B, T, 1)
        terminal_logits = self.terminal_head(h).view(B, T, 1)
        return reward, terminal_logits


# ── ConvEncoder, RewardTermModel, ActorCritic ────────────────────────────────


class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        block_channels=(32, 32, 64, 64),
        block_layers=(1, 1, 1, 1),
        cond_dim=None,
        num_groups=8,
    ):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, block_channels[0], 3, padding=1)

        stages = []
        prev_ch = block_channels[0]
        for out_ch, n_layers in zip(block_channels, block_layers):
            stage_blocks = []
            for _ in range(n_layers):
                stage_blocks.append(
                    ResidualBlock(prev_ch, out_ch, cond_dim, num_groups)
                )
                prev_ch = out_ch
            stages.append(nn.ModuleList(stage_blocks))
        self.stages = nn.ModuleList(stages)
        self.pool = nn.MaxPool2d(2)
        self.out_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, cond=None):
        h = self.in_conv(x)
        for stage in self.stages:
            for block in stage:
                h = block(h, cond)
            h = self.pool(h)
        h = self.out_pool(h)
        return h.flatten(1)


class RewardTermModel(nn.Module):
    def __init__(
        self,
        in_channels=3,
        action_dim=18,
        cond_dim=128,
        lstm_dim=512,
        num_groups=8,
    ):
        super().__init__()
        self.action_embed = nn.Embedding(action_dim, cond_dim)
        block_channels = (32, 32, 32, 32)
        self.encoder = ConvEncoder(
            in_channels=in_channels,
            block_channels=block_channels,
            block_layers=(2, 2, 2, 2),
            cond_dim=cond_dim,
            num_groups=num_groups,
        )
        enc_dim = block_channels[-1]
        self.lstm = nn.LSTMCell(enc_dim + cond_dim, lstm_dim)
        self.reward_head = nn.Linear(lstm_dim, 1)
        self.term_head = nn.Linear(lstm_dim, 1)

    def forward(self, frames, actions, state=None):
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        actions = actions.view(B * T, -1)
        act_emb = self.action_embed(actions.squeeze(-1).long())
        vis_emb = self.encoder(frames, act_emb)
        x = torch.cat([vis_emb, act_emb], dim=-1)

        hx = torch.zeros(B, self.lstm.hidden_size, device=frames.device)
        cx = torch.zeros(B, self.lstm.hidden_size, device=frames.device)
        if state is not None:
            hx, cx = state

        rewards = []
        terminals = []
        for t in range(T):
            inp = x[t::T] if T > 1 else x
            hx, cx = self.lstm(inp, (hx, cx))
            rewards.append(self.reward_head(hx))
            terminals.append(self.term_head(hx))

        rew = torch.stack(rewards, dim=1)
        term = torch.stack(terminals, dim=1)
        return rew, term, (hx, cx)


class ActorCritic(nn.Module):
    def __init__(
        self,
        in_channels=3,
        action_dim=18,
        lstm_dim=512,
        num_groups=8,
    ):
        super().__init__()
        self.action_dim = action_dim
        block_channels = (32, 32, 64, 64)
        self.encoder = ConvEncoder(
            in_channels=in_channels,
            block_channels=block_channels,
            block_layers=(1, 1, 1, 1),
            cond_dim=None,
            num_groups=num_groups,
        )
        enc_dim = block_channels[-1]
        self.lstm = nn.LSTMCell(enc_dim, lstm_dim)
        self.policy_head = nn.Linear(lstm_dim, action_dim)
        self.value_head = nn.Linear(lstm_dim, 1)

    def forward(self, frame, state=None):
        B, C, H, W = frame.shape
        vis_emb = self.encoder(frame)
        hx = torch.zeros(B, self.lstm.hidden_size, device=frame.device)
        cx = torch.zeros(B, self.lstm.hidden_size, device=frame.device)
        if state is not None:
            hx, cx = state
        hx, cx = self.lstm(vis_emb, (hx, cx))
        logits = self.policy_head(hx)
        value = self.value_head(hx)
        return logits, value, (hx, cx)

    def get_action(self, frame, state=None, deterministic=False):
        logits, value, state = self.forward(frame, state)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        return action, value, state, logits


# ── Example helpers ──────────────────────────────────────────────────────────


def example_infer(
    model: EDMModel, history, cond_vec, device='cpu', method='euler'
):
    model.eval()
    B = history.shape[0]
    cond = {'history': history, 'cond_vec': cond_vec}
    shape = (B, 3, history.shape[2], history.shape[3])

    if method == 'euler':
        out = sample_euler(model, cond, shape, device)
    else:
        out = sample_heun(model, cond, shape, device)

    return out


def example_train_step_wrapper(model, optimizer, batch, device='cpu'):
    return example_train_step(model, optimizer, batch, device)
