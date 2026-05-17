import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# TD-MPC2 Math Utilities
# ---------------------------------------------------------------------------


def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg):
    if x.ndim == 0:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 1:
        x = x.unsqueeze(-1)

    bin_size = (cfg.wm.vmax - cfg.wm.vmin) / (cfg.wm.num_bins - 1)
    x = torch.clamp(symlog(x), cfg.wm.vmin, cfg.wm.vmax)

    indices = (x - cfg.wm.vmin) / bin_size
    bin_idx = indices.floor().long()
    bin_offset = indices - bin_idx

    bin_idx = bin_idx.clamp(0, cfg.wm.num_bins - 2)

    soft_two_hot = torch.zeros(
        x.shape[0], cfg.wm.num_bins, device=x.device, dtype=x.dtype
    )
    soft_two_hot.scatter_(1, bin_idx, 1 - bin_offset)
    soft_two_hot.scatter_(1, bin_idx + 1, bin_offset)

    return soft_two_hot


def two_hot_inv(logits, cfg):
    device = logits.device
    bin_values = torch.linspace(
        cfg.wm.vmin, cfg.wm.vmax, cfg.wm.num_bins, device=device
    )
    probs = F.softmax(logits, dim=-1)
    x = torch.sum(probs * bin_values, dim=-1, keepdim=True)
    return symexp(x)


def log_std(x, low=-10, dif=12):
    return low + 0.5 * dif * (torch.tanh(x) + 1)


def gaussian_logprob(eps, log_std):
    residual = -0.5 * eps.pow(2) - log_std
    log_prob = residual - 0.9189385175704956
    return log_prob.sum(-1, keepdim=True)


def squash(mu, pi, log_pi):
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    squashed_pi = torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
    log_pi = log_pi - squashed_pi.sum(-1, keepdim=True)
    return mu, pi, log_pi


# ---------------------------------------------------------------------------
# TD-MPC2 Building Blocks
# ---------------------------------------------------------------------------


class SimNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # The paper uses V=8 as the default simplex dimensionality
        self.simplex_dim = cfg.wm.get('simnorm_dim', 8)

    def forward(self, x):
        shp = x.shape
        # Group the last dimension into L simplices of size V
        x = x.view(*shp[:-1], -1, self.simplex_dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)


class NormedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dropout=0., act=None):
        super().__init__(in_features, out_features, bias=bias)
        self.ln = nn.LayerNorm(out_features)
        self.act = act if act is not None else nn.Mish()
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))


class RunningScale(nn.Module):
    def __init__(self, tau=0.01):
        super().__init__()
        self.tau = tau
        self.register_buffer(
            'value', torch.ones(1, dtype=torch.float32) * 10.0
        )

    def update(self, x):
        with torch.no_grad():
            percentile_95 = torch.quantile(x.detach().float(), 0.95)
            percentile_05 = torch.quantile(x.detach().float(), 0.05)
            scale_val = torch.clamp(percentile_95 - percentile_05, min=1e-4)
            self.value.data.lerp_(scale_val.unsqueeze(0), self.tau)

    def forward(self, x, update=False):
        if update:
            self.update(x)
        return x / self.value


def mlp(in_dim, mlp_dim, out_dim, act=None, dropout=0.0):
    layers = [
        NormedLinear(in_dim, mlp_dim, dropout=dropout),
        NormedLinear(mlp_dim, mlp_dim),
    ]
    if act is not None:
        layers.append(NormedLinear(mlp_dim, out_dim, act=act))
    else:
        layers.append(nn.Linear(mlp_dim, out_dim))
    return nn.Sequential(*layers)


def weight_init(m):
    """Custom weight initialization for TD-MPC2."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def zero_init(params):
    """Zero-initialize specific parameters."""
    for p in params:
        if p is not None:
            p.data.fill_(0)
