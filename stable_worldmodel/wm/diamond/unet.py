import math
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        if cond_dim is not None:
            self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
        else:
            self.cond_proj = None

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, cond=None):
        h = self.conv1(x)
        h = self.norm1(h)
        if cond is not None:
            scale_shift = self.cond_proj(cond).view(-1, 2, h.size(1), 1, 1)
            scale = 1 + scale_shift[:, 0]
            shift = scale_shift[:, 1]
            h = h * scale + shift
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h + self.skip(x)


class SimpleUNet(nn.Module):
    """A small U-Net for 2D images with conditioning vector.

    This is intentionally compact but expressive enough for experiments.
    """

    def __init__(self, in_channels=3, base_ch=64, cond_dim=256):
        super().__init__()
        self.inc = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        self.down1 = ResidualBlock(base_ch, base_ch, cond_dim)
        self.down2 = ResidualBlock(base_ch, base_ch * 2, cond_dim)
        self.down3 = ResidualBlock(base_ch * 2, base_ch * 4, cond_dim)

        self.mid = ResidualBlock(base_ch * 4, base_ch * 4, cond_dim)

        self.up3 = ResidualBlock(base_ch * 8, base_ch * 2, cond_dim)
        self.up2 = ResidualBlock(base_ch * 4, base_ch, cond_dim)
        self.up1 = ResidualBlock(base_ch * 2, base_ch, cond_dim)

        self.outc = nn.Conv2d(base_ch, in_channels, 1)

        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, cond):
        # cond: (B, cond_dim)
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


__all__ = ['SimpleUNet']
