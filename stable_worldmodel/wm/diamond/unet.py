import math
import torch
from torch import nn


class AdaptiveGroupNorm(nn.Module):
    """Adaptive GroupNorm: group-norm then affine modulation from cond vector."""

    def __init__(self, num_groups, num_channels, cond_dim=None):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.proj = nn.Linear(cond_dim, num_channels * 2)
        else:
            self.proj = None

    def forward(self, x, cond=None):
        # x: (B, C, H, W)
        h = self.gn(x)
        if self.proj is not None and cond is not None:
            # cond: (B, cond_dim)
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
    """A U-Net for 2D images with conditioning vector and AdaptiveGroupNorm.

    Parameters:
    - in_channels: input image channels
    - out_channels: output image channels (defaults to in_channels)
    - base_ch: base number of channels
    - cond_dim: conditioning vector dimension
    - num_groups: number of groups for AdaptiveGroupNorm
    """

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
