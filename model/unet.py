"""SkeletonDiff v3 - Conditional U-Net with geometry + appearance injection.

The noisy image drives the main U-Net. Two parallel condition encoders supply
multi-scale features that are ADDED at every encoder level, the bottleneck and
every decoder level:

  geometry encoder   <- cond[:2] = (skeleton, confidence)
                         pins the stroke layout; confidence tells where the
                         skeleton is trustworthy (low => completion allowed)
  appearance encoder <- cond[2:4] = (stroke width map, soft LR appearance)
                         carries this input's thickness / ink / edge style

Input : noisy image (1ch) + 4-channel condition pack (passed separately).
Output: predicted noise (1ch).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device,
                                             dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, groups=8, dropout=0.0):
        super().__init__()
        g = min(groups, in_ch, out_ch)
        while in_ch % g or out_ch % g:
            g -= 1
        self.norm1 = nn.GroupNorm(g, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(g, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class ConvBlock(nn.Module):
    """Residual conv block without time embedding (condition encoders)."""

    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        g = min(groups, in_ch, out_ch)
        while in_ch % g or out_ch % g:
            g -= 1
        self.norm1 = nn.GroupNorm(g, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(g, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class ConditionEncoder(nn.Module):
    """Parallel encoder producing a condition feature at every resolution."""

    def __init__(self, in_ch, base_ch, ch_mult):
        super().__init__()
        self.input = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.blocks = nn.ModuleList()
        self.samples = nn.ModuleList()
        ch = base_ch
        for level, mult in enumerate(ch_mult):
            out = base_ch * mult
            self.blocks.append(ConvBlock(ch, out))
            ch = out
            if level < len(ch_mult) - 1:
                self.samples.append(Downsample(ch))
            else:
                self.samples.append(None)

    def forward(self, x):
        feats = []
        x = self.input(x)
        for block, sample in zip(self.blocks, self.samples):
            x = block(x)
            feats.append(x)
            if sample is not None:
                x = sample(x)
        return feats


class CondUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, geom_ch=2, app_ch=2,
                 base_ch=64, ch_mult=(1, 2, 2, 4), num_res_blocks=2,
                 dropout=0.0):
        super().__init__()
        self.base_ch = base_ch
        self.n_levels = len(ch_mult)
        self.geom_ch = geom_ch
        self.app_ch = app_ch
        time_dim = base_ch * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))

        # main image stream (noisy image only)
        self.input = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # two parallel condition streams
        self.geom_encoder = ConditionEncoder(geom_ch, base_ch, ch_mult)
        self.app_encoder = ConditionEncoder(app_ch, base_ch, ch_mult)
        self.geom_down = nn.ModuleList(
            [nn.Conv2d(base_ch * m, base_ch * m, 1) for m in ch_mult])
        self.app_down = nn.ModuleList(
            [nn.Conv2d(base_ch * m, base_ch * m, 1) for m in ch_mult])
        self.geom_mid = nn.Conv2d(base_ch * ch_mult[-1],
                                  base_ch * ch_mult[-1], 1)
        self.app_mid = nn.Conv2d(base_ch * ch_mult[-1],
                                 base_ch * ch_mult[-1], 1)
        self.geom_up = nn.ModuleList(
            [nn.Conv2d(base_ch * m, base_ch * m, 1) for m in reversed(ch_mult)])
        self.app_up = nn.ModuleList(
            [nn.Conv2d(base_ch * m, base_ch * m, 1) for m in reversed(ch_mult)])

        # encoder
        self.downs = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        chs = [base_ch]
        ch = base_ch
        for level, mult in enumerate(ch_mult):
            out = base_ch * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch, out, time_dim, dropout=dropout))
                ch = out
                chs.append(ch)
            self.downs.append(blocks)
            if level < len(ch_mult) - 1:
                self.down_samples.append(Downsample(ch))
                chs.append(ch)
            else:
                self.down_samples.append(None)

        self.mid1 = ResBlock(ch, ch, time_dim, dropout=dropout)
        self.mid2 = ResBlock(ch, ch, time_dim, dropout=dropout)

        # decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for level in reversed(range(self.n_levels)):
            out = base_ch * ch_mult[level]
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_ch = chs.pop()
                blocks.append(ResBlock(ch + skip_ch, out, time_dim,
                                       dropout=dropout))
                ch = out
            self.up_blocks.append(blocks)
            if level > 0:
                self.up_samples.append(Upsample(ch))
            else:
                self.up_samples.append(None)

        self.norm_out = nn.GroupNorm(8, ch)
        self.output = nn.Conv2d(ch, out_ch, 3, padding=1)

    def forward(self, x, t, cond):
        t_emb = self.time_mlp(timestep_embedding(t, self.base_ch))
        g = self.geom_encoder(cond[:, :self.geom_ch])
        a = self.app_encoder(cond[:, self.geom_ch:
                                 self.geom_ch + self.app_ch])

        # encoder
        x = self.input(x)
        hs = [x]
        for level, (blocks, sample) in enumerate(
                zip(self.downs, self.down_samples)):
            for block in blocks:
                x = block(x, t_emb)
                hs.append(x)
            x = x + self.geom_down[level](g[level]) + self.app_down[level](
                a[level])
            hs[-1] = x
            if sample is not None:
                x = sample(x)
                hs.append(x)

        # bottleneck
        x = self.mid1(x, t_emb)
        x = x + self.geom_mid(g[-1]) + self.app_mid(a[-1])
        x = self.mid2(x, t_emb)

        # decoder
        for rev, blocks in enumerate(self.up_blocks):
            level = self.n_levels - 1 - rev
            for block in blocks:
                x = torch.cat([x, hs.pop()], dim=1)
                x = block(x, t_emb)
            x = x + self.geom_up[rev](g[level]) + self.app_up[rev](a[level])
            if self.up_samples[rev] is not None:
                x = self.up_samples[rev](x)

        x = F.silu(self.norm_out(x))
        return self.output(x)
