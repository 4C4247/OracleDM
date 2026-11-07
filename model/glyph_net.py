import torch
import torch.nn as nn


def gn(c):
    return nn.GroupNorm(min(8, c), c)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1,
                               bias=False)
        self.norm1 = gn(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.norm2 = gn(cout)
        self.act = nn.SiLU()
        if stride != 1 or cin != cout:
            self.skip = nn.Conv2d(cin, cout, 1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class GlyphClassifier(nn.Module):

    def __init__(self, num_classes=241, feat_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), gn(32), nn.SiLU())
        self.block1 = ConvBlock(32, 32, stride=1)
        self.down1 = ConvBlock(32, 64, stride=2)
        self.block2 = ConvBlock(64, 64, stride=1)
        self.down2 = ConvBlock(64, 128, stride=2)
        self.block3 = ConvBlock(128, 128, stride=1)
        self.down3 = ConvBlock(128, 256, stride=2)
        self.block4 = ConvBlock(256, 256, stride=1)
        self.down4 = ConvBlock(256, feat_dim, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.SiLU(),
            nn.Linear(feat_dim, num_classes))
        self.feat_dim = feat_dim

    def feature_maps(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        f3 = self.block3(x)
        x = self.down3(f3)
        f4 = self.block4(x)
        x = self.down4(f4)
        return f3, f4, x

    def forward(self, x):
        f3, f4, f8 = self.feature_maps(x)
        p = self.pool(f8).flatten(1)
        return self.head(p)

    def features(self, x):
        f3, f4, f8 = self.feature_maps(x)
        p = self.pool(f8).flatten(1)
        return [f3, f4, p]


def build_glyph_classifier(num_classes=241, ckpt=None, device="cpu"):
    net = GlyphClassifier(num_classes=num_classes)
    if ckpt:
        sd = torch.load(ckpt, map_location=device)
        state = sd.get("model", sd)
        net.load_state_dict(state)
    net.to(device)
    return net
