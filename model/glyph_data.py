import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _drop_grain(bin_img, min_comp_frac=0.0025):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bin_img, 8)
    if n <= 1:
        return bin_img
    total = bin_img.shape[0] * bin_img.shape[1]
    min_area = max(8.0, min_comp_frac * total)
    out = np.zeros_like(bin_img)
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] >= min_area:
            out[lab == c] = 255
    return out


def _drop_solid_border(bin_img, margin=2, dens_thr=0.62):
    h, w = bin_img.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bin_img, 8)
    if n <= 1:
        return bin_img
    bad = set()
    for c in range(1, n):
        x = stats[c, cv2.CC_STAT_LEFT]
        y = stats[c, cv2.CC_STAT_TOP]
        cw = stats[c, cv2.CC_STAT_WIDTH]
        ch = stats[c, cv2.CC_STAT_HEIGHT]
        a = stats[c, cv2.CC_STAT_AREA]
        touches = (x <= margin or y <= margin or
                   x + cw >= w - margin or y + ch >= h - margin)
        dens = a / float(cw * ch)
        if touches and dens > dens_thr:
            bad.add(c)
    if not bad:
        return bin_img
    out = np.zeros_like(bin_img)
    for c in range(1, n):
        if c not in bad:
            out[lab == c] = 255
    return out


def glyph_quality(norm, dens_thr=0.62, min_fill=0.01, max_fill=0.42):
    fg = (norm > 0)
    fill = fg.mean()
    if not (min_fill <= fill <= max_fill):
        return False
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    if n <= 1:
        return False
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, a = stats[big]
    density = a / float(w * h)
    return density <= dens_thr


def _bbox_gap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    gx = max(0, max(ax, bx) - min(ax + aw, bx + bw))
    gy = max(0, max(ay, by) - min(ay + ah, by + bh))
    return max(gx, gy)


def _glyph_cluster(bin_img, radius_frac=0.28):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bin_img, 8)
    if n <= 2:
        return bin_img
    anchor = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    h, w = bin_img.shape
    radius = radius_frac * max(h, w)
    abox = (stats[anchor, cv2.CC_STAT_LEFT], stats[anchor, cv2.CC_STAT_TOP],
            stats[anchor, cv2.CC_STAT_WIDTH], stats[anchor, cv2.CC_STAT_HEIGHT])
    out = np.zeros_like(bin_img)
    for c in range(1, n):
        cbox = (stats[c, cv2.CC_STAT_LEFT], stats[c, cv2.CC_STAT_TOP],
                stats[c, cv2.CC_STAT_WIDTH], stats[c, cv2.CC_STAT_HEIGHT])
        if c == anchor or _bbox_gap(abox, cbox) <= radius:
            out[lab == c] = 255
    return out


def normalize_glyph(gray, size=128, frac=0.86, despeckle=True):
    g = gray
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float32)

    if np.median(g) > 127:
        g = 255.0 - g

    gu = np.clip(g, 0, 255).astype(np.uint8)
    _, b = cv2.threshold(gu, 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if despeckle:
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (3, 3)))
        b = _drop_solid_border(b)
        b = _drop_grain(b)
        b = _glyph_cluster(b)

    ys, xs = np.where(b > 0)
    if len(xs) == 0:
        b = gu
        ys, xs = np.where(b > 0)
        if len(xs) == 0:
            return np.zeros((size, size), np.uint8)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = b[y0:y1 + 1, x0:x1 + 1]
    h, w = crop.shape[:2]
    inner = int(round(size * frac))
    scale = inner / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((size, size), np.uint8)
    y = (size - nh) // 2
    x = (size - nw) // 2
    canvas[y:y + nh, x:x + nw] = crop
    return canvas


def _code_from_path(p):
    name = p.replace("\\", "/")
    for part in name.split("/"):
        if len(part) == 6 and part.isdigit():
            return part
    return None


def scan_files(root):
    root = Path(root)
    out = []
    for f in sorted(root.rglob("*.bmp")):
        code = _code_from_path(str(f))
        if code:
            out.append((f, code))
    return out


def handprint_files(root):
    return scan_files(root)


def build_index(handprint_root=None, scan_root=None, seed=42,
                val_frac=0.15):
    items = []
    if scan_root:
        for f, c in scan_files(scan_root):
            items.append((f, c, "scan"))
    if handprint_root:
        for f, c in handprint_files(handprint_root):
            items.append((f, c, "hand"))

    codes = sorted({c for _, c, _ in items})
    code2idx = {c: i for i, c in enumerate(codes)}

    rng = random.Random(seed)
    by_class = {}
    for it in items:
        by_class.setdefault(it[1], []).append(it)
    train, val = [], []
    for c in sorted(by_class):
        group = by_class[c]
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * val_frac)))
        for it in group[:n_val]:
            val.append((str(it[0]), code2idx[it[1]], it[2]))
        for it in group[n_val:]:
            train.append((str(it[0]), code2idx[it[1]], it[2]))
    return code2idx, train, val


def filter_valid(items, size=128, verbose=False):
    kept = []
    n_bad = 0
    n_empty = 0
    for i, (path, label, domain) in enumerate(items):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            n_empty += 1
            continue
        norm = normalize_glyph(gray, size=size)
        if not glyph_quality(norm):
            n_bad += 1
            continue
        kept.append((path, label, domain))
        if verbose and (i + 1) % 5000 == 0:
            print("  filtered %d/%d (kept %d)" % (i + 1, len(items), len(kept)))
    if verbose:
        print("filter_valid: kept %d, rejected %d (bad %d, unreadable %d)"
              % (len(kept), len(items) - len(kept), n_bad, n_empty))
    return kept


class GlyphDataset(Dataset):
    def __init__(self, items, size=128, train=False):
        self.items = items
        self.size = size
        self.train = train

    def __len__(self):
        return len(self.items)

    def _augment(self, img):
        s = self.size
        ang = random.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((s / 2, s / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (s, s), flags=cv2.INTER_NEAREST,
                             borderValue=0)
        sc = random.uniform(0.9, 1.1)
        nw = nh = int(round(s * sc))
        zoom = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
        if nw >= s:
            x0 = (nw - s) // 2
            y0 = (nh - s) // 2
            img = zoom[y0:y0 + s, x0:x0 + s]
        else:
            canvas = np.zeros((s, s), np.uint8)
            x0 = (s - nw) // 2 + random.randint(-6, 6)
            y0 = (s - nh) // 2 + random.randint(-6, 6)
            x0 = int(np.clip(x0, 0, s - nw))
            y0 = int(np.clip(y0, 0, s - nh))
            canvas[y0:y0 + nh, x0:x0 + nw] = zoom
            img = canvas
        if random.random() < 0.3:
            k = random.choice([2, 3])
            if random.random() < 0.5:
                img = cv2.dilate(img, np.ones((k, k), np.uint8))
            else:
                img = cv2.erode(img, np.ones((k, k), np.uint8))
        if random.random() < 0.3:
            m = (np.random.rand(s, s) > 0.992).astype(np.uint8) * 255
            img = cv2.bitwise_or(img, m)
        return img

    def __getitem__(self, i):
        path, label, domain = self.items[i]
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = normalize_glyph(gray, size=self.size)
        if self.train:
            img = self._augment(img)
        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        return x, int(label), domain
