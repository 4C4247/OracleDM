"""SkeletonDiff v3 oracle condition datasets.

Each sample pairs a clean HR oracle image with a 4-channel condition pack
(skeleton, confidence, stroke-width, soft appearance) - see condition.py.

The geometry is taken from the clean HR (teaching ideal geometry, confidence=1)
or from a degraded upscaled LR (realistic, with a data-driven confidence). The
appearance always comes from the degraded LR. A random stroke blob is erased
(completion augmentation) to teach the model to reconstruct missing/merged
strokes at 16x16. A fraction of conditions are dropped entirely for CFG.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from condition import build_condition, apply_completion, COND_CH
from degradation import OracleDegradation


def load_gray(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def to_tensor(x):
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    t = torch.from_numpy(x).float()
    if t.ndim == 2:  # grayscale image -> add channel dim
        t = t.unsqueeze(0)
    return t


def degrade_to_lr16(hr_tensor, degrade, lr_size, deterministic=False):
    """HR tensor (1,H,W) -> LR float [0,1] array (lr_size,lr_size)."""
    with torch.no_grad():
        lr = degrade.to_lr(hr_tensor.unsqueeze(0), target_size=lr_size,
                           deterministic=deterministic)[0, 0]
    return lr.numpy()


class OracleSkeletonDataset(Dataset):
    def __init__(self, hr_dir, hr_size=128, lr_size=16, dirty_prob=0.5,
                 drop_prob=0.1, comp_prob=0.3, train=True, degrade=None,
                 seed=0):
        self.files = sorted(Path(hr_dir).glob("*.jpg"))
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.dirty_prob = dirty_prob
        self.drop_prob = drop_prob
        self.comp_prob = comp_prob
        self.train = train
        self.degrade = degrade or OracleDegradation()
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hr = load_gray(self.files[idx])
        hr = cv2.resize(hr, (self.hr_size, self.hr_size),
                        interpolation=cv2.INTER_AREA)
        x0 = to_tensor(hr)

        lr = degrade_to_lr16(x0, self.degrade, self.lr_size,
                             deterministic=not self.train)
        clean = (not self.train) or (self.rng.random() >= self.dirty_prob)
        cond = build_condition(hr, lr, self.hr_size, clean=clean)

        if self.train and self.rng.random() < self.comp_prob:
            cond = apply_completion(cond, self.rng)

        dropped = 0
        if self.train and self.rng.random() < self.drop_prob:
            cond = np.zeros_like(cond)
            dropped = 1
        return {"x0": x0, "cond": to_tensor(cond), "dropped": dropped}


def collate(batch):
    return {
        "x0": torch.stack([b["x0"] for b in batch]),
        "cond": torch.stack([b["cond"] for b in batch]),
        "dropped": torch.tensor([b["dropped"] for b in batch],
                                dtype=torch.long),
    }
