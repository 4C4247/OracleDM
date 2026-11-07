"""Calibrated degradation for oracle characters.

Reproduces the real OBI pipeline: a clean HR image is blurred, area-downsampled
to a tiny LR, partially dilated (rubbing ink spread), given a gentle contrast
gain and brightness bias, and lightly noised. The LR is then upscaled to the
working low-resolution size (64), matching ``OBI_BlackLow_64``.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_unit_range(x: torch.Tensor) -> torch.Tensor:
    """Map a tensor to [0, 1] by its own min/max when out of range."""
    if x.min() < 0.0 or x.max() > 1.0:
        lo = x.amin(dim=(-2, -1), keepdim=True)
        hi = x.amax(dim=(-2, -1), keepdim=True)
        x = (x - lo) / (hi - lo + 1e-8)
    return x.clamp(0.0, 1.0)


def gaussian_kernel(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.view(1, 1, kernel_size, kernel_size)


class OracleDegradation(nn.Module):
    """Degradation calibrated to the real OBI LR statistics."""

    def __init__(
        self,
        blur_sigma: Tuple[float, float] = (0.5, 1.5),
        contrast: Tuple[float, float] = (0.7, 0.8),
        brightness: Tuple[float, float] = (0.15, 0.19),
        noise_std: Tuple[float, float] = (0.0, 0.02),
        dilate_blend: float = 0.4,
    ):
        super().__init__()
        self.blur_sigma = blur_sigma
        self.contrast = contrast
        self.brightness = brightness
        self.noise_std = noise_std
        self.dilate_blend = dilate_blend

    def _params(self, deterministic: bool):
        if deterministic:
            return (
                sum(self.blur_sigma) * 0.5,
                sum(self.contrast) * 0.5,
                sum(self.brightness) * 0.5,
                sum(self.noise_std) * 0.5,
            )
        return (
            torch.empty(1).uniform_(*self.blur_sigma).item(),
            torch.empty(1).uniform_(*self.contrast).item(),
            torch.empty(1).uniform_(*self.brightness).item(),
            torch.empty(1).uniform_(*self.noise_std).item(),
        )

    def to_lr(self, image: torch.Tensor, target_size: int = 16,
              deterministic: bool = False) -> torch.Tensor:
        """HR -> tiny LR (``target_size``), values in [0, 1]."""
        image = normalize_unit_range(image)
        src_size = image.shape[-1]
        scale = src_size / 256.0
        sigma, gain, bias, noise = self._params(deterministic)
        sigma = max(0.05, sigma * scale)

        k = max(5, int(6 * sigma + 1) | 1)
        image = F.conv2d(
            image, gaussian_kernel(k, sigma, image.device, image.dtype),
            padding=k // 2,
        )
        image = F.interpolate(image, size=(target_size, target_size), mode="area")

        if self.dilate_blend > 0:
            up = F.interpolate(image, scale_factor=2, mode="bilinear",
                               align_corners=False)
            up = F.max_pool2d(up, 3, stride=1, padding=1)
            dilated = F.interpolate(up, size=(target_size, target_size),
                                    mode="area")
            image = image * (1.0 - self.dilate_blend) + dilated * self.dilate_blend

        image = image * gain + bias
        if noise > 0:
            image = image + torch.randn_like(image) * noise
        return image.clamp(0.0, 1.0)

    def forward(self, image: torch.Tensor, lr_size: int = 16,
                low_size: int = 64, deterministic: bool = False) -> torch.Tensor:
        """HR -> LR (``lr_size``) -> upscaled working low-res (``low_size``)."""
        lr = self.to_lr(image, target_size=lr_size, deterministic=deterministic)
        if low_size == lr_size:
            return lr
        return F.interpolate(lr, size=(low_size, low_size), mode="bicubic",
                             align_corners=False).clamp(0.0, 1.0)
