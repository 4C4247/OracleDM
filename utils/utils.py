import cv2
import numpy as np
import torch


def to_uint8(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def psnr(pred, target):
    mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
    return float((-10.0 * torch.log10(mse + 1e-10)).mean())


def ssim_batch(pred, target):
    from skimage.metrics import structural_similarity as ssim
    p = to_uint8(pred)
    t = to_uint8(target)
    vals = []
    for i in range(p.shape[0]):
        vals.append(ssim(t[i, 0], p[i, 0], data_range=255))
    return float(np.mean(vals))


def dice_batch(pred, target, thresh=0.5):
    pb = (pred >= thresh).detach().cpu().numpy()
    tb = (target >= thresh).detach().cpu().numpy()
    inter = (pb & tb).sum(axis=(1, 2, 3))
    tot = pb.sum(axis=(1, 2, 3)) + tb.sum(axis=(1, 2, 3))
    return float(np.mean(2 * inter / (tot + 1e-8)))


def make_grid(tensors, nrow=4, pad=2):
    x = to_uint8(tensors)
    b, _, h, w = x.shape
    rows = []
    for i in range(0, b, nrow):
        chunk = x[i:i + nrow, 0]
        while chunk.shape[0] < nrow:
            chunk = np.concatenate([chunk, np.zeros_like(chunk[:1])], axis=0)
        stripe = np.hstack([np.pad(c, pad, constant_values=64) for c in chunk])
        rows.append(stripe)
    return np.vstack(rows)


def save_grid(tensors, path, nrow=4):
    cv2.imwrite(path, make_grid(tensors, nrow=nrow))


def save_ckpt(path, model, config, epoch, best_psnr, optimizer=None):
    data = {
        "model": model.state_dict(),
        "config": config,
        "epoch": epoch,
        "best_psnr": best_psnr,
    }
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()
    torch.save(data, path)


def read_meta(path):
    data = torch.load(path, map_location="cpu")
    return (data.get("config", {}), data.get("epoch", 0),
            data.get("best_psnr", 0.0))


def load_ckpt(path, model, device="cpu", load_optimizer=False):
    data = torch.load(path, map_location=device)
    model.load_state_dict(data["model"])
    out = (data.get("config", {}), data.get("epoch", 0),
           data.get("best_psnr", 0.0))
    if load_optimizer:
        return out + (data.get("optimizer"),)
    return out
