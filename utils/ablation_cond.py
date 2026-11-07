import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from data import OracleSkeletonDataset, collate, load_gray, to_tensor
from unet import CondUNet
from diffusion import SkeletonDiffusion
from degradation import OracleDegradation
from condition import build_condition
from utils import psnr, ssim_batch, dice_batch

CONFIGS = [
    ("full",          [],  0.3),
    ("no_geom",    [0, 1],  0.3),
    ("no_app",     [2, 3],  0.3),
    ("no_conf",      [1],  0.3),
    ("no_width",     [2],  0.3),
    ("no_completion", [],  0.0),
]


def neutralize(cond, channels):
    if not channels:
        return cond
    out = cond.clone()
    for c in channels:
        out[:, c] = 0.0
    return out


def build_val_batch(files, hr_size, lr_size, degrade):
    x0, cond = [], []
    for f in files:
        hr = cv2.resize(load_gray(f), (hr_size, hr_size),
                        interpolation=cv2.INTER_AREA)
        x = to_tensor(hr)
        with torch.no_grad():
            lr = degrade.to_lr(x.unsqueeze(0), target_size=lr_size,
                               deterministic=True)[0, 0].numpy()
        pack = build_condition(hr, lr, hr_size, clean=False)
        x0.append(x)
        cond.append(to_tensor(pack))
    return torch.stack(x0), torch.stack(cond)


@torch.no_grad()
def evaluate(diff, x0, cond, steps, device, batch_size=8):
    diff.eval()
    outs = []
    for i in range(0, cond.shape[0], batch_size):
        c = cond[i:i + batch_size].to(device)
        outs.append(diff.ddim_sample(c, num_steps=steps,
                                     guidance_scale=1.0).cpu())
    pred = torch.cat(outs)
    return psnr(pred, x0), ssim_batch(pred, x0), dice_batch(pred, x0), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="/root/autodl-tmp/train")
    ap.add_argument("--val-dir", default="/root/autodl-tmp/val")
    ap.add_argument("--out-dir", default="outputs/ablation_cond")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hr-size", type=int, default=128)
    ap.add_argument("--lr-size", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=2000,
                    help="number of training images per configuration (small subset)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--sample-steps", type=int, default=25)
    ap.add_argument("--val-n", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print("bf16 autocast:", use_bf16)
    degrade = OracleDegradation()

    val_files = sorted(Path(args.val_dir).glob("*.jpg"))[:args.val_n]
    val_x0, val_cond_full = build_val_batch(
        val_files, args.hr_size, args.lr_size, degrade)

    results = []
    for tag, zero_ch, comp_prob in CONFIGS:
        print("condition ablation: %s, zeroed channels=%s, comp_prob=%.1f"
              % (tag, zero_ch if zero_ch else "none", comp_prob))

        torch.manual_seed(args.seed)
        model = CondUNet(1, 1, geom_ch=2, app_ch=2, base_ch=64,
                         ch_mult=(1, 2, 2, 4), num_res_blocks=2)
        diff = SkeletonDiffusion(model).to(device)
        opt = torch.optim.AdamW(diff.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)

        ds = OracleSkeletonDataset(
            args.train_dir, hr_size=args.hr_size, lr_size=args.lr_size,
            comp_prob=comp_prob, train=True, seed=args.seed)
        n_use = min(args.n_train, len(ds))
        sub = Subset(ds, list(range(n_use)))
        dl = DataLoader(sub, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate,
                        drop_last=True, persistent_workers=args.num_workers > 0)
        print("train subset: %d  images, %d epochs, params %.3fM"
              % (n_use, args.epochs,
                 sum(p.numel() for p in model.parameters()) / 1e6))

        for ep in range(args.epochs):
            diff.train()
            t0 = time.time()
            running, skipped, nb = 0.0, 0, 0
            pbar = tqdm(dl, desc="%s ep%d" % (tag, ep), leave=False,
                        ncols=100)
            for batch in pbar:
                x0 = batch["x0"].to(device)
                cond = neutralize(batch["cond"].to(device), zero_ch)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=use_bf16):
                    loss = diff.training_loss(x0, cond)
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diff.parameters(),
                                               args.grad_clip)
                opt.step()
                running += loss.item()
                nb += 1
                pbar.set_postfix(loss="%.4f" % (running / max(1, nb)),
                                 skip=skipped)
            pbar.close()
            print("  ep %2d loss %.4f skip %d %.1fs"
                  % (ep, running / max(1, nb), skipped, time.time() - t0),
                  flush=True)

        val_cond = neutralize(val_cond_full, zero_ch)
        p, s, d, pred = evaluate(diff, val_x0, val_cond,
                                 args.sample_steps, device)
        print("  [VAL %s] PSNR %.3f  SSIM %.3f  Dice %.3f" % (tag, p, s, d),
              flush=True)
        results.append({"config": tag, "zero_channels": zero_ch,
                        "psnr": round(p, 3), "ssim": round(s, 3),
                        "dice": round(d, 3)})

        from utils import to_uint8
        rows = []
        u = to_uint8(pred)
        u0 = to_uint8(val_x0)
        for i in range(min(8, pred.shape[0])):
            rows.append(np.hstack([u0[i, 0], u[i, 0]]))
        cv2.imwrite(str(out_dir / ("%s_grid.png" % tag)), np.vstack(rows))

        with open(out_dir / "ablation_cond_results.json", "w",
                  encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)

    print("\n================ condition ablation summary (Table 5) ================")
    print("| configuration | PSNR | SSIM | Dice |")
    print("|---|---|---|---|")
    for r in results:
        print("| %s | %.3f | %.3f | %.3f |"
              % (r["config"], r["psnr"], r["ssim"], r["dice"]))
    print("\nALL CONDITION ABLATION DONE ->", out_dir)


if __name__ == "__main__":
    main()
