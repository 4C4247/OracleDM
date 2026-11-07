import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from data import load_gray, to_tensor
from unet import CondUNet
from diffusion import SkeletonDiffusion
from degradation import OracleDegradation
from condition import build_condition, build_condition_from_lr_image
from glyph_data import build_index, filter_valid
from glyph_perc import GlyphPerceptual
from stage2_data import Stage2Dataset, stage2_collate
from utils import (psnr, ssim_batch, dice_batch, to_uint8, load_ckpt,
                   save_ckpt)
from eval import count_small_holes
from eval_real import burr_ratio, count_specks

LOSS_CONFIGS = [
    ("mse_only",     (0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ("struct",       (0.5, 0.5, 0.5), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ("perc_rec",     (0.5, 0.5, 0.5), 0.1, 0.1, 0.0, 0.0, 0.0, 0.0),
    ("x0_fill_roll", (0.5, 0.5, 0.5), 0.1, 0.1, 1.0, 1.0, 1.0, 0.0),
    ("edge",         (0.5, 0.5, 0.5), 0.1, 0.1, 1.0, 1.0, 1.0, 0.5),
]


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


def defect_metrics(pred):
    u = to_uint8(pred)
    th = tb = ts = 0.0
    for i in range(u.shape[0]):
        img = u[i, 0]
        b = (img >= 127).astype(np.uint8)
        th += count_small_holes(img)
        br, _ = burr_ratio(b)
        tb += br
        ts += count_specks(b)
    n = max(1, u.shape[0])
    return th / n, tb / n, ts / n


@torch.no_grad()
def evaluate(diff, x0, cond, steps, device, batch_size=8):
    diff.eval()
    outs = []
    for i in range(0, cond.shape[0], batch_size):
        c = cond[i:i + batch_size].to(device)
        outs.append(diff.ddim_sample(c, num_steps=steps,
                                     guidance_scale=1.0).cpu())
    pred = torch.cat(outs)
    holes, burr, specks = defect_metrics(pred)
    return (psnr(pred, x0), ssim_batch(pred, x0), dice_batch(pred, x0),
            holes, burr, specks, pred)


@torch.no_grad()
def evaluate_real(diff, real_files, steps, device, hr_size, seed,
                  batch_size=16):
    diff.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    tot_h = tot_b = tot_s = 0
    grid_rows = []
    for i in range(0, len(real_files), batch_size):
        fb = real_files[i:i + batch_size]
        conds, ups = [], []
        for f in fb:
            x = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            up = cv2.resize(x, (hr_size, hr_size),
                            interpolation=cv2.INTER_CUBIC)
            pack = build_condition_from_lr_image(x, hr_size)
            conds.append(to_tensor(pack))
            ups.append(up)
        cond = torch.stack(conds).to(device)
        pred = diff.ddim_sample(cond, num_steps=steps, guidance_scale=1.0,
                                generator=gen).cpu()
        u = to_uint8(pred)
        for j in range(len(fb)):
            p = u[j, 0]
            b = (p >= 128).astype(np.uint8)
            tot_h += count_small_holes(p)
            br, _ = burr_ratio(b)
            tot_b += br
            tot_s += count_specks(b)
        if len(grid_rows) < 10:
            for j in range(len(fb)):
                grid_rows.append(np.hstack([ups[j], u[j, 0]]))
    n = max(1, len(real_files))
    grid = np.vstack(grid_rows[:20]) if grid_rows else None
    return tot_h / n, tot_b / n, tot_s / n, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="/root/autodl-tmp/train")
    ap.add_argument("--val-dir", default="/root/autodl-tmp/val")
    ap.add_argument("--handprint-dir",
                    default="/root/autodl-tmp/Oracle-241/handprint")
    ap.add_argument("--scan-dir",
                    default="/root/autodl-tmp/Oracle-241/scan/scanned")
    ap.add_argument("--glyph-encoder",
                    default="outputs/glyph_encoder/glyph_encoder.pth")
    ap.add_argument("--stage1-ckpt",
                    default="different_stage_final_checkpoint/stage1_final_ep69.pth",
                    help="common Stage1 starting point for all configurations (warm start)")
    ap.add_argument("--obi-dir", default="/root/autodl-tmp/OBI_BlackLow_64",
                    help="real oracle rubbing directory (no GT, in-domain evaluation)")
    ap.add_argument("--out-dir", default="outputs/ablation_loss")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-classes", type=int, default=241)
    ap.add_argument("--hr-size", type=int, default=128)
    ap.add_argument("--lr-size", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=2000,
                    help="maximum number of synthetic training images (small subset)")
    ap.add_argument("--glyph-limit", type=int, default=400,
                    help="maximum size of the labelled glyph sample pool")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--glyph-ratio", type=float, default=0.4)
    ap.add_argument("--sample-steps", type=int, default=25)
    ap.add_argument("--val-n", type=int, default=64)
    ap.add_argument("--real-n", type=int, default=94,
                    help="number of real rubbings for evaluation (fixed subset, identical across configurations)")
    ap.add_argument("--real-batch", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--roll-subset", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print("bf16 autocast:", use_bf16)
    degrade = OracleDegradation()

    print("indexing Oracle-241 ...")
    code2idx, gtr, _ = build_index(
        handprint_root=args.handprint_dir, scan_root=args.scan_dir,
        seed=args.seed)
    gtr = filter_valid(gtr, size=args.hr_size, verbose=True)
    if args.glyph_limit:
        gtr = gtr[:args.glyph_limit]
    print("labelled glyph pool = %d" % len(gtr))

    ds = Stage2Dataset(args.train_dir, gtr, hr_size=args.hr_size,
                       lr_size=args.lr_size, glyph_ratio=args.glyph_ratio,
                       train=True, seed=args.seed)
    ds.synth = ds.synth[:args.n_train]
    ds.n_synth = len(ds.synth)
    n_g = int(round(ds.n_synth * args.glyph_ratio /
                    max(1e-6, 1.0 - args.glyph_ratio)))
    ds.glyph = [gtr[i % len(gtr)] for i in range(n_g)] if gtr else []
    ds.n_glyph = len(ds.glyph)
    print("mixed epoch size: %d (synth %d + labelled %d)"
          % (len(ds), ds.n_synth, ds.n_glyph))

    glyph = GlyphPerceptual(args.glyph_encoder,
                            num_classes=args.num_classes, device=device)
    print("glyph encoder loaded (%d classes)" % args.num_classes)

    val_files = sorted(Path(args.val_dir).glob("*.jpg"))[:args.val_n]
    val_x0, val_cond = build_val_batch(
        val_files, args.hr_size, args.lr_size, degrade)

    real_rng = random.Random(args.seed)
    all_real = sorted(Path(args.obi_dir).glob("*.jpg"))
    if not all_real:
        raise SystemExit(
            "no .jpg in %s (pass --obi-dir to the real OBI folder)"
            % args.obi_dir)
    real_files = real_rng.sample(all_real, min(args.real_n, len(all_real)))
    print("real OBI subset = %d from %s" % (len(real_files), args.obi_dir))

    results = []
    for tag, sw, w_perc, w_rec, w_x0, w_fill, w_roll, roll_w_edge in LOSS_CONFIGS:
        print("loss ablation: %s" % tag)

        ckpt_path = ckpt_dir / ("%s.pth" % tag)
        torch.manual_seed(args.seed)
        model = CondUNet(1, 1, geom_ch=2, app_ch=2, base_ch=64,
                         ch_mult=(1, 2, 2, 4), num_res_blocks=2)
        diff = SkeletonDiffusion(
            model, w_edge=sw[0], w_occ=sw[1], w_skel=sw[2]).to(device)
        if ckpt_path.exists():
            _cfg0, ep0, bp0 = load_ckpt(str(ckpt_path), diff, device)
            print("RESUME: checkpoint exists, skip training (ep %d, "
                  "best %.3f)" % (ep0, bp0))
            opt = None
        else:
            _, s1ep, _ = load_ckpt(args.stage1_ckpt, diff, device)
            print("warm start from Stage1 (ep %d)" % s1ep)
            opt = torch.optim.AdamW(diff.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers,
                            collate_fn=stage2_collate, drop_last=True,
                            persistent_workers=args.num_workers > 0)
            print("params %.3fM"
                  % (sum(p.numel() for p in model.parameters()) / 1e6))

            for ep in range(args.epochs):
                diff.train()
                t0 = time.time()
                running, skipped, nb = 0.0, 0, 0
                pbar = tqdm(dl, desc="%s ep%d" % (tag, ep), leave=False,
                            ncols=100)
                for batch in pbar:
                    x0 = batch["x0"].to(device)
                    cond = batch["cond"].to(device)
                    labels = batch["label"].to(device)
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda",
                                        dtype=torch.bfloat16,
                                        enabled=use_bf16):
                        loss = diff.training_loss(
                            x0, cond, labels=labels, glyph=glyph,
                            w_perc=w_perc, w_rec=w_rec,
                            w_x0=w_x0, w_fill=w_fill, w_roll=w_roll,
                            roll_subset=args.roll_subset,
                            roll_w_edge=roll_w_edge)
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
                      % (ep, running / max(1, nb), skipped,
                         time.time() - t0), flush=True)

        p, s, d, holes, burr, specks, pred = evaluate(
            diff, val_x0, val_cond, args.sample_steps, device)
        print("  [SYNTH-GT %s] PSNR %.3f  SSIM %.3f  Dice %.3f  holes %.2f  "
              "burr %.4f  specks %.2f"
              % (tag, p, s, d, holes, burr, specks), flush=True)

        rh, rb, rs, real_grid = evaluate_real(
            diff, real_files, args.sample_steps, device, args.hr_size,
            args.seed, batch_size=args.real_batch)
        print("  [REAL-OBI %s] holes %.2f  burr %.4f  specks %.2f"
              % (tag, rh, rb, rs), flush=True)
        if real_grid is not None:
            cv2.imwrite(str(out_dir / ("real_%s_grid.png" % tag)), real_grid)

        if not ckpt_path.exists():
            model_config = {
                "geom_ch": 2, "app_ch": 2, "base_ch": 64,
                "ch_mult": [1, 2, 2, 4], "num_res_blocks": 2,
                "timesteps": 1000,
                "w_edge": sw[0], "w_occ": sw[1], "w_skel": sw[2],
            }
            save_ckpt(str(ckpt_path), diff, model_config,
                      epoch=args.epochs, best_psnr=p)

        results.append({
            "config": tag,
            "psnr": round(p, 3), "ssim": round(s, 3), "dice": round(d, 3),
            "holes": round(holes, 3), "burr": round(burr, 4),
            "specks": round(specks, 3),
            "real_holes": round(rh, 3), "real_burr": round(rb, 4),
            "real_specks": round(rs, 3),
        })

        rows = []
        u, u0 = to_uint8(pred), to_uint8(val_x0)
        for i in range(min(8, pred.shape[0])):
            rows.append(np.hstack([u0[i, 0], u[i, 0]]))
        cv2.imwrite(str(out_dir / ("%s_grid.png" % tag)), np.vstack(rows))

        with open(out_dir / "ablation_loss_results.json", "w",
                  encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)

    print("\n================ Table 6a: simple synthetic GT set (saturated, control) ================")
    print("| loss configuration | PSNR | SSIM | Dice | holes | burr | specks |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print("| %s | %.3f | %.3f | %.3f | %.2f | %.4f | %.2f |"
              % (r["config"], r["psnr"], r["ssim"], r["dice"],
                 r["holes"], r["burr"], r["specks"]))

    print("\n================ Table 6b: real rubbing domain (no GT, in-domain metrics) ================")
    print("| loss configuration | holes | burr | specks |")
    print("|---|---|---|---|")
    for r in results:
        print("| %s | %.2f | %.4f | %.2f |"
              % (r["config"], r["real_holes"], r["real_burr"],
                 r["real_specks"]))

    print("\nALL LOSS ABLATION DONE ->", out_dir)
    print("checkpoints ->", ckpt_dir)


if __name__ == "__main__":
    main()
