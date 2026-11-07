import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from unet import CondUNet
from diffusion import SkeletonDiffusion
from degradation import OracleDegradation
from condition import build_condition, upscale_lr
from data import to_tensor, load_gray
from glyph_data import build_index, filter_valid, normalize_glyph
from utils import psnr, ssim_batch, dice_batch


def count_small_holes(img, max_area=12):
    inv = cv2.bitwise_not(img)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    h, w = img.shape
    cnt = 0
    for c in range(1, n):
        x, y, cw, ch, a = stats[c]
        if a > max_area:
            continue
        if x > 0 and y > 0 and x + cw < w and y + ch < h:
            cnt += 1
    return cnt


def mean_holes(imgs):
    return float(np.mean([count_small_holes(x) for x in imgs]))


def select_held_out(items, n_total, per_class, seed):
    by_class = defaultdict(list)
    for path, label, domain in items:
        by_class[label].append((path, label, domain))

    chosen = []
    for label in sorted(by_class):
        group = by_class[label][:per_class]
        chosen.extend(group)
    chosen = chosen[:n_total]
    return chosen


def is_solid_flare(hr, solid_dmax=18.0, solid_r=10.0, solid_frac=0.10):
    fg = (hr > 0).astype(np.uint8)
    if fg.sum() == 0:
        return True
    d = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
    fr = (d > solid_r).sum() / max(1, int(fg.sum()))
    return bool(d.max() > solid_dmax or fr > solid_frac)


def gate_items(items, hr_size, solid_dmax, solid_r, solid_frac):
    kept, rejected = [], []
    for path, label, domain in items:
        hr = normalize_glyph(load_gray(path), size=hr_size)
        if is_solid_flare(hr, solid_dmax, solid_r, solid_frac):
            rejected.append(str(path))
        else:
            kept.append((path, label, domain))
    return kept, rejected


def build_sample(path, hr_size, lr_size, degrade):
    gray = load_gray(path)
    hr = normalize_glyph(gray, size=hr_size)
    x0 = to_tensor(hr)
    lr = degrade_lr(x0, degrade, lr_size)
    cond = build_condition(hr, lr, hr_size, clean=False)
    bic = upscale_lr(lr, hr_size, cv2.INTER_CUBIC)
    return hr, x0, to_tensor(cond), to_tensor(bic)


def degrade_lr(x0, degrade, lr_size):
    with torch.no_grad():
        lr = degrade.to_lr(x0.unsqueeze(0), target_size=lr_size,
                           deterministic=True)[0, 0]
    return lr.numpy()


def _label_band(text, width):
    band = np.full((16, width), 255, np.uint8)
    cv2.putText(band, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 0, 0), 1)
    return band


def montage(rows, save_path, pad=2):
    panels = []
    for t, b, p in rows:
        for img, name in ((t, "target"), (b, "bicubic"), (p, "pred")):
            v = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            band = cv2.cvtColor(_label_band(name, v.shape[1]),
                                cv2.COLOR_GRAY2BGR)
            panels.append(np.pad(np.vstack([band, v]),
                                 ((pad, pad), (pad, pad), (0, 0)),
                                 constant_values=64))
    triplet_w = panels[0].shape[1] * 3
    out_rows = []
    for i in range(0, len(panels), 3):
        triplet = np.hstack(panels[i:i + 3])
        out_rows.append(triplet)
    cv2.imwrite(str(save_path), np.vstack(out_rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="different_stage_final_checkpoint/stage3_final_ep4.pth")
    ap.add_argument("--handprint-dir",
                    default="/root/autodl-tmp/Oracle-241/handprint")
    ap.add_argument("--scan-dir",
                    default="/root/autodl-tmp/Oracle-241/scan/scanned")
    ap.add_argument("--out-dir", default="outputs/eval_unseen")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hr-size", type=int, default=128)
    ap.add_argument("--lr-size", type=int, default=16)
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--ch-mult", default="1,2,2,4")
    ap.add_argument("--num-res-blocks", type=int, default=2)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance-scale", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n", type=int, default=96, help="maximum total number of evaluated samples")
    ap.add_argument("--per-class", type=int, default=4, help="maximum number of samples per character class")
    ap.add_argument("--domain", choices=["scan", "hand", "both"], default="both",
                    help="evaluate only one domain; scan=real rubbings, hand=handwritten references")
    ap.add_argument("--solid-dmax", type=float, default=18.0,
                    help="quality gate: maximum distance-transform threshold, above which a solid flare is declared")
    ap.add_argument("--solid-r", type=float, default=10.0,
                    help="quality gate: distance threshold for identifying a solid interior")
    ap.add_argument("--solid-frac", type=float, default=0.10,
                    help="quality gate: pixel fraction threshold for a solid interior")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    ch_mult = tuple(int(x) for x in args.ch_mult.split(","))
    degrade = OracleDegradation()

    print("indexing Oracle-241 ...")
    code2idx, _gtr, gva = build_index(
        handprint_root=args.handprint_dir, scan_root=args.scan_dir,
        seed=args.seed)
    idx2code = {v: k for k, v in code2idx.items()}
    gva = filter_valid(gva, size=args.hr_size, verbose=True)
    print("held-out valid instances = %d" % len(gva))

    if args.domain != "both":
        gva = [it for it in gva if it[2] == args.domain]
        print("domain filter (%s) -> %d instances" % (args.domain, len(gva)))

    gva, rejected = gate_items(
        gva, args.hr_size, args.solid_dmax, args.solid_r, args.solid_frac)
    print("solid-flare gate: kept %d, rejected %d" % (len(gva), len(rejected)))
    if not gva:
        raise SystemExit("no valid held-out glyphs after the gate")

    chosen = select_held_out(gva, args.n, args.per_class, args.seed)
    used_classes = sorted({lab for (_p, lab, _d) in chosen})
    print("selected = %d across %d classes" % (len(chosen), len(used_classes)))

    model = CondUNet(1, 1, geom_ch=2, app_ch=2, base_ch=args.base_ch,
                     ch_mult=ch_mult, num_res_blocks=args.num_res_blocks)
    diff = SkeletonDiffusion(model, timesteps=args.timesteps).to(device)
    cfg, ep, bp = load_ckpt_local(args.checkpoint, diff, device)
    print("loaded %s (ep %s, best %.3f)" % (args.checkpoint, ep, bp))
    diff.eval()

    tgt_list, bic_list, pred_list = [], [], []
    montage_rows = []
    for start in range(0, len(chosen), args.batch_size):
        chunk = chosen[start:start + args.batch_size]
        tgt_b, bic_b, cond_b = [], [], []
        for path, _lab, _dom in chunk:
            hr, x0, cond, bic = build_sample(
                path, args.hr_size, args.lr_size, degrade)
            tgt_b.append(x0)
            bic_b.append(bic)
            cond_b.append(cond)
        tgt = torch.stack(tgt_b).to(device)
        bic = torch.stack(bic_b).to(device)
        cond = torch.stack(cond_b).to(device)

        gen = torch.Generator(device=device).manual_seed(args.seed)
        with torch.no_grad():
            pred = diff.ddim_sample(cond, num_steps=args.steps,
                                    guidance_scale=args.guidance_scale,
                                    eta=0.0, generator=gen)

        tgt_list.append(tgt.cpu())
        bic_list.append(bic.cpu())
        pred_list.append(pred.cpu())
        for i in range(tgt.shape[0]):
            montage_rows.append((
                _u8(tgt[i]), _u8(bic[i]), _u8(pred[i])))
        print("  processed %d/%d" % (min(start + args.batch_size, len(chosen)),
                                     len(chosen)))

    tgt_all = torch.cat(tgt_list)
    bic_all = torch.cat(bic_list)
    pred_all = torch.cat(pred_list)

    metrics = {
        "Bicubic": {
            "psnr": psnr(bic_all, tgt_all),
            "ssim": ssim_batch(bic_all, tgt_all),
            "dice": dice_batch(bic_all, tgt_all),
            "holes": mean_holes([_u8(x) for x in bic_all]),
        },
        "Ours": {
            "psnr": psnr(pred_all, tgt_all),
            "ssim": ssim_batch(pred_all, tgt_all),
            "dice": dice_batch(pred_all, tgt_all),
            "holes": mean_holes([_u8(x) for x in pred_all]),
        },
    }

    print("\n================ unseen-character generalisation (Table 2, held-out instances) ================")
    print("| method | PSNR | SSIM | Dice | holes |")
    print("|---|---|---|---|---|")
    for name in ("Bicubic", "Ours"):
        m = metrics[name]
        print("| %s | %.3f | %.3f | %.3f | %.2f |"
              % (name, m["psnr"], m["ssim"], m["dice"], m["holes"]))

    montage(montage_rows, out_dir / "unseen_grid.png")
    result = {
        "checkpoint": args.checkpoint, "seed": args.seed,
        "steps": args.steps, "guidance_scale": args.guidance_scale,
        "domain": args.domain,
        "solid_gate": {"solid_dmax": args.solid_dmax, "solid_r": args.solid_r,
                       "solid_frac": args.solid_frac, "n_rejected": len(rejected),
                       "rejected_paths": rejected},
        "n": len(chosen), "n_classes": len(used_classes),
        "held_out_classes": [idx2code[i] for i in used_classes],
        "note": "held-out instances (per-class stratified val), not whole classes; "
                "solid-flare gate removes targets that are not valid stroke networks",
        "metrics": metrics,
    }
    with open(out_dir / "unseen_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\nsaved grid -> %s" % (out_dir / "unseen_grid.png"))
    print("saved json -> %s" % (out_dir / "unseen_results.json"))


def _u8(x):
    return np.clip(x.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)[0]


def load_ckpt_local(path, model, device):
    data = torch.load(path, map_location=device)
    model.load_state_dict(data["model"])
    return data.get("config", {}), data.get("epoch", 0), data.get(
        "best_psnr", 0.0)


if __name__ == "__main__":
    main()
