import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import to_tensor
from condition import build_condition_from_lr_image
from eval import build_model, count_small_holes
from utils import read_meta, load_ckpt, to_uint8
from glyph_data import normalize_glyph

STAGES = [
    ("Stage1", "stage1_final_ep69.pth"),
    ("Stage2", "stage2_final_ep10.pth"),
    ("Stage3", "stage3_final_ep4.pth"),
]


def read_gray(p):
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if g is None:
        g = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8),
                         cv2.IMREAD_GRAYSCALE)
    return g


def burr_ratio(bin_img, k=3):
    fg = int(bin_img.sum())
    if fg == 0:
        return 0.0
    opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN,
                              np.ones((k, k), np.uint8), iterations=1)
    return (fg - int(opened.sum())) / fg


def count_specks(bin_img, max_area=8):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    if n <= 1:
        return 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    main = 1 + int(np.argmax(areas))
    return int(sum(1 for c in range(1, n)
                   if c != main and stats[c, cv2.CC_STAT_AREA] < max_area))


def clean_reference(ref_dir, hr_size=128, cap=200):
    files = sorted([p for p in Path(ref_dir).rglob("*")
                    if p.suffix.lower() in (".bmp", ".jpg", ".png")])[:cap]
    tb = th = ts = 0
    k = 0
    for f in files:
        g = read_gray(f)
        if g is None:
            continue
        norm = normalize_glyph(g, hr_size)
        b = (norm >= 128).astype(np.uint8)
        if b.sum() == 0:
            continue
        tb += burr_ratio(b)
        th += count_small_holes(norm)
        ts += count_specks(b)
        k += 1
    if k == 0:
        return None
    return {"n_ref": k,
            "burr_pct": 100.0 * tb / k,
            "holes_per_img": th / k,
            "specks_per_img": ts / k}


def header_band(width, labels, panel=128, h=22):
    band = np.full((h, width), 255, np.uint8)
    for i, t in enumerate(labels):
        cv2.putText(band, t, (i * panel + 6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return band


def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--obi-dir", default="/root/autodl-tmp/OBI_BlackLow")
    ap.add_argument("--ckpt-dir",
                    default=str(root / "different_stage_final_checkpoint"))
    ap.add_argument("--out-dir", default=str(root / "outputs" / "eval_real16"))
    ap.add_argument("--n", type=int, default=1440)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hr-size", type=int, default=128)
    ap.add_argument("--grid-rows", type=int, default=24)
    ap.add_argument("--clean-ref-dir", default="",
                    help="optional clean high-resolution glyph directory for the ideal reference floor")
    ap.add_argument("--random-sample", action="store_true",
                    help="when --n is below the total, sample across classes by seed instead of taking the first n")
    ap.add_argument("--save-individual", action="store_true",
                    help="save input|bicubic|Ours triplets per image for manual selection of examples")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obi = Path(args.obi_dir)
    files = sorted([p for p in obi.rglob("*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not files:
        raise SystemExit("no images found: %s" % obi)
    total = len(files)
    if args.n < total:
        if args.random_sample:
            files = sorted(random.Random(args.seed).sample(files, args.n))
        else:
            files = files[:args.n]
    n = len(files)
    print("found %d images (pool=%d) under %s" % (n, total, obi), flush=True)

    conds, input_panels, bic_panels = [], [], []
    for f in files:
        x = read_gray(f)
        if x is None:
            raise SystemExit("failed to read: %s" % f)
        pack = build_condition_from_lr_image(x, args.hr_size)
        conds.append(to_tensor(pack))
        input_panels.append(cv2.resize(x, (args.hr_size, args.hr_size),
                                      interpolation=cv2.INTER_NEAREST))
        bic_panels.append(cv2.resize(x, (args.hr_size, args.hr_size),
                                     interpolation=cv2.INTER_CUBIC))
    cond = torch.stack(conds).to(device)

    results = {}
    stage_imgs = {}
    for name, fn in STAGES:
        ckpt = Path(args.ckpt_dir) / fn
        config, ep, _ = read_meta(str(ckpt))
        diff = build_model(config, device)
        load_ckpt(str(ckpt), diff, device)
        gen = torch.Generator(device=device).manual_seed(args.seed)
        tot_h = tot_b = tot_s = 0
        preds = []
        pbar = tqdm(range(0, n, args.batch_size),
                    desc=name, unit="batch", ncols=100)
        with torch.no_grad():
            for i in pbar:
                cb = cond[i:i + args.batch_size]
                pr = diff.ddim_sample(cb, num_steps=args.steps,
                                      guidance_scale=1.0,
                                      generator=gen).cpu()
                pbar.set_postfix(images=min(i + args.batch_size, n))
                preds.append(pr)
                u = to_uint8(pr)
                for j in range(u.shape[0]):
                    p = u[j, 0]
                    b = (p >= 128).astype(np.uint8)
                    tot_h += count_small_holes(p)
                    tot_b += burr_ratio(b)
                    tot_s += count_specks(b)
        pred_u8 = to_uint8(torch.cat(preds))[:, 0]
        stage_imgs[name] = pred_u8
        results[name] = {
            "holes_per_img": tot_h / n,
            "burr_pct": 100.0 * tot_b / n,
            "specks_per_img": tot_s / n,
            "checkpoint": fn,
            "epoch": ep,
        }
        print("[%s] holes %.3f/img  burr %.3f%%  specks %.3f/img"
              % (name, tot_h / n, 100.0 * tot_b / n, tot_s / n), flush=True)
        del diff
        torch.cuda.empty_cache()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m = min(args.grid_rows, n)
    rows5 = []
    for i in range(m):
        rows5.append(np.hstack([
            input_panels[i], bic_panels[i],
            stage_imgs["Stage1"][i], stage_imgs["Stage2"][i],
            stage_imgs["Stage3"][i]]))
    labels5 = ["LR16", "Bicubic", "Stage1", "Stage2", "Stage3"]
    grid5 = np.vstack([header_band(5 * args.hr_size, labels5), np.vstack(rows5)])
    p5 = out_dir / "real_grid5.png"
    cv2.imwrite(str(p5), grid5)

    rows3 = []
    for i in range(m):
        rows3.append(np.hstack([input_panels[i], bic_panels[i],
                                stage_imgs["Stage3"][i]]))
    labels3 = ["LR16", "Bicubic", "Ours"]
    grid3 = np.vstack([header_band(3 * args.hr_size, labels3), np.vstack(rows3)])
    p3 = out_dir / "fig3.png"
    cv2.imwrite(str(p3), grid3)

    if args.save_individual:
        ind_dir = out_dir / "individual"
        ind_dir.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(n), desc="save", unit="img", ncols=100):
            trip = np.hstack([input_panels[i], bic_panels[i],
                              stage_imgs["Stage3"][i]])
            cv2.imwrite(str(ind_dir / ("%04d.png" % i)), trip)
        print("INDIVIDUAL ->", ind_dir, flush=True)

    clean_ref = None
    if args.clean_ref_dir:
        clean_ref = clean_reference(args.clean_ref_dir, args.hr_size)
        if clean_ref:
            print("[CleanRef] burr %.3f%%  holes %.3f/img  specks %.3f/img"
                  % (clean_ref["burr_pct"], clean_ref["holes_per_img"],
                     clean_ref["specks_per_img"]), flush=True)

    payload = {
        "dataset": str(obi),
        "n": n,
        "pool": total,
        "random_sample": args.random_sample,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": 1.0,
        "raw_output_no_solidify": True,
        "note": "condition packs are built strictly from real 16x16 inputs; the three "
                "stages share the same initial noise; holes/burr/specks are reference-free "
                "morphological defect metrics.",
        "clean_reference": clean_ref,
        "metrics": results,
    }
    jpath = out_dir / "real16_results.json"
    with open(jpath, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    print("=" * 52)
    print("GRID5 ->", p5)
    print("FIG3  ->", p3)
    print("JSON  ->", jpath)


if __name__ == "__main__":
    main()
