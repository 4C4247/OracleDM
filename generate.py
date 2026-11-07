import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from data import to_tensor
from eval import build_model
from condition import build_condition_from_lr_image
from postprocess import clean_specks, solidify, adaptive_deburr
from utils import read_meta, load_ckpt, to_uint8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="/root/autodl-tmp/OBI_BlackLow_64")
    ap.add_argument("--ckpt",
                    default="outputs/skeletondiff_v3/checkpoints/best.pth")
    ap.add_argument("--out-dir", default="outputs/obi_generate")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hr-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance-scale", type=float, default=1.0)
    ap.add_argument("--cleanup", type=int, default=1,
                    help="1=remove isolated unsupported specks (postprocess)")
    ap.add_argument("--max-area", type=int, default=55)
    ap.add_argument("--max-area-frac", type=float, default=0.02)
    ap.add_argument("--gap", type=float, default=2.0)
    ap.add_argument("--app-thr", type=float, default=0.35)
    ap.add_argument("--conf-thr", type=float, default=0.50)
    ap.add_argument("--deburr", type=int, default=0,
                    help="attached 1-2px edge spike removal: k for a kxk "
                         "morphological opening (3 is mild); -1 = width-adaptive "
                         "(thin strokes 2x2, thick 3x3); 0 disables")
    ap.add_argument("--solidify", type=int, default=1,
                    help="close pinholes + fill small enclosed holes (solid strokes)")
    ap.add_argument("--solidify-close", type=int, default=2,
                    help="morphological closing kernel px (0 to disable)")
    ap.add_argument("--solidify-hole", type=int, default=12,
                    help="fill enclosed holes up to this many px (0 to disable)")
    ap.add_argument("--max-n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42,
                    help="fixes both image selection and sampling noise")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    select_rng = random.Random(args.seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    config, ep, best = read_meta(args.ckpt)
    diff = build_model(config, device)
    load_ckpt(args.ckpt, diff, device)
    print("loaded ep %d" % ep)

    all_files = sorted(Path(args.input_dir).glob("*.jpg"))
    if args.max_n:
        files = select_rng.sample(all_files, args.max_n)
    else:
        files = all_files
    print("selected:", [f.name for f in files])

    out_dir = Path(args.out_dir)
    (out_dir / "recon").mkdir(parents=True, exist_ok=True)

    pairs = []
    for f in files:
        x = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        up = cv2.resize(x, (args.hr_size, args.hr_size),
                        interpolation=cv2.INTER_CUBIC)
        pack = build_condition_from_lr_image(x, args.hr_size)
        cond = to_tensor(pack).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = diff.ddim_sample(cond, num_steps=args.steps,
                                    guidance_scale=args.guidance_scale,
                                    generator=gen)
        p = to_uint8(pred)[0, 0]
        if args.cleanup:
            p, removed, _ = clean_specks(
                p, cond=pack, max_area=args.max_area,
                max_area_frac=args.max_area_frac, gap=args.gap,
                app_thr=args.app_thr, conf_thr=args.conf_thr)
        if args.deburr:
            b = (p >= 127).astype(np.uint8)
            if args.deburr == -1:
                b, k, w = adaptive_deburr(b)
                print("    [deburr-auto] width %.2f -> %dx%d opening" % (w, k, k))
            else:
                b = cv2.morphologyEx(
                    b, cv2.MORPH_OPEN,
                    np.ones((args.deburr, args.deburr), np.uint8))
            p = (b * 255).astype(np.uint8)
        if args.solidify:
            p = solidify(p, close_px=args.solidify_close,
                         max_hole=args.solidify_hole)
        out_name = (f.stem + ".png") if args.solidify else f.name
        cv2.imwrite(str(out_dir / "recon" / out_name), p)
        pairs.append((up, p))

    cols = min(5, len(pairs))
    rows_n = int(np.ceil(len(pairs) / cols))
    before = np.vstack([np.hstack([pairs[i*cols+j][0]
                                   for j in range(cols)])
                        for i in range(rows_n)])
    after = np.vstack([np.hstack([pairs[i*cols+j][1]
                                  for j in range(cols)])
                       for i in range(rows_n)])
    h = min(before.shape[0], after.shape[0])
    cv2.imwrite(str(out_dir / "montage.png"),
                np.hstack([before[:h], after[:h]]))
    print("GENERATE DONE: %d -> %s" % (len(files), out_dir))


if __name__ == "__main__":
    main()
