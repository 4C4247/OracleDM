"""4.3 未见字形泛化评估（表2，纯推理，不训练）。

口径说明（重要，务必如实）
--------------------------
真实 Stage2/Stage3 训练（train_stage2.py，--glyph-limit 默认 0）对全部
Oracle-241 字类的 *训练划分* 做了身份监督，240/241 个字类都有有效训练样本；
唯一没有训练样本的字类也没有任何可用图像。因此严格意义上的"整类未见"在
现有定版模型上无法构造。

本脚本采用分层留出的 **未见实例（held-out instances）**：build_index 对每个
字类按 15% 分层切出的 val 实例（gva），训练数据集只接收 train 划分（gtr），
从不接收 gva，因此这些具体字形实例模型从未见过，是可复现、可辩护的泛化测试，
也贴合真实部署（真实 OBI 输入都是已知字类的新实例）。

每张留出 HR（normalize_glyph 到 128）经确定性退化到 LR16，构建 4 通道条件包
（clean=False，几何取自退化 LR，模拟真实推理），与 Bicubic 一起对照，DDIM 25 步
固定 generator seed=42，报告 PSNR/SSIM/Dice/holes。

服务器示例：
  python eval_unseen.py \
    --checkpoint different_stage_final_checkpoint/stage3_final_ep4.pth \
    --handprint-dir /root/autodl-tmp/Oracle-241/handprint \
    --scan-dir /root/autodl-tmp/Oracle-241/scan/scanned \
    --out-dir outputs/eval_unseen --device cuda
"""

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


# ---------------------------------------------------------------------------
# 缺陷指标（与 eval.py 一致，复制以避免依赖旧文件的失效导入）
# ---------------------------------------------------------------------------

def count_small_holes(img, max_area=12):
    """Count enclosed black holes (area <= max_area px) inside a uint8 glyph."""
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


# ---------------------------------------------------------------------------
# 数据选择
# ---------------------------------------------------------------------------

def select_held_out(items, n_total, per_class, seed):
    """按字类均衡选择：每字类最多 per_class 个，总数不超过 n_total（确定性）。"""
    by_class = defaultdict(list)
    for path, label, domain in items:
        by_class[label].append((path, label, domain))

    chosen = []
    # 字类顺序固定；每类先取前 per_class 个（items 已按路径排序）
    for label in sorted(by_class):
        group = by_class[label][:per_class]
        chosen.extend(group)
    chosen = chosen[:n_total]
    return chosen


def is_solid_flare(hr, solid_dmax=18.0, solid_r=10.0, solid_frac=0.10):
    """判断 HR 目标是否含实心光斑/非笔画块。

    笔画网络是细线，距离变换值小；实心块内部有大量远离背景的像素。
    判据（任一命中即剔除）：最大距离变换 > solid_dmax，或距离 > solid_r 的
    前景像素占比 > solid_frac。该门只看目标本身是否为有效字形，与模型表现无关。
    阈值在本地 scan 域标定：有效粗字 dmax 多在 18 以内、fr10<=0.10，实心块
    明显越界，按此规则仅剔除约 7.8% 的坏目标。
    """
    fg = (hr > 0).astype(np.uint8)
    if fg.sum() == 0:
        return True
    d = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
    fr = (d > solid_r).sum() / max(1, int(fg.sum()))
    return bool(d.max() > solid_dmax or fr > solid_frac)


def gate_items(items, hr_size, solid_dmax, solid_r, solid_frac):
    """对候选样本做实心光斑门，返回 (kept, rejected_paths)。"""
    kept, rejected = [], []
    for path, label, domain in items:
        hr = normalize_glyph(load_gray(path), size=hr_size)
        if is_solid_flare(hr, solid_dmax, solid_r, solid_frac):
            rejected.append(str(path))
        else:
            kept.append((path, label, domain))
    return kept, rejected


def build_sample(path, hr_size, lr_size, degrade):
    """HR 目标 + 确定性 LR16 + 条件包 + bicubic。"""
    gray = load_gray(path)
    hr = normalize_glyph(gray, size=hr_size)          # uint8 目标
    x0 = to_tensor(hr)                                 # (1,H,W)
    lr = degrade_lr(x0, degrade, lr_size)              # float [0,1] LR16
    cond = build_condition(hr, lr, hr_size, clean=False)
    bic = upscale_lr(lr, hr_size, cv2.INTER_CUBIC)    # uint8 128
    return hr, x0, to_tensor(cond), to_tensor(bic)


def degrade_lr(x0, degrade, lr_size):
    with torch.no_grad():
        lr = degrade.to_lr(x0.unsqueeze(0), target_size=lr_size,
                           deterministic=True)[0, 0]
    return lr.numpy()


# ---------------------------------------------------------------------------
# 可视化：target | bicubic | pred
# ---------------------------------------------------------------------------

def _label_band(text, width):
    band = np.full((16, width), 255, np.uint8)
    cv2.putText(band, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 0, 0), 1)
    return band


def montage(rows, save_path, pad=2):
    """rows: list of (target_u8, bicubic_u8, pred_u8)."""
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


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

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
    ap.add_argument("--n", type=int, default=96, help="评估样本总数上限")
    ap.add_argument("--per-class", type=int, default=4, help="每字类最多样本数")
    ap.add_argument("--domain", choices=["scan", "hand", "both"], default="both",
                    help="只评估某一域；scan=真实拓片，hand=手写参考")
    ap.add_argument("--solid-dmax", type=float, default=18.0,
                    help="质量门：最大距离变换阈值，超过即判定实心光斑")
    ap.add_argument("--solid-r", type=float, default=10.0,
                    help="质量门：判定实心内部的距离阈值")
    ap.add_argument("--solid-frac", type=float, default=0.10,
                    help="质量门：实心内部像素占比阈值")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    ch_mult = tuple(int(x) for x in args.ch_mult.split(","))
    degrade = OracleDegradation()

    # --- 分层留出实例（训练从未见过）---
    print("indexing Oracle-241 ...")
    code2idx, _gtr, gva = build_index(
        handprint_root=args.handprint_dir, scan_root=args.scan_dir,
        seed=args.seed)
    idx2code = {v: k for k, v in code2idx.items()}
    gva = filter_valid(gva, size=args.hr_size, verbose=True)
    print("held-out valid instances = %d" % len(gva))

    # --- 域过滤（scan/hand/both）---
    if args.domain != "both":
        gva = [it for it in gva if it[2] == args.domain]
        print("domain filter (%s) -> %d instances" % (args.domain, len(gva)))

    # --- 客观质量门：剔除目标本身含实心光斑/非笔画块的样本 ---
    gva, rejected = gate_items(
        gva, args.hr_size, args.solid_dmax, args.solid_r, args.solid_frac)
    print("solid-flare gate: kept %d, rejected %d" % (len(gva), len(rejected)))
    if not gva:
        raise SystemExit("no valid held-out glyphs after the gate")

    chosen = select_held_out(gva, args.n, args.per_class, args.seed)
    used_classes = sorted({lab for (_p, lab, _d) in chosen})
    print("selected = %d across %d classes" % (len(chosen), len(used_classes)))

    # --- 模型 ---
    model = CondUNet(1, 1, geom_ch=2, app_ch=2, base_ch=args.base_ch,
                     ch_mult=ch_mult, num_res_blocks=args.num_res_blocks)
    diff = SkeletonDiffusion(model, timesteps=args.timesteps).to(device)
    cfg, ep, bp = load_ckpt_local(args.checkpoint, diff, device)
    print("loaded %s (ep %s, best %.3f)" % (args.checkpoint, ep, bp))
    diff.eval()

    # --- 逐批推理 ---
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

    # --- 指标（灰度口径）---
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

    print("\n================ 未见字形泛化（表2，held-out 实例）================")
    print("| 方法 | PSNR↑ | SSIM↑ | Dice↑ | holes↓ |")
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
