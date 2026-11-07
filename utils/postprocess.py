import cv2
import numpy as np

SK = 0
CONF = 1
WIDTH = 2
APP = 3


def _components(b):
    return cv2.connectedComponentsWithStats(b, connectivity=8)


def clean_specks(raw_uint8, cond=None, max_area_frac=0.02, max_area=55,
                 gap=2.0, app_thr=0.35, conf_thr=0.50, dilate=2,
                 legacy_frac=0.005, legacy_min_area=8):
    b = (raw_uint8 >= 128).astype(np.uint8)
    n, labels, stats, _ = _components(b)

    if n <= 1:
        return raw_uint8.copy(), np.zeros_like(b), []

    areas = stats[:, cv2.CC_STAT_AREA]
    main_id = 1 + int(np.argmax(areas[1:]))
    main_area = int(areas[main_id])
    size_cap = max(float(max_area), max_area_frac * main_area)
    legacy_cap = max(float(legacy_min_area), legacy_frac * main_area)

    k = None
    if cond is not None and dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * dilate + 1, 2 * dilate + 1))

    keep = np.zeros_like(b)
    removed = np.zeros_like(b)
    info = []

    for cid in range(1, n):
        comp = (labels == cid).astype(np.uint8)
        area = int(areas[cid])

        if cid == main_id:
            keep += comp
            info.append({"area": area, "dist": 0.0, "main": True,
                         "max_app": 1.0, "max_conf": 1.0,
                         "removed": False})
            continue

        others = (b - comp).astype(np.uint8)
        if others.sum() > 0 and gap > 0:
            dt = cv2.distanceTransform((1 - others).astype(np.uint8),
                                       cv2.DIST_L2, 3)
            min_dist = float(dt[comp > 0].min())
        else:
            min_dist = 0.0
        isolated = min_dist > gap

        max_app = max_conf = 1.0
        if cond is not None:
            region = comp if k is None else cv2.dilate(comp, k)
            max_app = float(cond[APP][region > 0].max())
            max_conf = float(cond[CONF][region > 0].max())
            unsupported = (max_app < app_thr) and (max_conf < conf_thr)
            is_speck = isolated and area < size_cap and unsupported
        else:
            is_speck = isolated and area < legacy_cap

        if is_speck:
            removed += comp
        else:
            keep += comp
        info.append({"area": area, "dist": round(min_dist, 1), "main": False,
                     "max_app": round(max_app, 2),
                     "max_conf": round(max_conf, 2),
                     "removed": bool(is_speck)})

    cleaned = np.where(keep.astype(bool), raw_uint8, 0).astype(np.uint8)
    return cleaned, removed, info


def solidify(cleaned_uint8, close_px=2, max_hole=12):
    b = (cleaned_uint8 >= 128).astype(np.uint8)

    if close_px and close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT,
                                      (int(close_px), int(close_px)))
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)

    if max_hole and max_hole > 0:
        inv = (1 - b).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(
            inv, connectivity=4)
        border = set(np.unique(np.concatenate(
            [lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        for cid in range(1, n):
            if cid in border:
                continue
            if stats[cid, cv2.CC_STAT_AREA] <= int(max_hole):
                b[lab == cid] = 1

    return (b * 255).astype(np.uint8)


def median_stroke_width(b):
    from skimage.morphology import skeletonize
    b = (b > 0).astype(np.uint8)
    sk = skeletonize(b > 0).astype(np.uint8)
    if sk.sum() == 0:
        return 0.0
    d = cv2.distanceTransform(b, cv2.DIST_L2, 3)
    return float(np.median(d[sk > 0]) * 2.0)


def adaptive_deburr(b, thin_thresh=5.2, thick_k=3, thin_k=2):
    b = (b > 0).astype(np.uint8)
    w = median_stroke_width(b)
    k = thin_k if w < thin_thresh else thick_k
    out = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    return out, k, w
