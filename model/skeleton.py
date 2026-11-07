import cv2
import numpy as np
from skimage.morphology import skeletonize


def to_uint8(x):
    if x.dtype == np.uint8:
        return x
    return np.clip(np.asarray(x) * 255.0, 0, 255).astype(np.uint8)


def binarize(x):
    x = to_uint8(x)
    b = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if b.mean() > 127:
        b = 255 - b
    return b


def clean_binary(b, open_k=2, close_k=3):
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    return b


def keep_glyph_components(b, min_size=10, min_frac=0.03, dist_frac=0.16):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (b > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return b
    areas = stats[1:, cv2.CC_STAT_AREA]
    main = int(areas.argmax()) + 1
    main_area = areas[main - 1]
    max_dist = dist_frac * max(b.shape)

    main_mask = (labels == main).astype(np.uint8)
    dt = cv2.distanceTransform(1 - main_mask, cv2.DIST_L2, 3)

    keep = np.zeros_like(b)
    for i in range(1, n):
        comp = labels == i
        if i == main:
            keep[comp] = 255
            continue
        area = areas[i - 1]
        if area < min_size or area < min_frac * main_area:
            continue
        if dt[comp].min() <= max_dist:
            keep[comp] = 255
    return keep


def extract_skeleton(x, open_k=2, close_k=3, min_size=10, min_frac=0.03,
                     dist_frac=0.16):
    b = clean_binary(binarize(x), open_k=open_k, close_k=close_k)
    b = keep_glyph_components(b, min_size=min_size, min_frac=min_frac,
                              dist_frac=dist_frac)
    sk = (skeletonize(b > 0) * 255).astype(np.uint8)
    return b, sk


def skeleton_mask(x, open_k=2, close_k=3):
    _, sk = extract_skeleton(x, open_k=open_k, close_k=close_k)
    return sk > 0
