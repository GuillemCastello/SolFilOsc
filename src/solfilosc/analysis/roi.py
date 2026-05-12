"""ROI and mask helpers for the analysis pipeline."""

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.measure import label, regionprops

def weighted_median_log(logx, w):
    logx = np.asarray(logx, np.float64)
    w = np.asarray(w, np.float64)

    m = np.isfinite(logx) & np.isfinite(w) & (w > 0)
    if m.sum() == 0:
        return np.nan

    logx = logx[m]
    w = w[m]
    order = np.argsort(logx)
    cdf = np.cumsum(w[order]) / np.sum(w)
    idx = np.searchsorted(cdf, 0.5)
    return float(logx[order][idx])

def local_nan_std(x, win=3):
    x = np.asarray(x, np.float32)
    m = np.isfinite(x).astype(np.float32)
    x0 = np.nan_to_num(x, nan=0.0)
    n = uniform_filter(m, size=win, mode="nearest")
    s1 = uniform_filter(x0 * m, size=win, mode="nearest")
    s2 = uniform_filter(x0**2 * m, size=win, mode="nearest")
    mean = s1 / np.maximum(n, 1e-9)
    var = np.maximum(s2 / np.maximum(n, 1e-9) - mean * mean, 0.0)
    std = np.sqrt(var)
    std[n < 1e-6] = np.nan
    return std.astype(np.float32)

def list_candidate_regions_from_mask(mask0, *, min_area=2000):
    lbl = label((mask0 > 0).astype(np.uint8))
    regs = [r for r in regionprops(lbl) if r.area >= int(min_area)]
    return sorted(regs, key=lambda x: x.area, reverse=True)

def select_roi_bbox_from_first_mask(mask0, *, min_area=2000, pick="largest", pick_index=0):
    regs = list_candidate_regions_from_mask(mask0, min_area=int(min_area))
    if not regs:
        raise RuntimeError("No regions found above min_area")

    if pick == "largest":
        return regs[0].bbox

    pick_index = int(pick_index)
    if pick_index >= len(regs):
        raise RuntimeError(f"pick_index {pick_index} out of range for {len(regs)} regions")
    return regs[pick_index].bbox

def expand_bbox(bbox, pad, H, W):
    minr, minc, maxr, maxc = bbox
    return (max(0, minr - pad), max(0, minc - pad), min(H, maxr + pad), min(W, maxc + pad))

def degraded_center_to_roi_xy(i, j, N, S):
    return float(j * S + 0.5 * N), float(i * S + 0.5 * N)

def patch_mask_to_roi(patch_mask, N, S, roi_H, roi_W):
    roi = np.zeros((roi_H, roi_W), dtype=bool)
    if patch_mask is None or not np.any(patch_mask):
        return roi

    rr, cc = np.where(patch_mask)
    if rr.size == 0:
        return roi

    y0s = (rr * S).astype(np.intp)
    x0s = (cc * S).astype(np.intp)
    y1s = np.minimum(y0s + N, roi_H)
    x1s = np.minimum(x0s + N, roi_W)

    for y0, y1, x0, x1 in zip(y0s, y1s, x0s, x1s):
        roi[y0:y1, x0:x1] = True
    return roi

def bbox_from_mask(mask_bool):
    mask_bool = np.asarray(mask_bool, bool)
    rr, cc = np.where(mask_bool)
    if rr.size == 0:
        return None
    return (int(rr.min()), int(cc.min()), int(rr.max() + 1), int(cc.max() + 1))
