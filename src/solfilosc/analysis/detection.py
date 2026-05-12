"""Multi-peak period detection and connected-component extraction."""

import math

import numpy as np
from joblib import Parallel, delayed, parallel_backend
from scipy.ndimage import uniform_filter
from scipy.signal import find_peaks
from skimage.measure import label, regionprops
from tqdm import tqdm

from .cnn import cnn_predict_noise_params, compute_ls_psd_safe, noise_model
from .constants import CNN_FREQUENCY_GRID, DETECTION_FMAX_HZ, DETECTION_FMIN_HZ, N_PIXEL_WORKERS
from .roi import (
    bbox_from_mask,
    degraded_center_to_roi_xy,
    local_nan_std,
    patch_mask_to_roi,
    weighted_median_log,
)

def analyze_degraded_stack_multipeak_cp(
    t,
    stack,
    cnn_model,
    scaler,
    m,
    s,
    qk,
    *,
    keep_mask=None,
    min_finite_frac=0.8,
    apply_hann=True,
    top_m_peaks=4,
    peak_min_prom_frac=0.05,
    n_jobs=N_PIXEL_WORKERS,
    batch_size_cnn=4096,
    detect_fmin_hz=DETECTION_FMIN_HZ,
    detect_fmax_hz=DETECTION_FMAX_HZ,
):
    freqs = CNN_FREQUENCY_GRID.astype(np.float64)
    detect_fmin_hz = float(detect_fmin_hz)
    detect_fmax_hz = float(detect_fmax_hz)

    if not (np.isfinite(detect_fmin_hz) and np.isfinite(detect_fmax_hz) and detect_fmin_hz < detect_fmax_hz):
        raise ValueError("Invalid detection band.")

    detect_band = (freqs >= detect_fmin_hz) & (freqs <= detect_fmax_hz)
    if not np.any(detect_band):
        raise RuntimeError("Detection band does not overlap CNN_FREQUENCY_GRID.")

    t = np.asarray(t, np.float64)
    m = np.asarray(m, np.float64)
    s = np.asarray(s, np.float64)
    qk = np.asarray(qk, np.float64)

    if not (len(freqs) == len(m) == len(s) == len(qk)):
        raise ValueError("freqs, m, s, qk must all have the same length.")

    tau_factor = np.exp(m + qk * s).astype(np.float64)

    T, h, w = stack.shape
    M = int(top_m_peaks)

    periods_m = np.full((h, w, M), np.nan, dtype=np.float32)
    freqs_m = np.full((h, w, M), np.nan, dtype=np.float32)
    z_m = np.full((h, w, M), np.nan, dtype=np.float32)
    best_period_min = np.full((h, w), np.nan, dtype=np.float32)
    best_z = np.full((h, w), np.nan, dtype=np.float32)
    sig_any = np.zeros((h, w), dtype=bool)

    if keep_mask is None:
        keep_mask = np.ones((h, w), dtype=bool)
    else:
        keep_mask = keep_mask.astype(bool)

    t_ok_base = np.isfinite(t)
    min_pts = max(5, int(min_finite_frac * T))

    def _one_pixel_psd(idx_flat):
        i = idx_flat // w
        j = idx_flat % w

        if not keep_mask[i, j]:
            return i, j, None

        y = stack[:, i, j].astype(np.float64)
        ok = t_ok_base & np.isfinite(y)
        if ok.sum() < min_pts:
            return i, j, None

        tt = t[ok]
        yy = y[ok]

        pxx = compute_ls_psd_safe(
            tt, yy, freqs,
            apply_hann=apply_hann,
            min_var=1e-12,
            min_energy=1e-20,
        )
        return i, j, pxx

    flat_ids = np.flatnonzero(keep_mask)
    with parallel_backend("loky", n_jobs=n_jobs):
        results = Parallel()(
            delayed(_one_pixel_psd)(k)
            for k in tqdm(flat_ids, desc="LS per degraded pixel (CP)", leave=False)
        )

    valid_results = [(i, j, pxx) for (i, j, pxx) in results if pxx is not None]
    if not valid_results:
        logp = np.full((h, w), np.nan, np.float32)
        coherence_std = local_nan_std(logp, win=3)
        patch_score = uniform_filter(np.nan_to_num(best_z, nan=0.0), size=3, mode="nearest").astype(np.float32)

        return dict(
            periods_m=periods_m,
            freqs_m=freqs_m,
            z_m=z_m,
            best_period_min=best_period_min,
            best_z=best_z,
            sig_any=sig_any,
            keep_mask=keep_mask,
            coherence_logp_std=coherence_std,
            patch_score=patch_score,
        )

    coords = [(i, j) for (i, j, _) in valid_results]
    psds = np.vstack([pxx for (_, _, pxx) in valid_results]).astype(np.float32)

    n_pix = psds.shape[0]
    n_batches = math.ceil(n_pix / batch_size_cnn)
    all_params = []
    for b in range(n_batches):
        sl = slice(b * batch_size_cnn, (b + 1) * batch_size_cnn)
        all_params.append(cnn_predict_noise_params(cnn_model, psds[sl], scaler))
    noise_params = np.vstack(all_params)

    for (i, j), pxx32, params in zip(coords, psds, noise_params):
        pxx = pxx32.astype(np.float64)
        shat = noise_model(freqs, *params)
        tau = np.maximum(shat, 1e-30) * tau_factor

        sigmask = detect_band & np.isfinite(pxx) & np.isfinite(tau) & (pxx > tau)
        if not np.any(sigmask):
            continue

        snr = pxx / (tau + 1e-300)
        snr_sig = np.where(sigmask, snr, -np.inf)
        finite_sig = np.isfinite(snr_sig)
        if not np.any(finite_sig):
            continue

        snr_max = np.max(snr_sig[finite_sig])
        if not np.isfinite(snr_max):
            continue

        prom = float(peak_min_prom_frac) * float(snr_max)
        peaks, _ = find_peaks(np.nan_to_num(snr_sig, nan=-np.inf), prominence=prom)

        if peaks.size == 0:
            cand = np.argsort(snr_sig)[::-1]
            peaks = cand[np.isfinite(snr_sig[cand])][:M]

        peaks = np.asarray(peaks, int)
        peaks = peaks[np.isfinite(snr_sig[peaks])]
        if peaks.size == 0:
            continue

        order = peaks[np.argsort(snr_sig[peaks])[::-1]][:M]

        sig_any[i, j] = True
        zz = snr[order].astype(np.float32)

        for m_idx, k in enumerate(order):
            bf = float(freqs[int(k)])
            bpmin = float((1.0 / bf) / 60.0)
            periods_m[i, j, m_idx] = bpmin
            freqs_m[i, j, m_idx] = bf
            z_m[i, j, m_idx] = float(zz[m_idx])

        best_period_min[i, j] = float(periods_m[i, j, 0])
        best_z[i, j] = float(z_m[i, j, 0])

    logp = np.full((h, w), np.nan, np.float32)
    m_ok = sig_any & np.isfinite(best_period_min) & (best_period_min > 0)
    logp[m_ok] = np.log(best_period_min[m_ok])

    coherence_std = local_nan_std(logp, win=3)
    patch_score = uniform_filter(np.nan_to_num(best_z, nan=0.0), size=3, mode="nearest").astype(np.float32)

    return dict(
        periods_m=periods_m,
        freqs_m=freqs_m,
        z_m=z_m,
        best_period_min=best_period_min,
        best_z=best_z,
        sig_any=sig_any,
        keep_mask=keep_mask,
        coherence_logp_std=coherence_std,
        patch_score=patch_score,
    )

def periods_match(Pa, Pb, *, period_tol_frac=0.15, period_abs_tol_min=2.0):
    Pa = float(Pa)
    Pb = float(Pb)
    if not (np.isfinite(Pa) and np.isfinite(Pb) and Pa > 0 and Pb > 0):
        return False

    if abs(Pa - Pb) <= float(period_abs_tol_min):
        return True

    return abs(math.log(Pa) - math.log(Pb)) <= math.log1p(float(period_tol_frac))

def cluster_period_values(
    periods,
    weights,
    *,
    period_tol_frac=0.15,
    period_abs_tol_min=2.0,
    max_groups=None,
):
    periods = np.asarray(periods, np.float64)
    weights = np.asarray(weights, np.float64)

    m = np.isfinite(periods) & (periods > 0) & np.isfinite(weights) & (weights > 0)
    if not m.any():
        return []

    p = periods[m]
    w = weights[m]
    order = np.argsort(w)[::-1]

    clusters = []

    def _recompute_center(periods_list, weights_list):
        pp = np.asarray(periods_list, np.float64)
        ww = np.asarray(weights_list, np.float64)
        lp_med = weighted_median_log(np.log(pp), ww)
        if np.isfinite(lp_med):
            return float(np.exp(lp_med))
        return float(np.nanmedian(pp))

    for idx in order:
        pi = float(p[idx])
        wi = float(w[idx])

        best_k = None
        best_dist = np.inf

        for k, cl in enumerate(clusters):
            c = float(cl["center"])
            if periods_match(
                pi, c,
                period_tol_frac=period_tol_frac,
                period_abs_tol_min=period_abs_tol_min,
            ):
                dist = abs(math.log(pi) - math.log(c))
                if dist < best_dist:
                    best_k = k
                    best_dist = dist

        if best_k is None:
            clusters.append(dict(
                periods=[pi],
                weights=[wi],
                center=pi,
                total_weight=wi,
                count=1,
            ))
        else:
            cl = clusters[best_k]
            cl["periods"].append(pi)
            cl["weights"].append(wi)
            cl["center"] = _recompute_center(cl["periods"], cl["weights"])
            cl["total_weight"] = float(np.sum(cl["weights"]))
            cl["count"] = int(len(cl["periods"]))

    merged = True
    while merged and len(clusters) > 1:
        merged = False
        clusters = sorted(clusters, key=lambda d: (d["total_weight"], d["count"]), reverse=True)
        used = np.zeros(len(clusters), dtype=bool)
        new_clusters = []

        for i in range(len(clusters)):
            if used[i]:
                continue

            base_p = list(clusters[i]["periods"])
            base_w = list(clusters[i]["weights"])
            base_center = float(clusters[i]["center"])
            used[i] = True

            for j in range(i + 1, len(clusters)):
                if used[j]:
                    continue

                cj = float(clusters[j]["center"])
                if periods_match(
                    base_center, cj,
                    period_tol_frac=period_tol_frac,
                    period_abs_tol_min=period_abs_tol_min,
                ):
                    base_p.extend(clusters[j]["periods"])
                    base_w.extend(clusters[j]["weights"])
                    used[j] = True
                    merged = True

            center = _recompute_center(base_p, base_w)
            new_clusters.append(dict(
                periods=base_p,
                weights=base_w,
                center=center,
                total_weight=float(np.sum(base_w)),
                count=int(len(base_p)),
            ))

        clusters = new_clusters

    clusters = sorted(clusters, key=lambda d: (d["total_weight"], d["count"]), reverse=True)
    if max_groups is not None:
        clusters = clusters[:int(max_groups)]
    return clusters

def period_group_maps(periods_m, z_m, keep_mask, sig_any, center_min, *, tol_frac, abs_tol_min):
    periods_m = np.asarray(periods_m, np.float32)
    z_m = np.asarray(z_m, np.float32)
    keep_mask = np.asarray(keep_mask, bool)
    sig_any = np.asarray(sig_any, bool)

    center = float(center_min)
    tol = float(tol_frac) * center

    valid = (
        sig_any[..., None] & keep_mask[..., None] &
        np.isfinite(periods_m) & (periods_m > 0) &
        np.isfinite(z_m) & (z_m > 0)
    )

    in_group = valid & ((np.abs(periods_m - center) <= tol) | (np.abs(periods_m - center) <= float(abs_tol_min)))
    z_group = np.where(in_group, z_m, np.float32(-np.inf))

    arg = np.argmax(z_group, axis=-1)
    score_map = z_group[
        np.arange(z_group.shape[0])[:, None],
        np.arange(z_group.shape[1])[None, :],
        arg
    ]
    group_mask = np.isfinite(score_map) & (score_map > -np.inf)

    period_map = np.take_along_axis(periods_m, arg[..., None], axis=-1)[..., 0]
    period_map = np.where(group_mask, period_map, np.nan).astype(np.float32)
    score_map = np.where(group_mask, score_map, np.nan).astype(np.float32)

    return group_mask.astype(bool), score_map, period_map

def build_component_record(
    *,
    day,
    roi_pick_index,
    detection_id,
    scale_idx,
    scale_name,
    N,
    S,
    period_group_id,
    period_group_center_min,
    component_idx,
    component_mask,
    score_map,
    period_map,
    roi_H,
    roi_W,
    roi_miny,
    roi_minx,
    H,
    W,
):
    rr, cc = np.where(component_mask)
    w = np.nan_to_num(score_map[rr, cc], nan=0.0, posinf=0.0, neginf=0.0)
    strength = float(w.sum())

    periods = period_map[rr, cc]
    m = np.isfinite(periods) & (periods > 0) & np.isfinite(w) & (w > 0)

    if m.sum() >= 1:
        p_comp = float(np.exp(weighted_median_log(np.log(periods[m].astype(np.float64)), w[m].astype(np.float64))))
        p_mean = float(np.average(periods[m].astype(np.float64), weights=w[m].astype(np.float64)))
    else:
        p_comp = np.nan
        p_mean = np.nan

    wsum = float(w.sum()) + 1e-12
    ci = float((w * rr).sum() / wsum)
    cj = float((w * cc).sum() / wsum)

    centroid_roi_x, centroid_roi_y = degraded_center_to_roi_xy(ci, cj, N, S)
    centroid_full_x = float(centroid_roi_x + roi_minx)
    centroid_full_y = float(centroid_roi_y + roi_miny)

    bbox_deg = bbox_from_mask(component_mask)
    if bbox_deg is None:
        bbox_deg = (None, None, None, None)

    roi_mask = patch_mask_to_roi(component_mask, N, S, roi_H, roi_W)
    bbox_roi = bbox_from_mask(roi_mask)
    if bbox_roi is None:
        bbox_roi = (None, None, None, None)

    if bbox_roi[0] is None:
        bbox_full = (None, None, None, None)
    else:
        bbox_full = (
            int(bbox_roi[0] + roi_miny),
            int(bbox_roi[1] + roi_minx),
            int(bbox_roi[2] + roi_miny),
            int(bbox_roi[3] + roi_minx),
        )

    return dict(
        detection_id=int(detection_id),
        day=str(day),
        roi_pick_index=int(roi_pick_index),
        scale_idx=int(scale_idx),
        scale_name=str(scale_name),
        N=int(N),
        S=int(S),
        period_group_id=int(period_group_id),
        period_group_center_min=float(period_group_center_min),
        component_idx=int(component_idx),
        period_min=float(p_comp) if np.isfinite(p_comp) else None,
        period_mean_min=float(p_mean) if np.isfinite(p_mean) else None,
        strength=float(strength),
        area_degpx=int(component_mask.sum()),
        centroid_deg_i=float(ci),
        centroid_deg_j=float(cj),
        centroid_roi_x=float(centroid_roi_x),
        centroid_roi_y=float(centroid_roi_y),
        centroid_full_x=float(centroid_full_x),
        centroid_full_y=float(centroid_full_y),
        bbox_deg_min_i=bbox_deg[0],
        bbox_deg_min_j=bbox_deg[1],
        bbox_deg_max_i=bbox_deg[2],
        bbox_deg_max_j=bbox_deg[3],
        bbox_roi_min_y=bbox_roi[0],
        bbox_roi_min_x=bbox_roi[1],
        bbox_roi_max_y=bbox_roi[2],
        bbox_roi_max_x=bbox_roi[3],
        bbox_full_min_y=bbox_full[0],
        bbox_full_min_x=bbox_full[1],
        bbox_full_max_y=bbox_full[2],
        bbox_full_max_x=bbox_full[3],
        roi_mask=roi_mask.astype(bool),
    )

def extract_scale_period_components(
    *,
    day,
    roi_pick_index,
    scale_idx,
    scale_name,
    N,
    S,
    maps,
    roi_H,
    roi_W,
    roi_miny,
    roi_minx,
    H,
    W,
    period_tol_frac=0.05,
    period_abs_tol_min=5.0,
    min_component_area=1,
    max_period_groups=None,
    max_components_per_group=None,
    base_detection_id=0,
):
    valid = (
        maps["sig_any"][..., None] & maps["keep_mask"][..., None] &
        np.isfinite(maps["periods_m"]) & (maps["periods_m"] > 0) &
        np.isfinite(maps["z_m"]) & (maps["z_m"] > 0)
    )

    periods = maps["periods_m"][valid].astype(np.float64)
    weights = maps["z_m"][valid].astype(np.float64)

    clusters = cluster_period_values(
        periods, weights,
        period_tol_frac=period_tol_frac,
        period_abs_tol_min=period_abs_tol_min,
        max_groups=max_period_groups,
    )

    detections = []
    group_summaries = []
    detection_id = int(base_detection_id)

    for gid, cl in enumerate(clusters):
        center_min = float(cl["center"])

        group_mask, score_map, period_map = period_group_maps(
            maps["periods_m"], maps["z_m"],
            maps["keep_mask"], maps["sig_any"],
            center_min,
            tol_frac=period_tol_frac,
            abs_tol_min=period_abs_tol_min,
        )

        if not np.any(group_mask):
            continue

        lbl = label(group_mask.astype(np.uint8), connectivity=2)
        props = regionprops(lbl)

        comp_infos = []
        for r in props:
            if r.area < int(min_component_area):
                continue

            comp_mask = (lbl == r.label)
            rr = r.coords[:, 0]
            cc = r.coords[:, 1]
            comp_strength = float(np.nansum(np.nan_to_num(score_map[rr, cc], nan=0.0)))
            if not np.isfinite(comp_strength) or comp_strength <= 0:
                continue

            comp_infos.append((comp_strength, comp_mask))

        comp_infos.sort(key=lambda x: x[0], reverse=True)
        if max_components_per_group is not None:
            comp_infos = comp_infos[:int(max_components_per_group)]

        if not comp_infos:
            continue

        group_summaries.append(dict(
            scale_idx=int(scale_idx),
            scale_name=str(scale_name),
            N=int(N),
            S=int(S),
            period_group_id=int(gid),
            period_group_center_min=float(center_min),
            n_pixels=int(group_mask.sum()),
            n_components=int(len(comp_infos)),
            total_group_strength=float(np.nansum(np.nan_to_num(score_map[group_mask], nan=0.0))),
        ))

        for comp_idx, (_, comp_mask) in enumerate(comp_infos):
            rec = build_component_record(
                day=day,
                roi_pick_index=roi_pick_index,
                detection_id=detection_id,
                scale_idx=scale_idx,
                scale_name=scale_name,
                N=N,
                S=S,
                period_group_id=gid,
                period_group_center_min=center_min,
                component_idx=comp_idx,
                component_mask=comp_mask,
                score_map=score_map,
                period_map=period_map,
                roi_H=roi_H,
                roi_W=roi_W,
                roi_miny=roi_miny,
                roi_minx=roi_minx,
                H=H,
                W=W,
            )
            detections.append(rec)
            detection_id += 1

    return group_summaries, detections

def cluster_global_period_families(detections, *, period_tol_frac=0.05, period_abs_tol_min=5.0):
    if not detections:
        return [], []

    periods = np.array([
        d["period_min"] if d["period_min"] is not None else np.nan for d in detections
    ], dtype=np.float64)
    weights = np.array([max(1e-12, float(d["strength"])) for d in detections], dtype=np.float64)

    clusters = cluster_period_values(
        periods, weights,
        period_tol_frac=period_tol_frac,
        period_abs_tol_min=period_abs_tol_min,
        max_groups=None,
    )

    family_centers = [float(cl["center"]) for cl in clusters]

    for d in detections:
        p = d["period_min"]
        if p is None or not np.isfinite(float(p)):
            d["family_id"] = None
            d["family_center_min"] = None
            continue

        best_fid = None
        best_dist = np.inf
        for fid, c in enumerate(family_centers):
            if periods_match(
                p, c,
                period_tol_frac=period_tol_frac,
                period_abs_tol_min=period_abs_tol_min,
            ):
                dist = abs(math.log(float(p)) - math.log(float(c)))
                if dist < best_dist:
                    best_dist = dist
                    best_fid = fid

        if best_fid is None:
            d["family_id"] = None
            d["family_center_min"] = None
        else:
            d["family_id"] = int(best_fid)
            d["family_center_min"] = float(family_centers[best_fid])

    family_summaries = []
    for fid, center in enumerate(family_centers):
        fam_det = [d for d in detections if d.get("family_id") == fid]
        if not fam_det:
            continue

        family_summaries.append(dict(
            family_id=int(fid),
            family_center_min=float(center),
            n_components=int(len(fam_det)),
            n_scales=int(len(set(int(d["scale_idx"]) for d in fam_det))),
            total_strength=float(sum(float(d["strength"]) for d in fam_det)),
            scales_present=sorted(set(int(d["N"]) for d in fam_det), reverse=True),
        ))

    family_summaries.sort(key=lambda d: (d["n_scales"], d["total_strength"]), reverse=True)
    return family_centers, family_summaries
