"""Plotting functions for analysis diagnostics and detection products."""

import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import CNN_FREQUENCY_GRID

def plot_full_disk_filaments_bboxes(outpath, image0, mask0, bboxes, *, title):
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image0, cmap="gray", origin="lower")
    plt.imshow((mask0 > 0).astype(float), cmap="Reds", alpha=0.35, origin="lower")
    for i, (miny, minx, maxy, maxx) in enumerate(bboxes):
        plt.plot([minx, maxx, maxx, minx, minx], [miny, miny, maxy, maxy, miny], "r-", lw=1.0)
        plt.text(
            minx, miny, f"{i}",
            color="yellow", fontsize=8,
            bbox=dict(facecolor="black", alpha=0.4, pad=1.0),
        )
    plt.title(title)
    plt.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

def plot_cp_calibration(outdir, m, s, qk, delta):
    freqs = CNN_FREQUENCY_GRID.astype(np.float64)
    periods_min = (1.0 / freqs) / 60.0
    factor = np.exp(m + qk * s)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.loglog(periods_min, factor, lw=2)
    ax.set_xlabel("Period (min)")
    ax.set_ylabel("Multiplicative factor")
    ax.set_title(f"Global CP factor ({100*(1-delta):.5f}% upper quantile)")
    ax.invert_xaxis()

    ax = axes[1]
    ax.plot(periods_min, m, label="m")
    ax.plot(periods_min, s, label="s")
    ax.plot(periods_min, qk, label="qk", ls="--")
    ax.set_xlabel("Period (min)")
    ax.set_ylabel("Value")
    ax.set_title("CP residual statistics")
    ax.invert_xaxis()
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "cp_calibration.png"), dpi=150)
    plt.close(fig)

def plot_scale_summary_maps(outdir, scale_name, stack0, union_cov, maps):
    os.makedirs(outdir, exist_ok=True)
    per = maps["best_period_min"]
    z = maps["best_z"]
    sig = maps["sig_any"]
    coh = maps["coherence_logp_std"]

    fig = plt.figure(figsize=(18, 10))
    ax = plt.subplot(231)
    ax.imshow(stack0, cmap="gray", origin="lower")
    ax.set_title(f"{scale_name}: degraded frame[0]")

    ax = plt.subplot(232)
    im = ax.imshow(per, origin="lower")
    ax.set_title(f"{scale_name}: best period (min)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = plt.subplot(233)
    im = ax.imshow(z, origin="lower")
    ax.set_title(f"{scale_name}: best pxx/tau")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = plt.subplot(234)
    ax.imshow(sig, cmap="gray", origin="lower")
    ax.set_title(f"{scale_name}: sig_any")

    ax = plt.subplot(235)
    im = ax.imshow(union_cov, origin="lower")
    ax.set_title(f"{scale_name}: union mask coverage")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = plt.subplot(236)
    im = ax.imshow(coh, origin="lower")
    ax.set_title(f"{scale_name}: coherence std(logP)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f"{scale_name}_scale_maps.png"), dpi=140)
    plt.close(fig)

def plot_scale_period_components(outdir, scale_name, roi_img0, union_mask, detections_in_scale):
    if not detections_in_scale:
        return

    fig = plt.figure(figsize=(10, 8))
    plt.imshow(roi_img0, cmap="gray", origin="lower")
    plt.contour(union_mask.astype(float), levels=[0.5], linewidths=0.8)

    for det in detections_in_scale:
        roi_mask = det["roi_mask"]
        if roi_mask is not None and np.any(roi_mask):
            plt.contour(roi_mask.astype(float), levels=[0.5], linewidths=1.2)
        plt.text(
            det["centroid_roi_x"], det["centroid_roi_y"],
            f"{det['period_min']:.1f}m",
            fontsize=7,
            ha="center", va="center",
            bbox=dict(facecolor="white", alpha=0.6, pad=0.5),
        )

    plt.title(f"{scale_name}: connected components by period")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f"{scale_name}_period_components.png"), dpi=150)
    plt.close(fig)

def plot_period_scale_scatter(outpath, detections, *, title):
    if not detections:
        return

    periods = np.array([d["period_min"] for d in detections if d["period_min"] is not None], dtype=np.float64)
    Ns = np.array([d["N"] for d in detections if d["period_min"] is not None], dtype=np.float64)
    strengths = np.array([d["strength"] for d in detections if d["period_min"] is not None], dtype=np.float64)
    fam_ids = np.array([
        -1 if d.get("family_id") is None else int(d["family_id"])
        for d in detections if d["period_min"] is not None
    ], dtype=np.int32)

    if periods.size == 0:
        return

    sizes = 20.0 + 12.0 * np.sqrt(np.maximum(strengths, 0.0))
    color_vals = fam_ids if np.any(fam_ids >= 0) else strengths

    fig = plt.figure(figsize=(10, 7))
    sc = plt.scatter(periods, Ns, s=sizes, c=color_vals, alpha=0.8)
    plt.xscale("log")
    plt.xlabel("Period (min)")
    plt.ylabel("Scale N")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.25)

    cbar = plt.colorbar(sc)
    if np.any(fam_ids >= 0):
        cbar.set_label("Family ID")
    else:
        cbar.set_label("Strength")

    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_period_family_spatial_maps(outdir, roi_img0, union_mask, scales, detections, family_summaries):
    os.makedirs(outdir, exist_ok=True)
    if not detections or not family_summaries:
        return

    scale_name_by_idx = {i: sc["name"] for i, sc in enumerate(scales)}

    for fam in family_summaries:
        fid = fam["family_id"]
        fam_center = fam["family_center_min"]
        fam_dets = [d for d in detections if d.get("family_id") == fid]
        if not fam_dets:
            continue

        dets_by_scale = defaultdict(list)
        for d in fam_dets:
            dets_by_scale[int(d["scale_idx"])].append(d)

        scale_idxs = sorted(dets_by_scale.keys())
        n_panels = len(scale_idxs)
        ncols = min(3, n_panels)
        nrows = int(math.ceil(n_panels / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
        axes = axes.ravel()

        for ax in axes[n_panels:]:
            ax.axis("off")

        for ax, sidx in zip(axes[:n_panels], scale_idxs):
            ax.imshow(roi_img0, cmap="gray", origin="lower")
            ax.contour(union_mask.astype(float), levels=[0.5], linewidths=0.8)

            for det in dets_by_scale[sidx]:
                roi_mask = det["roi_mask"]
                if roi_mask is not None and np.any(roi_mask):
                    ax.contour(roi_mask.astype(float), levels=[0.5], linewidths=1.5)
                ax.plot(det["centroid_roi_x"], det["centroid_roi_y"], marker="x", ms=8, mew=1.5)
                ax.text(
                    det["centroid_roi_x"], det["centroid_roi_y"],
                    f"{det['period_min']:.1f}m",
                    fontsize=7,
                    ha="left", va="bottom",
                    bbox=dict(facecolor="white", alpha=0.6, pad=0.5),
                )

            ax.set_title(f"{scale_name_by_idx[sidx]} | {len(dets_by_scale[sidx])} component(s)")

        fig.suptitle(
            f"Family {fid} | center ~ {fam_center:.2f} min | "
            f"{fam['n_scales']} scale(s), {fam['n_components']} component(s)"
        )
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, f"family_{fid:03d}_P{fam_center:.2f}min.png"), dpi=150)
        plt.close(fig)
