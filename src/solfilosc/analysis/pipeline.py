"""Driver for the CNN + conformal-prediction oscillation analysis pipeline."""

import argparse
import csv
import os

import h5py as h5
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import binary_dilation

from .constants import (
    CP_DELTA,
    DETECTION_FMAX_HZ,
    DETECTION_FMIN_HZ,
    N_FILAMENT_WORKERS,
    N_PIXEL_WORKERS,
)
from .cnn import _build_scaler, load_cnn
from .cp_calibration import ensure_cp_cache_for_day, get_cp_cache_path
from .degradation import (
    apply_null_time_transform,
    choose_scales_distinct_quotients,
    choose_scales_log_ladder,
    degrade_mask_coverage,
    degrade_stack_unweighted,
)
from .detection import analyze_degraded_stack_multipeak_cp, cluster_global_period_families, extract_scale_period_components
from .events import assign_event_ids_within_families, summarize_reported_families, write_event_summary_csv
from .plotting import (
    plot_cp_calibration,
    plot_full_disk_filaments_bboxes,
    plot_period_family_spatial_maps,
    plot_period_scale_scatter,
    plot_scale_period_components,
    plot_scale_summary_maps,
)
from .roi import expand_bbox, list_candidate_regions_from_mask, select_roi_bbox_from_first_mask
from .writers import _strip_masks_for_serialization, write_component_csv, write_json

def run_once(
    *,
    day,
    data_h5,
    masks_h5,
    outdir,
    cnn_weights_path,
    cp_n_calib=100_000,
    cp_delta=0.00001,
    cp_batch_size_cnn=4096,
    detect_fmin_hz=DETECTION_FMIN_HZ,
    detect_fmax_hz=DETECTION_FMAX_HZ,
    roi_pick="largest",
    roi_pick_index=0,
    roi_min_area=2000,
    roi_pad=15,
    union_min_frames=6 * 60,
    union_dilate_iter=2,
    cov_thr=0.30,
    top_m_peaks=4,
    peak_min_prom_frac=0.05,
    scales_mode="distinct_quotients",
    Nmax=51,
    Nmin=20,
    n_scales=12,
    overlap_frac=0.75,
    period_tol_frac=0.05,
    period_abs_tol_min=5.0,
    min_component_area=1,
    max_period_groups_per_scale=None,
    max_components_per_group=None,
    null_mode="none",
    null_seed=0,
    n_jobs=N_PIXEL_WORKERS,
    cp_cache_path=None,
):
    os.makedirs(outdir, exist_ok=True)

    plots_out = os.path.join(outdir, "plots")
    tables_out = os.path.join(outdir, "tables")
    families_out = os.path.join(outdir, "period_families")
    os.makedirs(plots_out, exist_ok=True)
    os.makedirs(tables_out, exist_ok=True)
    os.makedirs(families_out, exist_ok=True)

    print(f"\n[run] day={day}  roi_pick_index={roi_pick_index}  outdir={outdir}")

    with h5.File(data_h5, "r") as hf:
        images = np.array(hf["time_series"][:], dtype=np.float32)
        tdeltas = np.array(hf["tdeltas"][:], dtype=np.float32)

    with h5.File(masks_h5, "r") as hm:
        masks = np.array(hm["masks"][:], dtype=np.uint8)

    order = np.argsort(tdeltas)
    tdeltas = tdeltas[order]
    images = images[order]
    masks = masks[order]

    H, W = images.shape[1], images.shape[2]
    dt_med = np.median(np.diff(tdeltas[np.isfinite(tdeltas)]))
    print(f"[data] images={images.shape}  dt_med={dt_med/60:.2f} min")

    cnn_model = load_cnn(cnn_weights_path)
    scaler = _build_scaler()

    if cp_cache_path is None:
        cp_cache_path = get_cp_cache_path(
            os.path.abspath(os.path.join(outdir, os.pardir)),
            day,
            cp_delta
        )

    if not os.path.exists(cp_cache_path):
        raise FileNotFoundError(f"Expected precomputed CP cache was not found: {cp_cache_path}")

    print(f"[CP] Loading cached calibration stats from {cp_cache_path}")
    _cpdata = np.load(cp_cache_path)
    m_cp = _cpdata["m"]
    s_cp = _cpdata["s"]
    qk = _cpdata["qk"]

    if not (np.all(np.isfinite(m_cp)) and np.all(np.isfinite(s_cp)) and np.all(np.isfinite(qk))):
        raise RuntimeError(
            f"Loaded CP cache contains non-finite values: {cp_cache_path}. "
            "Delete it and rebuild calibration."
        )

    plot_cp_calibration(plots_out, m_cp, s_cp, qk, delta=float(cp_delta))

    day_root = os.path.abspath(os.path.join(outdir, os.pardir))
    os.makedirs(day_root, exist_ok=True)
    day_overview_png = os.path.join(day_root, "full_disk_filaments_bboxes.png")
    if not os.path.exists(day_overview_png):
        regs_sorted = list_candidate_regions_from_mask(masks[0], min_area=int(roi_min_area))
        plot_full_disk_filaments_bboxes(
            day_overview_png, images[0], masks[0],
            [r.bbox for r in regs_sorted],
            title=f"{day} full disk: filaments + bbox indices (area desc)\nroi_min_area={roi_min_area}"
        )

    bbox = expand_bbox(
        select_roi_bbox_from_first_mask(
            masks[0],
            min_area=roi_min_area,
            pick=roi_pick,
            pick_index=roi_pick_index,
        ),
        pad=roi_pad, H=H, W=W
    )
    roi_miny, roi_minx, roi_maxy, roi_maxx = bbox
    roi_H = roi_maxy - roi_miny
    roi_W = roi_maxx - roi_minx

    ROI_images = images[:, roi_miny:roi_maxy, roi_minx:roi_maxx]
    ROI_masks = masks[:, roi_miny:roi_maxy, roi_minx:roi_maxx]

    union_mask = (ROI_masks > 0).sum(axis=0) >= int(union_min_frames)
    if union_dilate_iter and union_dilate_iter > 0:
        union_mask = binary_dilation(union_mask, iterations=int(union_dilate_iter))

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(ROI_images[0], cmap="gray", origin="lower")
    plt.contour(union_mask.astype(float), levels=[0.5], linewidths=1.2)
    plt.title("ROI frame[0] + persistent union mask contour")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_out, "roi_union_mask.png"), dpi=140)
    plt.close(fig)

    if scales_mode == "distinct_quotients":
        Ns = choose_scales_distinct_quotients(roi_H, start=int(Nmax), stop=int(Nmin) - 1)
    elif scales_mode == "log_ladder":
        Ns = choose_scales_log_ladder(int(Nmax), int(Nmin), n=int(n_scales))
    else:
        raise ValueError("scales_mode must be distinct_quotients or log_ladder")

    scales = [
        dict(
            N=int(N),
            S=max(1, round(N * (1 - overlap_frac))),
            name=f"N{int(N)}_S{max(1, round(N * (1 - overlap_frac)))}"
        )
        for N in Ns
    ]
    print(f"[scales] {len(scales)}: " + ", ".join(sc["name"] for sc in scales))

    rng = np.random.default_rng(int(null_seed))

    all_group_summaries = []
    all_detections = []
    per_scale_cache = []
    next_detection_id = 0

    for scale_idx, sc in enumerate(scales):
        N, S, name = sc["N"], sc["S"], sc["name"]
        print(f"\n[scale {scale_idx+1}/{len(scales)}] {name}")

        stack = degrade_stack_unweighted(ROI_images, N=N, S=S)
        stack -= np.nanmean(stack, axis=0, keepdims=True)
        stack = apply_null_time_transform(stack, null_mode, rng)

        cov = degrade_mask_coverage(union_mask, N=N, S=S)
        keep_mask = (cov >= float(cov_thr))

        maps = analyze_degraded_stack_multipeak_cp(
            tdeltas, stack,
            cnn_model, scaler,
            m_cp, s_cp, qk,
            keep_mask=keep_mask,
            top_m_peaks=int(top_m_peaks),
            peak_min_prom_frac=float(peak_min_prom_frac),
            n_jobs=int(n_jobs),
            batch_size_cnn=int(cp_batch_size_cnn),
            detect_fmin_hz=float(detect_fmin_hz),
            detect_fmax_hz=float(detect_fmax_hz),
        )

        n_sig = int((maps["sig_any"] & maps["keep_mask"]).sum())
        print(f"[scale] sig pixels={n_sig}")

        plot_scale_summary_maps(plots_out, name, stack0=stack[0], union_cov=cov, maps=maps)

        group_summaries, detections = extract_scale_period_components(
            day=day,
            roi_pick_index=roi_pick_index,
            scale_idx=scale_idx,
            scale_name=name,
            N=N,
            S=S,
            maps=maps,
            roi_H=roi_H,
            roi_W=roi_W,
            roi_miny=roi_miny,
            roi_minx=roi_minx,
            H=H,
            W=W,
            period_tol_frac=float(period_tol_frac),
            period_abs_tol_min=float(period_abs_tol_min),
            min_component_area=int(min_component_area),
            max_period_groups=max_period_groups_per_scale,
            max_components_per_group=max_components_per_group,
            base_detection_id=next_detection_id,
        )

        print(f"[scale] period groups={len(group_summaries)}  components={len(detections)}")

        detections_sorted = sorted(
            detections,
            key=lambda d: (float(d["period_min"]) if d["period_min"] is not None else np.inf, -float(d["strength"]))
        )
        plot_scale_period_components(
            plots_out, name, ROI_images[0], union_mask, detections_sorted
        )

        scale_csv = os.path.join(tables_out, f"{name}_components.csv")
        scale_json = os.path.join(tables_out, f"{name}_components.json")
        write_component_csv(scale_csv, _strip_masks_for_serialization(detections))
        write_json(scale_json, dict(
            day=str(day),
            roi_pick_index=int(roi_pick_index),
            scale_idx=int(scale_idx),
            scale_name=str(name),
            N=int(N),
            S=int(S),
            n_groups=int(len(group_summaries)),
            n_components=int(len(detections)),
            groups=group_summaries,
            components=_strip_masks_for_serialization(detections),
        ))

        all_group_summaries.extend(group_summaries)
        all_detections.extend(detections)
        next_detection_id += len(detections)
        per_scale_cache.append(dict(scale_idx=scale_idx, scale_name=name, N=N, S=S, maps=maps))

    if not all_detections:
        print("[done] No connected components found.")
        return

    _, _all_family_summaries = cluster_global_period_families(
        all_detections,
        period_tol_frac=float(period_tol_frac),
        period_abs_tol_min=float(period_abs_tol_min),
    )

    event_summaries = assign_event_ids_within_families(
        all_detections,
        bbox_iou_thr=0.05,
        centroid_scale_factor=2,
        max_scale_idx_gap=3,
        allow_same_scale=False,
        min_event_scales=3,
    )

    # Keep only detections that belong to events seen in >= 3 scales
    all_detections = [d for d in all_detections if d.get("event_id") is not None]

    if not all_detections:
        print("[done] No events survived the min_event_scales >= 3 filter.")
        return

    family_summaries = summarize_reported_families(all_detections)

    plot_period_scale_scatter(
        os.path.join(plots_out, "period_vs_scale_components.png"),
        all_detections,
        title="Connected-component detections: period vs scale",
    )

    plot_period_family_spatial_maps(
        families_out,
        ROI_images[0],
        union_mask,
        scales,
        all_detections,
        family_summaries,
    )

    all_components_csv = os.path.join(tables_out, "all_components.csv")
    all_components_json = os.path.join(tables_out, "all_components.json")
    family_summary_json = os.path.join(tables_out, "period_families.json")
    family_summary_csv = os.path.join(tables_out, "period_families.csv")
    event_summary_json = os.path.join(tables_out, "events.json")
    event_summary_csv = os.path.join(tables_out, "events.csv")

    write_component_csv(all_components_csv, _strip_masks_for_serialization(all_detections))
    write_json(all_components_json, dict(
        day=str(day),
        roi_pick=str(roi_pick),
        roi_pick_index=int(roi_pick_index),
        roi_bbox=dict(
            min_y=int(roi_miny), min_x=int(roi_minx),
            max_y=int(roi_maxy), max_x=int(roi_maxx),
        ),
        cp_delta=float(cp_delta),
        n_scales=int(len(scales)),
        n_components=int(len(all_detections)),
        n_families=int(len(family_summaries)),
        n_events=int(len(event_summaries)),
        components=_strip_masks_for_serialization(all_detections),
    ))

    write_json(family_summary_json, dict(
        day=str(day),
        roi_pick_index=int(roi_pick_index),
        n_families=int(len(family_summaries)),
        families=family_summaries,
    ))

    write_json(event_summary_json, dict(
        day=str(day),
        roi_pick_index=int(roi_pick_index),
        n_events=int(len(event_summaries)),
        events=event_summaries,
    ))

    write_event_summary_csv(event_summary_csv, event_summaries)

    with open(family_summary_csv, "w", newline="") as f:
        fields = [
            "family_id", "family_center_min", "n_components",
            "n_scales", "total_strength", "scales_present",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in family_summaries:
            row = dict(rec)
            row["scales_present"] = ",".join(str(x) for x in rec["scales_present"])
            writer.writerow(row)

    summary_txt = os.path.join(tables_out, "summary.txt")
    with open(summary_txt, "w") as f:
        f.write("CONNECTED-COMPONENT PERIOD CATALOG\n")
        f.write("=================================\n\n")
        f.write(f"day: {day}\n")
        f.write(f"roi_pick_index: {roi_pick_index}\n")
        f.write(f"cp_delta: {cp_delta}\n")
        f.write(f"n_scales: {len(scales)}\n")
        f.write(f"n_components: {len(all_detections)}\n")
        f.write(f"n_families: {len(family_summaries)}\n")
        f.write(f"n_events: {len(event_summaries)}\n\n")

        f.write("Families:\n")
        for fam in family_summaries:
            f.write(
                f"  family_id={fam['family_id']:>3d}  "
                f"center={fam['family_center_min']:.3f} min  "
                f"n_scales={fam['n_scales']:>2d}  "
                f"n_components={fam['n_components']:>3d}  "
                f"strength={fam['total_strength']:.2f}\n"
            )

    print(
        f"[done] {len(all_detections)} component(s), "
        f"{len(family_summaries)} family/families, "
        f"{len(event_summaries)} event(s) -> {tables_out}"
    )


def run_filament(
    day: str,
    index: int = 0,
    *,
    data_root: str = "data",
    results_root: str = "results",
    cnn_weights_path: str = "CNN/BestFit/BestFitWeights.h5",
    cp_cache_path: str | None = None,
):
    """Run the notebook-equivalent analysis for one filament ROI."""
    data_h5 = os.path.join(data_root, day, "updated", f"{day}_data_modified.h5")
    masks_h5 = os.path.join(data_root, day, "updated", f"{day}_masks.h5")
    outdir = os.path.join(results_root, day, str(index))

    run_once(
        day=day,
        data_h5=data_h5,
        masks_h5=masks_h5,
        outdir=outdir,
        cnn_weights_path=cnn_weights_path,
        cp_n_calib=1_000_000,
        cp_delta=CP_DELTA,
        cp_batch_size_cnn=4096,
        detect_fmin_hz=DETECTION_FMIN_HZ,
        detect_fmax_hz=DETECTION_FMAX_HZ,
        roi_pick="index",
        roi_pick_index=index,
        roi_min_area=750,
        roi_pad=25,
        union_min_frames=6 * 60,
        union_dilate_iter=2,
        cov_thr=0.30,
        top_m_peaks=4,
        peak_min_prom_frac=0.10,
        scales_mode="distinct_quotients",
        Nmax=71,
        Nmin=9,
        n_scales=12,
        overlap_frac=0.75,
        period_tol_frac=0.05,
        period_abs_tol_min=5.0,
        min_component_area=1,
        max_period_groups_per_scale=None,
        max_components_per_group=None,
        null_mode="none",
        null_seed=0,
        n_jobs=N_PIXEL_WORKERS,
        cp_cache_path=cp_cache_path,
    )


def run_day(
    day: str,
    *,
    data_root: str = "data",
    results_root: str = "results",
    cnn_weights_path: str = "CNN/BestFit/BestFitWeights.h5",
    n_filament_workers: int = N_FILAMENT_WORKERS,
    n_pixel_workers: int = N_PIXEL_WORKERS,
):
    """Build the CP cache for one day and analyze every filament in masks[0]."""
    data_h5 = os.path.join(data_root, day, "updated", f"{day}_data_modified.h5")
    masks_h5 = os.path.join(data_root, day, "updated", f"{day}_masks.h5")

    with h5.File(masks_h5, "r") as hf:
        masks = np.array(hf["masks"][:], dtype=np.uint8)

    n_filaments = len(list_candidate_regions_from_mask(masks[0], min_area=750))
    print(f"\n=== Day {day}: {n_filaments} filaments ===")

    cp_cache_path = ensure_cp_cache_for_day(
        day=day,
        data_h5=data_h5,
        masks_h5=masks_h5,
        outroot=os.path.join(results_root, day),
        cnn_weights_path=cnn_weights_path,
        cp_n_calib=1_000_000,
        cp_delta=CP_DELTA,
        cp_batch_size_cnn=4096,
        n_jobs=int(n_pixel_workers),
    )

    Parallel(n_jobs=int(n_filament_workers), verbose=5)(
        delayed(run_filament)(
            day=day,
            index=i,
            data_root=data_root,
            results_root=results_root,
            cnn_weights_path=cnn_weights_path,
            cp_cache_path=cp_cache_path,
        )
        for i in range(n_filaments)
    )


def run_all_days(
    *,
    data_root: str = "data",
    results_root: str = "results",
    cnn_weights_path: str = "CNN/BestFit/BestFitWeights.h5",
    n_filament_workers: int = N_FILAMENT_WORKERS,
    n_pixel_workers: int = N_PIXEL_WORKERS,
):
    days = sorted(
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name))
    )
    for day in days:
        run_day(
            day,
            data_root=data_root,
            results_root=results_root,
            cnn_weights_path=cnn_weights_path,
            n_filament_workers=n_filament_workers,
            n_pixel_workers=n_pixel_workers,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run the solar-filament oscillation analysis pipeline.")
    parser.add_argument("--day", help="Day to analyze, e.g. 20140102. If omitted, all data/<day>/ folders are processed.")
    parser.add_argument("--filament-index", type=int, help="Analyze one filament index instead of every filament for the day.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--cnn-weights", default="CNN/BestFit/BestFitWeights.h5")
    parser.add_argument("--cp-cache", default=None, help="Optional existing CP cache for --filament-index runs.")
    parser.add_argument("--filament-workers", type=int, default=N_FILAMENT_WORKERS)
    parser.add_argument("--pixel-workers", type=int, default=N_PIXEL_WORKERS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.day and args.filament_index is not None:
        run_filament(
            args.day,
            args.filament_index,
            data_root=args.data_root,
            results_root=args.results_root,
            cnn_weights_path=args.cnn_weights,
            cp_cache_path=args.cp_cache,
        )
    elif args.day:
        run_day(
            args.day,
            data_root=args.data_root,
            results_root=args.results_root,
            cnn_weights_path=args.cnn_weights,
            n_filament_workers=args.filament_workers,
            n_pixel_workers=args.pixel_workers,
        )
    else:
        run_all_days(
            data_root=args.data_root,
            results_root=args.results_root,
            cnn_weights_path=args.cnn_weights,
            n_filament_workers=args.filament_workers,
            n_pixel_workers=args.pixel_workers,
        )


if __name__ == "__main__":
    main()
