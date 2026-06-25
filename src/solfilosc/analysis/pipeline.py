"""Driver for the CNN + conformal-prediction oscillation analysis pipeline."""

import os

import h5py as h5
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
from .events import assign_event_ids_within_families, summarize_reported_families
from .plotting import (
    plot_event,
    plot_full_disk_filaments_bboxes,
    plot_period_family_spatial_maps,
)
from .roi import expand_bbox, list_candidate_regions_from_mask, select_roi_bbox_from_first_mask
from .writers import write_json
from ..paths import day_dir, discover_days

def run_once(
    *,
    day,
    data_h5,
    masks_h5,
    day_dir,
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
    cp_cache_path=None,
    plot_period_families=False,
):
    os.makedirs(day_dir, exist_ok=True)

    print(f"\n[run] day={day}  roi_pick_index={roi_pick_index}  day_dir={day_dir}")

    with h5.File(data_h5, "r") as hf:
        images = np.array(hf["time_series"][:], dtype=np.float32)
        tdeltas = np.array(hf["tdeltas"][:], dtype=np.float32)

    with h5.File(masks_h5, "r") as hm:
        masks = np.array(hm["masks"][:], dtype=np.uint8)

    if images.shape[0] != tdeltas.shape[0] or masks.shape[0] != images.shape[0]:
        raise ValueError(
            f"Time-axis mismatch: images={images.shape}, "
            f"tdeltas={tdeltas.shape}, masks={masks.shape}"
        )

    order = np.argsort(tdeltas, kind="stable")
    if not np.array_equal(order, np.arange(order.size)):
        tdeltas = tdeltas[order]
        images = images[order]
        masks = masks[order]

    H, W = images.shape[1], images.shape[2]
    dt_med = np.median(np.diff(tdeltas[np.isfinite(tdeltas)]))
    print(f"[data] images={images.shape}  dt_med={dt_med/60:.2f} min")

    cnn_model = load_cnn(cnn_weights_path)
    scaler = _build_scaler()

    if cp_cache_path is None:
        cp_cache_path = get_cp_cache_path(day_dir, day, cp_delta)

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

    day_overview_png = os.path.join(day_dir, "full_disk_filaments_bboxes.png")
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

    all_detections = []
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

        all_detections.extend(detections)
        next_detection_id += len(detections)

    if not all_detections:
        print("[done] No connected components found.")
        return None

    _, _all_family_summaries = cluster_global_period_families(
        all_detections,
        period_tol_frac=float(period_tol_frac),
        period_abs_tol_min=float(period_abs_tol_min),
    )

    min_event_scales = 4
    event_summaries = assign_event_ids_within_families(
        all_detections,
        bbox_iou_thr=0.05,
        centroid_scale_factor=2,
        max_scale_idx_gap=3,
        allow_same_scale=False,
        min_event_scales=min_event_scales,
    )

    # Keep only detections that belong to events seen in >= min_event_scales scales
    all_detections = [d for d in all_detections if d.get("event_id") is not None]

    if not all_detections:
        print(f"[done] No events survived the min_event_scales >= {min_event_scales} filter.")
        return None

    family_summaries = summarize_reported_families(all_detections)

    if plot_period_families:
        fam_dir = os.path.join(day_dir, "period_families")
        os.makedirs(fam_dir, exist_ok=True)
        plot_period_family_spatial_maps(
            fam_dir,
            ROI_images[0],
            union_mask,
            scales,
            all_detections,
            family_summaries,
            fname_prefix=f"filament{roi_pick_index}_",
        )

    print(
        f"[done] {len(all_detections)} component(s), "
        f"{len(family_summaries)} family/families, "
        f"{len(event_summaries)} event(s)"
    )

    return dict(
        roi_pick_index=int(roi_pick_index),
        roi_bbox=dict(
            min_y=int(roi_miny), min_x=int(roi_minx),
            max_y=int(roi_maxy), max_x=int(roi_maxx),
        ),
        events=event_summaries,
        roi_image0=np.asarray(ROI_images[0], dtype=np.float32),
    )


def run_filament(
    day: str,
    index: int = 0,
    *,
    data_root: str = "data",
    results_root: str = "results",
    cnn_weights_path: str = "CNN/BestFit/BestFitWeights.h5",
    cp_cache_path: str | None = None,
    n_pixel_workers: int = N_PIXEL_WORKERS,
    plot_period_families: bool = False,
):
    """Run the notebook-equivalent analysis for one filament ROI.

    Returns the per-filament result dict (events + plot data) for day-level
    aggregation, or None if no events were found.
    """
    data_h5 = os.path.join(day_dir(data_root, day), f"{day}.h5")
    masks_h5 = os.path.join(day_dir(data_root, day), f"{day}_masks.h5")

    return run_once(
        day=day,
        data_h5=data_h5,
        masks_h5=masks_h5,
        day_dir=day_dir(results_root, day),
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
        n_jobs=int(n_pixel_workers),
        cp_cache_path=cp_cache_path,
        plot_period_families=plot_period_families,
    )


def write_day_outputs(day, results_root, filament_results):
    """Aggregate per-filament results into one per-day events JSON plus one plot per event.

    `filament_results` is the list returned by the per-filament runs (some entries may be
    None when a filament produced no events). Events are numbered with a global per-day
    index used both in the JSON and the plot filenames (``<day>_<event_index>.png``).
    """
    out_dir = day_dir(results_root, day)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    events_out = []
    n_filaments_with_events = 0
    event_index = 0

    for res in filament_results:
        if not res or not res.get("events"):
            continue
        n_filaments_with_events += 1
        roi_bbox = res["roi_bbox"]
        roi_image0 = res["roi_image0"]

        for ev in res["events"]:
            event_bbox = dict(
                min_y=ev["bbox_full_min_y"], min_x=ev["bbox_full_min_x"],
                max_y=ev["bbox_full_max_y"], max_x=ev["bbox_full_max_x"],
            )
            events_out.append(dict(
                event_index=int(event_index),
                filament_index=int(ev["roi_pick_index"]),
                filament_bbox=roi_bbox,
                event_bbox=event_bbox,
                period_min=ev["event_center_min"],
                period_mean_min=ev["event_mean_min"],
                scales=ev["scales_present"],
                strength=ev["total_strength"],
                centroid_x=ev["centroid_full_x"],
                centroid_y=ev["centroid_full_y"],
            ))

            plot_event(
                os.path.join(plots_dir, f"{day}_{event_index}.png"),
                roi_image0,
                roi_bbox,
                (event_bbox["min_y"], event_bbox["min_x"], event_bbox["max_y"], event_bbox["max_x"]),
                ev["event_center_min"],
                day=str(day),
                event_index=event_index,
            )
            event_index += 1

    write_json(
        os.path.join(out_dir, f"{day}_events.json"),
        dict(
            day=str(day),
            n_events=int(len(events_out)),
            n_filaments_with_events=int(n_filaments_with_events),
            events=events_out,
        ),
    )
    print(f"[day {day}] {len(events_out)} event(s) across {n_filaments_with_events} filament(s) -> {out_dir}")
    return events_out


def run_day(
    day: str,
    *,
    data_root: str = "data",
    results_root: str = "results",
    cnn_weights_path: str = "CNN/BestFit/BestFitWeights.h5",
    n_filament_workers: int = N_FILAMENT_WORKERS,
    n_pixel_workers: int = N_PIXEL_WORKERS,
    plot_period_families: bool = False,
):
    """Build the CP cache for one day and analyze every filament in masks[0]."""
    data_h5 = os.path.join(day_dir(data_root, day), f"{day}.h5")
    masks_h5 = os.path.join(day_dir(data_root, day), f"{day}_masks.h5")

    with h5.File(masks_h5, "r") as hf:
        mask0 = np.array(hf["masks"][0], dtype=np.uint8)

    n_filaments = len(list_candidate_regions_from_mask(mask0, min_area=750))
    print(f"\n=== Day {day}: {n_filaments} filaments ===")

    cp_cache_path = ensure_cp_cache_for_day(
        day=day,
        data_h5=data_h5,
        masks_h5=masks_h5,
        outroot=day_dir(results_root, day),
        cnn_weights_path=cnn_weights_path,
        cp_n_calib=1_000_000,
        cp_delta=CP_DELTA,
        cp_batch_size_cnn=4096,
        n_jobs=int(n_pixel_workers),
    )

    filament_results = Parallel(n_jobs=int(n_filament_workers), verbose=5)(
        delayed(run_filament)(
            day=day,
            index=i,
            data_root=data_root,
            results_root=results_root,
            cnn_weights_path=cnn_weights_path,
            cp_cache_path=cp_cache_path,
            n_pixel_workers=int(n_pixel_workers),
            plot_period_families=plot_period_families,
        )
        for i in range(n_filaments)
    )

    write_day_outputs(day, results_root, filament_results)


def run_all_days(
    *,
    data_root: str = "data",
    results_root: str = "results",
    cnn_weights_path: str = "CNN/BestFit/BestFitWeights.h5",
    n_filament_workers: int = N_FILAMENT_WORKERS,
    n_pixel_workers: int = N_PIXEL_WORKERS,
    plot_period_families: bool = False,
):
    days = discover_days(data_root)
    for day in days:
        run_day(
            day,
            data_root=data_root,
            results_root=results_root,
            cnn_weights_path=cnn_weights_path,
            n_filament_workers=n_filament_workers,
            n_pixel_workers=n_pixel_workers,
            plot_period_families=plot_period_families,
        )
