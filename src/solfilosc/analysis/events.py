"""Spatial event clustering within detected period families."""

import csv
import math
from collections import defaultdict

import numpy as np

from .roi import weighted_median_log

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(int(n)))
        self.rank = [0] * int(n)

    def find(self, x):
        x = int(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def groups(self):
        out = defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return list(out.values())

def _valid_box(box):
    if box is None:
        return False
    if len(box) != 4:
        return False
    if any(v is None for v in box):
        return False
    y0, x0, y1, x1 = box
    vals = np.array([y0, x0, y1, x1], dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        return False
    return (y1 > y0) and (x1 > x0)

def bbox_iou(box_a, box_b):
    """
    box format: (min_y, min_x, max_y, max_x)
    """
    if not (_valid_box(box_a) and _valid_box(box_b)):
        return 0.0

    ay0, ax0, ay1, ax1 = [float(v) for v in box_a]
    by0, bx0, by1, bx1 = [float(v) for v in box_b]

    iy0 = max(ay0, by0)
    ix0 = max(ax0, bx0)
    iy1 = min(ay1, by1)
    ix1 = min(ax1, bx1)

    ih = max(0.0, iy1 - iy0)
    iw = max(0.0, ix1 - ix0)
    inter = ih * iw
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ay1 - ay0) * max(0.0, ax1 - ax0)
    area_b = max(0.0, by1 - by0) * max(0.0, bx1 - bx0)
    union = area_a + area_b - inter

    if union <= 0.0:
        return 0.0
    return float(inter / union)

def detection_full_bbox(det):
    box = (
        det.get("bbox_full_min_y"),
        det.get("bbox_full_min_x"),
        det.get("bbox_full_max_y"),
        det.get("bbox_full_max_x"),
    )
    return box if _valid_box(box) else None

def detection_full_centroid(det):
    x = det.get("centroid_full_x", None)
    y = det.get("centroid_full_y", None)
    if x is None or y is None:
        return None
    vals = np.array([x, y], dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        return None
    return float(x), float(y)

def detections_spatially_match(
    d1,
    d2,
    *,
    bbox_iou_thr=0.10,
    centroid_scale_factor=1.5,
    max_scale_idx_gap=None,
    allow_same_scale=False,
):
    """
    Direct edge rule for event clustering.

    Two detections can be linked only if:
      - same family_id
      - optionally not from the same scale
      - optionally close enough in scale index
      - AND (bbox IoU high enough OR centroid distance small enough)
    """
    f1 = d1.get("family_id", None)
    f2 = d2.get("family_id", None)
    if f1 is None or f2 is None or int(f1) != int(f2):
        return False

    s1 = int(d1["scale_idx"])
    s2 = int(d2["scale_idx"])

    if (not allow_same_scale) and (s1 == s2):
        return False

    if max_scale_idx_gap is not None and abs(s1 - s2) > int(max_scale_idx_gap):
        return False

    # bbox overlap test
    box1 = detection_full_bbox(d1)
    box2 = detection_full_bbox(d2)
    iou = bbox_iou(box1, box2)
    iou_ok = iou >= float(bbox_iou_thr)

    # centroid-distance test
    c1 = detection_full_centroid(d1)
    c2 = detection_full_centroid(d2)
    dist_ok = False
    if c1 is not None and c2 is not None:
        dx = float(c1[0] - c2[0])
        dy = float(c1[1] - c2[1])
        dist = math.hypot(dx, dy)
        dist_thr = float(centroid_scale_factor) * max(float(d1["N"]), float(d2["N"]))
        dist_ok = dist <= dist_thr

    return bool(iou_ok or dist_ok)

def build_event_summary(event_id, family_id, members):
    strengths = np.array(
        [max(1e-12, float(d.get("strength", 0.0))) for d in members],
        dtype=np.float64
    )

    periods = np.array(
        [
            np.nan if d.get("period_min", None) is None else float(d["period_min"])
            for d in members
        ],
        dtype=np.float64
    )

    p_ok = np.isfinite(periods) & (periods > 0) & np.isfinite(strengths) & (strengths > 0)
    if np.any(p_ok):
        lp = weighted_median_log(np.log(periods[p_ok]), strengths[p_ok])
        if np.isfinite(lp):
            event_center_min = float(np.exp(lp))
        else:
            event_center_min = float(np.nanmedian(periods[p_ok]))
        event_mean_min = float(np.average(periods[p_ok], weights=strengths[p_ok]))
    else:
        event_center_min = None
        event_mean_min = None

    centroids = []
    centroid_weights = []
    for d in members:
        c = detection_full_centroid(d)
        if c is not None:
            centroids.append(c)
            centroid_weights.append(max(1e-12, float(d.get("strength", 0.0))))

    if centroids:
        centroids = np.asarray(centroids, dtype=np.float64)
        centroid_weights = np.asarray(centroid_weights, dtype=np.float64)
        centroid_full_x = float(np.average(centroids[:, 0], weights=centroid_weights))
        centroid_full_y = float(np.average(centroids[:, 1], weights=centroid_weights))
    else:
        centroid_full_x = None
        centroid_full_y = None

    boxes = [detection_full_bbox(d) for d in members]
    boxes = [b for b in boxes if b is not None]
    if boxes:
        bbox_full = (
            int(min(b[0] for b in boxes)),
            int(min(b[1] for b in boxes)),
            int(max(b[2] for b in boxes)),
            int(max(b[3] for b in boxes)),
        )
    else:
        bbox_full = (None, None, None, None)

    scale_idxs = sorted(set(int(d["scale_idx"]) for d in members))
    scales_present = sorted(set(int(d["N"]) for d in members), reverse=True)
    detection_ids = sorted(int(d["detection_id"]) for d in members)

    return dict(
        event_id=int(event_id),
        family_id=int(family_id),
        family_center_min=(
            None if members[0].get("family_center_min", None) is None
            else float(members[0]["family_center_min"])
        ),
        day=str(members[0]["day"]),
        roi_pick_index=int(members[0]["roi_pick_index"]),
        event_center_min=event_center_min,
        event_mean_min=event_mean_min,
        total_strength=float(np.sum(strengths)),
        n_components=int(len(members)),
        n_scales=int(len(scale_idxs)),
        scale_idxs=scale_idxs,
        scales_present=scales_present,
        detection_ids=detection_ids,
        centroid_full_x=centroid_full_x,
        centroid_full_y=centroid_full_y,
        bbox_full_min_y=bbox_full[0],
        bbox_full_min_x=bbox_full[1],
        bbox_full_max_y=bbox_full[2],
        bbox_full_max_x=bbox_full[3],
    )

def assign_event_ids_within_families(
    detections,
    *,
    bbox_iou_thr=0.10,
    centroid_scale_factor=1.5,
    max_scale_idx_gap=2,
    allow_same_scale=False,
    min_event_scales=4,
):
    """
    Adds:
      - detection['event_id']
      - detection['event_center_min']

    Returns:
      event_summaries
    """
    for d in detections:
        d["event_id"] = None
        d["event_center_min"] = None

    by_family = defaultdict(list)
    for d in detections:
        fid = d.get("family_id", None)
        if fid is None:
            continue
        by_family[int(fid)].append(d)

    event_summaries = []
    next_event_id = 0

    for family_id in sorted(by_family.keys()):
        fam_dets = by_family[family_id]
        n = len(fam_dets)

        uf = UnionFind(n)

        for i in range(n):
            for j in range(i + 1, n):
                if detections_spatially_match(
                    fam_dets[i],
                    fam_dets[j],
                    bbox_iou_thr=float(bbox_iou_thr),
                    centroid_scale_factor=float(centroid_scale_factor),
                    max_scale_idx_gap=max_scale_idx_gap,
                    allow_same_scale=bool(allow_same_scale),
                ):
                    uf.union(i, j)

        groups = uf.groups()

        def _group_sort_key(local_idxs):
            members = [fam_dets[k] for k in local_idxs]
            n_scales = len(set(int(d["scale_idx"]) for d in members))
            total_strength = sum(float(d["strength"]) for d in members)
            return (n_scales, total_strength, len(members))

        groups = sorted(groups, key=_group_sort_key, reverse=True)

        for local_idxs in groups:
            members = [fam_dets[k] for k in local_idxs]
            summary = build_event_summary(
                event_id=next_event_id,
                family_id=family_id,
                members=members,
            )

            # Keep only well-supported events
            if int(summary["n_scales"]) < int(min_event_scales):
                continue

            event_summaries.append(summary)

            for d in members:
                d["event_id"] = int(next_event_id)
                d["event_center_min"] = summary["event_center_min"]

            next_event_id += 1

    event_summaries = sorted(
        event_summaries,
        key=lambda d: (int(d["family_id"]), -int(d["n_scales"]), -float(d["total_strength"]))
    )
    return event_summaries

def summarize_reported_families(detections):
    by_family = defaultdict(list)

    for d in detections:
        fid = d.get("family_id", None)
        if fid is None:
            continue
        by_family[int(fid)].append(d)

    family_summaries = []
    for fid, fam_det in by_family.items():
        center = fam_det[0].get("family_center_min", None)

        family_summaries.append(dict(
            family_id=int(fid),
            family_center_min=(None if center is None else float(center)),
            n_components=int(len(fam_det)),
            n_scales=int(len(set(int(d["scale_idx"]) for d in fam_det))),
            total_strength=float(sum(float(d["strength"]) for d in fam_det)),
            scales_present=sorted(set(int(d["N"]) for d in fam_det), reverse=True),
        ))

    family_summaries.sort(
        key=lambda d: (d["n_scales"], d["total_strength"]),
        reverse=True
    )
    return family_summaries

def write_event_summary_csv(csv_path, event_summaries):
    fields = [
        "event_id", "family_id", "family_center_min",
        "day", "roi_pick_index",
        "event_center_min", "event_mean_min",
        "total_strength", "n_components", "n_scales",
        "centroid_full_x", "centroid_full_y",
        "bbox_full_min_y", "bbox_full_min_x", "bbox_full_max_y", "bbox_full_max_x",
        "scale_idxs", "scales_present", "detection_ids",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in event_summaries:
            row = dict(rec)
            row["scale_idxs"] = ",".join(str(x) for x in rec["scale_idxs"])
            row["scales_present"] = ",".join(str(x) for x in rec["scales_present"])
            row["detection_ids"] = ",".join(str(x) for x in rec["detection_ids"])
            writer.writerow(row)
