"""CSV and JSON writers for analysis outputs."""

import csv
import json
import math

def _strip_masks_for_serialization(detections):
    out = []
    for d in detections:
        dd = dict(d)
        dd.pop("roi_mask", None)
        out.append(dd)
    return out

def write_component_csv(csv_path, detections):
    fields = [
        "detection_id", "day", "roi_pick_index",
        "scale_idx", "scale_name", "N", "S",
        "period_group_id", "period_group_center_min", "component_idx",
        "period_min", "period_mean_min",
        "strength", "area_degpx",
        "family_id", "family_center_min",
        "event_id", "event_center_min",
        "centroid_deg_i", "centroid_deg_j",
        "centroid_roi_x", "centroid_roi_y",
        "centroid_full_x", "centroid_full_y",
        "bbox_deg_min_i", "bbox_deg_min_j", "bbox_deg_max_i", "bbox_deg_max_j",
        "bbox_roi_min_y", "bbox_roi_min_x", "bbox_roi_max_y", "bbox_roi_max_x",
        "bbox_full_min_y", "bbox_full_min_x", "bbox_full_max_y", "bbox_full_max_x",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for d in detections:
            row = {k: d.get(k, None) for k in fields}
            writer.writerow(row)

def write_json(json_path, payload):
    with open(json_path, "w") as f:
        json.dump(
            payload, f, indent=2,
            default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x
        )
