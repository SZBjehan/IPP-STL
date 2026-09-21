#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


SHELL_TYPES = {
    "external perimeter",
    "external perimeters",
    "overhang perimeter",
    "overhang perimeters",
}

INFILL_MARKERS = (
    "infill",
)


@dataclass
class Segment:
    x0: float
    y0: float
    x1: float
    y1: float
    z: float
    feature_type: str = "Unknown"
    category: str = "other"

    @property
    def length(self):
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)


@dataclass
class LayerData:
    z: float
    segments: list = field(default_factory=list)
    extrusion_amount_mm: float = 0.0
    extrusion_path_length_mm: float = 0.0
    extrusion_moves: int = 0

    def add_segments(self, segments):
        for segment in segments:
            if segment.length <= 0:
                continue
            self.segments.append(segment)
            self.extrusion_path_length_mm += segment.length


def parse_gcode_words(line):
    code = line.split(";", 1)[0].split("*", 1)[0].strip()
    if not code:
        return None, {}

    tokens = re.findall(
        r"([A-Za-z])\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        code,
    )
    if not tokens:
        return None, {}

    cmd = None
    params = {}
    for letter, value in tokens:
        letter = letter.upper()
        value_float = float(value)
        if letter in {"G", "M"} and cmd is None:
            cmd = f"{letter}{int(value_float)}"
        elif letter != "N":
            params[letter] = value_float

    return cmd, params


def normalize_feature_type(feature_type):
    return (feature_type or "Unknown").strip()


def classify_feature(feature_type):
    normalized = normalize_feature_type(feature_type).lower()
    if normalized in SHELL_TYPES:
        return "shell"
    if any(marker in normalized for marker in INFILL_MARKERS):
        return "infill"
    return "other"


def get_layer(layers, z_value, z_round_digits):
    z_key = round(float(z_value), z_round_digits)
    if z_key not in layers:
        layers[z_key] = LayerData(z=z_key)
    return layers[z_key]


def sweep_angle(start_angle, end_angle, clockwise):
    delta = end_angle - start_angle
    if clockwise:
        if delta >= 0:
            delta -= 2.0 * math.pi
    else:
        if delta <= 0:
            delta += 2.0 * math.pi
    return delta


def arc_center_from_radius(x0, y0, x1, y1, radius_word, clockwise):
    dx = x1 - x0
    dy = y1 - y0
    chord = math.hypot(dx, dy)
    if chord <= 0:
        return None

    radius = abs(radius_word)
    if radius < chord / 2.0:
        return None

    mx = 0.5 * (x0 + x1)
    my = 0.5 * (y0 + y1)
    h = math.sqrt(max(0.0, radius * radius - (chord * 0.5) ** 2))
    ux = -dy / chord
    uy = dx / chord

    candidates = [
        (mx + ux * h, my + uy * h),
        (mx - ux * h, my - uy * h),
    ]

    desired_large_arc = radius_word < 0
    best = None
    for cx, cy in candidates:
        start_angle = math.atan2(y0 - cy, x0 - cx)
        end_angle = math.atan2(y1 - cy, x1 - cx)
        delta = sweep_angle(start_angle, end_angle, clockwise)
        is_large_arc = abs(delta) > math.pi
        if is_large_arc == desired_large_arc:
            best = (cx, cy)
            break

    return best if best is not None else candidates[0]


def sample_arc_segments(
    x0,
    y0,
    z0,
    x1,
    y1,
    z1,
    params,
    clockwise,
    feature_type,
    arc_segment_length,
):
    if "I" in params or "J" in params:
        cx = x0 + params.get("I", 0.0)
        cy = y0 + params.get("J", 0.0)
    elif "R" in params:
        center = arc_center_from_radius(x0, y0, x1, y1, params["R"], clockwise)
        if center is None:
            return []
        cx, cy = center
    else:
        return []

    radius = math.hypot(x0 - cx, y0 - cy)
    if radius <= 0:
        return []

    start_angle = math.atan2(y0 - cy, x0 - cx)
    end_angle = math.atan2(y1 - cy, x1 - cx)
    delta = sweep_angle(start_angle, end_angle, clockwise)

    if abs(delta) < 1e-12 and math.hypot(x1 - x0, y1 - y0) <= 1e-9:
        delta = -2.0 * math.pi if clockwise else 2.0 * math.pi

    arc_length = abs(delta) * radius
    n_steps = max(2, int(math.ceil(arc_length / arc_segment_length)))
    category = classify_feature(feature_type)
    segments = []

    prev_x = x0
    prev_y = y0
    prev_z = z0
    for step in range(1, n_steps + 1):
        t = step / n_steps
        angle = start_angle + delta * t
        new_x = cx + radius * math.cos(angle)
        new_y = cy + radius * math.sin(angle)
        new_z = z0 + (z1 - z0) * t
        segments.append(Segment(prev_x, prev_y, new_x, new_y, new_z, feature_type, category))
        prev_x, prev_y, prev_z = new_x, new_y, new_z

    return segments


def parse_gcode(path, z_round_digits=3, extrusion_epsilon=1e-7, arc_segment_length=0.2):
    layers = {}
    feature_counts = {}
    category_counts = {"all": 0, "shell": 0, "infill": 0, "other": 0}

    x = y = z = e = 0.0
    absolute_xyz = True
    absolute_e = True
    current_feature_type = "Unknown"

    total_lines = 0
    motion_lines = 0
    extrusion_moves = 0
    arc_lines = 0
    unsupported_arc_lines = 0

    with open(path, "r", errors="ignore") as f:
        for raw_line in f:
            total_lines += 1
            stripped = raw_line.strip()

            if stripped.startswith(";TYPE:"):
                current_feature_type = stripped.split(":", 1)[1].strip()
                continue

            cmd, params = parse_gcode_words(raw_line)
            if cmd is None:
                continue

            if cmd == "G90":
                absolute_xyz = True
                continue
            if cmd == "G91":
                absolute_xyz = False
                continue
            if cmd == "M82":
                absolute_e = True
                continue
            if cmd == "M83":
                absolute_e = False
                continue
            if cmd == "G92":
                if "X" in params:
                    x = params["X"]
                if "Y" in params:
                    y = params["Y"]
                if "Z" in params:
                    z = params["Z"]
                if "E" in params:
                    e = params["E"]
                continue

            if cmd not in {"G0", "G1", "G2", "G3"}:
                continue

            old_x, old_y, old_z, old_e = x, y, z, e
            new_x, new_y, new_z, new_e = x, y, z, e

            if "X" in params:
                new_x = params["X"] if absolute_xyz else x + params["X"]
            if "Y" in params:
                new_y = params["Y"] if absolute_xyz else y + params["Y"]
            if "Z" in params:
                new_z = params["Z"] if absolute_xyz else z + params["Z"]

            if "E" in params:
                if absolute_e:
                    new_e = params["E"]
                    e_delta = new_e - e
                else:
                    e_delta = params["E"]
                    new_e = e + e_delta
            else:
                e_delta = 0.0

            if e_delta > extrusion_epsilon:
                category = classify_feature(current_feature_type)
                if cmd in {"G0", "G1"}:
                    segments = []
                    if math.hypot(new_x - old_x, new_y - old_y) > 0:
                        segments = [Segment(
                            old_x,
                            old_y,
                            new_x,
                            new_y,
                            new_z,
                            current_feature_type,
                            category,
                        )]
                else:
                    arc_lines += 1
                    segments = sample_arc_segments(
                        old_x,
                        old_y,
                        old_z,
                        new_x,
                        new_y,
                        new_z,
                        params,
                        clockwise=(cmd == "G2"),
                        feature_type=current_feature_type,
                        arc_segment_length=arc_segment_length,
                    )
                    if not segments:
                        unsupported_arc_lines += 1

                layer = get_layer(layers, new_z, z_round_digits)
                layer.extrusion_amount_mm += e_delta
                layer.extrusion_moves += 1
                layer.add_segments(segments)
                extrusion_moves += 1

                feature_counts[current_feature_type] = feature_counts.get(current_feature_type, 0) + 1
                category_counts["all"] += 1
                category_counts[category] += 1

            x, y, z, e = new_x, new_y, new_z, new_e
            motion_lines += 1

    metadata = {
        "path": str(path),
        "total_lines": total_lines,
        "motion_lines": motion_lines,
        "extrusion_moves": extrusion_moves,
        "num_z_values_detected": len(layers),
        "arc_lines_G2_G3": arc_lines,
        "unsupported_arc_lines_G2_G3": unsupported_arc_lines,
        "feature_counts_by_type": feature_counts,
        "extrusion_move_counts_by_category": category_counts,
    }

    return layers, metadata


def extrusion_layers(layers):
    return [
        layer for layer in layers.values()
        if layer.extrusion_amount_mm > 0 or layer.extrusion_path_length_mm > 0
    ]


def build_geometry_summary(layers):
    printable_layers = extrusion_layers(layers)
    if not printable_layers:
        return {"layer_count": 0, "final_build_height_mm": 0.0}
    return {
        "layer_count": len(printable_layers),
        "final_build_height_mm": float(max(layer.z for layer in printable_layers)),
    }


def total_extrusion_amount(layers):
    return float(sum(layer.extrusion_amount_mm for layer in layers.values()))


def extrusion_volume_mm3(extrusion_length_mm, filament_diameter_mm):
    return extrusion_length_mm * math.pi * (filament_diameter_mm / 2.0) ** 2


def typical_layer_height(layers):
    z_values = sorted(layer.z for layer in extrusion_layers(layers))
    diffs = [b - a for a, b in zip(z_values, z_values[1:]) if b - a > 1e-9]
    if not diffs:
        return None
    return float(np.median(diffs))


def choose_height_bin_size(layers_a, layers_b, requested_bin_size):
    if requested_bin_size is not None:
        return round(requested_bin_size, 6)
    candidates = [
        value for value in (typical_layer_height(layers_a), typical_layer_height(layers_b))
        if value is not None and value > 0
    ]
    return round(max(candidates), 6) if candidates else 0.2


def collect_segments(layers, category):
    segments = []
    for layer in layers.values():
        for segment in layer.segments:
            if category == "all" or segment.category == category:
                segments.append(segment)
    return segments


def collect_bounds(segments_a, segments_b, padding):
    xs = []
    ys = []
    for segment in segments_a + segments_b:
        xs.extend([segment.x0, segment.x1])
        ys.extend([segment.y0, segment.y1])
    if not xs:
        return None
    return (min(xs) - padding, min(ys) - padding, max(xs) + padding, max(ys) + padding)


def draw_segments_to_mask(segments, bounds, resolution, line_width):
    x_min, y_min, x_max, y_max = bounds
    width_px = int(math.ceil((x_max - x_min) / resolution)) + 1
    height_px = int(math.ceil((y_max - y_min) / resolution)) + 1
    mask = np.zeros((height_px, width_px), dtype=bool)

    radius_px = max(1, int(math.ceil((line_width / 2.0) / resolution)))
    offsets = []
    for dr in range(-radius_px, radius_px + 1):
        for dc in range(-radius_px, radius_px + 1):
            if dr * dr + dc * dc <= radius_px * radius_px:
                offsets.append((dr, dc))

    for segment in segments:
        length = segment.length
        if length <= 0:
            continue
        n_samples = max(2, int(math.ceil(length / (resolution * 0.5))) + 1)
        xs = np.linspace(segment.x0, segment.x1, n_samples)
        ys = np.linspace(segment.y0, segment.y1, n_samples)
        cols = np.rint((xs - x_min) / resolution).astype(int)
        rows = np.rint((ys - y_min) / resolution).astype(int)
        for r, c in zip(rows, cols):
            for dr, dc in offsets:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height_px and 0 <= cc < width_px:
                    mask[rr, cc] = True

    return mask


def rasterized_iou(segments_a, segments_b, resolution, line_width, max_pixels=50_000_000):
    if not segments_a and not segments_b:
        return {
            "applicable": False,
            "iou_percent": None,
            "intersection_pixels": 0,
            "union_pixels": 0,
            "a_pixels": 0,
            "b_pixels": 0,
            "mask_shape": None,
        }

    bounds = collect_bounds(segments_a, segments_b, line_width + 2 * resolution)
    x_min, y_min, x_max, y_max = bounds
    width_px = int(math.ceil((x_max - x_min) / resolution)) + 1
    height_px = int(math.ceil((y_max - y_min) / resolution)) + 1
    total_pixels = width_px * height_px
    if total_pixels > max_pixels:
        raise MemoryError(
            f"Mask would be too large: {height_px} x {width_px} = {total_pixels} pixels. "
            f"Increase --resolution."
        )

    mask_a = draw_segments_to_mask(segments_a, bounds, resolution, line_width)
    mask_b = draw_segments_to_mask(segments_b, bounds, resolution, line_width)

    a_pixels = int(np.sum(mask_a))
    b_pixels = int(np.sum(mask_b))
    intersection_pixels = int(np.sum(mask_a & mask_b))
    union_pixels = int(np.sum(mask_a | mask_b))
    iou_percent = None if union_pixels == 0 else float(intersection_pixels / union_pixels * 100.0)

    return {
        "applicable": union_pixels > 0,
        "iou_percent": iou_percent,
        "intersection_pixels": intersection_pixels,
        "union_pixels": union_pixels,
        "a_pixels": a_pixels,
        "b_pixels": b_pixels,
        "mask_shape": [height_px, width_px],
    }


def group_segments_by_height_bin(segments, z_min, height_bin_size):
    bins = {}
    for segment in segments:
        bin_index = int(math.floor((segment.z - z_min) / height_bin_size + 1e-9))
        bins.setdefault(bin_index, []).append(segment)
    return bins


def compare_category_by_height_bins(
    layers_a,
    layers_b,
    category,
    resolution,
    line_width,
    height_bin_size,
    global_z_min,
):
    segments_a = collect_segments(layers_a, category)
    segments_b = collect_segments(layers_b, category)

    if not segments_a and not segments_b:
        return [], {
            "category": category,
            "applicable": False,
            "height_bin_size_mm": height_bin_size,
            "num_bins": 0,
            "global_iou_percent": None,
            "mean_iou_percent": None,
            "median_iou_percent": None,
            "global_intersection_pixels": 0,
            "global_union_pixels": 0,
            "global_a_pixels": 0,
            "global_b_pixels": 0,
        }

    bins_a = group_segments_by_height_bin(segments_a, global_z_min, height_bin_size)
    bins_b = group_segments_by_height_bin(segments_b, global_z_min, height_bin_size)
    bin_indices = sorted(set(bins_a) | set(bins_b))

    rows = []
    total_intersection = 0
    total_union = 0
    total_a = 0
    total_b = 0

    for bin_index in bin_indices:
        z_min = global_z_min + bin_index * height_bin_size
        z_max = z_min + height_bin_size
        bin_segments_a = bins_a.get(bin_index, [])
        bin_segments_b = bins_b.get(bin_index, [])
        report = rasterized_iou(bin_segments_a, bin_segments_b, resolution, line_width)

        total_intersection += report["intersection_pixels"]
        total_union += report["union_pixels"]
        total_a += report["a_pixels"]
        total_b += report["b_pixels"]

        rows.append({
            "category": category,
            "height_bin_index": bin_index,
            "z_min_mm": round(float(z_min), 6),
            "z_max_mm": round(float(z_max), 6),
            "gt_segment_count": len(bin_segments_a),
            "modified_segment_count": len(bin_segments_b),
            "gt_pixels": report["a_pixels"],
            "modified_pixels": report["b_pixels"],
            "intersection_pixels": report["intersection_pixels"],
            "union_pixels": report["union_pixels"],
            "iou_percent": report["iou_percent"],
        })

    bin_ious = [row["iou_percent"] for row in rows if row["iou_percent"] is not None]
    global_iou = None if total_union == 0 else float(total_intersection / total_union * 100.0)

    summary = {
        "category": category,
        "applicable": total_union > 0,
        "height_bin_size_mm": height_bin_size,
        "num_bins": len(rows),
        "global_iou_percent": global_iou,
        "mean_iou_percent": float(np.mean(bin_ious)) if bin_ious else None,
        "median_iou_percent": float(np.median(bin_ious)) if bin_ious else None,
        "global_intersection_pixels": int(total_intersection),
        "global_union_pixels": int(total_union),
        "global_a_pixels": int(total_a),
        "global_b_pixels": int(total_b),
        "cross_check": "global_iou_percent = global_intersection_pixels / global_union_pixels * 100",
    }

    return rows, summary


def safe_percent_change(reference, modified):
    if reference == 0 and modified == 0:
        return 0.0
    if reference == 0:
        return None
    return (modified - reference) / reference * 100.0


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_output_dirs(gcode_a, gcode_b, output_root, case_name=None):
    if case_name is None:
        case_name = f"{Path(gcode_a).stem}__vs__{Path(gcode_b).stem}"
    base_dir = Path(output_root) / case_name
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, reports_dir


def fmt(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def evaluate_pair(args):
    base_dir, reports_dir = make_output_dirs(
        args.gcode_a,
        args.gcode_b,
        args.output_root,
        args.case_name,
    )

    layers_a, metadata_a = parse_gcode(
        args.gcode_a,
        z_round_digits=args.z_round_digits,
        arc_segment_length=args.arc_segment_length,
    )
    layers_b, metadata_b = parse_gcode(
        args.gcode_b,
        z_round_digits=args.z_round_digits,
        arc_segment_length=args.arc_segment_length,
    )

    geometry_a = build_geometry_summary(layers_a)
    geometry_b = build_geometry_summary(layers_b)
    extrusion_a = total_extrusion_amount(layers_a)
    extrusion_b = total_extrusion_amount(layers_b)
    volume_a = extrusion_volume_mm3(extrusion_a, args.filament_diameter)
    volume_b = extrusion_volume_mm3(extrusion_b, args.filament_diameter)
    volume_delta = volume_b - volume_a
    volume_delta_percent = safe_percent_change(volume_a, volume_b)

    height_bin_size = choose_height_bin_size(layers_a, layers_b, args.height_bin_size)
    all_segments = collect_segments(layers_a, "all") + collect_segments(layers_b, "all")
    global_z_min = min((segment.z for segment in all_segments), default=0.0)

    category_summaries = {}
    category_rows = {}
    for category in ("all", "shell", "infill"):
        rows, summary = compare_category_by_height_bins(
            layers_a,
            layers_b,
            category,
            resolution=args.resolution,
            line_width=args.line_width,
            height_bin_size=height_bin_size,
            global_z_min=global_z_min,
        )
        category_rows[category] = rows
        category_summaries[category] = summary
        write_csv(reports_dir / f"iou_by_height_bin_{category}.csv", rows)

    summary_row = {
        "gt_file": str(Path(args.gcode_a)),
        "modified_file": str(Path(args.gcode_b)),
        "gt_layers": geometry_a["layer_count"],
        "modified_layers": geometry_b["layer_count"],
        "layer_delta_vs_gt": geometry_b["layer_count"] - geometry_a["layer_count"],
        "gt_material_volume_mm3": fmt(volume_a, 4),
        "modified_material_volume_mm3": fmt(volume_b, 4),
        "material_delta_mm3": fmt(volume_delta, 4),
        "material_delta_percent": fmt(volume_delta_percent, 4),
        "toolpath_iou_percent": fmt(category_summaries["all"]["global_iou_percent"], 4),
        "toolpath_mean_iou_percent": fmt(category_summaries["all"]["mean_iou_percent"], 4),
        "shell_iou_percent": fmt(category_summaries["shell"]["global_iou_percent"], 4),
        "shell_mean_iou_percent": fmt(category_summaries["shell"]["mean_iou_percent"], 4),
        "infill_iou_percent": fmt(category_summaries["infill"]["global_iou_percent"], 4),
        "infill_mean_iou_percent": fmt(category_summaries["infill"]["mean_iou_percent"], 4),
        "gt_build_height_mm": fmt(geometry_a["final_build_height_mm"], 4),
        "modified_build_height_mm": fmt(geometry_b["final_build_height_mm"], 4),
        "build_height_delta_mm": fmt(geometry_b["final_build_height_mm"] - geometry_a["final_build_height_mm"], 4),
        "height_bin_size_mm": height_bin_size,
        "resolution_mm_per_pixel": args.resolution,
        "line_width_mm": args.line_width,
    }

    write_csv(reports_dir / "evaluation_summary.csv", [summary_row])

    final_report = {
        "input_files": {
            "gt_gcode": str(Path(args.gcode_a).resolve()),
            "modified_gcode": str(Path(args.gcode_b).resolve()),
        },
        "output_folders": {
            "base_dir": str(base_dir),
            "reports_dir": str(reports_dir),
        },
        "settings": {
            "resolution_mm_per_pixel": args.resolution,
            "line_width_mm": args.line_width,
            "filament_diameter_mm": args.filament_diameter,
            "height_bin_size_mm": height_bin_size,
            "z_round_digits": args.z_round_digits,
            "arc_segment_length_mm": args.arc_segment_length,
            "shell_types": sorted(SHELL_TYPES),
            "infill_rule": "Any ;TYPE containing 'infill' is treated as infill.",
            "whole_toolpath_iou_rule": "Sum intersection pixels over height bins, divide by summed union pixels.",
        },
        "metadata_gt": metadata_a,
        "metadata_modified": metadata_b,
        "geometry_gt": geometry_a,
        "geometry_modified": geometry_b,
        "material": {
            "gt_extrusion_length_mm": extrusion_a,
            "modified_extrusion_length_mm": extrusion_b,
            "gt_volume_mm3": volume_a,
            "modified_volume_mm3": volume_b,
            "delta_volume_mm3": volume_delta,
            "delta_volume_percent": volume_delta_percent,
            "filament_diameter_mm": args.filament_diameter,
        },
        "iou": category_summaries,
        "summary_row": summary_row,
    }

    with open(reports_dir / "summary.json", "w") as f:
        json.dump(final_report, f, indent=4)

    return base_dir, reports_dir, summary_row


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GT vs modified G-code with whole, shell, and infill raster IoU."
    )
    parser.add_argument("gcode_a", help="GT/reference G-code file.")
    parser.add_argument("gcode_b", help="Modified G-code file.")
    parser.add_argument("--resolution", type=float, default=0.1, help="Raster resolution in mm/pixel.")
    parser.add_argument("--line-width", type=float, default=0.45, help="Rasterized extrusion width in mm.")
    parser.add_argument("--filament-diameter", type=float, default=1.75, help="Filament diameter in mm.")
    parser.add_argument("--height-bin-size", type=float, default=None, help="Height bin size in mm. Default: larger median layer height.")
    parser.add_argument("--arc-segment-length", type=float, default=0.2, help="Max linear segment length for G2/G3 arc approximation.")
    parser.add_argument("--z-round-digits", type=int, default=3, help="Rounding precision for Z heights.")
    parser.add_argument("--output-root", default="eval_2", help="Output root folder. Default: eval_2.")
    parser.add_argument("--case-name", default=None, help="Optional output case folder name.")
    args = parser.parse_args()

    base_dir, reports_dir, summary_row = evaluate_pair(args)

    print("Output folder:")
    print(f"- {base_dir}")
    print("\n=== Evaluation Summary ===")
    print(f"Layers: GT={summary_row['gt_layers']}, modified={summary_row['modified_layers']}, delta={summary_row['layer_delta_vs_gt']}")
    print(f"Material delta: {summary_row['material_delta_mm3']} mm^3 ({summary_row['material_delta_percent']}%)")
    print(f"Toolpath IoU: {summary_row['toolpath_iou_percent']}%")
    print(f"Shell IoU: {summary_row['shell_iou_percent']}%")
    print(f"Infill IoU: {summary_row['infill_iou_percent']}%")
    print(f"Build height: GT={summary_row['gt_build_height_mm']} mm, modified={summary_row['modified_build_height_mm']} mm, delta={summary_row['build_height_delta_mm']} mm")
    print("\nSaved reports:")
    print(f"- {reports_dir / 'summary.json'}")
    print(f"- {reports_dir / 'evaluation_summary.csv'}")
    print(f"- {reports_dir / 'iou_by_height_bin_all.csv'}")
    print(f"- {reports_dir / 'iou_by_height_bin_shell.csv'}")
    print(f"- {reports_dir / 'iou_by_height_bin_infill.csv'}")


if __name__ == "__main__":
    main()
