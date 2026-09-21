#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------

@dataclass
class Segment:
    x0: float
    y0: float
    x1: float
    y1: float
    z: float

    @property
    def length(self):
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)


@dataclass
class LayerData:
    z: float
    extrusion_segments: list = field(default_factory=list)

    extrusion_amount_mm: float = 0.0
    extrusion_path_length_mm: float = 0.0
    travel_distance_mm: float = 0.0

    extrusion_moves: int = 0
    travel_moves: int = 0
    retraction_count: int = 0
    retraction_amount_mm: float = 0.0

    x_min: float = math.inf
    y_min: float = math.inf
    x_max: float = -math.inf
    y_max: float = -math.inf

    weighted_cx_sum: float = 0.0
    weighted_cy_sum: float = 0.0
    weight_sum: float = 0.0

    def add_extrusion_segment(self, segment: Segment):
        length = segment.length

        if length <= 0:
            return

        self.extrusion_segments.append(segment)
        self.extrusion_path_length_mm += length

        self.x_min = min(self.x_min, segment.x0, segment.x1)
        self.y_min = min(self.y_min, segment.y0, segment.y1)
        self.x_max = max(self.x_max, segment.x0, segment.x1)
        self.y_max = max(self.y_max, segment.y0, segment.y1)

        mid_x = 0.5 * (segment.x0 + segment.x1)
        mid_y = 0.5 * (segment.y0 + segment.y1)

        self.weighted_cx_sum += mid_x * length
        self.weighted_cy_sum += mid_y * length
        self.weight_sum += length

    def centroid(self):
        if self.weight_sum <= 0:
            return None, None

        return (
            self.weighted_cx_sum / self.weight_sum,
            self.weighted_cy_sum / self.weight_sum,
        )

    def bbox_area(self):
        if not math.isfinite(self.x_min):
            return 0.0

        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)


# ------------------------------------------------------------
# Output directory structure
# ------------------------------------------------------------

def make_output_dirs(gcode_a, gcode_b, output_root="./evaluate_gcode_output", case_name=None):
    a_stem = Path(gcode_a).stem
    b_stem = Path(gcode_b).stem

    if case_name is None:
        case_name = f"{a_stem}__vs__{b_stem}"

    base_dir = Path(output_root) / case_name
    reports_dir = base_dir / "reports"
    plots_dir = base_dir / "plots"
    layer_images_dir = base_dir / "layer_images"

    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    layer_images_dir.mkdir(parents=True, exist_ok=True)

    return {
        "base_dir": base_dir,
        "reports_dir": reports_dir,
        "plots_dir": plots_dir,
        "layer_images_dir": layer_images_dir,
    }


def fmt_float_for_filename(value):
    if value is None:
        return "None"

    return f"{value:.3f}".replace(".", "p").replace("-", "m")


# ------------------------------------------------------------
# G-code parser
# ------------------------------------------------------------

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


def get_layer(layers, z_value, z_round_digits):
    z_key = round(float(z_value), z_round_digits)

    if z_key not in layers:
        layers[z_key] = LayerData(z=z_key)

    return layers[z_key]


def parse_gcode(path, z_round_digits=3, extrusion_epsilon=1e-7):
    layers = {}

    x = 0.0
    y = 0.0
    z = 0.0
    e = 0.0

    absolute_xyz = True
    absolute_e = True

    total_lines = 0
    motion_lines = 0
    unsupported_arc_lines = 0

    with open(path, "r", errors="ignore") as f:
        for raw_line in f:
            total_lines += 1

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

            if cmd in {"G2", "G3"}:
                unsupported_arc_lines += 1
                continue

            if cmd not in {"G0", "G1"}:
                continue

            old_x, old_y, old_z, old_e = x, y, z, e

            new_x = x
            new_y = y
            new_z = z
            new_e = e

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

            xy_length = math.hypot(new_x - old_x, new_y - old_y)

            layer = get_layer(layers, new_z, z_round_digits)

            if e_delta > extrusion_epsilon:
                layer.extrusion_amount_mm += e_delta
                layer.extrusion_moves += 1

                if xy_length > 0:
                    segment = Segment(
                        x0=old_x,
                        y0=old_y,
                        x1=new_x,
                        y1=new_y,
                        z=new_z,
                    )
                    layer.add_extrusion_segment(segment)

            elif e_delta < -extrusion_epsilon:
                layer.retraction_count += 1
                layer.retraction_amount_mm += abs(e_delta)

            else:
                if xy_length > 0:
                    layer.travel_moves += 1
                    layer.travel_distance_mm += xy_length

            x, y, z, e = new_x, new_y, new_z, new_e
            motion_lines += 1

    metadata = {
        "path": str(path),
        "total_lines": total_lines,
        "motion_lines": motion_lines,
        "num_layers_detected": len(layers),
        "unsupported_arc_lines_G2_G3": unsupported_arc_lines,
    }

    return layers, metadata


# ------------------------------------------------------------
# Layer pairing
# ------------------------------------------------------------

def pair_layers_by_z(layers_a, layers_b, z_tolerance):
    z_a_list = sorted(layers_a.keys())
    z_b_list = sorted(layers_b.keys())

    unused_b = set(z_b_list)
    pairs = []

    for z_a in z_a_list:
        if not unused_b:
            pairs.append((z_a, None))
            continue

        closest_z_b = min(unused_b, key=lambda z_b: abs(z_b - z_a))

        if abs(closest_z_b - z_a) <= z_tolerance:
            pairs.append((z_a, closest_z_b))
            unused_b.remove(closest_z_b)
        else:
            pairs.append((z_a, None))

    for z_b in sorted(unused_b):
        pairs.append((None, z_b))

    return pairs


def pair_layers_by_index(layers_a, layers_b):
    z_a_list = sorted(layers_a.keys())
    z_b_list = sorted(layers_b.keys())

    max_len = max(len(z_a_list), len(z_b_list))
    pairs = []

    for i in range(max_len):
        z_a = z_a_list[i] if i < len(z_a_list) else None
        z_b = z_b_list[i] if i < len(z_b_list) else None
        pairs.append((z_a, z_b))

    return pairs


# ------------------------------------------------------------
# Rasterization
# ------------------------------------------------------------

def collect_bounds(segments_a, segments_b, padding):
    xs = []
    ys = []

    for seg in segments_a + segments_b:
        xs.extend([seg.x0, seg.x1])
        ys.extend([seg.y0, seg.y1])

    if not xs:
        return None

    return (
        min(xs) - padding,
        min(ys) - padding,
        max(xs) + padding,
        max(ys) + padding,
    )


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

    for seg in segments:
        length = seg.length

        if length <= 0:
            continue

        n_samples = max(2, int(math.ceil(length / (resolution * 0.5))) + 1)

        xs = np.linspace(seg.x0, seg.x1, n_samples)
        ys = np.linspace(seg.y0, seg.y1, n_samples)

        cols = np.rint((xs - x_min) / resolution).astype(int)
        rows = np.rint((ys - y_min) / resolution).astype(int)

        for r, c in zip(rows, cols):
            for dr, dc in offsets:
                rr = r + dr
                cc = c + dc

                if 0 <= rr < height_px and 0 <= cc < width_px:
                    mask[rr, cc] = True

    return mask


def rasterized_iou_with_masks(
    segments_a,
    segments_b,
    resolution,
    line_width,
    max_pixels=50_000_000,
):
    padding = line_width + 2 * resolution
    bounds = collect_bounds(segments_a, segments_b, padding)

    if bounds is None:
        mask_a = np.zeros((20, 20), dtype=bool)
        mask_b = np.zeros((20, 20), dtype=bool)

        report = {
            "toolpath_iou": 1.0,
            "toolpath_iou_percent": 100.0,
            "a_covered_by_b_percent": 100.0,
            "b_covered_by_a_percent": 100.0,
            "intersection_pixels": 0,
            "union_pixels": 0,
            "a_pixels": 0,
            "b_pixels": 0,
            "mask_shape": [20, 20],
        }

        return report, mask_a, mask_b

    x_min, y_min, x_max, y_max = bounds

    width_px = int(math.ceil((x_max - x_min) / resolution)) + 1
    height_px = int(math.ceil((y_max - y_min) / resolution)) + 1

    total_pixels = width_px * height_px

    if total_pixels > max_pixels:
        raise MemoryError(
            f"Layer mask would be too large: {height_px} x {width_px} = {total_pixels} pixels. "
            f"Increase --resolution, for example from {resolution} to {resolution * 2}."
        )

    mask_a = draw_segments_to_mask(segments_a, bounds, resolution, line_width)
    mask_b = draw_segments_to_mask(segments_b, bounds, resolution, line_width)

    a_pixels = int(np.sum(mask_a))
    b_pixels = int(np.sum(mask_b))

    intersection_pixels = int(np.sum(mask_a & mask_b))
    union_pixels = int(np.sum(mask_a | mask_b))

    if union_pixels == 0:
        iou = 1.0
    else:
        iou = intersection_pixels / union_pixels

    a_covered_by_b = intersection_pixels / a_pixels if a_pixels > 0 else 1.0
    b_covered_by_a = intersection_pixels / b_pixels if b_pixels > 0 else 1.0

    report = {
        "toolpath_iou": float(iou),
        "toolpath_iou_percent": float(iou * 100.0),
        "a_covered_by_b_percent": float(a_covered_by_b * 100.0),
        "b_covered_by_a_percent": float(b_covered_by_a * 100.0),
        "intersection_pixels": intersection_pixels,
        "union_pixels": union_pixels,
        "a_pixels": a_pixels,
        "b_pixels": b_pixels,
        "mask_shape": [int(height_px), int(width_px)],
    }

    return report, mask_a, mask_b


def all_extrusion_segments(layers):
    segments = []

    for layer in layers.values():
        segments.extend(layer.extrusion_segments)

    return segments


def typical_layer_height(layers):
    z_values = sorted(layer.z for layer in extrusion_layers(layers))
    diffs = [
        b - a for a, b in zip(z_values, z_values[1:])
        if b - a > 1e-9
    ]

    if not diffs:
        return None

    return float(np.median(diffs))


def choose_height_bin_size(layers_a, layers_b, requested_bin_size):
    if requested_bin_size is not None:
        return round(requested_bin_size, 6)

    candidates = [
        height for height in (
            typical_layer_height(layers_a),
            typical_layer_height(layers_b),
        )
        if height is not None and height > 0
    ]

    if not candidates:
        return 0.2

    return round(max(candidates), 6)


def group_segments_by_height_bin(layers, z_min, height_bin_size):
    bins = {}

    for layer in extrusion_layers(layers):
        bin_index = int(math.floor((layer.z - z_min) / height_bin_size + 1e-9))
        bins.setdefault(bin_index, []).extend(layer.extrusion_segments)

    return bins


def compare_whole_print_by_height_bins(
    layers_a,
    layers_b,
    resolution,
    line_width,
    height_bin_size,
):
    printable_a = extrusion_layers(layers_a)
    printable_b = extrusion_layers(layers_b)

    if not printable_a and not printable_b:
        return [], {
            "height_bin_size_mm": height_bin_size,
            "num_height_bins": 0,
            "mean_whole_print_iou_percent": 100.0,
            "global_whole_print_iou_percent": 100.0,
            "global_intersection_pixels": 0,
            "global_union_pixels": 0,
            "global_a_pixels": 0,
            "global_b_pixels": 0,
        }

    z_values = [layer.z for layer in printable_a + printable_b]
    z_min = min(z_values)

    bins_a = group_segments_by_height_bin(layers_a, z_min, height_bin_size)
    bins_b = group_segments_by_height_bin(layers_b, z_min, height_bin_size)
    bin_indices = sorted(set(bins_a) | set(bins_b))

    rows = []
    global_intersection_pixels = 0
    global_union_pixels = 0
    global_a_pixels = 0
    global_b_pixels = 0

    for bin_index in bin_indices:
        segments_a = bins_a.get(bin_index, [])
        segments_b = bins_b.get(bin_index, [])

        iou_report, _, _ = rasterized_iou_with_masks(
            segments_a,
            segments_b,
            resolution=resolution,
            line_width=line_width,
        )

        global_intersection_pixels += iou_report["intersection_pixels"]
        global_union_pixels += iou_report["union_pixels"]
        global_a_pixels += iou_report["a_pixels"]
        global_b_pixels += iou_report["b_pixels"]

        z_bin_min = z_min + bin_index * height_bin_size
        z_bin_max = z_bin_min + height_bin_size

        rows.append({
            "height_bin_index": bin_index,
            "z_min_mm": round(float(z_bin_min), 6),
            "z_max_mm": round(float(z_bin_max), 6),
            "a_segment_count": len(segments_a),
            "b_segment_count": len(segments_b),
            "toolpath_iou_percent": iou_report["toolpath_iou_percent"],
            "a_covered_by_b_percent": iou_report["a_covered_by_b_percent"],
            "b_covered_by_a_percent": iou_report["b_covered_by_a_percent"],
            "intersection_pixels": iou_report["intersection_pixels"],
            "union_pixels": iou_report["union_pixels"],
            "a_pixels": iou_report["a_pixels"],
            "b_pixels": iou_report["b_pixels"],
        })

    valid_rows = [row for row in rows if row["union_pixels"] > 0]
    ious = np.array([row["toolpath_iou_percent"] for row in valid_rows], dtype=float)

    if global_union_pixels == 0:
        global_iou_percent = 100.0
    else:
        global_iou_percent = (global_intersection_pixels / global_union_pixels) * 100.0

    summary = {
        "height_bin_size_mm": float(height_bin_size),
        "num_height_bins": len(rows),
        "num_height_bins_with_extrusion_toolpaths": len(valid_rows),
        "mean_whole_print_iou_percent": float(np.mean(ious)) if len(ious) else 100.0,
        "median_whole_print_iou_percent": float(np.median(ious)) if len(ious) else 100.0,
        "min_whole_print_iou_percent": float(np.min(ious)) if len(ious) else 100.0,
        "max_whole_print_iou_percent": float(np.max(ious)) if len(ious) else 100.0,
        "global_whole_print_iou_percent": float(global_iou_percent),
        "global_intersection_pixels": int(global_intersection_pixels),
        "global_union_pixels": int(global_union_pixels),
        "global_a_pixels": int(global_a_pixels),
        "global_b_pixels": int(global_b_pixels),
    }

    return rows, summary


# ------------------------------------------------------------
# Save layer image
# ------------------------------------------------------------

def save_layer_pair_image(mask_a, mask_b, row, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(mask_a, origin="lower", cmap="gray")
    axes[0].set_title(f"Original\nz={row['z_a_mm']} mm")
    axes[0].axis("off")

    axes[1].imshow(mask_b, origin="lower", cmap="gray")
    axes[1].set_title(f"Modified\nz={row['z_b_mm']} mm")
    axes[1].axis("off")

    fig.suptitle(
        f"Layer {row['pair_index']} | "
        f"Toolpath IoU = {row['toolpath_iou_percent']:.2f}% | "
        f"Extrusion change = {row['extrusion_amount_change_percent']}"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------
# Material/stat comparison
# ------------------------------------------------------------

def safe_percent_change(original, modified):
    if original == 0 and modified == 0:
        return 0.0

    if original == 0:
        return None

    return ((modified - original) / original) * 100.0


def layer_to_stats(layer):
    if layer is None:
        return {
            "extrusion_amount_mm": 0.0,
            "extrusion_path_length_mm": 0.0,
            "travel_distance_mm": 0.0,
            "extrusion_moves": 0,
            "travel_moves": 0,
            "retraction_count": 0,
            "retraction_amount_mm": 0.0,
            "bbox_area_mm2": 0.0,
            "centroid_x": None,
            "centroid_y": None,
        }

    cx, cy = layer.centroid()

    return {
        "extrusion_amount_mm": float(layer.extrusion_amount_mm),
        "extrusion_path_length_mm": float(layer.extrusion_path_length_mm),
        "travel_distance_mm": float(layer.travel_distance_mm),
        "extrusion_moves": int(layer.extrusion_moves),
        "travel_moves": int(layer.travel_moves),
        "retraction_count": int(layer.retraction_count),
        "retraction_amount_mm": float(layer.retraction_amount_mm),
        "bbox_area_mm2": float(layer.bbox_area()),
        "centroid_x": cx,
        "centroid_y": cy,
    }


def centroid_shift(stats_a, stats_b):
    if stats_a["centroid_x"] is None or stats_b["centroid_x"] is None:
        return None

    dx = stats_b["centroid_x"] - stats_a["centroid_x"]
    dy = stats_b["centroid_y"] - stats_a["centroid_y"]

    return math.hypot(dx, dy)


def total_stats(layers):
    totals = {
        "extrusion_amount_mm": 0.0,
        "extrusion_path_length_mm": 0.0,
        "travel_distance_mm": 0.0,
        "extrusion_moves": 0,
        "travel_moves": 0,
        "retraction_count": 0,
        "retraction_amount_mm": 0.0,
    }

    for layer in layers.values():
        totals["extrusion_amount_mm"] += layer.extrusion_amount_mm
        totals["extrusion_path_length_mm"] += layer.extrusion_path_length_mm
        totals["travel_distance_mm"] += layer.travel_distance_mm
        totals["extrusion_moves"] += layer.extrusion_moves
        totals["travel_moves"] += layer.travel_moves
        totals["retraction_count"] += layer.retraction_count
        totals["retraction_amount_mm"] += layer.retraction_amount_mm

    return totals


def extrusion_layers(layers):
    return [
        layer for layer in layers.values()
        if layer.extrusion_amount_mm > 0 or layer.extrusion_path_length_mm > 0
    ]


def build_geometry_summary(layers):
    printable_layers = extrusion_layers(layers)

    if not printable_layers:
        return {
            "layer_count": 0,
            "final_build_height_mm": 0.0,
        }

    return {
        "layer_count": len(printable_layers),
        "final_build_height_mm": float(max(layer.z for layer in printable_layers)),
    }


def extrusion_volume_mm3(extrusion_length_mm, filament_diameter_mm):
    filament_radius_mm = filament_diameter_mm / 2.0
    filament_area_mm2 = math.pi * filament_radius_mm * filament_radius_mm

    return extrusion_length_mm * filament_area_mm2


def aml_metric_summary(
    global_iou_report,
    whole_print_iou_summary,
    material_summary,
    geometry_a,
    geometry_b,
    filament_diameter_mm,
):
    a_volume = extrusion_volume_mm3(
        material_summary["a_total_extrusion_amount_mm"],
        filament_diameter_mm,
    )
    b_volume = extrusion_volume_mm3(
        material_summary["b_total_extrusion_amount_mm"],
        filament_diameter_mm,
    )

    return {
        "toolpath_similarity_percent": whole_print_iou_summary[
            "global_whole_print_iou_percent"
        ],
        "mean_whole_print_iou_percent": whole_print_iou_summary[
            "mean_whole_print_iou_percent"
        ],
        "paired_layer_global_iou_percent": global_iou_report[
            "global_rasterized_toolpath_iou_percent"
        ],
        "a_extrusion_volume_mm3": float(a_volume),
        "b_extrusion_volume_mm3": float(b_volume),
        "extrusion_volume_difference_mm3": float(b_volume - a_volume),
        "extrusion_volume_difference_percent": safe_percent_change(a_volume, b_volume),
        "filament_diameter_mm": float(filament_diameter_mm),
        "a_layer_count": geometry_a["layer_count"],
        "b_layer_count": geometry_b["layer_count"],
        "layer_count_difference": geometry_b["layer_count"] - geometry_a["layer_count"],
        "a_final_build_height_mm": geometry_a["final_build_height_mm"],
        "b_final_build_height_mm": geometry_b["final_build_height_mm"],
        "final_build_height_difference_mm": (
            geometry_b["final_build_height_mm"] - geometry_a["final_build_height_mm"]
        ),
    }


# ------------------------------------------------------------
# Compare layers
# ------------------------------------------------------------

def compare_layers(
    layers_a,
    layers_b,
    pairs,
    resolution,
    line_width,
    layer_images_dir,
    save_layer_images=True,
):
    rows = []

    global_intersection_pixels = 0
    global_union_pixels = 0
    global_a_pixels = 0
    global_b_pixels = 0

    for idx, (z_a, z_b) in enumerate(pairs):
        layer_a = layers_a.get(z_a) if z_a is not None else None
        layer_b = layers_b.get(z_b) if z_b is not None else None

        segments_a = layer_a.extrusion_segments if layer_a is not None else []
        segments_b = layer_b.extrusion_segments if layer_b is not None else []

        iou_report, mask_a, mask_b = rasterized_iou_with_masks(
            segments_a,
            segments_b,
            resolution=resolution,
            line_width=line_width,
        )

        global_intersection_pixels += iou_report["intersection_pixels"]
        global_union_pixels += iou_report["union_pixels"]
        global_a_pixels += iou_report["a_pixels"]
        global_b_pixels += iou_report["b_pixels"]

        stats_a = layer_to_stats(layer_a)
        stats_b = layer_to_stats(layer_b)

        c_shift = centroid_shift(stats_a, stats_b)

        row = {
            "pair_index": idx,
            "z_a_mm": z_a,
            "z_b_mm": z_b,

            "toolpath_iou_percent": iou_report["toolpath_iou_percent"],
            "a_covered_by_b_percent": iou_report["a_covered_by_b_percent"],
            "b_covered_by_a_percent": iou_report["b_covered_by_a_percent"],

            "toolpath_intersection_pixels": iou_report["intersection_pixels"],
            "toolpath_union_pixels": iou_report["union_pixels"],
            "toolpath_a_pixels": iou_report["a_pixels"],
            "toolpath_b_pixels": iou_report["b_pixels"],

            "a_extrusion_amount_mm": stats_a["extrusion_amount_mm"],
            "b_extrusion_amount_mm": stats_b["extrusion_amount_mm"],
            "extrusion_amount_change_percent": safe_percent_change(
                stats_a["extrusion_amount_mm"],
                stats_b["extrusion_amount_mm"],
            ),

            "a_extrusion_path_length_mm": stats_a["extrusion_path_length_mm"],
            "b_extrusion_path_length_mm": stats_b["extrusion_path_length_mm"],
            "extrusion_path_length_change_percent": safe_percent_change(
                stats_a["extrusion_path_length_mm"],
                stats_b["extrusion_path_length_mm"],
            ),

            "a_travel_distance_mm": stats_a["travel_distance_mm"],
            "b_travel_distance_mm": stats_b["travel_distance_mm"],
            "travel_distance_change_percent": safe_percent_change(
                stats_a["travel_distance_mm"],
                stats_b["travel_distance_mm"],
            ),

            "a_extrusion_moves": stats_a["extrusion_moves"],
            "b_extrusion_moves": stats_b["extrusion_moves"],
            "extrusion_moves_difference": stats_b["extrusion_moves"] - stats_a["extrusion_moves"],

            "a_travel_moves": stats_a["travel_moves"],
            "b_travel_moves": stats_b["travel_moves"],
            "travel_moves_difference": stats_b["travel_moves"] - stats_a["travel_moves"],

            "a_retraction_count": stats_a["retraction_count"],
            "b_retraction_count": stats_b["retraction_count"],
            "retraction_count_difference": stats_b["retraction_count"] - stats_a["retraction_count"],

            "a_retraction_amount_mm": stats_a["retraction_amount_mm"],
            "b_retraction_amount_mm": stats_b["retraction_amount_mm"],
            "retraction_amount_change_percent": safe_percent_change(
                stats_a["retraction_amount_mm"],
                stats_b["retraction_amount_mm"],
            ),

            "a_bbox_area_mm2": stats_a["bbox_area_mm2"],
            "b_bbox_area_mm2": stats_b["bbox_area_mm2"],
            "bbox_area_change_percent": safe_percent_change(
                stats_a["bbox_area_mm2"],
                stats_b["bbox_area_mm2"],
            ),

            "centroid_shift_mm": c_shift,
        }

        if save_layer_images:
            z_a_str = fmt_float_for_filename(z_a)
            z_b_str = fmt_float_for_filename(z_b)
            iou_str = f"{row['toolpath_iou_percent']:.2f}".replace(".", "p")

            image_name = (
                f"layer_{idx:04d}_"
                f"zA_{z_a_str}_"
                f"zB_{z_b_str}_"
                f"iou_{iou_str}.png"
            )

            image_path = layer_images_dir / image_name
            save_layer_pair_image(mask_a, mask_b, row, image_path)

            row["layer_image_path"] = str(image_path)

        rows.append(row)

    global_toolpath_iou = (
        global_intersection_pixels / global_union_pixels
        if global_union_pixels > 0
        else 1.0
    )

    global_a_covered_by_b = (
        global_intersection_pixels / global_a_pixels
        if global_a_pixels > 0
        else 1.0
    )

    global_b_covered_by_a = (
        global_intersection_pixels / global_b_pixels
        if global_b_pixels > 0
        else 1.0
    )

    global_iou_report = {
        "global_rasterized_toolpath_iou_percent": float(global_toolpath_iou * 100.0),
        "global_a_covered_by_b_percent": float(global_a_covered_by_b * 100.0),
        "global_b_covered_by_a_percent": float(global_b_covered_by_a * 100.0),
        "global_intersection_pixels": int(global_intersection_pixels),
        "global_union_pixels": int(global_union_pixels),
        "global_a_pixels": int(global_a_pixels),
        "global_b_pixels": int(global_b_pixels),
    }

    return rows, global_iou_report


# ------------------------------------------------------------
# Save reports and plots
# ------------------------------------------------------------

def save_csv(rows, path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_iou_plot(rows, path):
    x = [r["pair_index"] for r in rows]
    y = [r["toolpath_iou_percent"] for r in rows]

    plt.figure(figsize=(12, 5))
    plt.plot(x, y, marker="o", linewidth=1)
    plt.xlabel("Layer pair index")
    plt.ylabel("Rasterized toolpath IoU (%)")
    plt.title("Layer-wise G-code Toolpath IoU")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_extrusion_change_plot(rows, path):
    x = [r["pair_index"] for r in rows]
    y = [
        np.nan if r["extrusion_amount_change_percent"] is None
        else r["extrusion_amount_change_percent"]
        for r in rows
    ]

    plt.figure(figsize=(12, 5))
    plt.plot(x, y, marker="o", linewidth=1)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Layer pair index")
    plt.ylabel("Extrusion amount change (%)")
    plt.title("Layer-wise Extrusion/Material Change")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def summarize_layer_results(rows):
    valid = [r for r in rows if r["toolpath_union_pixels"] > 0]

    if not valid:
        return {}

    ious = np.array([r["toolpath_iou_percent"] for r in valid], dtype=float)

    worst_row = min(valid, key=lambda r: r["toolpath_iou_percent"])
    best_row = max(valid, key=lambda r: r["toolpath_iou_percent"])

    extrusion_changes = [
        r["extrusion_amount_change_percent"]
        for r in valid
        if r["extrusion_amount_change_percent"] is not None
    ]

    path_changes = [
        r["extrusion_path_length_change_percent"]
        for r in valid
        if r["extrusion_path_length_change_percent"] is not None
    ]

    return {
        "num_compared_layer_pairs": len(rows),
        "num_layer_pairs_with_extrusion_toolpaths": len(valid),

        "mean_toolpath_iou_percent": float(np.mean(ious)),
        "median_toolpath_iou_percent": float(np.median(ious)),
        "min_toolpath_iou_percent": float(np.min(ious)),
        "max_toolpath_iou_percent": float(np.max(ious)),
        "std_toolpath_iou_percent": float(np.std(ious)),

        "worst_layer_pair_index": worst_row["pair_index"],
        "worst_layer_z_a_mm": worst_row["z_a_mm"],
        "worst_layer_z_b_mm": worst_row["z_b_mm"],
        "worst_layer_toolpath_iou_percent": worst_row["toolpath_iou_percent"],

        "best_layer_pair_index": best_row["pair_index"],
        "best_layer_z_a_mm": best_row["z_a_mm"],
        "best_layer_z_b_mm": best_row["z_b_mm"],
        "best_layer_toolpath_iou_percent": best_row["toolpath_iou_percent"],

        "mean_extrusion_amount_change_percent": (
            float(np.mean(extrusion_changes)) if extrusion_changes else None
        ),
        "mean_extrusion_path_length_change_percent": (
            float(np.mean(path_changes)) if path_changes else None
        ),
    }


def compare_total_material(totals_a, totals_b):
    return {
        "a_total_extrusion_amount_mm": totals_a["extrusion_amount_mm"],
        "b_total_extrusion_amount_mm": totals_b["extrusion_amount_mm"],
        "total_extrusion_amount_change_percent": safe_percent_change(
            totals_a["extrusion_amount_mm"],
            totals_b["extrusion_amount_mm"],
        ),

        "a_total_extrusion_path_length_mm": totals_a["extrusion_path_length_mm"],
        "b_total_extrusion_path_length_mm": totals_b["extrusion_path_length_mm"],
        "total_extrusion_path_length_change_percent": safe_percent_change(
            totals_a["extrusion_path_length_mm"],
            totals_b["extrusion_path_length_mm"],
        ),

        "a_total_travel_distance_mm": totals_a["travel_distance_mm"],
        "b_total_travel_distance_mm": totals_b["travel_distance_mm"],
        "total_travel_distance_change_percent": safe_percent_change(
            totals_a["travel_distance_mm"],
            totals_b["travel_distance_mm"],
        ),

        "a_total_extrusion_moves": totals_a["extrusion_moves"],
        "b_total_extrusion_moves": totals_b["extrusion_moves"],
        "total_extrusion_moves_difference": totals_b["extrusion_moves"] - totals_a["extrusion_moves"],

        "a_total_travel_moves": totals_a["travel_moves"],
        "b_total_travel_moves": totals_b["travel_moves"],
        "total_travel_moves_difference": totals_b["travel_moves"] - totals_a["travel_moves"],

        "a_total_retraction_count": totals_a["retraction_count"],
        "b_total_retraction_count": totals_b["retraction_count"],
        "total_retraction_count_difference": totals_b["retraction_count"] - totals_a["retraction_count"],

        "a_total_retraction_amount_mm": totals_a["retraction_amount_mm"],
        "b_total_retraction_amount_mm": totals_b["retraction_amount_mm"],
        "total_retraction_amount_change_percent": safe_percent_change(
            totals_a["retraction_amount_mm"],
            totals_b["retraction_amount_mm"],
        ),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare two G-code files using rasterized layer-wise toolpath IoU and extrusion/material statistics."
    )

    parser.add_argument("gcode_a", help="Original/reference G-code file.")
    parser.add_argument("gcode_b", help="Modified/attacked G-code file.")

    parser.add_argument(
        "--resolution",
        type=float,
        default=0.1,
        help="Rasterization resolution in mm per pixel. Smaller is more accurate but slower. Default: 0.1",
    )

    parser.add_argument(
        "--line-width",
        type=float,
        default=0.45,
        help="Rasterized toolpath width in mm. Use nozzle width or extrusion width. Default: 0.45",
    )

    parser.add_argument(
        "--filament-diameter",
        type=float,
        default=1.75,
        help="Filament diameter in mm for converting extrusion length to volume. Default: 1.75",
    )

    parser.add_argument(
        "--z-tolerance",
        type=float,
        default=0.05,
        help="Tolerance for pairing layers by Z height. Default: 0.05 mm",
    )

    parser.add_argument(
        "--height-bin-size",
        type=float,
        default=None,
        help=(
            "Height bin size in mm for whole-print IoU when layer heights differ. "
            "Default: auto-select the larger median layer height."
        ),
    )

    parser.add_argument(
        "--pairing",
        choices=["z", "index"],
        default="z",
        help="How to pair layers: by Z height or by layer index. Default: z",
    )

    parser.add_argument(
        "--z-round-digits",
        type=int,
        default=3,
        help="Rounding precision for Z heights. Default: 3",
    )

    parser.add_argument(
        "--output-root",
        default="./evaluate_gcode_output",
        help="Root output folder. Default: ./evaluate_gcode_output",
    )

    parser.add_argument(
        "--case-name",
        default=None,
        help="Optional custom output case folder name.",
    )

    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument(
        "--save-layer-images",
        action="store_true",
        help="Save per-layer side-by-side images. Default: disabled.",
    )
    image_group.add_argument(
        "--no-layer-images",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    dirs = make_output_dirs(
        args.gcode_a,
        args.gcode_b,
        output_root=args.output_root,
        case_name=args.case_name,
    )

    print("Output folder:")
    print(f"- {dirs['base_dir']}")

    print("\nParsing G-code A...")
    layers_a, metadata_a = parse_gcode(
        args.gcode_a,
        z_round_digits=args.z_round_digits,
    )

    print("Parsing G-code B...")
    layers_b, metadata_b = parse_gcode(
        args.gcode_b,
        z_round_digits=args.z_round_digits,
    )

    print(f"A layers detected: {len(layers_a)}")
    print(f"B layers detected: {len(layers_b)}")

    if metadata_a["unsupported_arc_lines_G2_G3"] > 0:
        print(
            f"WARNING: G-code A contains {metadata_a['unsupported_arc_lines_G2_G3']} G2/G3 arc lines. "
            "This script currently skips arc moves."
        )

    if metadata_b["unsupported_arc_lines_G2_G3"] > 0:
        print(
            f"WARNING: G-code B contains {metadata_b['unsupported_arc_lines_G2_G3']} G2/G3 arc lines. "
            "This script currently skips arc moves."
        )

    if args.pairing == "z":
        pairs = pair_layers_by_z(layers_a, layers_b, args.z_tolerance)
    else:
        pairs = pair_layers_by_index(layers_a, layers_b)

    print(f"Layer pairs to compare: {len(pairs)}")

    save_layer_images = args.save_layer_images and not args.no_layer_images

    print("Computing rasterized toolpath IoU and material statistics...")
    if save_layer_images:
        print("Saving per-layer side-by-side images...")

    rows, global_iou_report = compare_layers(
        layers_a,
        layers_b,
        pairs,
        resolution=args.resolution,
        line_width=args.line_width,
        layer_images_dir=dirs["layer_images_dir"],
        save_layer_images=save_layer_images,
    )

    height_bin_size = choose_height_bin_size(
        layers_a,
        layers_b,
        args.height_bin_size,
    )
    whole_print_iou_rows, whole_print_iou_summary = compare_whole_print_by_height_bins(
        layers_a,
        layers_b,
        resolution=args.resolution,
        line_width=args.line_width,
        height_bin_size=height_bin_size,
    )

    layer_summary = summarize_layer_results(rows)

    totals_a = total_stats(layers_a)
    totals_b = total_stats(layers_b)
    material_summary = compare_total_material(totals_a, totals_b)
    geometry_a = build_geometry_summary(layers_a)
    geometry_b = build_geometry_summary(layers_b)
    aml_metrics = aml_metric_summary(
        global_iou_report,
        whole_print_iou_summary,
        material_summary,
        geometry_a,
        geometry_b,
        args.filament_diameter,
    )

    csv_path = dirs["reports_dir"] / "layerwise.csv"
    whole_print_iou_csv_path = dirs["reports_dir"] / "whole_print_iou_by_height_bin.csv"
    json_path = dirs["reports_dir"] / "summary.json"
    iou_plot_path = dirs["plots_dir"] / "toolpath_iou.png"
    extrusion_plot_path = dirs["plots_dir"] / "extrusion_change.png"

    save_csv(rows, csv_path)
    save_csv(whole_print_iou_rows, whole_print_iou_csv_path)
    save_iou_plot(rows, iou_plot_path)
    save_extrusion_change_plot(rows, extrusion_plot_path)

    final_report = {
        "input_files": {
            "gcode_a": str(Path(args.gcode_a).resolve()),
            "gcode_b": str(Path(args.gcode_b).resolve()),
        },
        "output_folders": {
            "base_dir": str(dirs["base_dir"]),
            "reports_dir": str(dirs["reports_dir"]),
            "plots_dir": str(dirs["plots_dir"]),
            "layer_images_dir": str(dirs["layer_images_dir"]),
        },
        "settings": {
            "resolution_mm_per_pixel": args.resolution,
            "line_width_mm": args.line_width,
            "filament_diameter_mm": args.filament_diameter,
            "z_tolerance_mm": args.z_tolerance,
            "height_bin_size_mm": height_bin_size,
            "pairing": args.pairing,
            "z_round_digits": args.z_round_digits,
            "layer_images_saved": save_layer_images,
        },
        "metadata_a": metadata_a,
        "metadata_b": metadata_b,
        "global_toolpath_iou": global_iou_report,
        "whole_print_iou_summary": whole_print_iou_summary,
        "layerwise_summary": layer_summary,
        "global_material_summary": material_summary,
        "build_geometry_a": geometry_a,
        "build_geometry_b": geometry_b,
        "aml_metrics": aml_metrics,
    }

    with open(json_path, "w") as f:
        json.dump(final_report, f, indent=4)

    print("\n=== Summary ===")
    print(
        f"Toolpath similarity: "
        f"{aml_metrics['toolpath_similarity_percent']:.2f}%"
    )
    print(
        f"Mean whole-print IoU: "
        f"{aml_metrics['mean_whole_print_iou_percent']:.2f}%"
    )
    print(
        f"Extrusion-volume difference: "
        f"{aml_metrics['extrusion_volume_difference_mm3']:.2f} mm^3 "
        f"({aml_metrics['extrusion_volume_difference_percent']}%)"
    )
    print(
        f"Layer count: A={aml_metrics['a_layer_count']}, "
        f"B={aml_metrics['b_layer_count']}, "
        f"difference={aml_metrics['layer_count_difference']}"
    )
    print(
        f"Final build height: A={aml_metrics['a_final_build_height_mm']:.3f} mm, "
        f"B={aml_metrics['b_final_build_height_mm']:.3f} mm, "
        f"difference={aml_metrics['final_build_height_difference_mm']:.3f} mm"
    )
    print(
        f"Global rasterized toolpath IoU: "
        f"{global_iou_report['global_rasterized_toolpath_iou_percent']:.2f}%"
    )

    if layer_summary:
        print(
            f"Mean layer-wise toolpath IoU: "
            f"{layer_summary['mean_toolpath_iou_percent']:.2f}%"
        )
        print(
            f"Worst layer pair: {layer_summary['worst_layer_pair_index']} "
            f"at zA={layer_summary['worst_layer_z_a_mm']} mm, "
            f"zB={layer_summary['worst_layer_z_b_mm']} mm, "
            f"IoU={layer_summary['worst_layer_toolpath_iou_percent']:.2f}%"
        )

    print(
        f"Total extrusion amount change: "
        f"{material_summary['total_extrusion_amount_change_percent']}%"
    )

    print(
        f"Total extrusion path length change: "
        f"{material_summary['total_extrusion_path_length_change_percent']}%"
    )

    print("\nSaved outputs:")
    print(f"- Base folder: {dirs['base_dir']}")
    print(f"- Layer-wise CSV: {csv_path}")
    print(f"- Whole-print IoU CSV: {whole_print_iou_csv_path}")
    print(f"- JSON summary: {json_path}")
    print(f"- Toolpath IoU plot: {iou_plot_path}")
    print(f"- Extrusion change plot: {extrusion_plot_path}")

    # if save_layer_images:
    #     print(f"- Layer images: {dirs['layer_images_dir']}")


if __name__ == "__main__":
    main()