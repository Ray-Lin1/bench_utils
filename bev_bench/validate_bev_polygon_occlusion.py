"""Validate and time BEV marking occlusion, then write the review artifacts."""

import argparse
import csv
import json
import os
from pathlib import Path
import platform
from time import perf_counter_ns

import numpy as np
import shapely

from bev_occlusion import bev_occlusion_mask
from bev_polygon_occlusion import bev_polygon_occlusion_ratio, _marking_intersections
from bev_polygon_scenarios import marking_frames
from bev_scenarios import CAR
from validate_bev_occlusion import check_vehicle_spacing, corners


GRID_STEP_M = 0.02
GRID_TOLERANCE = 0.02


def sampled_ratios(markings, boxes):
    """Estimate area independently with uniform cells and the point solver."""
    samples = []
    for marking in markings:
        xmin, ymin, xmax, ymax = marking.bounds
        columns, rows = np.maximum(1, np.ceil(
            np.array([xmax - xmin, ymax - ymin]) / GRID_STEP_M
        ).astype(int))
        x = xmin + (np.arange(columns) + 0.5) * (xmax - xmin) / columns
        y = ymin + (np.arange(rows) + 0.5) * (ymax - ymin) / rows
        grid = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
        inside = shapely.contains_xy(marking, grid[:, 0], grid[:, 1])
        samples.append(grid[inside])
    lengths = np.array([len(points) for points in samples])
    if np.any(lengths == 0):
        raise AssertionError("Reference grid did not sample every marking")
    flags = bev_occlusion_mask(np.concatenate(samples), boxes)
    cuts = np.cumsum(lengths)[:-1]
    return np.array([part.mean() for part in np.split(flags, cuts)]), int(lengths.sum())


def measure(frame, warmup, repeats):
    for _ in range(warmup):
        bev_polygon_occlusion_ratio(frame.markings, frame.boxes)
    times = np.empty(repeats)
    for index in range(repeats):
        start = perf_counter_ns()
        bev_polygon_occlusion_ratio(frame.markings, frame.boxes)
        times[index] = perf_counter_ns() - start
    return times / 1_000_000


def paint(ax, geometry, color):
    from matplotlib.path import Path as PlotPath
    from matplotlib.patches import PathPatch
    from shapely.geometry.polygon import orient

    for part in shapely.get_parts(geometry):
        if part.is_empty or part.area == 0:
            continue
        if part.geom_type != "Polygon":
            paint(ax, part, color)
            continue
        polygon = orient(part, sign=1.0)
        paths = [PlotPath(np.asarray(ring.coords), closed=True)
                 for ring in (polygon.exterior, *polygon.interiors)]
        ax.add_patch(PathPatch(PlotPath.make_compound_path(*paths), facecolor=color,
                               edgecolor="none", zorder=3))


def render(output, frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Polygon
    from matplotlib.lines import Line2D

    visible, blocked = "#087f69", "#cc4b32"
    box_edge, ego_color = "#586b80", "#225ca9"
    figure, axes = plt.subplots(3, 3, figsize=(15, 9.3), layout="constrained")
    figure.suptitle("Area occlusion of BEV road markings", fontsize=18, fontweight="bold")
    for index, ((frame, markings, overlaps, ratios), ax) in enumerate(zip(frames, axes.flat), 1):
        ax.set_facecolor("#f4f7fa")
        ax.set_axisbelow(True)
        ax.grid(color="#dde5ec", linewidth=0.6)
        for marker, overlap in zip(markings, overlaps):
            paint(ax, marker.difference(overlap), visible)
            paint(ax, overlap, blocked)
        for box in frame.boxes:
            ax.add_patch(Polygon(corners(box), facecolor="none", edgecolor=box_edge,
                                 linewidth=1, linestyle="--", zorder=4))
        ax.add_patch(Polygon(corners([0, 0, *CAR, 0]), facecolor="none", edgecolor=ego_color))
        ax.scatter([0], [0], color=ego_color, marker="*", s=35, zorder=5)
        coords = [np.array([[-2.3, -0.925], [2.3, 0.925]])]
        coords.extend(corners(box) for box in frame.boxes)
        coords.extend(np.asarray(marker.bounds).reshape(2, 2) for marker in markings)
        xy = np.concatenate(coords)
        xmin, ymin = xy.min(axis=0)
        xmax, ymax = xy.max(axis=0)
        ax.set_xlim(xmin - 1, xmax + 1)
        ymargin = max(1, ((xmax - xmin) / 2.8 - (ymax - ymin)) / 2)
        ax.set_ylim(ymin - ymargin, ymax + ymargin)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{index:02d}  {frame.name}", loc="left", fontsize=10, fontweight="bold")
        detail = " / ".join(f"{ratio:.1%}" for ratio in ratios) if len(ratios) <= 2 else (
            f"{np.count_nonzero(ratios)}/{len(ratios)} markings have occluded area"
        )
        ax.set_xlabel(f"x (m)   •   {detail}", fontsize=9)
        ax.set_ylabel("y (m)", fontsize=9)
        ax.tick_params(labelsize=8)
    handles = [Patch(facecolor=visible, label="Visible paint"),
               Patch(facecolor=blocked, label="Occluded paint"),
               Patch(facecolor="none", edgecolor=box_edge, linestyle="--", label="Obstacle box"),
               Line2D([], [], color=ego_color, marker="*", linestyle="None", label="Ego origin")]
    figure.legend(handles=handles, loc="outside lower center", ncols=4, frameon=False)
    figure.savefig(output / "overview.png", dpi=150)
    plt.close(figure)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def cpu_model():
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.is_file():
        for line in cpu_info.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be nonnegative and --repeats must be positive")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be a new or empty directory")

    frame_rows, marking_rows, render_inputs = [], [], []
    for frame_id, frame in enumerate(marking_frames(), 1):
        check_vehicle_spacing(frame)
        ratios = bev_polygon_occlusion_ratio(frame.markings, frame.boxes)
        if frame.expected is not None:
            np.testing.assert_allclose(ratios, frame.expected, atol=1e-12, rtol=0,
                                       err_msg=frame.name)
        markings, overlaps = _marking_intersections(frame.markings, frame.boxes, (0, 0))
        sampled, sample_count = sampled_ratios(markings, frame.boxes)
        np.testing.assert_allclose(ratios, sampled, atol=GRID_TOLERANCE, rtol=0,
                                   err_msg=frame.name)
        times = measure(frame, args.warmup, args.repeats)
        frame_row = dict(frame=frame_id, scene=frame.name, markings=len(markings),
                         boxes=len(frame.boxes), reference_cells=sample_count,
                         max_grid_error_pp=round(100 * float(np.max(np.abs(ratios - sampled))), 6),
                         min_ms=round(float(times.min()), 6),
                         median_ms=round(float(np.median(times)), 6),
                         mean_ms=round(float(times.mean()), 6),
                         p95_ms=round(float(np.percentile(times, 95)), 6))
        frame_rows.append(frame_row)
        for mark_id, (mark, overlap, ratio, approx) in enumerate(
            zip(markings, overlaps, ratios, sampled), 1
        ):
            marking_rows.append(dict(frame=frame_id, marking=mark_id, area_m2=mark.area,
                                     occluded_area_m2=overlap.area, ratio=ratio, grid_ratio=approx,
                                     expected_ratio=(None if frame.expected is None
                                                     else frame.expected[mark_id - 1])))
        render_inputs.append((frame, markings, overlaps, ratios))
        print(f"PASS {frame_id:02d} {frame.name:<46} "
              f"{frame_row['median_ms']:.4f} ms median, {frame_row['p95_ms']:.4f} ms p95, "
              f"{frame_row['max_grid_error_pp']:.4f} pp grid error")

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "frame_timings.csv", frame_rows)
    write_csv(args.output / "marking_ratios.csv", marking_rows)
    render(args.output, render_inputs)

    import matplotlib
    summary = dict(platform=platform.platform(), cpu=cpu_model(), python=platform.python_version(),
                   numpy=np.__version__, shapely=shapely.__version__, geos=shapely.geos_version_string,
                   matplotlib=matplotlib.__version__, openblas_num_threads=os.getenv("OPENBLAS_NUM_THREADS"),
                   frames_passed=len(frame_rows), markings_checked=len(marking_rows),
                   analytically_checked=sum(row['expected_ratio'] is not None for row in marking_rows),
                   grid_step_m=GRID_STEP_M, grid_tolerance_pp=100 * GRID_TOLERANCE,
                   max_grid_error_pp=max(row['max_grid_error_pp'] for row in frame_rows),
                   warmup_calls_per_frame=args.warmup, repeats_per_frame=args.repeats,
                   timing_scope="Full bev_polygon_occlusion_ratio call, including input validation")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
