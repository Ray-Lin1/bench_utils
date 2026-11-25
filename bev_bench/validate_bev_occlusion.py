"""Check 20 traffic frames, benchmark point masking, and render review plots."""

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import platform
from time import perf_counter_ns

import numpy as np

from bev_occlusion import bev_occlusion_mask
from bev_scenarios import CAR, traffic_scenes


VISIBLE = "#117a65"
OCCLUDED = "#d1452f"
BOX_FILL = "#c4d1df"
BOX_EDGE = "#52657b"


def corners(box):
    cx, cy, length, width, yaw = box
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    left = np.array([-np.sin(yaw), np.cos(yaw)])
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    return [cx, cy] + signs[:, :1] * length / 2 * forward + signs[:, 1:] * width / 2 * left


def reference_mask(points, boxes):
    """Independent check by separating-axis projections in world coordinates."""
    result = np.zeros(len(points), dtype=bool)
    normals = points[:, [1, 0]] * [-1, 1]
    length = np.linalg.norm(normals, axis=1)
    np.divide(normals, length[:, None], where=length[:, None] > 0, out=normals)
    tolerance = 1e-10  # Metres, to handle exact contacts in the synthetic scenes.
    for box in boxes:
        polygon = corners(box)
        axes = np.diff(polygon[:3], axis=0)
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        extent = polygon @ axes.T
        endpoint = points @ axes.T
        overlaps = (np.maximum(endpoint, 0) > extent.min(axis=0) + tolerance)
        overlaps &= (np.minimum(endpoint, 0) < extent.max(axis=0) - tolerance)
        side = normals @ polygon.T
        crosses = (side.min(axis=1) < -tolerance) & (side.max(axis=1) > tolerance)
        result |= overlaps.all(axis=1) & crosses
    return result


def check_boundaries():
    points = np.array([[8, 0], [10, 0], [12, 0], [20, 0], [20, 2],
                       [20, 3], [0, 0], [-20, 0], [20, 1.99]])
    boxes = np.array([[12, 0, 4, 2, 0]])
    expected = [False, False, True, True, False, False, False, False, True]
    np.testing.assert_array_equal(bev_occlusion_mask(points, boxes), expected)
    np.testing.assert_array_equal(reference_mask(points, boxes), expected)

    angle = 0.63
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, sine], [-sine, cosine]])
    origin = np.array([1.3, -4.2])
    rotated_box = np.array([[*(boxes[0, :2] @ rotation + origin), 4, 2, angle]])
    np.testing.assert_array_equal(
        bev_occlusion_mask(points @ rotation + origin, rotated_box, origin), expected
    )
    # The origin lies on the extension of the box's top edge.
    np.testing.assert_array_equal(
        bev_occlusion_mask(np.array([[20, 0], [20, -0.1]]), [[12, -1, 4, 2, 0]]),
        [False, True],
    )
    np.testing.assert_array_equal(
        bev_occlusion_mask(np.array([[-8, 0], [-20, 0]]), [[-12, 0, 4, 2, np.pi]]),
        [False, True],
    )
    assert bev_occlusion_mask(np.empty((0, 2)), boxes).shape == (0,)
    assert not bev_occlusion_mask(points, np.empty((0, 5))).any()


def check_vehicle_spacing(scene):
    footprints = [corners([0, 0, *CAR, 0]), *map(corners, scene.boxes)]
    for first, second in combinations(footprints, 2):
        axes = np.concatenate((np.diff(first[:3], axis=0), np.diff(second[:3], axis=0)))
        first_projection = first @ axes.T
        second_projection = second @ axes.T
        overlap = np.minimum(first_projection.max(axis=0), second_projection.max(axis=0))
        overlap -= np.maximum(first_projection.min(axis=0), second_projection.min(axis=0))
        if np.all(overlap > 0):
            raise AssertionError(f"Vehicle footprints overlap in {scene.name}")


def measure(points, boxes, warmup, repeats):
    for _ in range(warmup):
        bev_occlusion_mask(points, boxes)
    elapsed_ns = np.empty(repeats)
    for index in range(repeats):
        start = perf_counter_ns()
        bev_occlusion_mask(points, boxes)
        elapsed_ns[index] = perf_counter_ns() - start
    return elapsed_ns / 1_000_000


def draw_scene(ax, frame, scene, points, mask, compact=False):
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon

    ax.set_facecolor("#f5f7fa")
    ax.set_axisbelow(True)
    ax.grid(color="#dfe5eb", linewidth=0.6)
    for box in scene.boxes:
        ax.add_patch(Polygon(corners(box), facecolor=BOX_FILL, edgecolor=BOX_EDGE,
                             linewidth=0.85, zorder=2))
        cx, cy, length, _, yaw = box
        ax.plot([cx, cx + 0.42 * length * np.cos(yaw)],
                [cy, cy + 0.42 * length * np.sin(yaw)],
                color=BOX_EDGE, linewidth=1.0, solid_capstyle="round", zorder=3)
    ax.add_patch(Polygon(corners([0, 0, *CAR, 0]), facecolor="#d7e6ff",
                         edgecolor="#295da8", linewidth=1, zorder=4))
    ax.scatter([0], [0], color="#295da8", marker="*", s=34, zorder=6)

    # Colour each half-edge by its nearest sample's mask.
    start = 0
    for lane in scene.lanes:
        values = mask[start:start + len(lane)]
        start += len(lane)
        mid = (lane[:-1] + lane[1:]) / 2
        segments = np.concatenate((np.stack((lane[:-1], mid), axis=1),
                                   np.stack((mid, lane[1:]), axis=1)))
        colors = np.where(np.concatenate((values[:-1], values[1:])), OCCLUDED, VISIBLE)
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=1.6,
                                         capstyle="round", zorder=5))

    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    ax.set_xlim(x_min - 3, x_max + 3)
    y_margin = max(2, (20 - (y_max - y_min)) / 2)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7 if compact else 9)
    ax.set_title(f"{frame:02d}  {scene.name}", loc="left", fontweight="bold",
                 fontsize=9 if compact else 14)
    ax.set_xlabel("x (m)", fontsize=8 if compact else 10)
    ax.set_ylabel("y (m)", fontsize=8 if compact else 10)


def legend_handles():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    return [Line2D([], [], color=VISIBLE, linewidth=2, label="Visible"),
            Line2D([], [], color=OCCLUDED, linewidth=2, label="Occluded"),
            Patch(facecolor=BOX_FILL, edgecolor=BOX_EDGE, label="Obstacle footprint"),
            Line2D([], [], color="#295da8", marker="*", linestyle="None", label="Ego origin")]


def render(output, frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overview, panels = plt.subplots(5, 4, figsize=(19, 11), layout="constrained",
                                   height_ratios=[1, 1, 1, 1.8, 2.3])
    overview.suptitle("360° BEV lane occlusion • 20 traffic frames", fontsize=20,
                      fontweight="bold")
    for (frame, scene, points, mask), panel in zip(frames, panels.flat):
        draw_scene(panel, frame, scene, points, mask, compact=True)
        figure, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
        draw_scene(ax, frame, scene, points, mask)
        figure.legend(handles=legend_handles(), loc="outside lower center", ncols=4,
                      frameon=False)
        name = scene.name.lower().replace(" ", "_").replace("-", "_")
        figure.savefig(output / f"{frame:02d}_{name}.png", dpi=150)
        plt.close(figure)
    overview.legend(handles=legend_handles(), loc="outside lower center", ncols=4,
                    frameon=False)
    overview.savefig(output / "overview.png", dpi=150)
    plt.close(overview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("bev_review"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be nonnegative and --repeats must be positive")

    check_boundaries()
    scenes = traffic_scenes()
    assert len(scenes) == 20
    frames, rows = [], []
    for frame, scene in enumerate(scenes, 1):
        check_vehicle_spacing(scene)
        points = np.concatenate(scene.lanes)
        mask = bev_occlusion_mask(points, scene.boxes)
        np.testing.assert_array_equal(mask, reference_mask(points, scene.boxes),
                                      err_msg=scene.name)
        blocked = int(np.count_nonzero(mask))
        expected_mix = blocked == 0 if frame == 1 else 0 < blocked < len(points)
        if not expected_mix:
            raise AssertionError(f"Unexpected mask coverage in {scene.name}")
        elapsed = measure(points, scene.boxes, args.warmup, args.repeats)
        row = dict(frame=frame, scene=scene.name, lane_fragments=len(scene.lanes),
                   points=len(points), boxes=len(scene.boxes), occluded_points=blocked,
                   visible_points=len(points) - blocked, repeats=args.repeats,
                   min_ms=round(float(elapsed.min()), 6),
                   median_ms=round(float(np.median(elapsed)), 6),
                   mean_ms=round(float(elapsed.mean()), 6),
                   p95_ms=round(float(np.percentile(elapsed, 95)), 6))
        frames.append((frame, scene, points, mask))
        rows.append(row)
        print(f"PASS {frame:02d} {scene.name:<40} {blocked:>4}/{len(points):<4} "
              f"blocked  {row['median_ms']:>8.4f} ms median  {row['p95_ms']:>8.4f} ms p95")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "frame_timings.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    render(args.output, frames)

    import matplotlib
    summary = dict(platform=platform.platform(), python=platform.python_version(),
                   numpy=np.__version__, matplotlib=matplotlib.__version__,
                   frames_passed=len(rows), total_points=sum(row["points"] for row in rows),
                   total_occluded_points=sum(row["occluded_points"] for row in rows),
                   warmup_calls_per_frame=args.warmup, repeats_per_frame=args.repeats,
                   median_frame_median_ms=round(float(np.median(
                       [row["median_ms"] for row in rows]
                   )), 6),
                   max_frame_median_ms=max(row["median_ms"] for row in rows),
                   max_frame_p95_ms=max(row["p95_ms"] for row in rows),
                   timing_scope="bev_occlusion_mask call, including input checks and allocation")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                             encoding="utf-8")
    print(f"{len(rows)} traffic frames and boundary checks passed. Output {args.output}")


if __name__ == "__main__":
    main()
