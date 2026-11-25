"""Continuous area occlusion for painted road markings in BEV."""

from collections.abc import Sequence

import numpy as np
import shapely
from shapely.geometry import MultiPoint, Polygon


def bev_polygon_occlusion_ratio(
    polygons_xy: Sequence[np.ndarray | Sequence[np.ndarray]],
    boxes: np.ndarray,
    origin_xy: np.ndarray | tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Return one occluded-area fraction per marking in input order.

    A marking is a (K, 2) exterior ring, or a nonempty sequence of such
    rings for a multipart marking. Parts are unioned, so paint is counted
    once and gaps between stripes are excluded. Rings contain ordered XY
    vertices and may be open or closed. Invalid or zero-area rings raise
    ValueError. Boxes use [cx, cy, length, width, yaw] in the same BEV frame,
    with counterclockwise yaw in radians. Filter the ego box upstream.
    """
    markings, blocked = _marking_intersections(polygons_xy, boxes, origin_xy)
    return np.clip(shapely.area(blocked) / shapely.area(markings), 0.0, 1.0)


def _marking_intersections(polygons_xy, boxes, origin_xy):
    """Return aligned geometries for area measurement and review plots."""
    boxes = np.asarray(boxes, dtype=np.float64)
    origin = np.asarray(origin_xy, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError("boxes must have shape (M, 5)")
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("origin_xy must contain two finite coordinates")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:4] <= 0):
        raise ValueError("boxes must be finite with positive length and width")

    markings = np.array([
        _marking_geometry(marking, index) for index, marking in enumerate(polygons_xy)
    ], dtype=object)
    if not len(markings):
        return markings, markings.copy()

    bounds = np.abs(shapely.bounds(markings) - np.tile(origin, 2))
    extent = np.maximum(bounds[:, :2], bounds[:, 2:])
    max_range = np.max(np.hypot(extent[:, 0], extent[:, 1]))
    shadows = _shadow_union(boxes, origin, max_range)
    return markings, shapely.intersection(markings, shadows)


def _marking_geometry(marking, index):
    parts = (marking,) if isinstance(marking, np.ndarray) and marking.ndim == 2 else marking
    polygons = []
    for part in parts:
        vertices = np.asarray(part, dtype=np.float64)
        if (vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3
                or not np.isfinite(vertices).all()):
            raise ValueError(f"marking {index} must contain finite (K, 2) rings with K >= 3")
        polygon = Polygon(vertices)
        if not polygon.is_valid or not np.isfinite(polygon.area) or polygon.area <= 0:
            raise ValueError(f"marking {index} has an invalid or zero-area ring")
        polygons.append(polygon)
    if not polygons:
        raise ValueError(f"marking {index} has no rings")
    return polygons[0] if len(polygons) == 1 else shapely.union_all(polygons)


def _shadow_union(boxes, origin, max_range):
    shadows = []
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    roundoff = 8 * np.finfo(np.float64).eps
    for cx, cy, length, width, yaw in boxes:
        cosine, sine = np.cos(yaw), np.sin(yaw)
        axes = np.array([[cosine, -sine], [sine, cosine]])
        center = np.array([cx, cy]) - origin
        half = np.array([length, width]) / 2
        local_origin = -center @ axes
        tolerance = roundoff * max(np.max(np.abs(center)), np.max(half))
        gaps = np.abs(local_origin) - half
        distance = np.linalg.norm(np.maximum(gaps, 0))
        if distance >= max_range:
            continue

        if np.all(gaps < -tolerance):
            # Every nonzero ray starts inside this box.
            return shapely.box(*(origin - max_range), *(origin + max_range))

        if np.all(gaps <= tolerance):
            # At an edge/corner, only inward directions are blocked. Keep
            # the faces through the origin fixed under radial scaling.
            on_face = np.abs(gaps) <= tolerance
            local_origin[on_face] = np.copysign(half[on_face], local_origin[on_face])
            face_distances = np.abs(np.concatenate((half - local_origin, half + local_origin)))
            distance = np.min(face_distances[face_distances > 0])

        # Any marking position is at most max_range away. Extending the
        # box by this common radial scale covers the entire relevant shadow.
        corners = (signs * half - local_origin) @ axes.T
        scale = 1 + max_range / distance
        shadows.append(MultiPoint(np.concatenate((corners, scale * corners)) + origin).convex_hull)

    return shapely.union_all(shadows)
