"""Pointwise occlusion from a virtual 360-degree viewpoint in BEV."""

import numpy as np


def bev_occlusion_mask(
    points_xy: np.ndarray,
    boxes: np.ndarray,
    origin_xy: np.ndarray | tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Return an (N,) bool array for (N, 2) points and (M, 5) OBBs.

    Boxes are [cx, cy, length, width, yaw], with yaw counterclockwise
    from +x in radians and length along the box's local x axis. Inputs
    share one coordinate frame. Filter the ego box and evaluation ROI
    before calling. A segment must enter the box interior before its
    endpoint. A tangent, boundary-only contact, or zero-length segment
    does not count.
    """
    points = np.asarray(points_xy, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    origin = np.asarray(origin_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2)")
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError("boxes must have shape (M, 5)")
    if origin.shape != (2,):
        raise ValueError("origin_xy must have shape (2,)")
    if not all(np.isfinite(array).all() for array in (points, boxes, origin)):
        raise ValueError("points, boxes, and origin must be finite")
    if np.any(boxes[:, 2:4] <= 0):
        raise ValueError("box length and width must be positive")

    occluded = np.zeros(len(points), dtype=bool)
    if not len(points) or not len(boxes):
        return occluded

    directions = points - origin
    nonzero_segment = np.any(directions != 0.0, axis=1)
    # Ignore roundoff at exact rotated faces, corners, and endpoints.
    roundoff = 8 * np.finfo(np.float64).eps
    direction_scale = np.max(np.abs(directions), axis=1)

    for cx, cy, length, width, yaw in boxes:
        cosine, sine = np.cos(yaw), np.sin(yaw)
        axes = np.array([[cosine, -sine], [sine, cosine]])
        offset = origin - np.array([cx, cy])
        local_origin = offset @ axes
        local_directions = directions @ axes
        half_sizes = (length / 2, width / 2)

        enter = np.zeros(len(points))
        leave = np.ones(len(points))
        for start, delta, half_size in zip(
            local_origin, local_directions.T, half_sizes
        ):
            parallel = np.abs(delta) <= roundoff * direction_scale
            near = np.full(len(points), -np.inf)
            far = np.full(len(points), np.inf)
            np.divide(-half_size - start, delta, out=near, where=~parallel)
            np.divide(half_size - start, delta, out=far, where=~parallel)
            axis_enter = np.minimum(near, far)
            axis_leave = np.maximum(near, far)

            face_tolerance = roundoff * max(np.max(np.abs(offset)), half_size)
            if abs(start) >= half_size - face_tolerance:
                axis_leave[parallel] = -np.inf
            np.maximum(enter, axis_enter, out=enter)
            np.minimum(leave, axis_leave, out=leave)

        occluded |= nonzero_segment & (enter + roundoff < leave)

    return occluded
