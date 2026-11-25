"""Synthetic road marking frames with analytically labelled cases."""

from dataclasses import dataclass

import numpy as np

from bev_scenarios import traffic_scenes


@dataclass(frozen=True)
class MarkingFrame:
    name: str
    markings: tuple[np.ndarray | tuple[np.ndarray, ...], ...]
    boxes: np.ndarray
    expected: tuple[float, ...] | None


def rectangle(x_min, x_max, y_min, y_max):
    return np.array([[x_min, y_min], [x_max, y_min],
                     [x_max, y_max], [x_min, y_max]])


def arrow(x, y, direction=1):
    outline = np.array([[0, -0.2], [6, -0.2], [6, -0.8], [10, 0],
                        [6, 0.8], [6, 0.2], [0, 0.2]])
    return outline * direction + [x, y]


def zebra(x, centers):
    return tuple(rectangle(x, x + 3, y - 0.2, y + 0.2) for y in centers)


def frame(name, markings, boxes, expected=None):
    return MarkingFrame(name, tuple(markings), np.array(boxes, dtype=float).reshape(-1, 5),
                        expected)


def dense_arterial():
    markings = []
    for x in (25, 62):
        markings.extend(arrow(x - 15, y) for y in (-3.6, 0))
        markings.extend(arrow(x + 29, y, direction=-1) for y in (3.6, 7.2))
        markings.extend([rectangle(x - 1.7, x - 1.35, -5.4, 1.8),
                         rectangle(x + 16.35, x + 16.7, 1.8, 9.0),
                         zebra(x, np.arange(-5.0, 9, 0.75)),
                         zebra(x + 12, np.arange(-5.0, 9, 0.75))])
    markings.extend(rectangle(x, x + 4, -7.4, -5.7) for x in (9, 14, 36, 41))
    dense = next(scene for scene in traffic_scenes() if scene.name == "Dense four-lane arterial")
    return frame("Dense arterial: 20 markings, 20 boxes", markings, dense.boxes[::2])


def marking_frames():
    front_car = (12, 0, 4, 2, 0)
    return [
        frame("Unobstructed stop bar", [rectangle(7, 7.4, -1.7, 1.7)], [], (0,)),
        frame("Road patches ahead of and behind a car",
              [rectangle(6, 8, -0.5, 0.5), rectangle(17, 19, -0.5, 0.5)],
              [front_car], (0, 1)),
        frame("Half-covered rectangular marking", [rectangle(9, 11, -0.5, 0.5)],
              [front_car], (0.5,)),
        frame("Rectangular marking at a shadow edge", [rectangle(20, 22, 2, 3)],
              [front_car], (0.1,)),
        frame("Triangular painted gore at a merge",
              [np.array([[16, -1], [28, 0], [16, 1]])],
              [(24, 0, 4, 2, 0)], (0.25,)),
        frame("Zebra crossing behind a motorcycle", [zebra(20, np.arange(-1.5, 5, 0.75))],
              [(11, 0, 2, 1, 0)], (1 / 3,)),
        frame("Concave straight-ahead arrow", [arrow(16, 0)],
              [(22, 0, 4, 2, 0)], (5 / 7,)),
        frame("Side marking with overlapping car shadows", [rectangle(24, 26, 0.5, 3.5)],
              [front_car, (18, 0, 4, 2, 0)], (2 / 3,)),
        dense_arterial(),
    ]
