"""Deterministic synthetic road layouts and traffic for BEV validation."""

from dataclasses import dataclass

import numpy as np


CAR = (4.6, 1.85)
VAN = (5.5, 2.05)
BUS = (12.0, 2.55)
TRUCK = (16.5, 2.55)
MOTORCYCLE = (2.1, 0.8)


@dataclass(frozen=True)
class Scene:
    name: str
    lanes: tuple[np.ndarray, ...]
    boxes: np.ndarray


def road_xy(station, lateral, curvature=0.0):
    """Map station and leftward offset to a straight or constant-radius road."""
    station, lateral = np.broadcast_arrays(station, lateral)
    if curvature == 0:
        return np.stack((station, lateral), axis=-1)
    angle = curvature * station
    x = np.sin(angle) / curvature - lateral * np.sin(angle)
    y = (1 - np.cos(angle)) / curvature + lateral * np.cos(angle)
    return np.stack((x, y), axis=-1)


def stations(start, end):
    return np.linspace(start, end, int(np.ceil((end - start) / 0.25)) + 1)


def box(station, lateral, size=CAR, oncoming=False, curvature=0.0):
    x, y = road_xy(station, lateral, curvature)
    heading = station * curvature + (np.pi if oncoming else 0)
    return (x, y, *size, heading)


def scene(name, lanes, boxes):
    return Scene(name, tuple(lanes), np.asarray(boxes, dtype=float).reshape(-1, 5))


def road(name, traffic, offsets=(-1.8, 1.8, 5.4), curvature=0.0, bounds=(-32, 68)):
    s = stations(*bounds)
    return scene(
        name,
        [road_xy(s, offset, curvature) for offset in offsets],
        [box(*vehicle, curvature=curvature) for vehicle in traffic],
    )


def junction(name, traffic, t_junction=False):
    x_left, x_right = 10.4, 17.6
    y_lower, y_upper = -1.8, 5.4
    lanes = []
    for y in (y_lower, 1.8, y_upper):
        if t_junction and y == y_upper:
            lanes.append(road_xy(stations(-25, 55), y))
        else:
            lanes.extend([road_xy(stations(-25, x_left), y),
                          road_xy(stations(x_right, 55), y)])
    for x in (x_left, 14.0, x_right):
        south = stations(-32, y_lower)
        lanes.append(np.column_stack((np.full_like(south, x), south)))
        if not t_junction:
            north = stations(y_upper, 35)
            lanes.append(np.column_stack((np.full_like(north, x), north)))
    return scene(name, lanes, traffic)


def merge():
    s = stations(-35, 75)
    ramp = stations(-34, 25)
    # Right-hand entry ramp tapers into the through lane with zero final slope.
    offset = -7.2 * ((25 - ramp) / 59) ** 2
    nose = 25 - 59 / np.sqrt(2)
    lanes = [road_xy(s, 1.8), road_xy(s, 5.4),
             road_xy(stations(-35, nose), -1.8),
             road_xy(ramp, offset - 1.8), road_xy(stations(25, 75), -1.8)]
    ramp_approach = stations(-34, nose)
    lanes.append(road_xy(ramp_approach, 1.8 - 7.2 * ((25 - ramp_approach) / 59) ** 2))
    lanes.extend(road_xy(stations(x, x + 2), -1.8) for x in np.arange(nose, 16, 5))
    ramp_vehicle_x = -18.0
    ramp_vehicle_y = -7.2 * ((25 - ramp_vehicle_x) / 59) ** 2
    ramp_heading = np.arctan(14.4 * (25 - ramp_vehicle_x) / 59**2)
    return scene("Highway on-ramp merge", lanes, [
        (ramp_vehicle_x, ramp_vehicle_y, *CAR, ramp_heading),
        box(-10, 0, VAN), box(33, 0, TRUCK), box(47, 3.6, CAR, True),
    ])


def traffic_scenes():
    two_each_way = (-5.4, -1.8, 1.8, 5.4, 9.0)
    return [
        road("Clear two-way road", []),
        road("Following a passenger car", [(14, 0)]),
        road("Following a box truck", [(19, 0, TRUCK), (46, 0)]),
        road("Oncoming delivery van", [(18, 3.6, VAN, True)]),
        road("Close follower in rear", [(-10, 0), (-23, 3.6, CAR, True)]),
        road("Stop-and-go platoon", [(7, 0), (15, 0, VAN), (24, 0), (36, 0, BUS)]),
        road("City bus at curb stop", [(17, -0.5, BUS), (37, 3.6, CAR, True)]),
        road("Curbside parallel parking", [
            (10, -3.15), (20, -3.15, VAN), (33, -3.15),
            (23, 0), (44, 3.6, CAR, True),
        ], (-4.5, -1.8, 1.8, 5.4)),
        road("Passing in adjacent lane", [
            (-5, -3.6), (10, -3.6), (22, 0, VAN), (40, 3.6, CAR, True),
        ], two_each_way),
        road("Opposing queues", [
            (9, 0), (17, 0), (27, 0, VAN),
            (16, 3.6, CAR, True), (24, 3.6, VAN, True), (35, 3.6, BUS, True),
        ]),
        road("Motorcycle between cars", [(9, 0.55, MOTORCYCLE), (17, 0), (35, 0, VAN)]),
        road("Dense four-lane arterial", [
            (x, y, CAR if i % 3 else VAN, y > 0)
            for y, phase in [(-3.6, 0), (0, 4), (3.6, 3), (7.2, 6)]
            for i, x in enumerate(range(-28 + phase, 66, 9))
            if y != 0 or abs(x) >= 8
        ], two_each_way),
        road("Sweeping left bend", [(14, 0), (31, 3.6, VAN, True), (47, 0)],
             curvature=1 / 58),
        road("Sweeping right bend", [(15, 0, VAN), (-14, 0), (32, 3.6, CAR, True)],
             curvature=-1 / 52),
        road("Bus through tight curve", [(17, 0, BUS), (35, 3.6, CAR, True)],
             curvature=1 / 32, bounds=(-22, 46)),
        road("Adjacent traffic through bend", [
            (10, -3.6), (21, 0), (30, -3.6, VAN), (42, 3.6, CAR, True),
        ], two_each_way, curvature=-1 / 68),
        merge(),
        junction("Four-way junction with cross traffic", [
            box(6.4, 0), box(25, 3.6, CAR, True),
            (15.8, 2.1, *VAN, np.pi / 2), (12.2, 14, *CAR, -np.pi / 2),
        ]),
        junction("Four-way junction with stopped approaches", [
            box(6.4, 0), box(24, 3.6, VAN, True),
            (15.8, -8, *BUS, np.pi / 2), (12.2, 10, *CAR, -np.pi / 2),
        ]),
        junction("T-junction with turning car", [
            box(6.4, 0), box(28, 3.6, VAN, True),
            (15.9, -0.1, *CAR, 0.63), (15.8, -10, *CAR, np.pi / 2),
            (12.2, -8.5, *CAR, -np.pi / 2),
        ], t_junction=True),
    ]
