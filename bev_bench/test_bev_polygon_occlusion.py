"""Analytical tests for continuous BEV marking occlusion."""

import unittest

import numpy as np

from bev_polygon_occlusion import bev_polygon_occlusion_ratio
from bev_polygon_scenarios import marking_frames, rectangle


class PolygonOcclusionTests(unittest.TestCase):
    def test_traffic_frames(self):
        for frame in marking_frames():
            if frame.expected is not None:
                with self.subTest(frame=frame.name):
                    ratios = bev_polygon_occlusion_ratio(frame.markings, frame.boxes)
                    np.testing.assert_allclose(ratios, frame.expected, atol=1e-12, rtol=0)

    def test_rigid_transform_of_multipart_marking(self):
        frame = marking_frames()[5]
        yaw = 0.73
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[cosine, sine], [-sine, cosine]])
        origin = np.array([33.0, -19.0])
        transformed = tuple(part @ rotation + origin for part in frame.markings[0])
        boxes = frame.boxes.copy()
        boxes[:, :2] = boxes[:, :2] @ rotation + origin
        boxes[:, 4] += yaw
        ratios = bev_polygon_occlusion_ratio([transformed], boxes, origin)
        np.testing.assert_allclose(ratios, frame.expected, atol=1e-12, rtol=0)

    def test_paint_parts_are_unioned(self):
        # The shared area is behind the front face and must count only once.
        parts = (rectangle(9, 10.75, -0.5, 0.5), rectangle(10.25, 11, -0.5, 0.5))
        np.testing.assert_allclose(bev_polygon_occlusion_ratio([parts], [[12, 0, 4, 2, 0]]),
                                   [0.5], atol=1e-12, rtol=0)

    def test_contact_with_front_face_has_zero_area(self):
        ratios = bev_polygon_occlusion_ratio([rectangle(8, 10, -0.5, 0.5)], [[12, 0, 4, 2, 0]])
        np.testing.assert_array_equal(ratios, [0.0])

    def test_origin_on_box_face_and_corner(self):
        marking = rectangle(-3, 3, -3, 3)
        ratios = bev_polygon_occlusion_ratio([marking], [[1, 0, 2, 2, 0]])
        np.testing.assert_allclose(ratios, [0.5], atol=1e-12, rtol=0)
        ratios = bev_polygon_occlusion_ratio([marking], [[1, 1, 2, 2, 0]])
        np.testing.assert_allclose(ratios, [0.25], atol=1e-12, rtol=0)

    def test_origin_inside_box(self):
        ratios = bev_polygon_occlusion_ratio([rectangle(-3, 3, -3, 3)], [[0, 0, 4, 2, 0]])
        np.testing.assert_array_equal(ratios, [1.0])

    def test_empty_marking_list_and_empty_box_array(self):
        ratios = bev_polygon_occlusion_ratio([], np.empty((0, 5)))
        self.assertEqual(ratios.shape, (0,))
        ratios = bev_polygon_occlusion_ratio([rectangle(3, 4, 0, 1)], np.empty((0, 5)))
        np.testing.assert_array_equal(ratios, [0.0])

    def test_rear_obstacle_and_far_box_filter(self):
        marking = rectangle(-25, -23, -0.5, 0.5)
        ratios = bev_polygon_occlusion_ratio([marking], [[-12, 0, 4, 2, np.pi],
                                                        [100, 0, 4, 2, 0]])
        np.testing.assert_array_equal(ratios, [1.0])

    def test_invalid_markings_and_boxes_raise(self):
        invalid_rings = [np.array([[0, 0], [1, 1], [2, 2]]),
                         np.array([[0, 0], [2, 2], [0, 2], [2, 0]]),
                         np.array([[0, 0], [1, 0], [0, np.nan]]), ()]
        for invalid in invalid_rings:
            with self.subTest(ring=invalid), self.assertRaisesRegex(ValueError, "marking 0"):
                bev_polygon_occlusion_ratio([invalid], np.empty((0, 5)))
        with self.assertRaisesRegex(ValueError, "positive length and width"):
            bev_polygon_occlusion_ratio([], [[0, 0, 0, 1, 0]])


if __name__ == "__main__":
    unittest.main()
