from __future__ import annotations

import unittest

from src import config
from src.app import (
    camera_button_direction_at,
    camera_buttons,
    camera_dial_center_hit,
    camera_dial_hit,
    dial_x_to_yaw,
    should_draw_edge_payload,
    yaw_to_dial_x,
)


class MobileControlsTests(unittest.TestCase):
    def test_camera_button_hit_areas_are_in_ground_region(self) -> None:
        left, right = camera_buttons()

        self.assertGreaterEqual(left.y, config.GROUND_TOP_Y)
        self.assertGreaterEqual(right.y, config.GROUND_TOP_Y)
        self.assertEqual(camera_button_direction_at(left.x + 2, left.y + 2), -1)
        self.assertEqual(camera_button_direction_at(right.x + 2, right.y + 2), 1)

    def test_camera_button_hit_areas_ignore_cloud_region(self) -> None:
        self.assertIsNone(camera_button_direction_at(160.0, 190.0))

    def test_camera_dial_maps_track_to_continuous_yaw(self) -> None:
        center_x = (config.CAMERA_DIAL_LEFT + config.CAMERA_DIAL_RIGHT) / 2

        self.assertEqual(dial_x_to_yaw(config.CAMERA_DIAL_LEFT), config.CAMERA_MIN_YAW)
        self.assertEqual(dial_x_to_yaw(config.CAMERA_DIAL_RIGHT), config.CAMERA_MAX_YAW)
        self.assertEqual(dial_x_to_yaw(center_x), config.CAMERA_FRONT_YAW)
        self.assertAlmostEqual(
            yaw_to_dial_x(config.CAMERA_FRONT_YAW),
            center_x,
        )

    def test_camera_dial_hit_area_is_touch_padded(self) -> None:
        self.assertTrue(
            camera_dial_hit(
                config.CAMERA_DIAL_LEFT - config.CAMERA_DIAL_TOUCH_PADDING_X,
                config.CAMERA_DIAL_Y,
            )
        )
        self.assertFalse(
            camera_dial_hit(
                config.CAMERA_DIAL_LEFT - config.CAMERA_DIAL_TOUCH_PADDING_X - 1,
                config.CAMERA_DIAL_Y,
            )
        )

    def test_camera_dial_center_hit_returns_front_tick(self) -> None:
        center_x = (config.CAMERA_DIAL_LEFT + config.CAMERA_DIAL_RIGHT) / 2

        self.assertTrue(camera_dial_center_hit(center_x, config.CAMERA_DIAL_Y))
        self.assertFalse(camera_dial_center_hit(config.CAMERA_DIAL_LEFT, config.CAMERA_DIAL_Y))

    def test_edge_payloads_are_hidden_outside_debug_display(self) -> None:
        self.assertFalse(should_draw_edge_payload(debug_enabled=False))
        self.assertTrue(should_draw_edge_payload(debug_enabled=True))


if __name__ == "__main__":
    unittest.main()
