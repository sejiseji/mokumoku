from __future__ import annotations

import unittest

from src import config
from src.app import camera_button_direction_at, camera_buttons


class MobileControlsTests(unittest.TestCase):
    def test_camera_button_hit_areas_are_in_ground_region(self) -> None:
        left, right = camera_buttons()

        self.assertGreaterEqual(left.y, config.GROUND_TOP_Y)
        self.assertGreaterEqual(right.y, config.GROUND_TOP_Y)
        self.assertEqual(camera_button_direction_at(left.x + 2, left.y + 2), -1)
        self.assertEqual(camera_button_direction_at(right.x + 2, right.y + 2), 1)

    def test_camera_button_hit_areas_ignore_cloud_region(self) -> None:
        self.assertIsNone(camera_button_direction_at(160.0, 190.0))


if __name__ == "__main__":
    unittest.main()
