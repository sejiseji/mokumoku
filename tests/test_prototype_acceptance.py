from __future__ import annotations

import unittest

from src import config
from src.motion.atlas import WeatherMotionAtlas
from src.motion.runtime import WeatherMotionRuntime
from src.prototype_acceptance import (
    build_connected_cloud,
    max_ambient_offset,
    max_camera_tap_error,
    run_prototype_a_acceptance,
)


class PrototypeAcceptanceTests(unittest.TestCase):
    def test_prototype_a_acceptance_scenario_passes(self) -> None:
        report = run_prototype_a_acceptance(seed=12345)

        self.assertTrue(report.passed, report)
        self.assertGreaterEqual(report.node_count, 5)
        self.assertGreaterEqual(report.bridge_count, 2)
        self.assertTrue(report.normal_edges_hidden)

    def test_camera_tap_alignment_is_stable_across_all_a_cameras(self) -> None:
        error = max_camera_tap_error(seed=12345)

        self.assertLessEqual(error, config.PROTOTYPE_A_TAP_ALIGNMENT_TOLERANCE_PX)

    def test_quiet_ambient_keeps_node_centers_still_for_acceptance_window(self) -> None:
        simulation = build_connected_cloud(seed=12345)
        atlas = WeatherMotionAtlas.build(seed=12345)
        runtime = WeatherMotionRuntime()

        offset = max_ambient_offset(simulation, atlas, runtime)

        self.assertEqual(offset, 0.0)


if __name__ == "__main__":
    unittest.main()
