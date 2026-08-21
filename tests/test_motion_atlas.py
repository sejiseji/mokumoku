from __future__ import annotations

import unittest

from src import config
from src.camera.camera import build_camera_basis
from src.camera.projection import project_point
from src.cloud.simulation import CloudSimulation
from src.motion.atlas import WeatherMotionAtlas
from src.motion.cloud_motion import (
    CloudMotionState,
    cloud_motion_state_for_node,
    cloud_render_offset,
)
from src.motion.quantize import unpack_signed
from src.rng import RandomSource


class MotionAtlasTests(unittest.TestCase):
    def test_cloud_motion_banks_have_expected_dimensions_and_ranges(self) -> None:
        atlas = WeatherMotionAtlas.build(seed=123)
        expected = config.MOTION_PHASE_COUNT * config.MOTION_GROUP_COUNT

        self.assertEqual(atlas.phase_count, config.MOTION_PHASE_COUNT)
        self.assertEqual(atlas.group_count, config.MOTION_GROUP_COUNT)
        self.assertEqual(len(atlas.cloud_active_dx), expected)
        self.assertEqual(len(atlas.cloud_settling_dx), expected)
        self.assertEqual(len(atlas.cloud_mature_dx), expected)
        self.assertTrue(all(-2 <= unpack_signed(value) <= 2 for value in atlas.cloud_active_dx))
        self.assertTrue(all(-2 <= unpack_signed(value) <= 2 for value in atlas.cloud_active_dy))
        self.assertTrue(
            all(-1 <= unpack_signed(value) <= 1 for value in atlas.cloud_settling_dx)
        )
        self.assertTrue(
            all(-1 <= unpack_signed(value) <= 1 for value in atlas.cloud_mature_dx)
        )
        self.assertTrue(all(0 <= value <= 15 for value in atlas.cloud_active_pulse))

    def test_weather_motion_atlas_is_deterministic_by_seed(self) -> None:
        first = WeatherMotionAtlas.build(seed=777)
        second = WeatherMotionAtlas.build(seed=777)
        different = WeatherMotionAtlas.build(seed=778)

        self.assertEqual(first.cloud_active_dx, second.cloud_active_dx)
        self.assertEqual(first.cloud_settling_dy, second.cloud_settling_dy)
        self.assertEqual(first.rain_sway, second.rain_sway)
        self.assertEqual(first.rain_trajectories, second.rain_trajectories)
        self.assertEqual(first.lightning_templates, second.lightning_templates)
        self.assertNotEqual(first.cloud_active_dx, different.cloud_active_dx)

    def test_amplitude_table_settles_as_incubation_increases(self) -> None:
        atlas = WeatherMotionAtlas.build(seed=1)
        active_low_incubation = atlas.amplitude_table[15][0]
        active_high_incubation = atlas.amplitude_table[15][15]
        inactive_low_incubation = atlas.amplitude_table[0][0]

        self.assertGreater(active_low_incubation, active_high_incubation)
        self.assertGreater(active_low_incubation, inactive_low_incubation)

    def test_rain_trajectories_and_lightning_templates_are_bounded(self) -> None:
        atlas = WeatherMotionAtlas.build(seed=321)

        self.assertEqual(len(atlas.rain_trajectories), config.RAIN_TRAJECTORY_COUNT)
        for trajectory in atlas.rain_trajectories:
            self.assertEqual(len(trajectory.dx), config.RAIN_TRAJECTORY_STEPS)
            y_values = list(trajectory.dy)
            self.assertEqual(y_values, sorted(y_values))
            self.assertTrue(all(-16 <= unpack_signed(value) <= 16 for value in trajectory.dx))

        self.assertEqual(len(atlas.lightning_templates), config.LIGHTNING_TEMPLATE_COUNT)
        for template in atlas.lightning_templates:
            self.assertEqual(template.main_points[0], (0, 0))
            self.assertEqual(template.main_points[-1], (255, 0))
            self.assertLessEqual(len(template.branches), config.LIGHTNING_TEMPLATE_MAX_BRANCHES)
            self.assertTrue(all(-127 <= v <= 127 for _u, v in template.main_points))

    def test_cloud_motion_state_transitions_from_active_to_mature(self) -> None:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        node = simulation.state.nodes[result.node_id]

        self.assertEqual(cloud_motion_state_for_node(node), CloudMotionState.ACTIVE)

        node.activation = 0.2
        node.noise = 0.16
        self.assertEqual(cloud_motion_state_for_node(node), CloudMotionState.SETTLING)

        node.incubation = 0.8
        self.assertEqual(cloud_motion_state_for_node(node), CloudMotionState.MATURE)

    def test_cloud_render_offset_uses_lut_without_adjacent_frame_jitter(self) -> None:
        atlas = WeatherMotionAtlas.build(seed=12345)
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        node = simulation.state.nodes[result.node_id]
        projection = project_point(node.position, camera)
        self.assertTrue(projection.visible)

        early = cloud_render_offset(node, simulation.state, atlas, 0)
        adjacent = cloud_render_offset(node, simulation.state, atlas, 1)
        later = cloud_render_offset(node, simulation.state, atlas, 240)

        self.assertLessEqual(abs(adjacent[0] - early[0]), 1.0)
        self.assertLessEqual(abs(adjacent[1] - early[1]), 1.0)
        self.assertNotEqual(
            (round(early[0], 2), round(early[1], 2)),
            (round(later[0], 2), round(later[1], 2)),
        )

        node.incubation = 0.9
        mature = cloud_render_offset(node, simulation.state, atlas, 240)
        self.assertLessEqual(abs(mature[0]), abs(later[0]) + 0.01)
        self.assertLessEqual(abs(mature[1]), abs(later[1]) + 0.01)


if __name__ == "__main__":
    unittest.main()
