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
    cloud_shape_level,
)
from src.motion.quantize import unpack_signed
from src.motion.runtime import WeatherMotionRuntime, hysteresis_step
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
        self.assertEqual(len(atlas.cloud_shape_level), expected)
        self.assertEqual(len(atlas.cloud_growth_ease), config.CLOUD_GROWTH_EASE_FRAMES)
        self.assertTrue(
            all(-127 <= unpack_signed(value) <= 127 for value in atlas.cloud_active_dx)
        )
        self.assertTrue(
            all(-127 <= unpack_signed(value) <= 127 for value in atlas.cloud_settling_dx)
        )
        self.assertTrue(all(0 <= value <= 15 for value in atlas.cloud_active_pulse))
        self.assertTrue(
            all(0 <= value < config.CLOUD_SHAPE_LEVELS for value in atlas.cloud_shape_level)
        )

    def test_cloud_shape_bank_changes_on_a_slow_cycle(self) -> None:
        atlas = WeatherMotionAtlas.build(seed=123)
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        node = simulation.state.nodes[result.node_id]

        frames = range(0, int(config.CLOUD_SHAPE_PERIOD_SECONDS * config.FPS), config.FPS)
        levels = {cloud_shape_level(node, atlas, frame) for frame in frames}

        self.assertGreaterEqual(len(levels), 2)
        self.assertTrue(all(0 <= level < config.CLOUD_SHAPE_LEVELS for level in levels))

    def test_cloud_growth_ease_is_event_shaped(self) -> None:
        atlas = WeatherMotionAtlas.build(seed=1)
        ease = atlas.cloud_growth_ease

        self.assertEqual(ease[0], 0)
        self.assertEqual(ease[config.CLOUD_GROWTH_PEAK_FRAME], 10)
        self.assertGreater(ease[config.CLOUD_GROWTH_SETTLE_FRAME], 0)
        self.assertEqual(ease[-1], 0)

    def test_cloud_motion_bank_is_low_passed_before_quantization(self) -> None:
        atlas = WeatherMotionAtlas.build(seed=123)
        values = [unpack_signed(value) for value in atlas.cloud_active_dx[: atlas.phase_count]]
        adjacent_diffs = [
            abs(values[(index + 1) % len(values)] - values[index])
            for index in range(len(values))
        ]

        self.assertLessEqual(max(adjacent_diffs), 4)

    def test_hysteresis_keeps_one_pixel_offsets_from_chattering(self) -> None:
        current = 0
        for raw in (40, 70, 90):
            current = hysteresis_step(current, raw, 91, 48)
            self.assertEqual(current, 0)

        current = hysteresis_step(current, 100, 91, 48)
        self.assertEqual(current, 1)
        current = hysteresis_step(current, 60, 91, 48)
        self.assertEqual(current, 1)
        current = hysteresis_step(current, 40, 91, 48)
        self.assertEqual(current, 0)

    def test_motion_runtime_gates_offset_changes_by_state_and_size(self) -> None:
        runtime = WeatherMotionRuntime()
        active_small_interval = config.CLOUD_CLUSTER_X_INTERVAL_FRAMES[
            int(CloudMotionState.ACTIVE)
        ][0]
        active_large_interval = config.CLOUD_CLUSTER_X_INTERVAL_FRAMES[
            int(CloudMotionState.ACTIVE)
        ][-1]

        self.assertLess(active_large_interval, active_small_interval)
        self.assertEqual(
            runtime.cluster_offset(1, 100, 0, 0, int(CloudMotionState.ACTIVE), "s"),
            (1, 0),
        )
        self.assertEqual(
            runtime.cluster_offset(
                1,
                0,
                0,
                active_small_interval - 1,
                int(CloudMotionState.ACTIVE),
                "s",
            ),
            (1, 0),
        )
        self.assertEqual(
            runtime.cluster_offset(
                1,
                0,
                0,
                active_small_interval,
                int(CloudMotionState.ACTIVE),
                "s",
            ),
            (0, 0),
        )

        mature_interval = config.CLOUD_CLUSTER_X_INTERVAL_FRAMES[
            int(CloudMotionState.MATURE)
        ][0]
        self.assertGreater(mature_interval, active_small_interval)

    def test_local_motion_runtime_is_even_more_sparse_for_small_nodes(self) -> None:
        runtime = WeatherMotionRuntime()
        local_interval = config.CLOUD_LOCAL_X_INTERVAL_FRAMES[int(CloudMotionState.ACTIVE)][0]
        cluster_interval = config.CLOUD_CLUSTER_X_INTERVAL_FRAMES[
            int(CloudMotionState.ACTIVE)
        ][0]

        self.assertGreater(local_interval, cluster_interval)
        self.assertEqual(
            runtime.local_offset(1, 100, 0, 0, int(CloudMotionState.ACTIVE), "s"),
            (1, 0),
        )
        self.assertEqual(
            runtime.local_offset(
                1,
                0,
                0,
                local_interval - 1,
                int(CloudMotionState.ACTIVE),
                "s",
            ),
            (1, 0),
        )
        self.assertEqual(
            runtime.local_offset(
                1,
                0,
                0,
                local_interval,
                int(CloudMotionState.ACTIVE),
                "s",
            ),
            (0, 0),
        )

    def test_growth_runtime_reports_only_triggered_event_window(self) -> None:
        atlas = WeatherMotionAtlas.build(seed=123)
        runtime = WeatherMotionRuntime()

        self.assertEqual(runtime.growth_level(7, 99, atlas.cloud_growth_ease), 0)

        runtime.trigger_growth(7, 100)

        self.assertEqual(runtime.growth_level(7, 99, atlas.cloud_growth_ease), 0)
        self.assertEqual(runtime.growth_level(7, 100, atlas.cloud_growth_ease), 0)
        self.assertEqual(
            runtime.growth_level(
                7,
                100 + config.CLOUD_GROWTH_PEAK_FRAME,
                atlas.cloud_growth_ease,
            ),
            10,
        )
        self.assertEqual(
            runtime.growth_level(7, 100 + len(atlas.cloud_growth_ease), atlas.cloud_growth_ease),
            0,
        )

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

        runtime = WeatherMotionRuntime()
        offsets = [
            cloud_render_offset(node, simulation.state, atlas, frame, runtime)
            for frame in range(int(config.CLOUD_MOTION_PERIOD_SECONDS * config.FPS))
        ]
        early = offsets[0]
        adjacent = offsets[1]
        unique_positions = {(round(offset[0], 2), round(offset[1], 2)) for offset in offsets}

        self.assertEqual(adjacent, early)
        self.assertTrue(all(abs(offset[0]) <= 1.0 for offset in offsets))
        self.assertTrue(all(abs(offset[1]) <= 1.0 for offset in offsets))
        self.assertGreaterEqual(len(unique_positions), 2)

        node.incubation = 0.9
        mature_runtime = WeatherMotionRuntime()
        mature_offsets = [
            cloud_render_offset(node, simulation.state, atlas, frame, mature_runtime)
            for frame in range(0, int(config.CLOUD_MOTION_PERIOD_SECONDS * config.FPS), 30)
        ]
        self.assertTrue(all(abs(offset[0]) <= 1.0 for offset in mature_offsets))
        self.assertTrue(all(abs(offset[1]) <= 1.0 for offset in mature_offsets))


if __name__ == "__main__":
    unittest.main()
