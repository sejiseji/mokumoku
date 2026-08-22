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
from src.motion.runtime import (
    TouchResponseKind,
    WeatherMotionRuntime,
    choose_morph_node_ids,
    hysteresis_step,
    morph_interval_bounds,
)
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
        self.assertGreater(ease[config.CLOUD_GROWTH_PEAK_FRAME + 1], 0)
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

    def test_quiet_cloud_render_offset_disables_ambient_position_and_size_pulse(self) -> None:
        atlas = WeatherMotionAtlas.build(seed=12345)
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        node = simulation.state.nodes[result.node_id]
        runtime = WeatherMotionRuntime()

        offsets = [
            cloud_render_offset(node, simulation.state, atlas, frame, runtime)
            for frame in range(0, int(config.CLOUD_MOTION_PERIOD_SECONDS * config.FPS), 30)
        ]

        self.assertTrue(config.QUIET_CLOUD_MOTION_ENABLED)
        self.assertFalse(config.AMBIENT_LOCAL_POSITION_ENABLED)
        self.assertFalse(config.AMBIENT_SIZE_PULSE_ENABLED)
        self.assertFalse(config.ENABLE_CLUSTER_AMBIENT_OFFSET)
        self.assertEqual(set(offsets), {(0.0, 0.0, 1.0)})

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
            config.CLOUD_PULSE_STRENGTH_BY_DISTANCE[0],
        )
        self.assertEqual(
            runtime.growth_level(7, 100 + len(atlas.cloud_growth_ease), atlas.cloud_growth_ease),
            0,
        )

    def test_growth_wave_propagates_to_graph_distance_two_with_delay(self) -> None:
        atlas = WeatherMotionAtlas.build(seed=123)
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        root_result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(root_result.node_id)
        root = simulation.state.nodes[root_result.node_id]
        root_projection = project_point(root.position, camera)
        child = simulation.add_child_near_screen(
            root,
            root_projection.screen_x + 30.0,
            root_projection.screen_y,
            camera,
        )
        self.assertIsNotNone(child)
        child_id = child.id
        child_projection = project_point(child.position, camera)
        grandchild = simulation.add_child_near_screen(
            child,
            child_projection.screen_x + 30.0,
            child_projection.screen_y,
            camera,
        )
        self.assertIsNotNone(grandchild)
        grandchild_id = grandchild.id

        runtime = WeatherMotionRuntime()
        runtime.trigger_growth_wave(simulation.state, root_result.node_id, 200)

        root_pulse = runtime.growth_pulses[root_result.node_id]
        child_pulse = runtime.growth_pulses[child_id]
        grandchild_pulse = runtime.growth_pulses[grandchild_id]

        self.assertEqual(root_pulse.start_frame, 200)
        self.assertEqual(child_pulse.start_frame, 200 + config.CLOUD_PULSE_PROPAGATION_DELAY_FRAMES)
        self.assertEqual(
            grandchild_pulse.start_frame,
            200 + config.CLOUD_PULSE_PROPAGATION_DELAY_FRAMES * 2,
        )
        self.assertEqual(root_pulse.strength_level, config.CLOUD_PULSE_STRENGTH_BY_DISTANCE[0])
        self.assertEqual(child_pulse.strength_level, config.CLOUD_PULSE_STRENGTH_BY_DISTANCE[1])
        self.assertEqual(
            grandchild_pulse.strength_level,
            config.CLOUD_PULSE_STRENGTH_BY_DISTANCE[2],
        )

        self.assertGreater(
            runtime.growth_level(
                child_id,
                child_pulse.start_frame + config.CLOUD_GROWTH_PEAK_FRAME,
                atlas.cloud_growth_ease,
            ),
            runtime.growth_level(
                grandchild_id,
                grandchild_pulse.start_frame + config.CLOUD_GROWTH_PEAK_FRAME,
                atlas.cloud_growth_ease,
            ),
        )

    def test_touch_response_kinds_use_distinct_windows_and_distances(self) -> None:
        atlas = WeatherMotionAtlas.build(seed=123)
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        root_result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(root_result.node_id)
        root = simulation.state.nodes[root_result.node_id]
        root_projection = project_point(root.position, camera)
        child = simulation.add_child_near_screen(
            root,
            root_projection.screen_x + 30.0,
            root_projection.screen_y,
            camera,
        )
        self.assertIsNotNone(child)
        child_id = child.id
        child_projection = project_point(child.position, camera)
        grandchild = simulation.add_child_near_screen(
            child,
            child_projection.screen_x + 30.0,
            child_projection.screen_y,
            camera,
        )
        self.assertIsNotNone(grandchild)
        grandchild_id = grandchild.id

        runtime = WeatherMotionRuntime()
        runtime.trigger_response_wave(
            simulation.state,
            root_result.node_id,
            300,
            TouchResponseKind.DRAG_START,
        )

        self.assertEqual(
            runtime.growth_pulses[root_result.node_id].response_kind,
            TouchResponseKind.DRAG_START,
        )
        self.assertIn(child_id, runtime.growth_pulses)
        self.assertNotIn(grandchild_id, runtime.growth_pulses)
        self.assertEqual(
            runtime.growth_level(
                root_result.node_id,
                300 + config.CLOUD_GROWTH_PEAK_FRAME,
                atlas.cloud_growth_ease,
            ),
            config.CLOUD_DRAG_START_RESPONSE_STRENGTH,
        )

    def test_drag_hold_and_release_have_input_specific_ease(self) -> None:
        atlas = WeatherMotionAtlas.build(seed=123)
        runtime = WeatherMotionRuntime()

        runtime.trigger_drag_hold(7, 100)
        self.assertEqual(runtime.response_kind(7, 106), TouchResponseKind.DRAG_HOLD)
        self.assertEqual(
            runtime.growth_level(7, 107, atlas.cloud_growth_ease),
            config.CLOUD_DRAG_HOLD_RESPONSE_STRENGTH,
        )

        runtime.growth_pulses.clear()
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        runtime.trigger_response_wave(
            simulation.state,
            result.node_id,
            200,
            TouchResponseKind.RELEASE,
        )

        self.assertEqual(runtime.response_kind(result.node_id, 200), TouchResponseKind.RELEASE)
        self.assertEqual(
            runtime.growth_level(result.node_id, 200, atlas.cloud_growth_ease),
            config.CLOUD_RELEASE_RESPONSE_STRENGTH,
        )
        self.assertEqual(
            runtime.growth_level(
                result.node_id,
                200 + config.CLOUD_RELEASE_RESPONSE_DURATION_FRAMES,
                atlas.cloud_growth_ease,
            ),
            0,
        )

    def test_growth_response_blocks_ambient_morph_until_cooldown(self) -> None:
        runtime = WeatherMotionRuntime()
        runtime.trigger_growth(3, 100)

        self.assertTrue(runtime.response_blocks_ambient(3, 100))
        self.assertTrue(
            runtime.response_blocks_ambient(
                3,
                100 + config.CLOUD_TAP_PULSE_DURATION_FRAMES,
            )
        )
        self.assertFalse(
            runtime.response_blocks_ambient(
                3,
                100
                + config.CLOUD_TAP_PULSE_DURATION_FRAMES
                + config.POST_RESPONSE_AMBIENT_COOLDOWN_FRAMES,
            )
        )

    def test_sparse_morph_selection_respects_state_ratio_and_absolute_cap(self) -> None:
        active_nodes = choose_morph_node_ids(
            99,
            0,
            tuple(range(1, 21)),
            int(CloudMotionState.ACTIVE),
            20,
        )
        mature_nodes = choose_morph_node_ids(
            99,
            0,
            tuple(range(1, 21)),
            int(CloudMotionState.MATURE),
            20,
        )

        self.assertLessEqual(len(active_nodes), config.CLOUD_AMBIENT_MORPH_MAX_NODES)
        self.assertLessEqual(len(active_nodes), 4)
        self.assertLessEqual(len(mature_nodes), 2)
        self.assertGreaterEqual(len(active_nodes), len(mature_nodes))

    def test_sparse_morph_intervals_get_longer_as_cloud_settles(self) -> None:
        active = morph_interval_bounds(int(CloudMotionState.ACTIVE))
        settling = morph_interval_bounds(int(CloudMotionState.SETTLING))
        mature = morph_interval_bounds(int(CloudMotionState.MATURE))

        self.assertLess(active[0], settling[0])
        self.assertLess(settling[0], mature[0])
        self.assertLess(active[1], settling[1])
        self.assertLess(settling[1], mature[1])

    def test_sparse_morph_runtime_is_deterministic(self) -> None:
        first = WeatherMotionRuntime()
        second = WeatherMotionRuntime()
        kwargs = {
            "cluster_key": 17,
            "candidate_node_ids": tuple(range(1, 10)),
            "frame": 1000,
            "motion_state": int(CloudMotionState.ACTIVE),
            "node_count": 9,
        }

        self.assertEqual(
            first.ambient_morph_variants(**kwargs),
            second.ambient_morph_variants(**kwargs),
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

    def test_cloud_render_offset_stays_zero_through_quiet_ambient(self) -> None:
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

        self.assertEqual(set(offsets), {(0.0, 0.0, 1.0)})

        node.incubation = 0.9
        mature_runtime = WeatherMotionRuntime()
        mature_offsets = [
            cloud_render_offset(node, simulation.state, atlas, frame, mature_runtime)
            for frame in range(0, int(config.CLOUD_MOTION_PERIOD_SECONDS * config.FPS), 30)
        ]
        self.assertEqual(set(mature_offsets), {(0.0, 0.0, 1.0)})


if __name__ == "__main__":
    unittest.main()
