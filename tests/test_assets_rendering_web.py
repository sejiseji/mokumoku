from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pyxel
from scripts.build_web import disable_virtual_gamepad, prune_versioned_builds, write_build_info
from src import config
from src.assets.sprite_map import (
    CLOUD_SPRITE_VARIANT_COUNT,
    CloudSpriteFamily,
    cloud_sprite_rect,
    size_class_for_screen_radius,
)
from src.camera.camera import build_camera_basis
from src.camera.projection import project_point
from src.cloud.graph import add_edge, create_node, recompute_clusters
from src.cloud.rendering import (
    BridgePayload,
    EdgePayload,
    NodePayload,
    RenderItem,
    ambient_morph_priority,
    choose_cloud_sprite_family,
    cloud_bridge_count,
    cloud_node_wobble,
    collect_cloud_render_items,
    depth_sort_bucket,
    render_item_sort_key,
    single_node_mesh_intensity,
    single_node_mesh_phase,
)
from src.cloud.simulation import CloudSimulation
from src.enums import EdgeKind
from src.math3d import Vec3
from src.motion.atlas import WeatherMotionAtlas
from src.motion.runtime import TouchResponseKind, WeatherMotionRuntime
from src.rng import RandomSource

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_headless_pyxel() -> None:
    if pyxel.width == 0:
        pyxel.init(16, 16, headless=True)


def first_spawned_node_id(result) -> int:
    assert result.spawned_node_ids
    return result.spawned_node_ids[0]


class AssetsRenderingWebTests(unittest.TestCase):
    def test_cloud_resource_exists_and_loads_in_headless_pyxel(self) -> None:
        resource_path = PROJECT_ROOT / "assets" / "mokumoku.pyxres"

        self.assertTrue(resource_path.exists())
        ensure_headless_pyxel()
        pyxel.load(str(resource_path))

        rect = cloud_sprite_rect(CloudSpriteFamily.INTERNAL, "s")
        self.assertNotEqual(pyxel.images[rect.image].pget(rect.u + 8, rect.v + 8), 0)

    def test_cloud_resource_variants_share_anchor_and_load(self) -> None:
        resource_path = PROJECT_ROOT / "assets" / "mokumoku.pyxres"
        ensure_headless_pyxel()
        pyxel.load(str(resource_path))

        base = cloud_sprite_rect(CloudSpriteFamily.EDGE, "m", 0)
        for variant in range(CLOUD_SPRITE_VARIANT_COUNT):
            rect = cloud_sprite_rect(CloudSpriteFamily.EDGE, "m", variant)
            self.assertEqual((rect.u, rect.v, rect.width, rect.height), (base.u, base.v, 24, 24))
            self.assertEqual(rect.image, variant)
            self.assertNotEqual(pyxel.images[rect.image].pget(rect.u + 12, rect.v + 12), 0)

        with self.assertRaises(ValueError):
            cloud_sprite_rect(CloudSpriteFamily.EDGE, "m", CLOUD_SPRITE_VARIANT_COUNT)

    def test_size_class_uses_discrete_sprite_sizes(self) -> None:
        self.assertEqual(size_class_for_screen_radius(9.0), "s")
        self.assertEqual(size_class_for_screen_radius(12.0), "m")
        self.assertEqual(size_class_for_screen_radius(17.0), "l")
        self.assertEqual(size_class_for_screen_radius(22.0), "xl")

    def test_initial_seed_uses_small_sprite(self) -> None:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        node = simulation.state.nodes[result.node_id]
        projection = project_point(node.position, camera)

        size_class = size_class_for_screen_radius(node.radius * projection.scale)

        self.assertEqual(size_class, "s")

    def test_cloud_render_items_are_sorted_back_to_front(self) -> None:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        parent = simulation.state.nodes[result.node_id]
        child = create_node(
            simulation.state,
            parent.lineage_id,
            parent.cluster_id,
            parent.position + Vec3(32.0, 0.0, -10.0),
            simulation.rng,
            parent_node_id=parent.id,
            generation=1,
        )
        add_edge(
            simulation.state,
            parent.lineage_id,
            parent.cluster_id,
            parent.id,
            child.id,
            EdgeKind.PRIMARY,
        )
        recompute_clusters(simulation.state, parent.lineage_id)

        items = collect_cloud_render_items(simulation.state, camera)

        self.assertGreaterEqual(len(items), 3)
        expected_order = sorted(items, key=render_item_sort_key)
        self.assertEqual(items, expected_order)
        self.assertTrue(any(isinstance(item.payload, EdgePayload) for item in items))
        self.assertTrue(any(isinstance(item.payload, BridgePayload) for item in items))
        self.assertTrue(any(isinstance(item.payload, NodePayload) for item in items))

    def test_depth_sort_bucket_keeps_near_equal_items_stable(self) -> None:
        far_id = RenderItem(depth=100.10, layer_bias=2, stable_id=20, payload=object())
        near_id = RenderItem(depth=100.00, layer_bias=2, stable_id=10, payload=object())

        self.assertEqual(depth_sort_bucket(far_id.depth), depth_sort_bucket(near_id.depth))
        self.assertEqual(
            sorted((far_id, near_id), key=render_item_sort_key),
            [near_id, far_id],
        )

    def test_bridge_rendering_fills_gap_then_thins_under_strain(self) -> None:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        parent = simulation.state.nodes[result.node_id]
        child = create_node(
            simulation.state,
            parent.lineage_id,
            parent.cluster_id,
            parent.position + Vec3(64.0, 0.0, 0.0),
            simulation.rng,
            parent_node_id=parent.id,
            generation=1,
        )
        edge = add_edge(
            simulation.state,
            parent.lineage_id,
            parent.cluster_id,
            parent.id,
            child.id,
            EdgeKind.PRIMARY,
        )
        self.assertIsNotNone(edge)
        recompute_clusters(simulation.state, parent.lineage_id)

        filled_items = collect_cloud_render_items(simulation.state, camera)
        filled_bridges = [
            item.payload for item in filled_items if isinstance(item.payload, BridgePayload)
        ]
        self.assertGreaterEqual(len(filled_bridges), 2)
        self.assertTrue(
            all(bridge.family is CloudSpriteFamily.INTERNAL for bridge in filled_bridges)
        )

        edge.strain = config.CLOUD_BRIDGE_NECK_STRAIN
        strained_items = collect_cloud_render_items(simulation.state, camera)
        strained_bridges = [
            item.payload for item in strained_items if isinstance(item.payload, BridgePayload)
        ]
        self.assertLess(len(strained_bridges), len(filled_bridges))
        self.assertEqual(len(strained_bridges), 1)
        self.assertEqual(strained_bridges[0].family, CloudSpriteFamily.STRETCH)
        self.assertLess(strained_bridges[0].visual_radius, filled_bridges[0].visual_radius)

        edge.strain = config.CLOUD_BRIDGE_MAX_STRAIN
        broken_items = collect_cloud_render_items(simulation.state, camera)
        broken_bridges = [
            item.payload for item in broken_items if isinstance(item.payload, BridgePayload)
        ]
        self.assertEqual(len(broken_bridges), 0)

    def test_bridge_count_moves_from_fill_to_neck_by_strain(self) -> None:
        self.assertEqual(cloud_bridge_count(0.92, 0.0, 0.0), 4)
        self.assertEqual(cloud_bridge_count(0.72, 0.0, 0.0), 3)
        self.assertEqual(cloud_bridge_count(0.60, 0.0, 0.0), 2)
        self.assertEqual(cloud_bridge_count(0.92, config.CLOUD_BRIDGE_NECK_STRAIN, 0.4), 1)

    def test_connected_tap_places_child_close_enough_for_cloud_cohesion(self) -> None:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        parent = simulation.state.nodes[result.node_id]
        projection = project_point(parent.position, camera)

        child = simulation.tap_screen(
            projection.screen_x + config.DORMANT_SEED_TAP_RADIUS_PX - 4.0,
            projection.screen_y,
            camera,
        )
        child_id = first_spawned_node_id(child)
        child_node = simulation.state.nodes[child_id]
        child_projection = project_point(child_node.position, camera)
        distance = abs(child_projection.screen_x - projection.screen_x)
        radius_sum = parent.radius * projection.scale + child_node.radius * child_projection.scale

        self.assertLessEqual(distance / radius_sum, 1.05)

    def test_sprite_family_uses_projected_role_and_attribute_states(self) -> None:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        node = simulation.state.nodes[result.node_id]

        self.assertEqual(
            choose_cloud_sprite_family(node, simulation.state),
            CloudSpriteFamily.FRAGMENT,
        )

        node.density = 1.5
        self.assertEqual(
            choose_cloud_sprite_family(node, simulation.state),
            CloudSpriteFamily.BOTTOM,
        )

        node.density = 1.0
        node.updraft = 0.5
        self.assertEqual(
            choose_cloud_sprite_family(node, simulation.state),
            CloudSpriteFamily.UPDRAFT,
        )

        node.is_pruning = True
        self.assertEqual(
            choose_cloud_sprite_family(node, simulation.state),
            CloudSpriteFamily.FADE,
        )

    def test_sprite_family_uses_projected_top_and_bottom_exposure(self) -> None:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        lower = simulation.state.nodes[result.node_id]
        upper = create_node(
            simulation.state,
            lower.lineage_id,
            lower.cluster_id,
            lower.position + Vec3(0.0, 24.0, 0.0),
            simulation.rng,
            parent_node_id=lower.id,
            generation=1,
        )
        add_edge(
            simulation.state,
            lower.lineage_id,
            lower.cluster_id,
            lower.id,
            upper.id,
            EdgeKind.PRIMARY,
        )
        recompute_clusters(simulation.state, lower.lineage_id)

        lower_projection = project_point(lower.position, camera)
        upper_projection = project_point(upper.position, camera)

        self.assertEqual(
            choose_cloud_sprite_family(lower, simulation.state, camera, lower_projection),
            CloudSpriteFamily.BOTTOM,
        )
        self.assertEqual(
            choose_cloud_sprite_family(upper, simulation.state, camera, upper_projection),
            CloudSpriteFamily.UPDRAFT,
        )

    def test_runtime_holds_projected_sprite_role_briefly(self) -> None:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        lower = simulation.state.nodes[result.node_id]
        upper = create_node(
            simulation.state,
            lower.lineage_id,
            lower.cluster_id,
            lower.position + Vec3(0.0, 24.0, 0.0),
            simulation.rng,
            parent_node_id=lower.id,
            generation=1,
        )
        add_edge(
            simulation.state,
            lower.lineage_id,
            lower.cluster_id,
            lower.id,
            upper.id,
            EdgeKind.PRIMARY,
        )
        recompute_clusters(simulation.state, lower.lineage_id)
        runtime = WeatherMotionRuntime()

        first_items = collect_cloud_render_items(
            simulation.state,
            camera,
            frame=10,
            motion_runtime=runtime,
        )
        first_payloads = {
            item.payload.node.id: item.payload
            for item in first_items
            if isinstance(item.payload, NodePayload)
        }
        self.assertEqual(first_payloads[lower.id].family, CloudSpriteFamily.BOTTOM)

        upper.position = lower.position + Vec3(0.0, -24.0, 0.0)
        upper.previous_position = upper.position
        lower_projection = project_point(lower.position, camera)
        self.assertEqual(
            choose_cloud_sprite_family(lower, simulation.state, camera, lower_projection),
            CloudSpriteFamily.UPDRAFT,
        )

        held_items = collect_cloud_render_items(
            simulation.state,
            camera,
            frame=11,
            motion_runtime=runtime,
        )
        held_payloads = {
            item.payload.node.id: item.payload
            for item in held_items
            if isinstance(item.payload, NodePayload)
        }
        self.assertEqual(held_payloads[lower.id].family, CloudSpriteFamily.BOTTOM)

        switched_items = collect_cloud_render_items(
            simulation.state,
            camera,
            frame=10 + config.CLOUD_ROLE_MIN_HOLD_FRAMES + 1,
            motion_runtime=runtime,
        )
        switched_payloads = {
            item.payload.node.id: item.payload
            for item in switched_items
            if isinstance(item.payload, NodePayload)
        }
        self.assertEqual(switched_payloads[lower.id].family, CloudSpriteFamily.UPDRAFT)

    def test_sprite_family_uses_projected_internal_when_surrounded(self) -> None:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        center = simulation.state.nodes[result.node_id]
        offsets = (
            Vec3(-24.0, 0.0, 0.0),
            Vec3(24.0, 0.0, 0.0),
            Vec3(0.0, -24.0, 0.0),
            Vec3(0.0, 24.0, 0.0),
        )
        for offset in offsets:
            neighbor = create_node(
                simulation.state,
                center.lineage_id,
                center.cluster_id,
                center.position + offset,
                simulation.rng,
                parent_node_id=center.id,
                generation=1,
            )
            add_edge(
                simulation.state,
                center.lineage_id,
                center.cluster_id,
                center.id,
                neighbor.id,
                EdgeKind.PRIMARY,
            )
        recompute_clusters(simulation.state, center.lineage_id)

        center_projection = project_point(center.position, camera)

        self.assertEqual(
            choose_cloud_sprite_family(center, simulation.state, camera, center_projection),
            CloudSpriteFamily.INTERNAL,
        )

    def test_projected_sprite_rect_has_visible_center(self) -> None:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        node = simulation.state.nodes[result.node_id]
        projection = project_point(node.position, camera)
        size_class = size_class_for_screen_radius(node.radius * projection.scale)
        rect = cloud_sprite_rect(CloudSpriteFamily.INTERNAL, size_class)

        self.assertGreater(rect.width, 0)
        self.assertGreater(rect.height, 0)

    def test_cloud_node_body_offsets_do_not_animate(self) -> None:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        node = simulation.state.nodes[result.node_id]

        early = cloud_node_wobble(node, simulation.state, 0)
        later = cloud_node_wobble(node, simulation.state, 600)

        self.assertEqual(early, (0.0, 0.0, 1.0))
        self.assertEqual(later, (0.0, 0.0, 1.0))
        animated_items = collect_cloud_render_items(simulation.state, camera, frame=30)
        animated_payloads = [
            item.payload for item in animated_items if isinstance(item.payload, NodePayload)
        ]
        self.assertTrue(animated_payloads)
        self.assertTrue(
            all(
                payload.offset_x == 0.0 and payload.offset_y == 0.0
                for payload in animated_payloads
            )
        )

    def test_motion_atlas_does_not_move_node_centers_in_quiet_ambient(self) -> None:
        atlas = WeatherMotionAtlas.build(seed=12345)
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        runtime = WeatherMotionRuntime()

        payloads = []
        for frame in range(0, config.FPS * 30, config.FPS):
            items = collect_cloud_render_items(
                simulation.state,
                camera,
                frame=frame,
                motion_atlas=atlas,
                motion_runtime=runtime,
            )
            payloads.append(
                next(item.payload for item in items if isinstance(item.payload, NodePayload))
            )

        self.assertEqual(
            {(payload.offset_x, payload.offset_y) for payload in payloads},
            {(0.0, 0.0)},
        )

    def test_motion_atlas_does_not_change_size_class_in_quiet_ambient(self) -> None:
        atlas = WeatherMotionAtlas.build(seed=12345)
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        runtime = WeatherMotionRuntime()

        payloads = []
        for frame in range(0, config.FPS * 30, config.FPS):
            items = collect_cloud_render_items(
                simulation.state,
                camera,
                frame=frame,
                motion_atlas=atlas,
                motion_runtime=runtime,
            )
            payloads.append(
                next(item.payload for item in items if isinstance(item.payload, NodePayload))
            )

        sprite_shapes = {
            (payload.sprite.u, payload.sprite.v, payload.sprite.width, payload.sprite.height)
            for payload in payloads
        }
        expected_shape = (
            payloads[0].sprite.u,
            payloads[0].sprite.v,
            payloads[0].sprite.width,
            payloads[0].sprite.height,
        )
        self.assertEqual(sprite_shapes, {expected_shape})

    def test_sparse_ambient_morph_excludes_internal_nodes(self) -> None:
        self.assertIsNone(ambient_morph_priority(CloudSpriteFamily.INTERNAL))
        self.assertIsNone(ambient_morph_priority(CloudSpriteFamily.FADE))
        self.assertIsNotNone(ambient_morph_priority(CloudSpriteFamily.EDGE))

    def test_sparse_ambient_morph_uses_limited_sprite_variants(self) -> None:
        atlas = WeatherMotionAtlas.build(seed=12345)
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        parent = simulation.state.nodes[result.node_id]
        projection = project_point(parent.position, camera)
        child = simulation.tap_screen(projection.screen_x + 30.0, projection.screen_y, camera)
        self.assertGreaterEqual(len(child.spawned_node_ids), 1)

        runtime = WeatherMotionRuntime()
        variant_counts: list[int] = []
        for frame in range(0, config.FPS * 5):
            items = collect_cloud_render_items(
                simulation.state,
                camera,
                frame=frame,
                motion_atlas=atlas,
                motion_runtime=runtime,
            )
            payloads = [
                item.payload for item in items if isinstance(item.payload, NodePayload)
            ]
            variant_count = sum(1 for payload in payloads if payload.sprite.image > 0)
            variant_counts.append(variant_count)

        self.assertGreater(max(variant_counts), 0)
        self.assertLessEqual(max(variant_counts), 1)

    def test_growth_event_level_is_added_to_node_payload(self) -> None:
        atlas = WeatherMotionAtlas.build(seed=12345)
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        runtime = WeatherMotionRuntime()
        runtime.trigger_growth(result.node_id, 20)

        peak_items = collect_cloud_render_items(
            simulation.state,
            camera,
            frame=20 + config.CLOUD_GROWTH_PEAK_FRAME,
            motion_atlas=atlas,
            motion_runtime=runtime,
        )
        settled_items = collect_cloud_render_items(
            simulation.state,
            camera,
            frame=20 + len(atlas.cloud_growth_ease),
            motion_atlas=atlas,
            motion_runtime=runtime,
        )
        peak = next(item.payload for item in peak_items if isinstance(item.payload, NodePayload))
        settled = next(
            item.payload for item in settled_items if isinstance(item.payload, NodePayload)
        )

        self.assertEqual(peak.growth_level, config.CLOUD_PULSE_STRENGTH_BY_DISTANCE[0])
        self.assertEqual(peak.response_kind, TouchResponseKind.TAP)
        self.assertEqual(settled.growth_level, 0)
        self.assertIsNone(settled.response_kind)

    def test_input_response_kind_is_added_to_node_payload(self) -> None:
        atlas = WeatherMotionAtlas.build(seed=12345)
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        runtime = WeatherMotionRuntime()
        runtime.trigger_response_wave(
            simulation.state,
            result.node_id,
            20,
            TouchResponseKind.LONG_PRESS,
        )

        items = collect_cloud_render_items(
            simulation.state,
            camera,
            frame=20 + config.CLOUD_GROWTH_PEAK_FRAME,
            motion_atlas=atlas,
            motion_runtime=runtime,
        )
        payload = next(item.payload for item in items if isinstance(item.payload, NodePayload))

        self.assertEqual(payload.response_kind, TouchResponseKind.LONG_PRESS)
        self.assertEqual(payload.growth_level, config.CLOUD_LONG_PRESS_RESPONSE_STRENGTH)

    def test_growth_wave_reaches_neighbor_payload_after_delay(self) -> None:
        atlas = WeatherMotionAtlas.build(seed=12345)
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        parent = simulation.state.nodes[result.node_id]
        projection = project_point(parent.position, camera)
        child = simulation.tap_screen(projection.screen_x + 30.0, projection.screen_y, camera)
        child_id = first_spawned_node_id(child)
        runtime = WeatherMotionRuntime()
        runtime.trigger_growth_wave(simulation.state, result.node_id, 40)

        before_child_items = collect_cloud_render_items(
            simulation.state,
            camera,
            frame=40 + config.CLOUD_PULSE_PROPAGATION_DELAY_FRAMES - 1,
            motion_atlas=atlas,
            motion_runtime=runtime,
        )
        child_peak_items = collect_cloud_render_items(
            simulation.state,
            camera,
            frame=40
            + config.CLOUD_PULSE_PROPAGATION_DELAY_FRAMES
            + config.CLOUD_GROWTH_PEAK_FRAME,
            motion_atlas=atlas,
            motion_runtime=runtime,
        )
        before_payloads = {
            item.payload.node.id: item.payload
            for item in before_child_items
            if isinstance(item.payload, NodePayload)
        }
        peak_payloads = {
            item.payload.node.id: item.payload
            for item in child_peak_items
            if isinstance(item.payload, NodePayload)
        }

        self.assertEqual(before_payloads[child_id].growth_level, 0)
        self.assertEqual(
            peak_payloads[child_id].growth_level,
            config.CLOUD_PULSE_STRENGTH_BY_DISTANCE[1],
        )

    def test_small_single_cloud_uses_slow_mesh_overlay(self) -> None:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        node = simulation.state.nodes[result.node_id]
        projection = project_point(node.position, camera)

        intensity = single_node_mesh_intensity(
            node,
            simulation.state,
            node.radius * projection.scale,
        )
        phase_a = single_node_mesh_phase(node, 30)
        phase_b = single_node_mesh_phase(node, 31)

        self.assertGreater(intensity, 0.0)
        self.assertLess(phase_b - phase_a, 0.0045)
        items = collect_cloud_render_items(simulation.state, camera, frame=30)
        node_payloads = [
            item.payload for item in items if isinstance(item.payload, NodePayload)
        ]
        self.assertTrue(any(payload.mesh_intensity > 0.0 for payload in node_payloads))

    def test_connected_cloud_suppresses_single_mesh_overlay(self) -> None:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        parent = simulation.state.nodes[result.node_id]
        projection = project_point(parent.position, camera)
        child = simulation.tap_screen(projection.screen_x + 30.0, projection.screen_y, camera)
        self.assertGreaterEqual(len(child.spawned_node_ids), 1)

        parent_projection = project_point(parent.position, camera)
        intensity = single_node_mesh_intensity(
            parent,
            simulation.state,
            parent.radius * parent_projection.scale,
        )

        self.assertEqual(intensity, 0.0)

    def test_web_html_postprocess_disables_gamepad_and_touch_scrolling(self) -> None:
        html_path = PROJECT_ROOT / "docs" / "_postprocess_test.html"
        html_path.write_text(
            '<html><head><meta name="viewport" content="width=device-width, initial-scale=1.0">'
            "</head><body><script>"
            'launchPyxel({ command: "play", name: "mokumoku.pyxapp", base64: "abc" });'
            'const options = {, gamepad: "enabled"};'
            "</script></body></html>",
            encoding="utf-8",
        )
        try:
            disable_virtual_gamepad(html_path, "abc123def456")
            text = html_path.read_text(encoding="utf-8")
            self.assertNotIn("gamepad", text)
            self.assertIn("touch-action:none", text)
            self.assertIn("viewport-fit=cover", text)
            self.assertIn("user-scalable=no", text)
            self.assertIn("no-store, no-cache, must-revalidate", text)
            self.assertIn('name: "mokumoku-abc123def456.pyxapp"', text)
            self.assertIn('name="mokumoku-build" content="abc123def456"', text)
        finally:
            html_path.unlink(missing_ok=True)

    def test_web_build_info_stamp_is_written_into_package_copy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mokumoku_build_info_") as temp_dir_name:
            package_dir = Path(temp_dir_name)
            (package_dir / "src").mkdir()

            build_info_path = write_build_info(package_dir, "20260822044236")

            text = build_info_path.read_text(encoding="utf-8")
            self.assertIn('APP_BUILD_STAMP = "20260822044236"', text)
            self.assertIn('APP_BUILD_LABEL = "b044236"', text)

    def test_versioned_web_build_prunes_third_previous_build(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mokumoku_builds_") as temp_dir_name:
            builds_dir = Path(temp_dir_name)
            build_names = ("previous_three", "previous_two", "previous_one", "latest")
            for index, name in enumerate(build_names):
                build_dir = builds_dir / name
                build_dir.mkdir()
                (build_dir / "index.html").write_text(name, encoding="utf-8")
                timestamp = 1_800_000_000 + index
                os.utime(build_dir, (timestamp, timestamp))

            pruned_paths = prune_versioned_builds(builds_dir, retain=3)

            self.assertEqual([path.name for path in pruned_paths], ["previous_three"])
            self.assertFalse((builds_dir / "previous_three").exists())
            self.assertTrue((builds_dir / "previous_two").exists())
            self.assertTrue((builds_dir / "previous_one").exists())
            self.assertTrue((builds_dir / "latest").exists())


if __name__ == "__main__":
    unittest.main()
