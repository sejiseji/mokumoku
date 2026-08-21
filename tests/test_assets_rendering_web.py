from __future__ import annotations

import unittest
from pathlib import Path

import pyxel
from scripts.build_web import disable_virtual_gamepad
from src.assets.sprite_map import CloudSpriteFamily, cloud_sprite_rect, size_class_for_screen_radius
from src.camera.camera import build_camera_basis
from src.camera.projection import project_point
from src.cloud.graph import add_edge, create_node, recompute_clusters
from src.cloud.rendering import (
    BridgePayload,
    EdgePayload,
    NodePayload,
    choose_cloud_sprite_family,
    cloud_node_wobble,
    collect_cloud_render_items,
    single_node_mesh_intensity,
    single_node_mesh_phase,
)
from src.cloud.simulation import CloudSimulation
from src.enums import EdgeKind
from src.math3d import Vec3
from src.rng import RandomSource

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AssetsRenderingWebTests(unittest.TestCase):
    def test_cloud_resource_exists_and_loads_in_headless_pyxel(self) -> None:
        resource_path = PROJECT_ROOT / "assets" / "mokumoku.pyxres"

        self.assertTrue(resource_path.exists())
        pyxel.init(16, 16, headless=True)
        pyxel.load(str(resource_path))

        rect = cloud_sprite_rect(CloudSpriteFamily.INTERNAL, "s")
        self.assertNotEqual(pyxel.images[rect.image].pget(rect.u + 8, rect.v + 8), 0)

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
        expected_order = sorted(
            items,
            key=lambda item: (-item.depth, item.layer_bias, item.stable_id),
        )
        self.assertEqual(items, expected_order)
        self.assertTrue(any(isinstance(item.payload, EdgePayload) for item in items))
        self.assertTrue(any(isinstance(item.payload, BridgePayload) for item in items))
        self.assertTrue(any(isinstance(item.payload, NodePayload) for item in items))

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

    def test_cloud_wobble_changes_over_time(self) -> None:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        node = simulation.state.nodes[result.node_id]

        early = cloud_node_wobble(node, simulation.state, 0)
        later = cloud_node_wobble(node, simulation.state, 30)

        self.assertNotEqual(early, later)
        animated_items = collect_cloud_render_items(simulation.state, camera, frame=30)
        animated_payloads = [
            item.payload for item in animated_items if isinstance(item.payload, NodePayload)
        ]
        self.assertTrue(any(abs(payload.offset_x) > 0.0 for payload in animated_payloads))

    def test_cloud_wobble_does_not_jitter_between_adjacent_frames(self) -> None:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        node = simulation.state.nodes[result.node_id]

        first = cloud_node_wobble(node, simulation.state, 30)
        second = cloud_node_wobble(node, simulation.state, 31)
        delta = ((second[0] - first[0]) ** 2 + (second[1] - first[1]) ** 2) ** 0.5

        self.assertLess(delta, 0.006)

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
        self.assertLess(phase_b - phase_a, 0.008)
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
        self.assertIsNotNone(child.node_id)

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
            '</head><body><script>const options = {, gamepad: "enabled"};</script></body></html>',
            encoding="utf-8",
        )
        try:
            disable_virtual_gamepad(html_path)
            text = html_path.read_text(encoding="utf-8")
            self.assertNotIn("gamepad", text)
            self.assertIn("touch-action:none", text)
            self.assertIn("viewport-fit=cover", text)
            self.assertIn("user-scalable=no", text)
        finally:
            html_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
