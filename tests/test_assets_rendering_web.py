from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pyxel
from scripts.build_web import disable_virtual_gamepad, prune_versioned_builds
from src import config
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

        edge.strain = config.CLOUD_BRIDGE_MAX_STRAIN - 0.01
        strained_items = collect_cloud_render_items(simulation.state, camera)
        strained_bridges = [
            item.payload for item in strained_items if isinstance(item.payload, BridgePayload)
        ]
        self.assertLess(len(strained_bridges), len(filled_bridges))
        self.assertGreaterEqual(len(strained_bridges), 1)

        edge.strain = config.CLOUD_BRIDGE_MAX_STRAIN
        broken_items = collect_cloud_render_items(simulation.state, camera)
        broken_bridges = [
            item.payload for item in broken_items if isinstance(item.payload, BridgePayload)
        ]
        self.assertEqual(len(broken_bridges), 0)

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

    def test_cloud_node_body_offsets_do_not_jitter_between_adjacent_frames(self) -> None:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        node = simulation.state.nodes[result.node_id]

        first = cloud_node_wobble(node, simulation.state, 30)
        second = cloud_node_wobble(node, simulation.state, 31)

        self.assertEqual(first, second)

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
            self.assertIn('name: "mokumoku-abc123def456.pyxapp"', text)
            self.assertIn('name="mokumoku-build" content="abc123def456"', text)
        finally:
            html_path.unlink(missing_ok=True)

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
