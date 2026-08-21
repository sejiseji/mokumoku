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
    EdgePayload,
    NodePayload,
    choose_cloud_sprite_family,
    collect_cloud_render_items,
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
        self.assertEqual(size_class_for_screen_radius(7.0), "s")
        self.assertEqual(size_class_for_screen_radius(12.0), "m")
        self.assertEqual(size_class_for_screen_radius(17.0), "l")
        self.assertEqual(size_class_for_screen_radius(22.0), "xl")

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
            parent.position + Vec3(0.0, 0.0, -18.0),
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
        self.assertTrue(any(isinstance(item.payload, NodePayload) for item in items))

    def test_sprite_family_uses_fragment_and_fade_states(self) -> None:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        node = simulation.state.nodes[result.node_id]

        self.assertEqual(
            choose_cloud_sprite_family(node, simulation.state),
            CloudSpriteFamily.INTERNAL,
        )

        node.parent_node_id = 99
        self.assertEqual(
            choose_cloud_sprite_family(node, simulation.state),
            CloudSpriteFamily.FRAGMENT,
        )

        node.is_pruning = True
        self.assertEqual(
            choose_cloud_sprite_family(node, simulation.state),
            CloudSpriteFamily.FADE,
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
